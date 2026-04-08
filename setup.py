from setuptools import setup, Extension, find_packages, Command
from Cython.Build import cythonize
import numpy as np
import os
import shutil
import glob
import sys
import sysconfig


RELEASE_FLAG = '--release'
RELEASE_BUILD = RELEASE_FLAG in sys.argv

# Strip our custom flag before setuptools parses argv.
if RELEASE_BUILD:
    sys.argv = [arg for arg in sys.argv if arg != RELEASE_FLAG]

class TraceGameCommand(Command):
    """Trace a random game with human-readable output."""
    description = 'Play a random game and output a human-readable trace'
    user_options = [
        ('num-players=', 'p', 'Number of players (2-6, default 3)'),
        ('seed=', 's', 'Random seed (default 42)'),
        ('verbose', 'v', 'Full state dump every step'),
        ('output=', 'o', 'Output file (default: stdout)'),
    ]
    boolean_options = ['verbose']

    def initialize_options(self):
        self.num_players = 3
        self.seed = 42
        self.verbose = False
        self.output = None

    def finalize_options(self):
        self.num_players = int(self.num_players)
        self.seed = int(self.seed)

    def run(self):
        from tests.debug_trace import trace_random_game
        result = trace_random_game(self.num_players, self.seed, self.verbose)
        if self.output:
            with open(self.output, 'w') as f:
                f.write(result)
                f.write('\n')
            print(f'Trace written to {self.output}')
        else:
            print(result)


class BenchmarkCommand(Command):
    """Run performance benchmarks."""
    description = 'Run MCTS search benchmark'
    user_options = [
        ('num-simulations=', 'n', 'Simulations per search (default 800)'),
        ('num-runs=', 'r', 'Number of timed runs (default 10)'),
        ('num-players=', 'p', 'Number of players (default 3)'),
        ('device=', 'd', 'Torch device (default cpu)'),
        ('batch-size=', 'b', 'Search batch size for leaf eval (default 1)'),
    ]

    def initialize_options(self):
        self.num_simulations = 800
        self.num_runs = 10
        self.num_players = 3
        self.device = 'cpu'
        self.batch_size = 1

    def finalize_options(self):
        self.num_simulations = int(self.num_simulations)
        self.num_runs = int(self.num_runs)
        self.num_players = int(self.num_players)
        self.batch_size = int(self.batch_size)

    def run(self):
        from benchmarks.mcts_bench import run_mcts_benchmark
        run_mcts_benchmark(
            num_simulations=self.num_simulations,
            num_runs=self.num_runs,
            num_players=self.num_players,
            device=self.device,
            search_batch_size=self.batch_size,
        )


class CleanCommand(Command):
    """Custom clean command to remove Cython build artifacts."""
    description = 'Remove Cython build artifacts (.c, .so, .html, build/)'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Patterns to clean
        patterns = [
            '*.c',
            '*.h',
            '*.cpp',
            '*.so',
            '*.html',
            '**/*.c',
            '**/*.cpp',
            '**/*.so',
        ]
        # Clean Cython-generated .html annotation files, but NOT interp reports
        cython_html_dirs = ['core', 'entities', 'phases', 'mcts']

        files_removed = 0
        dirs_removed = 0

        for pattern in patterns:
            for path in glob.glob(pattern, recursive=True):
                os.remove(path)
                files_removed += 1
        for d in cython_html_dirs:
            for path in glob.glob(f'{d}/**/*.html', recursive=True):
                os.remove(path)
                files_removed += 1

        # Remove build directory
        if os.path.exists('build'):
            shutil.rmtree('build')
            dirs_removed += 1

        # Remove egg-info
        for path in glob.glob('*.egg-info'):
            shutil.rmtree(path)
            dirs_removed += 1

        print(f'Clean complete: {files_removed} files, {dirs_removed} directories removed.')

# Compiler directives for the default build profile.
compiler_directives = {
    'language_level': '3',
    'boundscheck': False,
    'wraparound': False,
    'cdivision': True,
    'initializedcheck': False,
    'nonecheck': False,
    'overflowcheck': False,
    'embedsignature': True,  # Useful for debugging
}

if RELEASE_BUILD:
    # Release builds trade Python introspection for faster generated code.
    compiler_directives.update({
        'binding': False,
        'embedsignature': False,
    })

# Find all .pyx files in specific directories
def find_pyx_files(directory):
    pyx_files = []
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and virtual environments
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'venv']
        for file in files:
            if file.endswith('.pyx'):
                pyx_files.append(os.path.join(root, file))
    return pyx_files

# Refactor in progress: only build core/data.pyx + core/state.pyx and
# entities/*.pyx for now. data.pyx is a pure data/constants module that
# state and every entity cimport from, so it must be in the build set.
pyx_files = ['core/data.pyx', 'core/state.pyx'] + find_pyx_files('entities')

extensions = []

def get_extra_compile_args():
    args = ['-Wno-unused-function']
    if not RELEASE_BUILD:
        return args

    if os.name == 'nt':
        args.extend(['/O2', '/GL'])
    else:
        args.extend(['-O3', '-march=native', get_lto_flag()])
    return args


def get_extra_link_args():
    if not RELEASE_BUILD:
        return []

    if os.name == 'nt':
        return ['/O2', '/LTCG']
    return ['-O3', get_lto_flag()]


def get_define_macros():
    macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    if RELEASE_BUILD:
        macros.append(("CYTHON_WITHOUT_ASSERTIONS", "1"))
    return macros


def get_lto_flag():
    cc = (sysconfig.get_config_var('CC') or '').lower()
    # GCC 13 on this toolchain ICEs with plain -flto but succeeds with -flto=auto.
    if 'gcc' in cc and 'clang' not in cc:
        return '-flto=auto'
    return '-flto'


extra_compile_args = get_extra_compile_args()
extra_link_args = get_extra_link_args()
define_macros = get_define_macros()

if RELEASE_BUILD:
    print(
        'Configuring release build: '
        f'-O3, -march=native, {get_lto_flag()}, Cython assertions off'
    )

for pyx_file in pyx_files:
    # Convert path to module name: cython_core/state.pyx -> cython_core.state
    module_name = pyx_file.replace('/', '.').replace('\\', '.').replace('.pyx', '')

    extensions.append(Extension(
        module_name,
        [pyx_file],
        include_dirs=[np.get_include()],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ))

# Skip cythonize for commands that don't need built extensions. Without this
# guard, even `setup.py clean` triggers a full Cython compile at module-load
# time, which fails (and blocks the clean) whenever any .pyx is mid-refactor.
SKIP_CYTHONIZE_COMMANDS = {'clean', 'trace_game', 'benchmark'}
if any(cmd in sys.argv for cmd in SKIP_CYTHONIZE_COMMANDS):
    ext_modules = []
else:
    ext_modules = cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate=False,  # Generates HTML annotation files showing Python interaction
    )

setup(
    name="rss-cython-core",
    packages=['phases', 'entities', 'core', 'mcts'],
    ext_modules=ext_modules,
    cmdclass={
        'benchmark': BenchmarkCommand,
        'clean': CleanCommand,
        'trace_game': TraceGameCommand,
    },
    include_package_data=True,
    zip_safe=False,
)
