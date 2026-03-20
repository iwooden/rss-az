from setuptools import setup, Extension, find_packages, Command
from Cython.Build import cythonize
import numpy as np
import os
import shutil
import glob

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
            '*.so',
            '*.html',
            '**/*.c',
            '**/*.so',
            '**/*.html',
        ]

        for pattern in patterns:
            for path in glob.glob(pattern, recursive=True):
                print(f'Removing {path}')
                os.remove(path)

        # Remove build directory
        if os.path.exists('build'):
            print('Removing build/')
            shutil.rmtree('build')

        # Remove egg-info
        for path in glob.glob('*.egg-info'):
            print(f'Removing {path}')
            shutil.rmtree(path)

        print('Clean complete.')

# Compiler directives for maximum performance
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

# Get pyx files from subdirectories and root level
pyx_files = find_pyx_files('phases') + find_pyx_files('core') + find_pyx_files('entities') + find_pyx_files('mcts')
# Add root level pyx files (non-recursive)
pyx_files += [f for f in os.listdir('.') if f.endswith('.pyx')]

extensions = []

# Extra compile args to suppress benign Cython-generated warnings
# -Wno-unused-function: Cython generates enum-to-Python converter functions
#   that are often unused when enums are only used in cdef code
extra_compile_args = ['-Wno-unused-function']

for pyx_file in pyx_files:
    # Convert path to module name: cython_core/state.pyx -> cython_core.state
    module_name = pyx_file.replace('/', '.').replace('\\', '.').replace('.pyx', '')

    extensions.append(Extension(
        module_name,
        [pyx_file],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=extra_compile_args,
    ))

setup(
    name="rss-cython-core",
    packages=['phases', 'entities', 'core', 'mcts'],
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate=False,  # Generates HTML annotation files showing Python interaction
    ),
    cmdclass={
        'benchmark': BenchmarkCommand,
        'clean': CleanCommand,
        'trace_game': TraceGameCommand,
    },
    include_package_data=True,
    zip_safe=False,
)
