from setuptools import setup, Extension, Command
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

pyx_files = (
    find_pyx_files('core')
    + find_pyx_files('entities')
    + find_pyx_files('phases')
)

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


def get_define_macros() -> list[tuple[str, str | None]]:
    macros: list[tuple[str, str | None]] = [
        ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
    ]
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
SKIP_CYTHONIZE_COMMANDS = {'clean'}
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
    packages=['core', 'entities', 'phases'],
    ext_modules=ext_modules,
    cmdclass={
        'clean': CleanCommand,
    },
    include_package_data=True,
    zip_safe=False,
)
