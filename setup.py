from setuptools import setup, Extension, find_packages, Command
from Cython.Build import cythonize
import numpy as np
import os
import shutil
import glob


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
pyx_files = find_pyx_files('helpers') + find_pyx_files('phases')
# Add root level pyx files (non-recursive)
pyx_files += [f for f in os.listdir('.') if f.endswith('.pyx')]

extensions = []
for pyx_file in pyx_files:
    # Convert path to module name: cython_core/state.pyx -> cython_core.state
    module_name = pyx_file.replace('/', '.').replace('\\', '.').replace('.pyx', '')

    extensions.append(Extension(
        module_name,
        [pyx_file],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ))

setup(
    name="rss-cython-core",
    packages=['phases', 'helpers'],
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate=True,  # Generates HTML annotation files showing Python interaction
    ),
    cmdclass={
        'clean': CleanCommand,
    },
    include_package_data=True,
    zip_safe=False,
)
