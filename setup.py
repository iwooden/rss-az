from setuptools import setup, Extension, find_packages, Command
from Cython.Build import cythonize
import numpy as np
import os
import shutil
import glob


class BenchmarkCommand(Command):
    """Benchmark command to measure games per minute."""
    description = 'Run benchmark to measure games completed per minute'
    user_options = [
        ('num-games=', 'n', 'Number of games to run (default: 1000)'),
        ('num-players=', 'p', 'Number of players (default: 3)'),
    ]

    def initialize_options(self):
        self.num_games = 1000
        self.num_players = 3

    def finalize_options(self):
        self.num_games = int(self.num_games)
        self.num_players = int(self.num_players)

    def run(self):
        import time
        import numpy as np
        from state import GameState
        from driver import apply_action
        from actions import get_valid_action_mask

        PHASE_GAME_OVER = 10
        num_games = self.num_games
        num_players = self.num_players
        max_steps = 10000

        print(f'Benchmarking {num_games} games with {num_players} players...')

        total_steps = 0
        start_time = time.perf_counter()

        for i in range(num_games):
            np.random.seed(i)
            state = GameState(num_players)
            state.setup_new_game(shuffle_seed=i)
            step = 0

            while state.phase != PHASE_GAME_OVER and step < max_steps:
                mask = get_valid_action_mask(state)
                valid_indices = np.where(mask == 1.0)[0]
                action = np.random.choice(valid_indices)
                apply_action(state, action)
                step += 1

            total_steps += step

            # Progress update every 10%
            if (i + 1) % (num_games // 10) == 0:
                elapsed = time.perf_counter() - start_time
                print(f'  {i + 1}/{num_games} games completed ({elapsed:.1f}s)')

        elapsed = time.perf_counter() - start_time
        games_per_min = (num_games / elapsed) * 60
        steps_per_sec = total_steps / elapsed
        avg_steps = total_steps / num_games

        print(f'\n=== Benchmark Results ===')
        print(f'Total games:      {num_games}')
        print(f'Total time:       {elapsed:.2f}s')
        print(f'Games per minute: {games_per_min:.1f}')
        print(f'Avg steps/game:   {avg_steps:.1f}')
        print(f'Steps per second: {steps_per_sec:.1f}')


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
        'benchmark': BenchmarkCommand,
    },
    include_package_data=True,
    zip_safe=False,
)
