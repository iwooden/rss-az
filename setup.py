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


class DebugGameCommand(Command):
    """Run a random game step-by-step with detailed debug output to GAME_OUTPUT.md"""
    description = 'Run a random game with detailed debug output to GAME_OUTPUT.md'
    user_options = [
        ('seed=', 's', 'Random seed for the game (default: 42)'),
        ('num-players=', 'p', 'Number of players (default: 3)'),
        ('output=', 'o', 'Output file path (default: GAME_OUTPUT.md)'),
    ]

    def initialize_options(self):
        self.seed = 42
        self.num_players = 3
        self.output = 'GAME_OUTPUT.md'

    def finalize_options(self):
        self.seed = int(self.seed)
        self.num_players = int(self.num_players)

    def run(self):
        import numpy as np
        from state import GameState, PHASE_NAMES
        from driver import apply_action
        from actions import get_valid_action_mask, decode_action_py, get_action_layout
        from data import (
            COMPANY_NAMES, CORP_NAMES,
            py_get_company_face_value, py_get_company_low_price,
            py_get_company_high_price, py_get_company_stars, py_get_company_income,
            py_get_corp_share_count, py_get_market_price, py_get_par_price
        )

        # Constants
        PHASE_GAME_OVER = 10
        NUM_CORPS = 8
        NUM_COMPANIES = 36
        MAX_STEPS = 10000

        # Action type names
        ACTION_TYPE_NAMES = {
            0: "PASS", 1: "AUCTION", 2: "BUY_SHARE", 3: "SELL_SHARE",
            4: "LEAVE_AUCTION", 5: "RAISE_BID", 6: "ACQ_PRICE",
            7: "ACQ_FI_HIGH", 8: "ACQ_FI_FACE", 9: "CLOSE",
            10: "DIVIDEND", 11: "ISSUE", 12: "IPO"
        }

        # Compute state layout offsets (based on state.pyx layout)
        # These formulas are from compute_layout() in state.pyx
        num_players = self.num_players
        NUM_PHASES = 11
        NUM_COO_LEVELS = 7
        NUM_MARKET_SPACES_LAYOUT = 27

        # Compute layout offsets
        phase_offset = 0
        coo_offset = phase_offset + NUM_PHASES
        players_offset = coo_offset + NUM_COO_LEVELS
        player_stride = 1 + 1 + num_players + 1 + NUM_COMPANIES + NUM_CORPS + NUM_CORPS + NUM_CORPS + NUM_CORPS
        players_size = player_stride * num_players
        fi_offset = players_offset + players_size
        fi_size = 1 + NUM_COMPANIES
        auction_companies_offset = fi_offset + fi_size
        revealed_companies_offset = auction_companies_offset + NUM_COMPANIES
        removed_companies_offset = revealed_companies_offset + NUM_COMPANIES
        company_incomes_offset = removed_companies_offset + NUM_COMPANIES
        market_offset = company_incomes_offset + NUM_COMPANIES
        corp_stride = 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + NUM_MARKET_SPACES_LAYOUT + NUM_COMPANIES + NUM_COMPANIES
        corps_offset = market_offset + NUM_MARKET_SPACES_LAYOUT
        corps_size = corp_stride * NUM_CORPS
        turn_offset = corps_offset + corps_size

        # Turn state sub-offsets
        turn_turn_number = 0
        turn_end_card_flipped = 1
        turn_consecutive_passes = 2
        turn_auction_company = 3
        turn_auction_price = turn_auction_company + NUM_COMPANIES
        turn_auction_high_bidder = turn_auction_price + 1
        turn_auction_starter = turn_auction_high_bidder + num_players
        turn_auction_passed = turn_auction_starter + num_players
        turn_dividend_corp = turn_auction_passed + num_players
        turn_dividend_impact = turn_dividend_corp + NUM_CORPS
        turn_dividend_remaining = turn_dividend_impact + 26
        turn_issue_corp = turn_dividend_remaining + NUM_CORPS
        turn_issue_remaining = turn_issue_corp + NUM_CORPS
        turn_ipo_company = turn_issue_remaining + NUM_CORPS
        turn_ipo_remaining = turn_ipo_company + NUM_COMPANIES
        turn_acq_active_corp = turn_ipo_remaining + NUM_COMPANIES
        turn_acq_target_company = turn_acq_active_corp + NUM_CORPS
        turn_acq_is_fi_offer = turn_acq_target_company + NUM_COMPANIES
        turn_closing_company = turn_acq_is_fi_offer + 1

        def get_turn_state_int(offset):
            """Get a one-hot index from turn state"""
            for i in range(NUM_COMPANIES):  # Use max size
                if state.as_numpy()[turn_offset + offset + i] == 1.0:
                    return i
            return -1

        def get_dividend_corp():
            """Get current dividend corp from one-hot"""
            for c in range(NUM_CORPS):
                if state.as_numpy()[turn_offset + turn_dividend_corp + c] == 1.0:
                    return c
            return -1

        def get_issue_corp():
            """Get current issue corp from one-hot"""
            for c in range(NUM_CORPS):
                if state.as_numpy()[turn_offset + turn_issue_corp + c] == 1.0:
                    return c
            return -1

        def get_ipo_company():
            """Get current IPO company from one-hot"""
            for c in range(NUM_COMPANIES):
                if state.as_numpy()[turn_offset + turn_ipo_company + c] == 1.0:
                    return c
            return -1

        def get_closing_company():
            """Get current closing company from one-hot"""
            for c in range(NUM_COMPANIES):
                if state.as_numpy()[turn_offset + turn_closing_company + c] == 1.0:
                    return c
            return -1

        def format_company(company_id):
            """Format company info: NAME (face=$X, low=$L, high=$H, stars=S)"""
            if company_id < 0:
                return "None"
            name = COMPANY_NAMES[company_id]
            face = py_get_company_face_value(company_id)
            low = py_get_company_low_price(company_id)
            high = py_get_company_high_price(company_id)
            stars = py_get_company_stars(company_id)
            income = py_get_company_income(company_id)
            return f"{name} (id={company_id}, face=${face}, low=${low}, high=${high}, stars={stars}, income=${income})"

        def format_corp(corp_id):
            """Format corp name"""
            if corp_id < 0:
                return "None"
            return CORP_NAMES[corp_id]

        def get_player_turn_order(player_id):
            """Get player's turn order position from one-hot"""
            arr = state.as_numpy()
            player_base = players_offset + player_id * player_stride
            # turn_order is at offset 2 within player data (after cash and net_worth)
            for pos in range(num_players):
                if arr[player_base + 2 + pos] == 1.0:
                    return pos
            return -1

        def format_player_state(state, player_id):
            """Format detailed player state"""
            lines = []
            cash = state.get_player_cash_py(player_id)
            net_worth = state.get_player_net_worth_py(player_id)
            turn_order = get_player_turn_order(player_id)

            lines.append(f"  **Player {player_id}**: cash=${cash}, net_worth=${net_worth}, turn_order={turn_order}")

            # Companies owned
            owned_companies = []
            for c in range(NUM_COMPANIES):
                if state.player_owns_company_py(player_id, c):
                    owned_companies.append(COMPANY_NAMES[c])
            if owned_companies:
                lines.append(f"    Companies: {', '.join(owned_companies)}")

            # Shares owned
            shares = []
            for corp_id in range(NUM_CORPS):
                s = state.get_player_shares_py(player_id, corp_id)
                if s > 0:
                    pres = " (P)" if state.is_player_president_py(player_id, corp_id) else ""
                    shares.append(f"{CORP_NAMES[corp_id]}:{s}{pres}")
            if shares:
                lines.append(f"    Shares: {', '.join(shares)}")

            return '\n'.join(lines)

        def format_corp_state(state, corp_id):
            """Format detailed corp state"""
            if not state.is_corp_active_py(corp_id):
                return f"  **{CORP_NAMES[corp_id]}**: inactive"

            lines = []
            cash = state.get_corp_cash_py(corp_id)
            price_index = state.get_corp_price_index_py(corp_id)
            price = py_get_market_price(price_index) if 0 <= price_index < 27 else 0
            issued = state.get_corp_issued_shares_py(corp_id)
            unissued = state.get_corp_unissued_shares_py(corp_id)
            bank = state.get_corp_bank_shares_py(corp_id)
            recv = " (RECEIVERSHIP)" if state.is_corp_in_receivership_py(corp_id) else ""

            lines.append(f"  **{CORP_NAMES[corp_id]}**{recv}: cash=${cash}, price=${price} (idx={price_index}), issued={issued}, unissued={unissued}, bank={bank}")

            # Companies owned
            owned_companies = []
            for c in range(NUM_COMPANIES):
                if state.corp_owns_company_py(corp_id, c):
                    owned_companies.append(COMPANY_NAMES[c])
            if owned_companies:
                lines.append(f"    Companies: {', '.join(owned_companies)}")

            # Acquisition pile
            acq_companies = []
            for c in range(NUM_COMPANIES):
                if state.corp_has_acquisition_company_py(corp_id, c):
                    acq_companies.append(COMPANY_NAMES[c])
            if acq_companies:
                lines.append(f"    Acquisition pile: {', '.join(acq_companies)}")

            # President
            for p in range(state.num_players):
                if state.is_player_president_py(p, corp_id):
                    lines.append(f"    President: Player {p}")
                    break

            return '\n'.join(lines)

        def format_auction_state(state):
            """Format auction state if in auction"""
            company_id = state.get_auction_company_py()
            if company_id < 0:
                return ""
            price = state.get_auction_price_py()
            high_bidder = state.get_auction_high_bidder_py()
            starter = state.get_auction_starter_py()
            return f"  Auction: {format_company(company_id)}, current_bid=${price}, high_bidder=P{high_bidder}, starter=P{starter}"

        def format_game_state(state, step):
            """Format complete game state"""
            lines = []
            lines.append(f"## Step {step} - Phase: {PHASE_NAMES[state.phase]}")
            lines.append(f"Turn: {state.turn_number}, Active Player: {state.active_player}, CoO Level: {state.coo_level}, End Card Flipped: {state.get_end_card_flipped_py()}")
            lines.append("")

            # Players
            lines.append("### Players")
            for p in range(state.num_players):
                lines.append(format_player_state(state, p))
            lines.append("")

            # Corporations
            lines.append("### Corporations")
            for c in range(NUM_CORPS):
                lines.append(format_corp_state(state, c))
            lines.append("")

            # Foreign Investor
            fi_cash = state.get_fi_cash_py()
            fi_companies = []
            for c in range(NUM_COMPANIES):
                if state.fi_owns_company_py(c):
                    fi_companies.append(COMPANY_NAMES[c])
            lines.append(f"### Foreign Investor: cash=${fi_cash}")
            if fi_companies:
                lines.append(f"  Companies: {', '.join(fi_companies)}")
            lines.append("")

            # Companies for auction
            auction_companies = []
            for c in range(NUM_COMPANIES):
                if state.is_company_for_auction_py(c):
                    auction_companies.append(format_company(c))
            if auction_companies:
                lines.append("### Companies for Auction")
                for c in auction_companies:
                    lines.append(f"  - {c}")
                lines.append("")

            # Auction state (if applicable)
            auction_str = format_auction_state(state)
            if auction_str:
                lines.append("### Auction State")
                lines.append(auction_str)
                lines.append("")

            # Market availability (brief)
            taken_spaces = []
            for i in range(27):
                if not state.is_market_space_available_py(i):
                    taken_spaces.append(f"{py_get_market_price(i)}(idx={i})")
            if taken_spaces:
                lines.append(f"### Market (taken spaces): {', '.join(taken_spaces)}")
                lines.append("")

            return '\n'.join(lines)

        def format_action(action_idx, state):
            """Format an action in human-readable form"""
            phase, action_type, slot, corp_id, amount = decode_action_py(action_idx, state.num_players)
            action_name = ACTION_TYPE_NAMES.get(action_type, f"UNKNOWN({action_type})")
            player = state.active_player

            if action_type == 0:  # PASS
                return f"Player {player} passes"

            elif action_type == 1:  # AUCTION
                # Find the company for this slot
                company_id = -1
                slot_count = 0
                for c in range(NUM_COMPANIES):
                    if state.is_company_for_auction_py(c):
                        if slot_count == slot:
                            company_id = c
                            break
                        slot_count += 1
                if company_id >= 0:
                    face = py_get_company_face_value(company_id)
                    bid = face + amount
                    return f"Player {player} starts auction for {COMPANY_NAMES[company_id]} at ${bid} (face=${face} + offset={amount})"
                return f"Player {player} starts auction (slot={slot}, offset={amount})"

            elif action_type == 2:  # BUY_SHARE
                return f"Player {player} buys share of {format_corp(corp_id)}"

            elif action_type == 3:  # SELL_SHARE
                return f"Player {player} sells share of {format_corp(corp_id)}"

            elif action_type == 4:  # LEAVE_AUCTION
                return f"Player {player} leaves auction"

            elif action_type == 5:  # RAISE_BID
                company_id = state.get_auction_company_py()
                if company_id >= 0:
                    face = py_get_company_face_value(company_id)
                    new_bid = face + amount + 1
                    return f"Player {player} raises bid to ${new_bid} on {COMPANY_NAMES[company_id]}"
                return f"Player {player} raises bid (offset={amount})"

            elif action_type == 6:  # ACQ_PRICE
                acq_corp = state.get_acq_active_corp_py()
                target = state.get_acq_target_company_py()
                if target >= 0:
                    low = py_get_company_low_price(target)
                    price = low + amount
                    return f"{format_corp(acq_corp)} acquires {COMPANY_NAMES[target]} for ${price} (low=${low} + offset={amount})"
                return f"Acquisition at price offset={amount}"

            elif action_type == 7:  # ACQ_FI_HIGH
                acq_corp = state.get_acq_active_corp_py()
                target = state.get_acq_target_company_py()
                if target >= 0:
                    high = py_get_company_high_price(target)
                    return f"{format_corp(acq_corp)} acquires {COMPANY_NAMES[target]} from FI at high price ${high}"
                return f"FI acquisition at high price"

            elif action_type == 8:  # ACQ_FI_FACE
                acq_corp = state.get_acq_active_corp_py()
                target = state.get_acq_target_company_py()
                if target >= 0:
                    face = py_get_company_face_value(target)
                    return f"{format_corp(acq_corp)} (OS) acquires {COMPANY_NAMES[target]} from FI at face value ${face}"
                return f"FI acquisition at face value"

            elif action_type == 9:  # CLOSE
                closing_company_id = get_closing_company()
                if closing_company_id >= 0:
                    return f"President closes {COMPANY_NAMES[closing_company_id]}"
                return f"Close company"

            elif action_type == 10:  # DIVIDEND
                div_corp = get_dividend_corp()
                if div_corp >= 0:
                    return f"{format_corp(div_corp)} pays ${amount} dividend per share"
                return f"Pay ${amount} dividend per share"

            elif action_type == 11:  # ISSUE
                issue_corp_id = get_issue_corp()
                if issue_corp_id >= 0:
                    return f"{format_corp(issue_corp_id)} issues one share"
                return f"Corp issues one share"

            elif action_type == 12:  # IPO
                ipo_company_id = get_ipo_company()
                if ipo_company_id >= 0 and corp_id >= 0:
                    par_price = py_get_par_price(slot) if slot >= 0 and slot < 14 else -1
                    return f"Player {player} IPOs {COMPANY_NAMES[ipo_company_id]} into {format_corp(corp_id)} at par ${par_price}"
                return f"IPO into {format_corp(corp_id)} at par slot={slot}"

            return f"Action {action_idx}: {action_name} (slot={slot}, corp={corp_id}, amount={amount})"

        # Set up random seed
        np.random.seed(self.seed)

        # Create game state
        state = GameState(self.num_players)
        state.setup_new_game(shuffle_seed=self.seed)

        # Open output file
        print(f"Running debug game with {self.num_players} players, seed={self.seed}")
        print(f"Output will be written to: {self.output}")

        with open(self.output, 'w') as f:
            f.write(f"# Rolling Stock Stars - Debug Game Output\n\n")
            f.write(f"**Players**: {self.num_players}, **Seed**: {self.seed}\n\n")
            f.write("---\n\n")

            # Write initial state
            f.write("# Initial Game State\n\n")
            f.write(format_game_state(state, 0))
            f.write("\n---\n\n")

            step = 0
            while state.phase != PHASE_GAME_OVER and step < MAX_STEPS:
                # Get valid actions
                mask = get_valid_action_mask(state)
                valid_indices = np.where(mask == 1.0)[0]

                if len(valid_indices) == 0:
                    f.write(f"\n**ERROR**: No valid actions at step {step}, phase={PHASE_NAMES[state.phase]}\n")
                    break

                # Choose random action
                action = np.random.choice(valid_indices)

                # Write action
                step += 1
                f.write(f"# Action {step}\n\n")
                f.write(f"**{format_action(action, state)}** (action_idx={action})\n\n")
                f.write(f"Valid actions: {len(valid_indices)}\n\n")

                # Apply action
                apply_action(state, action)

                # Write state after action
                f.write(format_game_state(state, step))
                f.write("\n---\n\n")

            # Final summary
            f.write("# Game Over Summary\n\n")
            f.write(f"Total steps: {step}\n\n")
            f.write("## Final Scores\n\n")
            scores = state.get_final_scores()
            for rank, (player_id, net_worth) in enumerate(scores, 1):
                f.write(f"{rank}. Player {player_id}: ${net_worth}\n")

        print(f"Debug game completed after {step} steps")
        print(f"Output written to {self.output}")

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
        annotate=False,  # Generates HTML annotation files showing Python interaction
    ),
    cmdclass={
        'clean': CleanCommand,
        'benchmark': BenchmarkCommand,
        'debug_game': DebugGameCommand,
    },
    include_package_data=True,
    zip_safe=False,
)
