"""
End-to-end replay test using real 18xx.games data.

Replays a complete game from 18xx.games through our Cython game engine,
verifying game state invariants after every action and player net worths
at phase boundaries.
"""

import pytest
import re
from pathlib import Path

from state import GameState
from driver import apply_action
from actions import get_valid_action_mask, get_action_layout
from data import COMPANY_NAME_TO_ID, CORP_NAME_TO_ID, py_get_company_face_value, py_get_company_stars
from test_random_game import check_invariants
from test_common import StateBuilder

from replay_utils import (
    Game18xxParser, ActionTranslator, ParsedAction,
    AcquisitionMatcher, ClosingMatcher, PendingAcquisition,
    parse_net_worth_table,
    PHASE_INVEST, PHASE_BID_IN_AUCTION, PHASE_WRAP_UP, PHASE_ACQUISITION,
    PHASE_CLOSING, PHASE_INCOME, PHASE_DIVIDENDS, PHASE_END_CARD,
    PHASE_ISSUE_SHARES, PHASE_IPO, PHASE_GAME_OVER, PHASE_ABBREV,
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Data directory
DATA_DIR = Path(__file__).parent / "18xx_data"

# Phase names for debugging
PHASE_NAMES = {
    0: "INVEST", 1: "BID_IN_AUCTION", 2: "WRAP_UP", 3: "ACQUISITION",
    4: "CLOSING", 5: "INCOME", 6: "DIVIDENDS", 7: "END_CARD",
    8: "ISSUE_SHARES", 9: "IPO", 10: "GAME_OVER"
}


# =============================================================================
# GAME REPLAY ENGINE
# =============================================================================

class GameReplayEngine:
    """
    Replays an 18xx.games game through our Cython engine.

    Handles:
    - Setting up initial state with correct deck order
    - Translating 18xx actions to our action indices
    - Verifying invariants after each action
    - Verifying net worths at phase boundaries
    """

    def __init__(self, json_path: str, md_path: str):
        """
        Initialize replay engine.

        Args:
            json_path: Path to 18xx.games JSON file
            md_path: Path to markdown file with game log and net worth table
        """
        self.parser = Game18xxParser(json_path, md_path)
        self.md_path = md_path

        # Map player names to seats (must be before _parse_net_worth_table)
        self.player_name_to_seat = {p.name: p.seat for p in self.parser.players}
        self.player_id_to_seat = {p.id: p.seat for p in self.parser.players}

        # Parse expected net worths
        self.expected_net_worth = self._parse_net_worth_table()

        # State tracking
        self.state: GameState = None
        self.builder: StateBuilder = None
        self.translator: ActionTranslator = None
        self.layout: dict = None

        # Phase-specific state machines
        self.acq_matcher = AcquisitionMatcher()
        self.closing_matcher = ClosingMatcher()

        # Auto-pass tracking
        self.auto_pass_players: set[int] = set()

        # Debug info
        self.step = 0
        self.prev_turn = 0
        self.last_phase = None
        self.last_turn = 0

    def _parse_net_worth_table(self) -> dict[tuple[str, int], dict[int, int]]:
        """Parse net worth table and convert player names to seats."""
        raw_table = parse_net_worth_table(self.md_path)

        result = {}
        for (phase_abbrev, turn), name_to_nw in raw_table.items():
            seat_to_nw = {}
            for name, nw in name_to_nw.items():
                if name in self.player_name_to_seat:
                    seat_to_nw[self.player_name_to_seat[name]] = nw
            result[(phase_abbrev, turn)] = seat_to_nw

        return result

    def setup_initial_state(self) -> GameState:
        """
        Create and initialize game state with correct deck order.

        Returns:
            Initialized GameState ready for replay
        """
        num_players = self.parser.num_players
        self.state = GameState(num_players)
        self.builder = StateBuilder(self.state, num_players)
        self.translator = ActionTranslator(num_players)
        self.layout = get_action_layout(num_players)

        # Set phase to INVEST
        self.state.phase = PHASE_INVEST
        self.state.coo_level = 1  # No cost of ownership initially
        self.state.active_player = 0
        self.state.turn_number = 1
        self.state.consecutive_passes = 0

        # Initialize player cash and turn order using StateBuilder
        for player_id in range(num_players):
            self.builder.set_player_cash(player_id, 30)
            self.builder.set_player_turn_order(player_id, player_id)
            self.builder.set_player_net_worth(player_id, 30)

        # Initialize FI cash (FI starts with 4● per rules)
        self.builder.set_fi_cash(4)

        # Initialize market (all spaces available)
        self.builder.init_all_market_available()

        # Get deck order and initial auction from game data
        initial_auction = self.parser.get_initial_auction_companies()
        deck_reveals = self.parser.get_deck_order()

        # Set up initial auction companies
        for company_name in initial_auction:
            company_id = COMPANY_NAME_TO_ID[company_name]
            self.builder.set_company_for_auction(company_id, True)

        # Set up deck (deck_reveals is in reveal order, first revealed = top of deck)
        # setup_deck expects first element to be top of deck, which is correct
        deck_ids = [COMPANY_NAME_TO_ID[name] for name in deck_reveals]
        self.builder.setup_deck(deck_ids)

        # Mark companies not in deck or auction as removed
        in_game = set(COMPANY_NAME_TO_ID[name] for name in initial_auction)
        in_game.update(COMPANY_NAME_TO_ID[name] for name in deck_reveals)

        for company_id in range(36):
            if company_id not in in_game:
                self.builder.set_company_removed(company_id, True)

        return self.state

    def get_available_auction_companies(self) -> list[int]:
        """Get list of company IDs currently available for auction."""
        available = []
        for company_id in range(36):
            if self.builder.has_company_for_auction(company_id):
                available.append(company_id)
        return available

    def run_replay(self):
        """
        Execute full game replay with verification.

        Raises AssertionError on any mismatch or invariant violation.
        """
        self.setup_initial_state()

        for action in self.parser.iterate_actions():
            self._process_action(action)

        # Verify final result
        self._verify_final_result()

    def _process_action(self, action: ParsedAction):
        """Process a single action from the 18xx action stream."""
        from conftest import debug_print

        # Debug: log every action with phase and market state
        idx6 = self.state.is_market_space_available_py(6)
        debug_print(f"Processing action #{action.original_id}: {action.action_type} {action.entity_id} ({action.entity_type}) - Engine: {PHASE_NAMES[self.state.phase]} Turn {self.state.turn_number}, idx6={idx6}")

        # Check for phase transition before processing
        current_phase = self.state.phase
        current_turn = self.state.turn_number

        if current_phase != self.last_phase or current_turn != self.last_turn:
            if self.last_phase is not None:
                # Don't trigger phase_end when transitioning INVEST <-> BID_IN_AUCTION
                # (BID_IN_AUCTION is a sub-phase of INVEST)
                is_invest_auction_transition = (
                    (self.last_phase == PHASE_INVEST and current_phase == PHASE_BID_IN_AUCTION) or
                    (self.last_phase == PHASE_BID_IN_AUCTION and current_phase == PHASE_INVEST)
                )
                if not is_invest_auction_transition:
                    self._on_phase_end(self.last_phase, self.last_turn)
            self._on_phase_start(current_phase, current_turn)
            self.last_phase = current_phase
            self.last_turn = current_turn

        # Skip if game is over
        if self.state.phase == PHASE_GAME_OVER:
            return

        # Translate and apply action
        engine_action = self._translate_action(action)

        if engine_action is None:
            # Action doesn't need engine action (e.g., buffered acquisition offer)
            return

        # Verify action is valid
        mask = get_valid_action_mask(self.state)
        context = self._get_context(action)

        assert mask[engine_action] == 1.0, \
            f"Translated action {engine_action} not valid at {context}"

        # Debug: show market state before IPO actions
        if action.action_type == "par":
            from state import get_market_index
            debug_print(f"Before apply: action={engine_action}, idx6={self.state.is_market_space_available_py(6)}")

        # Apply action
        apply_action(self.state, engine_action)
        self.step += 1

        # Debug: show market state after IPO actions
        if action.action_type == "par":
            from state import get_market_index
            debug_print(f"After apply: idx6={self.state.is_market_space_available_py(6)}")

        # Debug: show cash after acquisition actions
        if action.action_type == "offer" and self.state.phase != PHASE_ACQUISITION:
            # We just left ACQUISITION phase after this offer
            debug_print(f"  Post-acq: Player cash={[self.state.get_player_cash_py(i) for i in range(4)]}")

        # Debug: track auction state during Turn 4-5 transition
        if self.state.turn_number in (4, 5) and action.original_id in range(165, 180):
            from data import COMPANY_NAMES as COMP_NAMES
            avail = self.get_available_auction_companies()
            names = [COMP_NAMES[c] for c in avail]
            debug_print(f"  POST-ACTION: Turn={self.state.turn_number} Phase={PHASE_NAMES.get(self.state.phase)} Auction={names}")

        # Check invariants
        context = f"Step {self.step}, Phase {PHASE_NAMES[self.state.phase]}, Turn {self.state.turn_number}"
        self.prev_turn = check_invariants(self.state, self.prev_turn, context)

    def _translate_action(self, action: ParsedAction) -> int | None:
        """
        Translate 18xx action to engine action index.

        Returns None if the action should be buffered/skipped.

        Phase Synchronization:
        Our engine auto-advances through phases when there's nothing to do
        (CLOSING, INCOME, DIVIDENDS, END_CARD). But 18xx records explicit
        pass actions for each phase. We detect action type mismatches and
        skip stale 18xx actions.
        """
        phase = self.state.phase
        atype = action.action_type
        etype = action.entity_type

        # Phase synchronization: Skip actions that belong to phases
        # our engine has already auto-advanced through
        # Note: This may also advance the engine (e.g., auto-pass IPO)
        skip = self._should_skip_for_phase_sync(phase, action)
        if skip:
            return None

        # Re-read phase in case sync function advanced it
        phase = self.state.phase

        # INVEST phase
        if phase == PHASE_INVEST:
            return self._translate_invest_action(action)

        # BID_IN_AUCTION phase
        if phase == PHASE_BID_IN_AUCTION:
            return self._translate_bid_action(action)

        # ACQUISITION phase
        if phase == PHASE_ACQUISITION:
            return self._translate_acquisition_action(action)

        # CLOSING phase
        if phase == PHASE_CLOSING:
            return self._translate_closing_action(action)

        # DIVIDENDS phase
        if phase == PHASE_DIVIDENDS:
            return self._translate_dividends_action(action)

        # ISSUE_SHARES phase
        if phase == PHASE_ISSUE_SHARES:
            return self._translate_issue_action(action)

        # IPO phase
        if phase == PHASE_IPO:
            return self._translate_ipo_action(action)

        raise ValueError(f"Unknown phase {phase} for action {action}")

    def _should_skip_for_phase_sync(self, engine_phase: int, action: ParsedAction) -> bool:
        """
        Check if an 18xx action should be skipped due to phase mismatch.

        Our engine auto-advances through some phases. When 18xx has actions
        for phases we've already passed, skip them.
        """
        from conftest import debug_print
        atype = action.action_type
        etype = action.entity_type

        # DIVIDENDS phase - skip CLOSING passes
        if engine_phase == PHASE_DIVIDENDS:
            # Player passes from CLOSING/ACQUISITION phase
            if atype == "pass" and etype == "player":
                debug_print(f"SKIP: pass action in DIVIDENDS (was CLOSING/ACQ)")
                return True

        # ISSUE_SHARES phase - skip CLOSING/DIVIDENDS actions
        if engine_phase == PHASE_ISSUE_SHARES:
            # Player passes from CLOSING phase
            if atype == "pass" and etype == "player":
                debug_print(f"SKIP: pass action in ISSUE_SHARES (was CLOSING)")
                return True

            # dividend actions from DIVIDENDS phase - should have been processed
            if atype == "dividend":
                debug_print(f"SKIP: dividend action in ISSUE_SHARES - {action.entity_id} pays ${action.amount}")
                return True

        # IPO phase - handle phase desync
        if engine_phase == PHASE_IPO:
            if atype == "dividend":
                return True
            # Corp passes are ISSUE_SHARES - might already be processed
            if atype == "pass" and etype == "corporation":
                return True
            # Player passes in IPO phase might be "I pass on all my IPOs"
            if atype == "pass" and etype == "player":
                return True  # Skip - we'll handle IPO company by company

            # If 18xx has moved to next turn (INVEST action), we need to
            # auto-advance through remaining IPO companies
            if atype == "bid" and etype == "player":
                # This is an INVEST action - 18xx has moved to next turn
                # Auto-pass remaining IPO companies
                self._auto_pass_remaining_ipo()
                return False  # Don't skip - process in INVEST phase

        return False

    def _auto_pass_remaining_ipo(self):
        """Auto-pass all remaining IPO companies to advance to INVEST phase."""
        from driver import apply_action
        from actions import get_valid_action_mask

        while self.state.phase == PHASE_IPO:
            mask = get_valid_action_mask(self.state)
            ipo_pass = self.layout['ipo_pass']

            if mask[ipo_pass] == 1.0:
                apply_action(self.state, ipo_pass)
            else:
                # No more IPO actions possible - should transition automatically
                break

    def _translate_invest_action(self, action: ParsedAction) -> int:
        """Translate INVEST phase action."""
        atype = action.action_type

        if atype == "pass":
            return self.translator.translate_pass_invest()

        if atype == "bid":
            # Starting an auction
            available = self.get_available_auction_companies()
            return self.translator.translate_start_auction(
                action.company, action.price, available
            )

        if atype == "buy_shares":
            from conftest import debug_print
            from data import py_get_market_price
            # Extract corp name from shares list (format: "CORP_N")
            share_id = action.shares[0]
            corp_name = share_id.split('_')[0]
            corp_id = CORP_NAME_TO_ID[corp_name]
            bank_shares = self.state.get_corp_bank_shares_py(corp_id)
            player_cash = self.state.get_player_cash_py(self.state.active_player)
            price_idx = self.state.get_corp_price_index_py(corp_id)
            share_price = py_get_market_price(price_idx)
            debug_print(f"BUY: {corp_name} (ID={corp_id}): bank_shares={bank_shares}, player_cash=${player_cash}, price_idx={price_idx}, share_price=${share_price}")
            return self.translator.translate_buy_share(corp_name)

        if atype == "sell_shares":
            share_id = action.shares[0]
            corp_name = share_id.split('_')[0]
            return self.translator.translate_sell_share(corp_name)

        raise ValueError(f"Unknown INVEST action: {atype}")

    def _translate_bid_action(self, action: ParsedAction) -> int:
        """Translate BID_IN_AUCTION phase action."""
        atype = action.action_type

        if atype == "pass":
            return self.translator.translate_leave_auction()

        if atype == "bid":
            # Raising bid
            auction_company_id = self.state.get_auction_company_py()
            return self.translator.translate_raise_bid(action.price, auction_company_id)

        raise ValueError(f"Unknown BID_IN_AUCTION action: {atype}")

    def _translate_acquisition_action(self, action: ParsedAction) -> int | None:
        """
        Translate ACQUISITION phase action.

        In 18xx, acquisitions work differently than our engine:
        - "offer" where player owns company AND is corp president executes immediately
        - "pass" means "I'm done making acquisitions" (not declining a specific one)
        - Once all players pass, phase ends

        Our engine presents acquisitions one at a time. We need to match
        18xx offers to our engine's current acquisition opportunity.
        """
        atype = action.action_type

        if atype == "offer":
            from conftest import debug_print
            # In 18xx, self-deals execute immediately
            # Check if this matches what our engine is currently offering
            target_company = self.state.get_acq_target_company_py()
            target_corp = self.state.get_acq_active_corp_py()

            # Get company and corp IDs from the action
            company_id = COMPANY_NAME_TO_ID.get(action.company, -1)
            corp_id = CORP_NAME_TO_ID.get(action.corporation, -1)

            debug_print(f"ACQ offer: {action.company}→{action.corporation} at ${action.price}, engine has company={target_company} corp={target_corp}")

            if target_company == company_id and target_corp == corp_id:
                # This matches our engine's current offer - translate the price
                low_price = self._get_acquisition_low_price(company_id)
                debug_print(f"ACQ: MATCH - executing at ${action.price} (low=${low_price}, offset={action.price - low_price})")
                debug_print(f"  Pre-acq: Player cash={[self.state.get_player_cash_py(i) for i in range(4)]}")
                debug_print(f"  Corp {action.corporation} cash=${self.state.get_corp_cash_py(corp_id)}")
                return self.translator.translate_acquisition_price(action.price, low_price)
            else:
                # Doesn't match current engine offer - buffer it for later
                debug_print(f"ACQ: NO MATCH - buffering")
                self.acq_matcher.add_offer(
                    action.company, action.corporation, action.price,
                    from_fi=(action.entity_type != "player")
                )
                return None

        if atype == "respond":
            # Respond actions are for offers that require confirmation
            # (e.g., buying from another player)
            if action.accept:
                # Check if this matches current engine offer
                target_company = self.state.get_acq_target_company_py()
                company_id = COMPANY_NAME_TO_ID.get(action.company, -1)
                if target_company == company_id:
                    # Find the buffered offer price
                    price = self.acq_matcher.get_offer_price(action.company, action.corporation)
                    if price is not None:
                        return self.translator.translate_acquisition_price(
                            price, self._get_acquisition_low_price(company_id)
                        )
            return None

        if atype == "pass":
            from conftest import debug_print
            # In 18xx, pass means "I'm done with my acquisitions" - but 18xx
            # processes acquisitions in parallel while our engine does them sequentially.
            # We need to match buffered offers to our engine's sequential presentation.

            target_company = self.state.get_acq_target_company_py()
            target_corp = self.state.get_acq_active_corp_py()
            active_player = self.state.active_player
            passing_player = action.entity_id

            # Convert 18xx player ID to our seat number
            passing_seat = self.player_id_to_seat.get(passing_player, -1)

            debug_print(f"ACQ pass: from player {passing_player} (seat {passing_seat})")
            debug_print(f"  Engine: company={target_company} corp={target_corp}, active_player={active_player}, pending={len(self.acq_matcher.pending)}")

            # Only apply pass if it's from the player the engine is waiting for
            # (or if we have a matching buffered offer)
            if self.acq_matcher.has_pending_offers():
                offer = self.acq_matcher.get_matching_offer(target_company, target_corp)
                if offer:
                    low_price = self._get_acquisition_low_price(target_company)
                    debug_print(f"  -> MATCHED pending offer for {offer.company_name}→{offer.corp_name} at ${offer.price} (low=${low_price})")
                    debug_print(f"  Pre-acq: Player cash={[self.state.get_player_cash_py(i) for i in range(4)]}")
                    return self.translator.translate_acquisition_price(offer.price, low_price)

            # No matching buffered offer - only pass if this is from the active player
            if passing_seat == active_player:
                debug_print(f"  -> No match, passing on engine offer (same player)")
                return self.translator.translate_acquisition_pass()
            else:
                debug_print(f"  -> No match, but pass is from different player - skipping")
                return None

        raise ValueError(f"Unknown ACQUISITION action: {atype}")

    def _get_acquisition_low_price(self, company_id: int) -> int:
        """Get the low price for acquisition (about half face value)."""
        from data import py_get_company_low_price
        return py_get_company_low_price(company_id)

    def _translate_closing_action(self, action: ParsedAction) -> int:
        """Translate CLOSING phase action."""
        atype = action.action_type

        if atype == "pass":
            return self.translator.translate_close_pass()

        if atype == "sell_company":
            # This is a close action in 18xx terminology
            return self.translator.translate_close()

        raise ValueError(f"Unknown CLOSING action: {atype}")

    def _translate_dividends_action(self, action: ParsedAction) -> int:
        """Translate DIVIDENDS phase action."""
        from conftest import debug_print
        atype = action.action_type

        if atype == "dividend":
            debug_print(f"DIV action: {action.entity_id} pays ${action.amount} per share")
            return self.translator.translate_dividend(action.amount)

        raise ValueError(f"Unknown DIVIDENDS action: {atype}")

    def _translate_issue_action(self, action: ParsedAction) -> int:
        """Translate ISSUE_SHARES phase action."""
        atype = action.action_type

        if atype == "pass":
            return self.translator.translate_issue_pass()

        if atype == "sell_shares":
            # Corp issuing a share
            return self.translator.translate_issue()

        raise ValueError(f"Unknown ISSUE_SHARES action: {atype}")

    def _translate_ipo_action(self, action: ParsedAction) -> int | None:
        """Translate IPO phase action.

        In 18xx, IPO has two stages:
        1. Players in turn order can choose to IPO or pass (entity_type=player)
        2. Then companies are processed in face value order (entity_type=company)

        Our engine just presents each company to its owner in descending face value
        order. The player passes don't have an equivalent - skip them.
        """
        from conftest import debug_print
        from phases.ipo import get_phase_handler
        from data import COMPANY_NAMES, CORP_NAMES

        atype = action.action_type

        # Debug: show current IPO state
        ipo_handler = get_phase_handler(self.parser.num_players)
        current_company = ipo_handler.get_current_company(self.state)
        debug_print(f"IPO: action={atype}, entity={action.entity_id}, entity_type={action.entity_type}")
        debug_print(f"IPO: engine current_company={current_company} ({COMPANY_NAMES[current_company] if current_company >= 0 else 'none'})")

        if atype == "pass":
            # Skip player passes - they don't map to our engine
            if action.entity_type == "player":
                return None  # Skip this action
            # Company passes map to IPO pass
            return self.translator.translate_ipo_pass()

        if atype == "par":
            # Get par price from share_price_str field (format: "price,row,col")
            # Example: "10,0,6" means par price 10
            if not action.share_price_str:
                raise ValueError(f"par action missing share_price: {action}")

            price_parts = action.share_price_str.split(',')
            par_price = int(price_parts[0])

            # Get star tier from the converting company
            company_id = COMPANY_NAME_TO_ID[action.entity_id]
            star_tier = py_get_company_stars(company_id)
            corp_id = CORP_NAME_TO_ID[action.corporation]

            debug_print(f"IPO: par company={action.entity_id}(ID={company_id}), corp={action.corporation}(ID={corp_id}), price={par_price}, stars={star_tier}")

            # Check if engine's current company matches
            if current_company != company_id:
                debug_print(f"IPO: MISMATCH! Engine has company {current_company}, 18xx wants {company_id}")

            # Check corp active status
            corp_active = self.state.is_corp_active_py(corp_id)
            debug_print(f"IPO: corp {action.corporation} active={corp_active}")

            # Check valid IPO options
            valid_options = ipo_handler.get_valid_ipo_options(self.state)
            debug_print(f"IPO: valid options = {valid_options}")

            # Check can_ipo
            par_slot = self.translator._get_par_slot(par_price, star_tier)
            can = ipo_handler.can_ipo(self.state, corp_id, par_slot)
            debug_print(f"IPO: can_ipo(corp={corp_id}, par_slot={par_slot}) = {can}")

            # Check player cash
            player_id = self.state.active_player
            player_cash = self.state.get_player_cash_py(player_id)
            face_value = py_get_company_face_value(company_id)
            debug_print(f"IPO: player {player_id} cash=${player_cash}, company face=${face_value}")

            # Check market availability for all tier 1 prices
            from state import get_market_index
            from data import py_get_par_price
            debug_print(f"IPO: checking market for tier {star_tier}")
            for par_idx in range(5):  # First 5 par indices (tier 1 valid)
                p = py_get_par_price(par_idx)
                mkt_idx = get_market_index(p)
                mkt_avail = self.state.is_market_space_available_py(mkt_idx)
                debug_print(f"IPO: par_idx={par_idx} price=${p} mkt_idx={mkt_idx} available={mkt_avail}")

            return self.translator.translate_ipo(action.corporation, par_price, star_tier)

        raise ValueError(f"Unknown IPO action: {atype}")

    def _on_phase_start(self, phase: int, turn: int):
        """Called when entering a new phase."""
        from conftest import debug_print

        debug_print(f"=== Phase Start: {PHASE_NAMES.get(phase, phase)} Turn {turn} ===")

        # Debug player cash and share ownership at key phases
        if phase in (PHASE_DIVIDENDS, PHASE_IPO, PHASE_INVEST):
            debug_print(f"  Player cash: {[self.state.get_player_cash_py(i) for i in range(4)]}")
            if phase == PHASE_INVEST:
                from data import COMPANY_NAMES as COMP_NAMES
                available = self.get_available_auction_companies()
                names = [COMP_NAMES[c] for c in available]
                debug_print(f"  Auction companies: {names} (IDs: {available})")
                # Show FI state
                fi_cash = self.state.get_fi_cash_py()
                fi_companies = [c for c in range(36) if self.state.fi_owns_company_py(c)]
                fi_names = [COMP_NAMES[c] for c in fi_companies]
                debug_print(f"  FI: cash=${fi_cash}, companies={fi_names}")
            # Show player company ownership at DIVIDENDS start
            if phase == PHASE_DIVIDENDS:
                coo = self.state.coo_level
                debug_print(f"  CoO level: {coo}")
                for p in range(4):
                    companies = []
                    for c in range(36):
                        if self.state.player_owns_company_py(p, c):
                            companies.append(c)
                    if companies:
                        debug_print(f"  P{p} owns companies: {companies}")

        # Debug DIVIDENDS phase
        if phase == PHASE_DIVIDENDS:
            # Show all active corps and their dividend potential
            for corp_id in range(8):
                if self.state.is_corp_active_py(corp_id):
                    cash = self.state.get_corp_cash_py(corp_id)
                    issued = self.state.get_corp_issued_shares_py(corp_id)
                    max_div = cash // issued if issued > 0 else 0
                    corp_name = list(CORP_NAME_TO_ID.keys())[list(CORP_NAME_TO_ID.values()).index(corp_id)]
                    # Show who owns shares
                    owners = []
                    for p in range(4):
                        shares = self.state.get_player_shares_py(p, corp_id)
                        if shares > 0:
                            owners.append(f"P{p}:{shares}")
                    debug_print(f"  DIV: Corp {corp_name} - cash=${cash}, issued={issued}, max_div={max_div}, owners={owners}")

        # Clear phase-specific state
        if phase == PHASE_ACQUISITION:
            self.acq_matcher.clear()
            # Debug: show player company ownership at acquisition start
            for p in range(4):
                companies = []
                for c in range(36):
                    if self.state.player_owns_company_py(p, c):
                        name = list(COMPANY_NAME_TO_ID.keys())[list(COMPANY_NAME_TO_ID.values()).index(c)]
                        companies.append(name)
                if companies:
                    debug_print(f"  P{p} owns: {companies}")
        elif phase == PHASE_CLOSING:
            self.closing_matcher.clear()

    def _on_phase_end(self, phase: int, turn: int):
        """Called when exiting a phase."""
        from conftest import debug_print

        # BID_IN_AUCTION is a sub-phase of INVEST, skip net worth check
        if phase == PHASE_BID_IN_AUCTION:
            return

        # Skip phases not in the net worth table
        phase_abbrev = PHASE_ABBREV.get(phase)
        if phase_abbrev is None:
            return

        key = (phase_abbrev, turn)

        if key in self.expected_net_worth:
            expected = self.expected_net_worth[key]
            for seat, expected_nw in expected.items():
                actual_nw = self.state.get_player_net_worth_py(seat)
                if actual_nw != expected_nw:
                    debug_print(f"NET WORTH MISMATCH at {phase_abbrev} {turn}: player {seat} has ${actual_nw}, expected ${expected_nw}")
                assert actual_nw == expected_nw, \
                    f"Net worth mismatch at {phase_abbrev} {turn}: " \
                    f"player {seat} has ${actual_nw}, expected ${expected_nw}"

    def _verify_final_result(self):
        """Verify final game result matches expected."""
        expected_result = self.parser.get_final_result()

        for player_id, expected_nw in expected_result.items():
            seat = self.player_id_to_seat[player_id]
            actual_nw = self.state.get_player_net_worth_py(seat)
            player_name = self.parser.players[seat].name

            assert actual_nw == expected_nw, \
                f"Final net worth mismatch for {player_name}: " \
                f"got ${actual_nw}, expected ${expected_nw}"

    def _get_context(self, action: ParsedAction) -> str:
        """Get debug context string for an action."""
        return (
            f"Action #{action.original_id} ({action.action_type}), "
            f"Phase {PHASE_NAMES.get(self.state.phase, self.state.phase)}, "
            f"Turn {self.state.turn_number}, Step {self.step}"
        )


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_18xx_game_1():
    """
    Replay 18xx_game_1 through our engine.

    This test verifies:
    1. All actions translate correctly
    2. Game invariants hold after every action
    3. Player net worths match at phase boundaries
    4. Final scores match the 18xx result
    """
    engine = GameReplayEngine(
        str(DATA_DIR / "18xx_game_1.json"),
        str(DATA_DIR / "18xx_game_1.md"),
    )
    engine.run_replay()


# =============================================================================
# DEBUG HELPER
# =============================================================================

if __name__ == "__main__":
    """Run replay with extra debugging output."""
    import sys

    engine = GameReplayEngine(
        str(DATA_DIR / "18xx_game_1.json"),
        str(DATA_DIR / "18xx_game_1.md"),
    )

    try:
        engine.run_replay()
        print("Replay completed successfully!")
    except AssertionError as e:
        print(f"Replay failed: {e}")
        print(f"State: Phase={PHASE_NAMES.get(engine.state.phase)}, "
              f"Turn={engine.state.turn_number}, Step={engine.step}")
        sys.exit(1)
