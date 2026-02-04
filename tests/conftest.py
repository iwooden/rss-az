"""pytest configuration for tests."""
import sys
from pathlib import Path

# Add project root to Python path so we can import core, entities, etc.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import fixtures from phases/conftest.py so they're available to test_integration.py
# (which was moved from tests/phases/ to tests/)
import pytest
from tests.phases.conftest import (
    game_state,
    invest_state,
    bid_state,
    trade_state,
    apply_and_track,
)

# Add closing_offer_state fixture
from tests.phases.conftest import closing_offer_state

# Re-export fixtures so they're available at root level
__all__ = [
    'game_state',
    'invest_state',
    'bid_state',
    'trade_state',
    'apply_and_track',
    'closing_offer_state',
]
