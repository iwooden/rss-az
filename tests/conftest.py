"""Pytest configuration and fixtures for debug support."""

import os
import pytest

# Module-level debug flag that can be checked without fixtures
_debug_enabled = False


def pytest_addoption(parser):
    """Add --game-debug command line option."""
    parser.addoption(
        "--game-debug",
        action="store_true",
        default=False,
        help="Enable debug mode for game driver (records action history)",
    )


def pytest_configure(config):
    """Set module-level debug flag based on command line option."""
    global _debug_enabled
    _debug_enabled = config.getoption("--game-debug", default=False)


def is_debug_enabled():
    """Check if debug mode is enabled (via --game-debug flag or RSS_DEBUG env var)."""
    return _debug_enabled or os.environ.get("RSS_DEBUG", "").lower() in ("1", "true", "yes")


def debug_print(*args, **kwargs):
    """Print only if debug mode is enabled."""
    if is_debug_enabled():
        print(*args, **kwargs)


@pytest.fixture
def debug_enabled(request):
    """Returns True if --game-debug flag was passed."""
    return request.config.getoption("--game-debug")


@pytest.fixture
def debug_driver(debug_enabled):
    """
    Get a GameDriver with debug enabled.

    Usage in tests:
        def test_something(debug_driver):
            driver = debug_driver(num_players=3)
            # ... run game actions ...
            if driver.debug:
                print(driver.dump_history())
    """
    from driver import GameDriver

    def _make_driver(num_players=3):
        driver = GameDriver(num_players)
        if debug_enabled:
            driver.enable_debug()
        return driver

    return _make_driver


@pytest.fixture(autouse=True)
def auto_dump_on_failure(request, debug_enabled):
    """
    Automatically dump driver history on test failure when --game-debug is enabled.

    This fixture runs after each test and checks if it failed. If debug mode
    is enabled and the test has a 'driver' attribute or local variable,
    it will print the action history.
    """
    yield

    # Only process on failure with debug enabled
    if not debug_enabled:
        return

    if not hasattr(request.node, 'rep_call') or request.node.rep_call is None:
        return

    if not request.node.rep_call.failed:
        return

    # Try to find driver in test's local variables
    # This is a best-effort approach - tests should explicitly dump if needed
    print("\n=== DEBUG: Test failed with --game-debug enabled ===")
    print("Use driver.dump_history() in your test for detailed action history.")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Store test result for use in auto_dump_on_failure fixture."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)
