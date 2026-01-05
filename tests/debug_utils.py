"""Debug utilities for test infrastructure."""

import os

# Module-level debug flag that can be set by pytest or environment
_debug_enabled = False


def set_debug_enabled(enabled):
    """Set the debug flag (called by conftest.py)."""
    global _debug_enabled
    _debug_enabled = enabled


def is_debug_enabled():
    """Check if debug mode is enabled (via --game-debug flag or RSS_DEBUG env var)."""
    return _debug_enabled or os.environ.get("RSS_DEBUG", "").lower() in ("1", "true", "yes")


def debug_print(*args, **kwargs):
    """Print only if debug mode is enabled."""
    if is_debug_enabled():
        print(*args, **kwargs)
