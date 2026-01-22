"""Custom exceptions for game driver."""


class ForcedActionLoopError(RuntimeError):
    """Raised when forced action loop exceeds iteration limit."""
    pass


class ZeroLegalActionsError(RuntimeError):
    """Raised when zero legal actions exist outside GAME_OVER phase."""
    pass
