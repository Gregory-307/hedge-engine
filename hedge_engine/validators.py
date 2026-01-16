class StaleDataError(RuntimeError):
    """Raised when data is older than allowed staleness."""
