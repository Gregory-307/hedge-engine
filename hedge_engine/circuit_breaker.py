from __future__ import annotations

import time
from typing import Callable, Awaitable, TypeVar, ParamSpec

from prometheus_client import Gauge
from loguru import logger

__all__ = ["CircuitBreaker", "CircuitBreakerOpenError"]

_P = ParamSpec("_P")
_T = TypeVar("_T")

_CIRCUIT_STATE = Gauge(
    "hedge_decision_logger_circuit_state",
    "Circuit breaker state for decision logger (0=closed,1=open)",
)


class CircuitBreakerOpenError(RuntimeError):
    """Raised when the circuit is open and calls are short-circuited."""


class CircuitBreaker:
    """Simple count-based circuit breaker with time-based reset."""

    def __init__(self, max_failures: int = 3, reset_timeout: float = 30.0):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self._failures = 0
        self._state = "CLOSED"  # CLOSED, OPEN
        self._opened_at: float | None = None
        _CIRCUIT_STATE.set(0)

    # ---------------------------------------------------------------------
    def _trip(self) -> None:
        self._state = "OPEN"
        self._opened_at = time.monotonic()
        _CIRCUIT_STATE.set(1)
        logger.warning("Circuit breaker OPEN (failures >= {})", self.max_failures)

    def _reset(self) -> None:
        self._state = "CLOSED"
        self._failures = 0
        self._opened_at = None
        _CIRCUIT_STATE.set(0)
        logger.info("Circuit breaker reset to CLOSED")

    # ------------------------------------------------------------------
    def call(self, func: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:  # type: ignore[misc]
        if self._state == "OPEN":
            if self._opened_at and (time.monotonic() - self._opened_at) >= self.reset_timeout:
                # After timeout, attempt single call (half-open) and reset on success/failure.
                self._state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError

        try:
            result = func(*args, **kwargs)
        except Exception:
            self._failures += 1
            if self._failures >= self.max_failures:
                self._trip()
            raise
        else:
            if self._state == "HALF_OPEN":
                self._reset()
            self._failures = 0
            return result

    # Async helpers ------------------------------------------------------
    async def async_call(
        self, func: Callable[_P, Awaitable[_T]], *args: _P.args, **kwargs: _P.kwargs
    ) -> _T:  # type: ignore[misc]
        if self._state == "OPEN":
            if self._opened_at and (time.monotonic() - self._opened_at) >= self.reset_timeout:
                self._state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError

        try:
            result = await func(*args, **kwargs)
        except Exception:
            self._failures += 1
            if self._failures >= self.max_failures:
                self._trip()
            raise
        else:
            if self._state == "HALF_OPEN":
                self._reset()
            self._failures = 0
            return result 