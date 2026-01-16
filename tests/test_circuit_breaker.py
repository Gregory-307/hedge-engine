import asyncio
import pytest
from hedge_engine.decision_logger import DecisionLogger
from hedge_engine.circuit_breaker import CircuitBreakerOpenError


@pytest.mark.asyncio
async def test_circuit_breaker_trips(monkeypatch):
    record = {
        "asset": "BTC",
        "amount_usd": 1.0,
        "hedge_pct": 0.1,
        "confidence": 0.8,
        "ts_ms": 1,
    }

    # Patch _get_engine to always raise to simulate DB failure
    def broken_engine():
        raise RuntimeError("db down")

    monkeypatch.setattr(DecisionLogger, "_get_engine", broken_engine)

    # Reset DecisionLogger internal state for deterministic test
    DecisionLogger._cb = DecisionLogger._cb.__class__(max_failures=2, reset_timeout=60)  # type: ignore
    DecisionLogger._queue = asyncio.Queue()  # type: ignore

    # First attempt should raise internally but not propagate
    await DecisionLogger.log(record)
    assert DecisionLogger._queue.qsize() == 1  # failure queued

    # Second attempt exceeds failure threshold -> circuit opens
    await DecisionLogger.log(record)
    assert DecisionLogger._queue.qsize() == 2

    # Third attempt should immediately see circuit open and queue without engine access
    await DecisionLogger.log(record)
    assert DecisionLogger._queue.qsize() == 3

    # Ensure circuit breaker state is OPEN
    with pytest.raises(CircuitBreakerOpenError):
        await DecisionLogger._cb.async_call(lambda: None)  # type: ignore
