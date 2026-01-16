import asyncio
from hedge_engine.decision_logger import DecisionLogger


def test_queue_on_failure(monkeypatch):
    record = {
        "asset": "BTC",
        "amount_usd": 1.0,
        "hedge_pct": 0.1,
        "confidence": 0.8,
        "ts_ms": 1,
    }

    async def raise_exc():  # noqa: D401 – dummy coroutine raising DB error
        raise RuntimeError("DB unavailable")

    # Patch _get_engine to a function that raises to simulate outage
    def _broken_get_engine():
        raise RuntimeError("DB unavailable")

    monkeypatch.setattr(DecisionLogger, "_get_engine", _broken_get_engine)

    # Clear any existing items in queue for isolated assertion
    DecisionLogger._queue = asyncio.Queue()  # type: ignore[assignment]

    async def run():
        await DecisionLogger.log(record)

    asyncio.run(run())

    assert DecisionLogger._queue.qsize() == 1 