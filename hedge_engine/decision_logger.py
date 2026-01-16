from __future__ import annotations

import asyncio
from typing import Any, Dict

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .config import Settings
from .circuit_breaker import CircuitBreaker, CircuitBreakerOpenError

__all__ = ["DecisionLogger"]


class DecisionLogger:
    """Asynchronous Postgres decision logger with in-memory fallback queue."""

    _engine: AsyncEngine | None = None
    _queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
    _cb = CircuitBreaker(max_failures=3, reset_timeout=30.0)

    @classmethod
    def _build_engine(cls) -> AsyncEngine:
        settings = Settings()
        dsn = settings.db_dsn
        # Convert sync psycopg2 DSN → asyncpg DSN lazily if necessary.
        dsn = dsn.replace("+psycopg2", "+asyncpg")
        logger.debug("Creating async engine for DSN={}".format(dsn))
        return create_async_engine(dsn, pool_size=5, pool_recycle=1800, echo=False)

    @classmethod
    def _get_engine(cls) -> AsyncEngine:
        if cls._engine is None:
            cls._engine = cls._build_engine()
        return cls._engine

    # ---------------------------- public API ---------------------------- #

    @classmethod
    async def log(cls, record: Dict[str, Any]) -> None:  # noqa: D401 – imperative style
        """Persist *record* to Postgres; enqueue on error for later retry."""
        try:
            async def _write():
                engine = cls._get_engine()
                async with engine.begin() as conn:
                    await conn.execute(
                        text(
                            "INSERT INTO hedge_decisions(asset, amount_usd, hedge_pct, confidence, ts_ms) "
                            "VALUES (:asset, :amount_usd, :hedge_pct, :confidence, :ts_ms)"
                        ),
                        record,
                    )

            await cls._cb.async_call(_write)
        except CircuitBreakerOpenError:
            # Circuit open; enqueue directly
            await cls._queue.put(record)
            logger.warning("Circuit open – queued record without attempt")
        except Exception as exc:  # pragma: no cover – broad by design for resiliency
            logger.warning("Decision log insert failed – queuing for retry: {}", exc)
            await cls._queue.put(record)

    @classmethod
    async def flush_queue(cls) -> int:
        """Attempt to flush queued records; returns remaining queue length."""
        remaining: int = cls._queue.qsize()
        if remaining == 0:
            return 0
        # Drain snapshot to avoid infinite loop if writes keep failing.
        snapshot: list[Dict[str, Any]] = [cls._queue.get_nowait() for _ in range(remaining)]
        for rec in snapshot:
            try:
                await cls.log(rec)
            except Exception:
                # If persisting still fails, push back and abort to avoid busy loop.
                await cls._queue.put(rec)
                break
        return cls._queue.qsize() 