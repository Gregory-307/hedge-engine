"""
Backtesting framework for the signal-based hedge engine.

Validates signals on historical data with realistic assumptions:
- Transaction costs
- Position limits
- Proper train/test splits

Example:
    from hedge_engine.backtest import Backtest, fetch_crypto_data

    # Fetch real BTC data
    prices = fetch_crypto_data("bitcoin", days=365)

    # Define signal generators
    def momentum_signal(prices, i, lookback=20):
        if i < lookback:
            return 0.0
        returns = (prices[i] - prices[i - lookback]) / prices[i - lookback]
        return max(-1, min(1, returns / 0.2))  # Normalize

    # Run backtest
    bt = Backtest(prices)
    bt.add_signal("momentum", momentum_signal, weight=0.5)
    results = bt.run()
    print(results.summary())
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

import numpy as np

from .signals import HedgeEngine


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    initial_capital: float = 100_000  # Starting capital USD
    position_size_pct: float = 0.5  # Max position as % of capital
    transaction_cost_bps: float = 10  # 10 bps round trip
    slippage_bps: float = 5  # 5 bps slippage
    rebalance_threshold: float = 0.1  # Rebalance if signal changes by this much
    train_pct: float = 0.6  # 60% train, 40% test


@dataclass
class Trade:
    """A single trade record."""

    timestamp: datetime
    side: str  # "BUY" or "SELL"
    size: float
    price: float
    cost: float  # Transaction cost
    signal: float
    position_after: float


@dataclass
class BacktestResults:
    """Results from a backtest run."""

    # Core metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float

    # Trade stats
    num_trades: int
    avg_trade_return: float
    total_costs: float

    # Time series
    equity_curve: list[float]
    returns: list[float]
    positions: list[float]
    signals: list[float]
    timestamps: list[datetime]
    trades: list[Trade]

    # Train/test split
    train_sharpe: float
    test_sharpe: float

    def summary(self) -> str:
        """Human-readable summary."""
        return f"""
Backtest Results
================
Total Return:    {self.total_return:+.2%}
Sharpe Ratio:    {self.sharpe_ratio:.3f}
Max Drawdown:    {self.max_drawdown:.2%}
Win Rate:        {self.win_rate:.1%}

Trades:          {self.num_trades}
Avg Trade:       {self.avg_trade_return:+.2%}
Total Costs:     ${self.total_costs:,.2f}

Train Sharpe:    {self.train_sharpe:.3f}
Test Sharpe:     {self.test_sharpe:.3f}
"""


# Type alias for signal functions
SignalFunc = Callable[[list[float], int], float]


class Backtest:
    """
    Backtesting engine for signal-based hedging.

    Example:
        bt = Backtest(prices, timestamps)
        bt.add_signal("momentum", lambda p, i: compute_momentum(p, i), weight=0.5)
        bt.add_signal("volatility", lambda p, i: compute_vol(p, i), weight=0.3)
        results = bt.run()
    """

    def __init__(
        self,
        prices: list[float],
        timestamps: list[datetime] | None = None,
        config: BacktestConfig | None = None,
    ):
        """
        Initialize backtest.

        Args:
            prices: List of prices (daily close recommended)
            timestamps: Optional list of timestamps
            config: Backtest configuration
        """
        self.prices = prices
        self.timestamps = timestamps or [
            datetime.now() - timedelta(days=len(prices) - i)
            for i in range(len(prices))
        ]
        self.config = config or BacktestConfig()
        self.signals: dict[str, tuple[SignalFunc, float]] = {}

    def add_signal(
        self,
        name: str,
        func: SignalFunc,
        weight: float = 1.0,
    ) -> None:
        """
        Add a signal generator.

        Args:
            name: Signal name
            func: Function that takes (prices, index) and returns signal in [-1, 1]
            weight: Signal weight (auto-normalized)
        """
        self.signals[name] = (func, weight)

    def run(self) -> BacktestResults:
        """Run the backtest and return results."""
        if not self.signals:
            raise ValueError("No signals added. Use add_signal() first.")

        n = len(self.prices)
        cfg = self.config

        # Initialize tracking
        equity = [cfg.initial_capital]
        returns_list: list[float] = []
        positions: list[float] = []
        combined_signals: list[float] = []
        trades: list[Trade] = []

        current_position = 0.0  # In units of asset
        current_cash = cfg.initial_capital
        last_signal = 0.0

        # Main backtest loop
        for i in range(1, n):
            price = self.prices[i]

            # Compute all signals
            engine = HedgeEngine()
            for name, (func, weight) in self.signals.items():
                try:
                    signal_value = func(self.prices, i)
                    signal_value = max(-1.0, min(1.0, signal_value))
                except Exception:
                    signal_value = 0.0
                engine.add_signal(name, value=signal_value, weight=weight)

            # Get combined signal
            result = engine.compute(position_delta=current_position * price)
            combined_signal = result.combined_signal
            combined_signals.append(combined_signal)

            # Determine target position based on signal
            # Signal > 0 = bullish = want long exposure
            # Signal < 0 = bearish = want short/no exposure
            max_position_value = cfg.position_size_pct * (current_cash + current_position * price)
            target_position_value = combined_signal * max_position_value
            target_position = target_position_value / price if price > 0 else 0

            # Check if we should rebalance
            signal_change = abs(combined_signal - last_signal)
            should_rebalance = signal_change > cfg.rebalance_threshold

            if should_rebalance:
                # Execute trade
                trade_size = target_position - current_position
                trade_value = abs(trade_size * price)

                # Calculate costs
                cost_bps = cfg.transaction_cost_bps + cfg.slippage_bps
                cost = trade_value * cost_bps / 10000

                if trade_size > 0:
                    # Buying
                    current_cash -= trade_size * price + cost
                    current_position += trade_size
                    side = "BUY"
                else:
                    # Selling
                    current_cash -= trade_size * price - cost  # trade_size is negative
                    current_position += trade_size
                    side = "SELL"

                trades.append(Trade(
                    timestamp=self.timestamps[i],
                    side=side,
                    size=abs(trade_size),
                    price=price,
                    cost=cost,
                    signal=combined_signal,
                    position_after=current_position,
                ))

                last_signal = combined_signal

            # Mark to market
            portfolio_value = current_cash + current_position * price
            daily_return = (portfolio_value - equity[-1]) / equity[-1] if equity[-1] > 0 else 0

            equity.append(portfolio_value)
            returns_list.append(daily_return)
            positions.append(current_position)

        # Calculate metrics
        returns_arr = np.array(returns_list)
        equity_arr = np.array(equity)

        # Sharpe ratio (annualized, assuming daily data)
        if len(returns_arr) > 1 and np.std(returns_arr) > 0:
            sharpe = np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - peak) / peak
        max_dd = abs(np.min(drawdown))

        # Win rate
        if trades:
            # Calculate P&L per trade
            trade_pnls = []
            for j, trade in enumerate(trades):
                if j < len(trades) - 1:
                    exit_price = trades[j + 1].price
                else:
                    exit_price = self.prices[-1]

                if trade.side == "BUY":
                    pnl = (exit_price - trade.price) / trade.price
                else:
                    pnl = (trade.price - exit_price) / trade.price
                trade_pnls.append(pnl)

            win_rate = sum(1 for p in trade_pnls if p > 0) / len(trade_pnls) if trade_pnls else 0
            avg_trade = np.mean(trade_pnls) if trade_pnls else 0
        else:
            win_rate = 0
            avg_trade = 0

        # Train/test split metrics
        split_idx = int(len(returns_arr) * cfg.train_pct)
        train_returns = returns_arr[:split_idx]
        test_returns = returns_arr[split_idx:]

        if len(train_returns) > 1 and np.std(train_returns) > 0:
            train_sharpe = np.mean(train_returns) / np.std(train_returns) * np.sqrt(252)
        else:
            train_sharpe = 0.0

        if len(test_returns) > 1 and np.std(test_returns) > 0:
            test_sharpe = np.mean(test_returns) / np.std(test_returns) * np.sqrt(252)
        else:
            test_sharpe = 0.0

        total_costs = sum(t.cost for t in trades)
        total_return = (equity[-1] - cfg.initial_capital) / cfg.initial_capital

        return BacktestResults(
            total_return=total_return,
            sharpe_ratio=float(sharpe),
            max_drawdown=float(max_dd),
            win_rate=float(win_rate),
            num_trades=len(trades),
            avg_trade_return=float(avg_trade),
            total_costs=total_costs,
            equity_curve=equity,
            returns=returns_list,
            positions=positions,
            signals=combined_signals,
            timestamps=self.timestamps,
            trades=trades,
            train_sharpe=float(train_sharpe),
            test_sharpe=float(test_sharpe),
        )


def fetch_crypto_data(
    coin_id: str = "bitcoin",
    days: int = 365,
    vs_currency: str = "usd",
) -> tuple[list[float], list[datetime]]:
    """
    Fetch real crypto price data from CoinGecko (free, no API key).

    Args:
        coin_id: CoinGecko coin ID (e.g., "bitcoin", "ethereum")
        days: Number of days of history
        vs_currency: Quote currency

    Returns:
        Tuple of (prices, timestamps)
    """
    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        f"?vs_currency={vs_currency}&days={days}&interval=daily"
    )

    req = urllib.request.Request(url, headers={"Accept": "application/json"})

    with urllib.request.urlopen(req, timeout=30) as response:
        data = json.loads(response.read().decode())

    prices = [p[1] for p in data["prices"]]
    timestamps = [datetime.fromtimestamp(p[0] / 1000) for p in data["prices"]]

    return prices, timestamps


# =============================================================================
# Built-in signal generators
# =============================================================================


def momentum_signal(prices: list[float], i: int, lookback: int = 20) -> float:
    """
    Momentum signal: recent return normalized to [-1, 1].

    Positive momentum = bullish
    Negative momentum = bearish
    """
    if i < lookback:
        return 0.0

    ret = (prices[i] - prices[i - lookback]) / prices[i - lookback]
    # Normalize: assume ±20% is extreme
    return max(-1.0, min(1.0, ret / 0.20))


def volatility_regime_signal(
    prices: list[float],
    i: int,
    lookback: int = 20,
    baseline_vol: float = 0.02,
    high_vol: float = 0.05,
) -> float:
    """
    Volatility regime signal: high vol = bearish (risk-off).

    Returns signal in [-1, 1] where:
    - +1 = low volatility (risk-on, ok to hold)
    - -1 = high volatility (risk-off, reduce exposure)
    """
    if i < lookback:
        return 0.0

    # Calculate realized volatility
    returns = []
    for j in range(i - lookback + 1, i + 1):
        if j > 0:
            ret = (prices[j] - prices[j - 1]) / prices[j - 1]
            returns.append(ret)

    if not returns:
        return 0.0

    vol = float(np.std(returns))

    # Map to signal
    if vol <= baseline_vol:
        return 1.0  # Low vol = bullish
    elif vol >= high_vol:
        return -1.0  # High vol = bearish
    else:
        # Linear interpolation
        return 1.0 - 2.0 * (vol - baseline_vol) / (high_vol - baseline_vol)


def mean_reversion_signal(
    prices: list[float],
    i: int,
    lookback: int = 20,
    num_std: float = 2.0,
) -> float:
    """
    Mean reversion signal: price deviation from moving average.

    Returns signal in [-1, 1] where:
    - +1 = price far below MA (expect reversion up, bullish)
    - -1 = price far above MA (expect reversion down, bearish)
    """
    if i < lookback:
        return 0.0

    window = prices[i - lookback + 1 : i + 1]
    ma = np.mean(window)
    std = np.std(window)

    if std == 0:
        return 0.0

    z_score = (prices[i] - ma) / std
    # Invert: high price = bearish (expect reversion), low price = bullish
    signal = float(-z_score / num_std)
    return max(-1.0, min(1.0, signal))


def trend_following_signal(
    prices: list[float],
    i: int,
    fast_period: int = 10,
    slow_period: int = 30,
) -> float:
    """
    Trend following signal: fast MA vs slow MA.

    Returns signal in [-1, 1] where:
    - +1 = fast MA well above slow MA (uptrend)
    - -1 = fast MA well below slow MA (downtrend)
    """
    if i < slow_period:
        return 0.0

    fast_ma = np.mean(prices[i - fast_period + 1 : i + 1])
    slow_ma = np.mean(prices[i - slow_period + 1 : i + 1])

    if slow_ma == 0:
        return 0.0

    # Percentage difference
    diff_pct = float((fast_ma - slow_ma) / slow_ma)
    # Normalize: assume ±5% difference is extreme
    normalized = diff_pct / 0.05
    return max(-1.0, min(1.0, normalized))
