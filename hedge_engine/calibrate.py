"""
Signal weight calibration using Optuna.

Finds optimal signal weights by maximizing out-of-sample Sharpe ratio.

Example:
    from hedge_engine.calibrate import calibrate_weights, fetch_crypto_data

    prices, timestamps = fetch_crypto_data("bitcoin", days=365)

    best_weights, study = calibrate_weights(
        prices,
        timestamps,
        n_trials=100,
        metric="sharpe",
    )

    print(f"Best weights: {best_weights}")
    print(f"Best Sharpe: {study.best_value:.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Literal

import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from .backtest import (
    Backtest,
    BacktestConfig,
    BacktestResults,
    momentum_signal,
    volatility_regime_signal,
    mean_reversion_signal,
    trend_following_signal,
    fetch_crypto_data,
)


@dataclass
class CalibrationResult:
    """Results from weight calibration."""

    best_weights: dict[str, float]
    best_metric: float
    metric_name: str
    n_trials: int
    train_results: BacktestResults
    test_results: BacktestResults

    def summary(self) -> str:
        """Human-readable summary."""
        weights_str = ", ".join(f"{k}={v:.2f}" for k, v in self.best_weights.items())
        return f"""
Calibration Results
===================
Best {self.metric_name}: {self.best_metric:.4f}
Trials: {self.n_trials}

Optimal Weights:
  {weights_str}

Train Performance:
  Return: {self.train_results.total_return:+.2%}
  Sharpe: {self.train_results.sharpe_ratio:.3f}
  Max DD: {self.train_results.max_drawdown:.2%}

Test Performance (Out-of-Sample):
  Return: {self.test_results.total_return:+.2%}
  Sharpe: {self.test_results.sharpe_ratio:.3f}
  Max DD: {self.test_results.max_drawdown:.2%}
"""


# Default signal generators for calibration
DEFAULT_SIGNALS: dict[str, Callable[[list[float], int], float]] = {
    "momentum": lambda p, i: momentum_signal(p, i, lookback=20),
    "volatility": lambda p, i: volatility_regime_signal(p, i),
    "mean_reversion": lambda p, i: mean_reversion_signal(p, i),
    "trend": lambda p, i: trend_following_signal(p, i),
}


def calibrate_weights(
    prices: list[float],
    timestamps: list[datetime] | None = None,
    signals: dict[str, Callable[[list[float], int], float]] | None = None,
    n_trials: int = 100,
    train_pct: float = 0.6,
    metric: Literal["sharpe", "sortino", "calmar", "return"] = "sharpe",
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> CalibrationResult:
    """
    Calibrate signal weights using Optuna.

    Uses walk-forward optimization:
    1. Train on first train_pct of data
    2. Validate on remaining data
    3. Optimize for out-of-sample performance

    Args:
        prices: Price series
        timestamps: Optional timestamps
        signals: Dict of signal name -> signal function
        n_trials: Number of Optuna trials
        train_pct: Fraction of data for training
        metric: Optimization metric
        min_weight: Minimum weight for each signal
        max_weight: Maximum weight for each signal
        seed: Random seed
        verbose: Print progress

    Returns:
        CalibrationResult with optimal weights and performance
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna not installed. Run: pip install optuna"
        )

    if signals is None:
        signals = DEFAULT_SIGNALS

    if not signals:
        raise ValueError("No signals provided")

    # Split data
    split_idx = int(len(prices) * train_pct)
    train_prices = prices[:split_idx]
    test_prices = prices[split_idx:]

    if timestamps:
        train_timestamps = timestamps[:split_idx]
        test_timestamps = timestamps[split_idx:]
    else:
        train_timestamps = None
        test_timestamps = None

    signal_names = list(signals.keys())

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # Sample weights for each signal
        weights = {}
        for name in signal_names:
            weights[name] = trial.suggest_float(
                f"w_{name}", min_weight, max_weight
            )

        # Skip if all weights are zero
        if sum(weights.values()) < 0.01:
            return float("-inf")

        # Run backtest on training data
        bt = Backtest(
            train_prices,
            train_timestamps,
            config=BacktestConfig(train_pct=1.0),  # Use all train data
        )

        for name, func in signals.items():
            bt.add_signal(name, func, weight=weights[name])

        try:
            results = bt.run()
        except Exception:
            return float("-inf")

        # Return the optimization metric
        if metric == "sharpe":
            return results.sharpe_ratio
        elif metric == "sortino":
            # Sortino: only penalize downside volatility
            returns = np.array(results.returns)
            downside = returns[returns < 0]
            if len(downside) == 0 or np.std(downside) == 0:
                return results.sharpe_ratio
            mean_ret = float(np.mean(returns))
            down_std = float(np.std(downside))
            sortino = mean_ret / down_std * (252 ** 0.5)
            return float(sortino)
        elif metric == "calmar":
            # Calmar: return / max drawdown
            if results.max_drawdown == 0:
                return results.total_return * 100
            return results.total_return / results.max_drawdown
        elif metric == "return":
            return results.total_return
        else:
            return results.sharpe_ratio

    # Run optimization
    optuna.logging.set_verbosity(
        optuna.logging.INFO if verbose else optuna.logging.WARNING
    )

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    # Extract best weights
    best_weights = {
        name: study.best_params[f"w_{name}"]
        for name in signal_names
    }

    # Normalize weights to sum to 1
    total = sum(best_weights.values())
    if total > 0:
        best_weights = {k: v / total for k, v in best_weights.items()}

    # Run final backtest on train and test separately
    train_bt = Backtest(
        train_prices,
        train_timestamps,
        config=BacktestConfig(train_pct=1.0),
    )
    for name, func in signals.items():
        train_bt.add_signal(name, func, weight=best_weights[name])
    train_results = train_bt.run()

    test_bt = Backtest(
        test_prices,
        test_timestamps,
        config=BacktestConfig(train_pct=1.0),
    )
    for name, func in signals.items():
        test_bt.add_signal(name, func, weight=best_weights[name])
    test_results = test_bt.run()

    return CalibrationResult(
        best_weights=best_weights,
        best_metric=study.best_value,
        metric_name=metric,
        n_trials=n_trials,
        train_results=train_results,
        test_results=test_results,
    )


def quick_calibrate(
    coin: str = "bitcoin",
    days: int = 365,
    n_trials: int = 50,
    verbose: bool = True,
) -> CalibrationResult:
    """
    Quick calibration with default signals on crypto data.

    Args:
        coin: CoinGecko coin ID
        days: Days of history
        n_trials: Optuna trials
        verbose: Print progress

    Returns:
        CalibrationResult
    """
    if verbose:
        print(f"Fetching {days} days of {coin} data...")

    prices, timestamps = fetch_crypto_data(coin, days=days)

    if verbose:
        print(f"Got {len(prices)} data points")
        print(f"Running {n_trials} optimization trials...")

    result = calibrate_weights(
        prices,
        timestamps,
        signals=DEFAULT_SIGNALS,
        n_trials=n_trials,
        verbose=verbose,
    )

    return result


def compare_weights(
    prices: list[float],
    timestamps: list[datetime] | None = None,
    weight_sets: dict[str, dict[str, float]] | None = None,
    signals: dict[str, Callable[[list[float], int], float]] | None = None,
) -> dict[str, BacktestResults]:
    """
    Compare different weight configurations.

    Args:
        prices: Price series
        timestamps: Optional timestamps
        weight_sets: Dict of config_name -> {signal_name: weight}
        signals: Signal functions

    Returns:
        Dict of config_name -> BacktestResults
    """
    if signals is None:
        signals = DEFAULT_SIGNALS

    if weight_sets is None:
        # Default comparisons
        n = len(signals)
        equal_weights = {name: 1.0 / n for name in signals}
        weight_sets = {
            "equal": equal_weights,
            "momentum_heavy": {**{n: 0.1 for n in signals}, "momentum": 0.7},
            "volatility_heavy": {**{n: 0.1 for n in signals}, "volatility": 0.7},
        }

    results = {}
    for config_name, weights in weight_sets.items():
        bt = Backtest(prices, timestamps)
        for name, func in signals.items():
            bt.add_signal(name, func, weight=weights.get(name, 0.0))

        results[config_name] = bt.run()

    return results
