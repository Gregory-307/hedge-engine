"""
Distributional signals for the hedge engine.

Instead of point estimates, use full probability distributions to compute:
- P(loss): Probability of negative return
- CVaR: Expected loss in the worst α%
- Kelly: Optimal position sizing

Requires: temporalpdf library (pip install temporalpdf)

Example:
    from hedge_engine.distributional import DistributionalSignal, compute_downside_metrics
    import temporalpdf as tpdf

    # Create a return distribution (e.g., from your model)
    dist = tpdf.NIG()
    params = tpdf.NIGParameters(mu=0.001, delta=0.02, alpha=15.0, beta=-2.0)

    # Compute downside risk metrics
    metrics = compute_downside_metrics(dist, params)
    print(f"P(loss): {metrics.p_loss:.1%}")
    print(f"CVaR 5%: {metrics.cvar_5:.2%}")
    print(f"Kelly: {metrics.kelly:.1%}")

    # Use in hedge engine
    engine = HedgeEngine()
    signal = DistributionalSignal.from_distribution(dist, params, weight=0.5)
    engine.add_signal("my_forecast", value=signal.value, weight=signal.weight)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DistributionProtocol(Protocol):
    """Protocol for distributions compatible with temporalpdf."""

    def cdf(
        self, x: NDArray[np.float64], t: float, params: Any
    ) -> NDArray[np.float64]: ...

    def ppf(
        self, q: NDArray[np.float64], t: float, params: Any
    ) -> NDArray[np.float64]: ...

    def mean(self, t: float, params: Any) -> float: ...

    def variance(self, t: float, params: Any) -> float: ...

    def sample(
        self, n: int, t: float, params: Any, rng: np.random.Generator | None
    ) -> NDArray[np.float64]: ...


@dataclass
class DownsideMetrics:
    """
    Downside risk metrics computed from a distribution.

    Attributes:
        p_loss: Probability of negative return P(X < 0)
        expected_return: Mean of the distribution
        volatility: Standard deviation of the distribution
        var_5: Value at Risk at 5% (95% confidence)
        cvar_5: Conditional VaR at 5% (expected loss in worst 5%)
        kelly: Full Kelly fraction (optimal bet size)
        half_kelly: Half Kelly (more conservative)
    """

    p_loss: float  # P(return < 0)
    expected_return: float  # E[return]
    volatility: float  # sqrt(Var[return])
    var_5: float  # 5% VaR (loss at 95% confidence)
    cvar_5: float  # 5% CVaR (expected shortfall)
    kelly: float  # Full Kelly fraction
    half_kelly: float  # Half Kelly (conservative)

    @property
    def sharpe_proxy(self) -> float:
        """Sharpe-like ratio: expected_return / volatility."""
        if self.volatility <= 0:
            return 0.0
        return self.expected_return / self.volatility

    @property
    def risk_reward_ratio(self) -> float:
        """Risk/reward: CVaR / expected_return (lower is better)."""
        if self.expected_return <= 0:
            return float("inf")
        return self.cvar_5 / self.expected_return


def compute_downside_metrics(
    dist: DistributionProtocol,
    params: Any,
    t: float = 0.0,
    alpha: float = 0.05,
    n_samples: int = 100_000,
    rng: np.random.Generator | None = None,
) -> DownsideMetrics:
    """
    Compute downside risk metrics from a distribution.

    Args:
        dist: Distribution object (temporalpdf distribution)
        params: Distribution parameters
        t: Time point (default 0)
        alpha: Tail probability for VaR/CVaR (default 0.05 = 5%)
        n_samples: Monte Carlo samples for CVaR
        rng: Random number generator

    Returns:
        DownsideMetrics with p_loss, CVaR, Kelly, etc.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # P(loss) = P(X < 0)
    cdf_at_zero = dist.cdf(np.array([0.0]), t, params)
    p_loss = float(cdf_at_zero[0])

    # Expected return and volatility
    expected_return = dist.mean(t, params)
    variance = dist.variance(t, params)
    volatility = float(np.sqrt(variance)) if variance > 0 else 0.0

    # VaR: quantile at alpha
    var_quantile = dist.ppf(np.array([alpha]), t, params)
    var_5 = -float(var_quantile[0])  # Negate because VaR is positive for losses

    # CVaR: expected value in the tail (Monte Carlo)
    samples = dist.sample(n_samples, t, params, rng)
    var_threshold = float(var_quantile[0])
    tail_samples = samples[samples <= var_threshold]

    if len(tail_samples) > 0:
        cvar_5 = -float(np.mean(tail_samples))
    else:
        # Fallback: use VaR
        cvar_5 = var_5

    # Kelly criterion: f* = mu / sigma^2
    if variance > 0:
        kelly = expected_return / variance
    else:
        kelly = 0.0

    half_kelly = kelly * 0.5

    return DownsideMetrics(
        p_loss=p_loss,
        expected_return=expected_return,
        volatility=volatility,
        var_5=var_5,
        cvar_5=cvar_5,
        kelly=kelly,
        half_kelly=half_kelly,
    )


@dataclass
class DistributionalSignal:
    """
    A signal derived from a probability distribution.

    Converts distributional metrics into a normalized signal [-1, +1]
    that can be used in the HedgeEngine.

    The signal value is computed from the distribution's risk/reward profile:
    - High P(loss) + high CVaR = bearish signal (negative)
    - Low P(loss) + positive expected return = bullish signal (positive)
    """

    name: str
    value: float  # Normalized signal [-1, +1]
    weight: float  # Importance weight
    metrics: DownsideMetrics  # Underlying metrics

    @classmethod
    def from_distribution(
        cls,
        dist: DistributionProtocol,
        params: Any,
        name: str = "distributional",
        weight: float = 1.0,
        t: float = 0.0,
        method: str = "risk_adjusted",
    ) -> DistributionalSignal:
        """
        Create a signal from a probability distribution.

        Args:
            dist: Distribution object
            params: Distribution parameters
            name: Signal name
            weight: Signal weight for aggregation
            t: Time point
            method: How to convert distribution to signal:
                - "risk_adjusted": (E[r] - CVaR) / scale
                - "probability": 1 - 2*P(loss)  (maps [0,1] to [1,-1])
                - "sharpe": E[r] / sigma (capped)
                - "kelly": Kelly fraction (capped)

        Returns:
            DistributionalSignal with normalized value
        """
        metrics = compute_downside_metrics(dist, params, t=t)

        if method == "probability":
            # P(loss) = 0 -> signal = +1 (bullish)
            # P(loss) = 0.5 -> signal = 0 (neutral)
            # P(loss) = 1 -> signal = -1 (bearish)
            value = 1.0 - 2.0 * metrics.p_loss

        elif method == "risk_adjusted":
            # Combine expected return and downside risk
            # Positive when expected return exceeds downside risk
            if metrics.cvar_5 > 0:
                # Scale by CVaR to normalize
                risk_adjusted = (metrics.expected_return + metrics.cvar_5) / (
                    2 * metrics.cvar_5
                )
                # Center around 0 and cap
                value = max(-1.0, min(1.0, risk_adjusted - 0.5))
            else:
                value = 1.0 if metrics.expected_return > 0 else 0.0

        elif method == "sharpe":
            # Sharpe ratio, capped at [-1, +1]
            # Assume Sharpe > 2 is extremely bullish, < -2 is extremely bearish
            sharpe = metrics.sharpe_proxy
            value = max(-1.0, min(1.0, sharpe / 2.0))

        elif method == "kelly":
            # Kelly fraction, capped
            # Kelly > 1 means leverage (very bullish)
            # Kelly < 0 means short (bearish)
            kelly_capped = max(-1.0, min(1.0, metrics.kelly))
            value = kelly_capped

        else:
            raise ValueError(f"Unknown method: {method}")

        return cls(name=name, value=value, weight=weight, metrics=metrics)


def normalize_from_distribution(
    dist: DistributionProtocol,
    params: Any,
    method: str = "probability",
    t: float = 0.0,
) -> float:
    """
    Convenience function to get a normalized signal from a distribution.

    Args:
        dist: Distribution object
        params: Distribution parameters
        method: Conversion method (see DistributionalSignal.from_distribution)
        t: Time point

    Returns:
        Normalized signal value in [-1, +1]
    """
    signal = DistributionalSignal.from_distribution(
        dist, params, method=method, t=t
    )
    return signal.value


# =============================================================================
# Simple Distribution Wrappers (no temporalpdf dependency)
# =============================================================================


@dataclass
class SimpleNormal:
    """
    Simple Normal distribution for use without temporalpdf.

    This is a lightweight wrapper that provides the same interface
    as temporalpdf distributions but uses only numpy/scipy.
    """

    def cdf(
        self, x: NDArray[np.float64], t: float, params: SimpleNormalParams
    ) -> NDArray[np.float64]:
        """Cumulative distribution function."""
        from scipy import stats

        return stats.norm.cdf(x, loc=params.mu, scale=params.sigma)  # type: ignore[no-any-return]

    def ppf(
        self, q: NDArray[np.float64], t: float, params: SimpleNormalParams
    ) -> NDArray[np.float64]:
        """Percent point function (inverse CDF)."""
        from scipy import stats

        return stats.norm.ppf(q, loc=params.mu, scale=params.sigma)  # type: ignore[no-any-return]

    def mean(self, t: float, params: SimpleNormalParams) -> float:
        """Mean of the distribution."""
        return params.mu

    def variance(self, t: float, params: SimpleNormalParams) -> float:
        """Variance of the distribution."""
        return params.sigma**2

    def sample(
        self,
        n: int,
        t: float,
        params: SimpleNormalParams,
        rng: np.random.Generator | None,
    ) -> NDArray[np.float64]:
        """Sample from the distribution."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(loc=params.mu, scale=params.sigma, size=n)


@dataclass
class SimpleNormalParams:
    """Parameters for SimpleNormal distribution."""

    mu: float  # Mean (expected return)
    sigma: float  # Standard deviation (volatility)


def quick_downside_metrics(
    expected_return: float,
    volatility: float,
) -> DownsideMetrics:
    """
    Quick downside metrics assuming Normal distribution.

    This is a convenience function when you only have mean and volatility
    and want quick risk metrics without fitting a full distribution.

    Args:
        expected_return: Expected return (e.g., 0.02 for 2%)
        volatility: Standard deviation (e.g., 0.05 for 5%)

    Returns:
        DownsideMetrics computed assuming Normal distribution
    """
    dist = SimpleNormal()
    params = SimpleNormalParams(mu=expected_return, sigma=volatility)
    return compute_downside_metrics(dist, params)


def quick_signal(
    expected_return: float,
    volatility: float,
    method: str = "probability",
) -> float:
    """
    Quick signal from expected return and volatility.

    Assumes Normal distribution. For more accurate signals with
    fat tails, use temporalpdf with NIG distribution.

    Args:
        expected_return: Expected return
        volatility: Standard deviation
        method: Conversion method

    Returns:
        Normalized signal in [-1, +1]
    """
    dist = SimpleNormal()
    params = SimpleNormalParams(mu=expected_return, sigma=volatility)
    signal = DistributionalSignal.from_distribution(
        dist, params, method=method
    )
    return signal.value
