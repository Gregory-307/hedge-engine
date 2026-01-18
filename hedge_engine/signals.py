"""
Signal-based hedge engine.

This is the ACTUAL hedge engine - a modular signal aggregator.

How it works:
1. You provide signals (any market data you care about)
2. Each signal is normalized to [-1, +1] where:
   - +1 = strongly bullish (price will go UP)
   - -1 = strongly bearish (price will go DOWN)
   - 0 = neutral
3. You assign weights to each signal (how much you trust it)
4. Engine combines signals → direction + magnitude recommendation

Example:
    engine = HedgeEngine()
    engine.add_signal("funding_rate", value=0.7, weight=0.4)  # Bullish signal
    engine.add_signal("orderbook_imbalance", value=-0.3, weight=0.3)  # Slightly bearish
    engine.add_signal("vol_regime", value=0.5, weight=0.3)  # Bullish

    result = engine.compute(position_delta=100_000)  # You're $100k long
    # → "Reduce long by 15%" because mixed signals with slight bearish tilt
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Direction(str, Enum):
    """Recommended direction to adjust position."""
    REDUCE_LONG = "REDUCE_LONG"    # You're long, go shorter
    REDUCE_SHORT = "REDUCE_SHORT"  # You're short, go longer
    HOLD = "HOLD"                  # No action needed
    INCREASE_LONG = "INCREASE_LONG"   # Add to long
    INCREASE_SHORT = "INCREASE_SHORT"  # Add to short


@dataclass
class Signal:
    """
    A single market signal.

    Attributes:
        name: Identifier for this signal (e.g., "funding_rate", "orderbook_imbalance")
        value: Normalized signal value in [-1, +1]
                +1 = strongly bullish (expect price UP)
                -1 = strongly bearish (expect price DOWN)
                 0 = neutral
        weight: How much this signal matters (any positive number, auto-normalized)
        raw_value: Optional - the original unnormalized value (for debugging)
        description: Optional - what this signal represents
    """
    name: str
    value: float  # -1 to +1
    weight: float  # 0 to 1
    raw_value: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        # Clamp value to [-1, 1]
        self.value = max(-1.0, min(1.0, self.value))
        # Weight must be non-negative (no upper bound - auto-normalized by engine)
        self.weight = max(0.0, self.weight)

    @property
    def weighted_value(self) -> float:
        """Signal value × weight."""
        return self.value * self.weight


@dataclass
class HedgeRecommendation:
    """
    Output from the hedge engine.

    Attributes:
        direction: What to do (REDUCE_LONG, REDUCE_SHORT, HOLD, etc.)
        magnitude: How much to adjust (0.0 to 1.0, as fraction of position)
        confidence: How confident the engine is (based on signal agreement)
        combined_signal: The weighted average signal (-1 to +1)
        signal_breakdown: Individual signal contributions
        reasoning: Human-readable explanation
    """
    direction: Direction
    magnitude: float  # 0 to 1 (fraction of position to adjust)
    confidence: float  # 0 to 1 (how much signals agree)
    combined_signal: float  # -1 to +1
    signal_breakdown: dict[str, float]  # name → weighted contribution
    reasoning: str

    @property
    def action_str(self) -> str:
        """Human-readable action string."""
        if self.direction == Direction.HOLD:
            return "HOLD - no action"
        pct = self.magnitude * 100
        return f"{self.direction.value} by {pct:.0f}%"


@dataclass
class HedgeEngine:
    """
    Modular signal aggregator for hedge decisions.

    Usage:
        engine = HedgeEngine()
        engine.add_signal("my_signal", value=0.5, weight=0.3)
        result = engine.compute(position_delta=100_000)

    Position size scaling:
        Larger positions get more aggressive hedge recommendations.
        Formula: magnitude *= (1 + k * sqrt(position_size / baseline_size))

        This means:
        - $10k position with signal 0.5 → modest hedge
        - $1M position with signal 0.5 → aggressive hedge
        - Sqrt scaling gives diminishing returns for huge positions
    """
    signals: dict[str, Signal] = field(default_factory=dict)

    # Thresholds for action (configurable)
    action_threshold: float = 0.1  # Below this combined signal, HOLD
    high_conviction_threshold: float = 0.5  # Above this, recommend larger adjustment

    # Position size scaling parameters
    baseline_position_usd: float = 100_000  # $100k as baseline
    size_scaling_k: float = 0.5  # Scaling constant for sqrt(size) adjustment

    def add_signal(
        self,
        name: str,
        value: float,
        weight: float,
        raw_value: float | None = None,
        description: str = "",
    ) -> None:
        """
        Add or update a signal.

        Args:
            name: Signal identifier
            value: Normalized value in [-1, +1] (+1=bullish, -1=bearish)
            weight: Relative importance (any positive number, auto-normalized)
                    e.g., weights 1,2,1 and 0.25,0.5,0.25 produce same result
            raw_value: Optional original unnormalized value
            description: What this signal represents
        """
        self.signals[name] = Signal(
            name=name,
            value=value,
            weight=weight,
            raw_value=raw_value,
            description=description,
        )

    def remove_signal(self, name: str) -> None:
        """Remove a signal."""
        self.signals.pop(name, None)

    def clear_signals(self) -> None:
        """Remove all signals."""
        self.signals.clear()

    def compute(self, position_delta: float) -> HedgeRecommendation:
        """
        Compute hedge recommendation given current position.

        Args:
            position_delta: Your current position in USD notional
                           Positive = long exposure (you profit if price goes up)
                           Negative = short exposure (you profit if price goes down)

        Returns:
            HedgeRecommendation with direction, magnitude, and reasoning
        """
        if not self.signals:
            return HedgeRecommendation(
                direction=Direction.HOLD,
                magnitude=0.0,
                confidence=0.0,
                combined_signal=0.0,
                signal_breakdown={},
                reasoning="No signals provided. Add signals with add_signal().",
            )

        # Calculate weighted average signal
        total_weight = sum(s.weight for s in self.signals.values())
        if total_weight == 0:
            return HedgeRecommendation(
                direction=Direction.HOLD,
                magnitude=0.0,
                confidence=0.0,
                combined_signal=0.0,
                signal_breakdown={s.name: 0.0 for s in self.signals.values()},
                reasoning="All signal weights are zero.",
            )

        # Weighted sum
        combined = sum(s.weighted_value for s in self.signals.values()) / total_weight

        # Signal breakdown (contribution of each)
        breakdown = {
            s.name: (s.weighted_value / total_weight)
            for s in self.signals.values()
        }

        # Confidence: how much do signals agree?
        # High confidence if all signals point same direction
        # Low confidence if signals conflict
        if len(self.signals) > 1:
            values = [s.value for s in self.signals.values()]
            # Variance-based confidence: low variance = high agreement
            mean_val = sum(values) / len(values)
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)
            # Max variance is 1 (if signals are -1 and +1)
            confidence = 1.0 - min(1.0, variance)
        else:
            confidence = abs(list(self.signals.values())[0].value)

        # Determine direction and magnitude based on position + signal
        direction, magnitude, reasoning = self._decide_action(
            combined_signal=combined,
            position_delta=position_delta,
            confidence=confidence,
        )

        return HedgeRecommendation(
            direction=direction,
            magnitude=magnitude,
            confidence=confidence,
            combined_signal=combined,
            signal_breakdown=breakdown,
            reasoning=reasoning,
        )

    def _decide_action(
        self,
        combined_signal: float,
        position_delta: float,
        confidence: float,
    ) -> tuple[Direction, float, str]:
        """
        Decide action based on signal and position.

        Logic:
        - If signal is bullish (+) and you're short (-) → reduce short
        - If signal is bearish (-) and you're long (+) → reduce long
        - If signal agrees with position → could increase or hold
        - Magnitude scales with |combined_signal|, confidence, AND position size
        """
        signal_str = "bullish" if combined_signal > 0 else "bearish" if combined_signal < 0 else "neutral"
        position_str = "long" if position_delta > 0 else "short" if position_delta < 0 else "flat"
        pos_size = abs(position_delta)
        size_str = f"${pos_size/1000:.0f}k" if pos_size < 1_000_000 else f"${pos_size/1_000_000:.1f}M"

        # Flat position
        if position_delta == 0:
            if abs(combined_signal) < self.action_threshold:
                return Direction.HOLD, 0.0, f"Flat position, neutral signal ({combined_signal:.2f}). No action."
            elif combined_signal > 0:
                mag = self._scale_magnitude(combined_signal, confidence, pos_size)
                return Direction.INCREASE_LONG, mag, f"Flat position, {signal_str} signal ({combined_signal:.2f}). Consider going long."
            else:
                mag = self._scale_magnitude(abs(combined_signal), confidence, pos_size)
                return Direction.INCREASE_SHORT, mag, f"Flat position, {signal_str} signal ({combined_signal:.2f}). Consider going short."

        # Signal too weak to act
        if abs(combined_signal) < self.action_threshold:
            return Direction.HOLD, 0.0, f"Signal too weak ({combined_signal:.2f}). Hold {position_str} position."

        # Long position
        if position_delta > 0:
            if combined_signal < 0:
                # Bearish signal + long position → reduce long
                mag = self._scale_magnitude(abs(combined_signal), confidence, pos_size)
                return Direction.REDUCE_LONG, mag, (
                    f"You're {position_str} {size_str}, signal is {signal_str} ({combined_signal:.2f}). "
                    f"Reduce long by {mag*100:.0f}% (confidence: {confidence:.0%})."
                )
            else:
                # Bullish signal + long position → could add or hold
                if combined_signal > self.high_conviction_threshold:
                    mag = self._scale_magnitude(combined_signal, confidence, pos_size) * 0.5  # More conservative for adds
                    return Direction.INCREASE_LONG, mag, (
                        f"You're {position_str} {size_str}, signal is strongly {signal_str} ({combined_signal:.2f}). "
                        f"Could increase long by {mag*100:.0f}%."
                    )
                return Direction.HOLD, 0.0, f"You're {position_str} {size_str}, signal agrees ({combined_signal:.2f}). Hold."

        # Short position
        else:
            if combined_signal > 0:
                # Bullish signal + short position → reduce short
                mag = self._scale_magnitude(combined_signal, confidence, pos_size)
                return Direction.REDUCE_SHORT, mag, (
                    f"You're {position_str} {size_str}, signal is {signal_str} ({combined_signal:.2f}). "
                    f"Reduce short by {mag*100:.0f}% (confidence: {confidence:.0%})."
                )
            else:
                # Bearish signal + short position → could add or hold
                if abs(combined_signal) > self.high_conviction_threshold:
                    mag = self._scale_magnitude(abs(combined_signal), confidence, pos_size) * 0.5
                    return Direction.INCREASE_SHORT, mag, (
                        f"You're {position_str} {size_str}, signal is strongly {signal_str} ({combined_signal:.2f}). "
                        f"Could increase short by {mag*100:.0f}%."
                    )
                return Direction.HOLD, 0.0, f"You're {position_str} {size_str}, signal agrees ({combined_signal:.2f}). Hold."

    def _scale_magnitude(
        self,
        signal_strength: float,
        confidence: float,
        position_size: float,
    ) -> float:
        """
        Scale recommendation magnitude based on signal, confidence, AND position size.

        Formula: magnitude = base_signal * confidence_adj * size_adj

        Where:
        - base_signal = |combined_signal|
        - confidence_adj = 0.2 + 0.8 * confidence (floor at 20%)
        - size_adj = 1 + k * sqrt(position_size / baseline_size)

        This means larger positions get more aggressive recommendations.
        Derived from: loss_threshold = k * sqrt(position_size)
        """
        # Base magnitude from signal strength (0 to 1)
        base = min(1.0, signal_strength)

        # Scale by confidence (but never below 20% of base)
        confidence_adj = 0.2 + 0.8 * confidence

        # Scale by position size: larger positions → more aggressive
        # sqrt scaling: $1M position (10x baseline) → 1 + 0.5*sqrt(10) ≈ 2.58x
        size_ratio = abs(position_size) / self.baseline_position_usd
        size_adj = 1.0 + self.size_scaling_k * (size_ratio ** 0.5)

        # Combined scaling
        scaled = base * confidence_adj * size_adj

        # Cap at 100%
        return float(min(1.0, scaled))


# =============================================================================
# Signal Normalizers - helper functions to convert raw data to [-1, +1]
# =============================================================================

def normalize_funding_rate(funding_rate: float, neutral: float = 0.0, extreme: float = 0.1) -> float:
    """
    Normalize funding rate to [-1, +1].

    Funding rate > 0: longs pay shorts → bearish signal (crowded long)
    Funding rate < 0: shorts pay longs → bullish signal (crowded short)

    Args:
        funding_rate: Annual funding rate (e.g., 0.05 = 5%)
        neutral: Rate considered neutral (default 0)
        extreme: Rate considered extreme (default 10%)

    Returns:
        Normalized signal: -1 (very bearish) to +1 (very bullish)
    """
    # Invert because high funding = crowded longs = bearish
    deviation = neutral - funding_rate
    normalized = deviation / extreme
    return max(-1.0, min(1.0, normalized))


def normalize_orderbook_imbalance(bid_depth: float, ask_depth: float) -> float:
    """
    Normalize orderbook imbalance to [-1, +1].

    More bids than asks → bullish (buying pressure)
    More asks than bids → bearish (selling pressure)

    Args:
        bid_depth: USD depth on bid side
        ask_depth: USD depth on ask side

    Returns:
        Normalized signal: -1 (bearish) to +1 (bullish)
    """
    total = bid_depth + ask_depth
    if total == 0:
        return 0.0

    imbalance = (bid_depth - ask_depth) / total
    return max(-1.0, min(1.0, imbalance))


def normalize_price_momentum(returns: list[float], lookback: int = 20) -> float:
    """
    Normalize price momentum to [-1, +1].

    Positive recent returns → bullish
    Negative recent returns → bearish

    Args:
        returns: List of period returns (most recent last)
        lookback: How many periods to consider

    Returns:
        Normalized signal: -1 (bearish) to +1 (bullish)
    """
    if not returns:
        return 0.0

    recent = returns[-lookback:] if len(returns) > lookback else returns
    avg_return = sum(recent) / len(recent)

    # Scale: assume ±5% average return is extreme
    normalized = avg_return / 0.05
    return max(-1.0, min(1.0, normalized))


def normalize_volatility_regime(
    current_vol: float,
    baseline_vol: float = 0.05,
    low_vol: float = 0.02,
    high_vol: float = 0.10,
) -> float:
    """
    Normalize volatility regime to [-1, +1].

    This is RISK signal, not directional:
    - High vol → negative (risk-off, reduce exposure)
    - Low vol → positive (risk-on, can hold larger positions)

    Args:
        current_vol: Current realized volatility
        baseline_vol: Normal volatility level
        low_vol: Low volatility threshold
        high_vol: High volatility threshold

    Returns:
        Normalized signal: -1 (high vol, reduce) to +1 (low vol, comfortable)
    """
    if current_vol <= low_vol:
        return 1.0
    elif current_vol >= high_vol:
        return -1.0
    else:
        # Linear interpolation
        if current_vol < baseline_vol:
            # Low to baseline: 1.0 to 0.0
            return 1.0 - (current_vol - low_vol) / (baseline_vol - low_vol)
        else:
            # Baseline to high: 0.0 to -1.0
            return -(current_vol - baseline_vol) / (high_vol - baseline_vol)


def normalize_zscore(value: float, mean: float, std: float, cap: float = 3.0) -> float:
    """
    Generic z-score normalization.

    Converts any value to [-1, +1] based on its z-score.

    Args:
        value: The value to normalize
        mean: Historical mean
        std: Historical standard deviation
        cap: Z-score cap (default ±3)

    Returns:
        Normalized signal clamped to [-1, +1]
    """
    if std == 0:
        return 0.0

    zscore = (value - mean) / std
    capped = max(-cap, min(cap, zscore))
    return capped / cap
