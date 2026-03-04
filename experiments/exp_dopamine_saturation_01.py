"""
EXP-DOP-01 — Dopamine Saturation Threshold Detection
=====================================================
Models "saturation" as the point where marginal structural expansion
per unit of time approaches zero — analogous to receptor saturation.

In Phase 2, expansion rate decelerates. Saturation threshold is the
inflection point where second derivative of structural expansion changes sign.

Beyond this threshold, continued holding yields diminishing returns.
Collapse onset follows within a predictable window.

Output: saturation threshold estimates + post-saturation window.
"""

import math
import statistics
from dataclasses import dataclass


@dataclass
class SaturationState:
    bar: int
    expansion_rate: float      # first derivative of structural expansion
    acceleration: float        # second derivative
    saturated: bool            # True if d²E/dt² < 0 sustained
    bars_past_saturation: int
    collapse_window_open: bool # True if in predicted collapse window


SATURATION_CONFIRMATION = 3   # bars of negative acceleration to confirm saturation
COLLAPSE_WINDOW_BARS    = 5   # bars after saturation where collapse is expected


def compute_expansion_rate(price_series: list[float], window: int = 4) -> list[float]:
    """Rate of structural expansion: smoothed first difference."""
    rates = [0.0]
    for i in range(1, len(price_series)):
        start = max(0, i - window)
        hist = price_series[start:i+1]
        rate = (hist[-1] - hist[0]) / len(hist)
        rates.append(rate)
    return rates


def detect_saturation(price_series: list[float]) -> list[SaturationState]:
    rates = compute_expansion_rate(price_series)
    n = len(rates)
    states = []
    sat_confirmed = False
    sat_bar = None
    neg_accel_count = 0

    for i in range(1, n):
        accel = rates[i] - rates[i-1]
        if accel < 0:
            neg_accel_count += 1
        else:
            neg_accel_count = 0

        if neg_accel_count >= SATURATION_CONFIRMATION and not sat_confirmed:
            sat_confirmed = True
            sat_bar = i

        bars_past = (i - sat_bar) if sat_confirmed else 0
        collapse_open = sat_confirmed and 0 <= bars_past <= COLLAPSE_WINDOW_BARS

        states.append(SaturationState(
            bar=i,
            expansion_rate=round(rates[i], 4),
            acceleration=round(accel, 4),
            saturated=sat_confirmed,
            bars_past_saturation=bars_past,
            collapse_window_open=collapse_open,
        ))

    return states


if __name__ == "__main__":
    import random
    random.seed(31)

    price = [100.0]
    for i in range(24):
        if i < 12:
            rate = 0.5 * math.exp(-0.05 * i) + random.gauss(0, 0.1)
        else:
            rate = -0.1 + random.gauss(0, 0.1)
        price.append(price[-1] + rate)

    print("EXP-DOP-01 — Saturation Threshold Detection")
    print("=" * 55)

    states = detect_saturation(price)
    print(f"\n{'Bar':>4} {'Rate':>8} {'Accel':>8} {'Sat':>5} {'Past':>6} {'Window':>8}")
    print("─" * 45)
    for s in states:
        win = "◉ COLLAPSE" if s.collapse_window_open else ""
        sat = "✓" if s.saturated else "─"
        print(f"{s.bar:>4} {s.expansion_rate:>8.4f} {s.acceleration:>8.4f} "
              f"{sat:>5} {s.bars_past_saturation:>6} {win}")
