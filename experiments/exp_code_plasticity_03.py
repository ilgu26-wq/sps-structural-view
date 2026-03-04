"""
EXP-CPL-03 — Code Plasticity (Iteration 3)
============================================
Updated plasticity model incorporating:
- Regime-adaptive deformation threshold
- Multi-scale deformation measurement (short + long window)
- Cross-feature plasticity (price AND volume structure)

Replaces exp_code_plasticity_01 and _02 with improved
regime sensitivity and reduced false positives.

Changes from v2:
- Multi-scale: adds 12-bar long-window alongside 6-bar short-window
- Volume plasticity: structural resistance in volume dimension
- Composite: harmonic mean of price and volume plasticity

Output: composite plasticity score + regime-adjusted brittle flag.
"""

import math
import statistics
from dataclasses import dataclass


@dataclass
class PlasticityV3State:
    bar: int
    plasticity_price: float
    plasticity_volume: float
    plasticity_composite: float   # harmonic mean
    deformation_short: float      # 6-bar deformation
    deformation_long: float       # 12-bar deformation
    regime_factor: float          # volatility regime adjustment [0.5, 1.5]
    brittle: bool


BRITTLE_THRESHOLD_V3 = 0.28
SHORT_WINDOW = 6
LONG_WINDOW  = 12


def _deformation(series: list[float]) -> float:
    if len(series) < 2:
        return 0.0
    trend = (series[-1] - series[0]) / len(series)
    devs = [abs(series[j] - (series[0] + j * trend)) for j in range(len(series))]
    return statistics.mean(devs) / (abs(trend) + 1e-6)


def _plasticity_score(deformation: float, coherence: float, regime: float) -> float:
    raw = coherence / (1 + 0.25 * deformation * regime)
    return max(0.0, min(1.0, raw))


def harmonic_mean(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return 2 * a * b / (a + b)


def compute_plasticity_v3(
    price_series: list[float],
    volume_series: list[float],
    coherence_series: list[float],
) -> list[PlasticityV3State]:
    n = len(price_series)
    states = []

    for i in range(LONG_WINDOW, n):
        p_short = price_series[i-SHORT_WINDOW:i+1]
        p_long  = price_series[i-LONG_WINDOW:i+1]
        v_short = volume_series[i-SHORT_WINDOW:i+1]
        c = coherence_series[i]

        # Regime factor from volatility ratio
        vol_short = statistics.stdev(p_short) if len(p_short) > 1 else 0.1
        vol_long  = statistics.stdev(p_long) if len(p_long) > 1 else 0.1
        regime = min(1.5, max(0.5, vol_short / (vol_long + 1e-9)))

        # Deformations
        def_short = min(5.0, _deformation(p_short))
        def_long  = min(5.0, _deformation(p_long))
        def_vol   = min(5.0, _deformation(v_short))

        # Plasticity scores
        pl_price = _plasticity_score(0.6 * def_short + 0.4 * def_long, c, regime)
        pl_volume = _plasticity_score(def_vol, c, regime)
        pl_composite = harmonic_mean(pl_price, pl_volume)

        states.append(PlasticityV3State(
            bar=i,
            plasticity_price=round(pl_price, 4),
            plasticity_volume=round(pl_volume, 4),
            plasticity_composite=round(pl_composite, 4),
            deformation_short=round(def_short, 4),
            deformation_long=round(def_long, 4),
            regime_factor=round(regime, 3),
            brittle=pl_composite < BRITTLE_THRESHOLD_V3,
        ))

    return states


if __name__ == "__main__":
    import random
    random.seed(88)

    n = 30
    price = [100.0]
    for i in range(n - 1):
        volatility = 0.1 + 0.4 * (i / n)
        price.append(price[-1] + random.gauss(0.1, volatility))

    volume = [1000 + random.gauss(0, 120) for _ in range(n)]
    coherence = [max(0, min(1, 0.75 - 0.02 * i + random.gauss(0, 0.04))) for i in range(n)]

    print("EXP-CPL-03 — Code Plasticity v3")
    print("=" * 65)

    states = compute_plasticity_v3(price, volume, coherence)
    print(f"\n{'Bar':>4} {'P_price':>8} {'P_vol':>7} {'P_comp':>7} {'Regime':>7} {'Brittle':>8}")
    print("─" * 50)
    for s in states:
        brit = "⚠" if s.brittle else ""
        print(f"{s.bar:>4} {s.plasticity_price:>8.4f} {s.plasticity_volume:>7.4f} "
              f"{s.plasticity_composite:>7.4f} {s.regime_factor:>7.3f} {brit:>8}")
