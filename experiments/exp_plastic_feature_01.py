"""
EXP-PLS-01 — Structural Plasticity Feature
============================================
Measures the deformation resistance of price structure:
how much the trajectory can bend before coherence breaks.

High plasticity = structure absorbs perturbations → hold through noise
Low plasticity  = structure is brittle → minor perturbation triggers collapse

Plasticity coefficient P ∈ [0, 1].
"""

import math
import statistics
from dataclasses import dataclass


@dataclass
class PlasticityState:
    bar: int
    plasticity: float       # [0, 1]
    deformation: float      # current perturbation magnitude
    brittle: bool           # True if plasticity < BRITTLE_THRESHOLD


BRITTLE_THRESHOLD = 0.30


def compute_plasticity(
    price_series: list[float],
    coherence_series: list[float],
    window: int = 6,
) -> list[PlasticityState]:
    n = len(price_series)
    states = []

    for i in range(window, n):
        price_win = price_series[i-window:i+1]
        coh_win = coherence_series[i-window:i+1]

        # Deformation: normalized deviation from smooth trend
        trend = (price_win[-1] - price_win[0]) / window
        deviations = [abs(price_win[j] - (price_win[0] + j * trend)) for j in range(len(price_win))]
        deformation = statistics.mean(deviations) / (abs(trend) + 1e-6)
        deformation = min(5.0, deformation)  # cap

        # Plasticity: how well coherence survives deformation
        avg_coh = statistics.mean(coh_win)
        plasticity = max(0.0, min(1.0, avg_coh / (1 + 0.3 * deformation)))

        states.append(PlasticityState(
            bar=i,
            plasticity=round(plasticity, 4),
            deformation=round(deformation, 4),
            brittle=plasticity < BRITTLE_THRESHOLD,
        ))

    return states


if __name__ == "__main__":
    import random
    random.seed(17)

    n = 22
    price = [100.0]
    for i in range(n - 1):
        noise = 0.8 if i > 14 else 0.15
        price.append(price[-1] + random.gauss(0.2, noise))

    coherence = [max(0, min(1, 0.8 - 0.03 * i + random.gauss(0, 0.05))) for i in range(n)]

    print("EXP-PLS-01 — Structural Plasticity Feature")
    print("=" * 50)

    states = compute_plasticity(price, coherence)
    print(f"\n{'Bar':>4} {'Plasticity':>11} {'Deformation':>12} {'Brittle':>8}")
    print("─" * 40)
    for s in states:
        brit = "⚠ BRITTLE" if s.brittle else ""
        print(f"{s.bar:>4} {s.plasticity:>11.4f} {s.deformation:>12.4f} {brit:>8}")
