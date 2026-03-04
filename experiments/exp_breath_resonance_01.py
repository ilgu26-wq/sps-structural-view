"""
EXP-BRE-01 — Breath Resonance Cycle Detection
===============================================
Identifies periodic oscillations in coherence that correspond
to structural "breath" — cyclical expansion and contraction
within Phase 1–2 that should NOT trigger early exits.

Trades that exhibit breath resonance during Phase 1 are
incorrectly exited by EXIT-A, which mistakes oscillatory
coherence dips for collapse onset.

Output: resonance cycle markers + breath-corrected coherence.
"""

import math
import statistics
from dataclasses import dataclass


@dataclass
class BreathState:
    bar: int
    coherence_raw: float
    coherence_corrected: float
    in_resonance: bool
    cycle_phase: str     # "inhale" | "exhale" | "neutral"
    suppress_exit: bool  # True if exit should be suppressed (breath dip, not collapse)


RESONANCE_PERIOD = 6    # bars — typical breath cycle length
RESONANCE_AMP    = 0.12 # amplitude threshold to classify as resonance
SUPPRESSION_KAPPA = 0.28  # coherence floor below which suppression does NOT apply


def detect_breath_resonance(
    coherence_series: list[float],
    period: int = RESONANCE_PERIOD,
) -> list[BreathState]:
    """
    Detect breath resonance cycles in coherence series.

    Resonance = regular oscillation with period ≈ RESONANCE_PERIOD bars
    and amplitude ≈ RESONANCE_AMP.

    Corrected coherence smooths out resonance oscillations to expose
    the underlying structural trend.
    """
    n = len(coherence_series)
    states = []

    for i in range(n):
        raw = coherence_series[i]

        # Estimate local oscillation from recent window
        window = coherence_series[max(0, i - period):i + 1]
        if len(window) >= 4:
            local_mean = statistics.mean(window)
            local_std = statistics.stdev(window) if len(window) > 1 else 0
            in_res = local_std > RESONANCE_AMP * 0.7
            # Corrected: smooth toward local mean
            corrected = 0.7 * raw + 0.3 * local_mean
        else:
            in_res = False
            corrected = raw
            local_mean = raw

        # Cycle phase classification
        if in_res and i > 0:
            prev = coherence_series[i - 1]
            if raw > prev:
                cycle = "inhale"
            elif raw < prev:
                cycle = "exhale"
            else:
                cycle = "neutral"
        else:
            cycle = "neutral"

        # Suppress exit only if:
        # 1. In resonance
        # 2. Current dip is above suppression floor (not true collapse)
        suppress = in_res and raw > SUPPRESSION_KAPPA

        states.append(BreathState(
            bar=i,
            coherence_raw=round(raw, 4),
            coherence_corrected=round(corrected, 4),
            in_resonance=in_res,
            cycle_phase=cycle,
            suppress_exit=suppress,
        ))

    return states


if __name__ == "__main__":
    import random
    random.seed(55)

    n = 22
    # Coherence with embedded oscillation then true collapse
    coherence = []
    for i in range(n):
        base = 0.7 * math.exp(-0.015 * max(0, i - 10))
        oscillation = 0.14 * math.sin(2 * math.pi * i / 6) if i < 16 else 0
        noise = random.gauss(0, 0.03)
        coherence.append(max(-1, min(1, base + oscillation + noise)))

    print("EXP-BRE-01 — Breath Resonance Detection")
    print("=" * 55)

    states = detect_breath_resonance(coherence)

    print(f"\n{'Bar':>4} {'C_raw':>7} {'C_corr':>8} {'Resonance':>10} {'Phase':>8} {'Suppress':>9}")
    print("─" * 55)
    for s in states:
        res = "✓" if s.in_resonance else "─"
        sup = "⚑ HOLD" if s.suppress_exit else ""
        print(f"{s.bar:>4} {s.coherence_raw:>7.4f} {s.coherence_corrected:>8.4f} "
              f"{res:>10} {s.cycle_phase:>8} {sup:>9}")

    suppressed = sum(1 for s in states if s.suppress_exit)
    print(f"\nBars with exit suppressed (resonance): {suppressed}/{n}")
    print("These would be premature exits under EXIT-A logic.")
