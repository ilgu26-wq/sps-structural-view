"""
EXP-MOB-01 — Möbius Fold Detection
=====================================
Identifies fold crossing events on the Möbius manifold —
the geometric locus where surface orientation transitions.

Fold crossings empirically correspond to Phase 2→3 transitions.

Output: fold crossing timestamps with coherence state at crossing.
"""

import math
import statistics
from dataclasses import dataclass


@dataclass
class FoldEvent:
    bar: int
    elapsed_seconds: float
    coherence_at_crossing: float
    orientation_before: int   # +1 or -1
    orientation_after: int
    confidence: float         # [0, 1] — fold detection confidence


def _orientation(theta: float) -> int:
    """
    Orientation on Möbius band: +1 or -1 based on angular position.
    Orientation reversal occurs at seam crossing (~180° transition).
    """
    return 1 if (theta % 360) < 180 else -1


def _coherence(elapsed: float, structural_anchor: float = 0.7) -> float:
    """Simplified coherence estimator for fold detection."""
    peak = 290.0
    sigma = 120.0
    base = structural_anchor * math.exp(-((elapsed - peak) ** 2) / (2 * sigma ** 2))
    return max(-1.0, min(1.0, base))


def detect_fold_crossings(
    theta_series: list[float],
    elapsed_series: list[float],
    structural_anchor: float = 0.7,
    min_coherence_drop: float = 0.15,
) -> list[FoldEvent]:
    """
    Detect Möbius fold crossings in a θ(t) time series.

    A fold crossing is detected when:
    1. Orientation sign changes (+1 → -1 or -1 → +1)
    2. Accompanied by coherence drop ≥ min_coherence_drop

    Args:
        theta_series:       Rotational phase angles (degrees) per bar
        elapsed_series:     Elapsed seconds per bar
        structural_anchor:  Coherence anchor strength [0, 1]
        min_coherence_drop: Minimum coherence gradient to confirm fold

    Returns:
        List of FoldEvent
    """
    events = []
    prev_orientation = _orientation(theta_series[0]) if theta_series else 1

    for i in range(1, len(theta_series)):
        curr_orientation = _orientation(theta_series[i])
        elapsed = elapsed_series[i]
        coherence = _coherence(elapsed, structural_anchor)

        if curr_orientation != prev_orientation:
            # Orientation reversal detected — check coherence gradient
            if i >= 2:
                c_prev = _coherence(elapsed_series[i-2], structural_anchor)
                c_curr = coherence
                delta_c = c_prev - c_curr
            else:
                delta_c = 0.0

            if delta_c >= min_coherence_drop:
                confidence = min(1.0, delta_c / (min_coherence_drop * 2))
                events.append(FoldEvent(
                    bar=i,
                    elapsed_seconds=round(elapsed, 1),
                    coherence_at_crossing=round(coherence, 3),
                    orientation_before=prev_orientation,
                    orientation_after=curr_orientation,
                    confidence=round(confidence, 3),
                ))

        prev_orientation = curr_orientation

    return events


def mobius_position(theta: float, coherence: float) -> tuple[float, float]:
    """
    Map (θ, C) to Möbius band coordinates (u, w).
    u ∈ [0, 1] = phase progression
    w ∈ [-1, 1] = coherence signed amplitude
    """
    u = (theta % 360) / 360.0
    w = coherence * math.cos(math.pi * u)  # orientation-aware projection
    return round(u, 4), round(w, 4)


# ─── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(13)

    n_bars = 25
    bar_seconds = 30

    # Simulate θ progressing through a full phase lifecycle
    theta = [0.0]
    for i in range(1, n_bars):
        increment = random.gauss(8, 3)  # ~8 degrees per bar
        theta.append(theta[-1] + max(0, increment))

    elapsed = [i * bar_seconds for i in range(n_bars)]

    print("EXP-MOB-01 — Möbius Fold Detection")
    print("=" * 50)

    events = detect_fold_crossings(theta, elapsed, structural_anchor=0.75)

    print(f"\nθ progression ({n_bars} bars):")
    print(f"{'Bar':>4} {'θ':>8} {'Elapsed':>9} {'Orient':>7} {'(u, w)':>18}")
    print("─" * 50)
    for i, (t, e) in enumerate(zip(theta, elapsed)):
        orient = _orientation(t)
        coh = _coherence(e)
        u, w = mobius_position(t, coh)
        marker = " ← FOLD" if any(ev.bar == i for ev in events) else ""
        print(f"{i:>4} {t:>8.1f}° {e:>8.0f}s {orient:>+7} ({u:>6.3f}, {w:>+6.3f}){marker}")

    print(f"\nFold Crossings Detected: {len(events)}")
    for ev in events:
        print(f"\n  Bar {ev.bar} @ {ev.elapsed_seconds}s")
        print(f"    Orientation: {ev.orientation_before:+d} → {ev.orientation_after:+d}")
        print(f"    Coherence at crossing: {ev.coherence_at_crossing:.3f}")
        print(f"    Confidence: {ev.confidence:.3f}")
        print(f"    → EXIT SIGNAL" if ev.confidence > 0.6 else f"    → monitor (low confidence)")
