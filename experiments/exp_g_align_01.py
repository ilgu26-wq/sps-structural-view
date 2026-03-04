"""
EXP-GAL-01 — Geometric Alignment Probe
=========================================
Measures multi-dimensional structural alignment across the feature space.
High g_align = multiple dimensions coherently aligned → hold.
Low g_align  = dimensional decoherence → monitor for collapse.

g_align_v2 adds rotational correction for regime-dependent compression.

Output: g_align score series with alignment breakdown per dimension.
"""

import math
import statistics
from dataclasses import dataclass, field


@dataclass
class AlignmentState:
    bar: int
    g_align: float              # [0, 1] — overall alignment score
    dim_scores: dict            # per-dimension alignment
    rotational_correction: float
    regime_compressed: bool
    decoherence_flag: bool      # True if g_align < threshold


G_ALIGN_FLOOR = 0.35           # decoherence threshold
REGIME_COMPRESSION_THRESHOLD = 0.6  # volatility ratio above which regime compression applies


def _normalize(series: list[float]) -> list[float]:
    """Z-score normalize a series."""
    if len(series) < 2:
        return [0.0] * len(series)
    mu = statistics.mean(series)
    sigma = statistics.stdev(series) or 1e-9
    return [(x - mu) / sigma for x in series]


def _pairwise_alignment(a: list[float], b: list[float]) -> float:
    """
    Alignment between two normalized feature series.
    Returns cosine similarity ∈ [-1, 1] → remapped to [0, 1].
    """
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x**2 for x in a)) or 1e-9
    norm_b = math.sqrt(sum(x**2 for x in b)) or 1e-9
    cosine = dot / (norm_a * norm_b)
    return (cosine + 1) / 2  # remap to [0, 1]


def compute_g_align(
    price_series: list[float],
    volume_series: list[float],
    momentum_series: list[float],
    theta_series: list[float],
    window: int = 8,
    rotational_correction: bool = True,
) -> list[AlignmentState]:
    """
    Compute g_align_v2 alignment score series.

    Dimensions:
      - price_momentum alignment   (Δp vs momentum)
      - volume_momentum alignment  (volume vs momentum)
      - phase_momentum alignment   (θ vs momentum)
      - cross alignment            (all dimensions)

    Rotational correction: scales alignment by rotational coherence
    to account for regime-dependent phase compression.

    Args:
        window: lookback window for alignment computation (bars)

    Returns:
        List of AlignmentState per bar
    """
    n = len(price_series)
    if n < window:
        raise ValueError(f"Need at least {window} bars; got {n}")

    # Compute derived series
    delta_p = [price_series[i] - price_series[i-1] for i in range(1, n)]
    delta_p = [delta_p[0]] + delta_p  # pad

    norm_dp  = _normalize(delta_p)
    norm_vol = _normalize(volume_series)
    norm_mom = _normalize(momentum_series)
    norm_th  = _normalize(theta_series)

    states = []

    for i in range(window, n):
        w_dp  = norm_dp[i-window:i]
        w_vol = norm_vol[i-window:i]
        w_mom = norm_mom[i-window:i]
        w_th  = norm_th[i-window:i]

        # Pairwise alignments
        a_pm  = _pairwise_alignment(w_dp, w_mom)    # price vs momentum
        a_vm  = _pairwise_alignment(w_vol, w_mom)   # volume vs momentum
        a_phm = _pairwise_alignment(w_th, w_mom)    # phase vs momentum
        a_pv  = _pairwise_alignment(w_dp, w_vol)    # price vs volume

        # Weighted composite
        raw_score = 0.35 * a_pm + 0.25 * a_vm + 0.25 * a_phm + 0.15 * a_pv

        # Rotational correction: suppress alignment score in compressed regimes
        vol_ratio = (statistics.stdev(w_vol) / (statistics.mean([abs(x) for x in w_vol]) + 1e-9))
        compressed = vol_ratio > REGIME_COMPRESSION_THRESHOLD
        rot_corr = 1.0 - (0.15 * compressed)
        final_score = round(raw_score * rot_corr, 4)

        states.append(AlignmentState(
            bar=i,
            g_align=final_score,
            dim_scores={
                "price_momentum": round(a_pm, 3),
                "volume_momentum": round(a_vm, 3),
                "phase_momentum": round(a_phm, 3),
                "price_volume": round(a_pv, 3),
            },
            rotational_correction=round(rot_corr, 3),
            regime_compressed=compressed,
            decoherence_flag=final_score < G_ALIGN_FLOOR,
        ))

    return states


# ─── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(99)

    n = 25
    # Synthetic: aligned trend then decoherence
    price = [100.0]
    for i in range(1, n):
        drift = 0.4 if i < 16 else -0.1
        price.append(price[-1] + drift + random.gauss(0, 0.1))

    volume = [1000 + 50 * math.sin(i * 0.5) + random.gauss(0, 80)
              for i in range(n)]
    momentum = [price[i] - price[max(0, i-3)] for i in range(n)]
    theta = [i * 7.5 + random.gauss(0, 2) for i in range(n)]

    print("EXP-GAL-01 — Geometric Alignment Probe (g_align_v2)")
    print("=" * 60)

    states = compute_g_align(price, volume, momentum, theta)

    print(f"\n{'Bar':>4} {'g_align':>8} {'Rot.Corr':>9} {'Compressed':>11} {'Decoherence':>12}")
    print("─" * 50)
    for s in states:
        dc = "⚠ DECOHERE" if s.decoherence_flag else ""
        comp = "✓" if s.regime_compressed else "─"
        print(f"{s.bar:>4} {s.g_align:>8.4f} {s.rotational_correction:>9.3f} "
              f"{comp:>11} {dc:>12}")

    g_scores = [s.g_align for s in states]
    print(f"\ng_align stats:")
    print(f"  mean:  {statistics.mean(g_scores):.4f}")
    print(f"  min:   {min(g_scores):.4f}")
    print(f"  max:   {max(g_scores):.4f}")
    print(f"  decoherence events: {sum(1 for s in states if s.decoherence_flag)}/{len(states)}")

import math  # already imported above; guard for standalone usage
