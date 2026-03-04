"""
EXP-ROT-01 — Rotational Phase Detection
=========================================
Measures angular displacement θ(t) of price trajectory
in the (Δp, volume-momentum) feature plane.

Phase lifecycle maps to θ progression:
  Formation   : 0°–45°
  Expansion   : 45°–135°
  Saturation  : 135°–160°
  Collapse    : 160°+ or reversal

Output: θ(t) time series with phase annotations.
"""

import math
import statistics
from dataclasses import dataclass


@dataclass
class RotPhaseState:
    theta: float       # current angle (degrees)
    d_theta: float     # angular velocity (degrees/bar)
    phase: int         # 0=Formation, 1=Expansion, 2=Saturation, 3=Collapse
    lock_detected: bool


PHASE_THRESHOLDS = {
    "formation_exit":   45.0,
    "expansion_exit":  135.0,
    "saturation_exit": 160.0,
}

STALL_THRESHOLD = 0.5  # degrees/bar — below this = topology lock candidate


def compute_theta(delta_p: float, volume_momentum: float) -> float:
    """
    Angular displacement in (Δp, v_mom) feature plane.
    Returns angle in degrees [0, 360).
    """
    angle = math.degrees(math.atan2(volume_momentum, delta_p))
    return angle % 360


def classify_from_theta(theta: float, d_theta: float) -> int:
    """Classify phase from angular position and velocity."""
    if theta < PHASE_THRESHOLDS["formation_exit"]:
        return 0
    elif theta < PHASE_THRESHOLDS["expansion_exit"]:
        return 1
    elif theta < PHASE_THRESHOLDS["saturation_exit"]:
        if d_theta < STALL_THRESHOLD:
            return 3  # stall in saturation → early collapse
        return 2
    else:
        return 3


def run_rot_phase(price_series: list[float], volume_series: list[float]) -> list[RotPhaseState]:
    """
    Run EXP-ROT-01 on a price/volume series.

    Args:
        price_series:  list of prices (bars)
        volume_series: list of volumes (bars)

    Returns:
        list of RotPhaseState per bar
    """
    if len(price_series) < 3:
        raise ValueError("Minimum 3 bars required")

    states = []
    theta_history = []

    for i in range(1, len(price_series)):
        delta_p = price_series[i] - price_series[i - 1]

        # Volume momentum: deviation from rolling mean
        vol_window = volume_series[max(0, i-5):i+1]
        v_mean = statistics.mean(vol_window)
        v_mom = (volume_series[i] - v_mean) / max(1, v_mean)

        theta = compute_theta(delta_p, v_mom)
        theta_history.append(theta)

        # Angular velocity (smoothed over last 2 bars)
        if len(theta_history) >= 2:
            d_theta = abs(theta_history[-1] - theta_history[-2])
            # Wrap-around correction
            if d_theta > 180:
                d_theta = 360 - d_theta
        else:
            d_theta = 0.0

        phase = classify_from_theta(theta, d_theta)
        lock = (phase == 2 and d_theta < STALL_THRESHOLD)

        states.append(RotPhaseState(
            theta=round(theta, 2),
            d_theta=round(d_theta, 2),
            phase=phase,
            lock_detected=lock,
        ))

    return states


def summarize(states: list[RotPhaseState]) -> dict:
    phases = [s.phase for s in states]
    thetas = [s.theta for s in states]
    locks = sum(1 for s in states if s.lock_detected)

    phase_labels = {0: "Formation", 1: "Expansion", 2: "Saturation", 3: "Collapse"}
    phase_counts = {v: phases.count(k) for k, v in phase_labels.items()}

    return {
        "total_bars": len(states),
        "theta_mean": round(statistics.mean(thetas), 2),
        "theta_max":  round(max(thetas), 2),
        "theta_min":  round(min(thetas), 2),
        "topology_locks": locks,
        "phase_distribution": phase_counts,
        "final_phase": phase_labels[states[-1].phase] if states else "N/A",
    }


# ─── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(7)

    # Synthetic trade: trending then collapsing
    n_bars = 20
    price = [100.0]
    for i in range(n_bars - 1):
        drift = 0.3 if i < 12 else -0.1
        price.append(price[-1] + drift + random.gauss(0, 0.15))

    volume = [1000 + random.gauss(0, 100) for _ in range(n_bars)]

    print("EXP-ROT-01 — Rotational Phase Detection")
    print("=" * 45)

    states = run_rot_phase(price, volume)

    phase_labels = {0: "FORM", 1: "EXPN", 2: "SATN", 3: "COLL"}
    print(f"\n{'Bar':>4} {'Price':>8} {'θ':>8} {'dθ':>7} {'Phase':>6} {'Lock':>5}")
    print("─" * 45)
    for i, s in enumerate(states):
        lock_mark = "⚑" if s.lock_detected else ""
        print(f"{i+1:>4} {price[i+1]:>8.3f} {s.theta:>8.2f}° {s.d_theta:>6.2f}° "
              f"{phase_labels[s.phase]:>6} {lock_mark:>5}")

    print()
    summary = summarize(states)
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
