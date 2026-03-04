"""
EXP-TOP-01 — Topology Lock Identification
===========================================
Detects topology lock events: points where rotational phase θ
stalls in the saturation zone (135°–160°) — early warning of Phase 3 onset.

A topology lock does not guarantee collapse, but it precedes
Phase 3 in the majority of confirmed exit events.

Output: lock events with duration, depth, and collapse probability estimate.
"""

from dataclasses import dataclass


SATURATION_LOW  = 135.0   # degrees — saturation zone lower bound
SATURATION_HIGH = 160.0   # degrees — saturation zone upper bound
STALL_RATE      = 0.8     # degrees/bar — stall threshold
MIN_LOCK_BARS   = 2       # minimum consecutive bars in stall to confirm lock


@dataclass
class LockEvent:
    start_bar: int
    end_bar: int
    duration_bars: int
    theta_at_lock: float
    avg_d_theta: float              # average angular velocity during lock
    collapse_prob: float            # estimated probability of Phase 3 onset


def detect_topology_locks(
    theta_series: list[float],
    d_theta_series: list[float],
) -> list[LockEvent]:
    """
    Detect topology lock events from θ and dθ/dt series.

    Lock = MIN_LOCK_BARS+ consecutive bars satisfying:
      1. SATURATION_LOW <= θ <= SATURATION_HIGH
      2. |dθ| < STALL_RATE

    Collapse probability estimated from lock depth (how far θ is
    from expansion zone) and lock duration.
    """
    n = min(len(theta_series), len(d_theta_series))
    events = []
    lock_start = None
    lock_thetas = []
    lock_dthetas = []

    for i in range(n):
        theta = theta_series[i] % 360
        d_theta = abs(d_theta_series[i])
        in_saturation = SATURATION_LOW <= theta <= SATURATION_HIGH
        stalled = d_theta < STALL_RATE

        if in_saturation and stalled:
            if lock_start is None:
                lock_start = i
                lock_thetas = []
                lock_dthetas = []
            lock_thetas.append(theta)
            lock_dthetas.append(d_theta)
        else:
            if lock_start is not None:
                duration = i - lock_start
                if duration >= MIN_LOCK_BARS:
                    avg_theta = sum(lock_thetas) / len(lock_thetas)
                    avg_d = sum(lock_dthetas) / len(lock_dthetas)
                    # Depth into saturation zone
                    depth = (avg_theta - SATURATION_LOW) / (SATURATION_HIGH - SATURATION_LOW)
                    # Duration factor
                    dur_factor = min(1.0, duration / 6)
                    p_collapse = round(0.5 * depth + 0.5 * dur_factor, 3)
                    events.append(LockEvent(
                        start_bar=lock_start,
                        end_bar=i - 1,
                        duration_bars=duration,
                        theta_at_lock=round(avg_theta, 2),
                        avg_d_theta=round(avg_d, 3),
                        collapse_prob=p_collapse,
                    ))
            lock_start = None
            lock_thetas = []
            lock_dthetas = []

    # Close open lock at end
    if lock_start is not None:
        duration = n - lock_start
        if duration >= MIN_LOCK_BARS:
            avg_theta = sum(lock_thetas) / len(lock_thetas)
            avg_d = sum(lock_dthetas) / len(lock_dthetas)
            depth = (avg_theta - SATURATION_LOW) / (SATURATION_HIGH - SATURATION_LOW)
            dur_factor = min(1.0, duration / 6)
            p_collapse = round(0.5 * depth + 0.5 * dur_factor, 3)
            events.append(LockEvent(
                start_bar=lock_start,
                end_bar=n - 1,
                duration_bars=duration,
                theta_at_lock=round(avg_theta, 2),
                avg_d_theta=round(avg_d, 3),
                collapse_prob=p_collapse,
            ))

    return events


# ─── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(42)

    # Simulated θ series: rises, stalls in saturation, then breaks out/collapses
    theta = []
    d_theta = []
    t = 0.0
    for i in range(25):
        if i < 10:
            increment = random.gauss(9, 2)    # expansion sweep
        elif i < 16:
            increment = random.gauss(0.4, 0.3)  # stall in saturation
        else:
            increment = random.gauss(12, 3)   # collapse breakout
        increment = max(0, increment)
        t += increment
        theta.append(t % 360)
        d_theta.append(increment)

    print("EXP-TOP-01 — Topology Lock Identification")
    print("=" * 50)

    print(f"\n{'Bar':>4} {'θ':>8} {'dθ':>7} {'Zone':>12}")
    print("─" * 40)
    for i, (th, dt) in enumerate(zip(theta, d_theta)):
        th_mod = th % 360
        if SATURATION_LOW <= th_mod <= SATURATION_HIGH:
            zone = "SATURATION"
        elif th_mod < SATURATION_LOW:
            zone = "expansion"
        else:
            zone = "post-sat"
        print(f"{i:>4} {th_mod:>8.2f}° {dt:>6.2f}°  {zone:>12}")

    events = detect_topology_locks(theta, d_theta)
    print(f"\nTopology Locks Detected: {len(events)}")
    for ev in events:
        print(f"\n  Bars {ev.start_bar}–{ev.end_bar} (duration: {ev.duration_bars} bars)")
        print(f"    θ at lock:       {ev.theta_at_lock}°")
        print(f"    avg dθ:          {ev.avg_d_theta}°/bar")
        print(f"    collapse P:      {ev.collapse_prob:.3f}")
        print(f"    assessment:      {'HIGH RISK' if ev.collapse_prob > 0.65 else 'moderate'}")
