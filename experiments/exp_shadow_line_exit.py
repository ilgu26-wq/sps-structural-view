"""
EXP-SHD-01 — Shadow Line Exit Signal
=======================================
Dynamic structural boundary derived from the trailing edge
of the Möbius band as the trajectory progresses.

Shadow line = surface-relative support, not a fixed price level.
Moves with trajectory during Phases 0–1. Locks during Phase 2.
Exit fires when trajectory crosses shadow boundary AND coherence < κ_shadow.

Output: shadow line levels + crossing events.
"""

import statistics
from dataclasses import dataclass


@dataclass
class ShadowState:
    bar: int
    price: float
    shadow_level: float
    above_shadow: bool
    coherence: float
    exit_signal: bool
    signal_reason: str


KAPPA_SHADOW = 0.30       # coherence threshold for shadow exit confirmation
SHADOW_LOOKBACK = 5       # bars for trailing boundary computation
SHADOW_OFFSET = 0.95      # shadow as fraction of trailing structure


def compute_shadow_level(
    price_history: list[float],
    direction: str,
    lookback: int = SHADOW_LOOKBACK,
) -> float:
    """
    Compute shadow line level from trailing price structure.

    Long:  shadow = SHADOW_OFFSET × min(recent prices)
    Short: shadow = (1/SHADOW_OFFSET) × max(recent prices)
    """
    window = price_history[-lookback:]
    if direction == "long":
        return min(window) * SHADOW_OFFSET
    else:
        return max(window) / SHADOW_OFFSET


def shadow_line_exit(
    price_series: list[float],
    coherence_series: list[float],
    direction: str = "long",
    phase_series: list[int] | None = None,
) -> list[ShadowState]:
    """
    Run EXP-SHD-01 shadow line analysis.

    Shadow line is active only from Phase 2 onward.
    Before Phase 2: shadow level computed but no exit signal.

    Args:
        price_series:     Price per bar
        coherence_series: Coherence C(t) per bar
        direction:        "long" | "short"
        phase_series:     Phase label per bar (optional; derived if not provided)

    Returns:
        List of ShadowState per bar
    """
    if len(price_series) != len(coherence_series):
        raise ValueError("price_series and coherence_series must have equal length")

    states = []

    for i in range(len(price_series)):
        price = price_series[i]
        coherence = coherence_series[i]

        # Shadow level requires at least SHADOW_LOOKBACK bars of history
        history = price_series[max(0, i - SHADOW_LOOKBACK + 1):i + 1]
        shadow = compute_shadow_level(history, direction)

        above = price >= shadow if direction == "long" else price <= shadow

        # Phase from external series or simplified heuristic
        if phase_series is not None:
            phase = phase_series[i]
        else:
            phase = 0 if i < 7 else (1 if i < 11 else (2 if i < 14 else 3))

        # Exit conditions
        exit_signal = False
        reason = ""

        if phase >= 2:
            if not above:
                if coherence < KAPPA_SHADOW:
                    exit_signal = True
                    reason = "shadow_breach+coherence_low"
                else:
                    reason = "shadow_breach (coherence holding)"
            elif coherence < KAPPA_SHADOW * 0.7:
                # Extreme coherence collapse even without shadow breach
                exit_signal = True
                reason = "coherence_floor"

        states.append(ShadowState(
            bar=i,
            price=round(price, 4),
            shadow_level=round(shadow, 4),
            above_shadow=above,
            coherence=round(coherence, 3),
            exit_signal=exit_signal,
            signal_reason=reason,
        ))

    return states


# ─── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random, math
    random.seed(21)

    n_bars = 20
    # Uptrend then breakdown
    price = [100.0]
    for i in range(1, n_bars):
        if i < 12:
            price.append(price[-1] + random.gauss(0.25, 0.12))
        else:
            price.append(price[-1] + random.gauss(-0.35, 0.18))

    # Coherence rises then collapses
    coherence = []
    for i in range(n_bars):
        c = 0.8 * math.exp(-0.012 * max(0, (i - 10))) + random.gauss(0, 0.04)
        coherence.append(max(-1, min(1, c)))

    print("EXP-SHD-01 — Shadow Line Exit Signal")
    print("=" * 60)
    print(f"Direction: LONG | κ_shadow = {KAPPA_SHADOW} | Lookback = {SHADOW_LOOKBACK}\n")

    states = shadow_line_exit(price, coherence, direction="long")

    phase_labels = {0: "FORM", 1: "EXPN", 2: "SATN", 3: "COLL"}

    print(f"{'Bar':>4} {'Price':>8} {'Shadow':>8} {'Above':>6} {'C':>6} {'Signal':>8}  Reason")
    print("─" * 60)
    for s in states:
        ph = 0 if s.bar < 7 else (1 if s.bar < 11 else (2 if s.bar < 14 else 3))
        sig = "⚡ EXIT" if s.exit_signal else ("watch" if s.signal_reason else "─")
        print(f"{s.bar:>4} {s.price:>8.3f} {s.shadow_level:>8.3f} "
              f"{'✓' if s.above_shadow else '✗':>6} {s.coherence:>6.3f} "
              f"{sig:>8}  {s.signal_reason}")

    exit_bars = [s.bar for s in states if s.exit_signal]
    print(f"\nExit signals: {exit_bars}")
    print(f"First exit signal: Bar {exit_bars[0]}" if exit_bars else "No exit signal fired")
