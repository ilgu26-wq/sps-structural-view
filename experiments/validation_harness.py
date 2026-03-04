"""
VAL-01 — Validation Harness
============================
Compares baseline (EXIT-A) vs patched (EXIT-B) exit logic
on an identical trade universe.

Core (SOAR1) is held invariant. Only the EXIT function changes.

Usage:
    python experiments/validation_harness.py
    python experiments/validation_harness.py --condition EXIT-B
    python experiments/validation_harness.py --all --verbose
"""

import json
import argparse
import statistics
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class Trade:
    trade_id: str
    entry_time: float       # seconds from session open
    direction: str          # "long" | "short"
    entry_price: float
    structural_anchor: float
    phase_at_entry: int     # 0–3
    true_r_multiple: float  # realized R if held to structural exit
    duration_true: float    # true structural hold in seconds

@dataclass
class ExitResult:
    trade_id: str
    exit_time: float
    exit_price: float
    r_multiple: float
    hold_bars: float
    phase_at_exit: int
    condition: str
    exit_reason: str


# ─── Phase Classifier ────────────────────────────────────────────────────────

def classify_phase(elapsed_seconds: float, coherence: float) -> int:
    """
    Classify current phase based on elapsed time and coherence.
    Time thresholds are population medians, not hard rules.
    Coherence state can override time-based classification.
    """
    if elapsed_seconds < 200:
        return 0  # Formation
    elif elapsed_seconds < 320:
        if coherence < 0.4:
            return 3  # Early collapse
        return 1  # Expansion
    elif elapsed_seconds < 360:
        if coherence < 0.35:
            return 3  # Saturation→Collapse
        return 2  # Saturation
    else:
        return 3  # Collapse


# ─── Exit Conditions ─────────────────────────────────────────────────────────

def coherence_model(elapsed: float, trade: Trade) -> float:
    """
    Simplified coherence model for validation.
    Real implementation uses full Möbius manifold projection.
    """
    import math
    # Coherence peaks during expansion and decays post-saturation
    peak_time = 290.0
    decay_rate = 0.008
    base = math.exp(-decay_rate * max(0, elapsed - peak_time))
    noise = 0.05 * math.sin(elapsed * 0.12)  # structural oscillation
    return min(1.0, max(-1.0, base + noise))


def exit_a(trade: Trade, elapsed: float) -> Optional[str]:
    """
    EXIT-A — Baseline
    Exits at fixed τ = 6.3 bars (~189s) regardless of phase.
    """
    BAR_SECONDS = 30
    TAU_BARS = 6.3
    if elapsed >= TAU_BARS * BAR_SECONDS:
        return "tau_threshold"
    return None


def exit_b(trade: Trade, elapsed: float, coherence: float) -> Optional[str]:
    """
    EXIT-B — Extended τ (Structural)
    Exits on coherence collapse or Phase 3 onset, whichever comes first.
    Respects phase lifecycle. Does not exit during Phase 0–1.
    """
    KAPPA = 0.25          # coherence collapse threshold
    MIN_HOLD = 200.0      # minimum hold (Phase 0 floor)

    if elapsed < MIN_HOLD:
        return None  # never exit during formation

    phase = classify_phase(elapsed, coherence)

    if phase == 3 and coherence < KAPPA:
        return "coherence_collapse"

    # Safety: max hold 500s (structural outlier boundary)
    if elapsed > 500:
        return "max_hold"

    return None


def exit_c(trade: Trade, elapsed: float, coherence: float) -> Optional[str]:
    """
    EXIT-C — Aggressive (Research Only, Not Prop-Stable)
    Extends hold until coherence reaches floor.
    Highest expectancy, lowest structural validity.
    """
    KAPPA_AGGRESSIVE = 0.15
    MIN_HOLD = 250.0

    if elapsed < MIN_HOLD:
        return None

    if coherence < KAPPA_AGGRESSIVE:
        return "coherence_floor"

    if elapsed > 600:
        return "max_hold_aggressive"

    return None


# ─── Simulation Engine ────────────────────────────────────────────────────────

def simulate_condition(trades: list[Trade], condition: str, verbose: bool = False) -> list[ExitResult]:
    """
    Run exit simulation for a given condition on the trade universe.
    Time step: 1 second resolution.
    """
    results = []
    BAR_SECONDS = 30

    for trade in trades:
        exit_reason = None
        exit_elapsed = 0.0
        dt = 1.0  # 1-second steps

        elapsed = 0.0
        while elapsed <= 700:
            coherence = coherence_model(elapsed, trade)

            if condition == "EXIT-A":
                exit_reason = exit_a(trade, elapsed)
            elif condition == "EXIT-B":
                exit_reason = exit_b(trade, elapsed, coherence)
            elif condition == "EXIT-C":
                exit_reason = exit_c(trade, elapsed, coherence)

            if exit_reason:
                exit_elapsed = elapsed
                break
            elapsed += dt
        else:
            exit_reason = "timeout"
            exit_elapsed = 700.0

        # R-multiple: scaled by ratio of actual hold vs true structural hold
        hold_ratio = min(1.2, exit_elapsed / max(1, trade.duration_true))
        r_mult = trade.true_r_multiple * hold_ratio

        hold_bars = exit_elapsed / BAR_SECONDS
        phase_at_exit = classify_phase(exit_elapsed, coherence_model(exit_elapsed, trade))

        result = ExitResult(
            trade_id=trade.trade_id,
            exit_time=trade.entry_time + exit_elapsed,
            exit_price=trade.entry_price + (r_mult * 10),
            r_multiple=round(r_mult, 3),
            hold_bars=round(hold_bars, 2),
            phase_at_exit=phase_at_exit,
            condition=condition,
            exit_reason=exit_reason,
        )
        results.append(result)

        if verbose:
            print(f"  [{trade.trade_id}] {condition} → R={r_mult:.2f} "
                  f"τ={hold_bars:.1f}bars phase={phase_at_exit} reason={exit_reason}")

    return results


def compute_metrics(results: list[ExitResult]) -> dict:
    """Compute summary performance metrics from exit results."""
    r_multiples = [r.r_multiple for r in results]
    wins = [r for r in results if r.r_multiple > 0]
    losses = [r for r in results if r.r_multiple <= 0]

    avg_win = statistics.mean([r.r_multiple for r in wins]) if wins else 0
    avg_loss = abs(statistics.mean([r.r_multiple for r in losses])) if losses else 0
    rr = avg_win / avg_loss if avg_loss > 0 else float('inf')

    win_rate = len(wins) / len(results) if results else 0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    total_equity = sum(r_multiples)
    max_dd = min(r_multiples)
    avg_hold = statistics.mean([r.hold_bars for r in results])

    return {
        "n_trades": len(results),
        "win_rate": round(win_rate * 100, 1),
        "avg_rr": round(rr, 2),
        "expectancy": round(expectancy, 2),
        "total_equity": round(total_equity, 1),
        "max_dd": round(max_dd, 1),
        "avg_hold_bars": round(avg_hold, 1),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def load_trades(path: str = "data/sample_trade_logs.json") -> list[Trade]:
    """Load trade universe from JSON log file."""
    data_path = Path(path)
    if not data_path.exists():
        print(f"[warn] {path} not found — generating synthetic dataset (247 trades)")
        import random
        random.seed(42)
        trades = []
        for i in range(247):
            trades.append(Trade(
                trade_id=f"T{i+1:04d}",
                entry_time=random.uniform(3600, 28800),
                direction=random.choice(["long", "short"]),
                entry_price=random.uniform(100, 500),
                structural_anchor=random.uniform(-0.5, 0.5),
                phase_at_entry=0,
                true_r_multiple=random.uniform(-2.0, 8.0),
                duration_true=random.uniform(180, 520),
            ))
        return trades

    with open(data_path) as f:
        raw = json.load(f)
    return [Trade(**t) for t in raw]


def print_summary(condition: str, metrics: dict):
    width = 50
    print(f"\n{'─' * width}")
    print(f"  {condition}")
    print(f"{'─' * width}")
    print(f"  Trades        : {metrics['n_trades']}")
    print(f"  Win Rate      : {metrics['win_rate']}%")
    print(f"  Avg R:R       : {metrics['avg_rr']}")
    print(f"  Expectancy    : +{metrics['expectancy']}")
    print(f"  Total Equity  : +{metrics['total_equity']}R")
    print(f"  Max DD        : {metrics['max_dd']}R")
    print(f"  Avg Hold      : {metrics['avg_hold_bars']} bars")
    print(f"{'─' * width}")


def main():
    parser = argparse.ArgumentParser(description="SOAR Exit Validation Harness — VAL-01")
    parser.add_argument("--condition", choices=["EXIT-A", "EXIT-B", "EXIT-C"],
                        help="Run single condition")
    parser.add_argument("--all", action="store_true", default=True,
                        help="Run all conditions (default)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-trade results")
    parser.add_argument("--data", default="data/sample_trade_logs.json",
                        help="Path to trade log JSON")
    args = parser.parse_args()

    print("\n" + "═" * 50)
    print("  SOAR EXIT VALIDATION HARNESS — VAL-01")
    print("  Core: SOAR1 (sealed, invariant)")
    print("═" * 50)

    trades = load_trades(args.data)
    print(f"\n  Loaded {len(trades)} trades from {args.data}")

    conditions = (
        [args.condition] if args.condition
        else ["EXIT-A", "EXIT-B", "EXIT-C"]
    )

    all_metrics = {}
    for cond in conditions:
        print(f"\n  Running {cond}...")
        results = simulate_condition(trades, cond, verbose=args.verbose)
        metrics = compute_metrics(results)
        all_metrics[cond] = metrics
        print_summary(cond, metrics)

    if len(conditions) > 1:
        print(f"\n{'═' * 50}")
        print("  COMPARISON")
        print(f"{'═' * 50}")
        print(f"  {'Condition':<12} {'WinRate':>8} {'RR':>6} {'E[R]':>7} {'Equity':>9} {'MaxDD':>7} {'τ(bars)':>8}")
        print(f"  {'─'*12} {'─'*8} {'─'*6} {'─'*7} {'─'*9} {'─'*7} {'─'*8}")
        for cond, m in all_metrics.items():
            print(f"  {cond:<12} {m['win_rate']:>7}% {m['avg_rr']:>6} "
                  f"{m['expectancy']:>+7} {m['total_equity']:>+9}R "
                  f"{m['max_dd']:>+7}R {m['avg_hold_bars']:>8}")
        print(f"{'═' * 50}\n")


if __name__ == "__main__":
    main()
