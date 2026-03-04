# soar-phase-structure

Analysis of exit behavior in an algorithmic trading system.

---

## Summary

A fixed-time exit heuristic was replaced with a structure-aware exit signal.
Tested across 247 trades with entry logic and market data held constant.

| Condition | Win Rate | R:R  | Expectancy | Equity  | Max DD |
|-----------|----------|------|------------|---------|--------|
| EXIT-A (baseline, fixed τ)  | 82.2% | 0.99 | +1.21 | +606R  | −8R |
| EXIT-B (structural exit)    | 89.0% | 2.24 | +3.66 | +1832R | −5R |
| EXIT-C (aggressive, research only) | 93.2% | 2.40 | +4.31 | +2155R | −4R |

EXIT-C is excluded from operational consideration — structural validity rate is 37.5%.
Most of its gains come from holding through noise, not structure.

The operative finding is EXIT-B: roughly 3x expectancy improvement from holding
to a coherence-based signal rather than a fixed time threshold.

---

## The observation

Premature exits discarded valid alpha while the underlying structure was still intact.

EXIT-A exits at median ~6.3 bars (~189s). At that point, based on phase analysis
of 2,473 exit events, 57% of trades were still in the expansion phase.
The structure resolved correctly — it was released before it finished.

EXIT-B holds until coherence degrades below a threshold, which empirically
corresponds to a transition point at ~280–350s from entry.

---

## Proposed explanation

Exit timing may be better modeled as a phase observable than a time parameter.

The price trajectory appears to follow a lifecycle:
```
Entry → formation (~200s) → expansion (~280–320s) → saturation → collapse → exit
```

The collapse point varies per trade. A fixed time threshold will be early
for some trades and late for others. A coherence-based signal adapts.

The geometric model in `docs/03_mobius_manifold_model.md` formalizes this.
It is a framework for thinking about the problem, not a finished theory.

---

## Repository structure

```
docs/           Analysis documents (start with docs/00_START_HERE.md)
experiments/    Experiment scripts (all runnable standalone)
data/           Sample trade logs and signal data
results/        Analysis outputs
```

---

## Running

```bash
python experiments/validation_harness.py     # full three-condition comparison
python experiments/exp_rot_phase_01.py       # individual experiments
```

---

## Notes

- Core system (SOAR1) was not modified in any experiment
- All R-multiples are realized, flat-position (no compounding)
- Sample size is 247 trades — sufficient to observe the pattern, not to prove it
- This repository is analysis only; no live execution logic is included
