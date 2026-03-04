Research exploration of phase-structured exit behavior in algorithmic trading systems.

# SOAR Phase Structure Research

> **Hypothesis:** Trade exit timing follows a geometric phase transition,  
> not a time-parameter optimization problem.

---

## Abstract

This repository documents structural analysis of exit behavior in algorithmic trading systems.

Observed across 247 completed trades and 2,473 exit events, the data suggests that trade lifecycle dynamics cluster around a **coherence collapse phase** rather than fixed time thresholds. The system models price trajectory persistence as a phase manifold where exits occur when structural coherence degrades below a critical threshold.

---

## Core Hypothesis

Traditional systems treat exit timing as a **parameter tuning problem**.  
SOAR treats exit timing as a **geometric phase transition** problem.

```
Entry
  → Structure formation   (~200s)
  → Expansion phase       (~280–320s)
  → Coherence collapse    (~350s)
  → Exit
```

Phase sequence: `formation → alignment → expansion → collapse`

---

## Key Finding: Exit Ablation Study

Execution-only ablation holding core judgment (SOAR1) strictly invariant.  
Variable: EXIT logic only. Entry logic and market data identical across all conditions.

| Condition | Win Rate | RR   | Expectancy | Equity  | Max DD |
|-----------|----------|------|------------|---------|--------|
| EXIT-A (Baseline)   | 82.2%    | 0.99 | +1.21      | +606R   | −8R    |
| EXIT-B (Extended τ) | **89.0%**| **2.24** | **+3.66** | **+1832R** | **−5R** |
| EXIT-C (Aggressive) | 93.2%    | 2.40 | +4.31      | +2155R  | −4R    |

**Verdict:** Performance limits were execution-induced, not core-induced.  
The structural alpha was correct. The execution layer released it prematurely.

> EXIT-C excluded from evaluation regimes: structural validity collapses at 37.5% under live conditions.

---

## Repository Structure

```
soar-phase-structure/
│
├── README.md                         ← This file
│
├── docs/
│   ├── 01_phase_structure_hypothesis.md   ← Core theoretical framework
│   ├── 02_exit_ablation_study.md          ← Ablation methodology & results
│   ├── 03_mobius_manifold_model.md        ← Geometric phase model
│   ├── 04_architecture_overview.md        ← System architecture
│   └── 05_quick_reference.md              ← Glossary & key parameters
│
├── experiments/
│   ├── exp_rot_phase_01.py           ← Rotational phase detection
│   ├── exp_topology_lock_01.py       ← Topology lock identification
│   ├── exp_mobius_fold_01.py         ← Möbius fold geometry
│   ├── exp_shadow_line_exit.py       ← Shadow line exit signal
│   ├── exp_g_align_01.py             ← Geometric alignment probe
│   ├── exp_breath_resonance_01.py    ← Breath resonance cycle
│   ├── exp_dopamine_saturation_01.py ← Saturation threshold detection
│   ├── exp_plastic_feature_01.py     ← Structural plasticity feature
│   ├── exp_code_plasticity_03.py     ← Code plasticity (v3)
│   └── validation_harness.py         ← Baseline vs patched comparison
│
├── data/
│   ├── sample_trade_logs.json        ← 247-trade sample dataset
│   └── g_align_v2.csv               ← Geometric alignment signal data
│
└── results/
    ├── exit_phase_analysis.md        ← Phase clustering analysis
    └── ablation_summary.md           ← Ablation study full results
```

---

## Getting Started

```bash
# Run full validation harness (baseline vs patched exit)
python experiments/validation_harness.py

# Run individual phase experiments
python experiments/exp_rot_phase_01.py
python experiments/exp_mobius_fold_01.py
python experiments/exp_shadow_line_exit.py
```

---

## Theoretical Background

### Phase Manifold Model

Price trajectory is modeled as a **Möbius manifold** — a non-orientable surface where structural coherence is measured as a continuous scalar field. Exit signals are emitted when the coherence gradient crosses a critical inflection point.

Key geometric parameters:

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Structural persistence | τ | Coherence hold duration |
| Phase rotation angle | θ | Angular displacement per bar |
| Fold depth | φ | Möbius fold curvature |
| Coherence threshold | κ | Collapse trigger level |

### Why Not Time-Based?

Time-based exit heuristics assume i.i.d. market structure across intervals.  
Phase-based exits assume market structure is path-dependent and self-similar.

Empirical result: extending τ from 6.3 → 8.6 bars **tripled expectancy**  
without any modification to core judgment (entry selection remained constant).

---

## Research Status

- [x] Exit ablation study complete (EXIT-A / B / C)
- [x] Phase clustering analysis (2,473 exit events)
- [x] Geometric alignment signal validation (g_align_v2)
- [x] Rotational phase detection prototype
- [x] Möbius fold geometry implementation
- [ ] Live regime adaptation (in progress)
- [ ] Multi-asset phase correlation study

---

## Notes

This repository documents architectural analysis and experimental validation  
related to the SOAR trading system. All data is anonymized. No live execution logic is included.

Author: Song Ilgyu  
Repository: `soar-phase-structure`  
License: Research use only
