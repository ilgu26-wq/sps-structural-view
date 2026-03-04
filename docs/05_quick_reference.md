# Quick Reference Guide
## SOAR Exit Research — Document 05

---

## Key Parameters

| Symbol | Name | Description | Typical Range |
|--------|------|-------------|---------------|
| τ | Structural persistence | Coherence hold duration | 6–10 bars |
| κ | Collapse threshold | Coherence gradient trigger | 0.15–0.35 |
| θ | Rotational phase angle | Angular displacement in (Δp, v) plane | 0°–360° |
| φ | Fold depth | Möbius fold curvature coefficient | 0.5–2.0 |
| C(t) | Coherence scalar | Inner product of current vs anchor trajectory | [−1, 1] |
| g_align | Geometric alignment score | Multi-dimensional structural alignment | [0, 1] |
| Δu | Confirmation window | Bars required to confirm collapse | 2–4 bars |

---

## Phase Definitions

| Phase | Label | Duration (median) | Exit Action |
|-------|-------|-------------------|-------------|
| 0 | Formation | 0–200s | HOLD unconditionally |
| 1 | Expansion | 200–320s | HOLD, monitor coherence |
| 2 | Saturation | 320–350s | MONITOR, prepare exit |
| 3 | Collapse | 350s+ | EXIT on fold crossing |

---

## Exit Condition Glossary

**Coherence collapse:** dC/dt < −κ sustained for Δu bars  
**Fold crossing:** trajectory crosses Möbius band seam (orientation reversal)  
**Shadow line breach:** trajectory crosses dynamic structural boundary with C < κ_shadow  
**Topology lock:** θ progression stalls below saturation threshold  
**G-align decoherence:** g_align drops below alignment floor (multi-dim divergence)

---

## Experiment IDs

| EXP ID | Short Name | Key Output |
|--------|-----------|------------|
| EXP-ROT-01 | Rotational Phase | θ(t) time series |
| EXP-TOP-01 | Topology Lock | Lock signal binary |
| EXP-MOB-01 | Möbius Fold | Fold crossing timestamps |
| EXP-SHD-01 | Shadow Line | Dynamic boundary levels |
| EXP-GAL-01 | G-Align | Alignment score series |
| EXP-BRE-01 | Breath Resonance | Resonance cycle markers |
| EXP-DOP-01 | Dopamine Saturation | Saturation threshold events |
| EXP-PLS-01 | Plastic Feature | Plasticity coefficient |
| EXP-CPL-03 | Code Plasticity v3 | Updated plasticity model |
| VAL-01 | Validation Harness | Full ablation comparison |

---

## Ablation Result Reference

| Condition | τ_median | Win Rate | Expectancy | Equity |
|-----------|---------|----------|------------|--------|
| EXIT-A | 6.3 bars | 82.2% | +1.21 | +606R |
| EXIT-B | 8.6 bars | 89.0% | +3.66 | +1832R |
| EXIT-C | 9.9 bars | 93.2% | +4.31 | +2155R |

EXIT-C: research only. Prop-stability failure at 37.5% structural validity.

---

## File Index

```
README.md                          Repository overview and key finding
docs/01_phase_structure_hypothesis.md   Core hypothesis and phase model
docs/02_exit_ablation_study.md         Ablation methodology and results
docs/03_mobius_manifold_model.md       Geometric framework
docs/04_architecture_overview.md       System architecture
docs/05_quick_reference.md             This file

experiments/exp_rot_phase_01.py        EXP-ROT-01
experiments/exp_topology_lock_01.py    EXP-TOP-01
experiments/exp_mobius_fold_01.py      EXP-MOB-01
experiments/exp_shadow_line_exit.py    EXP-SHD-01
experiments/exp_g_align_01.py          EXP-GAL-01
experiments/exp_breath_resonance_01.py EXP-BRE-01
experiments/exp_dopamine_saturation_01.py EXP-DOP-01
experiments/exp_plastic_feature_01.py  EXP-PLS-01
experiments/exp_code_plasticity_03.py  EXP-CPL-03
experiments/validation_harness.py      VAL-01

data/sample_trade_logs.json            247-trade dataset
data/g_align_v2.csv                    G-align signal series

results/exit_phase_analysis.md         Phase clustering analysis
results/ablation_summary.md            Full ablation results
```
