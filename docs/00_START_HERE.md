# Where to Start

This document describes what is in this repository and what order to read it in.

---

## What this repository is

An analysis of exit behavior in a systematic trading system (SOAR).

The central observation is that trades held to a structural exit signal
outperformed trades exited at a fixed time threshold — not by a small margin,
but by a factor of roughly 3x in expectancy across 247 trades.

This repository documents that observation, the experimental design used to
isolate it, and the geometric model proposed to explain it.

It does not contain live trading code. It does not contain the core system.
It is analysis only.

---

## Suggested reading order

**If you have 5 minutes:**

- `README.md` — overview and key result table

**If you want the argument:**

1. `docs/01_phase_structure_hypothesis.md` — the core claim
2. `docs/02_exit_ablation_study.md` — how it was tested and what happened

**If you want the model:**

3. `docs/03_mobius_manifold_model.md` — geometric framework
4. `docs/04_architecture_overview.md` — system structure

**Reference:**

- `docs/05_quick_reference.md` — glossary, parameter table, file index

---

## Running the experiments

```bash
# Full three-condition comparison
python experiments/validation_harness.py

# Individual experiments
python experiments/exp_rot_phase_01.py
python experiments/exp_mobius_fold_01.py
python experiments/exp_shadow_line_exit.py
```

Each experiment file runs standalone and prints output to the console.

---

## Caveats

The 247-trade sample is real but limited. The phase timing estimates
(~200s formation, ~320s saturation) are population medians and vary
by regime. The geometric model is a framework for thinking about the
problem, not a finished theory.

The ablation result is genuine. The explanation for it is provisional.
