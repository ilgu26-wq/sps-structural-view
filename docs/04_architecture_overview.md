# Architecture Overview
## SOAR Exit Research — Document 04

---

## 1. System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        SOAR SYSTEM                          │
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  CORE       │    │  EXECUTION   │    │  RESEARCH     │  │
│  │  (SOAR1)    │───▶│  LAYER       │    │  SURFACE      │  │
│  │             │    │  (Exit Logic)│    │  (This Repo)  │  │
│  │  [SEALED]   │    │  [Variable]  │    │  [Analysis]   │  │
│  └─────────────┘    └──────────────┘    └───────────────┘  │
│        │                   │                    │           │
│   Alpha selection     Exit signal           Phase model     │
│   Entry filtering     Phase detection       Ablation study  │
│   Directional bias    τ management          Experiment log  │
└─────────────────────────────────────────────────────────────┘
```

**Core (SOAR1):** Sealed judgment layer. Not modified in any experiment.  
**Execution Layer:** Exit signal generation. Variable across ablation conditions.  
**Research Surface:** This repository. Analysis and experimental validation only.

---

## 2. Exit Logic Architecture

```
Market Data
    │
    ▼
┌───────────────────────────────┐
│       FEATURE EXTRACTION      │
│  price · volume · volatility  │
│  momentum · angular phase     │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│       PHASE CLASSIFIER        │
│  Formation / Expansion /      │
│  Saturation / Collapse        │
└──────┬──────────────┬─────────┘
       │              │
       ▼              ▼
┌──────────┐   ┌──────────────────────────┐
│  HOLD    │   │    EXIT SIGNAL LAYER     │
│  (Ph 0–1)│   │                          │
└──────────┘   │  ┌──────────────────┐    │
               │  │  Coherence C(t)  │    │
               │  │  Shadow Line     │    │
               │  │  Topology Lock   │    │
               │  │  G-Align Score   │    │
               │  └────────┬─────────┘    │
               │           │              │
               │           ▼              │
               │  ┌──────────────────┐    │
               │  │  EXIT DECISION   │    │
               │  │  (Phase 2→3      │    │
               │  │   fold crossing) │    │
               │  └──────────────────┘    │
               └──────────────────────────┘
```

---

## 3. Experiment Catalog

| File | EXP ID | Purpose | Status |
|------|--------|---------|--------|
| `exp_rot_phase_01.py` | EXP-ROT-01 | Rotational phase angle detection | Complete |
| `exp_topology_lock_01.py` | EXP-TOP-01 | Topology lock identification | Complete |
| `exp_mobius_fold_01.py` | EXP-MOB-01 | Möbius fold geometry | Complete |
| `exp_shadow_line_exit.py` | EXP-SHD-01 | Shadow line exit signal | Complete |
| `exp_g_align_01.py` | EXP-GAL-01 | Geometric alignment probe | Complete |
| `exp_breath_resonance_01.py` | EXP-BRE-01 | Breath resonance cycle detection | Complete |
| `exp_dopamine_saturation_01.py` | EXP-DOP-01 | Saturation threshold detection | Complete |
| `exp_plastic_feature_01.py` | EXP-PLS-01 | Structural plasticity feature | Complete |
| `exp_code_plasticity_03.py` | EXP-CPL-03 | Code plasticity (iteration 3) | Complete |
| `validation_harness.py` | VAL-01 | Baseline vs patched comparison | Complete |

---

## 4. Data Flow

```
data/sample_trade_logs.json
    │
    │  247 trades × n fields
    │  (entry time, exit time, direction, R-multiple, phase label)
    │
    ▼
experiments/validation_harness.py
    │
    │  Runs EXIT-A / EXIT-B / EXIT-C on identical trade universe
    │
    ▼
results/ablation_summary.md
results/exit_phase_analysis.md
```

---

## 5. Component Isolation Guarantee

The ablation design requires strict component isolation:

```
Core (SOAR1) ─── read-only during all experiments
Entry logic  ─── identical across all conditions
Market data  ─── identical tick stream, no look-ahead
Exit logic   ─── the ONLY variable
```

Any experiment that modifies more than one component simultaneously  
is invalid and is excluded from the results.

---

## 6. SOAR Organization (Reference)

```
SOAR
├── Research Division         ← This repository
│   ├── Phase Structure Lab
│   ├── Exit Mechanics Lab
│   └── Geometric Model Lab
│
├── Execution Division        ← Proprietary (not in this repo)
│   ├── EXIT-A (deprecated)
│   ├── EXIT-B (active)
│   └── EXIT-C (research only)
│
└── Core Division             ← Sealed
    └── SOAR1 (invariant)
```

---

## Next

→ `05_quick_reference.md` — Glossary and parameter reference  
→ `../experiments/validation_harness.py` — Run the full validation
