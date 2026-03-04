# SOAR Analysis Package — Summary
## March 4, 2026

---

## Contents

| File | Size | Description |
|------|------|-------------|
| `00_START_HERE.txt` | — | Navigation guide, read first |
| `PHASE_STRUCTURE_NOT_TIME.md` | — | Core argument: phase structure vs time parameter |
| `EXPERIMENT_PROTOCOL.md` | — | Patch procedure, success criteria, timeline |
| `VALIDATION_HARNESS.py` | — | Baseline vs patch comparison tool |
| `final_verdict.md` | — | 247-trade exit signal and duration analysis |
| `YOUR_HYPOTHESIS_CONFIRMED.md` | — | Initial 20-trade observation |
| `SOAR_SYSTEM_ANALYSIS.md` | 23 KB | Full system architecture |
| `MOBIUS_MANIFOLD_DEEP_DIVE.md` | 20 KB | Geometric model and math |
| `COHERENCE_PEAK_VALIDATION_REPORT.md` | — | Coherence curve analysis |
| `QUICK_REFERENCE_GUIDE.md` | 13 KB | Parameter tables, patch summaries |
| `ARCHITECTURE_DIAGRAMS.md` | 52 KB | Component diagrams, state machines |
| `README.md` | — | Document index |

---

## Core findings

**Exit signal timing across 2,473 events:**

| Signal | Avg elapsed | Role |
|--------|-------------|------|
| GRAMMAR_CUT | ~167s | Fires during formation — too early |
| MFE_SLOPE_CUT | ~244s | Fires at favorable moment |
| SHADOW_LINE_CUT | ~343s | Fires at coherence collapse — structural exit |

Gap between GRAMMAR and SHADOW: ~175s. This gap spans the predicted coherence peak zone.

**Coherence curve shape (observed):** rises 0–200s, plateaus 200–320s, collapses 320s+.

**Current system status:**

| Component | Status |
|-----------|--------|
| Entry system | Working |
| Orbit tracking | Working |
| SHADOW exit | Working |
| Grammar filter | Needs adjustment — fires as exit, should be filter |

**Three proposed patches:**

| Patch | File | Change |
|-------|------|--------|
| A — Orbit Guard | `exit_core.py` | Skip grammar check if orbit unknown and elapsed < 240s |
| B — Close History | `grammar_judge.py` | Use tick-level close history instead of 4-bar sample |
| C — Min Retries | `grammar_judge.py` | Reduce min_retries from 6 to 2–3 |

---

## Suggested reading paths

**For understanding the architecture (~45 min):**
1. `00_START_HERE.txt`
2. `PHASE_STRUCTURE_NOT_TIME.md`
3. `SOAR_SYSTEM_ANALYSIS.md` sections 1–3
4. `ARCHITECTURE_DIAGRAMS.md` for reference

**For implementation (~30 min + run time):**
1. `00_START_HERE.txt`
2. `EXPERIMENT_PROTOCOL.md`
3. `VALIDATION_HARNESS.py`
4. `QUICK_REFERENCE_GUIDE.md` for reference

**For deeper analysis (3+ hours):**
All of the above, plus `MOBIUS_MANIFOLD_DEEP_DIVE.md` and `COHERENCE_PEAK_VALIDATION_REPORT.md`.

---

## Implementation steps

**Step 1 — Apply patches (~30 min)**

```python
# Patch A: exit_core.py
if orbit == '?' and elapsed < 240:
    return HOLD

# Patch B: grammar_judge.py
tick_window = pos.close_history  # was: bar_window from 4 bars

# Patch C: grammar_judge.py
min_retries = 2  # was: 6
```

**Step 2 — Generate test runs (~1–2 hours)**

```bash
python run_live_orchestrator.py --file data.csv --batch > baseline.log
cp exit_events baseline_run.jsonl
# apply patches one by one, re-run each time
```

**Step 3 — Validate (~10 min)**

```bash
python VALIDATION_HARNESS.py \
  --baseline baseline_run.jsonl \
  --patch-a patch_a_run.jsonl \
  --patch-ab patch_ab_run.jsonl \
  --patch-abc patch_abc_run.jsonl
```

Target metrics (3 of 4 confirms hypothesis):

| Metric | Baseline | Target |
|--------|----------|--------|
| Grammar exit % | 18% | < 8% |
| Grammar exits at 240s+ | 8% | > 50% |
| Sample size | 80 | > 140 |
| Grammar migrations | 12 | > 35 |

---

## Evidence base

- 2,473 exit event records (Feb 20–21, 2026)
- 247 completed trades (Feb 20–25, 2026)
- GRAMMAR vs SHADOW timing gap: t-statistic > 12
- Duration clustering at predicted zone: 1.52× uniform expectation
- Current confidence estimate: ~75–95% depending on metric

---

## Notes

The core observation — grammar fires too early, shadow fires at the structural
exit point, and the gap between them spans the predicted coherence peak — is
consistent across both the 20-trade and 247-trade samples.

The patches address component ordering, not the core algorithm.
Whether the improvements hold across different market regimes is the
open question the validation harness is designed to answer.

---

*Analysis period: Feb 16–25, 2026*  
*Data: 2,473 events, 247 trades*
