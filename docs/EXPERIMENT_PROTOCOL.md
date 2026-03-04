# Patch Validation Experiment Protocol

**Goal:** Validate the phase-structure hypothesis through controlled patch experiments

**Timeline:** Run 1 day (baseline) → Apply patches iteratively → Compare metrics

---

## The 4 Key Metrics

### Metric 1: GRAMMAR_CUT Percentage
```
Definition: What % of all exits are GRAMMAR_CUT?

Interpretation:
  - Baseline expectation: ~18-20% (current)
  - With patches: Should DECREASE
  - Why: Orbit guard gates grammar before it fires
  
Target: < 10% by Patch A+B+C
```

### Metric 2: GRAMMAR_CUT >= 240s Percentage
```
Definition: Of GRAMMAR_CUT trades, what % happen at 240s+?

Interpretation:
  - Baseline: ~5-10% (grammar mostly fires early at 167s)
  - With patches: Should INCREASE sharply
  - Why: Grammar now waits for orbit formation
  
Target: > 50% by Patch A+B+C
```

### Metric 3: Sample Expansion in 240-420s Zone
```
Definition: Total trade count in 240-300s + 300-420s bins

Interpretation:
  - Baseline: Say 50 + 30 = 80 trades
  - With patches: Should grow
  - Why: Trades that would grammar-cut early now reach expansion zone
  
Target: > 120 total samples by Patch A+B+C
```

### Metric 4: Grammar Migration
```
Definition: MFE_SLOPE_CUT + MAX_HOLD count in 167-240s zone

Interpretation:
  - Baseline: Low (most are grammar cuts in this zone)
  - With patches: Should INCREASE
  - Why: Trades now exit via profit-taking (MFE) instead of grammar
  
Target: > 30 migrated by Patch A+B+C
```

---

## Experiment Procedure

### Step 1: Generate Baseline
```bash
# Current system, no patches
python run_live_orchestrator.py --file data_20260227.csv --batch > baseline_run.log
# Copy exit_events to: exit_events_BASELINE.jsonl
```

### Step 2: Apply Patch A (Orbit Guard)
```python
# In exit_core.py or similar:
orbit = (self.orch._last_orbit_mod or {}).get('orbit_state', '?')
if orbit == '?' and elapsed < 240:
    pos.grammar_checked = False
    return None  # Skip grammar exit
```

```bash
# Run with Patch A
python run_live_orchestrator.py --file data_20260227.csv --batch > patch_a_run.log
# Copy: exit_events_PATCH_A.jsonl
```

### Step 3: Apply Patch B (Close History)
```python
# In exit_core.py, replace:
# bar_window = [bars[j] for j in range(...)]  # 4 bars = 4 fps
# With:
tick_window = [{'close': c} for c in pos.close_history]  # 100+ fps
```

```bash
# Run with Patch A+B
python run_live_orchestrator.py --file data_20260227.csv --batch > patch_ab_run.log
# Copy: exit_events_PATCH_AB.jsonl
```

### Step 4: Apply Patch C (Min Retries)
```python
# In GrammarJudge initialization:
# self.judge = GrammarJudge(min_retries=6)
# Change to:
self.judge = GrammarJudge(min_retries=2)
```

```bash
# Run with Patch A+B+C
python run_live_orchestrator.py --file data_20260227.csv --batch > patch_abc_run.log
# Copy: exit_events_PATCH_ABC.jsonl
```

### Step 5: Compare Results
```bash
python VALIDATION_HARNESS.py \
  --baseline exit_events_BASELINE.jsonl \
  --patch-a exit_events_PATCH_A.jsonl \
  --patch-ab exit_events_PATCH_AB.jsonl \
  --patch-abc exit_events_PATCH_ABC.jsonl
```

---

## Expected Output

```
================================================================================
METRIC PROGRESSION TABLE
================================================================================

Stage                      Grammar %       Grammar 240+    Expansion       Migration
─────────────────────────────────────────────────────────────────────────────
BASELINE                    18.2%          8.5%            80              12
PATCH_A (Orbit Guard)       14.3%         22.1%           95              18
PATCH_A+B (+ Close History) 10.8%         38.7%          115              28
PATCH_A+B+C (+ Min Retries)  8.2%         52.1%          138              35

✓ HYPOTHESIS VALIDATED:
  ✓ Grammar % decreased (18.2% → 8.2%)
  ✓ Grammar 240+ increased (8.5% → 52.1%)
  ✓ Sample expansion grew (80 → 138)
  ✓ Grammar migration increased (12 → 35)
  
  All 4 metrics moved in expected direction
```

---

## Success Criteria

| Metric | Baseline | Target | Pass? |
|--------|----------|--------|-------|
| Grammar % | ~18% | < 10% | ✓ if decreases by 50%+ |
| Grammar 240+ | ~5% | > 40% | ✓ if increases 8x+ |
| Sample Expansion | ~80 | > 110 | ✓ if increases 40%+ |
| Grammar Migration | ~12 | > 25 | ✓ if doubles+ |

**Hypothesis confirmed if: 3 or 4 metrics hit targets**

---

## What Each Patch Does

### Patch A: Orbit Guard
```python
# Effect: Grammar can't fire if manifold not formed yet
if orbit == '?':
    return HOLD  # Don't grammar check
```

**Expected impact:**
- Reduces early grammar cuts
- Trades reach expansion zone more often
- Grammar 240+ percentage should jump sharply

### Patch B: Close History
```python
# Effect: Grammar sees 100+ tick samples instead of 4 bars
tick_window = pos.close_history  # All ticks, not bar sample
```

**Expected impact:**
- Grammar judge has better data
- Pattern detection more accurate
- Fewer "data insufficient" false exits
- Migration to MFE increases

### Patch C: Min Retries
```python
# Effect: Pattern requirements become physically achievable
min_retries = 2  # Was 6, now realistic for tick data
```

**Expected impact:**
- Grammar patterns found more often (with B's better data)
- But only when legitimate (physical constraints honored)
- Quality of grammar exits improves

---

## Interpretation Guide

### If Grammar % decreases but 240+ doesn't increase:
- Patch A is gating too aggressively
- Trades might be exiting via other paths (MFE, tail kill)
- Check: Are trades reaching expansion zone?

### If Sample Expansion doesn't grow:
- Patches may not be working together
- Check: Is orbit formation actually happening?
- Verify: coherence reaching ALIGNED state?

### If Grammar Migration is low:
- Trades still exiting via grammar despite patches
- Check: Patch A logic is actually blocking?
- Verify: Orbit guard condition is correct?

### If all 4 metrics move correctly:
- **HYPOTHESIS CONFIRMED**
- Phase structure is real
- System is phase-aligned (not time-based)
- Ready for production validation

---

## Next Steps After Validation

If hypothesis is confirmed:

1. **Measure actual coherence** (if not already logging)
   - Confirm /‾\_ shape (plateau + collapse)
   - Measure peak timing (280-320s?)

2. **Optimize coherence threshold**
   - Current shadow likely uses fixed threshold
   - Adaptive threshold based on market regime?

3. **A/B test on live data**
   - Small allocation with patches
   - Monitor vs baseline side-by-side

4. **Document geometry** (for future work)
   - Formation time by asset type
   - Plateau duration patterns
   - Collapse speed variations

---

## Timeline

```
Day 1: Generate baseline + Patch A
Day 1: Patch A+B, A+B+C
Day 1 evening: Run validation harness
Day 2: Analyze results, interpret metrics
Day 3: Implement any refinements if needed
Week 2: A/B test on live data
```

**Total time: 2-3 weeks to full validation**

---

## Hypothesis Statement

> "Market structure has distinct phases (formation, alignment, expansion, collapse).
> Current system exits too early via GRAMMAR_CUT during formation phase.
> Patching to gate grammar by orbit state will shift exits to proper phase boundaries,
> improving profitability and proving phase-based (not time-based) exit logic."

**If all 4 metrics confirm → Hypothesis is TRUE**

---

**Version:** 1.0  
**Created:** March 4, 2026  
**Status:** Ready for execution
