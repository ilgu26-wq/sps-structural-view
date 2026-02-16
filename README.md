# SPS — Structural Projection Surface

> This repository is not documentation.  
> It is a **live structural surface** for observing SOAR growth.

---

## ▶ View the Surface

- **Live Page (GitHub Pages)**  
> ⏳ If this shows 404, deployment may still be in progress.

🔗 https://ilgu26-wq.github.io/sps-structural-view/

---

- **Direct HTML (always works)**  
🔗 https://github.com/ilgu26-wq/sps-structural-view/blob/main/index.html

> Click **“View raw”** to open the exact HTML file.

---

## What This Repository Is

This repository exists to **project and display the structure of a growing SOAR system**.

- No execution
- No training
- No optimization

The HTML surface is authoritative.  
This repository only anchors and serves it.

---

## Reference Core vs Growing Core (Important)

To make growth **observable**, this surface includes:

- **SOAR1** — a sealed, invariant reference core  
- **Growing SOAR** — the actual evolving architecture under study

SOAR1 exists **only as a baseline coordinate**, not as the subject.

Growth is defined **relative to a fixed reference**.  
This surface provides that reference.

---

## Structural Evidence — Observation VETO Test

The following experiment was conducted to test whether
micro-structure / planetary observation,
when **lossily transformed into a VETO-only signal**,
can improve SOAR1 execution quality.

### Experiment
**EXP-VETO-MICRO-01**

- Core: **SOAR1 (fixed, sealed)**
- EXIT: **EXEC-B (fixed)**
- Variable: observation-based **BLOCK only** (no EXECUTE creation)
- Data: identical historical runs

### Verdict
**H0 RETAINED — Observation VETO is NOT beneficial for SOAR1**

### Results

| Arm | Trades | WR | RR | Expectancy | Equity |
|-----|--------|----|----|------------|--------|
| A: Baseline | 500 | 89.6% | 2.13 | +3.59 | **+1794R** |
| B: Soft-VETO | 361 | 90.6% | 2.27 | +3.90 | +1407R |
| C: Hard-VETO | 426 | 90.2% | 2.16 | +3.68 | +1568R |

### Analysis

- VETO blocked a large fraction of **winning trades**
- Among blocked trades:
  - Soft-VETO: only **13%** were actual losses
  - Hard-VETO: only **14%** were actual losses
- Net impact:
  - Soft-VETO: **−386R**
  - Hard-VETO: **−226R**

Observation-based blocking reduced total equity,
despite marginal improvements in WR / RR / Expectancy.

### Conclusion

> **Observation can act as a safety belt.  
> But SOAR1 already does not crash.  
> Adding a belt only slows it down.**

For SOAR1-level entry quality (~90% WR),
observation-based VETO removes more valid alpha
than it prevents losses.

### Artifacts

Evidence image:  

![EXP-VETO-MICRO-01 — Observation VETO Result](sps_veto_summary.png)