# Exit Ablation Study
## SOAR Exit Research — Document 02

---

## 1. Motivation

After observing consistent underperformance in the execution layer,  
an ablation was designed to isolate whether the performance limit originated from:

- **(A) Core judgment** — alpha identification, entry selection, directional bias, or
- **(B) Exit mechanics** — structural persistence, hold duration, exit trigger logic

The hypothesis was that core judgment was sound and exit mechanics were the bottleneck.

---

## 2. Experimental Design

**Controls (held invariant across all conditions):**

| Component | Value |
|-----------|-------|
| Core model | SOAR1 (sealed, no modification) |
| Entry logic | Identical |
| Market data | Identical tick stream |
| Trade universe | 247 trades, same sequence |
| Position sizing | Fixed 1R per trade |

**Variable:** EXIT logic only.

Three EXIT conditions tested:

| Condition | Description |
|-----------|-------------|
| EXIT-A | Baseline — original exit logic (τ = 6.3 bars median) |
| EXIT-B | Extended τ — coherence-respecting hold (τ = 8.6 bars median) |
| EXIT-C | Aggressive — maximum structural persistence (τ = 9.9 bars median) |

---

## 3. Results

### 3.1 Performance Summary

| Condition | Win Rate | Avg RR | Expectancy | Total Equity | Max DD |
|-----------|----------|--------|------------|--------------|--------|
| EXIT-A (Baseline)   | 82.2% | 0.99 | +1.21 | +606R  | −8R |
| EXIT-B (Extended τ) | 89.0% | 2.24 | +3.66 | +1832R | −5R |
| EXIT-C (Aggressive) | 93.2% | 2.40 | +4.31 | +2155R | −4R |

### 3.2 Structural Hold Duration

| Condition | Avg τ (bars) | Avg τ (seconds) |
|-----------|-------------|-----------------|
| EXIT-A | 6.3 | ~189s |
| EXIT-B | 8.6 | ~258s |
| EXIT-C | 9.9 | ~297s |

---

## 4. Observations

### 4.1 Core Judgment Remained Valid Throughout

Win rate exceeded 80% in all conditions without any modification to the entry model.  
This confirms that directional alpha identification was not the limiting factor.

### 4.2 Expectancy Scaled With Structural Persistence

EXIT-A expectancy: +1.21  
EXIT-B expectancy: +3.66  
→ **3.0× improvement** from τ extension alone, with no core modification.

The mechanism: premature exits (EXIT-A) truncated valid alpha mass  
while the underlying structure remained intact and expanding.

### 4.3 Drawdown Improved With Extended Hold

Counterintuitively, holding longer **reduced** drawdown:

```
EXIT-A Max DD: −8R
EXIT-B Max DD: −5R
EXIT-C Max DD: −4R
```

Explanation: early exits forced re-entry attempts on the same structure,  
generating additional transaction costs and re-entry slippage.  
Structural exits eliminate the re-entry problem entirely.

### 4.4 EXIT-C Is Not Prop-Stable

EXIT-C achieves highest raw metrics but fails the prop-stability test:

```
EXIT-C structural validity rate: 37.5%
```

At 37.5% structural validity, EXIT-C is exploiting noise tails rather  
than genuine structural persistence. Not suitable for live evaluation regimes.

**EXIT-B is the operationally valid finding.**

---

## 5. Verdict

> **The core did not improve.  
> The execution finally stopped interfering.**

The ablation confirms two independent results:

1. **Performance limits were execution-induced**, not core-induced.
2. **Structural persistence (τ) is not a free parameter** to be tuned —  
   it is a structural observable that should respect phase lifecycle.

---

## 6. Methodology Notes

- No learning, no optimization, no structural modification were introduced at any stage
- All conditions ran on identical market data with identical entry signals
- EXIT logic was the only differing variable
- Results reported as realized R-multiples (no hypothetical compounding applied)
- Structural validity defined as: exit occurring within Phase 2–3 transition window (280–380s)

---

## Next

→ `03_mobius_manifold_model.md` — Geometric framework explaining why τ-extension works  
→ `../results/ablation_summary.md` — Extended results table with per-trade breakdown
