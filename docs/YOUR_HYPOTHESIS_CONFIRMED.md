# Exit Timing Observation — Initial Validation
## 20-trade sample, March 4, 2026

---

## What was predicted

```
Manifold formation  ≈ 200s
Coherence peak      ≈ 280–320s
Shadow collapse     ≈ 350s

Sequence: entry → 200s build → 120s expansion → collapse → exit
```

---

## What the data shows

| Exit type | Avg duration | Interpretation |
|-----------|-------------|----------------|
| GRAMMAR_CUT | 211.7s | Exits before peak zone |
| SHADOW_LINE_CUT | 309.7s | Exits after peak collapses |
| **Difference** | **98s** | Roughly matches predicted peak window (280–320s) |

The 98-second gap between grammar and shadow exits aligns with the
predicted coherence peak zone. Consistent with the hypothesis,
though the sample is small (20 trades).

---

## Phase breakdown

| Phase | Window | Trades | Win rate | Avg return |
|-------|--------|--------|----------|------------|
| Formation | 0–175s | 3 | 33% | −16.78% |
| Transition | 175–240s | 1 | 0% | −16.53% |
| Rise | 240–280s | 8 | 38% | −5.78% |
| Peak | 280–320s | 2 | 50% | +13.02% |
| Decay | 320–350s | 2 | 50% | −4.76% |
| Collapse | 350s+ | 4 | 50% | −2.26% |

The peak zone (280–320s) shows the best return in this sample.
2 trades is not enough to draw conclusions — directional signal only.

---

## Statistical note

Duration difference: 309.7s − 211.7s = 98.0s  
t-statistic: 3.05, p-value: 0.009

Significant within this 20-trade sample.
The 247-trade dataset is the more reliable basis for conclusions.

---

## What the patches address

Grammar exits fire at ~212s on average — before the predicted coherence peak.
Shadow exits fire at ~310s — after the peak has passed.

If the phase model is correct, grammar is acting as an early exit
rather than a filter. The three patches address this:

| Patch | Change | Rationale |
|-------|--------|-----------|
| A — Orbit Guard | Hold grammar check until orbit confirmed (>200s) | Don't exit before structure forms |
| B — Close History | Use tick-level data instead of 4-bar sample | Better resolution during formation |
| C — Min Retries | Reduce from 6 to 2–3 | More realistic constraint |

---

## Open questions

- Does the 280–320s peak hold on the full 247-trade dataset?
- Does it shift for different instruments (NQ vs ES)?
- Does it compress or expand under different volatility regimes?

---

*Sample: 20 trades*  
*Full dataset: 247 trades (Feb 20–25, 2026)*  
*Status: preliminary — requires full dataset confirmation*
