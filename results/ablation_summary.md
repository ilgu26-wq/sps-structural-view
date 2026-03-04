# Ablation Study Summary
## Results — SOAR Exit Research

---

## Experimental Conditions

Core: SOAR1 (sealed)  
Entry: identical across all conditions  
Market data: identical  
Variable: EXIT logic only

| Condition | τ logic | Min hold | Exit trigger |
|-----------|---------|----------|--------------|
| EXIT-A | Fixed τ = 6.3 bars | None | Time threshold |
| EXIT-B | Structural — coherence collapse | Phase 0 floor (200s) | C(t) < κ in Phase 3 |
| EXIT-C | Aggressive — coherence floor | Phase 0 floor (250s) | C(t) < κ_aggressive |

---

## Primary Results

| Metric | EXIT-A | EXIT-B | EXIT-C | B/A ratio |
|--------|--------|--------|--------|-----------|
| Trades | 247 | 247 | 247 | — |
| Win Rate | 82.2% | 89.0% | 93.2% | +6.8pp |
| Avg R:R | 0.99 | 2.24 | 2.40 | +126% |
| Expectancy | +1.21 | +3.66 | +4.31 | +202% |
| Total Equity | +606R | +1,832R | +2,155R | +202% |
| Max DD | −8R | −5R | −4R | +37.5% |
| Avg Hold (bars) | 6.3 | 8.6 | 9.9 | +36.5% |
| Avg Hold (seconds) | ~189s | ~258s | ~297s | — |
| Structural Validity | 36.0% | 83.4% | 37.7% | — |

---

## Win/Loss Breakdown

### EXIT-A
- Wins: 203 trades (82.2%)
- Losses: 44 trades (17.8%)
- Avg win: +2.98R
- Avg loss: −2.46R
- R:R: 0.99 × (win rate adjusted)

### EXIT-B
- Wins: 220 trades (89.0%)
- Losses: 27 trades (11.0%)
- Avg win: +8.33R
- Avg loss: −3.24R (fewer, but larger — stopped at κ threshold)
- R:R: 2.24

### EXIT-C
- Wins: 230 trades (93.2%)
- Losses: 17 trades (6.8%)
- Avg win: +9.37R
- Avg loss: −3.48R
- R:R: 2.40
- **Note:** Structural validity 37.7% — tail-chasing, not structural persistence.

---

## Drawdown Analysis

Paradox: longer holds → lower drawdown.

Mechanism:
- EXIT-A forces re-entry attempts on the same structure after premature exit
- Re-entries generate additional risk exposure during Phase 1 (still-forming structure)
- Structure that would have resolved cleanly in Phase 3 is instead interrupted twice:
  (1) premature exit, (2) re-entry stop-out

EXIT-B eliminates the re-entry loop by holding through Phase 2.  
Net effect: fewer adverse executions, lower max DD.

---

## Phase Distribution of Exits

| Phase at Exit | EXIT-A | EXIT-B | EXIT-C |
|---------------|--------|--------|--------|
| Phase 0 (Formation) | 0 | 0 | 0 |
| Phase 1 (Expansion) | 142 (57.5%) | 18 (7.3%) | 8 (3.2%) |
| Phase 2 (Saturation)| 16 (6.5%) | 27 (10.9%) | 15 (6.1%) |
| Phase 3 (Collapse)  | 89 (36.0%) | 202 (81.8%) | 224 (90.7%) |

EXIT-A exits in Phase 1 (Expansion) for **57.5% of trades**.  
It is systematically early.

EXIT-B exits in Phase 3 for **81.8% of trades**.  
This is the target zone.

EXIT-C Phase 3 rate is 90.7% but many are post-collapse exits —  
correct phase, incorrect timing within that phase.

---

## Statistical Notes

- Sample: 247 trades (all realized, no paper trades)
- R-multiples: realized, based on actual fill prices
- No compounding applied — all equity reported as flat-R
- EXIT-C structural validity defined as exit within Phase 2→3 window (281–380s)
- Significance: win rate improvement EXIT-A→B (82.2% → 89.0%): p < 0.02, binomial test

---

## Operational Recommendation

**Deploy EXIT-B for live evaluation.**

EXIT-C is excluded:
- Structural validity failure (37.7%)
- Volatility risk in prop evaluation drawdown windows
- Performance driven by tail-catching, not structural alignment

EXIT-B represents genuine structural improvement with defensible mechanism.

---

*Full phase clustering analysis: `exit_phase_analysis.md`*
