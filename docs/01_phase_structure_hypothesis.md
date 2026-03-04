# Phase Structure Hypothesis
## SOAR Exit Research — Document 01

---

## 1. Problem Statement

Standard algorithmic exit logic treats the question *"when to exit a trade"*  
as equivalent to the question *"what exit parameter produces the best backtest?"*

This framing is wrong.

It assumes market structure is stationary and that an optimal τ (hold duration)  
can be discovered by searching the parameter space. In practice, this produces:

- Overfit exit parameters that fail out-of-sample
- Premature exits during structural expansion phases
- Late exits after coherence has already collapsed

**The SOAR hypothesis:** exit timing is not a parameter. It is a phase observable.

---

## 2. Phase Lifecycle Model

Every trade passes through four structural phases regardless of direction or instrument:

```
PHASE 0 — FORMATION
  Duration:  ~0–200s from entry
  Character: Structure assembles. Coherence gradient positive.
  Signal:    No exit. Hold unconditionally.

PHASE 1 — EXPANSION
  Duration:  ~200–320s from entry
  Character: Structure expands. Peak coherence window.
  Signal:    No exit. Maximum alpha mass present.

PHASE 2 — SATURATION
  Duration:  ~320–350s from entry
  Character: Coherence plateaus. Marginal expansion slows.
  Signal:    Monitor. Prepare exit criteria.

PHASE 3 — COLLAPSE
  Duration:  ~350s+ from entry
  Character: Coherence gradient reverses. Structure dissolves.
  Signal:    Exit. Alpha mass is depleting.
```

Empirical validation: 2,473 exit events across 247 trades show  
**phase-clustered** exit distributions, not uniform temporal distributions.

---

## 3. Why Time Is a Proxy, Not a Signal

The phase durations listed above (~200s, ~280–320s, ~350s) are **population medians**,  
not parameters.

An individual trade may reach Phase 3 in 180s or in 500s.  
A time-based exit at τ = 300s will be:
- Premature in slow-expanding trades (exits during Phase 1)
- Late in fast-collapsing trades (exits after Phase 3 onset)

The correct observable is **coherence gradient sign change**, not elapsed time.

Time is a correlate of phase. Phase is the signal.

---

## 4. Coherence as a Scalar Field

Coherence C(t) is defined as the inner product of the current trajectory vector  
with the structural eigenvector of the formation phase:

```
C(t) = <v(t), v_0> / (|v(t)| · |v_0|)
```

Where:
- `v(t)` = price displacement vector at time t (n-dimensional feature space)
- `v_0`  = structural eigenvector at entry (formation-phase anchor)

**Coherence collapse** is defined as:

```
dC/dt < -κ   for at least Δt consecutive bars
```

Where κ is the collapse sensitivity threshold and Δt is the confirmation window.

---

## 5. Empirical Phase Distribution

Exit events from the 247-trade sample, binned by elapsed time from entry:

```
 0–100s  ██░░░░░░░░░░░░░░░░░░░░░░░░  8%   (structural failures, stopped out)
100–200s ██░░░░░░░░░░░░░░░░░░░░░░░░  6%   (early collapse, fast-moving)
200–280s ████████░░░░░░░░░░░░░░░░░░ 18%   (EXIT-A premature zone)
280–350s ██████████████████████░░░░ 51%   (Phase 2–3 transition cluster)
350–500s ██████░░░░░░░░░░░░░░░░░░░░ 14%   (late collapse, extended structure)
500s+    ███░░░░░░░░░░░░░░░░░░░░░░░  3%   (structural persistence outliers)
```

**The 280–350s cluster is not a coincidence.**  
It is the empirical signature of the Phase 2→3 transition.

EXIT-A (baseline) exits at median 6.3 bars ≈ 189s.  
It exits inside the EXIT-A premature zone for the majority of its trades.

EXIT-B (extended τ) exits at median 8.6 bars ≈ 258s.  
It reaches the Phase 2–3 transition cluster.

---

## 6. Implications

1. **Do not tune τ directly.** Model the coherence field. Exit on gradient reversal.
2. **Phase duration varies by regime.** Volatile regimes compress phases; trending regimes extend them.
3. **Core judgment is separable from exit mechanics.** Ablation confirms this.
4. **Structural persistence (τ_eff) is an output**, not an input.

---

## Next

→ `02_exit_ablation_study.md` — Full ablation methodology and results  
→ `03_mobius_manifold_model.md` — Geometric framework for coherence field
