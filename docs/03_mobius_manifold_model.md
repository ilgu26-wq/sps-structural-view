# Möbius Manifold Model
## SOAR Exit Research — Document 03

---

## 1. Why Geometry

The phase lifecycle model (`01_phase_structure_hypothesis.md`) identifies four phases  
and defines coherence as a scalar field over the feature space.

The question this document addresses: **what is the geometric structure of that field?**

The answer shapes how we model, detect, and respond to coherence collapse.

---

## 2. Manifold Intuition

### 2.1 Standard Assumption (Euclidean)

Most price-action models assume feature space is locally Euclidean:  
structure is additive, signals are separable, trajectories are straight.

Under this assumption, exit timing is a threshold problem:  
*"exit when price moves X away from entry."*

### 2.2 SOAR Assumption (Möbius)

The SOAR model treats price trajectory as living on a **Möbius-like manifold** —  
a non-orientable surface with a single continuous edge.

Key geometric property: **orientation reverses over one full traversal.**

Practical implication:  
What looks like "trend continuation" on one face of the manifold  
is "mean reversion" on the other face — same trajectory, opposite structural meaning.

Exit signals must account for which face the trajectory is currently on.

---

## 3. Formal Construction

### 3.1 State Space

Define the n-dimensional state vector at time t:

```
s(t) = [p(t), Δp(t), σ(t), v(t), θ(t)]
```

Where:
- `p(t)` = normalized price
- `Δp(t)` = first-order displacement
- `σ(t)` = local volatility estimate
- `v(t)` = volume-weighted momentum
- `θ(t)` = rotational phase angle (see `exp_rot_phase_01.py`)

### 3.2 Möbius Embedding

The trajectory `{s(t)}` is embedded on a Möbius band M defined by:

```
M = [0,1] × [-1,1] / ~
where (0, x) ~ (1, -x)
```

Phase position is parameterized by (u, w) on M:
- `u` = progression through the phase lifecycle [0, 1]
- `w` = coherence signed amplitude [-1, 1]

### 3.3 Coherence Field

Coherence C on M is defined as the dot product of the current trajectory  
tangent vector with the formation-phase anchor vector:

```
C(u, w) = <T(u,w), T_anchor> / (|T| · |T_anchor|)
```

**Coherence collapse** = the point where ∂C/∂u < 0 and |∂C/∂u| > κ  
for a sustained window Δu.

### 3.4 Fold Detection

The manifold fold (`exp_mobius_fold_01.py`) is the geometric locus of points  
where surface orientation transitions — the seam of the Möbius band.

Empirically, fold events correspond to Phase 2→3 transitions:
- Pre-fold: coherence gradient positive (expansion)
- Fold crossing: gradient sign reversal
- Post-fold: coherence degrading (collapse onset)

**Exit signal fires at fold crossing, not at a fixed time.**

---

## 4. Rotational Phase

The rotational phase angle θ(t) (`exp_rot_phase_01.py`) measures  
angular displacement of the trajectory in the (Δp, v) plane:

```
θ(t) = arctan2(v(t), Δp(t))
```

Phase lifecycle maps to θ as follows:

| Phase | θ range | Description |
|-------|---------|-------------|
| Formation | 0° – 45° | Initial momentum alignment |
| Expansion | 45° – 135° | Full angular sweep |
| Saturation | 135° – 160° | Angular deceleration |
| Collapse | 160°+ or reversal | Momentum breakdown |

The topology lock (`exp_topology_lock_01.py`) fires when θ progression stalls  
below the saturation threshold — early warning of Phase 3 onset.

---

## 5. Shadow Line Exit

The shadow line (`exp_shadow_line_exit.py`) is a geometric construct derived  
from the trailing boundary of the Möbius band as the trajectory progresses.

It functions as a **dynamic structural support** — not a fixed price level,  
but a surface-relative boundary that moves with the trajectory.

Exit fires when:
1. Trajectory crosses below the shadow line boundary, **and**
2. Coherence C(t) < κ_shadow (shadow coherence threshold)

This prevents shadow line exits during temporary retracements within Phase 1–2.

---

## 6. Geometric Alignment Signal (g_align)

The g_align signal (`exp_g_align_01.py`, `data/g_align_v2.csv`) measures  
multi-dimensional structural alignment across the feature space.

High g_align = multiple feature dimensions aligned in coherent direction → hold  
Low g_align = dimensional decoherence → monitor for collapse

g_align_v2 incorporates rotational correction for regime-dependent compression.

---

## 7. Summary

The Möbius manifold model provides:

1. A **geometric reason** why phase transitions occur (fold crossing)
2. A **continuous observable** (coherence C) rather than a discrete threshold
3. A **directional signal** (fold detection) rather than a time-based rule
4. **Regime adaptability** — manifold curvature adjusts to local volatility

The key insight: exit timing is not "when enough time has passed"  
but "when the trajectory crosses the manifold fold."

---

## Next

→ `04_architecture_overview.md` — System-level component structure  
→ `../experiments/exp_mobius_fold_01.py` — Fold detection implementation  
→ `../experiments/exp_rot_phase_01.py` — Rotational phase implementation
