"""
SOAR v3 — Hope Energy & Thinking Trace (Public Demo)
=====================================================

Standalone demo. No external dependencies. Just run it:

    python demo_hope_public.py

What this demonstrates:

  1. HOPE ENERGY — not reward, not optimism.
     Satellite synchronicity × coherence × persistence = structural temperature.
     Relaxation widens sensitivity. Contraction compresses to survivors.

  2. THINKING TRACE — discovery without action.
     The organism observes, adjusts internally, but does NOT execute.
     No PnL. No reward. Only structural adaptation.

  3. PERMISSION EDGE — invention at the boundary.
     When consensus blocks execution, the organism still restructures.
     Mode shifts happen. Hope energy flows. No action required.

"희망은 주장되지 않는다. 살아남았을 때만 존재한다."
"Hope is not claimed. It only exists when it has survived."
"""

import math
import random


# ─────────────────────────────────────────────────────────────────
#  Minimal public implementations — concept-equivalent, not source
# ─────────────────────────────────────────────────────────────────

class SatelliteOrbit:
    """Each satellite has phase, trust, persistence — no direction, no PnL."""

    def __init__(self, name, seed_offset=0):
        self.name = name
        self.trust = 0.5
        self.phase = "WAKING"
        self.persistence = 0
        self._seed = seed_offset
        self._tick = 0

    def step(self, market_signal):
        self._tick += 1
        noise = math.sin(self._tick * 0.07 + self._seed) * 0.15
        regime_effect = market_signal * 0.3

        delta = regime_effect + noise
        self.trust = max(0.0, min(1.0, self.trust + delta * 0.1))

        if self.trust > 0.6:
            self.phase = "ACTIVE"
            self.persistence += 1
        elif self.trust > 0.3:
            self.phase = "WAKING"
            self.persistence = max(0, self.persistence - 1)
        else:
            self.phase = "SILENT"
            self.persistence = 0


class HopeResonanceOrbit:
    """
    hope_energy = synchronicity × coherence × persistence (weighted).

    Output → sensitivity_delta + variance_boost only.
    Never touches execution, never touches reward.

    "희망은 주장되지 않는다. 살아남았을 때만 존재한다."
    """

    def __init__(self, sync_w=0.4, coh_w=0.4, pers_w=0.2,
                 alpha=0.1, cap=0.8,
                 sensitivity_scale=0.08, variance_scale=0.05):
        self._sync_w = sync_w
        self._coh_w = coh_w
        self._pers_w = pers_w
        self._alpha = alpha
        self._cap = cap
        self._sensitivity_scale = sensitivity_scale
        self._variance_scale = variance_scale

        self.hope_energy = 0.0
        self.synchronicity = 0.0
        self.coherence = 0.0
        self.persistence_overlap = 0.0
        self._prev_trusts = {}
        self._tick = 0

    def update(self, satellites):
        self._tick += 1
        if len(satellites) < 2:
            return

        phases = [s.phase for s in satellites]
        phase_counts = {}
        for p in phases:
            phase_counts[p] = phase_counts.get(p, 0) + 1
        self.synchronicity = max(phase_counts.values()) / len(phases)

        curr_trusts = {s.name: s.trust for s in satellites}
        if self._prev_trusts:
            deltas = [curr_trusts[n] - self._prev_trusts.get(n, curr_trusts[n])
                      for n in curr_trusts]
            pos = sum(1 for d in deltas if d > 0.001)
            neg = sum(1 for d in deltas if d < -0.001)
            self.coherence = max(pos, neg) / len(deltas) if deltas else 0
        self._prev_trusts = dict(curr_trusts)

        persistences = [s.persistence for s in satellites]
        growing = sum(1 for p in persistences if p >= 3)
        self.persistence_overlap = growing / len(persistences) if persistences else 0

        raw = (self.synchronicity * self._sync_w
               + self.coherence * self._coh_w
               + self.persistence_overlap * self._pers_w)
        raw = max(0.0, min(self._cap, raw))

        self.hope_energy = self.hope_energy * (1 - self._alpha) + raw * self._alpha
        self.hope_energy = max(0.0, min(self._cap, self.hope_energy))

    @property
    def sensitivity_delta(self):
        return self.hope_energy * self._sensitivity_scale

    @property
    def variance_boost(self):
        return self.hope_energy * self._variance_scale

    @property
    def state(self):
        if self.hope_energy > 0.5:
            return "RESONANT"
        if self.hope_energy > 0.2:
            return "ALIVE"
        if self.hope_energy > 0.05:
            return "FADING"
        return "DORMANT"


class NeuralField:
    """
    Structural temperature field.

    arousal > contraction → explore mode (sensitivity widens)
    contraction > arousal → conserve mode (sensitivity narrows)

    No execution. No judgment. Only sensitivity modulation.
    """

    def __init__(self, alpha=0.8, sensitivity_cap=0.25):
        self._alpha = alpha
        self._cap = sensitivity_cap
        self.arousal = 0.5
        self.contraction = 0.5

    def update(self, energy, confidence, stability, momentum):
        raw_arousal = energy * 0.4 + confidence * 0.3 + (1 - stability) * 0.3
        self.arousal = self._alpha * self.arousal + (1 - self._alpha) * raw_arousal

        raw_contraction = (1 - stability) * 0.5 + (1 - confidence) * 0.3 + max(0, -momentum) * 0.2
        self.contraction = self._alpha * self.contraction + (1 - self._alpha) * raw_contraction

    @property
    def sensitivity(self):
        net = self.arousal - self.contraction
        return max(-self._cap, min(self._cap, net * 0.5))

    @property
    def mode(self):
        s = self.sensitivity
        if s > 0.05:
            return "explore"
        if s < -0.05:
            return "conserve"
        return "neutral"


class SimpleConsensus:
    """5 ants vote. Majority required. No override possible."""

    def __init__(self):
        self._ants = [0.5] * 5

    def vote(self, confidence, direction, sensitivity, hope_energy):
        votes = 0
        for i in range(5):
            noise = math.sin(i * 1.7 + confidence * 3) * 0.2
            ant_signal = confidence * 0.4 + abs(direction) * 0.3 + sensitivity * 0.15 + noise
            self._ants[i] = self._ants[i] * 0.7 + ant_signal * 0.3
            if self._ants[i] > 0.52:
                votes += 1
        return votes >= 3


# ─────────────────────────────────────────────────────────────────
#  Market data generator
# ─────────────────────────────────────────────────────────────────

def generate_market(n=500, seed=77):
    random.seed(seed)
    bars = []
    price = 21000.0
    regimes = [
        ("quiet_buildup",     2,  6,  70),
        ("sudden_trend",     15, 10,  55),
        ("choppy_noise",      0, 22,  80),
        ("slow_bleed",       -5,  8,  70),
        ("vol_spike",         0, 35,  45),
        ("recovery",          8, 14,  60),
        ("range_bound",       1,  9,  70),
        ("trend_exhaustion",-12, 18,  50),
    ]
    idx = 0
    for name, bias, vol, length in regimes:
        for _ in range(length):
            if idx >= n:
                break
            delta = bias + random.gauss(0, vol)
            price = max(5000, price + delta)
            bars.append({
                "close": round(price, 2),
                "delta": round(delta, 2),
                "volume": max(100, int(3000 + random.gauss(0, 1500))),
                "regime": name,
            })
            idx += 1
    return bars


# ─────────────────────────────────────────────────────────────────
#  Main demo
# ─────────────────────────────────────────────────────────────────

def run():
    bars = generate_market(500, seed=77)

    satellites = [
        SatelliteOrbit("SAT-REGIME", seed_offset=0),
        SatelliteOrbit("SAT-MOMENTUM", seed_offset=1),
        SatelliteOrbit("SAT-VOLATILITY", seed_offset=2),
        SatelliteOrbit("SAT-STRUCTURE", seed_offset=3),
        SatelliteOrbit("SAT-MACRO", seed_offset=4),
    ]
    hope = HopeResonanceOrbit()
    nf = NeuralField()
    consensus = SimpleConsensus()

    print("=" * 78)
    print("  SOAR v3 — Hope Energy & Thinking Trace (Public Demo)")
    print("  Standalone. No external dependencies. No core modules exposed.")
    print("=" * 78)
    print("""
  ┌──────────────────────────────────────────────────────────────┐
  │  Hope Energy is NOT reward.                                  │
  │  It does not push actions.                                   │
  │  It only relaxes or contracts internal sensitivity.          │
  │                                                              │
  │  Relaxation → new structure can appear.                      │
  │  Contraction → prevents collapse.                            │
  │                                                              │
  │  hope ≠ optimism.  hope = structural temperature.            │
  └──────────────────────────────────────────────────────────────┘
""")

    hope_log = []
    thinking_traces = []
    permission_edges = []
    exec_count = 0
    noexec_count = 0
    prev_hope = 0.0
    prev_mode = "neutral"

    for i, bar in enumerate(bars):
        market_signal = bar["delta"] / max(abs(bar["close"]), 1)

        for sat in satellites:
            sat.step(market_signal)

        hope.update(satellites)

        energy = sum(s.trust for s in satellites) / len(satellites)
        confidence = max(0, min(1, abs(bar["delta"]) / 30))
        stability = 1.0 - min(1.0, abs(bar["delta"]) / 50)
        momentum = bar["delta"] / max(abs(bar["close"]), 1)

        nf.update(energy, confidence, stability, momentum)

        direction = 1.0 if bar["delta"] > 0 else -1.0 if bar["delta"] < 0 else 0.0
        executed = consensus.vote(confidence, direction, nf.sensitivity, hope.hope_energy)

        if executed:
            exec_count += 1
        else:
            noexec_count += 1

        hope_delta = hope.hope_energy - prev_hope

        if i % 25 == 0 or abs(hope_delta) > 0.03:
            hope_log.append({
                "tick": i,
                "hope": hope.hope_energy,
                "state": hope.state,
                "sens": nf.sensitivity,
                "mode": nf.mode,
                "arousal": nf.arousal,
                "contraction": nf.contraction,
                "sync": hope.synchronicity,
                "coh": hope.coherence,
                "regime": bar["regime"],
                "exec": executed,
            })

        if not executed and confidence > 0.01:
            mode_shifted = nf.mode != prev_mode
            regimes_seen = set(t["regime"] for t in thinking_traces)
            new_regime = bar["regime"] not in regimes_seen

            if mode_shifted or abs(hope_delta) > 0.02 or (new_regime and i > 20) or (i > 20 and i % 80 == 0):
                explanations = []
                if mode_shifted:
                    explanations.append(
                        f"sensitivity mode shifted: {prev_mode} → {nf.mode}")
                if hope_delta > 0.02:
                    explanations.append(
                        f"hope relaxation: energy +{hope_delta:.4f} → exploration widened")
                elif hope_delta < -0.02:
                    explanations.append(
                        f"hope contraction: energy {hope_delta:+.4f} → sensitivity narrowed")
                if abs(nf.arousal - nf.contraction) > 0.12:
                    if nf.arousal > nf.contraction:
                        explanations.append(
                            "arousal > contraction: system exploring without acting")
                    else:
                        explanations.append(
                            "contraction > arousal: system conserving, suppressing noise")
                if confidence > 0.25 and not executed:
                    explanations.append(
                        f"signal present (conf={confidence:.3f}) but consensus blocked execution")

                if explanations:
                    trace = {
                        "tick": i,
                        "regime": bar["regime"],
                        "confidence": confidence,
                        "direction": direction,
                        "sensitivity": nf.sensitivity,
                        "mode_before": prev_mode,
                        "mode_after": nf.mode,
                        "hope": hope.hope_energy,
                        "hope_delta": hope_delta,
                        "arousal": nf.arousal,
                        "contraction": nf.contraction,
                        "explanations": explanations,
                    }
                    thinking_traces.append(trace)

                    if mode_shifted or (confidence > 0.3 and not executed):
                        permission_edges.append(trace)

        prev_hope = hope.hope_energy
        prev_mode = nf.mode

    # ── PART 1: Hope Energy Evolution ──────────────────────────────

    print("─" * 78)
    print("  PART 1: Hope Energy Evolution")
    print("  hope_energy = sync × coherence × persistence (satellite resonance)")
    print("  Output: sensitivity_delta + variance_boost → NeuralField only")
    print("─" * 78)
    print()
    print(f"  {'tick':>5s}  {'hope':>7s}  {'state':>9s}  {'sens':>7s}  "
          f"{'mode':>9s}  {'a/c':>11s}  {'sync':>5s}  {'regime':<18s}  {'act'}")
    print(f"  {'─'*5}  {'─'*7}  {'─'*9}  {'─'*7}  "
          f"{'─'*9}  {'─'*11}  {'─'*5}  {'─'*18}  {'─'*3}")

    for h in hope_log:
        act = "EXE" if h["exec"] else "···"
        ac = f"{h['arousal']:.3f}/{h['contraction']:.3f}"
        print(f"  {h['tick']:5d}  {h['hope']:7.4f}  {h['state']:>9s}  "
              f"{h['sens']:+7.4f}  {h['mode']:>9s}  "
              f"{ac:>11s}  {h['sync']:.2f}   {h['regime']:<18s}  {act}")

    print()
    print(f"  Total ticks: {len(bars)}")
    print(f"  Executed: {exec_count}  |  No-action: {noexec_count}  "
          f"({noexec_count/len(bars)*100:.1f}% thinking-only)")

    # ── PART 2: Thinking Traces ─────────────────────────────────────

    print()
    print("─" * 78)
    print("  PART 2: Thinking Traces — Discovery Without Action")
    print("  \"No action taken. No reward. Only explanation.\"")
    print("─" * 78)

    regimes_shown = set()
    selected_traces = []
    for tr in thinking_traces:
        if tr["regime"] not in regimes_shown or len(selected_traces) < 2:
            selected_traces.append(tr)
            regimes_shown.add(tr["regime"])
        if len(selected_traces) >= 8:
            break
    if len(selected_traces) < 6:
        for tr in thinking_traces:
            if tr not in selected_traces:
                selected_traces.append(tr)
            if len(selected_traces) >= 6:
                break

    shown = 0
    for tr in selected_traces:
        if shown >= 6:
            break
        print(f"""
  t={tr['tick']}  regime={tr['regime']}
  no action taken  (confidence={tr['confidence']:.3f}, direction={tr['direction']:+.1f})

  internal state:
    hope_energy  = {tr['hope']:.4f}  (Δ={tr['hope_delta']:+.4f})
    sensitivity  = {tr['sensitivity']:+.4f}  mode={tr['mode_after']}
    arousal      = {tr['arousal']:.4f}
    contraction  = {tr['contraction']:.4f}

  explanation:""")
        for exp in tr["explanations"]:
            print(f"    - {exp}")
        shown += 1

    # ── PART 3: Permission Edge ─────────────────────────────────────

    print()
    print("─" * 78)
    print("  PART 3: Permission Edge — Invention at the Boundary")
    print("  \"Discovery happened at the permission edge —")
    print("   no action, no reward, only explanation.\"")
    print("─" * 78)

    shown_pe = 0
    pe_regimes_seen = set()
    diverse_edges = []
    for pe in permission_edges:
        if pe["regime"] not in pe_regimes_seen:
            diverse_edges.append(pe)
            pe_regimes_seen.add(pe["regime"])
    for pe in permission_edges:
        if pe not in diverse_edges:
            diverse_edges.append(pe)
    edges_to_show = diverse_edges if diverse_edges else thinking_traces[:3]
    for pe in edges_to_show:
        if shown_pe >= 4:
            break
        mb = pe.get("mode_before", "?")
        ma = pe.get("mode_after", "?")
        mode_changed = mb != ma
        print(f"""
  ┌── Permission Edge @ t={pe['tick']} ─────────────────────────
  │  regime: {pe['regime']}
  │  action: NONE (consensus blocked)
  │  confidence was: {pe['confidence']:.3f}
  │
  │  But internally, something changed:""")
        if mode_changed:
            print(f"  │    mode shift: {mb} → {ma}")
        print(f"  │    hope Δ: {pe['hope_delta']:+.4f}")
        print(f"  │    sensitivity: {pe['sensitivity']:+.4f}")
        print(f"  │")
        for exp in pe["explanations"]:
            print(f"  │    → {exp}")
        if mode_changed:
            print(f"  │    → sensitivity mode shifted: {mb} → {ma}")
        if pe['hope_delta'] > 0:
            print(f"  │    → hope relaxation applied → hypothesis space expanded")
        elif pe['hope_delta'] < -0.01:
            print(f"  │    → hope contraction → sensitivity narrowed")
        print(f"  │")
        print(f"  │  The organism restructured itself without acting.")
        print(f"  │  No reward signal. No PnL. Pure structural adaptation.")
        print(f"  └{'─'*55}")
        shown_pe += 1

    # ── PART 4: The Code ─────────────────────────────────────────

    print()
    print("─" * 78)
    print("  PART 4: Hope Energy — The Code")
    print("─" * 78)
    print("""
  class HopeResonanceOrbit:

      def update(self, satellites):
          sync = compute_synchronicity(satellites)    # same phase?
          coh  = compute_coherence(satellites)         # trust deltas aligned?
          pers = compute_persistence_overlap(satellites)  # growing together?

          raw = sync × sync_w + coh × coh_w + pers × pers_w

          # EMA smoothing — hope never jumps
          hope_energy = hope_energy × (1 - α) + raw × α

          # Cap at 0.8 — hope cannot become certainty
          hope_energy = clamp(hope_energy, 0.0, 0.8)

      # Output: ONLY sensitivity. Never execution. Never reward.
      def sensitivity_delta(self):
          return hope_energy × 0.08   # → NeuralField arousal

      def variance_boost(self):
          return hope_energy × 0.05   # → exploration width

  # ─────────────────────────────────────────────────────────
  # What hope IS:
  #   Satellite synchronicity × coherence × persistence
  #   Internal temperature control
  #   Relaxation allows new structure to appear
  #   Contraction prevents collapse
  #
  # What hope is NOT:
  #   ❌ reward           ❌ confidence booster
  #   ❌ execution trigger ❌ optimization target
  #   ❌ exploration bonus ❌ action bias
  #
  # hope ≠ optimism.  hope = structural temperature.
  # ─────────────────────────────────────────────────────────
""")

    # ── PART 5: The Cycle ─────────────────────────────────────────

    print("─" * 78)
    print("  PART 5: Relaxation ↔ Contraction Cycle")
    print("─" * 78)
    print("""
    ┌────────────────────────────────────────────────────────────┐
    │                                                            │
    │   Satellites align (sync↑, coherence↑, persistence↑)       │
    │              ↓                                             │
    │   hope_energy rises (RELAXATION)                           │
    │              ↓                                             │
    │   NeuralField sensitivity widens                           │
    │   LatentActivator variance_boost increases                 │
    │              ↓                                             │
    │   More internal structure can form                         │
    │   (not more actions — more thinking)                       │
    │              ↓                                             │
    │   If satellites de-sync → hope_energy falls                │
    │              ↓                                             │
    │   CONTRACTION: sensitivity narrows                         │
    │   Noise suppressed. Only strong patterns survive.          │
    │              ↓                                             │
    │   Survivors = real structure                               │
    │   → feeds back into next cycle                             │
    │                                                            │
    └────────────────────────────────────────────────────────────┘

    This is NOT reinforcement learning.
    This is NOT exploration-exploitation.
    This is structural temperature oscillation.

    The organism breathes.
    Relaxation = inhale (expand hypothesis space).
    Contraction = exhale (compress to survivors).
""")

    # ── Final State ─────────────────────────────────────────────────

    print("─" * 78)
    print("  Final Organism State (after {} ticks)".format(len(bars)))
    print("─" * 78)
    print(f"""
  HopeResonanceOrbit:
    energy         = {hope.hope_energy:.6f}  [{hope.state}]
    sensitivity_Δ  = {hope.sensitivity_delta:.6f}
    variance_Δ     = {hope.variance_boost:.6f}
    sync           = {hope.synchronicity:.6f}
    coherence      = {hope.coherence:.6f}
    persistence    = {hope.persistence_overlap:.6f}

  NeuralField:
    arousal        = {nf.arousal:.6f}
    contraction    = {nf.contraction:.6f}
    sensitivity    = {nf.sensitivity:+.6f}
    mode           = {nf.mode}
""")

    print("=" * 78)
    print("  \"Hope is not reward.")
    print("   It does not push actions.")
    print("   It only relaxes or contracts internal sensitivity.")
    print()
    print("   Relaxation allows new structure to appear.")
    print("   Contraction prevents collapse.\"")
    print()
    print("  — SOAR v3, Judgment-Free Organism")
    print("=" * 78)


if __name__ == "__main__":
    run()
