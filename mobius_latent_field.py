# -*- coding: utf-8 -*-
"""
mobius_latent_field.py — Möbius Latent Manifold Engine (v3)

핵심 업그레이드 (v2 → v3):
  1. TV encoder  — phase-weighted total variation (위상 질서 붕괴 측정)
  2. Sparse memory — latent jump > eps일 때만 저장 (noise 제거)
  3. Phase-weight matrix W — P0/P1/P2/P3별 다른 6×6 가중 행렬
  4. wave_energy 기반 μ 재정의 (직접 곱셈 → 물리 의미 강화)
  5. Windmill 흡수 (wind_rotation, wind_persistence, collapse_spin)
  6. Entry impulse feed (entry 순간도 buffer에 주입)

설계 철학:
  runner   = 데이터 수집 (원시값)
  orch     = 물리량 번역 (OrchestratorPhysics)
  Möbius   = 관계 압축 (latent)
  Brain    = adaptive gate (판단)

TV 인코더 원리:
  TV_phase = Σ w_t * |x_t - x_{t-1}|
  w_t = phase_weight × energy_weight
  → "얼마나 움직였나"가 아니라 "위상 질서가 얼마나 무너졌나"

Sparse memory:
  if abs(latent_jump) > SPARSE_EPS: store frame
  → 대표점만 저장, 전부 저장 금지

W matrix:
  score = μᵀ @ W_phase @ μ
  각 phase별 다른 가중 행렬 (P0: curvature보호, P2: fold압력, P3: drift강조)

μ 재정의 (wave_energy 기반):
  μ1 = impulse * slope          (micro: 순간 힘)
  μ2 = shadow * torsion         (meso: 구조 비틀림)
  μ3 = pain * collapse_pressure (macro: 기억 붕괴)
  μ4 = coherence * fold         (구조 접힘)
  μ5 = rotation * persistence   (windmill 지속성)
  μ6 = tesla_amp * wave_energy  (Tesla 공진 에너지)
"""

import math
from collections import deque
from typing import NamedTuple, List, Optional, Dict, Tuple

# ── 상수 ──────────────────────────────────────────────────────────────────────

BUFFER_SIZE        = 64
BAND_LOW           = 16
BAND_MID           = 8
BAND_HIGH          = 4

BASE_TAU           = 420.0
TESLA_K            = 0.6
CHANNEL_TAU        = {0: 900.0, 1: 420.0, 2: 180.0}
CHANNEL_HYSTERESIS = 3

SPARSE_EPS         = 0.04   # latent jump threshold for sparse store
SPARSE_MAX         = 32     # sparse memory max size

LATENT_MAX_REDUCTION = 0.22

# TV phase weights (P0~P3)
TV_PHASE_W = {0: 0.4, 1: 1.0, 2: 1.4, 3: 1.6}

# ── Phase-weight W matrix (6×6, symmetric) ───────────────────────────────────
# score = μᵀ @ W @ μ
# P0: 진입 보호 — μ1 curvature 낮게, μ4 coherence 강조
# P1: 압축 — μ2 torsion 상승
# P2: collapse domain — μ3 fold, μ6 resonance 최강
# P3: force exit — μ5 drift, μ3 강제
def _make_W(diag: list, cross: dict = None) -> list:
    """6×6 대칭 행렬 생성. diag=대각, cross={(i,j):v} 비대각."""
    W = [[0.0]*6 for _ in range(6)]
    for i, v in enumerate(diag):
        W[i][i] = v
    if cross:
        for (i, j), v in cross.items():
            W[i][j] = v
            W[j][i] = v
    return W

W_MATRIX = {
    0: _make_W([0.10, 0.08, 0.12, 0.20, 0.08, 0.10],  # P0: coherence/fold 보호
               {(0,3): 0.05, (3,5): 0.04}),
    1: _make_W([0.15, 0.18, 0.15, 0.15, 0.10, 0.12],  # P1: torsion/curvature
               {(0,1): 0.07, (1,2): 0.06}),
    2: _make_W([0.18, 0.12, 0.28, 0.10, 0.12, 0.25],  # P2: fold+resonance DOMINANT
               {(2,5): 0.12, (0,2): 0.08, (4,5): 0.06}),
    3: _make_W([0.12, 0.10, 0.30, 0.08, 0.22, 0.28],  # P3: drift+resonance FORCE
               {(2,4): 0.10, (4,5): 0.10, (2,5): 0.08}),
}

# 12D frame 슬롯
FRAME_DIM = 12
F_SHADOW, F_GRAMMAR, F_SLOPE, F_FOLD, F_PAIN  = 0, 1, 2, 3, 4
F_COH,    F_CUR_R,   F_VEL,   F_DELTA         = 5, 6, 7, 8
F_TORQUE, F_MFE,     F_MAE                    = 9, 10, 11

MICRO_IDX = [F_SLOPE, F_VEL, F_DELTA, F_TORQUE]
MESO_IDX  = [F_SHADOW, F_GRAMMAR, F_FOLD, F_COH]
MACRO_IDX = [F_PAIN, F_MFE, F_MAE]


# ── OrchestratorPhysics ───────────────────────────────────────────────────────

class OrchestratorPhysics:
    """
    Runner raw → 5개 공통 물리량 + windmill.
    Entry/Exit 공통 좌표계.
    """

    @staticmethod
    def translate(packet: dict, elapsed: float = 0.0) -> dict:
        shadow    = float(packet.get("shadow_signal",  0.0))
        grammar   = float(packet.get("grammar_signal", 0.0))
        fold      = float(packet.get("fold",           0.0))
        pain      = float(packet.get("pain",           0.0))
        coherence = float(packet.get("coherence",      0.5))
        velocity  = float(packet.get("velocity",       0.0))
        delta_raw = float(packet.get("delta",          0.0))
        mfe       = float(packet.get("mfe",            0.0))
        mae       = float(packet.get("mae",            0.0))
        pinn_res  = float(packet.get("pinn_residual",  0.0))
        tesla_amp = float(packet.get("tesla_amp",      0.0))
        tesla_wave= float(packet.get("tesla_wave",     0.0))
        dopamine  = float(packet.get("dopamine",       0.5))
        slope     = float(packet.get("slope_signal",   0.0))

        # windmill
        wind_rot  = float(packet.get("wind_rotation",  0.0))
        wind_per  = float(packet.get("wind_persistence",0.0))
        coll_spin = float(packet.get("collapse_spin",  0.0))

        impulse     = round(min(1.0, abs(velocity) * max(abs(delta_raw), 0.01)), 4)
        pressure    = round(min(1.0, (shadow + grammar + fold) / 3.0), 4)
        elapsed_f   = round(min(1.0, elapsed / max(BASE_TAU, 1.0)), 4)
        inertia     = round(min(1.0, coherence * (0.5 + 0.5 * elapsed_f)), 4)
        mfe_decay   = round(min(1.0, max(0.0, mfe - mae + 0.5)), 4) if mfe > 0 else 0.0
        dissipation = round(min(1.0, (pain + mfe_decay + pinn_res * 0.5) / 3.0), 4)
        resonance   = round(min(1.0, tesla_wave * max(tesla_amp, dopamine * 0.3)), 4)
        wave_energy = round(impulse * resonance + pressure * inertia - dissipation, 4)

        # torsion: rotation(pressure, inertia)
        torsion = round(min(1.0, abs(pressure - inertia)), 4)

        # collapse_pressure proxy
        col_press = round(min(1.0, dissipation / max(impulse + 0.01, 0.01)), 4)

        return {
            "impulse":       impulse,
            "pressure":      pressure,
            "inertia":       inertia,
            "dissipation":   dissipation,
            "resonance":     resonance,
            "wave_energy":   wave_energy,
            "torsion":       torsion,
            "col_press":     col_press,
            "slope":         slope,
            "shadow":        shadow,
            "coherence":     coherence,
            "fold":          fold,
            "pain":          pain,
            "tesla_amp":     tesla_amp,
            "wind_rotation": wind_rot,
            "wind_persist":  wind_per,
            "collapse_spin": coll_spin,
            "elapsed":       elapsed,
        }


# ── TV encoder ────────────────────────────────────────────────────────────────

class TVEncoder:
    """
    Phase-weighted Total Variation encoder.

    TV_phase = Σ w_t * |x_t - x_{t-1}|
    w_t = TV_PHASE_W[phase] × energy_weight

    출력: signed_tv (방향성 포함), unsigned_tv (크기만)
    """

    def __init__(self, window: int = 7):
        self._window = window
        self._buf: deque = deque(maxlen=window)

    def push(self, value: float) -> None:
        self._buf.append(float(value))

    def compute(self, phase: int = 2, energy: float = 1.0) -> Tuple[float, float]:
        """
        Returns (signed_tv, unsigned_tv).
        signed_tv = Σ sign(Δx)*|Δx|*w  — 방향 붕괴 포함
        """
        buf = list(self._buf)
        if len(buf) < 2:
            return 0.0, 0.0
        pw   = TV_PHASE_W.get(phase, 1.0)
        ew   = max(0.1, min(2.0, energy))
        w    = pw * ew
        diffs = [buf[i] - buf[i-1] for i in range(1, len(buf))]
        unsigned_tv = round(w * sum(abs(d) for d in diffs) / max(len(diffs), 1), 6)
        signed_tv   = round(w * sum(d for d in diffs) / max(len(diffs), 1), 6)
        return signed_tv, unsigned_tv


# ── Sparse memory ─────────────────────────────────────────────────────────────

class SparseLatentMemory:
    """
    latent jump > SPARSE_EPS일 때만 저장.
    "모든 tick 동일 importance 금지"
    """

    def __init__(self, maxlen: int = SPARSE_MAX):
        self._buf: list = []
        self._maxlen    = maxlen
        self._prev_vec: Optional[List[float]] = None

    def try_push(self, vec: List[float]) -> bool:
        """Returns True if stored (significant jump)."""
        if self._prev_vec is None:
            self._buf.append(vec[:])
            self._prev_vec = vec[:]
            return True
        jump = math.sqrt(sum((a-b)**2 for a,b in zip(vec, self._prev_vec)))
        if jump > SPARSE_EPS:
            if len(self._buf) >= self._maxlen:
                self._buf.pop(0)
            self._buf.append(vec[:])
            self._prev_vec = vec[:]
            return True
        return False

    def get_recent(self, n: int = 8) -> List[List[float]]:
        return self._buf[-n:] if self._buf else []

    def __len__(self):
        return len(self._buf)


# ── MobiusState ───────────────────────────────────────────────────────────────

class MobiusState(NamedTuple):
    """6D Möbius latent + 3-layer 압축."""
    micro:  float
    meso:   float
    macro:  float

    # wave_energy 기반 μ (v3)
    mu1_impulse_slope:  float   # impulse × slope          (순간 힘)
    mu2_shadow_torsion: float   # shadow × torsion         (구조 비틀림)
    mu3_pain_collapse:  float   # pain × collapse_pressure (기억 붕괴)
    mu4_coherence_fold: float   # coherence × fold         (구조 접힘)
    mu5_wind_persist:   float   # rotation × persistence   (windmill 지속)
    mu6_tesla_energy:   float   # tesla_amp × wave_energy  (Tesla 공진)

    def to_vec(self) -> List[float]:
        return [self.mu1_impulse_slope,  self.mu2_shadow_torsion,
                self.mu3_pain_collapse,  self.mu4_coherence_fold,
                self.mu5_wind_persist,   self.mu6_tesla_energy]

    def to_brain_input(self) -> Dict[str, float]:
        return {
            "mobius_mu1":   self.mu1_impulse_slope,
            "mobius_mu2":   self.mu2_shadow_torsion,
            "mobius_mu3":   self.mu3_pain_collapse,
            "mobius_mu4":   self.mu4_coherence_fold,
            "mobius_mu5":   self.mu5_wind_persist,
            "mobius_mu6":   self.mu6_tesla_energy,
            "mobius_micro": self.micro,
            "mobius_meso":  self.meso,
            "mobius_macro": self.macro,
        }

    def score(self, phase: int = 2) -> float:
        """
        score = μᵀ @ W_phase @ μ  (v4: μ5 부호 처리)
        μ5는 [-1,1] 부호 있음 → quadratic form에 절대값 사용
        양 drift bonus는 별도 가산.
        """
        W = W_MATRIX.get(phase, W_MATRIX[2])
        v = list(self.to_vec())
        # μ5 index=4: 부호 있음 → quadratic에서 abs 사용
        v_abs = [abs(x) for x in v]
        s = 0.0
        for i in range(6):
            for j in range(6):
                s += v_abs[i] * W[i][j] * v_abs[j]
        # drift direction bonus: 음 drift는 score 올림 (danger), 양 drift는 낮춤 (safe)
        drift_penalty = -v[4] * 0.04   # μ5<0 → penalty 양수 → score 올라감
        return round(min(2.0, max(0.0, s + drift_penalty)), 4)

    def collapse_pressure(self) -> float:
        return round(min(1.0,
            0.35 * self.mu3_pain_collapse +
            0.35 * self.mu4_coherence_fold +
            0.30 * self.mu6_tesla_energy), 4)

    def neural_gain(self) -> float:
        """Simple linear gain for compatibility."""
        v = self.to_vec()
        w = [0.18, 0.12, 0.22, 0.20, 0.13, 0.25]
        return round(min(1.0, max(0.0, sum(w[i]*v[i] for i in range(6)))), 4)

    def is_collapse_ready(self) -> bool:
        return (self.mu3_pain_collapse > 0.4 and
                self.mu6_tesla_energy  > 0.3 and
                self.mu4_coherence_fold > 0.3)


class TeslaResonance(NamedTuple):
    tau_eff:    float
    tesla_amp:  float
    channel:    int
    phase:      float
    wave:       float


# ── MobiusLatentField v3 ──────────────────────────────────────────────────────

class MobiusLatentField:
    """
    Möbius Latent Manifold Engine v3.

    사용:
        field = MobiusLatentField()

        # entry 순간:
        field.feed_entry(entry_packet)

        # 매 tick (manage_position):
        phys  = OrchestratorPhysics.translate(packet, elapsed)
        field.push(packet, physics=phys, phase=_op)

        # compute:
        state = field.compute(packet, physics=phys)
        tesla = field.tesla(state, packet)
        thr   = field.modulate_thr(thr_base, state, tesla, elapsed, phase=_op)
        score = state.score(phase=_op)   # W matrix score
    """

    def __init__(self):
        self._buf: deque              = deque(maxlen=BUFFER_SIZE)
        self._sparse: SparseLatentMemory = SparseLatentMemory()
        self._drift_ema: float        = 0.0
        self._prev_channel: int       = 1
        self._channel_sustain: int    = 0
        self._push_count: int         = 0
        self._entry_count: int        = 0
        self._last_state: Optional[MobiusState]     = None
        self._last_tesla: Optional[TeslaResonance]  = None
        self._last_physics: dict                    = {}

        # TV encoders per field (slope, shadow, mfe, coherence)
        self._tv_slope    = TVEncoder(7)
        self._tv_shadow   = TVEncoder(7)
        self._tv_mfe      = TVEncoder(7)
        self._tv_coherence= TVEncoder(7)

        # prev for velocity calc
        self._prev_wave_energy: float = 0.0

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def feed_entry(self, packet: dict,
                   physics: Optional[dict] = None) -> None:
        """
        Entry 순간 buffer 주입.
        entry impulse continuity를 latent가 볼 수 있도록.
        packet에 'is_entry': True 태그.
        """
        entry_pkt = dict(packet)
        entry_pkt["is_entry"] = True
        self.push(entry_pkt, physics=physics, phase=0)
        self._entry_count += 1

    def push(self, packet: dict,
             physics: Optional[dict] = None,
             phase: int = 2) -> None:
        """
        12D frame 구성 + TV 갱신 + drift_ema 갱신.
        """
        ph = physics or {}
        frame = [
            float(packet.get("shadow_signal",  0.0)),
            float(packet.get("grammar_signal", 0.0)),
            float(ph.get("slope", packet.get("slope_signal", 0.0))),
            float(ph.get("fold",  packet.get("fold",         0.0))),
            float(packet.get("pain",           0.0)),
            float(ph.get("coherence", packet.get("coherence", 0.5))),
            float(packet.get("cur_r",          0.0)),
            float(packet.get("velocity",       0.0)),
            float(packet.get("delta",          0.0)),
            float(ph.get("torsion", packet.get("torque", 0.0))),
            float(packet.get("mfe",            0.0)),
            float(packet.get("mae",            0.0)),
        ]
        self._buf.append(frame)
        self._push_count += 1

        # TV 갱신
        self._tv_slope.push(frame[F_SLOPE])
        self._tv_shadow.push(frame[F_SHADOW])
        self._tv_mfe.push(frame[F_MFE])
        self._tv_coherence.push(frame[F_COH])

        # drift_ema
        we   = float(ph.get("wave_energy", 0.0))
        sign = 1.0 if we > 0.005 else (-1.0 if we < -0.005 else 0.0)
        self._drift_ema = round(0.92 * self._drift_ema + 0.08 * sign, 6)
        self._prev_wave_energy = we
        self._last_physics = ph

    def compute(self, packet: dict,
                physics: Optional[dict] = None,
                phase: int = 2) -> MobiusState:
        """
        v3 μ 계산 (wave_energy 기반 직접 곱셈).
        """
        ph = physics or self._last_physics or {}

        if len(self._buf) < BAND_LOW:
            state = MobiusState(0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0)
            self._last_state = state
            return state

        buf  = list(self._buf)
        low  = self._band_mean(buf, BAND_LOW)
        mid  = self._band_mean(buf, BAND_MID)
        high = self._band_mean(buf, BAND_HIGH)

        def _layer(indices, vec):
            return sum(abs(vec[i]) for i in indices) / max(len(indices), 1)

        micro = round(min(1.0, _layer(MICRO_IDX, high)), 4)
        meso  = round(min(1.0, _layer(MESO_IDX,  mid)),  4)
        macro = round(min(1.0, _layer(MACRO_IDX, low)),  4)

        # ── TV signals ───────────────────────────────────────────────────
        energy = abs(ph.get("wave_energy", 0.0))
        _stv, _utv_slope   = self._tv_slope.compute(phase, energy)
        _stv, _utv_shadow  = self._tv_shadow.compute(phase, energy)
        _stv, _utv_mfe     = self._tv_mfe.compute(phase, energy)
        _stv, _utv_coh     = self._tv_coherence.compute(phase, energy)

        # ── μ: wave_energy 기반 직접 곱셈 ───────────────────────────────
        impulse     = float(ph.get("impulse",     micro))
        slope       = float(ph.get("slope",       packet.get("slope_signal", 0.0)))
        shadow      = float(ph.get("shadow",      packet.get("shadow_signal", 0.0)))
        torsion     = float(ph.get("torsion",     0.0))
        pain        = float(ph.get("pain",        packet.get("pain", 0.0)))
        col_press   = float(ph.get("col_press",   macro))
        coherence   = float(ph.get("coherence",   packet.get("coherence", 0.5)))
        fold        = float(ph.get("fold",        packet.get("fold", 0.0)))
        wind_rot    = float(ph.get("wind_rotation", 0.0))
        wind_per    = float(ph.get("wind_persist",  0.0))
        tesla_amp   = float(ph.get("tesla_amp",   packet.get("tesla_amp", 0.0)))
        wave_energy = float(ph.get("wave_energy", 0.0))

        # μ1: 순간 힘 (TV-scaled)
        mu1 = round(min(1.0, max(0.0,
            impulse * abs(slope) * (1.0 + _utv_slope))), 4)

        # μ2: 구조 비틀림 (TV-scaled)
        mu2 = round(min(1.0, max(0.0,
            shadow * torsion * (1.0 + _utv_shadow))), 4)

        # μ3: 기억 붕괴 (TV MFE 반영)
        mu3 = round(min(1.0, max(0.0,
            pain * col_press * (1.0 + _utv_mfe))), 4)

        # μ4: 구조 접힘 (coherence TV 반영)
        mu4 = round(min(1.0, max(0.0,
            coherence * fold * (1.0 + _utv_coh))), 4)

        # μ5: windmill 지속성 + drift 방향 (v4: abs 제거 — 부호 정보 보존)
        # 실험 결과: 양 drift(good) vs 음 drift(bad) 분리 필요
        if wind_rot > 0 or wind_per > 0:
            _mu5_raw = wind_rot * wind_per
            mu5 = round(min(1.0, max(-1.0, _mu5_raw)), 4)
        else:
            mu5 = round(min(1.0, max(-1.0, self._drift_ema)), 4)  # 부호 보존

        # μ6: Tesla 공진 에너지 (v4: 음수 wave_energy 차단)
        # 실험: Ewave>0 → avg_R=+0.037 / 음수는 μ3(pain×collapse)가 이미 처리
        _we_pos = max(0.0, wave_energy)
        mu6 = round(min(1.0, max(0.0,
            tesla_amp * _we_pos * max(0.2, coherence))), 4)

        state = MobiusState(micro, meso, macro, mu1, mu2, mu3, mu4, mu5, mu6)

        # sparse memory
        self._sparse.try_push(state.to_vec())
        self._last_state = state
        return state

    def tesla(self, state: MobiusState, packet: dict) -> TeslaResonance:
        """Tesla resonance with channel hysteresis."""
        ph       = self._last_physics or {}
        fold     = float(ph.get("fold",    packet.get("fold", 0.0)))
        shadow   = float(ph.get("shadow",  packet.get("shadow_signal", 0.0)))
        slope    = float(ph.get("slope",   packet.get("slope_signal",  0.0)))

        tesla_amp = round(min(1.0, max(0.0, (shadow + slope + fold) / 3.0)), 4)
        tau_base  = BASE_TAU / max(1.0 + TESLA_K * tesla_amp, 0.01)

        # channel: mu1+mu6 energy
        energy = state.mu1_impulse_slope + state.mu6_tesla_energy
        if energy > 0.60:
            candidate = 2
        elif energy < 0.20:
            candidate = 0
        else:
            candidate = 1

        if candidate == self._prev_channel:
            self._channel_sustain += 1
        else:
            self._channel_sustain = 0

        if self._channel_sustain >= CHANNEL_HYSTERESIS:
            channel = candidate
            self._prev_channel = channel
        else:
            channel = self._prev_channel

        if channel == 2:
            tau_eff = min(tau_base, CHANNEL_TAU[2])
        elif channel == 0:
            tau_eff = max(tau_base, CHANNEL_TAU[0])
        else:
            tau_eff = tau_base

        elapsed = float(packet.get("elapsed", 0.0))
        phase   = elapsed * (math.pi / 2.0) / max(tau_eff, 1.0)
        wave    = max(0.0, math.sin(phase) * math.exp(-phase / 3.5))

        tesla = TeslaResonance(
            tau_eff   = round(tau_eff,   2),
            tesla_amp = round(tesla_amp, 4),
            channel   = channel,
            phase     = round(phase, 4),
            wave      = round(wave,  4),
        )
        self._last_tesla = tesla
        return tesla

    def modulate_field(self, thr_base: float,
                       state: MobiusState,
                       tesla: TeslaResonance,
                       elapsed: float,
                       threshold_floor: float = 0.08,
                       phase: int = 2) -> dict:
        """
        Möbius field modulation (v4) — 실험 결과 반영.

        핵심 전환:
          score는 exit trigger가 아니라 threshold를 휘게 하는 장(field).

          score > 0.35 → HOLD bias (threshold 상향)
          score < 0.08 → EXIT permission (threshold 하향)

        공식:
          thr_eff = base_thr
                  + phase_gain * score   (score높으면 thr 올림: hold)
                  - μ3 * 0.25            (danger gate: μ3 높으면 thr 내림: exit 촉진)
                  + wave_phase_term      (P0/P1: wave↑→thr↑ / P2/P3: wave↑→thr↓)

        phase별 wave 방향:
          P0 (birth):     wave 상승 → thr 올림 (보호)
          P1 (compress):  wave 상승 → thr 소폭 올림
          P2 (collapse):  wave 상승 → thr 내림 (exit 촉진)
          P3 (force):     wave 상승 → thr 크게 내림

        반환: dict {thr, field_score, mu3_gate, wave_term, mode}
        """
        score   = state.score(phase=phase)
        mu3     = state.mu3_pain_collapse
        wave    = tesla.wave
        mu5_dir = state.mu5_wind_persist   # 부호 있음 (v4)

        # phase별 score gain (score가 thr에 미치는 방향)
        # P0/P1: score↑ → thr↑ (HOLD bias)
        # P2: 약한 score gain (collapse 허용)
        # P3: score gain=0 (μ3 + wave만으로 결정)
        _SCORE_GAIN = {0: +0.30, 1: +0.20, 2: +0.08, 3: 0.0}
        phase_gain = _SCORE_GAIN.get(phase, 0.08)

        # wave term (phase-inverted)
        # P0/P1: wave↑ → threshold 올림 (entry zone 보호)
        # P2/P3: wave↑ → threshold 내림 (collapse zone exit 촉진)
        _WAVE_DIR = {0: +0.12, 1: +0.06, 2: -0.18, 3: -0.28}
        wave_term = _WAVE_DIR.get(phase, -0.18) * wave

        # μ3 danger gate — phase별 강도 다름
        # P0: μ3 약하게 (birth zone 보호 — 진입 직후 pain은 정상)
        # P1: 중간
        # P2/P3: 강하게 (collapse/force exit)
        _MU3_K = {0: 0.08, 1: 0.15, 2: 0.25, 3: 0.32}
        mu3_gate = -mu3 * _MU3_K.get(phase, 0.25)

        # μ5 drift direction bonus
        # 양 drift(μ5>0) = healthy = thr 소폭 올림
        # 음 drift(μ5<0) = collapsing = thr 소폭 내림
        drift_term = mu5_dir * 0.05

        delta = phase_gain * score + mu3_gate + wave_term + drift_term
        thr   = round(max(threshold_floor,
                          min(thr_base * 2.0,   # 상한: base의 2배
                              thr_base + delta)), 4)

        # mode 판별 (로그용)
        if score > 0.35:
            mode = "HOLD_BIAS"
        elif mu3 > 0.52:
            mode = "DANGER_GATE"
        elif score < 0.08 and mu3 < 0.20:
            mode = "EXIT_PERM"
        else:
            mode = "FIELD_NORM"

        return {
            "thr":         thr,
            "field_score": round(score,    4),
            "mu3_gate":    round(mu3_gate, 4),
            "wave_term":   round(wave_term,4),
            "drift_term":  round(drift_term,4),
            "delta":       round(delta,    4),
            "mode":        mode,
        }

    def modulate_thr(self, thr_base: float,
                     state: MobiusState,
                     tesla: TeslaResonance,
                     elapsed: float,
                     threshold_floor: float = 0.08,
                     phase: int = 2) -> float:
        """
        하위 호환용 (v3 API). modulate_field()를 내부 호출.
        """
        result = self.modulate_field(
            thr_base, state, tesla, elapsed, threshold_floor, phase)
        return result["thr"]

    def enrich_packet(self, packet: dict,
                      state: MobiusState,
                      tesla: TeslaResonance,
                      physics: dict) -> dict:
        """orbital packet에 latent + physics 전체 주입."""
        packet["mobius_mu"]        = state.to_vec()
        packet["mobius_micro"]     = state.micro
        packet["mobius_meso"]      = state.meso
        packet["mobius_macro"]     = state.macro
        packet["mobius_gain"]      = state.neural_gain()
        packet["mobius_cp"]        = state.collapse_pressure()
        packet["mobius_score_p2"]  = state.score(phase=2)
        packet["tesla_tau_eff"]    = tesla.tau_eff
        packet["tesla_amp"]        = tesla.tesla_amp
        packet["tesla_channel"]    = tesla.channel
        packet["tesla_wave"]       = tesla.wave
        packet["wave_energy"]      = physics.get("wave_energy",  0.0)
        packet["phys_impulse"]     = physics.get("impulse",      0.0)
        packet["phys_pressure"]    = physics.get("pressure",     0.0)
        packet["phys_inertia"]     = physics.get("inertia",      0.0)
        packet["phys_dissipation"] = physics.get("dissipation",  0.0)
        packet["phys_torsion"]     = physics.get("torsion",      0.0)
        return packet

    def build_brain_latent(self, state: MobiusState,
                           tesla: TeslaResonance,
                           physics: dict) -> dict:
        d = state.to_brain_input()
        d["tesla_tau_eff"]    = tesla.tau_eff
        d["tesla_channel"]    = tesla.channel
        d["tesla_wave"]       = tesla.wave
        d["wave_energy"]      = physics.get("wave_energy",  0.0)
        d["phys_pressure"]    = physics.get("pressure",     0.0)
        d["phys_dissipation"] = physics.get("dissipation",  0.0)
        d["phys_torsion"]     = physics.get("torsion",      0.0)
        return d

    def summary_log(self, state: MobiusState,
                    tesla: TeslaResonance,
                    physics: Optional[dict] = None,
                    phase: int = 2) -> str:
        we  = physics.get("wave_energy", 0.0) if physics else 0.0
        scr = state.score(phase=phase)
        return (
            f"mob μ=["
            f"{state.mu1_impulse_slope:.2f},"
            f"{state.mu2_shadow_torsion:.2f},"
            f"{state.mu3_pain_collapse:.2f},"
            f"{state.mu4_coherence_fold:.2f},"
            f"{state.mu5_wind_persist:.2f},"
            f"{state.mu6_tesla_energy:.2f}]"
            f" score={scr:.4f}"
            f" cp={state.collapse_pressure():.3f}"
            f" ch{tesla.channel}(τ={tesla.tau_eff:.0f}s"
            f",w={tesla.wave:.3f})"
            f" Ewave={we:+.4f}"
        )

    def get_stats(self) -> dict:
        return {
            "frames":     self._push_count,
            "entries":    self._entry_count,
            "buf_size":   len(self._buf),
            "sparse_n":   len(self._sparse),
            "drift_ema":  round(self._drift_ema, 4),
            "channel":    self._prev_channel,
            "ch_sustain": self._channel_sustain,
        }

    @staticmethod
    def _band_mean(buf: list, window: int) -> list:
        recent = buf[-window:]
        n = len(recent)
        if n == 0:
            return [0.0] * FRAME_DIM
        return [sum(f[i] for f in recent) / n for i in range(FRAME_DIM)]
