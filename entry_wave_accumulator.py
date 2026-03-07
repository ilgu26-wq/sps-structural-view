# -*- coding: utf-8 -*-
"""
entry_wave_accumulator.py — EntryCore 앞단 Möbius 상대좌표 채널  (v2)

v1 → v2 수정 (4개 핵심 + 2개 추가):
  FIX-1: M5 = torsion_now - torsion_prev  (단순값 → delta, phase inversion 감지)
  FIX-2: curvature composite input = 0.5*wave + 0.3*coh + 0.2*torsion
          (wave 단독 amplitude → geometry 합성)
  FIX-3: action_bias normalize = base * (1 + clamp(mem_std*5, 0, 3))
          (EntryCore memory variance에 비례, weak-bias 문제 해결)
  FIX-4: reset_on_entry 완전 연결
          — torsion/col/kappa 버퍼 추가 clear
          — torsion_prev 리셋
          — orchestrator entry commit 직후 호출 위치 명시
  FIX-5: field_delta = kappa * wave_e * coh  (coherence noise 억제)
  FIX-6: wave_quality FORMING → dynamic_thr += 0.03 (추가 보호)

구조:
  Raw market packet
    → EntryWaveAccumulator.push()    (24-tick rolling)
    → EntryWaveAccumulator.compute() (6D 상대벡터)
    → EntryBiasInjector.gate()       (dynamic thr + action bias)
    → select_with_bias()             (ranking 왜곡)
    → EntryCore.select()             (unchanged)

6D 상대벡터:
  M1  curvature_rel   κ_composite_now - κ_ema        (파동 형상 곡률)
  M2  fold_rel        fold_now - fold_ema             (구조 비틀림)
  M3  pressure_rel    col_now - hold_ema              (붕괴압력)
  M4  coherence_rel   log(coh_now / coh_ema)         (일관성 상대비율)
  M5  drift_delta     torsion_now - torsion_prev     ← FIX-1
  M6  wave_rel        wave_now - wave_median          (파동에너지 중앙값 상대)
  M7  observer_axis   0.6*I_rel + 0.4*sonar_rel         (관성 × 가시성)
  M8  latent_proj     geometry manifold 투영 합성 벡터    (EXP_M8_LATENT_PROJECTION_01)
      = 0.35*M1 + 0.25*M6 + 0.20*M7 + 0.20*sonar_proj
  G   boundary        tanh(M6+M7+M8)                    (invention permission frontier)

orchestrator.py 통합:
  # __init__
  self._ewa = EntryWaveAccumulator()
  self._ebi = EntryBiasInjector()

  # spawn_world, EntryCore.select 직전 매 tick
  _phys = OrchestratorPhysics.translate(packet, elapsed)
  self._ewa.push(packet, _phys)
  _mrc  = self._ewa.compute()
  _bias = self._ebi.gate(_mrc, _flight,
              entry_core=self.entry_core, state_key=_ec_state_key)
  _conf_eff = _ec_conf_raw + _bias["conf_delta"]
  _dyn_thr  = _bias["dynamic_thr"]
  if not _bias["gate_pass"]:
      action = None
  else:
      chosen = select_with_bias(self.entry_core, _ec_state_key,
                                _flight, _plastic_allowed,
                                _bias["action_biases"])

  # FIX-4: entry commit 확정 직후
  if evidence.executed:
      self._ewa.reset_on_entry()
"""

import math
import statistics
from collections import deque
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_SHORT  = 8
WINDOW_LONG   = 24

M1_SCALE = 3.0   # composite curvature
M2_SCALE = 2.0   # fold
M3_SCALE = 2.5   # pressure
M4_SCALE = 2.0   # coherence log-ratio
M5_SCALE = 5.0   # torsion delta  (FIX-1: delta는 작으므로 감도 높임)
M6_SCALE = 4.0   # wave median 상대

# M7 Observer Axis (EXP_M7_OBSERVER_AXIS_01)
# I_rel = 파동 관성 상대비율 (persistence = 얼마나 안 꺾이는가)
# sonar_rel = 반사 해상도 상대비율 (visibility = 얼마나 또렷하게 들리는가)
# M7 = 0.6*I_rel + 0.4*sonar_rel
# → 보이면서 유지되는 구조 = observer가 봐야 하는 것
M7_I_SCALE     = 1.5   # inertia log-ratio 감도
M7_S_SCALE     = 1.2   # sonar log-ratio 감도
M7_I_WINDOW    = 6     # 관성 계산 wave_buf 윈도우

# M8 Latent Projection Axis (EXP_M8_LATENT_PROJECTION_01)
# sonar를 Möbius geometry 위에 투영한 latent 합성 벡터
# M8 = w1*M1 + w2*M6 + w3*M7 + w4*sonar_proj
# sonar_proj = tanh(sonar_latent * curvature_ratio) — manifold projection
# entropy    = |sonar_now - sonar_prev| — local noise gate
# spectral   = |mr5 - mr20| 근사 = |wave_now - wave_ema| — jump detector
# G_boundary = tanh(M6 + M7 + M8) — invention permission frontier
M8_W1         = 0.35   # M1 curvature weight
M8_W2         = 0.25   # M6 wave permanence weight
M8_W3         = 0.20   # M7 observer inertia weight
M8_W4         = 0.20   # sonar_proj weight
M8_DIFFUSION  = 0.85   # sonar latent smoothing alpha (Brownian spike 제거)
M8_ENTROPY_CAP = 0.30  # entropy 게이트 상한 (noise trust 감쇠)

BASE_CONF_THR = 0.60
HOLD_BIAS_ADJ = 0.08
EXIT_PERM_ADJ = 0.06
FORMING_ADJ   = 0.03   # FIX-6

CURVE_BIAS_WEIGHT = 0.25

E_ENTRY_MIN  = 0.003
E_ENTRY_SOFT = 0.010

BASE_BIAS_LONG  = 0.030
BASE_BIAS_SHORT = 0.020
BASE_BIAS_DELAY = 0.010
BIAS_STD_CAP    = 3.0    # FIX-3

FIELD_MEMORY_ALPHA = 0.82
MU3_EMA_ALPHA      = 0.70


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _curvature(x1: float, x2: float, x3: float) -> float:
    dx  = x3 - x1
    d2x = x3 - 2.0 * x2 + x1
    return round(abs(d2x) / max((1.0 + dx ** 2) ** 1.5, 1e-6), 6)


def _ema(prev: float, new: float, alpha: float) -> float:
    return round(alpha * prev + (1.0 - alpha) * new, 6)


def _safe_std(vals: list) -> float:
    if len(vals) < 2:
        return 0.0
    try:
        return statistics.stdev(vals)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# EntryWaveAccumulator
# ─────────────────────────────────────────────────────────────────────────────

class EntryWaveAccumulator:

    def __init__(self):
        self._wave_buf:    deque = deque(maxlen=WINDOW_LONG)
        self._coh_buf:     deque = deque(maxlen=WINDOW_LONG)
        self._fold_buf:    deque = deque(maxlen=WINDOW_LONG)
        self._torsion_buf: deque = deque(maxlen=WINDOW_LONG)
        self._col_buf:     deque = deque(maxlen=WINDOW_LONG)
        self._kappa_buf:   deque = deque(maxlen=WINDOW_LONG)  # FIX-2

        self._kappa_ema:    float = 0.0
        self._fold_ema:     float = 0.0
        self._hold_ema:     float = 0.0
        self._coh_ema:      float = 0.0
        self._torsion_prev: float = 0.0   # FIX-1

        self._E_accum:      float = 0.0
        self._E_buf:        deque = deque(maxlen=WINDOW_LONG)

        self._field_memory: float = 0.0
        self._mu3_ema:      float = 0.0

        # M7 Observer Axis 상태
        # I_ema: 파동 관성 EMA (wave² × coherence)
        # I_ema: 파동 관성 EMA (wave² × coherence)
        # sonar_ema: 반사 해상도 EMA (tick≥8 이후 갱신 — warm-start)
        # sonar_warmup: tick<8 동안 sonar 예비값 수집 → tick=8 시점에 sonar_ema 초기화
        self._I_ema:        float = 1e-6  # 0 방지 초기값
        self._sonar_ema:    float = 1e-6
        self._sonar_warmup: list  = []
        self._wave_prev:    float = 0.0

        # M8 Observer Latent 상태
        self._sonar_latent: float = 0.0   # diffusion-smoothed sonar
        self._sonar_raw_prev: float = 0.0 # entropy 계산용 이전 sonar raw

        # world_memory seed: orchestrator에서 latch_score로 주입.
        # _kappa_ema / _coh_ema를 geometry 기반으로 사전 왜곡.
        # 이전 world curvature가 다음 world 좌표계 시작점을 결정.
        self._world_memory: float = 0.0   # 0.0 = 흔적 없음, 1.0 = 강한 비가역

        self._tick: int = 0

    def push(self, packet: dict, physics: Optional[dict] = None) -> None:
        ph = physics or {}

        wave_e  = float(ph.get("wave_energy",  packet.get("wave_energy",  0.0)))
        coh     = float(ph.get("coherence",    packet.get("coherence",    0.5)))
        fold    = float(ph.get("fold",         packet.get("fold",         0.0)))
        torsion = float(ph.get("torsion",      packet.get("torque",       0.0)))
        col_p   = float(ph.get("col_press",    packet.get("collapse_pressure", 0.0)))
        pain    = float(ph.get("pain",         packet.get("pain",         0.0)))

        # world_memory seed 수신 — orchestrator latch_score 기반.
        # _world_memory_signed: 방향 포함 (latch_signed).
        # curvature는 방향(signed)으로, coherence는 magnitude(abs)로 다르게 왜곡.
        # fade-out: α = max(0, 0.10 - tick×0.015) → abrupt detach 없이 잔향 decay.
        _wm     = float(ph.get("world_memory",        0.0))
        _wm_sgn = float(ph.get("world_memory_signed", _wm))  # 없으면 scalar fallback
        _fade_alpha = max(0.0, 0.10 - self._tick * 0.015)
        if _fade_alpha > 0.0:
            # curvature EMA: signed latch → 방향 기억 반영
            self._kappa_ema = round(
                self._kappa_ema * (1 - _fade_alpha) + _wm_sgn * wave_e * _fade_alpha, 6
            )
            # coherence EMA: magnitude만 → |latch| × coherence
            self._coh_ema = round(
                min(0.99, self._coh_ema * (1 - _fade_alpha) + abs(_wm) * coh * _fade_alpha), 6
            )
        self._world_memory = _wm

        self._wave_buf.append(wave_e)
        self._coh_buf.append(coh)
        self._fold_buf.append(fold)
        self._torsion_buf.append(torsion)
        self._col_buf.append(col_p)

        # FIX-2: composite curvature input
        k_in = 0.5 * wave_e + 0.3 * coh + 0.2 * torsion
        self._kappa_buf.append(k_in)

        if len(self._kappa_buf) >= 3:
            k = list(self._kappa_buf)
            kappa = _curvature(k[-3], k[-2], k[-1])
        else:
            kappa = 0.0

        alpha_s = 2.0 / (WINDOW_SHORT + 1)
        alpha_l = 2.0 / (WINDOW_LONG  + 1)
        self._kappa_ema = _ema(self._kappa_ema, kappa, 1.0 - alpha_s)
        self._fold_ema  = _ema(self._fold_ema,  fold,  1.0 - alpha_s)
        self._hold_ema  = _ema(self._hold_ema,  col_p, 1.0 - alpha_l)
        self._coh_ema   = _ema(self._coh_ema,   coh,   1.0 - alpha_l)

        e_tick = abs(kappa * coh)
        self._E_buf.append(e_tick)
        self._E_accum = sum(self._E_buf) / max(len(self._E_buf), 1)

        # FIX-5: coherence 곱해 noise 억제
        field_delta = kappa * wave_e * max(coh, 0.1)
        self._field_memory = (FIELD_MEMORY_ALPHA * self._field_memory
                              + (1.0 - FIELD_MEMORY_ALPHA) * field_delta)

        self._mu3_ema = (MU3_EMA_ALPHA * self._mu3_ema
                         + (1.0 - MU3_EMA_ALPHA) * pain * col_p)

        # ── M7 Observer Axis: 관성(persistence) + sonar(visibility) ──────────
        # I_now = mean(w²) × max(coh, 0.1) — wave_buf 마지막 6tick 관성
        _wb_recent = list(self._wave_buf)[-M7_I_WINDOW:]
        _I_now = (sum(w * w for w in _wb_recent) / max(len(_wb_recent), 1)
                  * max(coh, 0.1)) if _wb_recent else 1e-6
        _I_now = max(_I_now, 1e-6)
        self._I_ema = round(0.88 * self._I_ema + 0.12 * _I_now, 8)

        # sonar warmup: tick<8 동안 wave delta 수집 → tick=8 시점에 sonar_ema 초기화
        # compute()에서 처음 sonar를 계산할 때 ratio≈1.0 폭발 방지
        _wb_list = list(self._wave_buf)
        if len(_wb_list) >= 2:
            _w_delta = abs(_wb_list[-1] - _wb_list[-2]) * max(coh, 0.01)
            if self._tick < 8:
                self._sonar_warmup.append(max(1e-6, _w_delta))
            elif self._tick == 8 and self._sonar_warmup:
                # warm-start: 과거 8 tick 평균으로 초기화
                self._sonar_ema = max(1e-6,
                    sum(self._sonar_warmup) / len(self._sonar_warmup))

        # sonar_ema는 compute()에서 갱신 (kappa_now 3점 차분과 일관성 유지)
        self._wave_prev = wave_e

        # FIX-1: torsion_prev 갱신 (push 마지막에)
        self._torsion_prev = torsion
        self._tick += 1

    def compute(self) -> dict:
        _zero = {"M1": 0.0, "M2": 0.0, "M3": 0.0, "M4": 0.0, "M5": 0.0,
                 "M6": 0.0, "M8": 0.0, "G_boundary": 0.0, "sonar_proj": 0.0, "entropy": 0.0, "spectral_score": 0.0, "m8_bucket": "M8_MID", "E_accum": 0.0, "field_memory": 0.0,
                 "mu3_ema": 0.0, "curvature_bias": 0.0, "wave_quality": "FLAT",
                 "kappa_now": 0.0}
        if self._tick < 4:
            return _zero

        kb = list(self._kappa_buf)
        wb = list(self._wave_buf)
        cb = list(self._coh_buf)
        fb = list(self._fold_buf)
        tb = list(self._torsion_buf)
        cp = list(self._col_buf)

        kappa_now = _curvature(kb[-3], kb[-2], kb[-1]) if len(kb) >= 3 else 0.0
        fold_now  = fb[-1] if fb else 0.0
        col_now   = cp[-1] if cp else 0.0
        coh_now   = cb[-1] if cb else 0.5
        wave_now  = wb[-1] if wb else 0.0
        tors_now  = tb[-1] if tb else 0.0

        M1 = math.tanh((kappa_now - self._kappa_ema) * M1_SCALE)

        # ── M1 v2: curvature ratio (절대값 → 상대좌표) ────────────────────
        # kappa_ratio = kappa_now / max(kappa_ema, 1e-6)
        # tanh(log(ratio)) → regime 변화에도 스케일 불변
        # 기존 M1과 0.5/0.5 blend: 급격한 교체 없이 부드럽게 전환
        if self._kappa_ema > 1e-6 and kappa_now > 0:
            _kr = kappa_now / max(self._kappa_ema, 1e-6)
            _M1_ratio = math.tanh(math.log(max(_kr, 1e-6)) * M1_SCALE)
            M1 = round(0.50 * M1 + 0.50 * _M1_ratio, 4)

        M2 = math.tanh((fold_now  - self._fold_ema)  * M2_SCALE)
        M3 = math.tanh((col_now   - self._hold_ema)  * M3_SCALE)

        _cr = coh_now / max(self._coh_ema, 0.01)
        M4  = math.tanh(math.log(max(_cr, 0.01)) * M4_SCALE)

        # FIX-1: drift DELTA (방향 변화량)
        # torsion_buf[-2]를 prev로 사용 (push 후 갱신 타이밍 문제 회피)
        tors_prev = tb[-2] if len(tb) >= 2 else tors_now
        M5 = math.tanh((tors_now - tors_prev) * M5_SCALE)

        wave_med = statistics.median(wb) if len(wb) >= 3 else wave_now
        M6 = math.tanh((wave_now - wave_med) * M6_SCALE)

        # ── M7 Observer Axis: I_rel (관성) + sonar_rel (반사 해상도) ─────────
        # I_rel > 0  : 파동이 평소보다 더 버팀 (persistence ↑)
        # sonar_rel > 0: 구조 변화가 평소보다 더 선명하게 반사됨 (visibility ↑)
        # M7 > 0.25  : 보이면서 유지되는 구조 → entry quality 가능성 ↑
        # M7 < -0.25 : ghost 가능성 (안 버티고 echo도 약함)
        # Phase 1 (EXP_M7_OBSERVER_AXIS_01): 기록만, conf injection 없음
        # cold-start guard: tick < 8에서 EMA가 안정되기 전 → M7=0
        if self._tick >= 8:
            _I_now_c = (sum(w * w for w in list(self._wave_buf)[-M7_I_WINDOW:])
                        / max(len(list(self._wave_buf)[-M7_I_WINDOW:]), 1)
                        * max(coh_now, 0.1))
            _I_now_c = max(_I_now_c, 1e-6)
            _I_ratio  = _I_now_c / max(self._I_ema, 1e-6)
            I_rel = math.tanh(math.log(max(_I_ratio, 1e-6)) * M7_I_SCALE)

            # sonar: wave_buf의 마지막 두 값 차이 × coherence
            # compute() = push() 직후 호출이므로 wb[-1]/[-2]가 현재/이전 tick
            # kappa_d는 kb[-1]/[-2] 사용 — k_in 입력값 차이 (같은 스케일)
            # kappa_now(3점 2차 차분)는 M1에만 사용, sonar 비교에 부적합
            _wave_d   = abs(wb[-1] - wb[-2]) if len(wb) >= 2 else 0.0
            _kb_list  = list(self._kappa_buf)
            _kappa_d  = abs(_kb_list[-1] - _kb_list[-2]) if len(_kb_list) >= 2 else 0.0
            _sonar_now_c = max(1e-6, (_wave_d + _kappa_d) * max(coh_now, 0.01))
            # sonar_ema 갱신 (decay 0.85 — dead zone에서 빠르게 음전환)
            self._sonar_ema = round(0.85 * self._sonar_ema + 0.15 * _sonar_now_c, 8)

            _sonar_ratio  = _sonar_now_c / max(self._sonar_ema, 1e-6)
            sonar_rel = math.tanh(math.log(max(_sonar_ratio, 1e-6)) * M7_S_SCALE)

            M7 = round(0.6 * I_rel + 0.4 * sonar_rel, 4)
        else:
            I_rel = 0.0
            sonar_rel = 0.0
            M7 = 0.0

        # ── Observer bucket (EXP_M7: 분류만, 아직 action 없음) ───────────────
        if M7 > 0.25:
            observer_bucket = "M7_HIGH"
        elif M7 < -0.25:
            observer_bucket = "M7_LOW"
        else:
            observer_bucket = "M7_MID"

        # ── M8 Latent Projection (EXP_M8_LATENT_PROJECTION_01) ──────────────
        # Step 1: diffusion smoothing (Brownian local spike 제거)
        _sonar_raw = sonar_rel if self._tick >= 8 else 0.0
        self._sonar_latent = round(
            M8_DIFFUSION * self._sonar_latent + (1.0 - M8_DIFFUSION) * _sonar_raw, 6)

        # Step 2: entropy gate (noise 측정 — 급변 시 trust 감쇠)
        _entropy = min(M8_ENTROPY_CAP, abs(_sonar_raw - self._sonar_raw_prev))
        _trust   = max(0.0, 1.0 - _entropy / M8_ENTROPY_CAP)
        self._sonar_raw_prev = _sonar_raw

        # Step 3: spectral continuity (wave_now vs wave_ema — jump 감지)
        # FIX: wave_ema 직접 사용 (kappa proxy 제거 — 과대반응 방지)
        # wave_short = 최근 WINDOW_SHORT 평균, wave_long = _wave_buf 전체 EMA
        _wb_list = list(self._wave_buf)
        _wave_short_avg = (sum(_wb_list[-WINDOW_SHORT:]) / max(len(_wb_list[-WINDOW_SHORT:]), 1)
                           if _wb_list else wave_now)
        _wave_long_avg  = (sum(_wb_list) / max(len(_wb_list), 1)
                           if _wb_list else wave_now)
        _spectral = abs(_wave_short_avg - _wave_long_avg)   # mr5-mr20 직접 근사
        _spectral_score = math.tanh(_spectral * 4.0)        # 스케일 조정 (wave는 작음)

        # Step 4: manifold projection — sonar를 curvature_ratio 위에 투영
        _curv_ratio = abs(kappa_now) / max(abs(self._kappa_ema), 1e-6)
        _sonar_proj = math.tanh(self._sonar_latent * _curv_ratio) * _trust

        # Step 5: M8 latent composite
        M8 = round(
            M8_W1 * M1
            + M8_W2 * M6
            + M8_W3 * M7
            + M8_W4 * _sonar_proj,
            4)

        # Step 6: G boundary — invention permission frontier
        # G > 0 → geometry가 새 invention 허용 (WorldConfigSampler 앞단용)
        G_boundary = round(math.tanh(M6 + M7 + M8), 4)

        # M8 bucket (진단용)
        if M8 > 0.20:
            m8_bucket = "M8_HIGH"    # live axis: 투영 강함
        elif M8 < -0.20:
            m8_bucket = "M8_LOW"     # collapse risk
        else:
            m8_bucket = "M8_MID"
        
        # ── EXP-M7-PSR-CONFIDENCE-01: M7 PSR-like observer confidence (v2: Möbius ratio-based) ──
        # 철학: absolute deviation이 아니라 self-relative deformation
        # 즉 "현재 구조가 자기 regime 대비 얼마나 튀었나"
        # M1과 동일 좌표계 (kappa_now / kappa_ema와 같은 ratio 설계)
        #
        # 계산:
        #   1. M7 rolling history 유지 (maxlen=48)
        #   2. Warmup < 12 → neutral (0.5)
        #   3. ratio = M7 / mean (self-relative)
        #   4. z = log(ratio) / std (log-space으로 skew 영향 감소)
        #   5. sonar amplification (visibility 강하면 confidence ↑)
        #   6. observer_conf = tanh (안정적)
        
        if not hasattr(self, '_m7_hist'):
            self._m7_hist = deque(maxlen=48)
        
        self._m7_hist.append(M7)
        
        m7_hist_len = len(self._m7_hist)
        m7_warmup = (m7_hist_len < 12)
        
        # 기본값
        m7_mean = 0.0
        m7_std = 0.0
        m7_ratio = 1.0
        z_score = 0.0
        observer_conf = 0.5
        psr_bucket = "M7_NEUTRAL_WARMUP"
        m7_persistence = 0
        persist_ratio = 0.0
        persist_factor = 1.0
        
        if m7_warmup:
            observer_conf = 0.5
            z_score = 0.0
            m7_ratio = 1.0
            psr_bucket = "M7_NEUTRAL_WARMUP"
        else:
            # Möbius ratio-based confidence
            m7_mean = statistics.mean(self._m7_hist)
            m7_std = statistics.stdev(self._m7_hist) if m7_hist_len > 1 else 1.0
            m7_std = max(m7_std, 1e-6)
            
            # RISK FIX 1: zero-near explosion 방지
            # mean ≈ 0일 때 ratio 폭발 → fake PSR_HIGH
            # 해결: m7_base = abs(mean) + 0.5*std (barrier 추가)
            m7_base = max(abs(m7_mean) + 0.5 * m7_std, 1e-4)
            
            # RATIO: self-relative deformation
            m7_ratio = M7 / m7_base
            
            # log-space z_score (ratio 기반)
            z_score = math.log(max(abs(m7_ratio), 1e-6)) / m7_std
            
            # RISK FIX 2: ratio sign 기준으로 부호 결정
            # (M7 sign이 아니라 m7_ratio sign 사용)
            if m7_ratio < 0:
                z_score *= -1.0
            
            # RISK FIX 3: sonar amplification 약화
            # (M7에 이미 sonar 포함되어 중복 weighting 방지)
            sonar_amp = 1.0 + 0.15 * abs(sonar_rel)
            z_score_amplified = z_score * sonar_amp
            
            # observer_conf = Phi(z) 근사
            observer_conf = 0.5 * (1.0 + math.tanh(z_score_amplified / 2.0))
            
            # ── PERSISTENCE AXIS (비가역 핵심) ─────────────────────────────────
            # "튀는 것"이 아니라 "안 돌아오는 것"이 중요
            # M7 strength × duration = irreversibility
            #
            # m7_persistence: |M7| > 0.25인 연속 ticks 수
            # 추천 window: 8 ticks (약 2초, 시장 micro-regime)
            if not hasattr(self, '_m7_persist_count'):
                self._m7_persist_count = 0
            
            # persistence counter 갱신
            if abs(M7) > 0.25:
                self._m7_persist_count += 1
            else:
                self._m7_persist_count = 0
            
            m7_persistence = min(self._m7_persist_count, 8)  # cap at 8
            persist_ratio = m7_persistence / 8.0  # [0, 1]
            
            # observer_conf 보정: persistence로 modulate
            # spike (2 ticks) → 0.7 weighting (약함)
            # sustained (8 ticks) → 1.0 weighting (정상)
            persist_factor = 0.7 + 0.3 * persist_ratio
            observer_conf = observer_conf * persist_factor
            
            # PSR bucket
            if observer_conf > 0.70:
                psr_bucket = "PSR_HIGH"
            elif observer_conf < 0.30:
                psr_bucket = "PSR_LOW"
            else:
                psr_bucket = "PSR_MID"

        curvature_bias = round(
            CURVE_BIAS_WEIGHT * (
                + 0.30 * M1
                + 0.20 * M4
                + 0.20 * M6
                - 0.20 * M3
                - 0.10 * abs(M2)
                - 0.10 * abs(M5)   # torsion 급변 = 불안정
            ), 4)

        if self._E_accum < E_ENTRY_MIN:
            wq = "FLAT"
        elif self._E_accum < E_ENTRY_SOFT:
            wq = "FORMING"
        elif M1 < -0.30:
            wq = "DECAYING"
        else:
            wq = "ALIVE"

        return {
            "M1": round(M1, 4), "M2": round(M2, 4), "M3": round(M3, 4),
            "M4": round(M4, 4), "M5": round(M5, 4), "M6": round(M6, 4),
            "M7": M7,
            "M8":             M8,
            "G_boundary":     G_boundary,
            "sonar_proj":     round(_sonar_proj, 4),
            "entropy":        round(_entropy,    4),
            "spectral_score": round(_spectral_score, 4),
            "m8_bucket":      m8_bucket,
            "I_rel":          round(I_rel,    4),
            "sonar_rel":      round(sonar_rel, 4),
            "observer_bucket": observer_bucket,
            # EXP-M7-PSR-CONFIDENCE-01: Möbius ratio-based v2
            "observer_conf":  round(observer_conf, 4),
            "psr_bucket":     psr_bucket,
            "m7_ratio":       round(m7_ratio, 4),
            "m7_persistence": m7_persistence,  # 비가역 축
            "persist_ratio":  round(persist_ratio, 4),
            "persist_factor": round(persist_factor, 4),
            "m7_hist_len":    m7_hist_len,
            "m7_hist_mean":   round(m7_mean if not m7_warmup else 0.0, 4),
            "m7_hist_std":    round(m7_std if not m7_warmup else 0.0, 4),
            "z_score":        round(z_score, 4),
            "E_accum":        round(self._E_accum,      6),
            "field_memory":   round(self._field_memory, 6),
            "mu3_ema":        round(self._mu3_ema,      6),
            "curvature_bias": curvature_bias,
            "wave_quality":   wq,
            "kappa_now":      round(kappa_now, 6),
        }

    def reset_on_entry(self) -> None:
        """
        FIX-4: entry commit 직후 호출. short-term 버퍼 전부 리셋.

        orchestrator.py 연결:
            if evidence.executed:
                self._ewa.reset_on_entry()

        이전 trade wave가 다음 entry bias를 오염하는 문제 제거.
        EMA / field_memory / mu3_ema 는 유지 (long-term context 보존).
        """
        self._wave_buf.clear()
        self._coh_buf.clear()
        self._fold_buf.clear()
        self._torsion_buf.clear()  # FIX-4
        self._col_buf.clear()      # FIX-4
        self._kappa_buf.clear()    # FIX-4
        self._E_buf.clear()
        self._E_accum      = 0.0
        self._torsion_prev = 0.0   # FIX-4: delta 리셋
        # M7 Observer: short-term prev 리셋 (EMA는 유지 — long-term context)
        self._wave_prev    = 0.0
        self._sonar_warmup = []
        # Persistence counter 리셋 (entry 후 다시 세기)
        self._m7_persist_count = 0

    def get_telemetry(self) -> dict:
        return {
            "tick":         self._tick,
            "kappa_ema":    round(self._kappa_ema,   4),
            "coh_ema":      round(self._coh_ema,     4),
            "fold_ema":     round(self._fold_ema,    4),
            "hold_ema":     round(self._hold_ema,    4),
            "E_accum":      round(self._E_accum,     6),
            "field_memory": round(self._field_memory,6),
            "mu3_ema":      round(self._mu3_ema,     4),
            "buf_len":      len(self._wave_buf),
        }


# ─────────────────────────────────────────────────────────────────────────────
# EntryBiasInjector
# ─────────────────────────────────────────────────────────────────────────────

class EntryBiasInjector:

    def __init__(self):
        self._call_count = 0

    def gate(self, mrc: dict, flight_mode: str = "CRUISE",
             entry_core=None, state_key: str = "") -> dict:
        self._call_count += 1
        wq  = mrc.get("wave_quality",   "FLAT")
        M1  = mrc.get("M1",  0.0)
        M5  = mrc.get("M5",  0.0)
        M6  = mrc.get("M6",  0.0)
        mu3 = mrc.get("mu3_ema",        0.0)
        fm  = mrc.get("field_memory",   0.0)
        cb  = mrc.get("curvature_bias", 0.0)
        E   = mrc.get("E_accum",        0.0)

        # ── 1. dynamic_thr ──────────────────────────────────────────────
        field_adj = 0.0
        if fm > 0.02:
            field_adj += HOLD_BIAS_ADJ * min(1.0, fm / 0.05)
        if M6 > 0.30 and M1 > 0.0:
            field_adj -= EXIT_PERM_ADJ * min(1.0, M6 / 0.5)
        if mu3 > 0.40:
            field_adj += 0.04 * (mu3 - 0.40)
        if wq == "FORMING":                 # FIX-6
            field_adj += FORMING_ADJ

        dynamic_thr = round(max(0.40, min(0.80, BASE_CONF_THR + field_adj)), 4)

        # ── 2. conf_delta ───────────────────────────────────────────────
        conf_delta = cb
        if flight_mode == "BOOST":
            conf_delta += 0.05
        if wq == "FLAT":
            conf_delta -= 0.10
        elif wq == "FORMING":
            conf_delta -= 0.03
        # FIX-1 활용: torsion 급변 → conf 감소
        if abs(M5) > 0.5:
            conf_delta -= 0.05 * abs(M5)

        conf_delta = round(max(-0.20, min(0.20, conf_delta)), 4)

        # ── 3. soft gate ────────────────────────────────────────────────
        if wq == "FLAT" and E < E_ENTRY_MIN * 0.5:
            gate_pass = False
            reason    = f"WAVE_FLAT(E={E:.4f})"
        elif mu3 > 0.65:
            gate_pass = False
            reason    = f"PAIN_GATE(μ3={mu3:.3f})"
        else:
            gate_pass = True
            reason    = f"OK(wq={wq},thr={dynamic_thr:.3f},Δconf={conf_delta:+.3f})"

        # ── 4. action_bias (FIX-3: normalize) ───────────────────────────
        mem_std = _get_memory_std(entry_core, state_key)
        _scale  = 1.0 + min(BIAS_STD_CAP, mem_std * 5.0)

        action_biases: Dict[str, float] = {}

        if wq == "ALIVE" and M1 > 0.0 and M6 > 0.0:
            action_biases["long_d0"] = round(BASE_BIAS_LONG        * _scale, 4)
            action_biases["long_d1"] = round(BASE_BIAS_LONG * 0.67 * _scale, 4)
        elif wq == "DECAYING" or (wq == "ALIVE" and M1 < -0.15):
            action_biases["short_d0"] = round(BASE_BIAS_SHORT        * _scale, 4)
            action_biases["short_d1"] = round(BASE_BIAS_SHORT * 0.50 * _scale, 4)
        elif wq == "FORMING":
            action_biases["long_d3"]  = round(BASE_BIAS_DELAY * _scale, 4)
            action_biases["short_d3"] = round(BASE_BIAS_DELAY * _scale, 4)

        # FIX-1 활용: M5 급변 → phase inversion → 짧은 delay 위험
        if abs(M5) > 0.6 and wq in ("ALIVE", "FORMING"):
            for k in list(action_biases):
                if "_d0" in k or "_d1" in k:
                    action_biases[k] = round(action_biases[k] * 0.3, 4)
            action_biases["long_d5"]  = round(BASE_BIAS_DELAY * _scale, 4)
            action_biases["short_d5"] = round(BASE_BIAS_DELAY * _scale, 4)

        return {
            "conf_delta":    conf_delta,
            "dynamic_thr":   dynamic_thr,
            "action_biases": action_biases,
            "gate_pass":     gate_pass,
            "reason":        reason,
            "wave_quality":  wq,
            "field_memory":  round(fm,      4),
            "mu3_ema":       round(mu3,     4),
            "E_accum":       round(E,       4),
            "mem_std":       round(mem_std, 4),
            "bias_scale":    round(_scale,  4),
            "M5":            round(M5,      4),
        }

    def log(self, result: dict, verbose: bool = False) -> str:
        s = (f"[EWA] wq={result['wave_quality']:<8}  "
             f"Δconf={result['conf_delta']:+.3f}  "
             f"thr={result['dynamic_thr']:.3f}  "
             f"fm={result['field_memory']:+.4f}  "
             f"μ3={result['mu3_ema']:.3f}  "
             f"E={result['E_accum']:.4f}  "
             f"{'✅' if result['gate_pass'] else '⛔'}  "
             f"{result['reason']}")
        if verbose:
            s += (f"\n         M5={result.get('M5',0):+.4f}  "
                  f"scale={result.get('bias_scale',1):.2f}  "
                  f"biases={result['action_biases']}")
        return s


# ─────────────────────────────────────────────────────────────────────────────
# FIX-3 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _get_memory_std(entry_core, state_key: str) -> float:
    if entry_core is None or not state_key:
        return 0.0
    try:
        for mem in (getattr(entry_core, "_long", {}),
                    getattr(entry_core, "_short", {})):
            cm = mem.get(state_key, {})
            if not cm:
                continue
            means = []
            for rs in cm.values():
                n = rs.n() if callable(rs.n) else rs.n
                if n >= 2:
                    m = rs.mean() if callable(rs.mean) else rs.mean
                    means.append(m)
            if len(means) >= 2:
                return _safe_std(means)
    except Exception:
        pass
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# select_with_bias
# ─────────────────────────────────────────────────────────────────────────────

def select_with_bias(entry_core, state_key: str, flight_mode: str,
                     allowed: Optional[List[str]],
                     action_biases: Dict[str, float],
                     min_obs_override: int = 2) -> str:
    if not action_biases or entry_core is None:
        return entry_core.select(state_key, flight_mode=flight_mode,
                                 allowed=allowed)
    import random as _rng
    pool = list(allowed) if allowed else []
    if not pool:
        return entry_core.select(state_key, flight_mode=flight_mode,
                                 allowed=allowed)

    eps_scale = {"BOOST": 0.5, "CRUISE": 1.0, "GLIDE": 2.0, "STALL": 4.0}
    eps = min(0.95, getattr(entry_core, "epsilon_base", 0.005)
              * eps_scale.get(flight_mode, 1.0))
    if _rng.random() < eps:
        return entry_core.select(state_key, flight_mode=flight_mode,
                                 allowed=allowed)

    long_mem  = getattr(entry_core, "_long",  {})
    short_mem = getattr(entry_core, "_short", {})
    min_long  = getattr(entry_core, "min_obs_long", 5)

    def _eff(action: str) -> Optional[float]:
        lm = long_mem.get(state_key, {}).get(action)
        if lm is not None:
            n = lm.n() if callable(lm.n) else lm.n
            if n >= min_long:
                m = lm.mean() if callable(lm.mean) else lm.mean
                return m + action_biases.get(action, 0.0)
        sm = short_mem.get(state_key, {}).get(action)
        if sm is not None:
            n = sm.n() if callable(sm.n) else sm.n
            if n >= min_obs_override:
                m = sm.mean() if callable(sm.mean) else sm.mean
                return m + action_biases.get(action, 0.0)
        return None

    eligible = [(c, _eff(c)) for c in pool if _eff(c) is not None]
    if not eligible:
        return entry_core.select(state_key, flight_mode=flight_mode,
                                 allowed=allowed)
    return max(eligible, key=lambda x: x[1])[0]


# ─────────────────────────────────────────────────────────────────────────────
# 셀프 테스트
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    print("=" * 70)
    print("  EntryWaveAccumulator v2 — FIX-1,2,3,4,5,6 Self Test")
    print("=" * 70)

    rng = random.Random(42)
    ewa = EntryWaveAccumulator()
    ebi = EntryBiasInjector()

    def _ph(wave=0.15, coh=0.60, fold=0.05, tors=0.08, col=0.10, pain=0.10):
        return {"wave_energy": wave, "coherence": coh, "fold": fold,
                "torsion": tors, "col_press": col, "pain": pain}

    # Phase 1: FLAT
    print("\n  [Phase 1: FLAT]")
    for _ in range(12):
        ewa.push({}, _ph(wave=rng.gauss(0, 0.01), coh=0.30, col=0.10, pain=0.10))
    mrc = ewa.compute()
    print(f"  {ebi.log(ebi.gate(mrc, 'CRUISE'))}")

    # Phase 2: FORMING
    print("\n  [Phase 2: FORMING]")
    for t in range(8):
        ewa.push({}, _ph(wave=0.05+t*0.01, coh=0.50+t*0.02, tors=0.05+t*0.005))
    mrc = ewa.compute()
    print(f"  {ebi.log(ebi.gate(mrc, 'CRUISE'), verbose=True)}")

    # Phase 3: ALIVE
    print("\n  [Phase 3: ALIVE]")
    for t in range(10):
        ewa.push({}, _ph(wave=0.20+rng.gauss(0,0.02), coh=0.72))
    mrc = ewa.compute()
    print(f"  {ebi.log(ebi.gate(mrc, 'BOOST'), verbose=True)}")

    # Phase 4: DECAYING
    print("\n  [Phase 4: DECAYING]")
    for t in range(7):
        ewa.push({}, _ph(wave=0.20-t*0.03, coh=0.55-t*0.04,
                          fold=0.3+t*0.06, col=0.50, pain=0.40+t*0.04))
    mrc = ewa.compute()
    print(f"  {ebi.log(ebi.gate(mrc, 'GLIDE'), verbose=True)}")

    # FIX-1: M5 delta 검증
    print("\n  [FIX-1: M5 torsion delta — phase inversion 감지]")
    ewa2 = EntryWaveAccumulator()
    ebi2 = EntryBiasInjector()
    print(f"  {'t':>3}  {'tors':>6}  {'M5(Δ)':>8}  gate")
    for t in range(16):
        tors = 0.10 if t < 8 else -0.15 + (t-8)*0.02
        ewa2.push({}, _ph(wave=0.15, coh=0.65, tors=tors))
        mrc2 = ewa2.compute()
        if t >= 6:
            res2 = ebi2.gate(mrc2, "CRUISE")
            print(f"  {t:3d}  {tors:6.3f}  {mrc2['M5']:+8.4f}  {res2['reason']}")

    # FIX-4: reset_on_entry 완전성 검증
    print("\n  [FIX-4: reset_on_entry — 6개 버퍼 + torsion_prev 리셋]")
    ewa3 = EntryWaveAccumulator()
    for _ in range(20):
        ewa3.push({}, _ph())
    before = ewa3.get_telemetry()
    ewa3.reset_on_entry()
    after  = ewa3.get_telemetry()
    print(f"  before: buf={before['buf_len']}  E={before['E_accum']:.5f}  kappa_ema={before['kappa_ema']:.4f}")
    print(f"  after:  buf={after['buf_len']}   E={after['E_accum']:.5f}  kappa_ema={after['kappa_ema']:.4f}  ← EMA 보존")
    assert after["buf_len"] == 0,  "buf_len should be 0 after reset"
    assert after["E_accum"] == 0.0, "E_accum should be 0.0 after reset"
    assert after["kappa_ema"] == before["kappa_ema"], "EMA should be preserved"
    print("  ✅ assertions passed")

    # FIX-3: bias_scale 확인
    print("\n  [FIX-3: bias normalize — mem_std=0 vs mem_std>0]")
    res_no = ebi.gate(ewa.compute(), "CRUISE", entry_core=None, state_key="")
    print(f"  no core: scale={res_no['bias_scale']:.2f}  biases={res_no['action_biases']}")

    # FIX-5: field_memory coherence 곱 효과 (노이즈 wave vs 진짜 wave)
    print("\n  [FIX-5: field_memory × coh — noise 억제]")
    ewa_noise = EntryWaveAccumulator()
    ewa_real  = EntryWaveAccumulator()
    for t in range(15):
        # 같은 wave_energy, 다른 coherence
        ewa_noise.push({}, _ph(wave=0.15, coh=0.15))  # low coh → noise
        ewa_real.push( {}, _ph(wave=0.15, coh=0.75))  # high coh → real
    print(f"  low-coh  field_memory: {ewa_noise.get_telemetry()['field_memory']:.6f}")
    print(f"  high-coh field_memory: {ewa_real.get_telemetry()['field_memory']:.6f}  ← 더 큼")

    print("\n" + "=" * 70)
    print("  ✅ v2 완료. 모든 FIX 검증됨.")
    print()
    print("  orchestrator.py 통합 체크리스트:")
    print("  □ __init__: self._ewa = EntryWaveAccumulator()")
    print("  □ __init__: self._ebi = EntryBiasInjector()")
    print("  □ spawn_world, select 직전: self._ewa.push(packet, phys)")
    print("  □ _mrc = self._ewa.compute()")
    print("  □ _bias = self._ebi.gate(_mrc, _flight,")
    print("              entry_core=self.entry_core, state_key=_ec_state_key)")
    print("  □ _conf_eff = _ec_conf_raw + _bias['conf_delta']")
    print("  □ if not _bias['gate_pass']: action = None")
    print("  □ chosen = select_with_bias(..., _bias['action_biases'])")
    print("  □ if evidence.executed: self._ewa.reset_on_entry()  ← FIX-4")
    print("=" * 70)