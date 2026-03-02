"""
windmill_casebook.py — 병렬 윈드밀 + 판례 저장소

설계 원칙:
  판단 ❌  |  관측 + 기록 + 물성 수정 ✅

5개 윈드밀 (병렬):
  WM_G: Grammar 상태 전이 (coh 기반)
  WM_S: Shadow stall/curvature (rcb, amp 기반)
  WM_M: MFE tail shape (mfe_slope 기반)
  WM_C: conf/vitality trajectory (conf_rel, vitality 기반)
  WM_T: 시간 비율 (elapsed / t_cut 기반)

출력:
  death_pressure ∈ [0, 1]  → Orchestrator 세계 물성 수정용
  trajectory_signature      → Casebook 저장 + 검색용

연결 위치:
  orchestrator.py MFE tracker 업데이트 직후 → wm_trace.record()
  Exit 확정 직후 → wm_trace.finalize() + casebook.save()
  다음 진입 on_entry 직후 → casebook.retrieve_verdict()
"""

import json
import os
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────
# 단일 프레임 (매 tick 기록)
# ─────────────────────────────────────────────────────────────────

class WindmillFrame:
    __slots__ = ("tick", "coh", "rcb", "amp", "mfe_slope",
                 "conf_rel", "vitality", "elapsed_ratio")

    def __init__(self, tick, coh, rcb, amp, mfe_slope,
                 conf_rel, vitality, elapsed_ratio):
        self.tick          = tick
        self.coh           = coh
        self.rcb           = rcb
        self.amp           = amp
        self.mfe_slope     = mfe_slope
        self.conf_rel      = conf_rel
        self.vitality      = vitality
        self.elapsed_ratio = elapsed_ratio  # elapsed / t_cut ∈ [0, 1+]


# ─────────────────────────────────────────────────────────────────
# 5개 윈드밀 vote 계산
# ─────────────────────────────────────────────────────────────────

def _wm_g_vote(frames: List[WindmillFrame]) -> float:
    """
    WM_G: Grammar coherence 전이 감지
    coh가 EARLY에서 LATE로 가며 하락하면 vote↑
    coh가 유지/상승하면 vote↓
    """
    if len(frames) < 3:
        return 0.0
    n = len(frames)
    early_coh = sum(f.coh for f in frames[:n//3]) / max(1, n//3)
    late_coh  = sum(f.coh for f in frames[2*n//3:]) / max(1, n - 2*n//3)
    drop = early_coh - late_coh
    return max(0.0, min(1.0, drop * 2.5))  # 0.4 drop → vote=1.0


def _wm_s_vote(frames: List[WindmillFrame]) -> float:
    """
    WM_S: Shadow stall 감지
    rcb가 낮고 amp가 수축하면 vote↑
    """
    if len(frames) < 3:
        return 0.0
    recent = frames[-max(3, len(frames)//4):]
    avg_rcb = sum(f.rcb for f in recent) / len(recent)
    avg_amp = sum(f.amp for f in recent) / len(recent)
    # rcb < 0 이고 amp < 0.4 이면 stall
    stall_score = max(0.0, -avg_rcb) * max(0.0, (0.4 - avg_amp) * 5)
    return max(0.0, min(1.0, stall_score))


def _wm_m_vote(frames: List[WindmillFrame]) -> float:
    """
    WM_M: MFE tail shape 감지
    mfe_slope가 peak 이후 지속 하락하면 vote↑
    """
    if len(frames) < 4:
        return 0.0
    slopes = [f.mfe_slope for f in frames]
    peak_idx = max(range(len(slopes)), key=lambda i: slopes[i])
    # peak 이후 구간
    post_peak = slopes[peak_idx:]
    if len(post_peak) < 2:
        return 0.0
    # 하락 일관성: 연속 음수 비율
    declining = sum(1 for i in range(1, len(post_peak))
                    if post_peak[i] < post_peak[i-1])
    ratio = declining / max(1, len(post_peak) - 1)
    # peak 이후 낙폭
    drop = max(0.0, slopes[peak_idx] - slopes[-1])
    return max(0.0, min(1.0, ratio * 0.6 + min(1.0, drop * 3) * 0.4))


def _wm_c_vote(frames: List[WindmillFrame]) -> float:
    """
    WM_C: conf_rel + vitality 동반 하락 감지
    둘 다 하락하면 vote↑ / 하나라도 유지되면 vote↓
    """
    if len(frames) < 4:
        return 0.0
    n = len(frames)
    e_conf = sum(f.conf_rel for f in frames[:n//3]) / max(1, n//3)
    l_conf = sum(f.conf_rel for f in frames[2*n//3:]) / max(1, n - 2*n//3)
    e_vit  = sum(f.vitality for f in frames[:n//3]) / max(1, n//3)
    l_vit  = sum(f.vitality for f in frames[2*n//3:]) / max(1, n - 2*n//3)
    conf_drop = max(0.0, e_conf - l_conf)
    vit_drop  = max(0.0, e_vit  - l_vit)
    return max(0.0, min(1.0, (conf_drop + vit_drop) * 1.5))


def _wm_t_vote(frames: List[WindmillFrame]) -> float:
    """
    WM_T: 시간 비율 기반 — T_cut 초과 구간에서 vote↑
    elapsed_ratio > 1.0 이면 시간 초과, 빠른 exit 신호
    """
    if not frames:
        return 0.0
    last_ratio = frames[-1].elapsed_ratio
    # 0.8 이하 → 0  /  1.0 → 0.5  /  1.2+ → 1.0
    return max(0.0, min(1.0, (last_ratio - 0.8) * 2.5))


def compute_death_pressure(frames: List[WindmillFrame]) -> dict:
    """
    5개 윈드밀 vote → death_pressure 합성
    Returns: {"death_pressure": float, "votes": dict}
    """
    votes = {
        "wm_g": _wm_g_vote(frames),
        "wm_s": _wm_s_vote(frames),
        "wm_m": _wm_m_vote(frames),
        "wm_c": _wm_c_vote(frames),
        "wm_t": _wm_t_vote(frames),
    }
    # 가중 평균 — WM_M, WM_C가 핵심 신호
    weights = {"wm_g": 0.15, "wm_s": 0.20, "wm_m": 0.30, "wm_c": 0.25, "wm_t": 0.10}
    dp = sum(votes[k] * weights[k] for k in votes)
    return {
        "death_pressure": round(dp, 4),
        "votes":          {k: round(v, 4) for k, v in votes.items()},
    }


# ─────────────────────────────────────────────────────────────────
# trajectory_signature 생성
# ─────────────────────────────────────────────────────────────────

def _discretize(value: float, bins: int, lo: float, hi: float) -> int:
    """값을 [0, bins-1] 정수로 이산화."""
    ratio = (value - lo) / max(hi - lo, 1e-9)
    return max(0, min(bins - 1, int(ratio * bins)))


def make_signature(frames: List[WindmillFrame]) -> str:
    """
    프레임 시퀀스 → 압축된 trajectory_signature 문자열
    형식: "E3+- → M2+- → L1--"
    각 토큰: Phase(1) + conf_bin(0-4) + drift_dir(+/-) + rcb_sign(+/-)
    연속 동일 토큰은 run-length 압축
    """
    if not frames:
        return "EMPTY"

    n = len(frames)
    tokens = []
    for i, f in enumerate(frames):
        # Phase
        ratio = i / max(n - 1, 1)
        phase = "E" if ratio < 0.33 else ("M" if ratio < 0.67 else "L")
        # conf_rel bin [0,4]
        cb = _discretize(f.conf_rel, 5, 0.0, 1.0)
        # drift: coh 변화 방향
        if i > 0:
            d = "+" if f.coh >= frames[i-1].coh else "-"
        else:
            d = "+" if f.coh >= 0.5 else "-"
        # rcb 부호
        r = "+" if f.rcb >= 0 else "-"
        tokens.append(f"{phase}{cb}{d}{r}")

    # run-length 압축
    compressed = [tokens[0]]
    for t in tokens[1:]:
        if t != compressed[-1]:
            compressed.append(t)
    return " → ".join(compressed)


# ─────────────────────────────────────────────────────────────────
# WindmillTrace: 포지션 단위 프레임 수집기
# ─────────────────────────────────────────────────────────────────

class WindmillTrace:
    """
    포지션 진입 시 생성, 매 tick record(), Exit 시 finalize().

    사용 패턴 (orchestrator.py):
        # 진입 확정 시
        self.wm_trace = WindmillTrace(t_cut=t_cut)

        # MFE tracker 업데이트 직후 (매 tick)
        if self.wm_trace is not None and self._mfe_tracker is not None:
            self.wm_trace.record(
                tick=self._worlds_spawned,
                orbit_mod=self._last_orbit_mod,
                mfe_slope=self._mfe_tracker.get_slope(),
                conf_rel=self._last_entry_conf,
                vitality=...,   # 아래 참고
            )

        # Exit 확정 후
        if self.wm_trace is not None:
            wm_result = self.wm_trace.finalize()
            casebook.save(state_key, action, exit_reason, realized_R, wm_result)
            self.wm_trace = None
    """

    def __init__(self, t_cut: float = 180.0):
        self._frames: List[WindmillFrame] = []
        self._t_cut = max(t_cut, 1.0)
        self._entry_tick: Optional[int] = None

    def record(self,
               tick:       int,
               orbit_mod:  dict,
               mfe_slope:  float,
               conf_rel:   float,
               vitality:   float) -> None:
        """매 tick 호출. 실패해도 조용히 넘어간다."""
        try:
            if self._entry_tick is None:
                self._entry_tick = tick
            elapsed = (tick - self._entry_tick) * 5.0  # poll_sec≈5s 가정, 외부에서 보정 가능
            elapsed_ratio = elapsed / self._t_cut

            coh = float(orbit_mod.get("coherence", 0.5))
            rcb = float(orbit_mod.get("rcb", 0.0))
            amp = float(orbit_mod.get("amplitude", 0.5))

            self._frames.append(WindmillFrame(
                tick          = tick,
                coh           = coh,
                rcb           = rcb,
                amp           = amp,
                mfe_slope     = float(mfe_slope),
                conf_rel      = max(0.0, min(1.0, float(conf_rel))),
                vitality      = max(0.0, min(1.0, float(vitality))),
                elapsed_ratio = elapsed_ratio,
            ))
        except Exception:
            pass

    def record_with_elapsed(self,
                            elapsed_sec: float,
                            orbit_mod:   dict,
                            mfe_slope:   float,
                            conf_rel:    float,
                            vitality:    float) -> None:
        """elapsed_sec를 직접 받는 변형 — poll_sec 의존성 제거."""
        try:
            elapsed_ratio = elapsed_sec / self._t_cut
            coh = float(orbit_mod.get("coherence", 0.5))
            rcb = float(orbit_mod.get("rcb", 0.0))
            amp = float(orbit_mod.get("amplitude", 0.5))
            tick = len(self._frames)

            self._frames.append(WindmillFrame(
                tick          = tick,
                coh           = coh,
                rcb           = rcb,
                amp           = amp,
                mfe_slope     = float(mfe_slope),
                conf_rel      = max(0.0, min(1.0, float(conf_rel))),
                vitality      = max(0.0, min(1.0, float(vitality))),
                elapsed_ratio = elapsed_ratio,
            ))
        except Exception:
            pass

    def finalize(self) -> dict:
        """
        포지션 종료 시 호출.
        Returns: {
            "signature":       str,
            "death_pressure":  float,
            "votes":           dict,
            "n_frames":        int,
        }
        """
        if not self._frames:
            return {
                "signature":      "EMPTY",
                "death_pressure": 0.0,
                "votes":          {},
                "n_frames":       0,
            }
        sig    = make_signature(self._frames)
        dp_out = compute_death_pressure(self._frames)
        return {
            "signature":      sig,
            "death_pressure": dp_out["death_pressure"],
            "votes":          dp_out["votes"],
            "n_frames":       len(self._frames),
        }

    @property
    def current_pressure(self) -> float:
        """포지션 중 실시간 death_pressure (Exit 전 참조용)."""
        if len(self._frames) < 3:
            return 0.0
        return compute_death_pressure(self._frames)["death_pressure"]

    @property
    def n_frames(self) -> int:
        return len(self._frames)


# ─────────────────────────────────────────────────────────────────
# WindmillCasebook: 판례 저장 + 검색
# ─────────────────────────────────────────────────────────────────

class WindmillCasebook:
    """
    사례 저장소. 판결은 "물성 수정값"만 반환한다.

    verdict ∈ {
        "EXIT_TIGHT":    T_cut × 0.80  (유사 사례 tail_rate ≥ 0.40)
        "HOLD_EXTEND":   T_cut × 1.20  (유사 사례 median_R > 0.10, survive ≥ 0.60)
        "NEUTRAL":       변경 없음
    }

    사용 패턴:
        # 진입 after on_entry
        verdict = casebook.retrieve_verdict(
            current_partial_sig="",   # 진입 직후는 NEUTRAL
            state_key=state_key,
        )
        t_cut = int(t_cut * verdict["t_cut_factor"])

        # Exit 후
        casebook.save(
            state_key  = "UP_HVOL_VOLATILE",
            action     = "short_d10",
            exit_reason= "MFE_SLOPE_CUT",
            realized_R = -0.366,
            wm_result  = wm_trace.finalize(),
        )
    """

    VERDICT_T_CUT_FACTOR = {
        "EXIT_TIGHT":  0.80,
        "HOLD_EXTEND": 1.20,
        "NEUTRAL":     1.00,
    }

    def __init__(self, state_dir: str = ".", max_cases: int = 2000):
        self._state_dir  = state_dir
        self._max_cases  = max_cases
        self._cases: List[dict] = []
        self._case_counter = 0
        self._path = os.path.join(state_dir, "windmill_casebook.json")
        self._load()

    def _load(self):
        try:
            if os.path.exists(self._path):
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._cases = data.get("cases", [])
                self._case_counter = data.get("counter", len(self._cases))
                print(f"  [Casebook] loaded {len(self._cases)} cases ← {self._path}")
        except Exception as e:
            print(f"  [Casebook] load failed: {e} — starting fresh")
            self._cases = []

    def save_to_disk(self):
        try:
            os.makedirs(self._state_dir, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump({
                    "counter": self._case_counter,
                    "cases":   self._cases[-self._max_cases:],
                }, f, ensure_ascii=False, indent=None)
        except Exception as e:
            print(f"  [Casebook] save failed: {e}")

    def add_case(self,
                 state_key:   str,
                 action:      str,
                 exit_reason: str,
                 realized_R:  float,
                 wm_result:   dict) -> str:
        """
        사례 추가. case_id 반환.
        wm_result = WindmillTrace.finalize() 출력
        """
        self._case_counter += 1
        case_id = f"C{self._case_counter:05d}"

        case = {
            "case_id":        case_id,
            "state_key":      state_key,
            "action":         action,
            "exit_reason":    exit_reason,
            "realized_R":     round(float(realized_R), 4),
            "signature":      wm_result.get("signature", "EMPTY"),
            "death_pressure": wm_result.get("death_pressure", 0.0),
            "votes":          wm_result.get("votes", {}),
            "n_frames":       wm_result.get("n_frames", 0),
            "survived":       float(realized_R) > -0.5,
        }
        self._cases.append(case)

        # 용량 제한
        if len(self._cases) > self._max_cases:
            self._cases = self._cases[-self._max_cases:]

        # 10건마다 자동 저장
        if self._case_counter % 10 == 0:
            self.save_to_disk()

        return case_id

    def retrieve_verdict(self,
                         state_key:        str,
                         current_sig:      str = "",
                         k:                int = 5,
                         min_n:            int = 3) -> dict:
        """
        현재 state_key + signature와 유사한 사례 검색 → 판결 반환.

        Returns: {
            "verdict":       "EXIT_TIGHT" | "HOLD_EXTEND" | "NEUTRAL",
            "t_cut_factor":  float,
            "n_used":        int,
            "median_R":      float,
            "tail_rate":     float,
            "survive_rate":  float,
            "dp_mean":       float,   # 유사 사례의 death_pressure 평균
        }
        """
        _null = {
            "verdict": "NEUTRAL", "t_cut_factor": 1.00,
            "n_used": 0, "median_R": 0.0,
            "tail_rate": 0.0, "survive_rate": 0.5, "dp_mean": 0.0,
        }

        # 1. state_key 필터
        candidates = [c for c in self._cases if c["state_key"] == state_key]
        if len(candidates) < min_n:
            return _null

        # 2. signature 유사도 (토큰 레벨 공통 prefix 비율)
        def sig_dist(a: str, b: str) -> float:
            if not a or not b:
                return 1.0
            ta = a.split(" → ")
            tb = b.split(" → ")
            match = sum(x == y for x, y in zip(ta, tb))
            return 1.0 - match / max(len(ta), len(tb), 1)

        ranked = sorted(candidates,
                        key=lambda c: sig_dist(c.get("signature", ""), current_sig))
        top_k = ranked[:k]

        if len(top_k) < min_n:
            return _null

        # 3. 집계
        rs = sorted(c["realized_R"] for c in top_k)
        median_R    = rs[len(rs) // 2]
        tail_rate   = sum(1 for c in top_k if c["realized_R"] < -0.5) / len(top_k)
        survive_rate = sum(1 for c in top_k if c["survived"]) / len(top_k)
        dp_mean      = sum(c.get("death_pressure", 0.0) for c in top_k) / len(top_k)

        # 4. 판결
        if tail_rate >= 0.40:
            verdict = "EXIT_TIGHT"
        elif median_R > 0.10 and survive_rate >= 0.60:
            verdict = "HOLD_EXTEND"
        else:
            verdict = "NEUTRAL"

        return {
            "verdict":      verdict,
            "t_cut_factor": self.VERDICT_T_CUT_FACTOR[verdict],
            "n_used":       len(top_k),
            "median_R":     round(median_R, 4),
            "tail_rate":    round(tail_rate, 4),
            "survive_rate": round(survive_rate, 4),
            "dp_mean":      round(dp_mean, 4),
        }

    @property
    def n_cases(self) -> int:
        return len(self._cases)

    def get_stats(self) -> dict:
        if not self._cases:
            return {"n_cases": 0}
        dps = [c.get("death_pressure", 0.0) for c in self._cases]
        rs  = [c["realized_R"] for c in self._cases]
        return {
            "n_cases":    len(self._cases),
            "dp_mean":    round(sum(dps) / len(dps), 4),
            "R_mean":     round(sum(rs) / len(rs), 4),
            "tail_rate":  round(sum(1 for r in rs if r < -0.5) / len(rs), 4),
            "survive_rate": round(sum(1 for c in self._cases if c["survived"]) / len(self._cases), 4),
        }
