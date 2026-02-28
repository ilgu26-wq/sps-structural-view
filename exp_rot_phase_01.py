"""
EXP-ROT-PHASE-01
─────────────────────────────────────────────────────────────────
코크 자전 모델 — 위상 정렬(Phase Alignment) + SHOCK 방향 분리

핵심 전환:
  EXP-VIDEO-ALIGN-01이 밝힌 것:
    SHOCK = 상태가 아니라 '전이 중인 궤적'
    SHOCK_RISING (dr_g > 0)  → 에너지 회복, HOLD
    SHOCK_FALLING(dr_g < 0)  → 에너지 붕괴, CUT 준비

  EXP-ROT-PHASE-01이 추가하는 것:
    매 프레임마다 자전 위상 φ(t)를 누적 → 모든 신호를 phase-aligned로 재표현
    raw trajectory vs aligned trajectory 비교
    SHOCK_FALLING 3프레임 연속 → WEAKENING_CUT 승격 시뮬레이션

물리 모델:
  시장(코크)을 rotating frame으로 본다.
  φ(t+1) = φ(t) + Ω·Δt
  Ω = 코크 자전 각속도 (coherence decay rate로 추정)
  signal_aligned = signal * cos(φ) → phase-aligned 에너지

연결 포인트:
  STBObserver  : EXHAUSTION 감지 (신뢰, 건드리지 않음)
  VideoObserver: Frame Strip + trajectory
  RotPhaseProbe: 위상 정렬 + SHOCK 방향 분리 + WEAKENING_CUT 시뮬

결론:
  사진은 상태를 맞히고, 영상은 전이를 맞힌다.
  SHOCK은 상태가 아니라 궤적이다.
"""

import math
import random
from collections import defaultdict, Counter
from typing import List, Optional, Tuple

_EPS = 1e-8

# ── SHOCK 방향 레이블 ────────────────────────────────────────────
ROTATION      = "ROTATION"
SHOCK_RISING  = "SHOCK_RISING"   # dr_g > 0 : 에너지 회복 → HOLD
SHOCK_FALLING = "SHOCK_FALLING"  # dr_g < 0 : 에너지 붕괴 → CUT 준비
EXHAUSTION    = "EXHAUSTION"     # r_g < θ  : 소진 → 즉시 CUT


# ════════════════════════════════════════════════════════════════
# PhaseFrame — 위상 정렬이 포함된 프레임
# ════════════════════════════════════════════════════════════════

class PhaseFrame:
    __slots__ = (
        "idx", "elapsed_s", "R",
        "r_g", "dr_g",
        "phase", "d_phase",
        "coh", "d_coh",
        "stb_ratio",
        # 위상 자전 필드
        "phi",          # 누적 자전 위상 φ(t)
        "r_g_aligned",  # r_g * cos(φ)  — phase-aligned 에너지
        # SHOCK 방향
        "signal",       # ROTATION / SHOCK_RISING / SHOCK_FALLING / EXHAUSTION
    )

    def __init__(self, idx, elapsed_s, R,
                 r_g, dr_g, phase, d_phase,
                 coh, d_coh, stb_ratio,
                 phi, r_g_aligned, signal):
        for k, v in zip(self.__slots__,
                        (idx, elapsed_s, R, r_g, dr_g,
                         phase, d_phase, coh, d_coh, stb_ratio,
                         phi, r_g_aligned, signal)):
            object.__setattr__(self, k, v)

    def __repr__(self):
        return (f"PF(#{self.idx} ela={self.elapsed_s:.0f}s R={self.R:+.4f} "
                f"r_g={self.r_g:.3f}({self.r_g_aligned:.3f}↦) "
                f"dr_g={self.dr_g:+.3f} φ={self.phi:.2f} [{self.signal}])")


# ════════════════════════════════════════════════════════════════
# RotPhaseProbe — 포지션 하나를 위상 정렬 Frame Strip으로 기록
# ════════════════════════════════════════════════════════════════

class RotPhaseProbe:
    """
    Parameters
    ----------
    ticks_per_frame   : int   몇 tick = 1 frame (default=5)
    theta_exh         : float EXHAUSTION r_g 임계값 (default=0.35)
    shock_dr_thresh   : float SHOCK 기준 |dr_g| (default=0.08)
    omega             : float 자전 각속도 [rad/frame] (default=0.1)
    lookback          : int   CUT 직전 판단 프레임 수 (default=7)
    falling_streak    : int   SHOCK_FALLING 연속 N프레임 → WEAKENING_CUT 승격 (default=3)
    """

    def __init__(self, pos_id, entry_tick,
                 ticks_per_frame=5,
                 theta_exh=0.35,
                 shock_dr_thresh=0.08,
                 omega=0.10,
                 lookback=7,
                 falling_streak=3):
        self.pos_id = pos_id
        self.entry_tick = entry_tick
        self.ticks_per_frame = ticks_per_frame
        self.theta_exh = theta_exh
        self.shock_dr_thresh = shock_dr_thresh
        self.omega = omega
        self.lookback = lookback
        self.falling_streak = falling_streak

        # entry 기준
        self.ref_g     = _EPS
        self.ref_coh   = _EPS
        self.ref_slope = _EPS
        self.ref_curv  = _EPS
        self.ref_energy= _EPS

        # 누적 위상 (자전)
        self._phi = 0.0

        # raw ticks
        self._ticks: List[Tuple] = []
        self._cut_tick_idx: Optional[int] = None

        # frames
        self.frames: List[PhaseFrame] = []

        # 결과
        self.final_reason = "OPEN"
        self.final_R = 0.0

        # 시뮬레이션: WEAKENING_CUT 발생 여부 + 시점
        self.sim_weakening_cut_frame: Optional[int] = None

    # ── 진입 기준 ────────────────────────────────────────────────
    def set_entry(self, coh, slope, curv, rcb, energy):
        self.ref_coh    = max(abs(coh),    _EPS)
        self.ref_slope  = max(abs(slope),  _EPS)
        self.ref_curv   = max(abs(curv),   _EPS)
        self.ref_energy = max(abs(energy), _EPS)
        self.ref_g = max(
            math.sqrt(slope**2 + curv**2) * max(energy, _EPS),
            _EPS
        )

    # ── tick 기록 ────────────────────────────────────────────────
    def tick(self, elapsed_s, R, coh, slope, curv, rcb, energy,
             is_cut=False):
        idx = len(self._ticks)
        if is_cut and self._cut_tick_idx is None:
            self._cut_tick_idx = idx
        self._ticks.append((elapsed_s, R, coh, slope, curv, rcb, energy))

        if len(self._ticks) % self.ticks_per_frame == 0:
            self._build_frame()

    def close(self, reason, final_R):
        remainder = len(self._ticks) % self.ticks_per_frame
        if remainder > 0:
            self._build_frame(partial=True)
        self.final_reason = reason
        self.final_R = round(final_R, 4)
        # 시뮬: WEAKENING_CUT 탐색
        self._simulate_weakening_cut()

    # ── frame 생성 ────────────────────────────────────────────────
    def _build_frame(self, partial=False):
        n = self.ticks_per_frame
        start = len(self.frames) * n
        chunk = self._ticks[start:start + n] if not partial else self._ticks[start:]
        if not chunk:
            return

        ela   = chunk[-1][0]
        R     = chunk[-1][1]
        avg_coh    = sum(t[2] for t in chunk) / len(chunk)
        avg_slope  = sum(t[3] for t in chunk) / len(chunk)
        avg_curv   = sum(t[4] for t in chunk) / len(chunk)
        avg_energy = sum(t[6] for t in chunk) / len(chunk)

        g_now = math.sqrt(avg_slope**2 + avg_curv**2) * max(avg_energy, _EPS)
        r_g   = g_now / self.ref_g
        coh_rel = avg_coh / self.ref_coh

        if self.frames:
            prev   = self.frames[-1]
            dr_g   = r_g - prev.r_g
            d_phase= self._phase(avg_slope, avg_curv) - prev.phase
            d_coh  = coh_rel - prev.coh
        else:
            dr_g = d_phase = d_coh = 0.0

        phase = self._phase(avg_slope, avg_curv)
        stb_ratio = (
            (coh_rel +
             abs(avg_slope) / self.ref_slope +
             max(avg_energy, 0) / self.ref_energy) / 3.0
        )

        # ── 자전 위상 누적 φ(t+1) = φ(t) + Ω·1
        # Ω는 coherence decay rate에 비례 — coherence가 낮을수록 더 빠르게 자전
        omega_eff = self.omega * max(1.0 - coh_rel, 0.1)
        self._phi += omega_eff
        phi = self._phi

        # ── phase-aligned 에너지: r_g * |cos(φ)|
        # cos(φ)은 위상 정렬도 — φ=0이면 완전 정렬, φ=π/2면 직교 → 에너지 전달 0
        r_g_aligned = r_g * abs(math.cos(phi))

        # ── SHOCK 방향 분리 (핵심)
        signal = self._classify(r_g, dr_g, d_phase)

        # ── WEAKENING_CUT 시뮬: SHOCK_FALLING 연속 추적
        # (close() 이후 _simulate_weakening_cut에서 일괄 처리)

        f = PhaseFrame(
            idx        = len(self.frames),
            elapsed_s  = round(ela, 1),
            R          = round(R, 4),
            r_g        = round(r_g, 4),
            dr_g       = round(dr_g, 4),
            phase      = round(phase, 4),
            d_phase    = round(d_phase, 4),
            coh        = round(coh_rel, 4),
            d_coh      = round(d_coh, 4),
            stb_ratio  = round(stb_ratio, 4),
            phi        = round(phi, 4),
            r_g_aligned= round(r_g_aligned, 4),
            signal     = signal,
        )
        self.frames.append(f)

    @staticmethod
    def _phase(slope, curv):
        return math.atan2(abs(curv), abs(slope))

    def _classify(self, r_g, dr_g, d_phase) -> str:
        if r_g < self.theta_exh:
            return EXHAUSTION
        if abs(dr_g) > self.shock_dr_thresh or abs(d_phase) > 0.5:
            # ── SHOCK 방향 분리 ──────────────────────────────────
            return SHOCK_RISING if dr_g >= 0 else SHOCK_FALLING
        return ROTATION

    def _simulate_weakening_cut(self):
        """SHOCK_FALLING N프레임 연속 → WEAKENING_CUT 시뮬 (로직 변경 없이 관측만)"""
        streak = 0
        for f in self.frames:
            if f.signal == SHOCK_FALLING:
                streak += 1
                if streak >= self.falling_streak:
                    self.sim_weakening_cut_frame = f.idx
                    return
            else:
                streak = 0

    # ── 궤적 분석 ────────────────────────────────────────────────
    def analyze(self) -> dict:
        if not self.frames:
            return {"pos_id": self.pos_id, "error": "no frames"}

        cut_fi = (self._cut_tick_idx // self.ticks_per_frame
                  if self._cut_tick_idx is not None else None)

        # 스냅샷(사진) — CUT 프레임 단독
        sf = self.frames[cut_fi] if (cut_fi is not None and cut_fi < len(self.frames)) else self.frames[-1]
        snapshot_zone = sf.signal

        # 궤적(영상) — CUT 직전 lookback 프레임
        if cut_fi is not None:
            window = self.frames[max(0, cut_fi - self.lookback):cut_fi]
        else:
            window = self.frames[-self.lookback:]

        if window:
            cnt = Counter(f.signal for f in window)
            trajectory_zone = cnt.most_common(1)[0][0]
            avg_dr_g        = sum(f.dr_g        for f in window) / len(window)
            avg_phi         = sum(f.phi         for f in window) / len(window)
            avg_aligned     = sum(f.r_g_aligned for f in window) / len(window)
            falling_count   = cnt.get(SHOCK_FALLING, 0)
        else:
            trajectory_zone = snapshot_zone
            avg_dr_g = avg_phi = avg_aligned = 0.0
            falling_count = 0

        mismatch = (snapshot_zone != trajectory_zone)

        # WEAKENING_CUT 시뮬 — 실제 exit vs 시뮬 타이밍
        wc_frame = self.sim_weakening_cut_frame
        wc_R = None
        if wc_frame is not None and wc_frame < len(self.frames):
            wc_R = self.frames[wc_frame].R

        r_recovered = round(self.final_R - wc_R, 4) if wc_R is not None else None

        return dict(
            pos_id            = self.pos_id,
            final_reason      = self.final_reason,
            final_R           = self.final_R,
            n_frames          = len(self.frames),
            cut_frame_idx     = cut_fi,
            # 사진
            snapshot_zone     = snapshot_zone,
            snapshot_r_g      = round(sf.r_g, 4),
            snapshot_aligned  = round(sf.r_g_aligned, 4),
            # 영상
            trajectory_zone   = trajectory_zone,
            traj_avg_dr_g     = round(avg_dr_g, 4),
            traj_avg_phi      = round(avg_phi, 4),
            traj_avg_aligned  = round(avg_aligned, 4),
            traj_falling_cnt  = falling_count,
            mismatch          = mismatch,
            # WEAKENING_CUT 시뮬
            wc_frame          = wc_frame,
            wc_R              = wc_R,
            r_recovered       = r_recovered,  # 양수 = 조기 CUT이 더 좋았음
        )

    def print_strip(self, mark_cut=True):
        cut_fi = (self._cut_tick_idx // self.ticks_per_frame
                  if self._cut_tick_idx is not None else None)
        wc_fi = self.sim_weakening_cut_frame
        print(f"\n  ── {self.pos_id} ({self.final_reason}, R={self.final_R:+.4f}) ──")
        print(f"  {'#':>3} | {'ela':>6} | {'R':>7} | {'r_g':>5} | {'r_g↦':>5} | "
              f"{'dr_g':>6} | {'φ':>5} | {'signal':<14}")
        for f in self.frames:
            m = ""
            if mark_cut and f.idx == wc_fi:  m += "⚡WC"
            if mark_cut and f.idx == cut_fi: m += " ◀ACT"
            sig_colored = {
                ROTATION:      "ROTATION      ",
                SHOCK_RISING:  "SHOCK_RISING↑ ",
                SHOCK_FALLING: "SHOCK_FALLING↓",
                EXHAUSTION:    "EXHAUSTION    ",
            }.get(f.signal, f.signal)
            print(f"  {f.idx:>3} | {f.elapsed_s:>6.0f} | {f.R:>+7.4f} | "
                  f"{f.r_g:>5.3f} | {f.r_g_aligned:>5.3f} | "
                  f"{f.dr_g:>+6.3f} | {f.phi:>5.2f} | {sig_colored}{m}")


# ════════════════════════════════════════════════════════════════
# RotPhaseObserver — 라이브 연결용 (STBObserver / VideoObserver 동일 패턴)
# ════════════════════════════════════════════════════════════════

class RotPhaseObserver:
    """
    라이브 연결:
        from exp_rot_phase_01 import RotPhaseObserver
        self._rp_obs = RotPhaseObserver(self, out_dir=self.state_dir)
        # on_entry() / on_tick(tick, is_cut=...) / on_exit(reason, R)
        # → .analyze() / .weakening_cut_report()

    STBObserver와 병렬 운용 (독립적):
        self._stb_obs  = STBObserver(self, ...)
        self._rp_obs   = RotPhaseObserver(self, ...)
    """

    def __init__(self, runner, out_dir=".",
                 ticks_per_frame=5, theta_exh=0.35,
                 shock_dr_thresh=0.08, omega=0.10,
                 lookback=7, falling_streak=3):
        self.runner = runner
        self.out_dir = out_dir
        self.cfg = dict(
            ticks_per_frame=ticks_per_frame,
            theta_exh=theta_exh,
            shock_dr_thresh=shock_dr_thresh,
            omega=omega,
            lookback=lookback,
            falling_streak=falling_streak,
        )
        self._probe: Optional[RotPhaseProbe] = None
        self._probes: List[RotPhaseProbe] = []
        self._tick_n = 0
        import os; os.makedirs(out_dir, exist_ok=True)

    def on_entry(self):
        self._probe = RotPhaseProbe(f"pos_{self._tick_n}", self._tick_n, **self.cfg)
        self._probe.set_entry(
            self._coh(), self._slope(), self._curv(), self._rcb(), self._energy()
        )

    def on_tick(self, tick, is_cut=False):
        self._tick_n += 1
        if self._probe is None: return
        pos = getattr(self.runner, "position", None)
        if pos is None: return
        from datetime import datetime
        elapsed = pos.elapsed_sec(tick.get("dt", datetime.now())) if hasattr(pos, "elapsed_sec") else 0.0
        R = pos.pnl_r(tick.get("close", 0.0)) if hasattr(pos, "pnl_r") else 0.0
        self._probe.tick(elapsed, R, self._coh(), self._slope(),
                         self._curv(), self._rcb(), self._energy(), is_cut=is_cut)

    def on_exit(self, reason, final_R):
        if self._probe is None: return
        self._probe.close(reason, final_R)
        self._probes.append(self._probe)
        rec = self._probe.analyze()
        wc  = f"wc_frame={rec['wc_frame']} r_rec={rec['r_recovered']:+.4f}" if rec.get("wc_frame") is not None else "no_wc"
        mm  = "⚡MISMATCH" if rec.get("mismatch") else "─"
        print(f"  [RP] {rec['pos_id']} {reason} R={final_R:+.4f} "
              f"snap={rec['snapshot_zone'][:3]} traj={rec['trajectory_zone'][:3]} {mm} | {wc}")
        self._probe = None

    def _coh(self):
        try: return float((self.runner.orch._last_orbit_mod or {}).get("coherence", 0.5))
        except: return 0.5
    def _slope(self):
        try: tr = getattr(self.runner.orch, "_mfe_tracker", None); return float(tr.get_slope()) if tr else 0.0
        except: return 0.0
    def _curv(self):
        try: tr = getattr(self.runner.orch, "_mfe_tracker", None); return float(tr.get_curvature()) if tr else 0.0
        except: return 0.0
    def _rcb(self):
        try: se = getattr(self.runner.orch, "shadow_exit", None); return float(se.rcb_meter.rcb) if se else 0.0
        except: return 0.0
    def _energy(self):
        try:
            sp = getattr(self.runner, "_spinal", {})
            sk = getattr(self.runner, "_last_ec_state_key", "")
            ac = getattr(self.runner, "_last_ec_chosen", "")
            return float(sp.get(sk, {}).get(ac, {}).get("energy", 0.0))
        except: return 0.0

    def analyze(self, verbose=False):
        recs = [p.analyze() for p in self._probes]
        print_phase_analysis(recs, self.cfg)
        if verbose:
            for p in self._probes: p.print_strip()
        return recs

    def weakening_cut_report(self):
        recs = [p.analyze() for p in self._probes]
        wc = [r for r in recs if r.get("wc_frame") is not None]
        print(f"\n  [RP] WEAKENING_CUT 시뮬: {len(wc)}/{len(recs)}")
        benefit = [r for r in wc if (r.get("r_recovered") or 0) > 0]
        loss    = [r for r in wc if (r.get("r_recovered") or 0) <= 0]
        if benefit:
            avg_b = sum(r["r_recovered"] for r in benefit) / len(benefit)
            print(f"    개선: {len(benefit)}건  avg_R_recovered={avg_b:+.4f}")
        if loss:
            avg_l = sum(r["r_recovered"] for r in loss) / len(loss)
            print(f"    역효과: {len(loss)}건  avg_R_lost={avg_l:+.4f}")
        return wc


# ════════════════════════════════════════════════════════════════
# 분석 출력
# ════════════════════════════════════════════════════════════════

def print_phase_analysis(records, cfg=None):
    n = max(len(records), 1)
    n_mm = sum(1 for r in records if r.get("mismatch"))

    def _correct(zone, fR):
        if fR >= 0: return zone in (ROTATION, SHOCK_RISING)
        else:       return zone in (EXHAUSTION, SHOCK_FALLING)

    snap_c = sum(1 for r in records if _correct(r["snapshot_zone"], r["final_R"]))
    traj_c = sum(1 for r in records if _correct(r["trajectory_zone"], r["final_R"]))
    delta  = traj_c - snap_c

    # 신호별 평균 R
    sig_r = defaultdict(list)
    for r in records:
        sig_r[r["trajectory_zone"]].append(r["final_R"])

    # WEAKENING_CUT 시뮬
    wc_recs = [r for r in records if r.get("wc_frame") is not None]
    wc_benefit = [r for r in wc_recs if (r.get("r_recovered") or 0) > 0]
    sim_delta  = (sum(r["r_recovered"] for r in wc_benefit) / max(len(wc_benefit), 1)
                  if wc_benefit else 0.0)

    verdict = ("✅ 궤적 우세" if delta > 0 else ("⚠️ 동등" if delta == 0 else "❌ 스냅샷 우세"))

    print(f"\n{'═'*72}")
    print(f"  EXP-ROT-PHASE-01  n={n}" + (f"  θ={cfg['theta_exh']:.2f} ω={cfg['omega']:.2f} lb={cfg['lookback']}" if cfg else ""))
    print(f"{'═'*72}")
    print(f"  불일치 (사진≠영상):  {n_mm}/{n} ({n_mm/n*100:.1f}%)")
    print(f"  판단 정확도")
    print(f"    스냅샷(사진): {snap_c}/{n} ({snap_c/n*100:.1f}%)")
    print(f"    궤적(영상):   {traj_c}/{n} ({traj_c/n*100:.1f}%)  {verdict} (Δ={delta:+d})")
    print(f"\n  궤적 신호별 평균 R:")
    for sig in (ROTATION, SHOCK_RISING, SHOCK_FALLING, EXHAUSTION):
        rs = sig_r.get(sig, [])
        if rs:
            print(f"    {sig:<16} n={len(rs):3d}  avg_R={sum(rs)/len(rs):+.4f}")
    print(f"\n  WEAKENING_CUT 시뮬 (SHOCK_FALLING {cfg['falling_streak'] if cfg else 3}프레임 연속):")
    print(f"    발동: {len(wc_recs)}/{n}  개선: {len(wc_benefit)}건  avg_gain={sim_delta:+.4f}")
    if len(wc_recs) > 0:
        v = ("✅ 실장 권고" if sim_delta > 0.02 else ("⚠️ 조건 조정" if sim_delta >= 0 else "❌ 역효과"))
        print(f"    {v}")

    # 사진의 맹점
    print(f"\n  [증거] 동일 스냅샷 → 다른 궤적:")
    groups = defaultdict(list)
    for r in records:
        groups[r["snapshot_zone"]].append(r["trajectory_zone"])
    for snap, trajs in sorted(groups.items()):
        cnt = Counter(trajs)
        total = len(trajs)
        parts = ", ".join(f"{z[:3]}:{c}({c/total*100:.0f}%)" for z, c in sorted(cnt.items()))
        print(f"    snap={snap:<16} → {parts}")
    print(f"{'═'*72}")


# ════════════════════════════════════════════════════════════════
# ω (자전 각속도) sweep — 최적 omega 탐색
# ════════════════════════════════════════════════════════════════

def omega_sweep(probes_factory, omegas=None, lookback=7, falling_streak=3):
    if omegas is None:
        omegas = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    print(f"\n{'─'*72}")
    print(f"  ω sweep (자전 각속도 최적화)")
    print(f"{'─'*72}")
    print(f"  {'ω':>5} | {'traj_acc%':>10} | {'snap_acc%':>10} | {'wc_gain':>9} |")
    print(f"  {'─'*48}")

    best_ω, best_delta = None, -999
    for ω in omegas:
        probes = probes_factory(omega=ω, lookback=lookback, falling_streak=falling_streak)
        recs = [p.analyze() for p in probes]
        n = max(len(recs), 1)

        def _c(zone, fR):
            if fR >= 0: return zone in (ROTATION, SHOCK_RISING)
            else:       return zone in (EXHAUSTION, SHOCK_FALLING)

        sc = sum(1 for r in recs if _c(r["snapshot_zone"], r["final_R"]))
        tc = sum(1 for r in recs if _c(r["trajectory_zone"], r["final_R"]))

        wc_b = [r for r in recs if r.get("wc_frame") is not None and (r.get("r_recovered") or 0) > 0]
        gain = sum(r["r_recovered"] for r in wc_b) / max(len(wc_b), 1) if wc_b else 0.0

        delta = tc - sc
        tag = " ★" if delta > best_delta else ""
        if delta > best_delta:
            best_delta = delta; best_ω = ω
        print(f"  {ω:>5.2f} | {tc/n*100:>9.1f}% | {sc/n*100:>9.1f}% | {gain:>+9.4f}{tag}")

    print(f"  {'─'*48}")
    if best_ω:
        print(f"  권고 ω = {best_ω:.2f}  (traj_acc Δ={best_delta:+d})")
    print(f"{'─'*72}")


# ════════════════════════════════════════════════════════════════
# 시뮬레이션 데이터 생성 (exp_shadow_stb_01 동일 시나리오)
# ════════════════════════════════════════════════════════════════

def make_rot_probes(n=80, seed=42,
                   ticks_per_frame=5, theta_exh=0.35,
                   shock_dr_thresh=0.08, omega=0.10,
                   lookback=7, falling_streak=3):
    rng = random.Random(seed)
    emap = {
        "good":       "MAX_HOLD",
        "slow_death": "SHADOW_LINE_CUT",
        "fast_death": "GRAMMAR_CUT",
        "choppy":     "WAVE_CONVERGED",
    }
    probes = []
    for i in range(n):
        p = RotPhaseProbe(
            f"sim_{i:03d}", i * 100,
            ticks_per_frame=ticks_per_frame,
            theta_exh=theta_exh,
            shock_dr_thresh=shock_dr_thresh,
            omega=omega,
            lookback=lookback,
            falling_streak=falling_streak,
        )
        ec  = rng.uniform(.55, .85)
        es  = rng.uniform(.003, .018)
        ecv = rng.uniform(.0005, .003)
        er  = rng.uniform(.10, .70)
        ee  = rng.uniform(.30, 1.20)
        p.set_entry(ec, es, ecv, er, ee)

        pt  = rng.choice(list(emap.keys()))
        nt  = rng.randint(25, 90)
        fR  = 0.0

        for t in range(nt):
            f  = t / max(nt - 1, 1)
            el = t * 5.0

            if pt == "good":
                coh = ec*(1-.06*f); slo = es*(1-.25*f); crv = ecv*(1-.10*f)
                rcb = er*(1-.12*f); nrg = ee*(1-.18*f); R = rng.uniform(0,1)*f

            elif pt == "slow_death":
                coh = ec*max(.08,1-1.6*f); slo = es*max(.04,1-2.2*f)
                crv = ecv*max(.02,1-2.0*f); rcb = er*max(.03,1-2.5*f)
                nrg = ee*max(.04,1-2.8*f); R = rng.uniform(-1,.1)*f

            elif pt == "fast_death":
                d = max(.02,1-3.5*f)
                coh = ec*d; slo = es*max(.02,d-.3); crv = ecv*d
                rcb = er*max(.02,d-.2); nrg = ee*max(.02,d-.1)
                R = rng.uniform(-1.8,-.05)*f

            else:  # choppy
                nv = rng.gauss(0,.12)
                coh = max(.1,ec*(.75+nv)); slo = max(_EPS,es*(.80+nv))
                crv = max(_EPS,ecv*(.70+nv)); rcb = max(_EPS,er*(.85+nv))
                nrg = max(_EPS,ee*(.78+nv)); R = rng.gauss(0,.35)

            is_cut = (t == int(nt*.85)) and pt in ("slow_death","fast_death")
            p.tick(el, R, coh, slo, crv, rcb, nrg, is_cut=is_cut)
            fR = R

        p.close(emap[pt], fR)
        probes.append(p)
    return probes


# ════════════════════════════════════════════════════════════════
# 메인 실험
# ════════════════════════════════════════════════════════════════

def run_experiment():
    print("=" * 72)
    print("EXP-ROT-PHASE-01 — 코크 자전 모델 + SHOCK 방향 분리 + 위상 정렬")
    print("=" * 72)

    cfg_base = dict(ticks_per_frame=5, theta_exh=0.35,
                    shock_dr_thresh=0.08, omega=0.10,
                    lookback=7, falling_streak=3)

    # ── [1] 기본 실험 seed=42
    print("\n[1] seed=42  n=80  lookback=7  ω=0.10  falling_streak=3")
    pa = make_rot_probes(80, 42, **cfg_base)
    ra = [p.analyze() for p in pa]
    print_phase_analysis(ra, cfg_base)

    # ── [2] seed=99
    print("\n[2] seed=99  n=80")
    pb = make_rot_probes(80, 99, **cfg_base)
    rb = [p.analyze() for p in pb]
    print_phase_analysis(rb, cfg_base)

    # ── [3] ω sweep
    print("\n[3] ω sweep  seed=42  lookback=7")
    def _factory(**kw):
        cfg = dict(cfg_base); cfg.update(kw)
        return make_rot_probes(80, 42, **cfg)
    omega_sweep(_factory)

    # ── [4] falling_streak 민감도
    print("\n[4] falling_streak 민감도  seed=42  ω=0.10")
    print(f"  {'streak':>8} | {'wc_발동%':>9} | {'개선건':>7} | {'avg_gain':>9}")
    print(f"  {'─'*42}")
    for fs in [2, 3, 4, 5]:
        probes = make_rot_probes(80, 42, **dict(cfg_base, falling_streak=fs))
        recs = [p.analyze() for p in probes]
        wc  = [r for r in recs if r.get("wc_frame") is not None]
        b   = [r for r in wc if (r.get("r_recovered") or 0) > 0]
        g   = sum(r["r_recovered"] for r in b) / max(len(b), 1) if b else 0.0
        print(f"  {fs:>8} | {len(wc)/max(len(recs),1)*100:>8.1f}% | "
              f"{len(b):>7} | {g:>+9.4f}")

    # ── [5] 결합 320 trades
    print("\n[5] 결합  320 trades")
    all_p = []
    for s in [42, 99, 123, 456]:
        all_p.extend(make_rot_probes(80, s, **cfg_base))
    all_r = [p.analyze() for p in all_p]
    print_phase_analysis(all_r, cfg_base)

    # ── [6] SHOCK_RISING vs SHOCK_FALLING 분리 효과 확인
    print("\n[6] SHOCK_RISING vs SHOCK_FALLING — 평균 R 분리 확인 (n=320)")
    sr_r  = [r["final_R"] for r in all_r if r["trajectory_zone"] == SHOCK_RISING]
    sf_r  = [r["final_R"] for r in all_r if r["trajectory_zone"] == SHOCK_FALLING]
    print(f"  SHOCK_RISING  n={len(sr_r)}  avg_R={sum(sr_r)/max(len(sr_r),1):+.4f}  "
          f"(양수 비율={sum(1 for r in sr_r if r>0)/max(len(sr_r),1)*100:.0f}%)")
    print(f"  SHOCK_FALLING n={len(sf_r)}  avg_R={sum(sf_r)/max(len(sf_r),1):+.4f}  "
          f"(음수 비율={sum(1 for r in sf_r if r<0)/max(len(sf_r),1)*100:.0f}%)")
    print(f"  분리 유효성: "
          f"{'✅ RISING(+) / FALLING(-) 방향 분리 성공' if sr_r and sf_r and sum(sr_r)/len(sr_r) > sum(sf_r)/len(sf_r) else '⚠️ 추가 조정 필요'}")

    # ── [7] Frame Strip — WEAKENING_CUT 발동 케이스 샘플
    print("\n[7] WEAKENING_CUT 발동 케이스 Frame Strip (최대 2개)")
    wc_probes = [p for p in pa if p.sim_weakening_cut_frame is not None
                 and (pa[0].analyze().get("r_recovered") or 0) > 0][:2]
    if not wc_probes:
        wc_probes = [p for p in pa if p.sim_weakening_cut_frame is not None][:2]
    for p in wc_probes:
        p.print_strip()

    # ── LIVE 연결 안내
    print("\n" + "─" * 72)
    print("LIVE 연결:")
    print("  from exp_rot_phase_01 import RotPhaseObserver")
    print("  self._rp_obs = RotPhaseObserver(self, out_dir=self.state_dir,")
    print("                                  lookback=7, falling_streak=3, omega=0.10)")
    print("  # on_entry() / on_tick(tick, is_cut=...) / on_exit(reason, R)")
    print()
    print("3-레이어 병렬 운용 (권장):")
    print("  self._stb_obs  = STBObserver(self, ...)      # EXHAUSTION 감지")
    print("  self._video_obs= VideoObserver(self, ...)     # Frame Strip 궤적")
    print("  self._rp_obs   = RotPhaseObserver(self, ...)  # 위상 정렬 + WC 시뮬")
    print()
    print("WEAKENING_CUT 실장 조건 (시뮬 검증 후):")
    print("  if rp_obs.sim_weakening_cut_frame is not None:")
    print("    → WEAKENING_CUT 승격 (STB EXHAUSTION gate bypass 없음)")
    print("─" * 72)


if __name__ == "__main__":
    run_experiment()
