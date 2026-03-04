"""
EXP-DOPAMINE-SATURATION-01
===========================
목표: 포화의 원인을 2축으로 분리한다.

  시장 포화 (Market Saturation)
    파동 에너지가 천장/바닥에 붙어 변화량이 죽음
    → Var(alignment), Var(relative_drag), Var(amp*coh) → 0

  SOAR 포화 (Controller Saturation)
    파동은 살아있는데 파라미터가 바닥/천장에 고착
    → time_at_floor(entry_conf), time_at_cap(eps) 높음

도파민 = 파동 생존 기반 (수익 기반 ❌)
  dopamine = sigmoid(k1*align + k2*amp*coh - k3*drag - k4*converged - bias)

  파동 ALIVE + align 높음 → dopamine↑ → ε↓, risk 안정, gate↑
  drag↑ 또는 CONVERGED → dopamine↓ → ε↑, risk 보수, gate↓

4가지 시장 모드:
  LIVE_WAVE  : 파동 살아있음 (amp/coh 높고 drag 낮음)
  SAT_WAVE   : 파동 포화 (amp/coh 천장 고착, 변화량 죽음)
  CHOPPY     : 정렬 깨짐 (drag 잦고 align 낮음)
  CRASH_REC  : 방향 급전환 + drag 강한 구간

4군:
  G0 : Baseline — relative drag + 고정 ε 상한
  G1 : Dopamine-ε — ε 상한/목표를 dopamine으로 조절
  G2 : Dopamine-Risk — risk_scale을 dopamine으로 조절
  G3 : Full — ε + risk + gate 모두 dopamine 기반

판정:
  케이스 A (Market Sat) : Var(시장 지표)≈0, 군 간 차이 없음
  케이스 B (SOAR Sat)   : 시장 Var 있음, G3이 G0보다 고착 시간 짧고 DD↓
  케이스 C (둘 다)      : Axis 분해 단계로 이동

외부 의존 없음. python3 표준 라이브러리만 사용.
"""

import math
import random
import json
import statistics
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# 0) 글로벌 파라미터
# ─────────────────────────────────────────────────────────────────────────────
SEEDS        = [11, 42, 99, 123, 777]
N_BARS       = 2000
ETA_PE       = 0.005
EPSILON_BASE = 0.005
EPS_ABS_MIN  = 0.001
EPS_ABS_MAX  = 0.15
EPS_HARD_CAP = EPSILON_BASE * 10   # G0 고정 상한
SLOPE_WIN    = 3
EPS_NORM     = 1e-5

# 도파민 계수
D_K_ALIGN    = 2.5
D_K_ORBIT    = 1.5
D_K_DRAG     = 3.0
D_K_CONV     = 2.0
D_BIAS       = -1.2

# ─────────────────────────────────────────────────────────────────────────────
# 1) 4가지 시장 모드 합성 데이터
# ─────────────────────────────────────────────────────────────────────────────
def _regime_segment(rng, regime_type: str) -> dict:
    """regime_type → (slope_bias, noise, align_base, amp_base, coh_base)"""
    if regime_type == "LIVE_WAVE":
        return dict(bias=rng.uniform(0.0010, 0.0020), noise=0.0006,
                    align=rng.uniform(0.72, 0.88), amp=rng.uniform(0.65, 0.80),
                    coh=rng.uniform(0.65, 0.78), drag_boost=0.0)
    elif regime_type == "SAT_WAVE":
        return dict(bias=rng.uniform(0.0003, 0.0008), noise=0.0002,
                    align=rng.uniform(0.55, 0.62), amp=rng.uniform(0.78, 0.92),
                    coh=rng.uniform(0.72, 0.88), drag_boost=0.0)
    elif regime_type == "CHOPPY":
        return dict(bias=0.0, noise=0.0022,
                    align=rng.uniform(0.30, 0.55), amp=rng.uniform(0.35, 0.55),
                    coh=rng.uniform(0.35, 0.52), drag_boost=0.3)
    else:   # CRASH_REC
        return dict(bias=rng.uniform(-0.0035, -0.0020), noise=0.0018,
                    align=rng.uniform(0.08, 0.22), amp=rng.uniform(0.20, 0.40),
                    coh=rng.uniform(0.25, 0.45), drag_boost=0.5)


MARKET_MODES = ["LIVE_WAVE", "SAT_WAVE", "CHOPPY", "CRASH_REC"]
_MODE_RATIOS = {        # 시장 모드별 레짐 구성 비율
    "LIVE_WAVE":  {"LIVE_WAVE": 0.60, "SAT_WAVE": 0.20, "CHOPPY": 0.10, "CRASH_REC": 0.10},
    "SAT_WAVE":   {"LIVE_WAVE": 0.10, "SAT_WAVE": 0.65, "CHOPPY": 0.15, "CRASH_REC": 0.10},
    "CHOPPY":     {"LIVE_WAVE": 0.15, "SAT_WAVE": 0.10, "CHOPPY": 0.60, "CRASH_REC": 0.15},
    "CRASH_REC":  {"LIVE_WAVE": 0.20, "SAT_WAVE": 0.10, "CHOPPY": 0.10, "CRASH_REC": 0.60},
}


def generate_market(seed: int, market_mode: str, n_bars: int = N_BARS) -> List[dict]:
    rng   = random.Random(seed)
    bars: List[dict] = []
    price = 100.0
    ratios = _MODE_RATIOS[market_mode]

    # 레짐 시퀀스 생성
    regime_types = list(ratios.keys())
    weights      = [ratios[r] for r in regime_types]
    seq: List[str] = []
    while len(seq) < n_bars:
        # 가중 선택
        r_idx = random.Random(rng.random()).choices(range(len(regime_types)), weights=weights)[0]
        rt = regime_types[r_idx]
        length = rng.randint(40, 100)
        seq.extend([rt] * length)
    seq = seq[:n_bars]

    prev_seg = None
    for i, rt in enumerate(seq):
        if prev_seg is None or (i > 0 and seq[i] != seq[i-1]):
            seg = _regime_segment(rng, rt)
        else:
            seg = prev_seg
        prev_seg = seg

        delta = seg["bias"] + rng.gauss(0, seg["noise"])
        price = max(1.0, price * (1 + delta))

        # align/amp/coh에 약간의 노이즈 추가
        align = max(0.0, min(1.0, seg["align"] + rng.gauss(0, 0.04)))
        amp   = max(0.0, min(1.0, seg["amp"]   + rng.gauss(0, 0.03)))
        coh   = max(0.0, min(1.0, seg["coh"]   + rng.gauss(0, 0.03)))

        # 수렴 플래그: amp*coh 높고 |delta|가 작으면 수렴 가능성
        converged = (amp * coh > 0.55 and abs(delta) < seg["noise"] * 0.5)

        bars.append({
            "idx":      i,
            "close":    round(price, 4),
            "regime":   rt,
            "delta":    delta,
            "align":    round(align, 4),
            "amp":      round(amp, 4),
            "coh":      round(coh, 4),
            "converged": converged,
            "drag_boost": seg["drag_boost"],
        })

    return bars


# ─────────────────────────────────────────────────────────────────────────────
# 2) 도파민 계산 (파동 생존 기반)
# ─────────────────────────────────────────────────────────────────────────────
def compute_dopamine(
    align: float,
    amp: float,
    coh: float,
    wave_drag_norm: float,
    converged: bool,
) -> float:
    """
    dopamine ∈ [0, 1]
    파동이 ALIVE + align 높음 → high (ε↓, risk 안정)
    drag↑ 또는 CONVERGED   → low  (ε↑, risk 보수)
    """
    raw = (
        D_K_ALIGN  * align
        + D_K_ORBIT  * amp * coh
        - D_K_DRAG   * wave_drag_norm
        - D_K_CONV   * (1.0 if converged else 0.0)
        + D_BIAS
    )
    # sigmoid
    raw = max(-10.0, min(10.0, raw))
    dopa = 1.0 / (1.0 + math.exp(-raw))

    # 정렬도 낮고 surprise 없으면 suppression
    if align < 0.35 and wave_drag_norm < 0.05:
        dopa *= 0.3

    return round(dopa, 6)


# ─────────────────────────────────────────────────────────────────────────────
# 3) 도파민 기반 제어 함수
# ─────────────────────────────────────────────────────────────────────────────
def dopamine_eps_cap(dopa: float) -> float:
    """
    G1/G3: ε 상한을 dopamine으로 조절
    dopamine↑ (파동 살아있음) → cap 낮음 (과탐색 방지)
    dopamine↓ (위험)          → cap 높음 (재탐색 허용)
    """
    cap = EPS_ABS_MIN + (1.0 - dopa) * (EPS_HARD_CAP - EPS_ABS_MIN)
    return round(max(EPS_ABS_MIN, min(EPS_ABS_MAX, cap)), 6)


def dopamine_risk_scale(dopa: float, wave_drag: float) -> float:
    """
    G2/G3: risk_scale을 dopamine으로 조절
    dopamine 높음 → 안정 진입 (risk_scale 1.0 근처)
    dopamine 낮음 → 보수 진입 (risk_scale↓)
    drag 추가 압력 포함
    """
    base = 0.5 + 0.5 * dopa
    drag_pen = wave_drag * 2.0          # drag → 추가 감쇠
    rs = base / (1.0 + drag_pen)
    return round(max(0.2, min(1.0, rs)), 4)


def dopamine_gate_open(dopa: float) -> bool:
    """
    G3: 진입 gate — dopamine < 0.3이면 SOAR 코어 진입 기준 높임
    (block이 아니라 ε↑로 진입 확률 낮춤 — 규칙 아님)
    """
    return dopa >= 0.25


# ─────────────────────────────────────────────────────────────────────────────
# 4) 미니 PlasticEngine (dopamine_cap 주입형)
# ─────────────────────────────────────────────────────────────────────────────
class MiniPlasticEngine:
    def __init__(self, eta: float = ETA_PE):
        self._params = {"ec": 0.55, "ep": 0.05}
        self._prev_ec = 0.55
        self._aw = deque(maxlen=10)
        self._lm = 0
        self._eta = eta
        self._h: deque = deque(maxlen=200)
        # 포화 추적
        self._ticks_at_cap  = 0
        self._ticks_at_floor= 0
        self._total_ticks   = 0

    def tick(self, alignment: Optional[float], current_eps: float,
             wave_drag: float, eps_cap: float) -> dict:
        self._total_ticks += 1

        if alignment is not None:
            for k, (lo, hi) in [("ec", (0.35, 0.80)), ("ep", (0.01, 0.20))]:
                t = lo + (hi - lo) * alignment
                self._params[k] += self._eta * (t - self._params[k])

        ec = self._params["ec"]
        ep = self._params["ep"]
        d  = ec - self._prev_ec

        if alignment is not None:
            self._aw.append(alignment)
            ar = (max(self._aw) - min(self._aw)) if len(self._aw) >= 3 else 0.0
            self._lm = 1 if (ar > 0.05 and abs(d) < 0.002) else 0
        self._prev_ec = ec

        lb   = 0.005 if self._lm else 0.0
        cr   = max(0.0, -d * 20) * 0.002
        db   = wave_drag * 8.0
        et   = max(EPS_ABS_MIN, min(eps_cap, ep + lb - cr + db))
        eo   = max(EPS_ABS_MIN, min(eps_cap, current_eps * 0.8 + et * 0.2))

        # 포화 추적
        if eo >= eps_cap * 0.98:
            self._ticks_at_cap += 1
        if eo <= EPS_ABS_MIN * 1.02:
            self._ticks_at_floor += 1

        self._h.append(eo)
        return {
            "eo": eo, "db": db, "ep_in": current_eps,
            "ec": round(ec, 6),
        }

    def saturation_stats(self) -> dict:
        total = max(self._total_ticks, 1)
        return {
            "cap_ratio":   round(self._ticks_at_cap   / total, 4),
            "floor_ratio": round(self._ticks_at_floor / total, 4),
            "total":       self._total_ticks,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5) SOAR 진입 판단
# ─────────────────────────────────────────────────────────────────────────────
def soar_eval(bar: dict) -> Tuple[bool, float]:
    al    = bar["align"]
    amp   = bar["amp"]
    coh   = bar["coh"]
    ss    = max(0.0, min(1.0, abs(al - 0.5) * 4))
    orbit = amp * coh
    tau_r = 6 if abs(bar["delta"]) > 0.001 else 2
    nr    = 0.7 if bar["regime"] in ("CHOPPY", "CRASH_REC") else 0.2
    rr    = 0.6 if bar["regime"] in ("CRASH_REC",) else 0.25
    eq    = ss * 0.25 + (tau_r / 20) * 0.20 + orbit * 0.30 + (1 - nr) * 0.15 + (1 - rr) * 0.10
    return eq >= 0.42, round(eq, 4)


ACTIONS = ["long_d0", "long_d1", "short_d0", "short_d1"]


class MiniEntryCore:
    def __init__(self, eps: float = EPSILON_BASE):
        self.eps = eps
        self._q: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {a: 0.0 for a in ACTIONS}
        )

    def select(self, sk: str, rng: random.Random) -> str:
        if rng.random() < self.eps:
            return rng.choice(ACTIONS)
        q = self._q[sk]
        return max(q, key=q.get)

    def update(self, sk: str, ac: str, reward: float):
        old = self._q[sk][ac]
        self._q[sk][ac] = old + 0.1 * (reward - old)


# ─────────────────────────────────────────────────────────────────────────────
# 6) 단일 시뮬 실행
# ─────────────────────────────────────────────────────────────────────────────
def run_one(
    bars:       List[dict],
    seed:       int,
    group:      str,   # "G0" | "G1" | "G2" | "G3"
) -> Tuple[List[dict], dict]:
    rng = random.Random(seed)
    pe  = MiniPlasticEngine()
    ec  = MiniEntryCore(eps=EPSILON_BASE)

    trades: List[dict] = []
    pos        = None
    cur_eps    = EPSILON_BASE
    prev_slope = 0.0
    sw         = deque(maxlen=SLOPE_WIN)

    # 진단 수집
    diag = {
        "eps_h":        [],
        "dopa_h":       [],
        "drag_h":       [],
        "risk_h":       [],
        "gate_h":       [],    # gate_open True/False
        "sat_market":   [],    # align * amp * coh (시장 에너지 지표)
        "ovc":          0,
    }

    for i, bar in enumerate(bars):
        sw.append(bar["delta"])
        cur_slope = sum(sw) / len(sw)

        # Relative drag (EXP-03 정규화)
        abs_dec   = max(0.0, -(cur_slope - prev_slope))
        wd_rel    = abs_dec / (abs(prev_slope) + EPS_NORM)
        wd_norm   = math.tanh(wd_rel)
        # drag_boost: CHOPPY/CRASH에서 추가 압력
        wd_norm   = min(1.0, wd_norm + bar.get("drag_boost", 0.0) * wd_norm)
        prev_slope = cur_slope

        # 도파민 계산
        dopa = compute_dopamine(
            align      = bar["align"],
            amp        = bar["amp"],
            coh        = bar["coh"],
            wave_drag_norm = wd_norm,
            converged  = bar.get("converged", False),
        )

        # 군별 제어 파라미터
        if group == "G0":
            eps_cap    = EPS_HARD_CAP
            risk_fn    = lambda d, w: max(0.2, 1.0 / (1.0 + 4.0 * w)) if w > 0.01 else 1.0
            gate_open  = True
        elif group == "G1":
            eps_cap    = dopamine_eps_cap(dopa)
            risk_fn    = lambda d, w: max(0.2, 1.0 / (1.0 + 4.0 * w)) if w > 0.01 else 1.0
            gate_open  = True
        elif group == "G2":
            eps_cap    = EPS_HARD_CAP
            risk_fn    = lambda d, w: dopamine_risk_scale(d, w)
            gate_open  = True
        else:   # G3
            eps_cap    = dopamine_eps_cap(dopa)
            risk_fn    = lambda d, w: dopamine_risk_scale(d, w)
            gate_open  = dopamine_gate_open(dopa)

        diag["dopa_h"].append(round(dopa, 4))
        diag["drag_h"].append(round(wd_norm, 4))
        diag["gate_h"].append(gate_open)
        diag["sat_market"].append(round(bar["align"] * bar["amp"] * bar["coh"], 4))

        # 포지션 보유 중
        if pos is not None:
            hb  = i - pos["i"]
            cp  = bar["close"]
            ep  = pos["ep"]
            dr  = pos["dir"]
            raw = ((cp - ep) / ep) if dr == "long" else ((ep - cp) / ep)
            close_now = (
                hb >= 20 or raw >= 0.008 or raw <= -0.005
                or (bar["regime"] == "CRASH_REC" and dr == "long")
            )
            if close_now:
                rr_v = raw * 100 * pos["rs"]
                ec.update(bar["regime"], pos["ac"], rr_v)
                trades.append({
                    "idx": i, "regime": bar["regime"], "dir": dr,
                    "R":  round(rr_v, 4),
                    "wd": round(wd_norm, 4),
                    "dopa": round(dopa, 4),
                    "ei": pos["ei"], "eo": pos["eo"],
                    "rs": pos["rs"],
                })
                pos = None
            else:
                pe.tick(bar["align"], cur_eps, wd_norm, eps_cap)
            continue

        # 진입 판단
        ok, _ = soar_eval(bar)
        if not ok or not gate_open:
            r2 = pe.tick(bar["align"], cur_eps, wd_norm, eps_cap)
            ne = r2["eo"]
            if abs(ne - cur_eps) > 1e-6:
                diag["ovc"] += 1
            cur_eps = ne
            diag["eps_h"].append(round(cur_eps, 6))
            diag["risk_h"].append(1.0)
            continue

        r2 = pe.tick(bar["align"], cur_eps, wd_norm, eps_cap)
        ne = r2["eo"]
        if abs(ne - cur_eps) > 1e-6:
            diag["ovc"] += 1
        cur_eps = ne

        diag["eps_h"].append(round(cur_eps, 6))
        rs = risk_fn(dopa, wd_norm)
        diag["risk_h"].append(rs)

        ec.eps = cur_eps
        sk  = bar["regime"]
        ac  = ec.select(sk, rng)
        dr  = "long" if "long" in ac else "short"

        pos = {
            "i": i, "dir": dr, "ep": bar["close"], "ac": ac,
            "ei": round(EPSILON_BASE, 6), "eo": round(cur_eps, 6), "rs": rs,
        }

    diag["pe_sat"] = pe.saturation_stats()
    return trades, diag


# ─────────────────────────────────────────────────────────────────────────────
# 7) 지표 계산
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(trades: List[dict], diag: dict) -> dict:
    if not trades:
        return {"error": "no_trades", "trades": 0, "max_dd": 999,
                "total_R": 0, "pf": 0, "ws": 0, "eps_std": 0,
                "dopa_mean": 0, "drag_mean": 0, "risk_mean": 1.0,
                "gate_closed_pct": 0, "market_energy_var": 0,
                "eps_cap_ratio": 0, "eps_floor_ratio": 0}
    rs     = [t["R"] for t in trades]
    cumul  = 0.0; peak = 0.0; max_dd = 0.0
    streak = 0;   ws   = 0
    in_dd  = False; dd_start = 0; rec = []
    for i, r in enumerate(rs):
        cumul += r
        if cumul > peak:
            if in_dd: rec.append(i - dd_start)
            peak = cumul; in_dd = False
        else:
            dd = peak - cumul
            if dd > max_dd:
                max_dd = dd; dd_start = i; in_dd = True
        streak = (streak + 1) if r < 0 else 0
        ws = max(ws, streak)

    tail  = sum(1 for r in rs if r <= -2.0)
    total = round(sum(rs), 4)
    wins  = [r for r in rs if r > 0]
    loss  = [abs(r) for r in rs if r < 0]
    pf    = round(sum(wins) / max(sum(loss), 1e-9), 4)

    eh  = diag.get("eps_h",       [])
    dh  = diag.get("dopa_h",      [])
    drh = diag.get("drag_h",      [])
    rh  = diag.get("risk_h",      [])
    gh  = diag.get("gate_h",      [])
    mh  = diag.get("sat_market",  [])

    dopa_mean  = round(statistics.mean(dh), 4)  if dh  else 0.0
    drag_mean  = round(statistics.mean(drh), 4) if drh else 0.0
    risk_mean  = round(statistics.mean(rh), 4)  if rh  else 1.0
    eps_std    = round(statistics.stdev(eh), 6) if len(eh) >= 2 else 0.0
    gate_off   = round(sum(1 for g in gh if not g) / max(len(gh), 1), 4)
    mkt_var    = round(statistics.variance(mh), 6) if len(mh) >= 2 else 0.0

    sat = diag.get("pe_sat", {})

    return {
        "trades":           len(rs),
        "total_R":          total,
        "avg_R":            round(total / max(len(rs), 1), 4),
        "pf":               pf,
        "max_dd":           round(max_dd, 4),
        "tail":             tail,
        "ws":               ws,
        "avg_rec":          round(statistics.mean(rec), 1) if rec else None,
        # 도파민 / 물리 지표
        "dopa_mean":        dopa_mean,
        "drag_mean":        drag_mean,
        "risk_mean":        risk_mean,
        "eps_std":          eps_std,
        "gate_closed_pct":  gate_off,
        # 포화 판정 지표
        "market_energy_var":mkt_var,          # 시장 포화 지표
        "eps_cap_ratio":    sat.get("cap_ratio",   0.0),   # SOAR 포화 지표
        "eps_floor_ratio":  sat.get("floor_ratio", 0.0),
        "ovc":              diag.get("ovc", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8) 포화 원인 자동 판정
# ─────────────────────────────────────────────────────────────────────────────
def saturation_diagnosis(
    market_mode: str,
    g0_avg: dict,
    g3_avg: dict,
    all_avg: Dict[str, dict],
) -> dict:
    """
    케이스 A: 시장 포화
    케이스 B: SOAR 포화
    케이스 C: 둘 다
    """
    mkt_var   = g0_avg.get("market_energy_var", 0)
    g0_cap    = g0_avg.get("eps_cap_ratio", 0)
    g3_cap    = g3_avg.get("eps_cap_ratio", 0)
    g0_dd     = g0_avg.get("max_dd", 1)
    g3_dd     = g3_avg.get("max_dd", 1)
    dd_ratio  = g3_dd / max(g0_dd, 1e-9)

    market_saturated  = mkt_var < 0.005
    soar_saturated    = g0_cap > 0.20   # 20%+ 시간이 cap에 붙어있으면 포화
    dopamine_helps    = dd_ratio < 0.90 and g3_cap < g0_cap * 0.80

    if market_saturated and not soar_saturated:
        case = "A_MARKET_SAT"
        verdict = "파동이 없어서 먹을 게 없음 — 환경 문제"
        next_step = "시장 모드 전환 대기 또는 더 작은 Axis 분해"
    elif not market_saturated and soar_saturated and dopamine_helps:
        case = "B_SOAR_SAT"
        verdict = "SOAR 포화 확인 — dopamine plasticity가 해법"
        next_step = "G3을 실거래 데이터에 적용, friction/inertia 미세조정"
    elif not market_saturated and soar_saturated and not dopamine_helps:
        case = "B_SOAR_SAT_NO_HELP"
        verdict = "SOAR 포화이나 도파민 효과 없음 — ε 이외 게이트 확인"
        next_step = "CollectiveEnergy + DopamineWave 결합 확인"
    elif market_saturated and soar_saturated:
        case = "C_BOTH_SAT"
        verdict = "시장+SOAR 둘 다 포화 — Axis 분해 단계로 이동"
        next_step = "E/S/I Axis 재분해 후 weak-market 전용 축 추가"
    else:
        case = "OK_NORMAL"
        verdict = "포화 없음 — 정상 작동"
        next_step = "PnL 개선 단계로 이동"

    return {
        "market_mode":      market_mode,
        "case":             case,
        "verdict":          verdict,
        "next_step":        next_step,
        "market_energy_var":mkt_var,
        "g0_cap_ratio":     g0_cap,
        "g3_cap_ratio":     g3_cap,
        "dd_ratio_G3_G0":   round(dd_ratio, 4),
        "dopamine_helps":   dopamine_helps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9) 메인
# ─────────────────────────────────────────────────────────────────────────────
GROUPS = ["G0", "G1", "G2", "G3"]


def run_experiment():
    print("=" * 80)
    print("EXP-DOPAMINE-SATURATION-01")
    print(f"  Seeds={SEEDS}  N_bars={N_BARS}  Market modes={MARKET_MODES}")
    print(f"  도파민: 파동 생존 기반 | k_align={D_K_ALIGN} k_orbit={D_K_ORBIT} "
          f"k_drag={D_K_DRAG} k_conv={D_K_CONV}")
    print("=" * 80)

    # all_res[market_mode][group][seed] = metrics
    all_res: Dict[str, Dict[str, Dict[int, dict]]] = {
        mm: {g: {} for g in GROUPS} for mm in MARKET_MODES
    }
    all_tr: Dict[str, Dict[str, Dict[int, list]]] = {
        mm: {g: {} for g in GROUPS} for mm in MARKET_MODES
    }

    for mm in MARKET_MODES:
        for seed in SEEDS:
            bars = generate_market(seed, mm, N_BARS)
            for g in GROUPS:
                tr, diag = run_one(bars, seed, g)
                all_res[mm][g][seed] = compute_metrics(tr, diag)
                all_tr[mm][g][seed]  = tr

    # ── [1] 시장 모드별 메트릭 테이블 ─────────────────────────────────────────
    print("\n[1] 시장 모드별 MaxDD (5-seed 평균)")
    hdr = f"  {'Mode':>12}" + "".join(f"  {g:>8}" for g in GROUPS) + "  판정"
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    mode_avg: Dict[str, Dict[str, dict]] = {}
    for mm in MARKET_MODES:
        row = f"  {mm:>12}"
        gavg: Dict[str, dict] = {}
        for g in GROUPS:
            vs = list(all_res[mm][g].values())
            def av(k):
                v = [x[k] for x in vs if k in x and isinstance(x[k], (int, float))]
                return round(sum(v) / max(len(v), 1), 4)
            a = {k: av(k) for k in ["max_dd", "total_R", "pf", "trades",
                                      "dopa_mean", "drag_mean", "risk_mean",
                                      "eps_std", "market_energy_var",
                                      "eps_cap_ratio", "gate_closed_pct"]}
            gavg[g] = a
            row += f"  {a['max_dd']:>8.3f}"
        mode_avg[mm] = gavg
        g3_dd = gavg["G3"]["max_dd"]
        g0_dd = gavg["G0"]["max_dd"]
        ratio = g3_dd / max(g0_dd, 1e-9)
        flag  = "✅" if ratio < 0.85 else ("⚠️" if ratio < 1.0 else "❌")
        row += f"  G3/G0={ratio:.3f} {flag}"
        print(row)

    # ── [2] 도파민 / 포화 지표 ────────────────────────────────────────────────
    print("\n[2] 도파민 & 포화 지표 (G0 vs G3, 5-seed 평균)")
    for mm in MARKET_MODES:
        g0 = mode_avg[mm]["G0"]
        g3 = mode_avg[mm]["G3"]
        print(f"\n  {mm}")
        print(f"    시장 에너지 분산: {g0['market_energy_var']:.5f}  "
              f"({'저분산=포화 의심' if g0['market_energy_var'] < 0.005 else '분산 정상'})")
        print(f"    G0: dopa={g0['dopa_mean']:.3f}  drag={g0['drag_mean']:.3f}  "
              f"eps_cap_ratio={g0['eps_cap_ratio']:.3f}  risk={g0['risk_mean']:.3f}")
        print(f"    G3: dopa={g3['dopa_mean']:.3f}  drag={g3['drag_mean']:.3f}  "
              f"eps_cap_ratio={g3['eps_cap_ratio']:.3f}  risk={g3['risk_mean']:.3f}  "
              f"gate_off={g3['gate_closed_pct']:.3f}")

    # ── [3] PnL 지표 ─────────────────────────────────────────────────────────
    print("\n[3] PnL 지표 (total_R, PF) — G0 vs G3")
    for mm in MARKET_MODES:
        g0 = mode_avg[mm]["G0"]
        g3 = mode_avg[mm]["G3"]
        r_dd  = g3["max_dd"]  / max(g0["max_dd"],  1e-9)
        r_pnl = g3["total_R"] / max(g0["total_R"], 1e-9) if g0["total_R"] > 0 else 0
        print(f"  {mm:>12}: MaxDD G3/G0={r_dd:.3f}  TotalR G3/G0={r_pnl:.3f}  "
              f"G3_trades={int(g3['trades'])}")

    # ── [4] 포화 원인 자동 판정 ──────────────────────────────────────────────
    print("\n[4] 포화 원인 자동 판정")
    diag_results = {}
    for mm in MARKET_MODES:
        dx = saturation_diagnosis(mm, mode_avg[mm]["G0"], mode_avg[mm]["G3"], mode_avg[mm])
        diag_results[mm] = dx
        print(f"\n  [{mm}]  → {dx['case']}")
        print(f"    {dx['verdict']}")
        print(f"    mkt_var={dx['market_energy_var']:.5f}  "
              f"g0_cap={dx['g0_cap_ratio']:.3f}  g3_cap={dx['g3_cap_ratio']:.3f}  "
              f"dd_ratio={dx['dd_ratio_G3_G0']:.3f}  dopa_helps={dx['dopamine_helps']}")
        print(f"    다음: {dx['next_step']}")

    # ── [5] seed별 상세 (seed=99, CRASH_REC) ────────────────────────────────
    print("\n[5] seed=99, CRASH_REC 군별 상세")
    mm_focus = "CRASH_REC"
    print(f"  {'Group':>5}  {'MaxDD':>7}  {'TotalR':>8}  "
          f"{'dopa':>6}  {'drag':>6}  {'risk':>6}  {'eps_cap%':>8}  {'gate_off':>8}")
    for g in GROUPS:
        m = all_res[mm_focus][g].get(99, {})
        print(f"  {g:>5}  {m.get('max_dd',0):>7.3f}  {m.get('total_R',0):>8.3f}  "
              f"{m.get('dopa_mean',0):>6.3f}  {m.get('drag_mean',0):>6.3f}  "
              f"{m.get('risk_mean',1):>6.3f}  {m.get('eps_cap_ratio',0):>8.3f}  "
              f"{m.get('gate_closed_pct',0):>8.3f}")

    # ── [6] worst-DD 구간 drill-down (CRASH_REC, G0 vs G3, seed=99) ─────────
    print("\n[6] CRASH_REC seed=99 worst-DD (G0 vs G3)")
    for g in ["G0", "G3"]:
        trades = all_tr[mm_focus][g].get(99, [])
        _dd_drilldown(trades, g, top_n=5)

    # ── [FINAL] ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("[FINAL VERDICT]")
    passed = 0
    for mm in MARKET_MODES:
        dx = diag_results[mm]
        if dx["dopamine_helps"]:
            passed += 1
    if passed >= 3:
        v = "✅ DOPAMINE EFFECTIVE — 3개+ 시장 모드에서 도파민이 DD 개선"
    elif passed >= 2:
        v = "⚠️ PARTIAL — 일부 모드에서만 효과, 계수 조정 필요"
    else:
        v = "❌ DOPAMINE NOT YET — 포화 원인 재진단 필요"
    print(f"  {v}  ({passed}/{len(MARKET_MODES)} 모드에서 G3<G0)")
    print("=" * 80)

    # JSON 저장
    out = {
        "config": {
            "seeds": SEEDS, "n_bars": N_BARS,
            "d_k_align": D_K_ALIGN, "d_k_orbit": D_K_ORBIT,
            "d_k_drag": D_K_DRAG, "d_k_conv": D_K_CONV,
        },
        "mode_avg":    {mm: {g: mode_avg[mm][g] for g in GROUPS} for mm in MARKET_MODES},
        "diagnosis":   diag_results,
        "seed_detail": {
            mm: {g: {str(s): all_res[mm][g][s] for s in SEEDS} for g in GROUPS}
            for mm in MARKET_MODES
        },
    }
    path = "/mnt/user-data/outputs/exp_dopamine_sat_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n결과: {path}")


def _dd_drilldown(trades: List[dict], label: str, top_n: int = 5):
    if not trades:
        print(f"  {label}: no trades"); return
    rs = [t["R"] for t in trades]
    cumul = peak = max_dd = ds = 0.0
    ws_slice: List[dict] = []
    for i, r in enumerate(rs):
        cumul += r
        if cumul > peak: peak = cumul; ds = i
        elif peak - cumul > max_dd:
            max_dd = peak - cumul
            ws_slice = trades[max(0, int(ds) - 2): i + 1]
    print(f"  {label}  {'bar':>5}  {'regime':>12}  {'dir':>6}  "
          f"{'R':>7}  {'dopa':>6}  {'drag':>6}  {'rs':>6}")
    for t in ws_slice[-top_n:]:
        print(f"       {t['idx']:>5}  {t['regime']:>12}  {t['dir']:>6}  "
              f"{t['R']:>7.3f}  {t.get('dopa',0):>6.3f}  "
              f"{t.get('wd',0):>6.4f}  {t.get('rs',1):>6.3f}")


if __name__ == "__main__":
    run_experiment()
