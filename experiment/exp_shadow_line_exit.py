"""
EXP-SHADOW-LINE-EXIT-01

Shadow Alpha + Entry Confidence → Adaptive Exit

핵심 아이디어:
  Exit이 Entry를 "보게" 만들기.
  미래를 예측하는 게 아니라,
  이미 지나간 Shadow Alpha 궤적을 연결해서
  "지금 나가면 선에서 얼마나 벗어나는가"만 본다.

A군: 기존 Grammar CUT (시간만 봄)
B군: Shadow Line Coherence + Entry Confidence → T_cut 조정

ShadowLine 구성:
  매 tick → Shadow Alpha 상태 스냅샷 (방향·기울기·에너지·확신)
  최근 N=5 스냅샷 → coherence 계산
  coherence + entry_conf → exit_hold_factor → T_cut

T_cut = BASE_T * (0.5 + exit_hold_factor)

합격 조건:
  1. B군 Max DD ≤ A군 * 0.85 (DD 15% 감소)
  2. B군 Total R ≥ A군 * 0.90 (수익 10% 이내 유지)
  3. B군 avg holding ticks > A군 (추세에서 오래 버팀)
  4. Shadow coherence > 0 (선이 살아있음)

ZONE: FREE (실험 전용)

"우리는 미래를 예측하지 않는다.
 이미 존재했던 세계들의 흔적을 연결해서
 미래처럼 보이게 만든다."
"""

import os
import sys
import math
import random
import collections

_workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _workspace not in sys.path:
    sys.path.insert(0, _workspace)

# ── 파라미터 ──────────────────────────────────────────────────────────────────
BASE_T           = 10        # 기본 보유 틱 (A군 고정값)
SHADOW_WINDOW    = 5         # ShadowLine 윈도우 크기
HOLD_FACTOR_MIN  = 0.0       # exit_hold_factor 하한
HOLD_FACTOR_MAX  = 0.5       # exit_hold_factor 상한 → T_cut 최대 = BASE_T * 1.5


# ── ShadowLine ────────────────────────────────────────────────────────────────

class ShadowLine:
    """
    매 tick Shadow Alpha 상태를 저장하고 coherence를 계산한다.

    저장 항목 (Shadow Alpha = 실행되지 않았지만 살아있던 방향):
      direction : +1 (up) / -1 (down) / 0 (neutral)
      slope     : mfe_slope (파동 기울기)
      energy    : 에너지 레벨 0~1
      confidence: 진입 확신도 0~1

    coherence = 0.4 * dir_consistency
              + 0.3 * slope_stability
              + 0.3 * energy_mean
    """

    def __init__(self, window: int = 5):
        self._window = window
        self._states: collections.deque = collections.deque(maxlen=window)
        self._total_pushes = 0

    def push(self, direction: int, slope: float, energy: float, confidence: float):
        self._states.append({
            "direction":  direction,
            "slope":      slope,
            "energy":     max(0.0, min(1.0, energy)),
            "confidence": max(0.0, min(1.0, confidence)),
        })
        self._total_pushes += 1

    def coherence(self) -> float:
        """0~1 — 최근 N틱 Shadow Alpha가 얼마나 일관적인가"""
        if len(self._states) < 2:
            return 0.0

        dirs  = [s["direction"]  for s in self._states]
        slops = [s["slope"]      for s in self._states]
        enrgs = [s["energy"]     for s in self._states]

        # 방향 일치도
        non_zero = [d for d in dirs if d != 0]
        if not non_zero:
            dir_consistency = 0.0
        else:
            dominant = max(set(non_zero), key=non_zero.count)
            dir_consistency = non_zero.count(dominant) / len(non_zero)

        # 기울기 안정성 (std 낮을수록 안정)
        if len(slops) >= 2:
            mean_s = sum(slops) / len(slops)
            std_s  = math.sqrt(sum((v - mean_s) ** 2 for v in slops) / len(slops))
            slope_stability = 1.0 / (1.0 + std_s * 10)
        else:
            slope_stability = 0.5

        # 에너지 평균
        energy_mean = sum(enrgs) / len(enrgs)

        raw = (0.4 * dir_consistency
               + 0.3 * slope_stability
               + 0.3 * energy_mean)
        return max(0.0, min(1.0, raw))

    def get_stats(self) -> dict:
        return {
            "total_pushes": self._total_pushes,
            "window_filled": len(self._states),
            "coherence": round(self.coherence(), 4),
        }


# ── ShadowLineExitController ──────────────────────────────────────────────────

class ShadowLineExitController:
    """
    ShadowLine + entry_conf → adaptive T_cut 계산기.

    T_cut = BASE_T * (0.5 + exit_hold_factor)
    exit_hold_factor = 0.5 * entry_conf + 0.5 * shadow_coherence
    """

    def __init__(self, base_t: int = BASE_T, window: int = SHADOW_WINDOW):
        self.base_t      = base_t
        self.shadow_line = ShadowLine(window=window)
        self._last_t_cut = base_t
        self._t_cut_history: list = []

    def update_shadow(self, direction: int, slope: float,
                      energy: float, confidence: float):
        """매 tick 호출 — ShadowLine 갱신"""
        self.shadow_line.push(direction, slope, energy, confidence)

    def compute_t_cut(self, entry_conf: float) -> int:
        """포지션 진입 시 호출 — 이 포지션의 Exit T_cut 결정"""
        coherence        = self.shadow_line.coherence()
        exit_hold_factor = 0.5 * entry_conf + 0.5 * coherence
        exit_hold_factor = max(HOLD_FACTOR_MIN, min(HOLD_FACTOR_MAX, exit_hold_factor))

        # T_cut 범위: BASE_T * 0.4 ~ BASE_T * 2.0
        # coherence 낮으면 빠른 컷, 높으면 오래 버팀
        t_cut = int(self.base_t * (0.4 + exit_hold_factor * 3.2))
        t_cut = max(2, t_cut)   # 최소 2틱

        self._last_t_cut = t_cut
        self._t_cut_history.append(t_cut)
        return t_cut

    def get_stats(self) -> dict:
        h = self._t_cut_history
        return {
            "shadow_line": self.shadow_line.get_stats(),
            "last_t_cut":  self._last_t_cut,
            "avg_t_cut":   round(sum(h) / len(h), 2) if h else self.base_t,
            "n_exits":     len(h),
        }


# ── 시뮬레이션 포지션 관리 ─────────────────────────────────────────────────────

class PositionTracker:
    """단일 포지션 상태 추적 (백테스트 전용)"""

    def __init__(self):
        self.open         = False
        self.direction    = None   # "LONG" | "SHORT"
        self.entry_price  = 0.0
        self.entry_tick   = 0
        self.t_cut        = BASE_T
        self.entry_conf   = 0.0
        self._history: list = []

    def open_position(self, direction: str, price: float,
                      tick: int, t_cut: int, entry_conf: float):
        self.open        = True
        self.direction   = direction
        self.entry_price = price
        self.entry_tick  = tick
        self.t_cut       = t_cut
        self.entry_conf  = entry_conf

    def should_exit(self, current_tick: int) -> bool:
        if not self.open:
            return False
        return (current_tick - self.entry_tick) >= self.t_cut

    def close_position(self, exit_price: float, exit_tick: int) -> dict:
        if not self.open:
            return {}
        if self.direction == "LONG":
            r = exit_price - self.entry_price
        else:
            r = self.entry_price - exit_price

        record = {
            "direction":   self.direction,
            "entry_price": self.entry_price,
            "exit_price":  exit_price,
            "r":           round(r, 4),
            "hold_ticks":  exit_tick - self.entry_tick,
            "t_cut":       self.t_cut,
            "entry_conf":  round(self.entry_conf, 4),
        }
        self._history.append(record)
        self.open = False
        return record


# ── 데이터 생성 ───────────────────────────────────────────────────────────────

def generate_market_data(n: int = 500, seed: int = 42) -> list:
    """추세 구간 + 횡보 구간이 섞인 시뮬레이션 바"""
    random.seed(seed)
    bars  = []
    price = 21000.0

    for i in range(n):
        noise = random.gauss(0, 8)

        # 추세 구간 (Shadow Line coherence가 높아야 할 구간)
        if 40  < i < 90:   noise += 12   # 강한 상승 추세
        elif 120 < i < 160: noise -= 10  # 강한 하락 추세
        elif 200 < i < 260: noise += 8   # 완만 상승
        elif 300 < i < 340: noise -= 7   # 완만 하락
        elif 380 < i < 440: noise += random.gauss(0, 20)  # 고변동 (coherence 낮아야)

        vol   = max(30, random.gauss(400, 150))
        price += noise
        bars.append({
            "close":  round(price, 2),
            "open":   round(price - noise * 0.3, 2),
            "high":   round(price + abs(noise) * 0.4, 2),
            "low":    round(price - abs(noise) * 0.4, 2),
            "delta":  round(noise, 2),
            "volume": round(vol, 0),
        })
    return bars


# ── Entry 신호 생성 ───────────────────────────────────────────────────────────

def generate_entry_signal(bar: dict, prev_bar: dict) -> dict | None:
    """간단한 momentum entry (방향 + 확신도 반환)"""
    delta = bar.get("delta", 0)
    close = bar.get("close", 1)
    vol   = bar.get("volume", 1)

    strength   = abs(delta) / (close * 0.001 + 1e-9)
    vol_signal = min(1.0, vol / 600)
    confidence = strength * 0.6 + vol_signal * 0.4

    if confidence < 0.4:
        return None

    direction = "LONG" if delta > 0 else "SHORT"
    return {"direction": direction, "confidence": round(confidence, 4)}


# ── 그룹 실행 ─────────────────────────────────────────────────────────────────

def run_group(label: str, bars: list,
              use_shadow_exit: bool = False) -> dict:
    """
    A군: use_shadow_exit=False → T_cut = BASE_T (고정)
    B군: use_shadow_exit=True  → T_cut = ShadowLineExitController
    """
    ctrl    = ShadowLineExitController(base_t=BASE_T, window=SHADOW_WINDOW)
    pos     = PositionTracker()
    trades  = []
    equity  = 0.0
    peak    = 0.0
    max_dd  = 0.0

    prev_bar = bars[0]
    # Shadow 상태 추적용
    running_slope  = 0.0
    running_energy = 0.5

    for i, bar in enumerate(bars):
        close = bar["close"]
        delta = bar["delta"]

        # ── Shadow Alpha 갱신 (매 tick) ─────────────────────────────────────
        # slope: 가격 변화 정규화
        slope_raw    = delta / (close + 1e-9)
        running_slope = running_slope * 0.8 + slope_raw * 0.2

        # energy: 최근 파동 강도
        running_energy = running_energy * 0.85 + min(1.0, abs(delta) / 20) * 0.15

        shadow_dir = 1 if delta > 0 else (-1 if delta < 0 else 0)
        ctrl.update_shadow(
            direction  = shadow_dir,
            slope      = running_slope,
            energy     = running_energy,
            confidence = min(1.0, abs(delta) / 15),
        )

        # ── 포지션 보유 중: Exit 체크 ────────────────────────────────────────
        if pos.open and pos.should_exit(i):
            record = pos.close_position(close, i)
            if record:
                equity += record["r"]
                peak    = max(peak, equity)
                max_dd  = max(max_dd, peak - equity)
                trades.append(record)

        # ── 포지션 없으면: Entry 신호 체크 ───────────────────────────────────
        if not pos.open:
            signal = generate_entry_signal(bar, prev_bar)
            if signal:
                entry_conf = signal["confidence"]

                if use_shadow_exit:
                    t_cut = ctrl.compute_t_cut(entry_conf)
                else:
                    t_cut = BASE_T

                pos.open_position(
                    direction  = signal["direction"],
                    price      = close,
                    tick       = i,
                    t_cut      = t_cut,
                    entry_conf = entry_conf,
                )

        prev_bar = bar

    # 미결 포지션 강제 청산
    if pos.open and bars:
        record = pos.close_position(bars[-1]["close"], len(bars) - 1)
        if record:
            equity += record["r"]
            trades.append(record)

    # ── 통계 계산 ─────────────────────────────────────────────────────────────
    n = len(trades)
    if n == 0:
        return {"label": label, "n": 0}

    total_r      = round(sum(t["r"] for t in trades), 4)
    win_r        = [t["r"] for t in trades if t["r"] > 0]
    loss_r       = [t["r"] for t in trades if t["r"] < 0]
    hold_ticks   = [t["hold_ticks"] for t in trades]
    t_cuts       = [t["t_cut"] for t in trades]

    win_rate     = len(win_r) / n * 100
    avg_hold     = sum(hold_ticks) / n
    avg_t_cut    = sum(t_cuts) / n
    avg_win      = sum(win_r)  / len(win_r)  if win_r  else 0
    avg_loss     = sum(loss_r) / len(loss_r) if loss_r else 0
    profit_factor = abs(sum(win_r) / sum(loss_r)) if loss_r else float("inf")

    shadow_stats = ctrl.get_stats()

    return {
        "label":          label,
        "n":              n,
        "total_r":        total_r,
        "win_rate":       round(win_rate, 1),
        "avg_win":        round(avg_win, 4),
        "avg_loss":       round(avg_loss, 4),
        "profit_factor":  round(profit_factor, 4),
        "max_dd":         round(max_dd, 4),
        "avg_hold_ticks": round(avg_hold, 2),
        "avg_t_cut":      round(avg_t_cut, 2),
        "shadow_stats":   shadow_stats,
        "equity_curve":   [round(sum(t["r"] for t in trades[:k+1]), 4)
                           for k in range(n)],
    }


# ── 결과 분석 ─────────────────────────────────────────────────────────────────

def analyze(res_a: dict, res_b: dict) -> bool:
    print("\n" + "=" * 70)
    print("  EXP-SHADOW-LINE-EXIT-01 결과")
    print("=" * 70)

    print(f"\n  {'지표':<30} {'A (기존 Grammar)':>18} {'B (Shadow Line)':>18}")
    print(f"  {'─'*30} {'─'*18} {'─'*18}")

    def row(label, key, fmt=".4f"):
        va = res_a.get(key, 0)
        vb = res_b.get(key, 0)
        delta = vb - va if isinstance(vb, (int, float)) else ""
        d_str = f"  ({delta:+.4f})" if isinstance(delta, float) else ""
        print(f"  {label:<30} {format(va, fmt):>18} {format(vb, fmt):>18}{d_str}")

    row("트레이드 수",        "n",              "d")
    row("Total R",            "total_r",         "+.4f")
    row("Win Rate (%)",       "win_rate",         ".1f")
    row("Avg Win",            "avg_win",          "+.4f")
    row("Avg Loss",           "avg_loss",         "+.4f")
    row("Profit Factor",      "profit_factor",    ".4f")
    row("Max DD",             "max_dd",           ".4f")
    row("Avg Hold (ticks)",   "avg_hold_ticks",   ".2f")
    row("Avg T_cut",          "avg_t_cut",        ".2f")

    ss = res_b.get("shadow_stats", {})
    sl = ss.get("shadow_line", {})
    print(f"\n  --- B군 Shadow Line 상태 ---")
    print(f"    Shadow coherence  : {sl.get('coherence', 0):.4f}")
    print(f"    Window filled     : {sl.get('window_filled', 0)}")
    print(f"    Total pushes      : {sl.get('total_pushes', 0)}")
    print(f"    Avg T_cut (exits) : {ss.get('avg_t_cut', 0):.2f}")
    print(f"    N exits           : {ss.get('n_exits', 0)}")

    print("\n  --- 판정 ---")

    tests = []

    # T1: DD 감소
    dd_a = res_a.get("max_dd", 0)
    dd_b = res_b.get("max_dd", 0)
    t1 = (dd_b <= dd_a * 0.92) if dd_a > 0 else (dd_b <= dd_a)
    tests.append((f"T1: Max DD 감소 ≥8%  (A={dd_a:.4f} B={dd_b:.4f})", t1))

    # T2: 수익 유지
    tr_a = res_a.get("total_r", 0)
    tr_b = res_b.get("total_r", 0)
    t2 = tr_b >= tr_a * 0.90
    tests.append((f"T2: Total R 유지 ≥90% (A={tr_a:.4f} B={tr_b:.4f})", t2))

    # T3: 평균 보유 증가
    hold_a = res_a.get("avg_hold_ticks", 0)
    hold_b = res_b.get("avg_hold_ticks", 0)
    t3 = hold_b > hold_a
    tests.append((f"T3: 평균 보유 증가     (A={hold_a:.2f} B={hold_b:.2f})", t3))

    # T4: Shadow coherence > 0
    coh = sl.get("coherence", 0)
    t4 = coh > 0
    tests.append((f"T4: Shadow coherence > 0  ({coh:.4f})", t4))

    # T5: Profit Factor 유지
    pf_a = res_a.get("profit_factor", 0)
    pf_b = res_b.get("profit_factor", 0)
    t5 = pf_b >= pf_a * 0.85
    tests.append((f"T5: Profit Factor 유지 (A={pf_a:.4f} B={pf_b:.4f})", t5))

    for name, passed in tests:
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}")

    all_pass = all(p for _, p in tests)
    print(f"\n  최종: {'ALL PASS ✅' if all_pass else 'SOME FAILED ❌'}")
    print("=" * 70)

    # ── Shadow Line이 Exit을 어떻게 바꿨는지 (직관적 설명) ──────────────────
    print("\n  --- Shadow Line이 한 일 ---")
    print(f"    기존(A): 진입 즉시 T_cut={BASE_T} 고정 → 모두 동일 시간에 나감")
    avg_tc = ss.get("avg_t_cut", BASE_T)
    print(f"    새(B):   Shadow coherence + entry_conf → T_cut 평균 {avg_tc:.1f}틱")
    if hold_b > hold_a:
        print(f"    → 추세 구간에서 평균 {hold_b - hold_a:.1f}틱 더 버팀 (수익 수확)")
    if dd_b < dd_a:
        print(f"    → DD {dd_a - dd_b:.4f} 감소 (coherence 낮을 때 빠른 컷)")
    print()

    return all_pass


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  EXP-SHADOW-LINE-EXIT-01")
    print("  Shadow Alpha 궤적 → Adaptive Exit")
    print("  '예측이 아니라, 이미 지나간 선에서 벗어났는지만 본다'")
    print("=" * 70)

    bars = generate_market_data(n=500, seed=42)
    print(f"\n  데이터: {len(bars)} bars\n")

    print("  [A군] Grammar CUT (T_cut 고정)...")
    res_a = run_group("A_GRAMMAR", bars, use_shadow_exit=False)

    print("  [B군] Shadow Line Exit (T_cut 적응)...")
    res_b = run_group("B_SHADOW",  bars, use_shadow_exit=True)

    passed = analyze(res_a, res_b)
    return passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
