# exp_g_align_01.py
# EXP-G-ALIGN-01 v2
#
# 핵심 수정: g 절대값이 아닌 "상대비" 사용
#   g_ratio_grammar = g_grammar / g_entry  (에너지 잔존율)
#   g_ratio_shadow  = g_shadow  / g_entry
#
# 발견: corr(|dg|/g_entry, R) = 0.338 — 절대값(0.17)보다 신호 강함
# 즉, 에너지가 "얼마나 남았는지"가 수익과 연결됨

from __future__ import annotations
import argparse, csv, os
import random
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np


# ─────────────────────────────────────────────────────────────
# g(t) — 상대 변화율 (peak 정규화)
# ─────────────────────────────────────────────────────────────

def compute_g_rel(mfe: List[float], center: int, window: int = 3) -> float:
    """
    |dMFE/dt| / peak — 에너지 흐름 속도, peak 정규화.
    절대값이 아닌 상대 변화율 → MFE 크기에 무관하게 비교 가능.
    """
    peak = max(mfe) if mfe else 1e-9
    if peak < 0.1:
        return 0.0
    t0 = max(0, center - window)
    t1 = min(len(mfe), center + 1)
    slc = mfe[t0:t1]
    if len(slc) < 2:
        return 0.0
    diffs = [abs(slc[i] - slc[i-1]) for i in range(1, len(slc))]
    return float(np.mean(diffs)) / peak if diffs else 0.0


# ─────────────────────────────────────────────────────────────
# 시뮬레이터
# ─────────────────────────────────────────────────────────────

@dataclass
class GTrade:
    mfe: np.ndarray
    label: str
    true_R: float
    grammar_bar: int
    shadow_bar: int
    g_entry:   float = 0.0
    g_grammar: float = 0.0
    g_shadow:  float = 0.0
    # 핵심 지표: 상대비
    r_grammar: float = 0.0   # g_grammar / g_entry
    r_shadow:  float = 0.0   # g_shadow  / g_entry
    r_min:     float = 0.0   # min(r_grammar, r_shadow) — 에너지 잔존율 최솟값
    stall_grammar: bool = False
    collapse_grammar: bool = False
    stall_shadow: bool = False
    collapse_shadow: bool = False


def _make_mfe(T: int, seed: int, label: str,
              base: float = 40.0, noise: float = 1.5) -> np.ndarray:
    rng  = random.Random(seed)
    nrng = np.random.default_rng(seed)
    peak_t = rng.randint(T // 4, T // 2)
    mfe = np.zeros(T)
    for t in range(T):
        mfe[t] = base * min(1.0, t / max(peak_t, 1)) + nrng.normal(0, noise)
    mfe = np.clip(mfe, 0, None)
    if label == "collapse":
        start    = rng.randint(peak_t, min(peak_t + 3, T - 4))
        peak_val = float(np.max(mfe[:start + 1]))
        drop     = rng.uniform(0.38, 0.55)
        slope    = rng.uniform(0.025, 0.06)
        for t in range(start, T):
            mfe[t] = max(0, peak_val * (1 - drop) - slope * (t - start)
                         + nrng.normal(0, noise * 0.3))
    elif label == "stall":
        start   = rng.randint(int(T * 0.35), int(T * 0.55))
        sv      = float(np.max(mfe[:start + 1])) * rng.uniform(0.85, 0.97)
        for t in range(start, T):
            mfe[t] = max(0, sv + nrng.normal(0, noise * 0.12))
    return np.clip(np.convolve(mfe, np.ones(3)/3, mode="same"), 0, None)


def simulate_trades(n: int, T: int, seed: int,
                    p_collapse: float, p_stall: float) -> List[GTrade]:
    rng = random.Random(seed)
    grammar_bar = int(T * 0.55)
    shadow_bar  = int(T * 0.78)
    trades = []
    for i in range(n):
        r = rng.random()
        if r < p_collapse:
            label  = "collapse"; true_R = rng.uniform(-0.40, -0.08)
        elif r < p_collapse + p_stall:
            label  = "stall";    true_R = rng.uniform(-0.18, 0.04)
        else:
            label  = "alive";    true_R = rng.uniform(0.06,  0.38)
        mfe = _make_mfe(T, seed=seed * 10000 + i, label=label)
        trades.append(GTrade(mfe=mfe, label=label, true_R=true_R,
                             grammar_bar=min(grammar_bar, T-1),
                             shadow_bar=min(shadow_bar, T-1)))
    return trades


# ─────────────────────────────────────────────────────────────
# 측정
# ─────────────────────────────────────────────────────────────

class CollapseDetector:
    def __init__(self, drop_frac=0.30, hold_bars=3, min_peak=0.5):
        self.drop=drop_frac; self.hold=hold_bars; self.min_peak=min_peak; self.reset()
    def reset(self): self.peak=None; self.below=0; self.confirmed=False
    def update(self, now):
        if not now or now <= 0: return False
        self.peak = now if self.peak is None else max(self.peak, now)
        if self.peak < self.min_peak: self.below=0; self.confirmed=False; return False
        thr = self.peak*(1-self.drop)
        self.below = (self.below+1) if now<=thr else 0
        self.confirmed = (self.below>=self.hold); return self.confirmed

class StallDetector:
    def __init__(self, rel_eps=0.010, stall_bars=4, smooth_w=3, min_peak=0.5):
        self.rel_eps=rel_eps; self.stall_bars=stall_bars
        self.smooth_w=smooth_w; self.min_peak=min_peak; self.reset()
    def reset(self): self._h=[]; self._peak=None; self.below=0; self.stalled=False
    def update(self, now):
        now=max(0.0,float(now)); self._h.append(now)
        self._peak = now if self._peak is None else max(self._peak, now)
        n=len(self._h)
        if n<2 or self._peak<self.min_peak: return False
        w=min(self.smooth_w,n-1); rec=self._h[-w-1:]
        abs_d=[abs(rec[i]-rec[i-1]) for i in range(1,len(rec))]
        rc=(sum(abs_d)/len(abs_d))/self._peak if abs_d else 0.0
        self.below=(self.below+1) if rc<self.rel_eps else 0
        self.stalled=(self.below>=self.stall_bars); return self.stalled


def measure(tr: GTrade) -> GTrade:
    mfe = list(tr.mfe)
    EARLY = 4
    early = mfe[:EARLY+1]
    peak  = max(mfe) if mfe else 1e-9

    # g_entry: 초반 에너지 주입 속도
    if len(early) >= 2:
        diffs = [abs(early[i]-early[i-1]) for i in range(1,len(early))]
        tr.g_entry = float(np.mean(diffs)) / peak
    else:
        tr.g_entry = 0.0

    tr.g_grammar = compute_g_rel(mfe, tr.grammar_bar)
    tr.g_shadow  = compute_g_rel(mfe, tr.shadow_bar)

    eps = tr.g_entry + 1e-9
    tr.r_grammar = tr.g_grammar / eps
    tr.r_shadow  = tr.g_shadow  / eps
    tr.r_min     = min(tr.r_grammar, tr.r_shadow)

    cd_g=CollapseDetector(); sd_g=StallDetector()
    cd_s=CollapseDetector(); sd_s=StallDetector()
    for t, v in enumerate(mfe):
        cd_g.update(v); sd_g.update(v)
        cd_s.update(v); sd_s.update(v)
        if t == tr.grammar_bar:
            tr.collapse_grammar = cd_g.confirmed
            tr.stall_grammar    = sd_g.stalled
        if t == tr.shadow_bar:
            tr.collapse_shadow = cd_s.confirmed
            tr.stall_shadow    = sd_s.stalled
    return tr


# ─────────────────────────────────────────────────────────────
# 분석
# ─────────────────────────────────────────────────────────────

def analyze(trades: List[GTrade]) -> Dict:
    tail  = [t for t in trades if t.true_R < -0.15]
    clean = [t for t in trades if t.true_R >  0.05]
    n_t, n_c = len(tail) or 1, len(clean) or 1

    def m(lst, key): return round(float(np.mean([getattr(t,key) for t in lst])),5) if lst else 0.0
    def s(lst, key): return round(float(np.std([getattr(t,key) for t in lst])),5) if lst else 0.0

    # Q1: r_min (에너지 잔존율 최솟값) — tail vs clean
    r_min_tail  = m(tail, "r_min")
    r_min_clean = m(clean, "r_min")

    # Q2: r_grammar 분포
    r_gram_tail  = m(tail, "r_grammar")
    r_gram_clean = m(clean, "r_grammar")

    # Q3: stall/collapse 감지율
    stall_g_tail  = sum(1 for t in tail  if t.stall_grammar) / n_t
    stall_g_clean = sum(1 for t in clean if t.stall_grammar) / n_c
    stall_s_tail  = sum(1 for t in tail  if t.stall_shadow)  / n_t
    stall_s_clean = sum(1 for t in clean if t.stall_shadow)  / n_c
    coll_g_tail   = sum(1 for t in tail  if t.collapse_grammar) / n_t
    coll_g_clean  = sum(1 for t in clean if t.collapse_grammar) / n_c

    # Q4: 상관관계
    r_min_all = [t.r_min     for t in trades]
    R_all     = [t.true_R    for t in trades]
    corr_rmin_R = float(np.corrcoef(r_min_all, R_all)[0,1])

    # label별 r_grammar 분포 (핵심)
    label_stats = {}
    for lbl in ("collapse", "stall", "alive"):
        sub = [t for t in trades if t.label == lbl]
        if sub:
            label_stats[lbl] = {
                "r_grammar": round(float(np.mean([t.r_grammar for t in sub])),4),
                "r_shadow":  round(float(np.mean([t.r_shadow  for t in sub])),4),
                "r_min":     round(float(np.mean([t.r_min     for t in sub])),4),
            }

    return {
        "n_tail": len(tail), "n_clean": len(clean),
        "r_min_tail":  r_min_tail,  "r_min_clean": r_min_clean,
        "r_gram_tail": r_gram_tail, "r_gram_clean": r_gram_clean,
        "r_ratio": round(r_min_clean / (r_min_tail + 1e-9), 3),
        "stall_g_tail": round(stall_g_tail,3), "stall_g_clean": round(stall_g_clean,3),
        "stall_s_tail": round(stall_s_tail,3), "stall_s_clean": round(stall_s_clean,3),
        "coll_g_tail":  round(coll_g_tail,3),  "coll_g_clean":  round(coll_g_clean,3),
        "corr_rmin_R": round(corr_rmin_R, 4),
        "label_stats": label_stats,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades",     type=int,   default=600)
    ap.add_argument("--T",          type=int,   default=20)
    ap.add_argument("--p-collapse", type=float, default=0.25)
    ap.add_argument("--p-stall",    type=float, default=0.30)
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    trades = simulate_trades(args.trades, args.T, args.seed, args.p_collapse, args.p_stall)
    trades = [measure(t) for t in trades]
    a = analyze(trades)
    lc = {k: sum(1 for t in trades if t.label==k) for k in ("collapse","stall","alive")}

    print("\n==========================================================")
    print("  EXP-G-ALIGN-01 v2 — g(t) 상대 에너지 잔존율 검증")
    print("==========================================================")
    print(f"trades={args.trades}  T={args.T}  p_collapse={args.p_collapse}  p_stall={args.p_stall}")
    print(f"distribution: collapse={lc['collapse']} stall={lc['stall']} alive={lc['alive']}")
    print(f"tail(R<-0.15)={a['n_tail']}  clean(R>0.05)={a['n_clean']}")

    print("\n── Q1. r_min (에너지 잔존율): tail vs clean ─────────────")
    print(f"  tail  r_min: {a['r_min_tail']:.4f}")
    print(f"  clean r_min: {a['r_min_clean']:.4f}")
    print(f"  clean/tail ratio: {a['r_ratio']:.2f}x")

    print("\n── Q2. r_grammar (Grammar 시점 에너지 잔존율) ───────────")
    print(f"  tail  r_grammar: {a['r_gram_tail']:.4f}")
    print(f"  clean r_grammar: {a['r_gram_clean']:.4f}")

    print("\n── label별 에너지 잔존율 (핵심) ─────────────────────────")
    for lbl, st in a["label_stats"].items():
        print(f"  [{lbl:8s}] r_grammar={st['r_grammar']:.4f}  "
              f"r_shadow={st['r_shadow']:.4f}  r_min={st['r_min']:.4f}")

    print("\n── Q3. Stall/Collapse 감지율 ────────────────────────────")
    print(f"  stall@grammar:    tail={a['stall_g_tail']:.3f}  clean={a['stall_g_clean']:.3f}  "
          f"ratio={a['stall_g_tail']/(a['stall_g_clean']+1e-9):.1f}x")
    print(f"  collapse@grammar: tail={a['coll_g_tail']:.3f}  clean={a['coll_g_clean']:.3f}")
    print(f"  stall@shadow:     tail={a['stall_s_tail']:.3f}  clean={a['stall_s_clean']:.3f}  "
          f"ratio={a['stall_s_tail']/(a['stall_s_clean']+1e-9):.1f}x")

    print("\n── Q4. corr(r_min, R) ───────────────────────────────────")
    print(f"  {a['corr_rmin_R']:.4f}  (절대값 기준: 이전 corr(Δg_max,R)=0.171 → 상대비로 재계산)")

    print("\n── 판정 ─────────────────────────────────────────────────")
    findings = []

    if a["r_ratio"] >= 1.5:
        findings.append(f"✅ Q1 PASS: clean의 r_min이 tail보다 {a['r_ratio']:.2f}x 높음")
        findings.append("   → clean exit은 Grammar/Shadow 시점에도 에너지가 남아있음")
    else:
        findings.append(f"🟡 Q1: r_min ratio {a['r_ratio']:.2f}x")

    stall_ratio = a["stall_g_tail"] / (a["stall_g_clean"] + 1e-9)
    if stall_ratio >= 2.0:
        findings.append(f"✅ Q3 PASS: Grammar 시점 STALL이 tail에 {stall_ratio:.1f}x 집중")
    else:
        findings.append(f"🟡 Q3: stall ratio {stall_ratio:.1f}x")

    st = a["label_stats"]
    if (st.get("stall", {}).get("r_grammar", 1) <
            st.get("alive", {}).get("r_grammar", 0) * 0.6):
        findings.append(f"✅ LABEL: stall의 r_grammar({st['stall']['r_grammar']:.4f})가 "
                        f"alive({st['alive']['r_grammar']:.4f})보다 확연히 낮음")
        findings.append("   → g 잔존율로 stall/alive 분리 가능")

    if abs(a["corr_rmin_R"]) >= 0.20:
        findings.append(f"✅ Q4 PASS: corr(r_min, R) = {a['corr_rmin_R']:.4f}")
    else:
        findings.append(f"🟡 Q4: corr(r_min, R) = {a['corr_rmin_R']:.4f}")

    for f in findings: print(f"  {f}")
    passed = sum(1 for f in findings if f.strip().startswith("✅"))
    total  = sum(1 for f in findings if f.strip().startswith(("✅","🟡")))

    print(f"\n  [{passed}/{total} PASS]", end=" ")
    if passed >= 2:
        print("→ g 좌표계 불일치 확인")
        print("  다음: r_grammar / r_shadow를 live 로그에 추가")
        print("  → Grammar/Shadow의 실제 에너지 잔존율을 측정하기 시작")
    else:
        print("→ 추가 분석 필요")
    print("==========================================================\n")

    # CSV
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "g_align_v2.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label","true_R","g_entry","g_grammar","g_shadow",
                    "r_grammar","r_shadow","r_min",
                    "stall_g","collapse_g","stall_s","collapse_s"])
        for t in trades:
            w.writerow([t.label, round(t.true_R,4),
                        round(t.g_entry,6), round(t.g_grammar,6), round(t.g_shadow,6),
                        round(t.r_grammar,5), round(t.r_shadow,5), round(t.r_min,5),
                        int(t.stall_grammar), int(t.collapse_grammar),
                        int(t.stall_shadow),  int(t.collapse_shadow)])
    print(f"  [CSV] {out}")


if __name__ == "__main__":
    main()
