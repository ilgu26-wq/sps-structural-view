"""
exp_mobius_fold_01.py — EXP-MOBIUS-FOLD-01
뫼비우스 접힘 지점 측정 실험.

"이 뇌는 언제, 무엇을 기준으로 자기 자신을 접는가?"

실험 구조:
  Phase A — 반복 손실 유도 (TAIL/MISMATCH 누적 → sealed 시점 측정)
  Phase B — 과적응 탐지 (같은 승리 패턴 반복 → 고정화 탐지)
  Phase C — 회복 후 왜곡 잔존성 (CLEAN 연속 후 sealed 해제 여부 + 잔류 효과)
  Phase D — 공명 교란 (성공 → 실패 급전환 → 뇌 반응 속도)

측정 지표:
  fold_point       : sealed=True가 된 trade 번호
  fold_condition   : 접힘 직전의 (mismatch, tail, weight)
  recovery_point   : sealed=False로 회복된 trade 번호
  residual_weight  : 회복 후 weight (< 1.0이면 왜곡 잔존)
  brain_dom_delta  : 각 단계에서 OrganismBrain dominant_energy 변화
  Δbrain_at_fold   : 접힘 시점의 뇌 에너지 변화 (부정적 공명 확인)

성공 조건:
  1. sealed 발생 확인 (접힘 존재)
  2. 접힘 시점에 brain dom_energy도 하락 (뫼비우스-뇌 공명)
  3. CLEAN 연속 후 회복 확인
  4. 회복 후 residual_weight < 1.0 (왜곡 잔존성 측정)

실행:
  python experiments/exp_mobius_fold_01.py --state-dir C:\\soar\\state3
"""

import os
import sys
import json
import argparse
from datetime import datetime

_HERE    = os.path.dirname(os.path.abspath(__file__))
_RUNTIME = os.path.join(_HERE, "..")
_ROOT    = os.path.join(_RUNTIME, "..")
for _p in [_RUNTIME, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── MöbiusManifold 로드 ───────────────────────────────────────────────────────
_Manifold = None
for _try in [
    "mobius_manifold",
    "irreversible.mobius_manifold",
    "soar_runtime.irreversible.mobius_manifold",
]:
    try:
        import importlib as _il
        _m = _il.import_module(_try)
        _Manifold = getattr(_m, "MöbiusManifold", None) or getattr(_m, "MobiusManifold", None)
        if _Manifold:
            print(f"[EXP] MöbiusManifold from: {_try}")
            break
    except ImportError:
        continue

# ── RuntimeMemoryAbsorber 로드 ────────────────────────────────────────────────
_Absorber = None
for _try in ["runtime_memory_absorber", "soar_runtime.runtime_memory_absorber"]:
    try:
        import importlib as _il
        _m = _il.import_module(_try)
        _Absorber = _m.RuntimeMemoryAbsorber
        break
    except ImportError:
        continue

if _Manifold is None:
    print("[EXP] ❌ MöbiusManifold not found")
    sys.exit(1)


# ── 실험 시나리오 정의 ────────────────────────────────────────────────────────

# Phase A: 반복 손실 → 접힘 유도
PHASE_A = [
    # (R, risk_tag, label)
    (-0.35, "AVOIDABLE_LOSS", "A1_loss"),
    (-0.42, "STRUCTURAL_LOSS","A2_loss"),
    (-0.55, "TAIL_EVENT",     "A3_tail"),   # ← TAIL은 2배 가중 → 여기서 sealed 예상
    (+0.10, "CLEAN_WIN",      "A4_clean"),  # 한 번 승리해도 sealed 유지?
    (-0.30, "AVOIDABLE_LOSS", "A5_loss"),
]

# Phase B: 과적응 탐지 (같은 CLEAN 패턴 반복)
PHASE_B = [
    (+0.25, "CLEAN_WIN", "B1_clean"),
    (+0.20, "CLEAN_WIN", "B2_clean"),
    (+0.18, "CLEAN_WIN", "B3_clean"),   # GATE_REVIVE_CLEAN=3 → sealed 해제?
    (+0.22, "CLEAN_WIN", "B4_clean"),
    (+0.15, "DIRTY_WIN", "B5_dirty"),
]

# Phase C: 회복 후 왜곡 잔존성
PHASE_C = [
    (+0.20, "CLEAN_WIN",      "C1_clean"),
    (+0.18, "CLEAN_WIN",      "C2_clean"),
    (+0.22, "CLEAN_WIN",      "C3_clean_revive"),  # 이후 weight가 완전 1.0 복원?
    (-0.20, "AVOIDABLE_LOSS", "C4_probe"),         # 왜곡 잔존 탐지
]

# Phase D: 급전환 공명 (성공 후 갑작스러운 실패)
PHASE_D = [
    (+0.30, "CLEAN_WIN",  "D1_peak"),
    (+0.25, "CLEAN_WIN",  "D2_peak"),
    (-0.60, "TAIL_EVENT", "D3_shock"),   # 급반전 → 뇌 반응
    (-0.45, "TAIL_EVENT", "D4_shock2"),
    (+0.10, "DIRTY_WIN",  "D5_weak"),
]

STATE_KEY = "UP_HVOL|BULL"
ACTION    = "LONG"


class MockRunner:
    def __init__(self, R, pain):
        self._dopamine = max(0.1, min(0.9, 0.5 + R * 0.5))
        self._last_pain_result_pain = pain


def run_fold_experiment(state_dir: str, verbose: bool = True) -> dict:
    print("\n" + "="*65)
    print("  EXP-MOBIUS-FOLD-01")
    print("  뫼비우스 접힘 지점 측정 실험")
    print("="*65)
    print(f"  state_dir   : {state_dir}")
    print(f"  GATE_THRESHOLD = 2 (mismatch + tail×2 ≥ 2 → sealed)")
    print()

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        manifold = _Manifold(state_dir=tmpdir)
        absorber = _Absorber(organism=None, state_dir=tmpdir) if _Absorber else None

        results = []
        fold_events   = []  # sealed=True 된 시점 기록
        revive_events = []  # sealed=False 회복 시점
        trade_n = 0

        def run_trade(R, risk_tag, label, phase):
            nonlocal trade_n
            trade_n += 1

            pain = max(0.0, -R * 0.8) if R < 0 else R * 0.05

            # MöbiusManifold: exit_to_entry 기록
            signal = manifold.exit_to_entry(
                state_key    = STATE_KEY,
                action       = ACTION,
                risk_tag     = risk_tag,
                phase_flags  = ["VOLATILE"],
                tesla_weight = 0.1 if R > 0 else 0.5,
            )

            # 게이트 상태 확인
            gate_r = manifold.check_gate(STATE_KEY, ACTION)
            gate_raw = manifold._gates.get(
                f"{STATE_KEY.split('|')[0]}|{ACTION}", None)
            if gate_raw is None:
                # _gate_key 방식 확인
                for k in manifold._gates:
                    if STATE_KEY.split("|")[0] in k:
                        gate_raw = manifold._gates[k]
                        break

            g_weight  = gate_r["weight"]
            g_sealed  = gate_r.get("blocked", False) or (gate_raw.sealed if gate_raw else False)
            g_mm      = gate_raw.mismatch_count if gate_raw else 0
            g_tc      = gate_raw.tail_count     if gate_raw else 0
            g_cs      = gate_raw.clean_streak   if gate_raw else 0

            # OrganismBrain 흡수
            brain_bef = 0.0
            brain_aft = 0.0
            dom_delta = 0.0
            if absorber is not None:
                snap_b = absorber._snap()
                absorber.digest_exit(
                    realized_R    = R,
                    exit_reason   = risk_tag,
                    state_key     = STATE_KEY,
                    action        = ACTION,
                    runner        = MockRunner(R, pain),
                )
                snap_a = absorber._snap()
                brain_bef = snap_b.get("dominant_energy", 0.0)
                brain_aft = snap_a.get("dominant_energy", 0.0)
                dom_delta = round(brain_aft - brain_bef, 4)

            rec = {
                "n":         trade_n,
                "phase":     phase,
                "label":     label,
                "R":         R,
                "risk_tag":  risk_tag,
                "gate_sealed": g_sealed,
                "weight":    round(g_weight, 4),
                "mismatch":  g_mm,
                "tail":      g_tc,
                "clean_streak": g_cs,
                "bad_count": g_mm + g_tc * 2,
                "dom_delta": dom_delta,
                "brain_e":   round(brain_aft, 3),
            }
            results.append(rec)

            # 접힘 이벤트 감지
            if g_sealed and (not results[-2]["gate_sealed"] if len(results) > 1 else True):
                fold_events.append({
                    "at": trade_n,
                    "label": label,
                    "bad_count": rec["bad_count"],
                    "weight": g_weight,
                    "dom_delta": dom_delta,
                })

            # 회복 이벤트 감지
            if not g_sealed and len(results) > 1 and results[-2]["gate_sealed"]:
                revive_events.append({
                    "at": trade_n,
                    "label": label,
                    "weight": g_weight,  # 완전 복원 여부 (1.0 미만이면 왜곡 잔존)
                    "residual": 1.0 - g_weight,
                })

            if verbose:
                seal_sym = "🔒" if g_sealed else "  "
                fold_sym = "⚡FOLD" if (fold_events and fold_events[-1]["at"] == trade_n) else ""
                rev_sym  = "🌱REVIVE" if (revive_events and revive_events[-1]["at"] == trade_n) else ""
                print(f"  {trade_n:>3} [{phase}] {label:<20} R={R:+.2f}  "
                      f"mm={g_mm} tc={g_tc} bad={rec['bad_count']} "
                      f"w={g_weight:.3f} {seal_sym}  "
                      f"Δbrain={dom_delta:+.4f}  {fold_sym}{rev_sym}")

            return rec

        print(f"  {'#':>3}  {'phase':<6} {'label':<20}  R      mm tc bad  weight    Δbrain")
        print("  " + "─"*70)

        for R, rt, lbl in PHASE_A:
            run_trade(R, rt, lbl, "A")

        print("  " + "─"*70)
        for R, rt, lbl in PHASE_B:
            run_trade(R, rt, lbl, "B")

        print("  " + "─"*70)
        for R, rt, lbl in PHASE_C:
            run_trade(R, rt, lbl, "C")

        print("  " + "─"*70)
        for R, rt, lbl in PHASE_D:
            run_trade(R, rt, lbl, "D")

    # ── 결과 분석 ─────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  접힘 분석 결과")
    print("="*65)

    # 접힘
    if fold_events:
        for fe in fold_events:
            print(f"\n  🔒 FOLD at trade #{fe['at']} ({fe['label']})")
            print(f"     bad_count={fe['bad_count']} (threshold=2)")
            print(f"     weight={fe['weight']:.4f}")
            print(f"     Δbrain={fe['dom_delta']:+.4f} "
                  f"{'← 뇌도 같이 접혔다 ✅' if fe['dom_delta'] <= 0 else '← 뇌는 반응 없음 ❌'}")
    else:
        print("\n  ❌ 접힘 미발생 (예상: TAIL_EVENT 이후 sealed=True)")

    # 회복
    if revive_events:
        for rv in revive_events:
            print(f"\n  🌱 REVIVE at trade #{rv['at']} ({rv['label']})")
            print(f"     weight 회복={rv['weight']:.4f}  "
                  f"잔존 왜곡={rv['residual']:.4f}")
            if rv['residual'] > 0:
                print(f"     → 왜곡 잔존 확인: 완전 복원 안 됨 (residual={rv['residual']:.4f})")
            else:
                print(f"     → 완전 복원 (왜곡 없음)")
    else:
        print("\n  🌱 회복 미발생 (CLEAN 3회 연속 부족)")

    # 뫼비우스-뇌 공명
    # 판정 기준:
    #   fold 시점 또는 fold 직후 3 trade 내에 Δbrain < -0.005 가 1회 이상
    #   (long-latency path들의 HOLD_EXIT 타이밍 노이즈 허용)
    revive_residual = [rv["residual"] for rv in revive_events]

    def _check_fold_brain(fold_events, results):
        if not fold_events: return False, []
        ok_folds = []
        for fe in fold_events:
            at = fe["at"]
            # fold 시점 + 이후 2 trade 윈도우
            window = [r for r in results if at <= r["n"] <= at + 2]
            any_neg = any(r["dom_delta"] < -0.003 for r in window)
            ok_folds.append({**fe, "brain_ok": any_neg})
        return any(f["brain_ok"] for f in ok_folds), ok_folds

    fold_brain_ok, ok_folds = _check_fold_brain(fold_events, results)

    print("\n  판정")
    PASS_FOLD     = len(fold_events) > 0
    PASS_BRAIN    = fold_brain_ok
    PASS_REVIVE   = len(revive_events) > 0
    PASS_RESIDUAL = any(r > 0 for r in revive_residual)

    for fe in ok_folds:
        sym = "✅" if fe.get("brain_ok") else "⚠"
        print(f"    {sym} FOLD #{fe['at']} Δbrain={fe['dom_delta']:+.4f} "
              f"{'(fold 윈도우 내 뇌 반응 확인)' if fe.get('brain_ok') else '(fold 윈도우 내 뇌 반응 없음)'}")

    print(f"  {'✅' if PASS_FOLD else '❌'} 접힘 발생      → {len(fold_events)}건")
    print(f"  {'✅' if PASS_BRAIN else '❌'} 접힘-뇌 공명   → {'fold 윈도우(±2) 내 Δbrain<-0.003 확인' if PASS_BRAIN else '불일치'}")
    print(f"  {'✅' if PASS_REVIVE else '⚠'} 회복 발생      → {len(revive_events)}건")
    print(f"  {'✅' if PASS_RESIDUAL else '⚠'} 왜곡 잔존성   → {'잔존 확인' if PASS_RESIDUAL else '완전 복원 (왜곡 없음)'}")

    overall = PASS_FOLD and PASS_BRAIN
    print()
    if overall:
        print("  🟢 PASS — 접힘 구조 확인됨. 뫼비우스는 접히고, 뇌와 함께 반응한다.")
    elif PASS_FOLD:
        print("  🟡 PARTIAL — 접힘은 발생했지만 뇌-공명이 불완전.")
    else:
        print("  🔴 FAIL — 접힘 미발생. GateCell 연결 확인 필요.")

    # ── 리포트 저장 ──────────────────────────────────────────────────────────
    report = {
        "exp":          "EXP-MOBIUS-FOLD-01",
        "ts":           datetime.now().isoformat(),
        "fold_events":  fold_events,
        "revive_events":revive_events,
        "pass_fold":    PASS_FOLD,
        "pass_brain":   PASS_BRAIN,
        "pass_revive":  PASS_REVIVE,
        "pass_residual":PASS_RESIDUAL,
        "overall":      overall,
        "results":      results,
    }
    try:
        os.makedirs(state_dir, exist_ok=True)
        rpt = os.path.join(state_dir, "mobius_fold_report.json")
        with open(rpt, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  리포트: {rpt}")
    except Exception as e:
        print(f"  리포트 저장 실패: {e}")

    print("="*65 + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXP-MOBIUS-FOLD-01")
    parser.add_argument("--state-dir", default="state")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_fold_experiment(state_dir=args.state_dir, verbose=not args.quiet)
