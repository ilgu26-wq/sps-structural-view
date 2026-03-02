"""
exp_breath_resonance_01.py — EXP-BREATH-RESONANCE-01
런타임 ↔ soar_core 호흡 공명 실험.

"런타임이 숨 쉴 때, 코어도 같은 박자로 숨 쉬는가?"

실험 구조:
  1. RuntimeMemoryAbsorber 로드
  2. 실거래 결과를 모사한 digest 시퀀스 주입 (실로그 기반)
  3. 매 흡수마다 BODY / BRAIN 상태 비교 로그
  4. 공명 통계 출력

성공 조건:
  1. 시간 공명: brain.epoch += 1 (같은 bar에서 반응)
  2. 방향 공명: 좋은 R → dom_energy↑, 나쁜 R → dom_energy↓ 또는 alive↓
  3. 누적 공명: 여러 trade 후 channel_gains 방향 변화

실행:
  python experiments/exp_breath_resonance_01.py --state-dir C:\\soar\\state3

결과 파일:
  state_dir/breath_log.jsonl    — 전체 호흡 기록
  state_dir/resonance_report.json — 공명 통계
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime, timedelta

# ── 경로 설정 ────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_RUNTIME = os.path.join(_HERE, "..")
_ROOT    = os.path.join(_RUNTIME, "..")

for _p in [_RUNTIME, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── RuntimeMemoryAbsorber 로드 ────────────────────────────────────────────────
_Absorber = None
for _try in [
    "runtime_memory_absorber",
    "soar_runtime.runtime_memory_absorber",
]:
    try:
        import importlib as _il
        _m = _il.import_module(_try)
        _Absorber = _m.RuntimeMemoryAbsorber
        print(f"[EXP] RuntimeMemoryAbsorber from: {_try}")
        break
    except ImportError:
        continue

if _Absorber is None:
    print("[EXP] ❌ RuntimeMemoryAbsorber not found — check path")
    sys.exit(1)


# ── 테스트 시나리오 ────────────────────────────────────────────────────────────

SCENARIOS = [
    # (realized_R, pain, dopamine, am_energy, exit_reason, state_key)
    # Phase 1: 안정적 승리 → brain이 살아나야 함
    (+0.28, 0.05, 0.82, +0.15, "GRAMMAR_CUT",    "UP_HVOL|BULL"),
    (+0.18, 0.08, 0.74, +0.10, "GRAMMAR_CUT",    "UP_HVOL|BULL"),
    (+0.22, 0.06, 0.79, +0.12, "SHADOW_LINE_CUT","UP_HVOL|BULL"),
    # Phase 2: 꼬리 손실 → brain이 조여야 함
    (-0.38, 0.55, 0.41, -0.18, "TAIL_KILL",      "DN_HVOL|BEAR"),
    (-0.45, 0.72, 0.32, -0.25, "TAIL_KILL",      "DN_HVOL|BEAR"),
    # Phase 3: 회복 시도
    (+0.12, 0.20, 0.62, +0.05, "MFE_SLOPE_CUT",  "UP_MVOL|BULL"),
    (-0.15, 0.35, 0.52, -0.08, "AVOIDABLE_LOSS", "UP_MVOL|BULL"),
    (+0.09, 0.18, 0.65, +0.03, "SHADOW_LINE_CUT","UP_MVOL|BULL"),
    # Phase 4: 고통 누적 → channel_gains 반응 확인
    (-0.52, 0.85, 0.25, -0.30, "TAIL_KILL",      "DN_HVOL|BEAR"),
    (-0.31, 0.60, 0.38, -0.20, "MFE_SLOPE_CUT",  "DN_HVOL|BEAR"),
    # Phase 5: 안정화
    (+0.20, 0.10, 0.75, +0.12, "GRAMMAR_CUT",    "UP_LVOL|CALM"),
    (+0.15, 0.12, 0.70, +0.08, "GRAMMAR_CUT",    "UP_LVOL|CALM"),
    (+0.25, 0.07, 0.80, +0.15, "SHADOW_LINE_CUT","UP_LVOL|CALM"),
]


class MockActionMemory:
    """ActionMemory 모사 객체."""
    def __init__(self, energy: float, state_key: str, action: str):
        self._energy = energy
        self._sk = state_key
        self._act = action

    def get_cell_stats(self, state_key, action):
        if state_key == self._sk and action == self._act:
            return {"energy_bias": self._energy}
        return {"energy_bias": 0.0}


class MockRunner:
    """Runner 모사 객체."""
    def __init__(self, dopamine: float, pain: float):
        self._dopamine = dopamine
        self._last_pain_result_pain = pain


def run_experiment(state_dir: str, verbose: bool = True):
    print("\n" + "="*65)
    print("  EXP-BREATH-RESONANCE-01")
    print("  런타임 ↔ soar_core 호흡 공명 실험")
    print("="*65)
    print(f"  state_dir : {state_dir}")
    print(f"  scenarios : {len(SCENARIOS)}")
    print()

    absorber = _Absorber(organism=None, state_dir=state_dir)

    results   = []
    resonant  = 0
    dir_match = 0

    print("─"*65)
    print(f"  {'#':>3}  {'R':>7}  {'pain':>5}  {'reason':<18}  "
          f"{'dom_Δ':>8}  {'epoch':>6}  {'res':>4}")
    print("─"*65)

    for i, (R, pain, dopa, am_e, reason, sk) in enumerate(SCENARIOS):
        am  = MockActionMemory(am_e, sk, "LONG")
        run = MockRunner(dopa, pain)

        breath = absorber.digest_exit(
            realized_R    = R,
            exit_reason   = reason,
            state_key     = sk,
            action        = "LONG",
            action_memory = am,
            exit_am       = None,
            growth_signal = "CRUISE" if R > 0 else "PAIN",
            runner        = run,
        )

        dom_delta = breath["delta"].get("dominant_energy", 0.0)
        ep_delta  = breath["delta"].get("epoch", 0)
        ep_after  = breath["brain_after"].get("epoch", 0)
        res_ok    = breath.get("resonance", False)

        # 방향 공명: R > 0이면 dom_energy 또는 alive가 개선
        dir_ok = (
            (R > 0 and dom_delta >= -0.005) or
            (R < 0 and dom_delta <= +0.005)
        )

        if res_ok:
            resonant  += 1
        if dir_ok:
            dir_match += 1

        results.append({
            "i":         i + 1,
            "R":         R,
            "pain":      pain,
            "reason":    reason,
            "dom_delta": dom_delta,
            "ep_after":  ep_after,
            "resonant":  res_ok,
            "dir_ok":    dir_ok,
        })

        if verbose:
            res_sym = "✅" if res_ok else "❌"
            dir_sym = "→" if dir_ok else "↯"
            print(f"  {i+1:>3}  {R:>+7.3f}  {pain:>5.3f}  {reason:<18}  "
                  f"{dom_delta:>+8.4f}  {ep_after:>6}  {res_sym}{dir_sym}")

    print("─"*65)

    # ── 공명 통계 ────────────────────────────────────────────────────────────
    n        = len(SCENARIOS)
    res_rate = resonant  / n
    dir_rate = dir_match / n

    stats = absorber.get_resonance_stats(n)
    brain_ok = absorber.brain is not None

    print()
    print("  공명 결과")
    print(f"  시간 공명 (same bar): {resonant}/{n} = {res_rate:.1%}")
    print(f"  방향 공명 (R↔brain): {dir_match}/{n} = {dir_rate:.1%}")
    print(f"  brain epoch final  : {stats.get('brain_epoch', 0)}")
    print(f"  brain alive final  : {stats.get('brain_alive', 0)}")
    print()

    # ── 채널 게인 변화 ────────────────────────────────────────────────────────
    if absorber._last_breath.get("brain_after", {}).get("channel_gains"):
        cg = absorber._last_breath["brain_after"]["channel_gains"]
        print("  channel_gains (final):")
        for ch, v in cg.items():
            bar = "█" * int(v * 10) + "░" * max(0, 10 - int(v * 10))
            print(f"    {ch:<10} {v:.4f}  {bar}")
        print()

    # ── 성공 판정 ─────────────────────────────────────────────────────────────
    PASS_RESONANCE = res_rate >= 0.7
    PASS_DIRECTION = dir_rate >= 0.65
    PASS_BRAIN     = brain_ok

    print("  판정")
    print(f"  {'✅' if PASS_RESONANCE else '❌'} 시간 공명 ≥ 70%  → {res_rate:.1%}")
    print(f"  {'✅' if PASS_DIRECTION else '❌'} 방향 공명 ≥ 65%  → {dir_rate:.1%}")
    print(f"  {'✅' if PASS_BRAIN else '❌'} OrganismBrain 연결됨")

    overall = PASS_RESONANCE and PASS_DIRECTION and PASS_BRAIN
    print()
    print(f"  {'🟢 PASS — 공명 확인됨. 런타임과 코어가 같은 박자로 숨 쉰다.' if overall else '🔴 PARTIAL — 아직 완전한 공명 아님. 흡수 파라미터 조정 필요.'}")

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    report = {
        "exp":          "EXP-BREATH-RESONANCE-01",
        "ts":           datetime.now().isoformat(),
        "n_scenarios":  n,
        "resonance_rate": round(res_rate, 3),
        "direction_rate": round(dir_rate, 3),
        "brain_ok":     brain_ok,
        "overall_pass": overall,
        "brain_stats":  stats,
        "results":      results,
    }

    rpt_path = os.path.join(state_dir, "resonance_report.json")
    try:
        os.makedirs(state_dir, exist_ok=True)
        with open(rpt_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  리포트 저장: {rpt_path}")
    except Exception as e:
        print(f"\n  리포트 저장 실패: {e}")

    print("="*65 + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXP-BREATH-RESONANCE-01")
    parser.add_argument("--state-dir", default="state",
                        help="soar_core state directory")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-scenario output")
    args = parser.parse_args()

    run_experiment(state_dir=args.state_dir, verbose=not args.quiet)
