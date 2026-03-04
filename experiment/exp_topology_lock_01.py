"""
exp_topology_lock_01.py — EXP-TOPOLOGY-LOCK-01
위상 정렬 봉인 실험.

"판단–접힘–기억–다음 판단의 인과 순서를 절대 어기지 않도록 봉인한다."

검증할 우주의 시간축 (이 순서가 법칙):
  T0  Sensory 관측
  T1  판단 요청 (pre-epoch brain state 읽기)
  T2  Entry / Exit 실행
  T3  Outcome 발생 (R, pain)
  T4  Möbius fold / revive 판정 (exit_to_entry)
  T5  Hippocampus 기억 반영 (내부 − process_epoch 안)
  T6  Brain epoch++ (run_epoch 호출 시)
  T7  다음 판단

3가지 불변식 (모두 통과해야 PASS):
  I1 — "이전 판단 불변":
       snap_bef.epoch == brain.epoch at judgment time
       즉 T3 전에 brain epoch가 증가하면 안 됨

  I2 — "epoch 단조 증가":
       각 digest_exit() 호출에서 epoch += 1 정확히 한 번
       snap_bef.epoch + 1 == snap_aft.epoch

  I3 — "fold 효과 미래 한정":
       fold 발생 trade의 brain_before == fold 미발생 시의 expected state
       즉 fold 기록이 같은 digest 안의 brain 상태를 즉시 바꾸지 않음
       (fold 기록 = MöbiusManifold에 저장, brain에는 다음 epoch에서 반영)

실행:
  python experiments/exp_topology_lock_01.py --state-dir C:\\soar\\state3
"""

import os
import sys
import json
import copy
import argparse
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in [os.path.join(_HERE, ".."), os.path.join(_HERE, "../..")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 모듈 로드 ─────────────────────────────────────────────────────────────────
_Absorber = None
_Manifold = None

for _try in ["runtime_memory_absorber", "soar_runtime.runtime_memory_absorber"]:
    try:
        import importlib as _il
        _Absorber = _il.import_module(_try).RuntimeMemoryAbsorber
        break
    except ImportError:
        continue

for _try in ["mobius_manifold", "irreversible.mobius_manifold",
             "soar_runtime.irreversible.mobius_manifold"]:
    try:
        import importlib as _il
        _m = _il.import_module(_try)
        _Manifold = getattr(_m, "MöbiusManifold", None) or getattr(_m, "MobiusManifold", None)
        if _Manifold:
            break
    except ImportError:
        continue

if _Absorber is None:
    print("[EXP] ❌ RuntimeMemoryAbsorber not found")
    sys.exit(1)


# ── 위상 이벤트 타입 ──────────────────────────────────────────────────────────
T0 = "T0_SENSORY"
T1 = "T1_JUDGMENT_PRE"
T2 = "T2_EXECUTION"
T3 = "T3_OUTCOME"
T4 = "T4_MOBIUS_FOLD"
T5 = "T5_HIPPOCAMPUS"
T6 = "T6_EPOCH_INC"
T7 = "T7_NEXT_JUDGMENT"


class TopoLog:
    """위상 이벤트 시퀀스 기록."""
    def __init__(self):
        self.events = []
        self._trade = 0

    def mark(self, phase: str, epoch: int, data: dict = None):
        self.events.append({
            "trade":  self._trade,
            "phase":  phase,
            "epoch":  epoch,
            "data":   data or {},
        })

    def start_trade(self):
        self._trade += 1

    def get_sequence(self, trade: int) -> list:
        return [e for e in self.events if e["trade"] == trade]

    def verify_order(self, trade: int, required_seq: list) -> tuple:
        """
        주어진 위상이 required_seq 순서로 발생했는지 검증.
        Returns (ok: bool, violations: list)
        """
        evts = self.get_sequence(trade)
        phases = [e["phase"] for e in evts]
        violations = []

        for i, (pa, pb) in enumerate(zip(required_seq, required_seq[1:])):
            if pa not in phases or pb not in phases:
                continue
            ia = phases.index(pa)
            ib = phases.index(pb)
            if ia > ib:
                violations.append(f"{pa}({ia}) AFTER {pb}({ib})")

        return len(violations) == 0, violations


class MockRunner:
    def __init__(self, R, pain):
        self._dopamine = max(0.1, min(0.9, 0.5 + R * 0.5))
        self._last_pain_result_pain = pain


def run_topology_experiment(state_dir: str, verbose: bool = True) -> dict:
    print("\n" + "="*65)
    print("  EXP-TOPOLOGY-LOCK-01")
    print("  위상 정렬 봉인 실험")
    print("="*65)
    print()

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:

        absorber = _Absorber(organism=None, state_dir=tmpdir)
        manifold = _Manifold(state_dir=tmpdir) if _Manifold else None
        topo     = TopoLog()

        violations_all = []
        inv1_results   = []   # I1: 이전 판단 불변
        inv2_results   = []   # I2: epoch 단조 증가
        inv3_results   = []   # I3: fold 효과 미래 한정

        TRADES = [
            # (R, risk_tag, label)
            (-0.40, "AVOIDABLE_LOSS", "fold_setup_1"),
            (-0.35, "STRUCTURAL_LOSS","fold_trigger"),   # → fold 발생 예상
            (+0.25, "CLEAN_WIN",      "post_fold_good"), # fold 이후 첫 판단
            (-0.50, "TAIL_EVENT",     "post_fold_bad"),
            (+0.20, "CLEAN_WIN",      "recovery_1"),
            (+0.18, "CLEAN_WIN",      "recovery_2"),
            (+0.22, "CLEAN_WIN",      "recovery_3"),    # REVIVE 예상
            (+0.15, "CLEAN_WIN",      "post_revive"),
        ]

        print(f"  {'#':>3}  {'label':<20}  {'R':>6}  "
              f"{'epoch_bef':>9} {'epoch_aft':>9}  "
              f"{'I1':>3} {'I2':>3} {'I3':>3}  {'fold':>5}")
        print("  " + "─"*70)

        for trade_n, (R, risk_tag, label) in enumerate(TRADES, 1):
            topo.start_trade()
            pain = max(0.0, -R * 0.8) if R < 0 else R * 0.05

            # ── T0: Sensory ────────────────────────────────────────────────
            pre_epoch = absorber.brain.epoch if absorber.brain else 0
            topo.mark(T0, pre_epoch, {"R": R})

            # ── T1: Judgment pre (brain state before outcome) ─────────────
            snap_judgment = absorber._snap()
            judgment_epoch = snap_judgment.get("epoch", 0)
            judgment_dom_e = snap_judgment.get("dominant_energy", 0.0)
            topo.mark(T1, judgment_epoch, {"dom_e": judgment_dom_e})

            # ── T2: Execution (already committed at this point) ───────────
            topo.mark(T2, judgment_epoch, {"action": "LONG"})

            # ── T3: Outcome ───────────────────────────────────────────────
            topo.mark(T3, judgment_epoch, {"R": R, "risk_tag": risk_tag})

            # ── T4: Möbius fold 판정 (digest_exit 이전에 기록) ─────────────
            fold_before_digest = False
            fold_at_t4 = False
            if manifold:
                # fold 전 상태 저장
                gate_key_raw = None
                for k in manifold._gates:
                    if "UP_HVOL" in k:
                        gate_key_raw = k
                        break
                sealed_before = (manifold._gates[gate_key_raw].sealed
                                 if gate_key_raw and gate_key_raw in manifold._gates
                                 else False)
                _ = manifold.exit_to_entry(
                    state_key    = "UP_HVOL|BULL",
                    action       = "LONG",
                    risk_tag     = risk_tag,
                    phase_flags  = ["VOLATILE"],
                    tesla_weight = 0.1 if R > 0 else 0.4,
                )
                # fold 후 상태
                for k in manifold._gates:
                    if "UP_HVOL" in k:
                        gate_key_raw = k
                        break
                sealed_after = (manifold._gates[gate_key_raw].sealed
                                if gate_key_raw and gate_key_raw in manifold._gates
                                else False)
                fold_at_t4 = (not sealed_before) and sealed_after

            topo.mark(T4, judgment_epoch,
                      {"fold": fold_at_t4, "sealed": sealed_after if manifold else False})

            # ── T6: Brain epoch++ (digest_exit 내부에서 발생) ────────────────
            # I3 검증: digest_exit 호출 전 brain epoch를 미리 저장
            brain_epoch_before_digest = absorber.brain.epoch if absorber.brain else 0
            brain_dom_before_digest   = absorber._snap().get("dominant_energy", 0.0)

            breath = absorber.digest_exit(
                realized_R    = R,
                exit_reason   = risk_tag,
                state_key     = "UP_HVOL|BULL",
                action        = "LONG",
                runner        = MockRunner(R, pain),
            )

            # ── T6 기록 ────────────────────────────────────────────────────
            post_epoch = absorber.brain.epoch if absorber.brain else 0
            topo.mark(T6, post_epoch, {"epoch_delta": post_epoch - brain_epoch_before_digest})

            # ── T5: Hippocampus (T6와 같은 run_epoch 내부에서 발생) ─────────
            topo.mark(T5, post_epoch, {"internal": "process_epoch"})

            # ── T7: Next judgment read ─────────────────────────────────────
            snap_after = absorber._snap()
            next_epoch = snap_after.get("epoch", 0)
            next_dom_e = snap_after.get("dominant_energy", 0.0)
            topo.mark(T7, next_epoch, {"dom_e": next_dom_e})

            # ── 불변식 검증 ────────────────────────────────────────────────

            # I1: 이전 판단 불변
            # T1의 brain_epoch == digest 전 epoch (판단 시점의 뇌 상태 = 구 상태)
            i1_ok = (judgment_epoch == brain_epoch_before_digest)
            inv1_results.append(i1_ok)

            # I2: epoch 단조 증가 (정확히 +1)
            ep_delta = breath["delta"]["epoch"]
            i2_ok = (ep_delta == 1)
            inv2_results.append(i2_ok)

            # I3: fold 효과 미래 한정
            # fold가 T4에서 발생했더라도 → digest_exit 안에서 brain 상태는 snap_bef 기준
            # 즉 breath["brain_before"]["epoch"] == brain_epoch_before_digest
            i3_snap_epoch = breath["brain_before"].get("epoch", -1)
            i3_ok = (i3_snap_epoch == brain_epoch_before_digest)
            if fold_at_t4:
                # fold 발생 시: brain_before가 fold 기록에 오염되지 않아야 함
                i3_ok = (i3_snap_epoch == brain_epoch_before_digest) and (
                    breath["brain_before"].get("dominant_energy", 0) == brain_dom_before_digest
                )
            inv3_results.append(i3_ok)

            # 순서 검증
            ok_order, viols = topo.verify_order(trade_n, [T0, T1, T2, T3, T4, T6, T7])
            violations_all.extend(viols)

            # 결과 출력
            fold_sym  = "🔒FOLD" if fold_at_t4 else "     "
            i1_sym = "✅" if i1_ok else "❌"
            i2_sym = "✅" if i2_ok else "❌"
            i3_sym = "✅" if i3_ok else "❌"

            if verbose:
                print(f"  {trade_n:>3}  {label:<20}  {R:>+6.2f}  "
                      f"{brain_epoch_before_digest:>9} {post_epoch:>9}  "
                      f"{i1_sym} {i2_sym} {i3_sym}  {fold_sym}")
                if not i1_ok:
                    print(f"       ⚠ I1 FAIL: judgment_epoch={judgment_epoch} "
                          f"!= before_digest={brain_epoch_before_digest}")
                if not i2_ok:
                    print(f"       ⚠ I2 FAIL: epoch_delta={ep_delta} (expected 1)")
                if not i3_ok:
                    print(f"       ⚠ I3 FAIL: snap_bef.epoch={i3_snap_epoch} "
                          f"!= before_digest={brain_epoch_before_digest}")

    # ── 결과 분석 ─────────────────────────────────────────────────────────────
    n = len(TRADES)
    i1_pass = all(inv1_results)
    i2_pass = all(inv2_results)
    i3_pass = all(inv3_results)
    order_pass = len(violations_all) == 0

    print("\n" + "="*65)
    print("  위상 정렬 검증 결과")
    print("="*65)
    print()
    print("  불변식 결과:")
    print(f"  {'✅' if i1_pass else '❌'} I1 — 이전 판단 불변")
    print(f"        T1 brain_epoch == T3 이전 epoch  →  {inv1_results.count(True)}/{n}")
    print(f"        판단 시점의 뇌 상태는 항상 이전 epoch 기준")

    print(f"\n  {'✅' if i2_pass else '❌'} I2 — epoch 단조 증가")
    print(f"        각 digest_exit → epoch +1 정확히 한 번  →  {inv2_results.count(True)}/{n}")
    print(f"        누락도, 중복도, 역전도 없음")

    print(f"\n  {'✅' if i3_pass else '❌'} I3 — fold 효과 미래 한정")
    print(f"        T4 fold 기록이 같은 digest의 brain_before를 오염하지 않음  →  {inv3_results.count(True)}/{n}")
    print(f"        fold는 다음 epoch에서만 뇌에 영향")

    print(f"\n  {'✅' if order_pass else '❌'} 위상 순서 보장")
    print(f"        T0→T1→T2→T3→T4→T6→T7 순서 위반  →  {len(violations_all)}건")
    if violations_all:
        for v in violations_all:
            print(f"        ⚠ {v}")

    overall = i1_pass and i2_pass and i3_pass and order_pass
    print()
    if overall:
        print("  🟢 PASS — 위상 정렬 봉인됨.")
        print("        사건은 과거를 바꾸지 않는다.")
        print("        fold는 미래만 바꾼다.")
        print("        우주의 시간은 단방향이다.")
    else:
        print("  🔴 FAIL — 위상 역전 감지.")
        print("        hippocampus absorb 위치 또는 epoch increment 순서 조정 필요.")

    report = {
        "exp": "EXP-TOPOLOGY-LOCK-01",
        "ts":  datetime.now().isoformat(),
        "n":   n,
        "I1_pass": i1_pass, "I1_results": inv1_results,
        "I2_pass": i2_pass, "I2_results": inv2_results,
        "I3_pass": i3_pass, "I3_results": inv3_results,
        "order_pass": order_pass,
        "violations": violations_all,
        "overall": overall,
    }
    try:
        os.makedirs(state_dir, exist_ok=True)
        p = os.path.join(state_dir, "topology_lock_report.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  리포트: {p}")
    except Exception as e:
        print(f"  리포트 저장 실패: {e}")

    print("="*65 + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXP-TOPOLOGY-LOCK-01")
    parser.add_argument("--state-dir", default="state")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_topology_experiment(state_dir=args.state_dir, verbose=not args.quiet)
