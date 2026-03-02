# -*- coding: utf-8 -*-
"""
tesla_field.py — 비가역 테슬라 공명장 (Irreversible Tesla Resonance Field)

======================================================================
핵심 원칙
======================================================================

  학습 ❌    평균 ❌    파라미터 튜닝 ❌
  비가역 ✅  위상 판례 ✅  척수 반사 ✅

"과거에 구조적으로 틀렸던 위상이
 다시는 같은 각도로 진입하지 못하게 막는다."

======================================================================
작동 원리
======================================================================

STEP 1 — 위상 추출 (Windmill Phase Extractor)
  Entry 시점의 (state, action, conf_bin, shadow_phase, grammar_event)
  → 5차원 위상 튜플 (이산화, 가격 없음)

STEP 2 — 기록 (Police 판결 시만)
  verdict ∈ {STRUCTURAL_LOSS, AVOIDABLE_LOSS, TAIL_EVENT}
  → TeslaField.record(phase, severity)
  severity는 평균 ❌  max + decay (비가역 성질 유지)

STEP 3 — 공명 체크 (Entry 직전)
  TeslaField.resonates(current_phase)
  → similarity ≥ THR (0.80) 이면 (True, weight)
  → weight 기반 size_scale 감소 + Dopamine 감소

======================================================================
위상 정의 (5축)
======================================================================

  axis 0: state_key      e.g. "UP_HVOL_VOLATILE|short_d1"
  axis 1: conf_bin       int [0,4]  (conf_rel 이산화)
  axis 2: rcb_sign       int -1/0/+1 (rcb < -0.5 → -1, > +0.5 → +1, else 0)
  axis 3: shadow_phase   str "HOLD"/"STALL"/"BREAK"
  axis 4: drift_bin      int 0/1/2  (drift < 0.25 → 0, < 0.60 → 1, else 2)

축별 유사도 가중치 (합=1.0):
  state_key    0.35   ← 가장 중요 (구조 자체)
  conf_bin     0.20
  rcb_sign     0.20
  shadow_phase 0.15
  drift_bin    0.10

======================================================================
비가역 특성 (중요)
======================================================================

  저장 방식: max + decay (평균 ❌)
    weight[phase] = max(weight[phase] * DECAY, new_severity)
  → 한 번 기록된 판례는 천천히 사라지지, 평균으로 희석 안 됨
  → 같은 판례 반복 시 weight가 쌓임 (누적 → 더 강한 차단)

======================================================================
사용법
======================================================================

  # 인스턴스화 (run_live_orchestrator.__init__)
  self._tesla = TeslaField(state_dir="state")

  # Entry 직전 공명 체크
  phase = TeslaField.extract_phase(state_key, conf_rel, rcb, shadow_phase, drift)
  hit, w = self._tesla.resonates(phase)
  if hit:
      size *= (1 - w)
      dopamine -= 0.05 * w

  # Police 판결 후 기록 (_close_position 후)
  self._tesla.record(phase, severity, verdict)

ZONE: SOFT — Entry 절대 차단 없음. 반드시 size_scale 감소만.
"""

import json
import math
import os
from typing import Optional, Tuple


# ── 파라미터 ──────────────────────────────────────────────────────────────────
FIELD_DECAY      = 0.97    # 매 World 후 weight 자연 감쇠 (비가역 완화 속도)
SIM_THRESHOLD    = 0.80    # 공명 임계 (이 이상 유사하면 Hit)
MAX_FIELD_SIZE   = 500     # 최대 저장 위상 수
SAVE_EVERY       = 20      # N판결 마다 자동 저장
FILENAME         = "tesla_field_snapshot.json"

# 축별 유사도 가중치 (합=1.0)
W_STATE  = 0.35
W_CONF   = 0.20
W_RCB    = 0.20
W_SHADOW = 0.15
W_DRIFT  = 0.10

# 판결 severity 기본값 (CaseReviewer가 severity를 안 주는 경우 폴백)
VERDICT_SEVERITY = {
    "STRUCTURAL_LOSS": 0.75,
    "AVOIDABLE_LOSS":  0.55,
    "DIRTY_WIN":       0.35,
    "TAIL_EVENT":      0.65,
    "SURVIVED_LUCK":   0.40,
}

# 공명 발동 조건 — 이 판결 종류만 기록 + 차단
RESONANCE_VERDICTS = frozenset({
    "STRUCTURAL_LOSS",
    "AVOIDABLE_LOSS",
    "TAIL_EVENT",
})


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _conf_bin(conf_rel: float) -> int:
    """conf_rel [0,1] → bin [0,4]"""
    return max(0, min(4, int(conf_rel * 5)))


def _rcb_sign(rcb: float) -> int:
    """-1 / 0 / +1"""
    if rcb < -0.50:
        return -1
    if rcb > +0.50:
        return +1
    return 0


def _drift_bin(drift: float) -> int:
    """drift_decay → 0/1/2"""
    if drift < 0.25:
        return 0
    if drift < 0.60:
        return 1
    return 2


def _shadow_phase(coh: float, rcb: float, stall_duration: float = 0.0) -> str:
    """
    Shadow 위상을 3단계로 이산화.
    HOLD:  coh 높고 rcb 중립 — 아직 살아있음
    STALL: 변화율 없음 (stall_duration 기반)
    BREAK: coh 낮거나 rcb 극단
    """
    if stall_duration >= 20.0:
        return "STALL"
    if coh < 0.55 or abs(rcb) > 0.75:
        return "BREAK"
    return "HOLD"


# ── 핵심 클래스 ───────────────────────────────────────────────────────────────

class TeslaField:
    """
    비가역 테슬라 공명장.
    판례 위상을 저장하고, 새 Entry의 공명 여부를 판단한다.
    """

    def __init__(self,
                 state_dir:      str   = "state",
                 sim_threshold:  float = SIM_THRESHOLD,
                 field_decay:    float = FIELD_DECAY,
                 save_every:     int   = SAVE_EVERY):

        self.state_dir     = state_dir
        self.sim_threshold = sim_threshold
        self.field_decay   = field_decay
        self.save_every    = save_every

        # 공명장 본체: phase_key → weight (0~1)
        self._field: dict = {}

        # 메타 통계
        self._total_records = 0
        self._total_hits    = 0
        self._total_checked = 0
        self._recent_hits:  list = []   # 최근 50 hit 기록

        self._save_path = os.path.join(state_dir, FILENAME)
        self._load()

    # ── 위상 추출 (정적 유틸) ─────────────────────────────────────────────────

    @staticmethod
    def extract_phase(
        state_key:    str,
        conf_rel:     float,
        rcb:          float,
        coh:          float,
        drift:        float,
        stall_dur:    float = 0.0,
    ) -> tuple:
        """
        Entry 시점 피처 → 5차원 위상 튜플.

        이 위상은 (state_key, conf_bin, rcb_sign, shadow_phase, drift_bin)으로
        이산화됨. 가격 없음. 확률 없음. PnL 없음.
        """
        return (
            str(state_key),
            _conf_bin(conf_rel),
            _rcb_sign(rcb),
            _shadow_phase(coh, rcb, stall_dur),
            _drift_bin(drift),
        )

    # ── 공명 유사도 계산 ──────────────────────────────────────────────────────

    @staticmethod
    def similarity(phase_a: tuple, phase_b: tuple) -> float:
        """
        두 위상의 유사도 [0,1].
        축별 가중치 합산 — 이산값이므로 일치/불일치만 판단.
        """
        if len(phase_a) < 5 or len(phase_b) < 5:
            return 0.0

        score = 0.0
        # axis 0: state_key (exact match → full weight)
        score += W_STATE  * (1.0 if phase_a[0] == phase_b[0] else 0.0)
        # axis 1: conf_bin (|diff| ≤ 1 → partial credit)
        score += W_CONF   * max(0.0, 1.0 - abs(phase_a[1] - phase_b[1]) * 0.5)
        # axis 2: rcb_sign
        score += W_RCB    * (1.0 if phase_a[2] == phase_b[2] else 0.0)
        # axis 3: shadow_phase
        score += W_SHADOW * (1.0 if phase_a[3] == phase_b[3] else 0.0)
        # axis 4: drift_bin
        score += W_DRIFT  * max(0.0, 1.0 - abs(phase_a[4] - phase_b[4]) * 0.5)

        return round(score, 4)

    # ── 핵심 API ──────────────────────────────────────────────────────────────

    def resonates(self, phase: tuple) -> Tuple[bool, float]:
        """
        Entry 직전 공명 체크.

        Returns:
            (hit: bool, weight: float)
            hit    = 유사도 ≥ SIM_THRESHOLD 인 판례가 존재
            weight = 가장 강한 매칭의 field weight (차단 강도)
        """
        self._total_checked += 1

        best_sim   = 0.0
        best_weight = 0.0

        for past_key, w in self._field.items():
            past_phase = _phase_from_key(past_key)
            sim = self.similarity(phase, past_phase)
            if sim >= self.sim_threshold:
                effective = sim * w
                if effective > best_weight:
                    best_sim    = sim
                    best_weight = effective

        if best_weight > 0:
            self._total_hits += 1
            self._recent_hits.append({
                "phase":   phase,
                "sim":     round(best_sim, 4),
                "weight":  round(best_weight, 4),
            })
            if len(self._recent_hits) > 50:
                self._recent_hits = self._recent_hits[-50:]
            return True, round(best_weight, 4)

        return False, 0.0

    def record(self, phase: tuple, severity: float, verdict: str = "") -> None:
        """
        Police 판결 후 공명장에 위상을 각인한다.

        비가역 특성:
          weight = max(old * DECAY, severity)
          → 기존 판례가 더 강하면 유지 (덮어씌우기 ❌)
          → 같은 판례 반복 시 severity 합산 효과
        """
        if verdict and verdict not in RESONANCE_VERDICTS:
            return  # CLEAN_WIN, SURVIVED_LUCK 등 차단 불필요

        key = _phase_to_key(phase)
        old = self._field.get(key, 0.0)

        # 비가역: max(감쇠된 기존값, 새 severity)
        new_w = max(old * self.field_decay, float(severity))
        new_w = round(min(1.0, max(0.0, new_w)), 4)
        self._field[key] = new_w

        self._total_records += 1

        # 장 크기 제한 (오래된 약한 판례 제거)
        if len(self._field) > MAX_FIELD_SIZE:
            # weight 낮은 순으로 제거
            items = sorted(self._field.items(), key=lambda x: x[1])
            for k, _ in items[:len(self._field) - MAX_FIELD_SIZE]:
                del self._field[k]

        if self._total_records % self.save_every == 0:
            self.save()

    def decay_all(self) -> None:
        """
        매 World 종료 후 호출 — 전체 field weight 자연 감쇠.
        시간이 지날수록 판례가 희미해짐 (완전한 비가역 ❌, 천천히 잊음).
        """
        for k in list(self._field.keys()):
            self._field[k] = round(self._field[k] * self.field_decay, 4)
            if self._field[k] < 0.05:
                del self._field[k]

    # ── 저장 / 복원 ──────────────────────────────────────────────────────────

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._save_path)),
                        exist_ok=True)
            data = {
                "version":        "1",
                "field":          dict(self._field),
                "total_records":  self._total_records,
                "total_hits":     self._total_hits,
                "total_checked":  self._total_checked,
                "sim_threshold":  self.sim_threshold,
                "field_decay":    self.field_decay,
            }
            tmp = self._save_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._save_path)
        except Exception as e:
            print(f"  [TeslaField] save error: {e}")

    def _load(self) -> None:
        if not os.path.exists(self._save_path):
            return
        try:
            with open(self._save_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            self._field         = dict(d.get("field", {}))
            self._total_records = int(d.get("total_records", 0))
            self._total_hits    = int(d.get("total_hits", 0))
            self._total_checked = int(d.get("total_checked", 0))
            n = len(self._field)
            print(f"  [TeslaField] loaded: {self._total_records} records  "
                  f"{n} phases  hits={self._total_hits}")
        except Exception as e:
            print(f"  [TeslaField] load error: {e}")

    # ── 통계 / 디버그 ─────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        weights = list(self._field.values())
        return {
            "n_phases":      len(self._field),
            "total_records": self._total_records,
            "total_hits":    self._total_hits,
            "total_checked": self._total_checked,
            "hit_rate":      round(self._total_hits / max(self._total_checked, 1), 4),
            "weight_mean":   round(sum(weights) / max(len(weights), 1), 4),
            "weight_max":    round(max(weights) if weights else 0, 4),
        }

    def top_phases(self, n: int = 10) -> list:
        """가장 강한 판례 위상 상위 n개."""
        items = sorted(self._field.items(), key=lambda x: -x[1])
        result = []
        for k, w in items[:n]:
            p = _phase_from_key(k)
            result.append({"phase": p, "weight": round(w, 4)})
        return result

    def print_summary(self) -> None:
        stats = self.get_stats()
        print(f"\n  [TeslaField] phases={stats['n_phases']}  "
              f"hits={stats['total_hits']}/{stats['total_checked']}"
              f"({100*stats['hit_rate']:.1f}%)  "
              f"w_max={stats['weight_max']:.3f}")
        for item in self.top_phases(5):
            p = item["phase"]
            print(f"    w={item['weight']:.3f}  "
                  f"state={p[0] if p else '?'}  "
                  f"conf_bin={p[1] if len(p)>1 else '?'}  "
                  f"rcb_sign={p[2] if len(p)>2 else '?'}  "
                  f"shadow={p[3] if len(p)>3 else '?'}  "
                  f"drift={p[4] if len(p)>4 else '?'}")


# ── 위상 직렬화 헬퍼 ──────────────────────────────────────────────────────────
# 구분자 §  — state_key 내부에 | 가 포함될 수 있으므로 § 사용

_SEP = "§"


def _phase_to_key(phase: tuple) -> str:
    """(state, cb, rs, sp, db) → "state§cb§rs§sp§db" 문자열 키."""
    return _SEP.join(str(v) for v in phase)


def _phase_from_key(key: str) -> tuple:
    """"state§cb§rs§sp§db" → 튜플 (타입 복원)."""
    parts = key.split(_SEP)
    if len(parts) < 5:
        # 구버전 | 구분자 호환 시도
        parts = key.split("|")
        if len(parts) < 5:
            return tuple(parts)
        # | 구분: state|action|cb|rs|sp|db (6부분) → state_key = parts[0]|parts[1]
        if len(parts) == 6:
            try:
                return (
                    f"{parts[0]}|{parts[1]}",
                    int(parts[2]),
                    int(parts[3]),
                    parts[4],
                    int(parts[5]),
                )
            except (ValueError, IndexError):
                return tuple(parts)
        return tuple(parts)
    try:
        return (
            parts[0],           # state_key (str)
            int(parts[1]),      # conf_bin
            int(parts[2]),      # rcb_sign
            parts[3],           # shadow_phase (str)
            int(parts[4]),      # drift_bin
        )
    except (ValueError, IndexError):
        return tuple(parts)


# ── 단독 검증 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    print("── TeslaField v1 검증 ──\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        tf = TeslaField(state_dir=tmpdir, sim_threshold=0.80, field_decay=0.97)

        # 1. 위상 추출 검증
        p1 = TeslaField.extract_phase("UP_HVOL_VOLATILE|short_d1",
                                       conf_rel=0.69, rcb=-0.93,
                                       coh=0.72, drift=0.33)
        p2 = TeslaField.extract_phase("UP_HVOL_VOLATILE|short_d1",
                                       conf_rel=0.65, rcb=-0.88,
                                       coh=0.68, drift=0.40)
        p3 = TeslaField.extract_phase("DOWN_HVOL_RANGING|long_d5",
                                       conf_rel=0.50, rcb=0.20,
                                       coh=0.80, drift=0.20)

        print(f"Phase 1: {p1}")
        print(f"Phase 2: {p2}")
        print(f"Phase 3: {p3}")
        sim12 = TeslaField.similarity(p1, p2)
        sim13 = TeslaField.similarity(p1, p3)
        print(f"sim(p1,p2)={sim12:.3f}  sim(p1,p3)={sim13:.3f}")
        assert sim12 > sim13, "p1-p2 유사도 > p1-p3"

        # 2. 판례 없을 때 공명 없음
        hit, w = tf.resonates(p1)
        assert not hit and w == 0, f"빈 장에서 공명 없어야 함: {hit} {w}"
        print(f"빈 장 공명: {hit} ✅")

        # 3. 판례 기록 후 공명
        tf.record(p1, severity=0.72, verdict="STRUCTURAL_LOSS")
        hit, w = tf.resonates(p1)
        assert hit, "동일 위상 공명"
        print(f"동일 위상 공명: {hit} w={w:.3f} ✅")

        # 4. 유사 위상도 공명
        hit2, w2 = tf.resonates(p2)
        print(f"유사 위상 공명: {hit2} w={w2:.3f}  (sim={sim12:.3f})")
        assert hit2, "유사 위상도 공명해야 함"

        # 5. 다른 위상은 공명 안 함
        hit3, w3 = tf.resonates(p3)
        print(f"다른 위상 공명: {hit3} w={w3:.3f}  (sim={sim13:.3f})")
        assert not hit3, "다른 위상 공명 없어야 함"

        # 6. 비가역 특성 — 반복 기록 시 weight 증가
        old_w = tf._field.get(_phase_to_key(p1), 0)
        tf.record(p1, severity=0.80, verdict="STRUCTURAL_LOSS")
        new_w = tf._field.get(_phase_to_key(p1), 0)
        print(f"비가역 누적: {old_w:.3f} → {new_w:.3f}  (더 크거나 같아야 함)")
        assert new_w >= old_w * tf.field_decay - 0.01

        # 7. 저장/복원
        tf.save()
        tf2 = TeslaField(state_dir=tmpdir)
        assert tf2._total_records == tf._total_records
        hit_r, w_r = tf2.resonates(p1)
        assert hit_r, "복원 후 공명"
        print(f"저장/복원 후 공명: {hit_r} w={w_r:.3f} ✅")

        tf.print_summary()
        print("\nALL PASS ✅")
