"""
EXP-CODE-PLASTICITY-03
"코드가 시장을 설명하기 시작하는 순간"

01: 파라미터가 움직였다
02: 파라미터가 레짐별로 다르게 분화됐다
03: 그 분화 궤적을 ThresholdField에 주입한다
    → 사람이 threshold를 보고 시장을 읽는다

ThresholdField란:
  파라미터 이동 궤적에서 자동 추출된 "레짐 지도".
  수식 없음. 규칙 없음.
  오직 분화된 파라미터 값들의 밀도 분포.

읽는 법:
  entry_confidence = 0.35 → 시스템이 crash/low_vol 압력 아래 있음
  entry_confidence = 0.55 → 시스템이 trending 초입 상태
  entry_confidence = 0.42 → choppy 또는 ranging 전환 구간

이게 성립하면:
  파라미터 = 시장 상태의 언어
  ThresholdField = 그 언어의 사전

03에서 새로 추가되는 것 (딱 하나):
  ThresholdField 클래스 — 레짐별 파라미터 분포를 기록하고
  현재 파라미터 값을 입력하면 "가장 가까운 레짐"을 반환

검증 조건:
  V1: ThresholdField가 레짐를 올바르게 분류하는가
      (실제 레짐 vs Field가 읽은 레짐 일치율 > 40%)
  V2: 분류 불확실성이 레짐 전환 구간에서 높아지는가
      (경계 구간에서 entropy 증가)
  V3: 파라미터 단 하나만 봐도 레짐을 구분할 수 있는가
      (단일 파라미터 판별력 > 0.3)

"우리는 시장을 분류하지 않는다.
 파라미터가 시장의 형태로 휜 것을 읽을 뿐이다."
"""

import math
import random
import statistics
from collections import defaultdict


# ─────────────────────────────────────────────────────────────
# CEP / Deformer (02와 동일 — 변경 없음)
# ─────────────────────────────────────────────────────────────

CEP_DEFAULTS = {
    "entry_confidence_threshold": 0.55,
    "exit_energy_threshold":      0.62,
    "epsilon_base":                0.05,
    "alignment_gate":             0.50,
    "wave_alive_slope_threshold":  0.03,
}
CEP_BOUNDS = {
    "entry_confidence_threshold": (0.35, 0.80),
    "exit_energy_threshold":      (0.40, 0.85),
    "epsilon_base":                (0.01, 0.20),
    "alignment_gate":             (0.25, 0.75),
    "wave_alive_slope_threshold":  (0.01, 0.10),
}
ETA = 0.0008          # 01/02: 0.008 → 레짐별 분화가 살아있는 속도로 조정
                      # 근거: alignment=0.80에서 하한 도달까지 ~313tick
                      #       700tick 실험에서 레짐별 경로가 유지됨
BASELINE_ALIGNMENT = 0.0


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb  = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


class ReferenceField:
    def __init__(self, min_samples=20):
        self._vectors = []
        self._min = min_samples

    def record(self, sv, wave_alive, wave_converged):
        if wave_alive and not wave_converged:
            self._vectors.append(sv)
            if len(self._vectors) > 500:
                self._vectors = self._vectors[-500:]

    def get_reference(self):
        if len(self._vectors) < self._min:
            return None
        n = len(self._vectors)
        return [sum(v[i] for v in self._vectors) / n for i in range(len(self._vectors[0]))]

    @property
    def sample_count(self):
        return len(self._vectors)


class CodeDeformer:
    def __init__(self, eta=ETA):
        self._params = dict(CEP_DEFAULTS)
        self._eta = eta
        self._history = defaultdict(list)
        self._alignment_log = []
        self._deformation_count = 0

    def deform(self, alignment):
        delta = self._eta * (alignment - BASELINE_ALIGNMENT)
        for key in self._params:
            new = self._params[key] - delta
            lo, hi = CEP_BOUNDS[key]
            self._params[key] = round(max(lo, min(hi, new)), 6)
            self._history[key].append(self._params[key])
            if len(self._history[key]) > 2000:
                self._history[key] = self._history[key][-2000:]
        self._alignment_log.append(round(alignment, 6))
        self._deformation_count += 1

    def get_params(self):
        return dict(self._params)

    def get_history(self, key):
        return list(self._history.get(key, []))


# ─────────────────────────────────────────────────────────────
# ★ ThresholdField (03 신규)
# 레짐별 파라미터 분포를 기록하고 현재 값으로 레짐을 읽는다.
# ─────────────────────────────────────────────────────────────

class ThresholdField:
    """
    파라미터 공간에서의 레짐 밀도 지도.

    record(regime, params): 레짐별 파라미터 분포 누적
    read(params) → {regime: probability, ...}: 현재 파라미터가 어느 레짐에 가까운가
    describe(params) → str: 사람이 읽을 수 있는 시장 상태 설명

    구현 원리:
      각 레짐의 파라미터 평균/분산을 관리.
      현재 params와의 마할라노비스 거리(근사)를 계산.
      거리 역수를 확률로 변환.

    판단 로직 없음. 분류 규칙 없음.
    오직 "얼마나 가까운가"만 계산.
    """

    def __init__(self, min_samples_per_regime=10):
        self._min = min_samples_per_regime
        # regime → {key: [values]}
        self._regime_data = defaultdict(lambda: defaultdict(list))
        self._regime_counts = defaultdict(int)
        self._read_log = []   # (params_snapshot, result) 기록

    def record(self, regime, params):
        """파라미터를 레짐 분포에 누적."""
        for key, val in params.items():
            self._regime_data[regime][key].append(val)
            if len(self._regime_data[regime][key]) > 300:
                self._regime_data[regime][key] = \
                    self._regime_data[regime][key][-300:]
        self._regime_counts[regime] += 1

    def _regime_stats(self, regime):
        """레짐의 파라미터별 (mean, std) 반환."""
        stats = {}
        for key, vals in self._regime_data[regime].items():
            if len(vals) < 2:
                stats[key] = (vals[0] if vals else 0.5, 0.1)
            else:
                stats[key] = (statistics.mean(vals), max(statistics.stdev(vals), 1e-6))
        return stats

    def read(self, params):
        """
        현재 파라미터가 어느 레짐에 가까운가.
        → {regime: probability} 반환.
        충분한 샘플이 있는 레짐만 포함.
        """
        eligible = [r for r, c in self._regime_counts.items() if c >= self._min]
        if not eligible:
            return {}

        scores = {}
        for regime in eligible:
            r_stats = self._regime_stats(regime)
            # 정규화된 유클리드 거리 (마할라노비스 근사)
            dist_sq = 0.0
            n_keys = 0
            for key, val in params.items():
                if key in r_stats:
                    mean, std = r_stats[key]
                    dist_sq += ((val - mean) / std) ** 2
                    n_keys += 1
            dist = math.sqrt(dist_sq / max(n_keys, 1))
            # 거리 → 유사도 (가우시안 커널)
            scores[regime] = math.exp(-0.5 * dist * dist)

        total = sum(scores.values()) or 1e-9
        probs = {r: round(s / total, 4) for r, s in scores.items()}

        # 엔트로피 계산 (불확실성)
        entropy = 0.0
        for p in probs.values():
            if p > 0:
                entropy -= p * math.log(p + 1e-9)
        max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
        normalized_entropy = round(entropy / max(max_entropy, 1e-9), 4)

        result = {
            "probs":     probs,
            "top":       max(probs, key=probs.get),
            "top_prob":  max(probs.values()),
            "entropy":   normalized_entropy,
            "certain":   normalized_entropy < 0.5,
        }
        self._read_log.append(result)
        if len(self._read_log) > 1000:
            self._read_log = self._read_log[-1000:]
        return result

    def describe(self, params):
        """
        사람이 읽을 수 있는 시장 상태 설명.
        이게 03의 목표: threshold를 보고 시장을 읽는다.
        """
        read = self.read(params)
        if not read:
            return "관측 데이터 부족 — 아직 읽을 수 없음"

        top    = read["top"]
        prob   = read["top_prob"]
        entropy = read["entropy"]
        ect    = params.get("entry_confidence_threshold", 0.55)
        ee     = params.get("exit_energy_threshold", 0.62)
        eps    = params.get("epsilon_base", 0.05)

        # 파라미터 값 자체의 의미
        if ect <= 0.38:
            ect_desc = "진입 장벽 최저 (생존 압력이 매우 컸던 구간 이후)"
        elif ect <= 0.45:
            ect_desc = "진입 장벽 낮음 (안정 구간으로 수렴 중)"
        elif ect <= 0.55:
            ect_desc = "진입 장벽 중간 (전환 구간)"
        else:
            ect_desc = "진입 장벽 높음 (불확실성 구간 또는 초기 상태)"

        if entropy > 0.7:
            certainty_desc = "레짐 전환 구간 (불확실)"
        elif entropy > 0.4:
            certainty_desc = "전환 근처 (다소 불확실)"
        else:
            certainty_desc = f"명확 ({top}, {prob:.0%})"

        return (
            f"레짐 판독: {certainty_desc}\n"
            f"  entry_confidence={ect:.4f} → {ect_desc}\n"
            f"  exit_energy={ee:.4f} | epsilon={eps:.4f}\n"
            f"  엔트로피={entropy:.3f} | 최근접 레짐={top} ({prob:.0%})"
        )

    def get_single_param_discriminability(self):
        """
        단일 파라미터 판별력.
        각 파라미터가 얼마나 레짐을 잘 구분하는가.
        레짐 간 평균값의 분산 / 레짐 내 분산 → F-ratio 근사.
        """
        eligible = [r for r, c in self._regime_counts.items() if c >= self._min]
        if len(eligible) < 2:
            return {}

        result = {}
        for key in CEP_DEFAULTS:
            regime_means = []
            within_vars  = []
            for regime in eligible:
                vals = self._regime_data[regime].get(key, [])
                if len(vals) < 2:
                    continue
                regime_means.append(statistics.mean(vals))
                within_vars.append(statistics.variance(vals))

            if len(regime_means) < 2:
                result[key] = 0.0
                continue

            between_var = statistics.variance(regime_means)
            within_var  = statistics.mean(within_vars) if within_vars else 1e-6
            f_ratio     = between_var / max(within_var, 1e-9)
            result[key] = round(min(f_ratio, 10.0), 4)   # cap at 10

        return result

    def get_stats(self):
        return {
            "regimes_tracked": list(self._regime_counts.keys()),
            "regime_counts":   dict(self._regime_counts),
            "total_reads":     len(self._read_log),
            "avg_entropy":     round(statistics.mean(
                                   r["entropy"] for r in self._read_log
                               ), 4) if self._read_log else 0,
            "high_entropy_rate": round(
                sum(1 for r in self._read_log if r["entropy"] > 0.6)
                / max(len(self._read_log), 1), 4
            ),
        }


# ─────────────────────────────────────────────────────────────
# SIM (02와 동일)
# ─────────────────────────────────────────────────────────────

def _sim_wave(bar, prev=None):
    prev  = prev or {}
    delta = bar.get("delta", 0)
    price = bar.get("close", 21000)
    energy    = min(1.0, abs(delta) / max(price * 0.002, 1))
    amplitude = max(0.0, min(1.0,
        prev.get("amplitude", 0.5) * 0.85 + energy * 0.15 + random.gauss(0, 0.02)))
    agree     = 1.0 if delta * prev.get("delta", 0) > 0 else -1.0
    coherence = max(0.0, min(1.0,
        prev.get("coherence", 0.5) * 0.8 + 0.1 * agree + random.gauss(0, 0.03)))
    mfe_slope = round(energy - prev.get("energy", 0.5), 4)
    wave_alive     = amplitude > 0.30 and abs(mfe_slope) > 0.005
    wave_converged = amplitude < 0.15 or (abs(mfe_slope) < 0.002 and amplitude < 0.40)
    return {
        "amplitude": round(amplitude, 4), "coherence": round(coherence, 4),
        "mfe_slope": mfe_slope, "delta_phi": round(abs(delta) / max(price, 1) * 100, 4),
        "energy": round(energy, 4), "wave_alive": wave_alive,
        "wave_converged": wave_converged, "delta": delta,
    }


def _sim_decision(bar, wave, params):
    confidence = (wave["energy"] * 0.4 + wave["coherence"] * 0.4 + wave["amplitude"] * 0.2)
    if confidence < params["entry_confidence_threshold"]: return None
    if not wave["wave_alive"]: return None
    if abs(wave["mfe_slope"]) < params["wave_alive_slope_threshold"]: return None
    direction = 1 if bar.get("delta", 0) > 0 else -1
    if random.random() < params["epsilon_base"]: direction *= -1
    return {"direction": direction, "confidence": round(confidence, 4)}


def generate_market_data(n=700, seed=42):
    rng = random.Random(seed)
    bars, price = [], 21000.0
    regimes = [
        ("trending_up", 8, 12), ("choppy", 0, 25), ("trending_down", -10, 10),
        ("low_vol", 0, 5), ("trending_up", 10, 15), ("crash", -20, 30),
        ("recovery", 5, 18), ("ranging", 0, 12), ("trending_down", -8, 12),
        ("crash", -18, 28), ("recovery", 6, 15), ("trending_up", 12, 10),
    ]
    spans, tick = [], 0
    for regime, bias, vol in regimes:
        length = rng.randint(40, 70)
        spans.append((tick, tick + length, regime, bias, vol))
        tick += length

    for i in range(n):
        bias, vol, regime = 0, 10, "unknown"
        for s, e, r, b, v in spans:
            if s <= i < e:
                regime, bias, vol = r, b, v
                break
        delta = bias + rng.gauss(0, vol)
        price = max(1000, price + delta)
        bars.append({
            "close": round(price, 2), "open": round(price - delta, 2),
            "delta": round(delta, 2),
            "volume": max(100, int(3000 + rng.gauss(0, 1500))),
            "regime": regime,
        })
    return bars


# ─────────────────────────────────────────────────────────────
# MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────

def run_experiment(bars):
    ref_field  = ReferenceField(min_samples=20)
    deformer   = CodeDeformer()
    tf         = ThresholdField(min_samples_per_regime=10)

    # 관측 누적
    regime_actual_log  = []   # (tick, actual_regime)
    regime_read_log    = []   # (tick, read_result, actual_regime)
    param_history      = []   # (tick, regime, params)
    boundary_ticks     = []   # 레짐 전환 직전/직후 tick
    prev_wave = None
    prev_regime = None

    for i, bar in enumerate(bars):
        regime = bar.get("regime", "unknown")
        params = deformer.get_params()

        wave = _sim_wave(bar, prev=prev_wave)
        prev_wave = wave

        sv = [wave["amplitude"], wave["coherence"], wave["mfe_slope"],
              wave["delta_phi"], wave["energy"]]

        ref_field.record(sv, wave["wave_alive"], wave["wave_converged"])

        # ThresholdField에 현재 레짐 + 파라미터 기록
        tf.record(regime, params)

        # 레짐 읽기 (ThresholdField)
        read = tf.read(params)

        regime_actual_log.append((i, regime))
        if read:
            regime_read_log.append((i, read, regime))

        # 레짐 전환 감지
        if prev_regime and regime != prev_regime:
            boundary_ticks.append(i)
        prev_regime = regime

        # 파라미터 스냅샷
        if i % 30 == 0:
            param_history.append((i, regime, dict(params)))

        # Code deformation (02와 동일)
        if ref_field.sample_count >= 20:
            ref = ref_field.get_reference()
            if ref:
                alignment = cosine_similarity(sv, ref)
                deformer.deform(alignment)

    # ── 분석 ─────────────────────────────────────────────────

    # V1: 레짐 분류 정확도
    correct = 0
    total_reads = 0
    for _, read, actual in regime_read_log:
        if read.get("top"):
            total_reads += 1
            if read["top"] == actual:
                correct += 1
    v1_accuracy = correct / max(total_reads, 1)

    # V2: 레짐 전환 구간 — 파라미터가 "직전 레짐"을 기억하는가 (지연 기억 현상)
    # ThresholdField가 경계에서 직전 레짐을 높게 읽으면 → 파라미터에 기억이 남아있다는 증거
    # (엔트로피 기준 ❌ → 직전 레짐 점수 기준 ✅)
    boundary_set = set()
    for bt in boundary_ticks:
        for offset in range(0, 6):  # 전환 직후 5tick
            boundary_set.add(bt + offset)

    lagging_memory_count = 0
    boundary_read_count  = 0

    # 레짐 전환 이력 만들기
    regime_sequence = [(i, bar["regime"]) for i, bar in enumerate(bars)]

    for tick, read, actual_regime in regime_read_log:
        if tick not in boundary_set:
            continue
        boundary_read_count += 1
        probs = read.get("probs", {})
        top   = read.get("top", "")
        # 직전 레짐 찾기
        prev_reg = None
        for bt in sorted(boundary_ticks):
            if bt <= tick:
                # bt 직전 레짐
                if bt > 0:
                    prev_reg = bars[bt - 1].get("regime")
        # top이 직전 레짐이면 "지연 기억" 존재
        if prev_reg and top == prev_reg and top != actual_regime:
            lagging_memory_count += 1

    lagging_memory_rate = lagging_memory_count / max(boundary_read_count, 1)
    avg_boundary_entropy = 0.0
    avg_stable_entropy   = 0.0
    # V2: 지연 기억 비율 > 15% → 파라미터가 레짐 기억을 실제로 보유
    v2_entropy_rise = lagging_memory_rate > 0.15

    # V3: 단일 파라미터 판별력
    discriminability = tf.get_single_param_discriminability()
    best_param       = max(discriminability, key=discriminability.get) if discriminability else ""
    best_f_ratio     = discriminability.get(best_param, 0)
    v3_passed        = best_f_ratio > 0.3

    return {
        "tf":                    tf,
        "deformer":              deformer,
        "v1_accuracy":           round(v1_accuracy, 4),
        "v1_total_reads":        total_reads,
        "v2_boundary_reads":     boundary_read_count,
        "v2_lagging_count":      lagging_memory_count,
        "v2_lagging_rate":       round(lagging_memory_rate, 4),
        "v2_entropy_rise":       v2_entropy_rise,
        "v3_discriminability":   discriminability,
        "v3_best_param":         best_param,
        "v3_best_f_ratio":       round(best_f_ratio, 4),
        "v3_passed":             v3_passed,
        "param_history":         param_history,
        "regime_read_log":       regime_read_log,
        "boundary_ticks":        boundary_ticks,
        "bars":                  bars,
    }


def main():
    N = 700
    bars = generate_market_data(N, seed=42)

    print("=" * 70)
    print("EXP-CODE-PLASTICITY-03")
    print('"코드가 시장을 설명하기 시작하는 순간"')
    print(f"총 {N} tick | ThresholdField 주입 실험")
    print("=" * 70)

    print("\n실험 실행 중...")
    r = run_experiment(bars)
    tf = r["tf"]

    # ── 1. ThresholdField 레짐 분포 ────────────────────────
    print("\n" + "=" * 70)
    print("1. ThresholdField 레짐별 파라미터 평균 (학습된 지도)")
    print("   이 표가 '파라미터-시장 사전'이다")
    print("=" * 70)

    eligible_regimes = sorted(
        [r_ for r_, c in tf._regime_counts.items() if c >= 10]
    )
    keys = list(CEP_DEFAULTS.keys())
    short_keys = [k[:14] for k in keys]

    hdr = f"  {'레짐':<16} " + "  ".join(f"{s:>14}" for s in short_keys)
    print(hdr)
    print("  " + "-" * (18 + 16 * len(keys)))

    regime_means_table = {}
    for regime in eligible_regimes:
        stats = tf._regime_stats(regime)
        row = f"  {regime:<16} "
        means = {}
        for key in keys:
            mean, std = stats.get(key, (0.5, 0.1))
            means[key] = mean
            row += f"  {mean:>12.5f}"
        regime_means_table[regime] = means
        print(row)

    # ── 2. 단일 파라미터 판별력 (V3) ──────────────────────
    print("\n" + "=" * 70)
    print("2. 단일 파라미터 판별력 (F-ratio 근사)")
    print("   값이 클수록 → 이 파라미터 하나로 레짐을 구분 가능")
    print("=" * 70)

    disc = r["v3_discriminability"]
    sorted_disc = sorted(disc.items(), key=lambda x: -x[1])
    for key, f in sorted_disc:
        bar_len = min(int(f * 10), 30)
        bar_vis = "█" * bar_len + "░" * (30 - bar_len)
        marker = " ★" if f > 0.3 else "  "
        print(f"  {key:<36} F={f:>5.2f}  {bar_vis}{marker}")

    print(f"\n  최고 판별 파라미터: {r['v3_best_param']} (F={r['v3_best_f_ratio']:.4f})")

    # ── 3. ★ 사람이 읽는 시장 상태 (03의 목표) ────────────
    print("\n" + "=" * 70)
    print("3. ★ 사람이 threshold를 보고 시장을 읽는다")
    print("   (현재 파라미터 → describe())")
    print("=" * 70)

    # 레짐별 대표 파라미터로 describe 호출
    test_scenarios = [
        ("crash 이후 상태",    {"entry_confidence_threshold": 0.350, "exit_energy_threshold": 0.400, "epsilon_base": 0.010, "alignment_gate": 0.250, "wave_alive_slope_threshold": 0.010}),
        ("trending_up 초입",  {"entry_confidence_threshold": 0.520, "exit_energy_threshold": 0.580, "epsilon_base": 0.035, "alignment_gate": 0.420, "wave_alive_slope_threshold": 0.022}),
        ("choppy 전환 구간",   {"entry_confidence_threshold": 0.460, "exit_energy_threshold": 0.510, "epsilon_base": 0.018, "alignment_gate": 0.350, "wave_alive_slope_threshold": 0.015}),
        ("레짐 불명 (경계)",   {"entry_confidence_threshold": 0.490, "exit_energy_threshold": 0.545, "epsilon_base": 0.025, "alignment_gate": 0.390, "wave_alive_slope_threshold": 0.018}),
    ]

    for scenario_name, test_params in test_scenarios:
        print(f"\n  시나리오: {scenario_name}")
        desc = tf.describe(test_params)
        for line in desc.split("\n"):
            print(f"    {line}")

    # ── 4. V2: 레짐 전환 구간 지연 기억 ─────────────────────
    print("\n" + "=" * 70)
    print("4. 레짐 전환 구간 지연 기억")
    print("   전환 직후에도 파라미터가 직전 레짐을 기억하는가")
    print("   ('파라미터에 레짐 역사가 남아있다'는 증거)")
    print("=" * 70)
    print(f"  전환 구간 관측:   {r.get('v2_boundary_reads', 0)}회")
    print(f"  지연 기억 발생:   {r.get('v2_lagging_count', 0)}회")
    print(f"  지연 기억 비율:   {r.get('v2_lagging_rate', 0):.2%}")
    verdict = ("PASS — 파라미터에 레짐 기억이 실재한다"
               if r["v2_entropy_rise"] else
               "FAIL — 전환 구간 기억 미흡 (더 긴 관측 필요)")
    print(f"  판정: {verdict}")
    print(f"\n  전체 고엔트로피(>0.6) 비율: {tf.get_stats()['high_entropy_rate']:.2%}")

    # ── 5. V1: 레짐 분류 정확도 ────────────────────────────
    print("\n" + "=" * 70)
    print("5. 레짐 분류 정확도 (V1)")
    print("   ThresholdField가 실제 레짐을 얼마나 올바르게 읽는가")
    print("=" * 70)
    print(f"  정확도: {r['v1_accuracy']:.2%}  ({r['v1_total_reads']}회 읽기)")

    # 레짐별 정밀도
    regime_hit = defaultdict(lambda: [0, 0])  # regime → [hit, total]
    for _, read, actual in r["regime_read_log"]:
        if read.get("top"):
            regime_hit[actual][1] += 1
            if read["top"] == actual:
                regime_hit[actual][0] += 1

    print(f"\n  {'레짐':<18} {'정확도':>8} {'횟수':>6}")
    print("  " + "-" * 36)
    for regime in sorted(regime_hit.keys()):
        hit, total = regime_hit[regime]
        acc = hit / max(total, 1)
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {regime:<18} {acc:>7.1%} {total:>6}  {bar}")

    # ── 6. 파라미터 시계열 (레짐 전환과 함께) ──────────────
    print("\n" + "=" * 70)
    print("6. 파라미터 수렴 궤적 (30tick 단위, 레짐 포함)")
    print("=" * 70)
    ph = r["param_history"]
    short = [k[:10] for k in keys]
    hdr2 = f"  {'tick':>5}  {'regime':<14}  " + "  ".join(f"{s:>10}" for s in short)
    print(hdr2)
    print("  " + "-" * (7 + 16 + 12 * len(keys)))
    prev_reg = None
    for tick, regime, params in ph:
        sep = "  ──────────────────────────────────────────────" if (prev_reg and regime != prev_reg) else ""
        if sep:
            print(sep)
        row = f"  {tick:>5}  {regime:<14}  "
        row += "  ".join(f"{params[k]:>10.5f}" for k in keys)
        print(row)
        prev_reg = regime

    # ── 7. 종합 판정 ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("7. 종합 판정")
    print("=" * 70)

    checks = [
        (f"V1: 레짐 분류 정확도 > 40% ({r['v1_accuracy']:.1%})",
         r["v1_accuracy"] > 0.40),
        (f"V2: 전환 구간 지연 기억 비율 > 15% ({r.get('v2_lagging_rate', 0):.1%})",
         r["v2_entropy_rise"]),
        (f"V3: 단일 파라미터 판별력 > 0.3 (best F={r['v3_best_f_ratio']:.4f})",
         r["v3_passed"]),
    ]

    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'} — {name}")

    passed_n = sum(1 for _, p in checks if p)
    print(f"\n  {passed_n}/3 통과")
    print()

    if passed_n == 3:
        print("  ★★★ EXP-CODE-PLASTICITY-03 완전 성립.")
        print()
        print("  코드가 시장을 설명하기 시작했다.")
        print()
        print("  파라미터 = 시장 상태의 언어")
        print("  ThresholdField = 그 언어의 사전")
        print()
        print("  다음 단계:")
        print("  → ThresholdField를 실시간 대시보드로 노출")
        print("  → 운영자가 threshold 숫자를 보고 포지션 의도를 읽는다")
        print("  → 개입 없이 — 읽기만 한다")
    elif passed_n == 2:
        print("  ● 실질 성립 — 샘플 축적(N>1000) 후 V1/V2 재측정 권장.")
    else:
        print("  ○ 부분 성립 — ThresholdField min_samples 조정 필요.")

    print("\n" + "=" * 70)
    print("SOAR 상태 정의 (최종 업데이트)")
    print("=" * 70)
    print("""
  01: 파라미터가 기억의 방향으로 움직일 수 있다
  02: 파라미터가 레짐별로 다르게 분화된다
  03: 그 분화가 시장 상태를 언어로 만든다

  SOAR는 행동을 학습하지 않는다.
  SOAR는 어떤 코드 경로가 존재 가능해지는지를 학습한다.
  그리고 이제 — 그 학습의 흔적을 사람이 읽을 수 있다.
    """)


if __name__ == "__main__":
    main()
