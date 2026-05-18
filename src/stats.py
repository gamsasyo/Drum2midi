"""요약 통계 + 스윙 비율 계산."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def summarize(rows: List[Dict], output_txt: Path) -> Dict:
    """
    rows: grid.compute_deviations()의 반환 row 리스트.
    output_txt에 사람이 읽을 요약 저장 + dict로도 반환.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        output_txt.write_text("No drum onsets detected.\n")
        return {}

    lines = []
    lines.append("=" * 64)
    lines.append("DRUM MICROTIMING SUMMARY")
    lines.append("=" * 64)
    lines.append("")

    # 클래스별 통계
    lines.append("[클래스별 deviation 통계]")
    lines.append(f"{'class':<8}{'count':>8}{'mean_8th':>12}{'std_8th':>12}{'mean_16th':>12}{'std_16th':>12}")
    lines.append(f"{'':8}{'':>8}{'(ms)':>12}{'(ms)':>12}{'(ms)':>12}{'(ms)':>12}")

    summary: Dict = {"per_class": {}}
    for cls in ("kick", "snare", "hihat", "tom", "cymbal"):
        sub = df[df.drum_class == cls]
        if sub.empty:
            continue
        m8 = float(sub.deviation_8th_ms.mean())
        s8 = float(sub.deviation_8th_ms.std())
        m16 = float(sub.deviation_16th_ms.mean())
        s16 = float(sub.deviation_16th_ms.std())
        lines.append(f"{cls:<8}{len(sub):>8}{m8:>+12.2f}{s8:>12.2f}{m16:>+12.2f}{s16:>12.2f}")
        summary["per_class"][cls] = {
            "count": len(sub),
            "mean_8th_ms": m8, "std_8th_ms": s8,
            "mean_16th_ms": m16, "std_16th_ms": s16,
        }

    lines.append("")
    lines.append("부호 규약: 양수 = 그리드보다 늦음 (laid-back), 음수 = 일찍 (rushed/pushing)")
    lines.append("")

    # 스윙 비율 (8분음표 쌍의 long/short)
    swing, swing_msg = compute_swing_ratio(df)
    lines.append("[스윙 비율]")
    if swing is None:
        lines.append(f"  계산 불가 — {swing_msg}")
    else:
        lines.append(f"  swing_ratio = {swing:.3f}  "
                     f"({'균등 8분' if swing < 1.10 else 'light swing' if swing < 1.35 else 'triplet swing (~1.5)' if swing < 1.70 else 'shuffle (~2.0)'})")
        lines.append(f"  해석: 8분음표 long/short 길이 비율. 1.0=균등, 1.5=트리플렛, 2.0=완전 셔플.")
        lines.append(f"  진단: {swing_msg}")
        summary["swing_ratio"] = swing
    lines.append("")

    # 푸시/풀 비교
    lines.append("[push/pull 상대 경향]")
    if "kick" in summary["per_class"] and "snare" in summary["per_class"]:
        k = summary["per_class"]["kick"]["mean_8th_ms"]
        s = summary["per_class"]["snare"]["mean_8th_ms"]
        diff = s - k
        if diff > 5:
            lines.append(f"  스네어가 킥보다 평균 {diff:+.2f} ms 늦음 → 백비트 laid-back, 덥/레게 lazy feel 가능성")
        elif diff < -5:
            lines.append(f"  스네어가 킥보다 평균 {diff:+.2f} ms 빠름 → 백비트 pushing, dnb/footwork urgency 가능성")
        else:
            lines.append(f"  킥/스네어 deviation 거의 동일 ({diff:+.2f} ms) → 박자 일관성 강함")
    if "hihat" in summary["per_class"]:
        h = summary["per_class"]["hihat"]["mean_8th_ms"]
        lines.append(f"  하이햇 평균 deviation: {h:+.2f} ms")
        if "kick" in summary["per_class"]:
            k = summary["per_class"]["kick"]["mean_8th_ms"]
            if abs(h - k) > 5:
                lines.append(f"    → 킥과 {h-k:+.2f} ms 차이. 하이햇이 메인 그리드보다 따로 노는 경향.")

    lines.append("")
    lines.append("=" * 64)

    out = "\n".join(lines) + "\n"
    output_txt.write_text(out)
    print(out)
    return summary


def compute_swing_ratio(df: pd.DataFrame) -> tuple[float | None, str]:
    """
    8분음표 swing 비율 (long / short).

    Gap-based 접근:
      1. 8분 그리드의 두 자리(downbeat 8th, upbeat 8th) = frac 분포의 top 2 모드
      2. frac 은 원형(0=1) 이므로 두 모드는 원을 두 호로 나눔.
         호의 길이 비 long/short 가 swing ratio.
         straight 8th = 0.5/0.5 = 1.0
         triplet      = 0.667/0.333 ≈ 2.0  (보통 1.5로 표기)
         shuffle      = 0.75/0.25 = 3.0
      3. 라벨링 ("어느 게 downbeat" vs "upbeat") 는 swing ratio 값에 영향 없음.
    """
    if df.empty or "beat_position" not in df.columns:
        return None, "beat_position 컬럼 없음"
    bp = pd.to_numeric(df.beat_position, errors="coerce").dropna()
    if len(bp) < 16:
        return None, "데이터 부족"

    frac = (bp - np.floor(bp)).values

    # 히스토그램 → top 2 모드 (서로 ≥0.3 떨어진, wraparound 고려)
    hist, edges = np.histogram(frac, bins=40, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2

    def circ_dist(a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, 1.0 - d)

    order = np.argsort(hist)[::-1]
    peaks: list[int] = []
    for idx in order:
        if all(circ_dist(centers[idx], centers[p]) > 0.3 for p in peaks):
            peaks.append(int(idx))
        if len(peaks) == 2:
            break

    if len(peaks) < 2:
        return None, "두 봉우리 분리 안 됨 (8분 모드 1개만 — onset 분포 단일)"

    # 두 모드 위치 + 호 길이
    m_a, m_b = sorted(centers[p] for p in peaks)
    arc1 = m_b - m_a            # m_a 에서 m_b 까지 (직진)
    arc2 = 1.0 - arc1           # m_b 에서 m_a 까지 (wraparound)

    # 봉우리 신뢰도: 작은 봉우리가 큰 봉우리의 최소 25%
    h_a = hist[peaks[0]] if centers[peaks[0]] == m_a else hist[peaks[1]]
    h_b = hist[peaks[1]] if centers[peaks[1]] == m_b else hist[peaks[0]]
    if min(h_a, h_b) < 0.25 * max(h_a, h_b):
        return None, (f"두 봉우리 비대칭 너무 큼 "
                      f"(modes at {m_a:.2f}/{m_b:.2f}, 카운트 {int(h_a)}/{int(h_b)})")

    long, short = max(arc1, arc2), min(arc1, arc2)
    ratio = long / short
    return ratio, f"modes at {m_a:.2f} / {m_b:.2f}  (호 길이 {long:.3f} / {short:.3f})"
