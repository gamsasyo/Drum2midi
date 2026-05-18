"""
Ghost note 검출 + accent/ghost 분리 분석.

각 클래스(특히 snare)의 velocity 분포에서 두 봉우리(accent vs ghost)를
자동 분리해, microtiming 통계를 두 군집으로 나눠 본다.

덥/펑크 그루브에서 ghost note 는 백비트의 *fill-in* 요소로 작동:
- accent snare 의 push/pull 과 ghost 의 push/pull 이 다를 수 있음
- ghost 밀도가 곡 진행 중 변하면서 dynamics 만듦
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def find_velocity_threshold(velocities: List[int], min_count: int = 8) -> Tuple[int | None, str]:
    """
    velocity 분포에서 accent/ghost 경계를 자동 추정.

    방법: KDE 대신 단순 히스토그램에서 두 봉우리 사이의 valley(국소 최소) 검출.
    봉우리가 1개거나 분리가 불충분하면 None 반환 (= ghost 라벨링 안 함).
    """
    if len(velocities) < min_count:
        return None, "데이터 부족"

    v = np.asarray(velocities)
    # 히스토그램 (1-127 범위, bin=8)
    hist, edges = np.histogram(v, bins=16, range=(1, 128))
    centers = (edges[:-1] + edges[1:]) / 2

    # 봉우리 후보: 양옆보다 큰 점
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] >= 3:
            peaks.append(i)

    if len(peaks) < 2:
        # 봉우리 1개 → ghost 없음 (모두 비슷한 강세)
        return None, f"velocity 분포 단봉 (peaks={len(peaks)})"

    # 두 가장 큰 봉우리만 사용
    peaks.sort(key=lambda i: hist[i], reverse=True)
    p1, p2 = sorted(peaks[:2])

    # 두 봉우리 사이 valley
    valley_region = hist[p1:p2 + 1]
    if len(valley_region) < 3:
        return None, "봉우리 너무 가까움"
    valley_idx_local = int(np.argmin(valley_region))
    valley_idx = p1 + valley_idx_local
    threshold = int(centers[valley_idx])

    # 신뢰도: valley 가 양봉의 최대 60% 이하여야 진짜 분리
    valley_h = hist[valley_idx]
    max_peak_h = max(hist[p1], hist[p2])
    if valley_h > 0.6 * max_peak_h:
        return None, (f"두 봉우리 분리 약함 "
                      f"(peaks h={hist[p1]}/{hist[p2]}, valley h={valley_h})")

    return threshold, (f"accent peak @ v={int(centers[p2])}, ghost peak @ v={int(centers[p1])}, "
                       f"valley @ v={threshold}")


def annotate_ghosts(
    refined_onsets: Dict[str, List[Dict]],
) -> Dict[str, Dict]:
    """
    각 클래스마다 ghost/accent 라벨링.
    refined_onsets 의 각 event dict 에 'is_ghost' 키를 in-place 추가.

    반환: 클래스별 진단 정보
        {"snare": {"threshold": 47, "n_ghost": 38, "n_accent": 107, "msg": "..."}, ...}
    """
    diagnostics: Dict[str, Dict] = {}
    for cls, events in refined_onsets.items():
        velocities = [int(ev.get("velocity", 64)) for ev in events]
        threshold, msg = find_velocity_threshold(velocities)

        if threshold is None:
            for ev in events:
                ev["is_ghost"] = False
            diagnostics[cls] = {
                "threshold": None, "n_ghost": 0, "n_accent": len(events), "msg": msg,
            }
            continue

        n_ghost = 0
        for ev in events:
            is_g = int(ev.get("velocity", 64)) < threshold
            ev["is_ghost"] = bool(is_g)
            if is_g:
                n_ghost += 1
        diagnostics[cls] = {
            "threshold": threshold,
            "n_ghost": n_ghost,
            "n_accent": len(events) - n_ghost,
            "msg": msg,
        }
    return diagnostics
