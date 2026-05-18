"""
Stage 3: 정밀 onset 재측정.

Stage 2의 onset은 hop_length=512에 묶여 ~10ms 양자화되어 있다.
각 onset 주변 ±50ms 윈도우에서 클래스별 밴드 필터 적용 후
에너지 envelope의 미분 피크를 잡아 샘플 단위 정밀도 확보.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np

from .transcription import _bandpass, DRUM_CLASSES

# 클래스별 정밀 측정용 밴드 (transcription의 BAND_CONFIG와 동일 의도)
REFINE_BAND = {
    "kick":   (30, 200),
    "snare":  (150, 800),
    "hihat":  (5000, None),
    "tom":    (80, 300),
    "cymbal": (3000, None),
}


def _refine_one(y: np.ndarray, sr: int, coarse_t: float, cls: str, window_ms: float = 50.0) -> float:
    """
    coarse_t 주변에서 정밀 onset 시각 (attack START) 검출.

    2단계:
      1) 클래스 밴드 필터 → |signal|² 의 짧은 smoothing → 미분 최대점 (attack peak)
      2) 그 peak 직전의 envelope 최저점으로 backtrack → attack 시작점

    이전 버전은 1단계(attack peak)만 썼는데, drum hit 의 attack 이 ~10-30ms
    길어서 transient START 와 PEAK 사이 일관된 +15ms 시스템 바이어스 발생.
    Backtrack 으로 attack 시작점 정렬 (librosa.onset.onset_backtrack 동일 알고리즘).
    """
    low, high = REFINE_BAND[cls]
    y_f = _bandpass(y, sr, low, high)

    center = int(coarse_t * sr)
    half = int(window_ms / 1000 * sr)
    start = max(0, center - half)
    end = min(len(y_f), center + half)
    seg = y_f[start:end]
    if len(seg) < 4:
        return coarse_t

    env = seg.astype(np.float64) ** 2
    # 1ms 짧은 박스카 smoothing — 트랜지언트 모양 안 망가뜨림
    win = max(1, sr // 1000)
    if win > 1:
        env = np.convolve(env, np.ones(win) / win, mode="same")

    diff = np.diff(env)
    if len(diff) == 0 or diff.max() <= 0:
        return coarse_t

    # 1) attack peak
    peak_idx = int(np.argmax(diff))

    # 2) Backtrack: peak 직전의 envelope 최저점 (attack start) 찾기.
    #    peak_idx 에서 좌측으로 이동하면서 env 값이 더 이상 감소 안 할 때까지.
    #    상한 ~30ms (typical drum attack 길이).
    max_backtrack = min(peak_idx, int(0.030 * sr))
    start_idx = peak_idx
    for k in range(peak_idx, peak_idx - max_backtrack, -1):
        if k <= 0:
            break
        if env[k] < env[start_idx]:
            start_idx = k
    # peak 와 start_idx 사이 envelope 최저값 자리 = attack start
    if start_idx < peak_idx:
        attack_start = start_idx + int(np.argmin(env[start_idx:peak_idx + 1]))
    else:
        attack_start = peak_idx

    refined_sample = start + attack_start
    return refined_sample / sr


def _estimate_velocity(y: np.ndarray, sr: int, t: float, cls: str, window_ms: float = 30.0) -> int:
    """대략적인 velocity (1-127). 클래스 밴드의 윈도우 RMS를 동적 정규화."""
    low, high = REFINE_BAND[cls]
    y_f = _bandpass(y, sr, low, high)
    center = int(t * sr)
    half = int(window_ms / 1000 * sr)
    seg = y_f[max(0, center - half): center + half]
    if len(seg) == 0:
        return 64
    rms = float(np.sqrt(np.mean(seg ** 2)))
    return rms  # 정규화는 호출자가 일괄 처리


def refine_onsets(
    drums_wav: Path, raw_onsets: Dict[str, List[float]], output_dir: Path
) -> Dict[str, List[Dict[str, float]]]:
    """
    각 coarse onset을 정밀 재측정 + velocity 추정.
    반환: {"kick": [{"time": float, "velocity": int}, ...], ...}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "refined_onsets.json"
    if cache.exists():
        print(f"[refinement] cached refined_onsets.json 사용")
        return json.loads(cache.read_text())

    y, sr = librosa.load(str(drums_wav), sr=None, mono=True)

    refined: Dict[str, List[Dict[str, float]]] = {}
    raw_vels: Dict[str, List[float]] = {}

    # raw_onsets 에 실제 들어있는 클래스만 처리 (method마다 다름)
    classes_in_data = [c for c in DRUM_CLASSES if c in raw_onsets and raw_onsets[c]]
    for cls in classes_in_data:
        coarse_list = raw_onsets.get(cls, [])
        refined[cls] = []
        raw_vels[cls] = []
        for ct in coarse_list:
            rt = _refine_one(y, sr, ct, cls)
            rv = _estimate_velocity(y, sr, rt, cls)
            refined[cls].append({"time": rt})
            raw_vels[cls].append(rv)

    # velocity 정규화: 전체 클래스 통합 분포의 95-percentile을 127에 매핑
    all_vels = np.concatenate([np.array(v) for v in raw_vels.values() if len(v) > 0])
    if len(all_vels) == 0:
        vel_max = 1.0
    else:
        vel_max = float(np.percentile(all_vels, 95)) or 1.0

    for cls in classes_in_data:
        for i, rv in enumerate(raw_vels[cls]):
            v = int(np.clip(round((rv / vel_max) * 110 + 17), 1, 127))
            refined[cls][i]["velocity"] = v

        # 시각 기준 정렬
        refined[cls].sort(key=lambda d: d["time"])
        # 너무 가까운 중복 (5ms 미만) 제거
        cleaned = []
        last_t = -1.0
        for ev in refined[cls]:
            if ev["time"] - last_t >= 0.005:
                cleaned.append(ev)
                last_t = ev["time"]
        refined[cls] = cleaned
        print(f"[refinement] {cls}: {len(refined[cls])} refined onsets")

    cache.write_text(json.dumps(refined, indent=2))
    return refined
