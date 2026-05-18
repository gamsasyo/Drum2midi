"""
Stage 4: 비트 그리드 추출 + deviation 계산.

madmom DBNBeatTracker → 박(beat) 시각 리스트 → 8분/16분 그리드로 보간.
각 onset의 가장 가까운 그리드점을 찾고 deviation을 ms로 기록.

★ 스냅 없음. onset의 원본 시각을 그대로 보존.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np


def correct_bpm_octave(beats: np.ndarray, bpm_hint: float, tol: float = 0.05) -> np.ndarray:
    """
    검출된 BPM 이 bpm_hint 의 2배/0.5배 (또는 3배/1/3배) 인 경우 자동 보정.

    librosa beat_track 은 laid-back / half-time / double-time 곡에서
    octave error 가 자주 발생한다. 진짜 BPM 한 번이라도 알면 보정 가능.

    예: hint=88, detected=178 → ratio=2.02 → 매 2번째 beat 만 사용
        hint=88, detected=44  → ratio=0.50 → beat 사이에 보간

    tol: 비율이 정수배(2,3,1/2,1/3)에서 ±5% 이내면 보정 트리거.
    """
    if len(beats) < 4:
        return beats
    median_ibi = float(np.median(np.diff(beats)))
    detected_bpm = 60.0 / median_ibi if median_ibi > 0 else 0
    if detected_bpm <= 0:
        return beats

    ratio = detected_bpm / bpm_hint
    # 가까운 정수배 찾기 (2, 3, 1/2, 1/3)
    candidates = [(2.0, "halve"), (3.0, "third"),
                  (0.5, "double"), (1.0 / 3.0, "triple")]
    for r, action in candidates:
        if abs(ratio - r) / r < tol:
            print(f"[grid] BPM hint 보정: detected={detected_bpm:.2f}, "
                  f"hint={bpm_hint:.2f}, ratio={ratio:.3f} → {action}")
            if action == "halve":
                return beats[::2]   # 매 2번째 beat
            elif action == "third":
                return beats[::3]
            elif action == "double":
                # beat 사이에 중점 삽입
                new_beats = []
                for i in range(len(beats) - 1):
                    new_beats.append(beats[i])
                    new_beats.append((beats[i] + beats[i + 1]) / 2)
                new_beats.append(beats[-1])
                return np.asarray(new_beats)
            elif action == "triple":
                new_beats = []
                for i in range(len(beats) - 1):
                    new_beats.append(beats[i])
                    step = (beats[i + 1] - beats[i]) / 3
                    new_beats.append(beats[i] + step)
                    new_beats.append(beats[i] + 2 * step)
                new_beats.append(beats[-1])
                return np.asarray(new_beats)
    print(f"[grid] BPM hint 보정 안 함: detected={detected_bpm:.2f} 이 "
          f"hint={bpm_hint} 의 정수배가 아님 (ratio={ratio:.3f})")
    return beats


def detect_phase_shift(
    beats: np.ndarray, raw_onsets: Dict[str, List[float]]
) -> float:
    """
    librosa beat_track 이 박을 ½ IBI 어긋난 자리(=실제 offbeat)에 잡았는지 판정.

    원리: kick 은 보통 진짜 downbeat에 떨어진다. kick fractional position 의
    circular mean 이 0/1 근처면 beats 가 맞음. 0.5 근처면 ½ shift 필요.

    반환: 보정용 shift 값(초). 0.0 = no shift, IBI/2 = beats 를 그만큼 *앞으로*
    당겨야 한다는 의미 (사실상 beats[i] - IBI/2).
    """
    if len(beats) < 4:
        return 0.0
    kicks = raw_onsets.get("kick", [])
    if len(kicks) < 8:
        return 0.0

    ibi = float(np.median(np.diff(beats)))
    fracs = []
    for t in kicks:
        idx = int(np.searchsorted(beats, t, side="right") - 1)
        if 0 <= idx < len(beats) - 1:
            local_ibi = beats[idx + 1] - beats[idx]
            if local_ibi > 0:
                fracs.append((t - beats[idx]) / local_ibi)
    if len(fracs) < 8:
        return 0.0

    # Circular mean on [0,1): theta = 2π·frac
    theta = 2 * np.pi * np.asarray(fracs)
    mean_theta = np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta)))
    if mean_theta < 0:
        mean_theta += 2 * np.pi
    dominant_kick_frac = mean_theta / (2 * np.pi)

    # 0/1 근처면 beats OK, 0.5 근처면 ½ shift 필요
    # 임계: 0.3 ~ 0.7 사이면 shift
    if 0.3 < dominant_kick_frac < 0.7:
        print(f"[grid] ★ phase shift 감지: kick circular mean = {dominant_kick_frac:.3f} "
              f"(0.5 근처) → beats 를 IBI/2={ibi/2*1000:.1f}ms 만큼 당김")
        return ibi / 2
    print(f"[grid] phase OK: kick circular mean = {dominant_kick_frac:.3f}")
    return 0.0


def extract_beat_grid(
    drums_wav: Path, output_dir: Path, tracker: str = "auto",
    bpm_hint: Optional[float] = None,
) -> Tuple[np.ndarray, str]:
    """
    비트(quarter-note) 시각 배열 반환. 단위: 초.
    """
    cache = output_dir / "beat_grid.json"
    if cache.exists():
        data = json.loads(cache.read_text())
        print(f"[grid] cached beat_grid.json 사용 (tracker={data.get('tracker')})")
        return np.array(data["beats"]), data.get("tracker", "unknown")

    methods = ["beat_this", "madmom", "librosa"] if tracker == "auto" else [tracker]
    beats = None
    used = None
    for m in methods:
        if m == "beat_this":
            beats = _beat_this_beats(drums_wav)
        elif m == "madmom":
            beats = _madmom_beats(drums_wav)
        elif m == "librosa":
            beats = _librosa_beats(drums_wav, start_bpm=bpm_hint)
        else:
            raise ValueError(f"Unknown tracker: {m}")
        if beats is not None and len(beats) >= 4:
            used = m
            break

    if beats is None or len(beats) < 4:
        raise RuntimeError("비트 트래킹 실패 (검출 박 수 < 4)")

    print(f"\n[grid] ★ USED BEAT TRACKER: {used} ({len(beats)} beats)")
    median_ibi = float(np.median(np.diff(beats)))
    bpm = 60.0 / median_ibi if median_ibi > 0 else 0
    print(f"[grid] median BPM: {bpm:.2f}")

    cache.write_text(json.dumps({"tracker": used, "beats": beats.tolist(), "bpm": bpm}, indent=2))
    return beats, used


def _beat_this_beats(drums_wav: Path) -> Optional[np.ndarray]:
    """
    Beat This! (Foscarin·Schlüter·Widmer, ISMIR 2024) — SOTA 비트 트래커.
    Transformer 기반, 가변 BPM 자동 추적, octave error 거의 없음.

    주의: Beat-this! 는 confidence 기반이라 atmospheric intro / 드럼 sparse
    구간에서 비트를 안 잡음. 후처리로 큰 gap 을 median IBI 간격으로 보간해
    contiguous 그리드 만든다.
    """
    try:
        import importlib.util
        if importlib.util.find_spec("beat_this") is None:
            return None
        from beat_this.inference import File2Beats
        # final0 = 논문 default checkpoint, dbn=False (post-processing 없이도 충분 정확)
        f2b = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
        beats, _downbeats = f2b(str(drums_wav))
        beats = np.asarray(beats)
        if len(beats) < 4:
            return beats

        median_ibi = float(np.median(np.diff(beats)))

        # ★ Dense-cull: 0.5× median IBI 보다 가까운 인접 비트 제거 (false dense)
        keep = [beats[0]]
        n_culled = 0
        for i in range(1, len(beats)):
            if beats[i] - keep[-1] >= 0.5 * median_ibi:
                keep.append(beats[i])
            else:
                n_culled += 1
        beats = np.asarray(keep)
        if n_culled > 0:
            print(f"[grid/beat_this] culled {n_culled} too-dense beats "
                  f"(< 0.5× median IBI {median_ibi*1000:.0f}ms)")

        # median 재계산 (cull 후)
        median_ibi = float(np.median(np.diff(beats)))
        filled = [beats[0]]
        n_filled = 0
        for i in range(1, len(beats)):
            gap = beats[i] - filled[-1]
            if gap > 1.5 * median_ibi:
                # 보간: 사이에 몇 개 넣을지 (round to nearest int)
                n_insert = int(round(gap / median_ibi)) - 1
                if n_insert > 0:
                    step = gap / (n_insert + 1)
                    for k in range(1, n_insert + 1):
                        filled.append(filled[-1] + step)
                    n_filled += n_insert
            filled.append(beats[i])

        # ★ Backward extrapolate to t=0 (intro 같이 시작 부분 비트 없는 경우)
        result = np.asarray(filled)
        front = []
        t = result[0] - median_ibi
        n_front = 0
        while t > 0:
            front.insert(0, t)
            t -= median_ibi
            n_front += 1
        if front:
            result = np.concatenate([np.asarray(front), result])

        if n_filled > 0 or n_front > 0:
            print(f"[grid/beat_this] gap-filled: +{n_filled} interior + "
                  f"{n_front} front, median IBI {median_ibi*1000:.1f}ms "
                  f"({60/median_ibi:.2f} BPM)")
        return result
    except Exception as e:
        print(f"[grid/beat_this] failed: {e}")
        return None


def _madmom_beats(drums_wav: Path) -> Optional[np.ndarray]:
    try:
        from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
        rnn = RNNBeatProcessor()(str(drums_wav))
        dbn = DBNBeatTrackingProcessor(min_bpm=60, max_bpm=200, fps=100)
        return np.asarray(dbn(rnn))
    except Exception as e:
        print(f"[grid/madmom] failed: {e}")
        return None


def _librosa_beats(drums_wav: Path, start_bpm: Optional[float] = None) -> Optional[np.ndarray]:
    try:
        y, sr = librosa.load(str(drums_wav), sr=None, mono=True)
        kwargs = {"y": y, "sr": sr, "units": "frames"}
        if start_bpm is not None and start_bpm > 0:
            # librosa 가 start_bpm 근처에서 검색하도록 강력히 bias.
            # 기본 120, jungle/dnb 면 170 같이 명시해야 octave error 안 발생.
            kwargs["start_bpm"] = float(start_bpm)
            kwargs["tightness"] = 400  # 기본 100 → 더 strict 하게 BPM 유지
            print(f"[grid/librosa] start_bpm hint = {start_bpm} (tightness 400)")
        tempo, beat_frames = librosa.beat.beat_track(**kwargs)
        beats = librosa.frames_to_time(beat_frames, sr=sr)
        return np.asarray(beats)
    except Exception as e:
        print(f"[grid/librosa] failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────────
# Subdivision grid
# ──────────────────────────────────────────────────────────────────

def build_subdivision_grid(
    beats: np.ndarray,
    subdivisions_per_beat: int,
    min_time: float = 0.0,
    max_time: float | None = None,
) -> np.ndarray:
    """
    quarter-note beats 사이를 subdivisions_per_beat 만큼 등분해서 보간.

    핵심: 비트 트래커가 곡 시작/끝을 못 잡은 경우 (librosa가 흔히 그럼)
    grid를 [min_time, max_time] 범위로 양쪽으로 extrapolate 한다.
    그렇지 않으면 grid 밖 onset들이 grid 끝점에 매칭돼 deviation이 폭주.

    예: beats=[0.5, 1.0, 1.5], subdivisions=2, min=0, max=2 →
        [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    """
    if len(beats) < 2:
        return beats.copy()

    # 1) 비트 사이 보간
    grid = []
    for i in range(len(beats) - 1):
        b0, b1 = beats[i], beats[i + 1]
        step = (b1 - b0) / subdivisions_per_beat
        for k in range(subdivisions_per_beat):
            grid.append(b0 + k * step)

    # 2) 뒤로 extrapolate: 마지막 IBI를 유지하며 max_time 까지
    last_step = (beats[-1] - beats[-2]) / subdivisions_per_beat
    if max_time is None:
        # 한 박만 (기존 동작)
        max_time = beats[-1] + (beats[-1] - beats[-2])
    t = beats[-1]
    while t <= max_time + last_step:
        grid.append(t)
        t += last_step

    # 3) 앞으로 extrapolate: 첫 IBI를 유지하며 min_time 까지
    first_step = (beats[1] - beats[0]) / subdivisions_per_beat
    t = beats[0] - first_step
    front = []
    while t >= min_time - first_step:
        front.append(t)
        t -= first_step

    grid = sorted(front) + grid
    return np.asarray(grid)


def _nearest_grid_deviation(t: float, grid: np.ndarray) -> Tuple[float, float]:
    """
    t에 가장 가까운 grid 점과 deviation(초) 반환. deviation = t - grid_point.
    양수면 onset이 그리드보다 늦음 (laid-back).
    """
    idx = int(np.argmin(np.abs(grid - t)))
    return float(grid[idx]), float(t - grid[idx])


# ──────────────────────────────────────────────────────────────────
# Main deviation table
# ──────────────────────────────────────────────────────────────────

def compute_deviations(
    refined_onsets: Dict[str, List[Dict[str, float]]],
    beats: np.ndarray,
) -> List[Dict[str, float]]:
    """
    각 onset에 대해 8분/16분 그리드 deviation 계산.
    반환: row dict 리스트 — analyze.py에서 CSV로 저장.
    """
    # onset 범위를 알아내서 grid를 양쪽으로 충분히 extrapolate
    all_times = [ev["time"] for evs in refined_onsets.values() for ev in evs]
    if all_times:
        t_min = min(min(all_times), float(beats[0])) - 0.5
        t_max = max(max(all_times), float(beats[-1])) + 0.5
    else:
        t_min, t_max = float(beats[0]), float(beats[-1])

    grid_8 = build_subdivision_grid(beats, 2, min_time=t_min, max_time=t_max)
    grid_16 = build_subdivision_grid(beats, 4, min_time=t_min, max_time=t_max)

    rows: List[Dict[str, float]] = []
    for cls, events in refined_onsets.items():
        for ev in events:
            t = ev["time"]
            g8, d8 = _nearest_grid_deviation(t, grid_8)
            g16, d16 = _nearest_grid_deviation(t, grid_16)
            # beat position within bar (0~3.99): t를 직전 beat로부터의 비율로 표현
            beat_idx = int(np.searchsorted(beats, t, side="right") - 1)
            if 0 <= beat_idx < len(beats) - 1:
                ibi = beats[beat_idx + 1] - beats[beat_idx]
                pos_in_beat = (t - beats[beat_idx]) / ibi if ibi > 0 else 0
                # 4박자 가정 — 1박부터 4박까지 순환
                beat_in_bar = beat_idx % 4
                beat_position = beat_in_bar + pos_in_beat
            else:
                beat_position = float("nan")

            rows.append({
                "onset_time_sec":         round(t, 6),
                "drum_class":             cls,
                "nearest_grid_8th_sec":   round(g8, 6),
                "deviation_8th_ms":       round(d8 * 1000, 3),
                "nearest_grid_16th_sec":  round(g16, 6),
                "deviation_16th_ms":      round(d16 * 1000, 3),
                "beat_position":          round(beat_position, 4) if not np.isnan(beat_position) else "",
                "velocity":               int(ev.get("velocity", 64)),
                "is_ghost":               int(bool(ev.get("is_ghost", False))),
            })

    rows.sort(key=lambda r: r["onset_time_sec"])
    return rows
