"""
원본 절대 타이밍을 그대로 보존한 MIDI 파일 작성. 스냅 금지.

전략:
- PPQ=1920 (충분히 높은 해상도)
- 단일 tempo (전체 곡의 median BPM 사용)
- 각 onset의 절대 시각(초)을 tick으로 정확히 변환 → delta-time 누적
- 결과: 일반 DAW에서 열어도 그리드와 어긋난 노트들이 그대로 보임

만약 BPM이 곡 내내 크게 변동한다면 (madmom이 변동 BPM도 추적함) 정확한
표현을 위해 tempo map을 추가해야 하지만, 일단 곡 내 BPM 변화가 거의
없다는 가정으로 단일 tempo 사용.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import mido
import numpy as np

# General MIDI drum mapping (channel 9, 0-indexed → MIDI 채널 10)
GM_DRUM_NOTE = {
    "kick":   36,  # Bass Drum 1
    "snare":  38,  # Acoustic Snare
    "hihat":  42,  # Closed Hi-Hat (ADTOF는 open/closed 통합 → 42 로 표현)
    "tom":    47,  # Low-Mid Tom (ADTOF TT 클래스의 대표값)
    "cymbal": 49,  # Crash Cymbal 1 (ADTOF CY 클래스의 대표값)
}

PPQ = 1920


def export_midi_tempo_map(
    refined_onsets: Dict[str, List[Dict[str, float]]],
    beats: np.ndarray,
    output_path: Path,
    note_duration_sec: float = 0.05,
    audio_duration_sec: float | None = None,
) -> None:
    """
    매 비트마다 set_tempo event 가 있는 multi-tempo MIDI 출력.

    가변 BPM 곡에서 각 비트의 실제 IBI 를 tempo 로 인코딩 → MIDI
    musical-time 과 audio 실시각이 정확히 일치.

    DAW 호환성:
      ✓ Reaper, Logic Pro     : 완벽 작동 (tempo map 인식)
      △ Ableton Live          : 첫 tempo 만 인식, audio warp 필요
      ✓ Studio One, Cubase    : 작동
    """
    if len(beats) < 2:
        raise ValueError("Tempo map 에는 최소 2개 비트 필요")

    def time_to_musical_tick(t: float) -> float:
        """audio 절대시각 t (초) → MIDI musical tick.
        가변 IBI 를 따라 segment-wise 선형 보간."""
        if t <= beats[0]:
            ibi = beats[1] - beats[0]
            return (t - beats[0]) / ibi * PPQ
        if t >= beats[-1]:
            ibi = beats[-1] - beats[-2]
            return (len(beats) - 1 + (t - beats[-1]) / ibi) * PPQ
        idx = int(np.searchsorted(beats, t, side="right") - 1)
        ibi = beats[idx + 1] - beats[idx]
        return (idx + (t - beats[idx]) / ibi) * PPQ

    mid = mido.MidiFile(ticks_per_beat=PPQ)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    initial_ibi = beats[1] - beats[0]
    initial_tempo = mido.bpm2tempo(60 / initial_ibi)
    track.append(mido.MetaMessage("set_tempo", tempo=initial_tempo, time=0))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    track.append(mido.MetaMessage("track_name", name="drums_tempo_map", time=0))

    # (tick, kind, args) — 모든 이벤트 평탄화
    events: list = []

    # 매 비트 boundary 에 set_tempo (첫 번째는 위에서 이미 깐 거)
    for i in range(1, len(beats) - 1):
        ibi = beats[i + 1] - beats[i]
        tempo_us = mido.bpm2tempo(60 / ibi)
        events.append((i * PPQ, "tempo", tempo_us))

    # 노트 이벤트 (musical tick 으로 변환)
    for cls, evs in refined_onsets.items():
        note = GM_DRUM_NOTE.get(cls)
        if note is None:
            continue
        for ev in evs:
            t = ev["time"]
            vel = int(ev.get("velocity", 100))
            tick_on = int(round(time_to_musical_tick(t)))
            tick_off = int(round(time_to_musical_tick(t + note_duration_sec)))
            events.append((tick_on, "on", note, vel))
            events.append((max(tick_on + 1, tick_off), "off", note))

    # 정렬: 같은 tick 에서 off → tempo → on 순
    order = {"off": 0, "tempo": 1, "on": 2}
    events.sort(key=lambda e: (e[0], order[e[1]]))

    last_tick = 0
    for evt in events:
        tick = evt[0]
        delta = max(0, tick - last_tick)
        kind = evt[1]
        if kind == "tempo":
            track.append(mido.MetaMessage("set_tempo", tempo=evt[2], time=delta))
        elif kind == "on":
            track.append(mido.Message("note_on", channel=9, note=evt[2], velocity=evt[3], time=delta))
        elif kind == "off":
            track.append(mido.Message("note_off", channel=9, note=evt[2], velocity=0, time=delta))
        last_tick = tick

    # Audio duration 까지 패딩
    if audio_duration_sec is not None:
        end_tick = int(round(time_to_musical_tick(audio_duration_sec)))
        pad = max(0, end_tick - last_tick)
        if pad > 0:
            track.append(mido.MetaMessage("end_of_track", time=pad))

    mid.save(str(output_path))
    n_notes = sum(1 for e in events if e[1] == "on")
    n_tempos = sum(1 for e in events if e[1] == "tempo") + 1
    print(f"[midi/tempo-map] saved: {output_path} ({n_notes} notes, {n_tempos} tempo events)")


def estimate_precise_bpm(beats: np.ndarray) -> tuple[float, float]:
    """
    선형 회귀로 BPM 정밀 추정. 모든 비트 시각을 활용해 median 보다 정확.

    fit: t_i = intercept + slope * i  (i = beat index)
    → IBI = slope, BPM = 60/slope

    반환: (bpm, max_drift_ms)
        max_drift_ms = 회귀 적합값과 실제 beat 시각의 최대 편차 — 가변 템포 진단용
    """
    if len(beats) < 4:
        return 120.0, 0.0
    idx = np.arange(len(beats))
    slope, intercept = np.polyfit(idx, beats, 1)
    bpm = 60.0 / slope if slope > 0 else 120.0
    fitted = intercept + slope * idx
    max_drift_ms = float(np.max(np.abs(beats - fitted))) * 1000
    return bpm, max_drift_ms


def export_midi(
    refined_onsets: Dict[str, List[Dict[str, float]]],
    beats: np.ndarray,
    output_path: Path,
    note_duration_sec: float = 0.05,
    audio_duration_sec: float | None = None,
) -> None:
    """원본 타이밍을 그대로 보존한 GM 드럼 MIDI 생성."""
    bpm, max_drift_ms = estimate_precise_bpm(beats)
    print(f"[midi] 정밀 BPM (선형 회귀): {bpm:.4f}  "
          f"(곡 내 fit 최대 편차 {max_drift_ms:.1f}ms — "
          f"{'거의 constant tempo' if max_drift_ms < 30 else '가변 템포 가능성, 별도 tempo map 필요'})")
    tempo_us = mido.bpm2tempo(bpm)

    # 모든 이벤트 평탄화: (time_sec, type, note, velocity)
    events = []
    for cls, evs in refined_onsets.items():
        note = GM_DRUM_NOTE.get(cls)
        if note is None:
            print(f"[midi] WARN: GM_DRUM_NOTE에 없는 클래스 '{cls}' 스킵")
            continue
        for ev in evs:
            t = ev["time"]
            vel = int(ev.get("velocity", 100))
            events.append((t, "on", note, vel))
            events.append((t + note_duration_sec, "off", note, 0))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))  # off가 같은 시각이면 먼저

    mid = mido.MidiFile(ticks_per_beat=PPQ)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    track.append(mido.MetaMessage("track_name", name="drums", time=0))

    last_tick = 0
    for t_sec, kind, note, vel in events:
        abs_tick = int(round(mido.second2tick(t_sec, PPQ, tempo_us)))
        delta = max(0, abs_tick - last_tick)
        if kind == "on":
            track.append(mido.Message("note_on", channel=9, note=note, velocity=vel, time=delta))
        else:
            track.append(mido.Message("note_off", channel=9, note=note, velocity=0, time=delta))
        last_tick = abs_tick

    # 오디오 길이만큼 트랙 길이 패딩 (Ableton에서 clip 길이가 audio와 일치하도록)
    if audio_duration_sec is not None:
        target_tick = int(round(mido.second2tick(audio_duration_sec, PPQ, tempo_us)))
        pad = max(0, target_tick - last_tick)
        if pad > 0:
            # mido는 트랙 끝에 자동으로 end_of_track 추가하지만, delta=pad로 명시
            track.append(mido.MetaMessage("end_of_track", time=pad))

    mid.save(str(output_path))
    print(f"[midi] saved: {output_path} (BPM={bpm:.2f}, {len(events)//2} notes"
          f"{', padded to ' + format(audio_duration_sec, '.2f') + 's' if audio_duration_sec else ''})")
