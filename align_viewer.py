"""
align_viewer.py — Interactive audio + MIDI alignment viewer.

오디오의 mel-spectrogram 위에 MIDI 노트를 컬러 vertical line 으로
오버레이해서 alignment 를 시각·청각으로 검증.

USAGE:
    python align_viewer.py <audio.wav> <midi.mid> [--click] [--sr 22050]

    # run 디렉토리 path 만 주면 자동으로 drums.wav (캐시) + drums.mid 찾음
    python align_viewer.py "outputs/<song>/runs/<timestamp>"

CONTROLS:
    space        play / pause
    click        해당 시각으로 seek
    home         처음으로 rewind
    esc          stop

OPTIONS:
    --click      각 MIDI 노트 자리에 짧은 클릭 소리 믹스 → 귀로도 alignment 확인
    --tempo-map  drums_tempo_map.mid 우선 사용 (variable BPM 곡용)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import librosa
import librosa.display
import matplotlib

# macOS 에선 native MacOSX backend, 그 외 시스템은 TkAgg 폴백
import platform
if platform.system() == "Darwin":
    matplotlib.use("MacOSX")
else:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pretty_midi
import sounddevice as sd


# GM drum pitch → (class, color)
PITCH_TO_CLASS = {
    35: ("kick",   "#d62728"),
    36: ("kick",   "#d62728"),
    38: ("snare",  "#1f77b4"),
    40: ("snare",  "#1f77b4"),
    42: ("hihat",  "#2ca02c"),
    44: ("hihat",  "#2ca02c"),
    46: ("hihat",  "#2ca02c"),
    47: ("tom",    "#ff7f0e"),
    49: ("cymbal", "#9467bd"),
    51: ("cymbal", "#9467bd"),
}
CLASS_Y = {"kick": 0, "snare": 1, "hihat": 2, "tom": 3, "cymbal": 4}


def resolve_inputs(audio_arg: str, midi_arg: str | None,
                   prefer_tempo_map: bool) -> tuple[Path, Path]:
    """audio_arg 가 디렉토리면 그 run 의 drums.{wav,mid} 자동 탐색."""
    ap = Path(audio_arg)
    if ap.is_dir():
        run_dir = ap
        # drums.wav 는 캐시 (상위 폴더에)
        cache_dir = run_dir.parent.parent
        drums_wav = cache_dir / "drums.wav"
        if not drums_wav.exists():
            raise FileNotFoundError(f"drums.wav not found in {cache_dir}")
        # MIDI 선택
        mid_name = "drums_tempo_map.mid" if prefer_tempo_map else "drums.mid"
        midi_path = run_dir / mid_name
        if not midi_path.exists():
            # fallback
            alt = run_dir / ("drums.mid" if prefer_tempo_map else "drums_tempo_map.mid")
            if alt.exists():
                midi_path = alt
            else:
                cand = list(run_dir.glob("*.mid"))
                if not cand:
                    raise FileNotFoundError(f"no .mid in {run_dir}")
                midi_path = cand[0]
        return drums_wav, midi_path

    if midi_arg is None:
        raise ValueError("audio 가 파일이면 midi 인자 필수")
    return ap, Path(midi_arg)


def load_midi_notes(midi_path: Path) -> list[tuple[float, int, int]]:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [(n.start, n.pitch, n.velocity)
             for inst in pm.instruments for n in inst.notes]
    notes.sort()
    return notes


def make_click_track(notes, audio_dur: float, sr: int,
                      click_duration: float = 0.025) -> np.ndarray:
    """각 MIDI 노트 자리에 짧은 noise burst click."""
    track = np.zeros(int(audio_dur * sr) + sr)
    n_click = int(sr * click_duration)
    envelope = np.exp(-np.linspace(0, 7, n_click))
    np.random.seed(0)
    click_template = envelope * np.random.randn(n_click)
    for t, _pitch, vel in notes:
        i = int(t * sr)
        if 0 <= i < len(track) - n_click:
            track[i:i + n_click] += click_template * (vel / 127) * 0.5
    return track


def main():
    p = argparse.ArgumentParser(
        description="Interactive audio + MIDI alignment viewer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("audio", help="audio wav OR run 디렉토리")
    p.add_argument("midi", nargs="?", help="MIDI 파일 (audio 가 디렉토리면 생략)")
    p.add_argument("--click", action="store_true",
                   help="MIDI 노트 자리에 클릭 소리 믹스 (alignment 청각 확인)")
    p.add_argument("--tempo-map", action="store_true",
                   help="drums_tempo_map.mid 우선 (variable BPM 곡)")
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--n-mels", type=int, default=96)
    args = p.parse_args()

    audio_path, midi_path = resolve_inputs(args.audio, args.midi, args.tempo_map)
    print(f"audio: {audio_path}", flush=True)
    print(f"midi:  {midi_path}", flush=True)

    print("[1/4] loading audio…", flush=True)
    y, sr = librosa.load(str(audio_path), sr=args.sr, mono=True)
    audio_dur = len(y) / sr
    print(f"[2/4] computing mel-spectrogram ({audio_dur:.1f}s @ {sr}Hz)…", flush=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=args.n_mels, hop_length=512
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 매트릭스가 너무 크면 매우 느림 (190s = 8000+ time bin × 96 mel = 768K cells).
    # 시간축으로 max-pooling 하여 2000 time bin 으로 축소. 렌더 30s → 1s.
    MAX_TIME_BINS = 2000
    if mel_db.shape[1] > MAX_TIME_BINS:
        pool = mel_db.shape[1] // MAX_TIME_BINS
        n = (mel_db.shape[1] // pool) * pool
        mel_db = mel_db[:, :n].reshape(mel_db.shape[0], n // pool, pool).max(axis=2)
        print(f"      downsampled mel-spec time-axis ÷{pool} → shape {mel_db.shape}", flush=True)
    print(f"[3/4] loading MIDI…", flush=True)
    notes = load_midi_notes(midi_path)
    print(f"      → {audio_dur:.1f}s audio, {len(notes)} MIDI notes", flush=True)

    # 재생용 mix
    if args.click:
        clicks = make_click_track(notes, audio_dur, sr)
        play_audio = (y * 0.6 + clicks[:len(y)] * 1.4).astype(np.float32)
        m = float(np.max(np.abs(play_audio)))
        if m > 0.99:
            play_audio = (play_audio / m * 0.95).astype(np.float32)
    else:
        play_audio = y.astype(np.float32)

    # ── 플롯 ──
    print("[4a] creating figure…", flush=True)
    fig, (ax_mel, ax_pr) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(
        f"{audio_path.name}   |   {len(notes)} MIDI notes"
        f"   {'[click ON]' if args.click else ''}",
        fontsize=11,
    )

    # specshow 가 pcolormesh 라 큰 매트릭스에 매우 느림 (190s 곡 = 8000+ time bin).
    # imshow 로 직접 그리는 게 10~50x 빠름.
    print(f"[4b] imshow mel-spec shape {mel_db.shape}…", flush=True)
    ax_mel.imshow(
        mel_db,
        origin="lower", aspect="auto",
        extent=(0, audio_dur, 0, args.n_mels),
        cmap="magma", interpolation="nearest",
    )
    ax_mel.set_ylabel("mel bin")
    ax_mel.set_yticks([])  # mel 축 라벨 단순화

    print(f"[4c] overlaying {len(notes)} MIDI notes (vectorized)…", flush=True)
    # MIDI vertical line — 클래스별로 묶어서 vectorized 한 번에 그림 (훨씬 빠름)
    by_class: dict[str, list[float]] = {c: [] for c in CLASS_Y}
    by_class_vels: dict[str, list[int]] = {c: [] for c in CLASS_Y}
    for t, pitch, vel in notes:
        cls, _ = PITCH_TO_CLASS.get(pitch, ("?", "gray"))
        if cls in by_class:
            by_class[cls].append(t)
            by_class_vels[cls].append(vel)

    # imshow 후 ylim 이 (0, n_mels) 로 설정됨
    y_lo, y_hi = 0, args.n_mels
    for cls, times in by_class.items():
        if not times:
            continue
        color = {"kick": "#d62728", "snare": "#1f77b4", "hihat": "#2ca02c",
                 "tom": "#ff7f0e", "cymbal": "#9467bd"}[cls]
        ax_mel.vlines(times, y_lo, y_hi, colors=color, alpha=0.4, linewidth=0.7)
        # piano roll: scatter 가 점 1071개 마다 path 그려서 느림.
        # vlines 로 각 클래스 행에 짧은 stripe 만 표시 — 훨씬 빠름.
        cy = CLASS_Y[cls]
        ax_pr.vlines(times, cy - 0.3, cy + 0.3, colors=color, alpha=0.8, linewidth=1.2)
    ax_pr.set_yticks(list(CLASS_Y.values()))
    ax_pr.set_yticklabels(list(CLASS_Y.keys()))
    ax_pr.set_xlabel("time (s)")
    ax_pr.set_xlim(0, audio_dur)
    ax_pr.set_ylim(-0.5, len(CLASS_Y) - 0.5)
    ax_pr.grid(True, axis="x", alpha=0.3)

    cursor_mel = ax_mel.axvline(0, color="cyan", linewidth=2, alpha=0.9)
    cursor_pr  = ax_pr.axvline(0, color="cyan", linewidth=2, alpha=0.9)

    # 재생 상태 (mutable dict 으로 closure 공유)
    state = {"start_wall": None, "start_pos": 0.0, "playing": False}

    def play_from(t_sec: float):
        sd.stop()
        i = int(t_sec * sr)
        if i >= len(play_audio):
            return
        state["start_wall"] = time.time()
        state["start_pos"] = t_sec
        state["playing"] = True
        sd.play(play_audio[i:], sr)

    def stop():
        sd.stop()
        state["playing"] = False

    def toggle(_event=None):
        if state["playing"]:
            stop()
        else:
            play_from(state["start_pos"])

    def update_cursor(_frame):
        # 재생 중일 때만 cursor 업데이트, 그 외엔 no-op (CPU 절약)
        if not (state["playing"] and state["start_wall"] is not None):
            return cursor_mel, cursor_pr
        elapsed = time.time() - state["start_wall"]
        t = state["start_pos"] + elapsed
        if t >= audio_dur:
            stop()
            t = audio_dur
        cursor_mel.set_xdata([t, t])
        cursor_pr.set_xdata([t, t])
        return cursor_mel, cursor_pr

    # 100ms (10fps) 면 cursor 따라가는 데 충분. blit=True 로 cursor 만 redraw
    anim = FuncAnimation(fig, update_cursor, interval=100,
                          blit=True, cache_frame_data=False)

    def on_click(event):
        if event.inaxes not in (ax_mel, ax_pr) or event.xdata is None:
            return
        was_playing = state["playing"]
        stop()
        t = max(0.0, min(float(event.xdata), audio_dur - 0.01))
        state["start_pos"] = t
        cursor_mel.set_xdata([t, t])
        cursor_pr.set_xdata([t, t])
        if was_playing:
            play_from(t)

    def on_key(event):
        if event.key == " ":
            toggle()
        elif event.key == "escape":
            stop()
        elif event.key == "home":
            stop()
            state["start_pos"] = 0.0
            cursor_mel.set_xdata([0, 0])
            cursor_pr.set_xdata([0, 0])

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print("\n─── controls ────────────────────")
    print("  space  play / pause")
    print("  click  seek to time")
    print("  home   rewind to start")
    print("  esc    stop")
    print("─────────────────────────────────\n")

    print("[4d] tight_layout + show…", flush=True)
    try:
        plt.tight_layout()
        plt.show()
    finally:
        sd.stop()


if __name__ == "__main__":
    main()
