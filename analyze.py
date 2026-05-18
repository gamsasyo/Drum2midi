"""
Drum2midi — main CLI.

사용법:
    python analyze.py path/to/mix.wav

각 단계의 산출물은 outputs/<filename>/ 에 저장되고, --skip-* 플래그로
중간부터 재실행 가능. 자세한 옵션은 --help.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from src.separation import separate_drums
from src.transcription import transcribe_drums, DRUM_CLASSES
from src.refinement import refine_onsets
from src.ghost import annotate_ghosts
from src.grid import (extract_beat_grid, compute_deviations,
                       detect_phase_shift, correct_bpm_octave)
from src.midi_export import export_midi
from src.stats import summarize
from src.viz import render_all


CSV_COLUMNS = [
    "onset_time_sec", "drum_class",
    "nearest_grid_8th_sec", "deviation_8th_ms",
    "nearest_grid_16th_sec", "deviation_16th_ms",
    "beat_position", "velocity", "is_ghost",
]


def main():
    p = argparse.ArgumentParser(description="Drum microtiming analyzer")
    p.add_argument("input_wav", type=Path, help="입력 WAV 파일")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="출력 디렉토리 (기본: outputs/<input_stem>)")
    p.add_argument("--skip-separation", action="store_true",
                   help="<output>/drums.wav 가 이미 있으면 Demucs 스킵 (자동 캐싱도 됨)")
    p.add_argument("--skip-transcription", action="store_true",
                   help="raw_onsets.json 있으면 Stage 2 스킵")
    p.add_argument("--transcription-method", default="auto",
                   choices=["auto", "adtof", "oaf", "librosa"],
                   help="Stage 2 method 강제")
    p.add_argument("--adtof-thresholds", default=None,
                   help="ADTOF per-class 임계값 (낮을수록 더 잡힘). "
                        "comma-separated 5개: kick,snare,tom,hihat,cymbal. "
                        "기본 [0.22,0.24,0.32,0.22,0.30]. 예: 0.15,0.18,0.25,0.15,0.22")
    p.add_argument("--beat-tracker", default="auto",
                   choices=["auto", "madmom", "librosa"],
                   help="Stage 4 비트 트래커 강제")
    p.add_argument("--bpm-hint", type=float, default=None,
                   help="진짜 BPM 힌트 (예: 88). librosa 가 octave error "
                        "(2x / 0.5x / 3x / 1/3x) 일 때 자동 보정. "
                        "Shazam 등에서 BPM 확인 후 사용 권장.")
    p.add_argument("--demucs-model", default="htdemucs_ft",
                   help="Demucs 모델 (기본: htdemucs_ft)")
    p.add_argument("--run-name", default=None,
                   help="이번 실행 결과 폴더 이름 suffix (기본: 타임스탬프만)")
    args = p.parse_args()

    if not args.input_wav.exists():
        print(f"ERROR: input file not found: {args.input_wav}", file=sys.stderr)
        sys.exit(1)

    out = args.output_dir or (Path("outputs") / args.input_wav.stem)
    out.mkdir(parents=True, exist_ok=True)

    # 이번 실행 결과를 담을 timestamped 서브폴더
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{ts}_{args.run_name}" if args.run_name else ts
    run_dir = out / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = run_dir / "viz"

    # ──────────── Stage 1 ────────────
    print("\n" + "=" * 64)
    print("Stage 1: SEPARATION (Demucs)")
    print("=" * 64)
    drums_wav = out / "drums.wav"
    if args.skip_separation and drums_wav.exists():
        print(f"[separation] --skip-separation: 기존 {drums_wav} 사용")
    else:
        drums_wav = separate_drums(args.input_wav, out, model=args.demucs_model)

    # ──────────── Stage 2 ────────────
    print("\n" + "=" * 64)
    print("Stage 2: TRANSCRIPTION (coarse onsets)")
    print("=" * 64)
    if args.skip_transcription and (out / "raw_onsets.json").exists() and not args.adtof_thresholds:
        print(f"[transcription] --skip-transcription: 기존 raw_onsets.json 사용")
        import json
        cached = json.loads((out / "raw_onsets.json").read_text())
        raw_onsets, method = cached["onsets"], cached.get("method", "unknown")
    else:
        # threshold 바꾸면 캐시 무효화
        if args.adtof_thresholds and (out / "raw_onsets.json").exists():
            (out / "raw_onsets.json").unlink()
            (out / "refined_onsets.json").unlink(missing_ok=True)
            print("[transcription] --adtof-thresholds 지정됨 → 캐시 삭제 후 재실행")
        adtof_th = None
        if args.adtof_thresholds:
            try:
                adtof_th = [float(x) for x in args.adtof_thresholds.split(",")]
                if len(adtof_th) != 5:
                    raise ValueError(f"5개 필요, {len(adtof_th)}개 받음")
            except Exception as e:
                print(f"ERROR: --adtof-thresholds 형식 오류: {e}", file=sys.stderr)
                sys.exit(1)
        raw_onsets, method = transcribe_drums(
            drums_wav, out,
            method=args.transcription_method,
            adtof_thresholds=adtof_th,
        )

    # ──────────── Stage 3 ────────────
    print("\n" + "=" * 64)
    print("Stage 3: REFINEMENT (sample-precision onsets)")
    print("=" * 64)
    refined = refine_onsets(drums_wav, raw_onsets, out)

    # Ghost note 라벨링 (velocity 분포 bimodal 검출)
    print("\n[ghost] velocity 분포 기반 ghost/accent 분리")
    ghost_diag = annotate_ghosts(refined)
    for cls, d in ghost_diag.items():
        if d["threshold"] is not None:
            print(f"  {cls:8s} threshold={d['threshold']:3d}  "
                  f"accent={d['n_accent']:4d}  ghost={d['n_ghost']:4d}  "
                  f"({d['msg']})")
        else:
            print(f"  {cls:8s} ghost 검출 안 됨 ({d['msg']})")

    # ──────────── Stage 4 ────────────
    print("\n" + "=" * 64)
    print("Stage 4: BEAT GRID + DEVIATION")
    print("=" * 64)
    beats, tracker = extract_beat_grid(drums_wav, out, tracker=args.beat_tracker)

    # BPM hint 가 있으면 octave error (2x/0.5x/3x/1/3x) 자동 보정
    if args.bpm_hint is not None:
        beats = correct_bpm_octave(beats, args.bpm_hint)

    # Phase shift 자동 보정: librosa-beats 가 ½ IBI 어긋난 자리에 잡힌 경우 당김
    shift = detect_phase_shift(beats, raw_onsets)
    if shift > 0:
        beats = beats - shift  # beats 를 앞으로 당기면 onset 들이 새 grid 의 downbeat 자리로 옴
        # 음수가 된 첫 박은 잘라냄
        beats = beats[beats >= 0]

    rows = compute_deviations(refined, beats)

    # ──────────── Outputs ────────────
    print("\n" + "=" * 64)
    print("OUTPUTS")
    print("=" * 64)

    # CSV
    csv_path = run_dir / "timing_analysis.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] saved: {csv_path}  ({len(rows)} rows)")

    # MIDI (Ableton clip 길이 매칭 위해 원본 오디오 길이까지 패딩)
    import soundfile as sf
    from src.midi_export import estimate_precise_bpm
    audio_dur = sf.info(str(args.input_wav)).duration
    midi_path = run_dir / "drums.mid"
    export_midi(refined, beats, midi_path, audio_duration_sec=audio_dur)

    # DAW 동기화 가이드
    precise_bpm, drift_ms = estimate_precise_bpm(beats)
    guide_path = run_dir / "ableton_sync.txt"
    guide_path.write_text(
        f"""DAW 동기화 가이드
================================================================
이 곡의 정밀 BPM:  {precise_bpm:.4f}
첫 비트 시각:       {beats[0]:.4f}s
fit 최대 편차:      {drift_ms:.1f} ms

★ Ableton 사용시:
  1. 프로젝트 BPM 을 정확히 {precise_bpm:.4f} 로 설정
     (Tempo 필드 클릭 → 소수점 4자리까지 입력)
  2. drums.mid 와 원본 오디오 둘 다 동일 트랙 시작점에 정렬
  3. MIDI clip 의 "Original BPM" 도 {precise_bpm:.4f} 로 설정
  4. 오디오는 warp 비활성화 (원본 그대로)

★ 그래도 드리프트 발견시:
  - 곡 자체가 가변 BPM (fit 최대 편차 {drift_ms:.1f}ms 가 30ms+ 면 의심)
  - 또는 librosa beat tracker 의 한계 (madmom / Beat-this! 같은 SOTA 필요)
================================================================
"""
    )
    print(f"[guide] saved: {guide_path}")

    # Summary
    summary_path = run_dir / "summary.txt"
    summarize(rows, summary_path)

    # Viz
    render_all(rows, viz_dir)

    print("\n" + "=" * 64)
    print(f"DONE. 이번 실행 결과: {run_dir}")
    print(f"     캐시 (재실행시 재사용): {out}")
    print(f"  transcription method: {method}")
    print(f"  beat tracker:         {tracker}")
    print("=" * 64)


if __name__ == "__main__":
    main()
