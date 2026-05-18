"""Stage 1: Demucs로 드럼 스템 분리."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import soundfile as sf


def separate_drums(input_wav: Path, output_dir: Path, model: str = "htdemucs_ft") -> Path:
    """
    Demucs v4로 drums 스템만 분리해 output_dir/drums.wav 로 저장.

    Demucs CLI를 subprocess로 호출 — Python API보다 메모리 관리가 깔끔하고,
    모델 캐싱도 알아서 함.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    drums_out = output_dir / "drums.wav"

    if drums_out.exists():
        print(f"[separation] drums.wav 이미 존재 — 스킵: {drums_out}")
        return drums_out

    # Demucs는 자체 출력 폴더 구조를 만든다: <out>/<model>/<stem>/drums.wav
    tmp_dir = output_dir / "_demucs_tmp"
    tmp_dir.mkdir(exist_ok=True)

    cmd = [
        sys.executable, "-m", "demucs.separate",
        "-n", model,
        "--two-stems", "drums",  # drums vs. no_drums 만 분리 (빠름)
        "-o", str(tmp_dir),
        str(input_wav),
    ]
    print(f"[separation] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Demucs 출력 경로 찾기
    stem_name = input_wav.stem
    demucs_drums = tmp_dir / model / stem_name / "drums.wav"
    if not demucs_drums.exists():
        # 일부 버전은 .mp3로 저장하기도 함 — 일단 검색
        candidates = list((tmp_dir / model / stem_name).glob("drums.*"))
        if not candidates:
            raise FileNotFoundError(f"Demucs output not found in {tmp_dir / model / stem_name}")
        demucs_drums = candidates[0]

    shutil.move(str(demucs_drums), str(drums_out))
    shutil.rmtree(tmp_dir, ignore_errors=True)

    info = sf.info(drums_out)
    print(f"[separation] drums stem saved: {drums_out} ({info.duration:.2f}s, {info.samplerate}Hz)")
    return drums_out
