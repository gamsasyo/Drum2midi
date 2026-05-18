"""
Stage 2: 드럼 전사 (coarse onset detection).

폴백 채인: ADTOF → OaF → librosa 대역별 onset detection.
어느 method가 쓰였는지 명확히 반환.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

DRUM_CLASSES = ("kick", "snare", "hihat", "tom", "cymbal")


def _bandpass(y: np.ndarray, sr: int, low: Optional[float], high: Optional[float], order: int = 4) -> np.ndarray:
    nyq = sr / 2
    if low is None:
        sos = butter(order, high / nyq, btype="lowpass", output="sos")
    elif high is None:
        sos = butter(order, low / nyq, btype="highpass", output="sos")
    else:
        sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, y)


# ──────────────────────────────────────────────────────────────────
# Method 1: ADTOF (PyTorch port — xavriley/ADTOF-pytorch)
# ──────────────────────────────────────────────────────────────────

# ADTOF Frame_RNN 출력 5클래스 → 우리 내부 이름
# LABELS_5 = [35, 38, 47, 42, 49] (model output 순서)
ADTOF_PITCH_TO_CLASS = {
    35: "kick",    # BD - bass drum
    38: "snare",   # SD - snare drum
    47: "tom",     # TT - all toms 통합
    42: "hihat",   # HH - all hi-hats 통합 (closed/open/pedal)
    49: "cymbal",  # CY - all cymbals 통합 (ride/crash/splash)
}


def _try_adtof(drums_wav: Path, output_dir: Path) -> Optional[Dict[str, List[float]]]:
    """ADTOF-pytorch (xavriley) 사용. 5클래스 반환."""
    try:
        import importlib.util
        if importlib.util.find_spec("adtof_pytorch") is None:
            return None

        from adtof_pytorch import transcribe_to_midi
        import pretty_midi

        tmp_mid = output_dir / "_adtof_raw.mid"
        transcribe_to_midi(str(drums_wav), str(tmp_mid), device="cpu")

        pm = pretty_midi.PrettyMIDI(str(tmp_mid))
        onsets: Dict[str, List[float]] = {c: [] for c in DRUM_CLASSES}
        for inst in pm.instruments:
            for note in inst.notes:
                cls = ADTOF_PITCH_TO_CLASS.get(note.pitch)
                if cls:
                    onsets[cls].append(float(note.start))

        for cls in onsets:
            onsets[cls].sort()

        if not any(onsets.values()):
            return None
        return onsets
    except Exception as e:
        print(f"[transcription/adtof] failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────────
# Method 2: OaF Drums (Magenta)
# ──────────────────────────────────────────────────────────────────

def _try_oaf(drums_wav: Path) -> Optional[Dict[str, List[float]]]:
    """Magenta OaF Drums 시도. 무거운 tensorflow 의존성 — 실패 가능성 높음."""
    try:
        import importlib.util
        if importlib.util.find_spec("magenta") is None:
            return None

        # OaF Drums는 magenta.models.onsets_frames_transcription.drums_pipeline 모듈 사용
        # 환경마다 인터페이스 변동 심함 — 일단 스켈레톤만 두고 실제 호출은 placeholder
        # (사용자 환경에서 magenta 설치 성공시 채워넣을 수 있게 함)
        print("[transcription/oaf] magenta가 설치되어 있지만 OaF 드럼 모델은 환경 의존성이 커서 "
              "현재 구현은 librosa 폴백으로 떨어집니다. (필요시 외부에서 OaF 결과를 raw_onsets.json으로 미리 만들어 두면 활용 가능)")
        return None
    except Exception as e:
        print(f"[transcription/oaf] failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────────
# Method 3: librosa 대역별 폴백 (항상 작동)
# ──────────────────────────────────────────────────────────────────

# 클래스별 밴드패스 + onset detect 파라미터
# 주: librosa 폴백은 5클래스 분류 정확도가 낮아 일반적으로 ADTOF 권장.
# tom/cymbal은 hihat/kick과 스펙트럼 겹침 커서 false positive 가능성 있음.
BAND_CONFIG = {
    "kick":   {"low": 30,   "high": 200,  "delta": 0.30, "wait_ms": 30},
    "snare":  {"low": 150,  "high": 800,  "delta": 0.25, "wait_ms": 30},
    "hihat":  {"low": 5000, "high": None, "delta": 0.20, "wait_ms": 15},
    "tom":    {"low": 80,   "high": 300,  "delta": 0.35, "wait_ms": 40},
    "cymbal": {"low": 3000, "high": None, "delta": 0.35, "wait_ms": 80},
}


def _librosa_band_onsets(drums_wav: Path) -> Dict[str, List[float]]:
    """
    클래스별로 분리 드럼 스템을 밴드패스 후 onset detection.
    스네어는 추가로 high-freq transient 가중치를 줘서 킥과 분리.
    """
    y, sr = librosa.load(str(drums_wav), sr=None, mono=True)
    onsets: Dict[str, List[float]] = {}

    for cls, cfg in BAND_CONFIG.items():
        y_band = _bandpass(y, sr, cfg["low"], cfg["high"])

        # 스네어의 경우 high-freq transient 보강 (snare wire 노이즈)
        if cls == "snare":
            y_hi = _bandpass(y, sr, 5000, None)
            # 두 envelope를 곱해서 "mid-band 임팩트 + high-freq 노이즈" 둘 다 있는 시각만 강조
            env_mid = librosa.onset.onset_strength(y=y_band, sr=sr)
            env_hi = librosa.onset.onset_strength(y=y_hi, sr=sr)
            # 두 envelope의 곱(요소별) — 길이 맞추기
            n = min(len(env_mid), len(env_hi))
            env = env_mid[:n] * env_hi[:n]
            # 정규화
            env = env / (env.max() + 1e-9)
        else:
            env = librosa.onset.onset_strength(y=y_band, sr=sr)
            env = env / (env.max() + 1e-9)

        wait_frames = max(1, int(cfg["wait_ms"] / 1000 * sr / 512))
        frames = librosa.onset.onset_detect(
            onset_envelope=env,
            sr=sr,
            hop_length=512,
            delta=cfg["delta"],
            wait=wait_frames,
            backtrack=False,  # Stage 3에서 정밀 재측정할 거라 여기선 backtrack 안 함
            units="frames",
        )
        times = librosa.frames_to_time(frames, sr=sr, hop_length=512)
        onsets[cls] = times.tolist()
        print(f"[transcription/librosa] {cls}: {len(times)} onsets")

    # 킥/스네어가 거의 같은 시각에 동시 검출되는 경우 (저주파 누설) — 더 강한 쪽만 남김
    onsets = _resolve_kick_snare_conflicts(onsets, y, sr, tolerance_ms=15)
    return onsets


def _resolve_kick_snare_conflicts(
    onsets: Dict[str, List[float]], y: np.ndarray, sr: int, tolerance_ms: float = 15
) -> Dict[str, List[float]]:
    """킥/스네어가 ±tolerance_ms 안에 둘 다 잡히면 에너지 비교해 한 쪽만 남김."""
    tol = tolerance_ms / 1000
    kicks = sorted(onsets["kick"])
    snares = sorted(onsets["snare"])
    keep_kick = set(range(len(kicks)))
    keep_snare = set(range(len(snares)))

    for i, kt in enumerate(kicks):
        for j, st in enumerate(snares):
            if abs(kt - st) > tol:
                continue
            # 각각의 밴드에서 RMS 비교
            center = int(((kt + st) / 2) * sr)
            half = int(0.020 * sr)  # ±20ms
            seg = y[max(0, center - half): center + half]
            if len(seg) == 0:
                continue
            kick_band = _bandpass(seg, sr, 30, 200)
            snare_band = _bandpass(seg, sr, 150, 800)
            kick_e = float(np.sqrt(np.mean(kick_band ** 2)))
            snare_e = float(np.sqrt(np.mean(snare_band ** 2)))
            if kick_e > snare_e:
                keep_snare.discard(j)
            else:
                keep_kick.discard(i)

    return {
        "kick":  [kicks[i] for i in sorted(keep_kick)],
        "snare": [snares[j] for j in sorted(keep_snare)],
        "hihat": onsets["hihat"],
    }


# ──────────────────────────────────────────────────────────────────
# 메인 진입점
# ──────────────────────────────────────────────────────────────────

def transcribe_drums(
    drums_wav: Path, output_dir: Path, method: str = "auto"
) -> tuple[Dict[str, List[float]], str]:
    """
    드럼 onset 검출. 반환: (onsets_dict, used_method).
    onsets_dict: {"kick": [t_sec, ...], "snare": [...], "hihat": [...]}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "raw_onsets.json"
    if cache.exists():
        data = json.loads(cache.read_text())
        print(f"[transcription] cached raw_onsets.json 사용 (method={data.get('method')})")
        return data["onsets"], data.get("method", "unknown")

    methods_to_try = []
    if method == "auto":
        methods_to_try = ["adtof", "oaf", "librosa"]
    else:
        methods_to_try = [method]

    used = None
    onsets = None
    for m in methods_to_try:
        if m == "adtof":
            onsets = _try_adtof(drums_wav, output_dir)
        elif m == "oaf":
            onsets = _try_oaf(drums_wav)
        elif m == "librosa":
            onsets = _librosa_band_onsets(drums_wav)
        else:
            raise ValueError(f"Unknown method: {m}")
        if onsets is not None:
            used = m
            break

    if onsets is None:
        raise RuntimeError("모든 transcription method 실패")

    print(f"\n[transcription] ★ USED METHOD: {used}")
    for cls in DRUM_CLASSES:
        print(f"  {cls}: {len(onsets.get(cls, []))} onsets")

    cache.write_text(json.dumps({"method": used, "onsets": onsets}, indent=2))
    return onsets, used
