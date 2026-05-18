# Drum2midi — Drum Microtiming Analysis

WAV 음원에서 드럼의 인간적 마이크로타이밍(스윙, push/pull)을 분석한다.
드럼을 그리드에 스냅하지 않고, **각 드럼 히트가 비트 그리드 대비 몇 ms
밀리거나 당겨졌는지**를 측정한다.

입력: 다른 멜로딕 악기가 섞인 전자음악 (덥/EDM, 보컬 거의 없음 가정).

## 파이프라인

1. **Demucs**로 drums 스템 분리
2. **드럼 전사** (ADTOF → OaF → librosa 폴백): kick/snare/hihat onset 검출
3. **정밀 onset 재측정**: 분리된 드럼 스템에서 샘플 단위 정밀도 확보
4. **비트 그리드 + deviation 계산**: madmom DBNBeatTracker로 그리드 추출,
   각 onset의 그리드 대비 deviation(ms) 기록 (스냅 없음)

## 설치

### 0. 가상환경

```bash
cd /Users/xyl/Documents/GitHub/Drum2midi
python3.9 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
```

> Python 3.9 권장. 3.11+는 madmom Cython 빌드가 깨질 수 있다.
> 그래도 코드는 madmom 실패시 librosa 폴백으로 동작한다.

### 1. 코어 + Demucs + madmom

```bash
pip install -r requirements.txt
```

처음 실행하면 Demucs가 htdemucs_ft 모델 가중치(~300MB)를 자동 다운로드한다.

### 2. (선택) ADTOF — 더 정확한 드럼 클래스 분류

ADTOF는 pip로 안 깔린다. 별도 clone + 설치 필요:

```bash
git clone https://github.com/MZehren/ADTOF.git external/ADTOF
cd external/ADTOF
pip install -e .
# 모델 가중치는 ADTOF 레포 README의 다운로드 링크 참고
cd ../..
```

설치 실패해도 파이프라인은 librosa 대역별 폴백으로 계속 동작한다.
어느 method가 쓰였는지는 콘솔에 명시된다.

### 3. (대안) Magenta OaF Drums

ADTOF가 안 되면 Magenta OaF를 시도해본다:

```bash
pip install magenta  # tensorflow 의존성으로 무겁다, 실패 가능
```

이것도 안 되면 자동으로 librosa 폴백으로 떨어진다.

## 사용

```bash
python analyze.py path/to/track.wav
```

옵션:

```bash
python analyze.py track.wav \
  --output-dir outputs/track \
  --skip-separation         # 이미 drums.wav 분리됐으면 스킵
  --skip-transcription      # raw_onsets.json 있으면 스킵
  --transcription-method auto|adtof|oaf|librosa  # 강제 지정
  --beat-tracker auto|madmom|librosa
```

## 산출물 구조

```
outputs/<track>/
├── drums.wav               ← 캐시: Demucs 분리 드럼 스템
├── raw_onsets.json         ← 캐시: Stage 2 거친 onset
├── refined_onsets.json     ← 캐시: Stage 3 정밀 onset
├── beat_grid.json          ← 캐시: librosa/madmom 비트 시각
└── runs/                   ★ 매 실행마다 timestamped 서브폴더
    ├── 2026-05-19_00-15-30/                (시간 자동)
    └── 2026-05-19_00-25-00_with-ghost/     (--run-name 으로 라벨)
        ├── drums.mid              원본 절대 타이밍 보존 (PPQ 1920)
        ├── timing_analysis.csv    onset별 deviation + ghost 라벨
        ├── summary.txt            클래스별 통계 + swing + ghost 분석
        ├── ableton_sync.txt       정밀 BPM 과 DAW 동기화 가이드
        └── viz/
            ├── dev_timeline.png
            ├── dev_histogram.png
            ├── beat_position_heatmap.png
            ├── velocity_distribution.png
            └── ghost_vs_accent.png
```

**왜 이렇게:** Stage 1~4 무거운 캐시는 top-level 에 두고 (재실행시 스킵),
실행마다 다른 결과물(MIDI/CSV/summary/viz)은 timestamped 폴더에 분리.
서로 다른 실험(다른 비트 트래커, 다른 threshold 등) 결과 비교가 깔끔.

## 단계별 재실행

각 단계가 산출물을 파일로 저장하기 때문에 `--skip-*` 플래그로 중간부터
재실행 가능. 예: 비트 트래커만 다시 돌리고 싶으면

```bash
python analyze.py track.wav --skip-separation --skip-transcription
```

## 출력 해석

- `deviation_8th_ms` 양수 = 그리드보다 **늦게** (laid-back, behind the beat)
- 음수 = 그리드보다 **일찍** (rushed, on top / pushing)
- `swing_ratio` = 8분음표 쌍의 long/short 길이 비. 1.0=균등, 1.5=트리플렛 스윙,
  2.0=완전 셔플
