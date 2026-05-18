"""시각화 3종: deviation 시계열, 클래스별 히스토그램, 비트 위치 히트맵."""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import matplotlib

matplotlib.use("Agg")  # headless 환경 안전
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASS_COLORS = {
    "kick":   "#d62728",
    "snare":  "#1f77b4",
    "hihat":  "#2ca02c",
    "tom":    "#ff7f0e",
    "cymbal": "#9467bd",
}


def plot_deviation_timeline(rows: List[Dict], output_path: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    for cls, color in CLASS_COLORS.items():
        sub = df[df.drum_class == cls]
        if sub.empty:
            continue
        ax.scatter(sub.onset_time_sec, sub.deviation_8th_ms,
                   c=color, s=18, alpha=0.7, label=cls)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("time (sec)")
    ax.set_ylabel("deviation from 8th grid (ms)")
    ax.set_title("Microtiming deviation over time   (+ = laid-back, − = pushing)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    print(f"[viz] saved: {output_path}")


def plot_deviation_histogram(rows: List[Dict], output_path: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    # 데이터에 실제로 있는 클래스만 그림 (n>=2)
    all_classes = ["kick", "snare", "hihat", "tom", "cymbal"]
    classes = [c for c in all_classes if (df.drum_class == c).sum() >= 2]
    if not classes:
        return
    n = len(classes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, cls in zip(axes, classes):
        sub = df[df.drum_class == cls]
        if sub.empty:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(cls)
            continue
        vals = sub.deviation_8th_ms.values
        ax.hist(vals, bins=40, color=CLASS_COLORS[cls], alpha=0.8, edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(float(np.mean(vals)), color="red", linestyle="-", linewidth=1.2,
                   label=f"mean={np.mean(vals):+.1f} ms")
        ax.set_xlabel("deviation (ms)")
        ax.set_title(f"{cls}  (n={len(vals)})")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("count")
    fig.suptitle("Deviation histogram by drum class")
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    print(f"[viz] saved: {output_path}")


def plot_beat_position_heatmap(rows: List[Dict], output_path: Path) -> None:
    """
    x축: 비트 위치 (1박, 1.5박, 2박, ... 4.5박) = 8분 단위 8칸
    y축: 클래스 (kick/snare/hihat)
    셀 색: 평균 deviation (ms)
    """
    df = pd.DataFrame(rows)
    if df.empty or "beat_position" not in df.columns:
        return
    df = df.copy()
    df.beat_position = pd.to_numeric(df.beat_position, errors="coerce")
    df = df.dropna(subset=["beat_position"])
    if df.empty:
        return

    # 8분 bin: 0, 0.5, 1, 1.5, ..., 3.5 (총 8개)
    df["beat_bin"] = (df.beat_position * 2).round().astype(int) % 8

    all_classes = ["kick", "snare", "hihat", "tom", "cymbal"]
    classes = [c for c in all_classes if (df.drum_class == c).any()]
    if not classes:
        return
    nc = len(classes)
    matrix = np.full((nc, 8), np.nan)
    counts = np.zeros((nc, 8), dtype=int)
    for i, cls in enumerate(classes):
        for b in range(8):
            sub = df[(df.drum_class == cls) & (df.beat_bin == b)]
            if not sub.empty:
                matrix[i, b] = float(sub.deviation_8th_ms.mean())
                counts[i, b] = len(sub)

    fig, ax = plt.subplots(figsize=(11, max(3, 1.2 * nc + 1)))
    vmax = np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 1
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(8))
    ax.set_xticklabels(["1", "1&", "2", "2&", "3", "3&", "4", "4&"])
    ax.set_yticks(range(nc))
    ax.set_yticklabels(classes)
    ax.set_xlabel("beat position (8th grid)")
    ax.set_title("Mean deviation by beat position   (red=late, blue=early)")

    for i in range(nc):
        for b in range(8):
            if not np.isnan(matrix[i, b]):
                ax.text(b, i, f"{matrix[i,b]:+.1f}\n(n={counts[i,b]})",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(matrix[i,b]) > vmax * 0.5 else "black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean deviation (ms)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    print(f"[viz] saved: {output_path}")


def plot_velocity_distribution(rows: List[Dict], output_path: Path) -> None:
    """클래스별 velocity 분포 + ghost/accent threshold 선."""
    df = pd.DataFrame(rows)
    if df.empty or "velocity" not in df.columns:
        return
    all_classes = ["kick", "snare", "hihat", "tom", "cymbal"]
    classes = [c for c in all_classes if (df.drum_class == c).sum() >= 2]
    if not classes:
        return
    n = len(classes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, cls in zip(axes, classes):
        sub = df[df.drum_class == cls]
        v = sub.velocity.values
        ax.hist(v, bins=16, range=(1, 128), color=CLASS_COLORS[cls], alpha=0.85, edgecolor="white")
        # ghost / accent 분리 선
        if "is_ghost" in sub.columns:
            ghosts = sub[sub.is_ghost == 1]
            if len(ghosts) > 0:
                threshold = int(ghosts.velocity.max()) + 1
                ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
                           label=f"ghost ≤ {threshold-1}")
                ax.legend()
        ax.set_xlabel("velocity")
        ax.set_title(f"{cls}  (n={len(sub)})")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("count")
    fig.suptitle("Velocity distribution + ghost/accent threshold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    print(f"[viz] saved: {output_path}")


def plot_ghost_vs_accent(rows: List[Dict], output_path: Path) -> None:
    """Snare (또는 가장 ghost 많은 클래스)의 ghost vs accent deviation 분포 비교."""
    df = pd.DataFrame(rows)
    if df.empty or "is_ghost" not in df.columns:
        return
    df["is_ghost"] = pd.to_numeric(df.is_ghost, errors="coerce").fillna(0).astype(int)
    if df.is_ghost.sum() == 0:
        return

    # ghost 가 가장 많은 클래스 선택 (보통 snare)
    candidate = (df[df.is_ghost == 1].groupby("drum_class").size()
                 .sort_values(ascending=False).index)
    if len(candidate) == 0:
        return
    cls = candidate[0]
    sub = df[df.drum_class == cls]
    accent = sub[sub.is_ghost == 0]
    ghost = sub[sub.is_ghost == 1]
    if len(accent) < 2 or len(ghost) < 2:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: 시계열 산포
    ax1.scatter(accent.onset_time_sec, accent.deviation_16th_ms,
                c=CLASS_COLORS[cls], s=22, alpha=0.7, label=f"{cls} accent (n={len(accent)})")
    ax1.scatter(ghost.onset_time_sec, ghost.deviation_16th_ms,
                c=CLASS_COLORS[cls], s=14, alpha=0.5, marker="x",
                label=f"{cls} ghost (n={len(ghost)})")
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("time (sec)")
    ax1.set_ylabel("deviation from 16th grid (ms)")
    ax1.set_title(f"{cls} accent vs ghost — timeline")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: deviation 히스토그램 비교
    bins = np.linspace(min(sub.deviation_16th_ms.min(), -30),
                       max(sub.deviation_16th_ms.max(), 30), 30)
    ax2.hist(accent.deviation_16th_ms, bins=bins, alpha=0.6,
             color=CLASS_COLORS[cls], label=f"accent  μ={accent.deviation_16th_ms.mean():+.1f}ms")
    ax2.hist(ghost.deviation_16th_ms, bins=bins, alpha=0.6,
             color="gray", label=f"ghost   μ={ghost.deviation_16th_ms.mean():+.1f}ms")
    ax2.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("deviation (ms)")
    ax2.set_ylabel("count")
    ax2.set_title(f"{cls} deviation distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    print(f"[viz] saved: {output_path}")


def render_all(rows: List[Dict], viz_dir: Path) -> None:
    viz_dir.mkdir(parents=True, exist_ok=True)
    plot_deviation_timeline(rows, viz_dir / "dev_timeline.png")
    plot_deviation_histogram(rows, viz_dir / "dev_histogram.png")
    plot_beat_position_heatmap(rows, viz_dir / "beat_position_heatmap.png")
    plot_velocity_distribution(rows, viz_dir / "velocity_distribution.png")
    plot_ghost_vs_accent(rows, viz_dir / "ghost_vs_accent.png")
