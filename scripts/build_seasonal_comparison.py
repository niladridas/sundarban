"""Stitch the four seasonal MP4s into a synchronized 2×2 comparison video.

Order:                Top:    Pre-monsoon (Apr–May) | Peak monsoon (Aug)
                      Bottom: Post-monsoon (Oct–Nov)| Dry season (Jan–Feb)

Each cell is the original season's animation, downscaled to fit. Frames
are matched by index (all four runs use the same dt and runtime, so step
N = the same elapsed simulation time).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from rich.console import Console

from driftscope.config import DATA

console = Console()

ANIM_DIR = DATA / "animations"
SEASON_FILES = {
    "Pre-monsoon": ANIM_DIR / "gbm_premonsoon.mp4",
    "Peak monsoon": ANIM_DIR / "gbm_monsoon.mp4",
    "Post-monsoon": ANIM_DIR / "gbm_postmonsoon.mp4",
    "Dry season": ANIM_DIR / "gbm_dry.mp4",
}
OUT_PATH = ANIM_DIR / "gbm_seasonal_comparison.mp4"


def main():
    # Load all four
    videos = {}
    for label, path in SEASON_FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        frames = list(iio.imiter(path))
        videos[label] = frames
        console.print(f"   {label}: {len(frames)} frames {frames[0].shape}")

    # Take the minimum frame count (all should be ~200)
    n = min(len(v) for v in videos.values())
    h, w, _ = next(iter(videos.values()))[0].shape
    console.print(f"[cyan]·[/cyan] stitching {n} synchronized frames @ 2×{h}×{w}×3")

    # Stitch
    out_frames = []
    order = ["Pre-monsoon", "Peak monsoon", "Post-monsoon", "Dry season"]
    for step in range(n):
        top = np.concatenate([videos[order[0]][step], videos[order[1]][step]], axis=1)
        bot = np.concatenate([videos[order[2]][step], videos[order[3]][step]], axis=1)
        full = np.concatenate([top, bot], axis=0)
        # Pad to even dimensions for libx264 if needed
        if full.shape[0] % 2:
            full = full[:-1]
        if full.shape[1] % 2:
            full = full[:, :-1]
        out_frames.append(full)
        if (step + 1) % 25 == 0:
            console.print(f"   {step+1}/{n}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(OUT_PATH, out_frames, fps=15, codec="libx264", quality=7)
    console.print(
        f"[green]✓[/green] {OUT_PATH} "
        f"({len(out_frames)} frames, {OUT_PATH.stat().st_size/1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
