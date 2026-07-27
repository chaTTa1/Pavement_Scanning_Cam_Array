"""Match supplied crack masks to the DJI frames and convert them for OpenMVG.

The input masks can have any resolution.  Each is matched to the source photo by
comparing its white linework with dark local-contrast lines in the three road
images.  Outputs are strict 8-bit binary PNGs at the exact source resolution and
are named ``<image_stem>_mask.png`` beside the source images.
"""

from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


SCAN_DIR = Path(__file__).resolve().parent / "scan"
PREVIEW_DIR = SCAN_DIR / "_mask_previews"
PHOTO_STEMS = (
    "DJI_20240623182734_0008_D",
    "DJI_20240623182741_0009_D",
    "DJI_20240623182745_0010_D",
)
MATCH_SIZE = (1000, 562)
BINARY_THRESHOLD = 48


def load_mask_weight(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        mask = ImageOps.grayscale(opened).resize(MATCH_SIZE, Image.Resampling.LANCZOS)
    values = np.asarray(mask, dtype=np.float32) / 255.0
    # Suppress nearly black antialias/compression residue while retaining faint
    # thin cracks from the supplied artwork.
    values[values < 0.08] = 0.0
    return values


def load_dark_line_response(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        photo = ImageOps.grayscale(ImageOps.exif_transpose(opened)).resize(
            MATCH_SIZE, Image.Resampling.LANCZOS
        )
    gray = np.asarray(photo, dtype=np.float32)
    local_tone = np.asarray(photo.filter(ImageFilter.GaussianBlur(6.0)), dtype=np.float32)
    dark_lines = np.clip(local_tone - gray, 0.0, None)
    # A small tolerance compensates for antialiasing and tiny drawing offsets.
    dark_lines = np.asarray(
        Image.fromarray(np.clip(dark_lines * 4.0, 0, 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(1.5)
        ),
        dtype=np.float32,
    )
    scale = float(np.percentile(dark_lines, 99.5)) or 1.0
    return np.clip(dark_lines / scale, 0.0, 1.0)


def alignment_score(mask: np.ndarray, dark_lines: np.ndarray) -> float:
    weight = float(mask.sum())
    if weight == 0:
        return float("-inf")
    foreground = float((mask * dark_lines).sum() / weight)
    return foreground - float(dark_lines.mean())


def choose_mapping(inputs: list[Path], photos: list[Path]) -> tuple[int, ...]:
    masks = [load_mask_weight(path) for path in inputs]
    responses = [load_dark_line_response(path) for path in photos]
    scores = np.array(
        [[alignment_score(mask, response) for response in responses] for mask in masks]
    )

    print("Alignment scores (rows=input masks, columns=DJI photos):")
    print(np.array2string(scores, precision=4, suppress_small=True))
    best = max(permutations(range(len(photos))), key=lambda order: sum(scores[i, j] for i, j in enumerate(order)))
    return best


def make_overlay(photo: Image.Image, mask: Image.Image) -> Image.Image:
    preview = photo.copy()
    preview.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    small_mask = mask.resize(preview.size, Image.Resampling.NEAREST)
    red = Image.new("RGB", preview.size, (255, 25, 25))
    alpha = small_mask.point(lambda value: 112 if value else 0)
    return Image.composite(red, preview, alpha)


def convert(input_path: Path, photo_path: Path) -> tuple[Path, Path]:
    with Image.open(photo_path) as opened:
        photo = ImageOps.exif_transpose(opened).convert("RGB")
        target_size = photo.size
    with Image.open(input_path) as opened:
        supplied = ImageOps.grayscale(opened)
        resized = supplied.resize(target_size, Image.Resampling.LANCZOS)
    values = np.asarray(resized, dtype=np.uint8)
    binary = Image.fromarray(np.where(values >= BINARY_THRESHOLD, 255, 0).astype(np.uint8), mode="L")
    output_path = photo_path.with_name(f"{photo_path.stem}_mask.png")
    binary.save(output_path, format="PNG", optimize=True)
    PREVIEW_DIR.mkdir(exist_ok=True)
    overlay_path = PREVIEW_DIR / f"{photo_path.stem}_supplied_mask_overlay.png"
    make_overlay(photo, binary).save(overlay_path, format="PNG", optimize=True)
    return output_path, overlay_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("masks", nargs=3, type=Path, help="Three supplied mask images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [path.resolve() for path in args.masks]
    photos = [SCAN_DIR / f"{stem}.JPG" for stem in PHOTO_STEMS]
    missing = [path for path in inputs + photos if not path.is_file()]
    if missing:
        raise SystemExit("Missing file(s): " + ", ".join(str(path) for path in missing))

    mapping = choose_mapping(inputs, photos)
    for input_path, photo_index in zip(inputs, mapping):
        output, overlay = convert(input_path, photos[photo_index])
        print(f"{input_path.name} -> {output.name} (overlay: {overlay.name})")


if __name__ == "__main__":
    main()
