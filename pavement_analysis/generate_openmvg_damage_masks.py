"""Generate OpenMVG masks that retain pavement cracks and potholes.

OpenMVG treats zero-valued (black) pixels as excluded and non-zero pixels as
valid.  The generated masks are therefore white over detected damage and black
everywhere else.  Each mask is saved beside its source image as
``<image_stem>_mask.png``.

This script is tuned for the three DJI frames in ``pavement_analysis/scan``.
Cracks are detected from their local dark-line contrast.  The badly ravelled
pothole is included with a hand-checked polygon per frame so its bright exposed
aggregate is not lost by a dark-crack detector.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


SCAN_DIR = Path(__file__).resolve().parent / "scan"
PREVIEW_DIR = SCAN_DIR / "_mask_previews"
WORK_WIDTH = 2000

# Polygon vertices are normalized to the displayed image width and height.
# They cover the visibly ravelled/pothole area, including its irregular rim.
POTHOLE_POLYGONS: dict[str, list[tuple[float, float]]] = {
    "DJI_20240623182734_0008_D": [
        (0.426, 0.320),
        (0.472, 0.286),
        (0.548, 0.313),
        (0.631, 0.356),
        (0.669, 0.423),
        (0.645, 0.505),
        (0.586, 0.557),
        (0.505, 0.563),
        (0.443, 0.505),
        (0.410, 0.413),
    ],
    "DJI_20240623182741_0009_D": [
        (0.410, 0.337),
        (0.456, 0.302),
        (0.535, 0.317),
        (0.605, 0.357),
        (0.631, 0.415),
        (0.600, 0.489),
        (0.525, 0.535),
        (0.448, 0.512),
        (0.404, 0.445),
    ],
    "DJI_20240623182745_0010_D": [
        (0.403, 0.344),
        (0.450, 0.314),
        (0.523, 0.326),
        (0.586, 0.367),
        (0.616, 0.432),
        (0.588, 0.508),
        (0.519, 0.551),
        (0.448, 0.527),
        (0.398, 0.453),
    ],
}


def shifted(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift a boolean array without wrapping values at the image borders."""
    result = np.zeros_like(mask, dtype=np.uint8)
    height, width = mask.shape
    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx)
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy)
    target_x0 = source_x0 + dx
    target_x1 = source_x1 + dx
    target_y0 = source_y0 + dy
    target_y1 = source_y1 + dy
    result[target_y0:target_y1, target_x0:target_x1] = mask[
        source_y0:source_y1, source_x0:source_x1
    ]
    return result


def directional_support(mask: np.ndarray, radius: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Return strongest and total support over several line orientations."""
    strongest = np.zeros_like(mask, dtype=np.uint8)
    total = np.zeros_like(mask, dtype=np.uint16)
    directions = (
        (1, 0),
        (0, 1),
        (1, 1),
        (1, -1),
        (2, 1),
        (1, 2),
        (2, -1),
        (1, -2),
    )
    for step_x, step_y in directions:
        count = np.zeros_like(mask, dtype=np.uint8)
        for offset in range(-radius, radius + 1):
            count += shifted(mask, offset * step_x, offset * step_y)
        strongest = np.maximum(strongest, count)
        total += count
    return strongest, total


def keep_linear_components(mask: np.ndarray) -> np.ndarray:
    """Remove short isolated texture responses while preserving crack segments."""
    working = mask.copy()
    kept = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    seed_rows, seed_columns = np.nonzero(working)

    for seed_y, seed_x in zip(seed_rows.tolist(), seed_columns.tolist()):
        if not working[seed_y, seed_x]:
            continue
        working[seed_y, seed_x] = False
        queue: deque[tuple[int, int]] = deque([(seed_y, seed_x)])
        pixels: list[tuple[int, int]] = []
        min_x = max_x = seed_x
        min_y = max_y = seed_y

        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
                for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                    if working[neighbor_y, neighbor_x]:
                        working[neighbor_y, neighbor_x] = False
                        queue.append((neighbor_y, neighbor_x))

        span = max(max_x - min_x + 1, max_y - min_y + 1)
        if len(pixels) >= 8 and span >= 9:
            rows, columns = zip(*pixels)
            kept[rows, columns] = True

    return kept


def detect_cracks(image: Image.Image) -> Image.Image:
    """Return a work-resolution binary crack mask for one RGB image."""
    width, height = image.size
    work_height = round(height * WORK_WIDTH / width)
    work = image.resize((WORK_WIDTH, work_height), Image.Resampling.LANCZOS)

    gray = np.asarray(ImageOps.grayscale(work), dtype=np.int16)
    # Large blur approximates the local pavement tone.  Dark cracks then have
    # a positive response even across the road's gradual illumination changes.
    local_tone = np.asarray(
        Image.fromarray(gray.astype(np.uint8)).filter(ImageFilter.GaussianBlur(10.0)),
        dtype=np.int16,
    )
    dark_response = local_tone - gray

    weak = dark_response >= 25
    strong = dark_response >= 43
    support, all_direction_support = directional_support(weak)

    # A crack has a continuous dark centerline.  Requiring several collinear
    # samples plus a dominant orientation removes most randomly oriented
    # aggregate texture and tire specks.
    coherent = support.astype(np.uint16) * 4 >= all_direction_support
    cracks = ((weak & (support >= 5)) | (strong & (support >= 4))) & coherent
    cracks = keep_linear_components(cracks)
    crack_image = Image.fromarray((cracks * 255).astype(np.uint8), mode="L")

    # Give retained OpenMVG features a modest band around each crack centerline.
    crack_image = crack_image.filter(ImageFilter.MaxFilter(11))
    return crack_image


def add_pothole(mask: Image.Image, stem: str) -> None:
    polygon = POTHOLE_POLYGONS.get(stem)
    if not polygon:
        return
    width, height = mask.size
    points = [(round(x * width), round(y * height)) for x, y in polygon]
    ImageDraw.Draw(mask).polygon(points, fill=255)


def make_overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    preview = image.copy()
    preview.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    small_mask = mask.resize(preview.size, Image.Resampling.NEAREST)
    tint = Image.new("RGB", preview.size, (255, 28, 28))
    alpha = small_mask.point(lambda value: 116 if value else 0)
    return Image.composite(tint, preview, alpha)


def process(source: Path) -> tuple[Path, Path]:
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    mask = detect_cracks(image)
    add_pothole(mask, source.stem)
    mask = mask.resize(image.size, Image.Resampling.NEAREST)
    # Enforce a strict 8-bit binary PNG: exactly 0 or 255.
    mask = mask.point(lambda value: 255 if value else 0, mode="L")

    target = source.with_name(f"{source.stem}_mask.png")
    mask.save(target, format="PNG", optimize=True)

    PREVIEW_DIR.mkdir(exist_ok=True)
    overlay_path = PREVIEW_DIR / f"{source.stem}_mask_overlay.png"
    make_overlay(image, mask).save(overlay_path, format="PNG", optimize=True)
    return target, overlay_path


def main() -> None:
    sources = sorted(SCAN_DIR.glob("*.JPG"))
    if not sources:
        raise SystemExit(f"No JPG files found in {SCAN_DIR}")
    for source in sources:
        target, overlay = process(source)
        print(f"{source.name} -> {target.name} (overlay: {overlay.name})")


if __name__ == "__main__":
    main()
