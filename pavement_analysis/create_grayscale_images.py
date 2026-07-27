"""Create lossless grayscale copies of the original pavement photographs."""

from pathlib import Path

from PIL import Image, ImageOps


SCAN_DIR = Path(__file__).resolve().parent / "scan"
OUTPUT_DIR = SCAN_DIR / "grayscale"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    sources = sorted(SCAN_DIR.glob("*.JPG"))
    if not sources:
        raise SystemExit(f"No JPG images found in {SCAN_DIR}")

    for source in sources:
        with Image.open(source) as opened:
            gray = ImageOps.grayscale(ImageOps.exif_transpose(opened))
            output = OUTPUT_DIR / f"{source.stem}_gray.png"
            gray.save(output, format="PNG", optimize=True)
            print(f"{source.name} -> {output.name} ({gray.width}x{gray.height}, {gray.mode})")


if __name__ == "__main__":
    main()
