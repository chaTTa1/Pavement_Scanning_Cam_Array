from pathlib import Path

from PIL import Image, ImageOps


SCAN_DIR = Path(__file__).resolve().parent / "scan"
PREVIEW_DIR = SCAN_DIR / "_mask_previews"


def main() -> None:
    PREVIEW_DIR.mkdir(exist_ok=True)
    for source in sorted(SCAN_DIR.glob("*.JPG")):
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            target = PREVIEW_DIR / f"{source.stem}_preview.png"
            image.save(target, format="PNG", optimize=True)
            print(f"{source.name}: {image.size[0]}x{image.size[1]} -> {target}")


if __name__ == "__main__":
    main()
