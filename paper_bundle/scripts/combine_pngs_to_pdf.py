"""Combine all PNG figures into a single PDF with captions.

The script walks the repository (excluding common temporary folders),
adds a caption with the relative file path to each image, and saves
the pages into one PDF for quick review.
"""

import argparse
import textwrap
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageDraw, ImageFont


# Folders that should be skipped while crawling for PNG files.
SKIP_PARTS = {"node_modules", ".git", ".trae", ".vscode", ".history", ".serena", ".cursor"}


def find_pngs(root: Path) -> List[Path]:
    """Return sorted PNG paths under the root while skipping unwanted folders."""

    candidates: List[Path] = []
    for path in root.rglob("*.png"):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.is_file():
            candidates.append(path)

    candidates.sort(key=lambda p: (str(p.parent).lower(), p.name.lower()))
    return candidates


def wrap_caption(text: str, font: ImageFont.ImageFont, max_width_px: int) -> str:
    """Wrap caption text to fit within the target pixel width."""

    max_width_px = max(max_width_px, 40)
    approx_char_width = max(int(max_width_px / 7), 10)
    wrapped = textwrap.wrap(text, width=approx_char_width)
    return "\n".join(wrapped) if wrapped else text


def add_caption(image: Image.Image, caption: str, font: ImageFont.ImageFont) -> Image.Image:
    """Create a new image with a caption area placed above the figure."""

    base = image.convert("RGB")
    wrapped_caption = wrap_caption(caption, font, base.width - 20)

    dummy = Image.new("RGB", (base.width, base.height), "white")
    draw = ImageDraw.Draw(dummy)
    bbox = draw.multiline_textbbox((0, 0), wrapped_caption, font=font, align="center")
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    padding = 16
    caption_height = text_height + padding * 2
    page = Image.new("RGB", (base.width, base.height + caption_height), "white")
    page.paste(base, (0, caption_height))

    draw_page = ImageDraw.Draw(page)
    x_pos = max(padding, int((base.width - text_width) / 2))
    y_pos = max(int((caption_height - text_height) / 2), 4)
    draw_page.multiline_text((x_pos, y_pos), wrapped_caption, fill="black", font=font, align="center")
    return page


def load_font(size: int) -> ImageFont.ImageFont:
    """Load a readable font; fall back to default if system fonts are unavailable."""

    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def build_pages(paths: Iterable[Path], root: Path, font_size: int) -> List[Image.Image]:
    """Open images and attach captions showing relative paths."""

    font = load_font(font_size)
    pages: List[Image.Image] = []
    for img_path in paths:
        with Image.open(img_path) as img:
            caption = str(img_path.resolve().relative_to(root))
            pages.append(add_caption(img, caption, font))
    return pages


def save_pdf(pages: List[Image.Image], output_path: Path) -> None:
    """Persist pages to a single PDF file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first, *rest = pages
    first.save(output_path, save_all=True, append_images=rest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine PNG files into a labeled PDF.")
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Project root to search for PNG files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "figures" / "all_png_figures.pdf",
        help="Destination PDF path.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=30,
        help="Caption font size in points.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_path = args.output.resolve()

    png_paths = find_pngs(root)
    if not png_paths:
        raise SystemExit("No PNG files found; nothing to combine.")

    pages = build_pages(png_paths, root, args.font_size)
    save_pdf(pages, output_path)
    print(f"[OK] Combined {len(png_paths)} PNG files into {output_path}")


if __name__ == "__main__":
    main()
