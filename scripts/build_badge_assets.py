#!/usr/bin/env python3
"""Build clean flat badge assets strictly from the exact source icon PNGs.

Design contract:
- Bronze / silver / gold badges use plain flat coin shapes.
- Diamond badges use a plain flat light-blue diamond shape.
- Non-diamond icons are silhouettes made from the exact source icon images and rendered black.
- Diamond icons are silhouettes made from the exact source icon images and rendered black.
- There are no generated fallback icons. Missing source art stops the build.

Run from the project root:
    python3 scripts/bootstrap_badge_sources.py
    python3 scripts/build_badge_assets.py
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "frontend" / "public" / "badges" / "source"
OUTPUT_DIR = PROJECT_ROOT / "frontend" / "public" / "badges" / "generated"
BADGE_SIZE = 192
CANVAS_SCALE = 4
TIERS = ("diamond", "gold", "silver", "bronze")
SOURCE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

TIER_FILL_COLORS: Dict[str, Tuple[int, int, int, int]] = {
    "diamond": (114, 232, 255, 255),
    "gold": (242, 193, 61, 255),
    "silver": (198, 205, 214, 255),
    "bronze": (181, 105, 52, 255),
}

ICON_COLORS: Dict[str, Tuple[int, int, int, int]] = {
    "diamond": (0, 0, 0, 255),
    "gold": (0, 0, 0, 255),
    "silver": (0, 0, 0, 255),
    "bronze": (0, 0, 0, 255),
}

BADGE_SOURCE_IDS: Dict[str, str] = {
    "deep_range_bomber": "deep_range_bomber",
    "catch_and_shoot_converter": "catch_and_shoot_converter",
    "contested_3pt_maker": "contested_3pt_maker",
    "pull_up_3pt_machine": "pull_up_3pt_machine",
    "volume_3pt_shooter": "volume_3pt_shooter",
    "three_pt_sniper": "three_pt_sniper",
    "volume_mid_range_shooter": "volume_mid_range_shooter",
    "mid_range_assassin": "mid_range_assassin",
    "volume_slasher": "volume_slasher",
    "efficient_driver": "efficient_driver",
    "free_throw_generator": "free_throw_generator",
    "drive_and_kicker": "drive_and_kicker",
    "inside_the_arc_scorer": "inside_the_arc_scorer",
    "walking_bucket": "walking_bucket",
    "dunker": "dunker",
    "active_hands": "active_hands",
    "defensive_lock_down": "defensive_lock_down",
    "assist_generator": "assist_generator",
    "efficient_passer": "efficient_passer",
}

# These are not placeholders. They only control how large each exact source icon
# is composited into the flat tier shape after being converted into a silhouette.
ICON_SIZES: Dict[str, int] = {
    "deep_range_bomber": 106,
    "catch_and_shoot_converter": 112,
    "contested_3pt_maker": 108,
    "pull_up_3pt_machine": 118,
    "volume_3pt_shooter": 108,
    "three_pt_sniper": 108,
    "volume_mid_range_shooter": 112,
    "mid_range_assassin": 112,
    "volume_slasher": 112,
    "efficient_driver": 112,
    "free_throw_generator": 116,
    "drive_and_kicker": 118,
    "inside_the_arc_scorer": 104,
    "walking_bucket": 112,
    "dunker": 116,
    "active_hands": 110,
    "defensive_lock_down": 110,
    "assist_generator": 110,
    "efficient_passer": 110,
}

ICON_VERTICAL_OFFSETS: Dict[str, int] = {
    "pull_up_3pt_machine": 4,
    "drive_and_kicker": 4,
    "free_throw_generator": 2,
    "dunker": 2,
}


def find_source_image(source_id: str) -> Optional[Path]:
    for extension in SOURCE_EXTENSIONS:
        candidate = SOURCE_DIR / f"{source_id}{extension}"
        if candidate.exists():
            return candidate
    return None


def require_source_image(source_id: str) -> Path:
    source_path = find_source_image(source_id)
    if source_path is not None:
        return source_path
    expected = ", ".join(f"{source_id}{extension}" for extension in SOURCE_EXTENSIONS)
    raise FileNotFoundError(
        f"Missing exact badge source image for '{source_id}'. Expected one of: {expected} in {SOURCE_DIR}.\n"
        "Run: python3 scripts/bootstrap_badge_sources.py\n"
        "No fallback silhouette will be generated."
    )


def validate_all_sources() -> None:
    missing = [source_id for source_id in sorted(set(BADGE_SOURCE_IDS.values())) if find_source_image(source_id) is None]
    if not missing:
        return
    formatted = "\n".join(f"  - {source_id}" for source_id in missing)
    raise FileNotFoundError(
        "Cannot build badge assets because exact source icon images are missing:\n"
        f"{formatted}\n\n"
        f"Expected files under: {SOURCE_DIR}\n"
        "Run: python3 scripts/bootstrap_badge_sources.py\n"
        "For local-only art, make sure these files exist on your machine before bootstrapping:\n"
        "  - /Users/harsha/Downloads/pngegg.png\n"

        "No generated fallback silhouettes are allowed in this build."
    )


def trim_alpha(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    return rgba.crop(bbox) if bbox else rgba


def fit_centered(image: Image.Image, canvas_size: int, max_size: int, y_offset: int = 0) -> Image.Image:
    fitted = trim_alpha(image)
    fitted.thumbnail((max_size, max_size), Image.LANCZOS)

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    x = (canvas_size - fitted.width) // 2
    y = (canvas_size - fitted.height) // 2 + y_offset
    canvas.alpha_composite(fitted, (x, y))
    return canvas


def make_flat_coin_base(tier: str) -> Image.Image:
    scale = CANVAS_SCALE
    size = BADGE_SIZE * scale
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    fill = TIER_FILL_COLORS[tier]
    margin = 14 * scale
    bbox = [margin, margin, size - margin, size - margin]

    # Plain filled coin shape only. This intentionally removes any imported coin
    # texture, 2K-style art, embossing, or interior design.
    draw.ellipse(bbox, fill=fill)
    draw.ellipse(bbox, outline=(255, 255, 255, 84), width=3 * scale)
    draw.ellipse(
        [margin + 7 * scale, margin + 7 * scale, size - margin - 7 * scale, size - margin - 7 * scale],
        outline=(0, 0, 0, 46),
        width=2 * scale,
    )
    return image.resize((BADGE_SIZE, BADGE_SIZE), Image.LANCZOS)


def make_flat_diamond_base() -> Image.Image:
    scale = CANVAS_SCALE
    size = BADGE_SIZE * scale
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    fill = TIER_FILL_COLORS["diamond"]
    points = [
        (32 * scale, 32 * scale),
        (160 * scale, 32 * scale),
        (188 * scale, 78 * scale),
        (96 * scale, 184 * scale),
        (4 * scale, 78 * scale),
    ]
    draw.polygon(points, fill=fill)
    draw.line(points + [points[0]], fill=(255, 255, 255, 132), width=3 * scale, joint="curve")
    facet_lines = [
        ((32, 32), (58, 78), (96, 184)),
        ((160, 32), (134, 78), (96, 184)),
        ((96, 32), (96, 184)),
        ((4, 78), (188, 78)),
    ]
    for line in facet_lines:
        scaled_line = [(x * scale, y * scale) for x, y in line]
        draw.line(scaled_line, fill=(255, 255, 255, 72), width=2 * scale, joint="curve")
    return image.resize((BADGE_SIZE, BADGE_SIZE), Image.LANCZOS)


def make_tier_base(tier: str) -> Image.Image:
    if tier == "diamond":
        return make_flat_diamond_base()
    return make_flat_coin_base(tier)


def source_image_to_mask(source_image: Image.Image) -> Image.Image:
    icon = trim_alpha(source_image)
    alpha = icon.getchannel("A")
    min_alpha, max_alpha = alpha.getextrema()

    if min_alpha < 250 or max_alpha < 255:
        mask = alpha
    else:
        # For opaque files on white backgrounds, keep any visibly non-white pixels.
        # This preserves the exact linked icon shape without keeping its original color.
        rgb_pixels = list(icon.convert("RGB").getdata())
        mask_values = []
        for red, green, blue in rgb_pixels:
            distance_from_white = max(abs(255 - red), abs(255 - green), abs(255 - blue))
            brightness = (red + green + blue) / 3
            mask_values.append(255 if distance_from_white > 18 or brightness < 238 else 0)
        mask = Image.new("L", icon.size, 0)
        mask.putdata(mask_values)

    mask = ImageOps.autocontrast(mask)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=0.35))
    return mask


def make_silhouette_from_source(source_path: Path, color: Tuple[int, int, int, int]) -> Image.Image:
    source_image = Image.open(source_path).convert("RGBA")
    mask = source_image_to_mask(source_image)
    silhouette = Image.new("RGBA", mask.size, color)
    silhouette.putalpha(mask)
    return silhouette


def make_icon_layer(badge_id: str, tier: str) -> Image.Image:
    color = ICON_COLORS[tier]
    source_id = BADGE_SOURCE_IDS[badge_id]
    source_path = require_source_image(source_id)
    icon = make_silhouette_from_source(source_path, color)

    icon_size = ICON_SIZES.get(badge_id, 108)
    y_offset = ICON_VERTICAL_OFFSETS.get(badge_id, 0)
    if tier == "diamond":
        icon_size = min(104, int(icon_size * 0.90))
    return fit_centered(icon, BADGE_SIZE, icon_size, y_offset=y_offset)


def make_overlay_layer(badge_id: str) -> Image.Image:
    output = Image.new("RGBA", (BADGE_SIZE, BADGE_SIZE), (0, 0, 0, 0))
    if badge_id not in {"assist_generator", "efficient_passer"}:
        return output

    draw = ImageDraw.Draw(output)
    glyph = "⚡" if badge_id == "assist_generator" else "!"
    font_size = 34 if glyph == "!" else 30
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except BaseException:  # noqa: BLE001
        font = None
    bbox = draw.textbbox((0, 0), glyph, font=font)
    x = (BADGE_SIZE - (bbox[2] - bbox[0])) / 2 - bbox[0]
    y = (BADGE_SIZE - (bbox[3] - bbox[1])) / 2 - bbox[1] - 1
    draw.text((x, y), glyph, fill=(255, 255, 255, 255), font=font)
    return output


def compose_badge(badge_id: str, tier: str) -> Image.Image:
    output = Image.new("RGBA", (BADGE_SIZE, BADGE_SIZE), (0, 0, 0, 0))
    output.alpha_composite(make_tier_base(tier))
    output.alpha_composite(make_icon_layer(badge_id, tier))
    output.alpha_composite(make_overlay_layer(badge_id))
    return output


def png_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def write_embedded_svg(path: Path, image: Image.Image) -> None:
    data_uri = png_data_uri(image)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{BADGE_SIZE}" height="{BADGE_SIZE}" '
        f'viewBox="0 0 {BADGE_SIZE} {BADGE_SIZE}" role="img" aria-hidden="true">\n'
        f'  <image href="{data_uri}" x="0" y="0" width="{BADGE_SIZE}" height="{BADGE_SIZE}" '
        f'preserveAspectRatio="xMidYMid meet"/>\n'
        '</svg>\n'
    )
    path.write_text(svg, encoding="utf-8")


def remove_old_assets() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".svg"}:
            path.unlink()


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    remove_old_assets()
    validate_all_sources()

    written = 0
    for badge_id in BADGE_SOURCE_IDS:
        for tier in TIERS:
            image = compose_badge(badge_id, tier)
            png_path = OUTPUT_DIR / f"{badge_id}_{tier}.png"
            svg_path = OUTPUT_DIR / f"{badge_id}_{tier}.svg"
            image.save(png_path)
            write_embedded_svg(svg_path, image)
            written += 2

    (OUTPUT_DIR / ".gitkeep").write_text("", encoding="utf-8")
    print(f"Built {written} exact-source silhouette badge assets in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
