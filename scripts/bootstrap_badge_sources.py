#!/usr/bin/env python3
"""Download/copy the exact badge source icon PNGs requested by the project owner.

This script fills frontend/public/badges/source with canonical icon images used by
scripts/build_badge_assets.py. It is strict by default: if a remote image cannot
be downloaded or a local-only file is missing, it exits with an error instead of
allowing generated fallback icon art.

Run from the project root:
    python3 scripts/bootstrap_badge_sources.py
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.request import Request, urlopen

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "frontend" / "public" / "badges" / "source"


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    url: Optional[str] = None
    local_path: Optional[str] = None


# These are the source icon images. The builder will turn these exact images into
# black/white silhouettes; it will not generate replacement/fallback icon shapes.
SOURCE_SPECS: Dict[str, SourceSpec] = {
    "deep_range_bomber": SourceSpec("deep_range_bomber", url="https://upload.wikimedia.org/wikipedia/commons/7/74/Bomb-png-46599.png?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=original"),
    "catch_and_shoot_converter": SourceSpec("catch_and_shoot_converter", url="https://png.pngtree.com/png-vector/20221212/ourmid/pngtree-slingshots-png-image_6520484.png"),
    "contested_3pt_maker": SourceSpec("contested_3pt_maker", url="https://png.pngtree.com/png-clipart/20250218/original/pngtree-comfortable-blue-sleep-mask-isolated-on-a-transparent-background-png-image_20458929.png"),
    "pull_up_3pt_machine": SourceSpec("pull_up_3pt_machine", url="https://images.vexels.com/media/users/3/141727/isolated/preview/8b594b42b746c398aaa1f8b0c04ff83f-shot-ball-player.png"),
    "volume_3pt_shooter": SourceSpec("volume_3pt_shooter", url="https://www.shareicon.net/download/2015/10/05/651365_hand_512x512.png"),
    "volume_mid_range_shooter": SourceSpec("volume_mid_range_shooter", local_path="/Users/harsha/Downloads/pngegg.png"),
    "mid_range_assassin": SourceSpec("mid_range_assassin", url="https://static.vecteezy.com/system/resources/thumbnails/011/887/515/small/black-knife-isolated-png.png"),
    "volume_slasher": SourceSpec("volume_slasher", url="https://upload.wikimedia.org/wikipedia/commons/4/49/Claw_Marks.png"),
    "efficient_driver": SourceSpec("efficient_driver", url="https://static.vecteezy.com/system/resources/previews/009/398/196/non_2x/steering-wheel-clipart-design-illustration-free-png.png"),
    "free_throw_generator": SourceSpec("free_throw_generator", url="https://png.pngtree.com/png-vector/20231018/ourmid/pngtree-basketball-referee-stop-clock-for-foul-hand-signal-retro-black-png-image_10277659.png"),
    "drive_and_kicker": SourceSpec("drive_and_kicker", url="https://images.vexels.com/media/users/3/141900/isolated/lists/bf643ea0ef37a22a5e12eaf8eec3f714-karate-high-kick-training.png"),
    "inside_the_arc_scorer": SourceSpec("inside_the_arc_scorer", url="https://pngimg.com/uploads/number2/Number%202%20PNG%20images%20free%20download_PNG14925.png"),
    "dunker": SourceSpec("dunker", url="https://cdn.creazilla.com/silhouettes/2915/basketball-dunk-silhouette-000000-md.png"),
    "active_hands": SourceSpec("active_hands", url="https://static.thenounproject.com/png/828812-200.png"),
    "defensive_lock_down": SourceSpec("defensive_lock_down", url="https://cdn-icons-png.flaticon.com/512/115/115681.png"),
    "assist_generator": SourceSpec("assist_generator", url="https://cdn-icons-png.flaticon.com/512/11498/11498928.png"),
    "efficient_passer": SourceSpec("efficient_passer", url="https://static.vecteezy.com/system/resources/previews/021/282/254/non_2x/be-careful-sign-symbol-exclamation-mark-in-yellow-free-png.png"),
    "walking_bucket": SourceSpec("walking_bucket", url="https://png.pngtree.com/png-clipart/20210309/original/pngtree-bucket-with-scale-clip-art-png-image_5892554.png"),
    "three_pt_sniper": SourceSpec("three_pt_sniper", url="https://static.thenounproject.com/png/3968826-200.png"),
}


def canonical_path(source_id: str) -> Path:
    return SOURCE_DIR / f"{source_id}.png"


def existing_source_path(source_id: str) -> Optional[Path]:
    for extension in (".png", ".jpg", ".jpeg", ".webp"):
        path = SOURCE_DIR / f"{source_id}{extension}"
        if path.exists():
            return path
    return None


def save_as_png(raw_bytes: bytes, destination: Path) -> None:
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def copy_local_image(source_path: Path, destination: Path) -> None:
    image = Image.open(source_path).convert("RGBA")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def download_image(url: str, destination: Path, attempts: int = 3) -> None:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Referer": "https://www.google.com/",
    }
    last_error: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=35) as response:
                raw_bytes = response.read()
            save_as_png(raw_bytes, destination)
            return
        except BaseException as exc:  # noqa: BLE001
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(1 + attempt)
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def materialize_source(spec: SourceSpec, refresh: bool) -> None:
    destination = canonical_path(spec.source_id)
    if not refresh and existing_source_path(spec.source_id) is not None:
        print(f"exists       {spec.source_id}")
        return

    if spec.local_path:
        source_path = Path(spec.local_path).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing local badge source for {spec.source_id}: {source_path}\n"
                f"Copy the intended image there, or manually place it at: {destination}"
            )
        copy_local_image(source_path, destination)
        print(f"copied       {spec.source_id} <- {source_path}")
        return

    if spec.url:
        download_image(spec.url, destination)
        print(f"downloaded   {spec.source_id} <- {spec.url}")
        return

    raise ValueError(f"No URL or local_path configured for {spec.source_id}")


def validate_sources() -> None:
    missing = [source_id for source_id in SOURCE_SPECS if existing_source_path(source_id) is None]
    if not missing:
        return
    formatted = "\n".join(f"  - {source_id}" for source_id in missing)
    raise FileNotFoundError(
        "Badge source bootstrap did not produce all required exact source icon files:\n"
        f"{formatted}\n\n"
        f"Expected files under: {SOURCE_DIR}\n"
        "No fallback silhouettes are allowed in this build."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/copy exact badge source icon art.")
    parser.add_argument("--refresh", action="store_true", help="Re-download/re-copy sources even when local copies already exist.")
    args = parser.parse_args()

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for source_id, spec in SOURCE_SPECS.items():
        try:
            materialize_source(spec, refresh=args.refresh)
        except BaseException as exc:  # noqa: BLE001
            failures.append((source_id, exc))
            print(f"FAILED {source_id}: {exc}", file=sys.stderr)

    if failures:
        formatted = "\n".join(f"  - {source_id}: {exc}" for source_id, exc in failures)
        raise SystemExit(
            "\nCould not bootstrap exact badge source icon art. No fallback silhouettes were generated.\n"
            f"{formatted}\n"
        )

    validate_sources()
    print(f"All exact badge source icon art is present in: {SOURCE_DIR}")


if __name__ == "__main__":
    main()
