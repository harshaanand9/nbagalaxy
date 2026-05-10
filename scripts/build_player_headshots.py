"""
Builds backend/data/player_headshots.csv and optionally downloads NBA headshots.

Run from the project root:

    python3 scripts/build_player_headshots.py

This script reads player names from the locked assignment CSV, optional similar-player
CSVs, and optional dataset CSVs. It maps names to NBA person IDs using nba_api's static
player index, writes backend/data/player_headshots.csv, and downloads images into
frontend/public/headshots/.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: requests. Install with: python3 -m pip install requests") from exc

try:
    from nba_api.stats.static import players as nba_static_players
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: nba_api. Install with: python3 -m pip install nba_api") from exc


NBA_HEADSHOT_URL_TEMPLATE = "https://cdn.nba.com/headshots/nba/latest/1040x760/{person_id}.png"
FALLBACK_WEB_PATH = "/headshots/fallback.svg"

# Source-name -> explicit NBA person ID for players missing from nba_api's static index.
# Keys are normalized by compact_player_key().
MANUAL_PERSON_ID_OVERRIDES: dict[str, tuple[str, str]] = {
    "jahmaimashack": ("1642942", "Jahmai Mashack"),
    "jamaimashack": ("1642942", "Jahmai Mashack"),

    # Names that the NBA static player index may not resolve consistently across
    # nba_api versions, but whose NBA.com person IDs are stable for headshots.
    "enesfreedom": ("202683", "Enes Kanter"),
    "eneskanter": ("202683", "Enes Kanter"),
    "kevinknox": ("1628995", "Kevin Knox II"),
    "kevinknoxii": ("1628995", "Kevin Knox II"),
    "marcusmorris": ("202694", "Marcus Morris Sr."),
    "marcusmorrissr": ("202694", "Marcus Morris Sr."),
    "ronaldhollandii": ("1641842", "Ron Holland II"),
    "ronholland": ("1641842", "Ron Holland II"),
    "ronhollandii": ("1641842", "Ron Holland II"),
}

# Source-name -> NBA API full_name. Keys are normalized by compact_player_key().
# Keep this list conservative. Bad automatic headshot matches are worse than fallbacks.
MANUAL_NAME_ALIASES: dict[str, str] = {
    # Existing project aliases / common source differences.
    "jimmybutleriii": "Jimmy Butler III",
    "kenyonmartinjr": "KJ Martin",
    "kenyonmartinjrjr": "KJ Martin",
    "kjmartinjr": "KJ Martin",
    "nicolasclaxton": "Nic Claxton",
    "enesfreedom": "Enes Kanter",
    "wesmatthews": "Wesley Matthews",
    "juanhernangomez": "Juancho Hernangomez",
    "louiswilliams": "Lou Williams",
    "robertwilliams": "Robert Williams III",
    "robertwilliamsiii": "Robert Williams III",
    "marvinbagleyiii": "Marvin Bagley III",
    "marvinbagley": "Marvin Bagley III",
    "frankmason": "Frank Mason III",
    "reggiebullock": "Reggie Bullock Jr.",
    "wandellmoorejr": "Wendell Moore Jr.",
    "wendellmoore": "Wendell Moore Jr.",
    "kevindurantii": "Kevin Durant",
    "kellyoubrejr": "Kelly Oubre Jr.",
    "ottoporterjr": "Otto Porter Jr.",
    "garytrentjr": "Gary Trent Jr.",
    "timhardawayjr": "Tim Hardaway Jr.",
    "larrynancejr": "Larry Nance Jr.",
    "dennissmithjr": "Dennis Smith Jr.",
    "derrickjonesjr": "Derrick Jones Jr.",
    "derecklivelyii": "Dereck Lively II",
    "lonniewalkeriv": "Lonnie Walker IV",
    "nickeilalexanderwalker": "Nickeil Alexander-Walker",
    "michaelporterjr": "Michael Porter Jr.",
    "wendellcarterjr": "Wendell Carter Jr.",
    "patrickbaldwinjr": "Patrick Baldwin Jr.",
    "jarenjacksonjr": "Jaren Jackson Jr.",
    "jabarismithjr": "Jabari Smith Jr.",
    "gregbrowniii": "Greg Brown III",
    "vincewilliamsjr": "Vince Williams Jr.",
    "scottypippenjr": "Scotty Pippen Jr.",
    "ronaldhollandii": "Ron Holland II",
    "ronholland": "Ron Holland II",
    "ronhollandii": "Ron Holland II",
    "kevinknox": "Kevin Knox II",
    "marcusmorris": "Marcus Morris Sr.",
    "vjedgcombe": "VJ Edgecombe",
    "ajlawson": "A.J. Lawson",
    "ajjohnson": "AJ Johnson",
    "pjdozier": "PJ Dozier",
    "pjwashington": "P.J. Washington",
    "tjmcconnell": "T.J. McConnell",
    "tjleaf": "T.J. Leaf",
    "jjsullinger": "J.J. Sullinger",
    "jjredick": "JJ Redick",
    "jrsmith": "JR Smith",
    "ogananoby": "OG Anunoby",
    "ogannunoby": "OG Anunoby",
    "kevinporterjr": "Kevin Porter Jr.",
}

# A few names can map to the wrong historical player if only fuzzy matching is used.
# These names should stay unresolved unless explicitly aliased above or exactly found.
DO_NOT_FUZZY_MATCH_KEYS: set[str] = {
    "player",
    "team",
}

PLAYER_NAME_COLUMNS = [
    "Player Name",
    "player_name",
    "related_player_name",
    "source_player_name",
    "name",
]


@dataclass(frozen=True)
class HeadshotRow:
    player_name: str
    normalized_player_name: str
    nba_person_id: str
    nba_full_name: str
    headshot_url: str
    local_headshot_path: str
    match_status: str
    match_method: str
    notes: str


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(character for character in normalized if not unicodedata.combining(character))


def canonical_display_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def compact_player_key(value: object) -> str:
    """Aggressively normalize a player name for matching across CSV/NBA variants."""
    text = canonical_display_name(value)
    text = strip_accents(text)
    text = text.replace("’", "'").replace("`", "'")
    text = text.lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    text = text.replace("'", "")
    text = re.sub(r"\b(junior)\b", "jr", text)
    text = re.sub(r"\b(senior)\b", "sr", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"[^a-z0-9]+", "", text)


def load_nba_index() -> tuple[dict[str, dict], dict[str, str]]:
    nba_players = nba_static_players.get_players()
    nba_by_key: dict[str, dict] = {}
    duplicate_keys: dict[str, str] = {}

    for nba_player in nba_players:
        full_name = canonical_display_name(nba_player.get("full_name", ""))
        key = compact_player_key(full_name)
        if not key:
            continue
        if key in nba_by_key:
            duplicate_keys[key] = full_name
            # Prefer active player when exact normalized names collide.
            current = nba_by_key[key]
            if bool(nba_player.get("is_active")) and not bool(current.get("is_active")):
                nba_by_key[key] = nba_player
        else:
            nba_by_key[key] = nba_player

    return nba_by_key, duplicate_keys


def source_csv_paths(project_root: Path) -> list[Path]:
    return [
        project_root / "backend" / "data" / "euclidean_kmeans_locked_assignments.csv",
        project_root / "backend" / "data" / "similar_players_precomputed_production.csv",
        project_root / "backend" / "data" / "similar_players.csv",
        project_root / "data" / "similar_players_precomputed_production.csv",
        project_root / "data" / "similar_players.csv",
        project_root / "similar_players_precomputed_production.csv",
        project_root / "similar_players.csv",
    ]


def collect_player_names(project_root: Path, extra_csv_paths: Iterable[Path]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    for csv_path in [*source_csv_paths(project_root), *extra_csv_paths]:
        if not csv_path.exists():
            continue
        try:
            dataframe = pd.read_csv(csv_path, low_memory=False)
        except Exception as exc:
            print(f"WARNING: could not read {csv_path}: {exc}", file=sys.stderr)
            continue

        for column_name in PLAYER_NAME_COLUMNS:
            if column_name not in dataframe.columns:
                continue
            for raw_name in dataframe[column_name].dropna().tolist():
                display_name = canonical_display_name(raw_name)
                key = compact_player_key(display_name)
                if not display_name or not key or key in seen:
                    continue
                seen.add(key)
                names.append(display_name)

    return sorted(names, key=lambda value: compact_player_key(value))


def resolve_player(player_name: str, nba_by_key: dict[str, dict]) -> tuple[str, str, str, str]:
    """Return (person_id, nba_full_name, match_status, match_method)."""
    source_key = compact_player_key(player_name)
    if not source_key:
        return "", "", "unresolved", "empty_name"

    person_id_override = MANUAL_PERSON_ID_OVERRIDES.get(source_key)
    if person_id_override:
        person_id, nba_full_name = person_id_override
        return person_id, nba_full_name, "matched", "manual_person_id"

    alias_name = MANUAL_NAME_ALIASES.get(source_key)
    if alias_name:
        alias_key = compact_player_key(alias_name)
        nba_player = nba_by_key.get(alias_key)
        if nba_player:
            return str(nba_player["id"]), nba_player["full_name"], "matched", "manual_alias"
        return "", alias_name, "unresolved", "manual_alias_missing_from_nba_api"

    nba_player = nba_by_key.get(source_key)
    if nba_player:
        return str(nba_player["id"]), nba_player["full_name"], "matched", "exact_normalized"

    if source_key in DO_NOT_FUZZY_MATCH_KEYS:
        return "", "", "unresolved", "blocked_fuzzy_match"

    # Ultra-conservative fallback: only accept a match if removing a suffix makes it exact.
    suffix_stripped = re.sub(r"(jr|sr|ii|iii|iv|v)$", "", source_key)
    if suffix_stripped != source_key and suffix_stripped in nba_by_key:
        nba_player = nba_by_key[suffix_stripped]
        return str(nba_player["id"]), nba_player["full_name"], "matched", "suffix_stripped_exact"

    return "", "", "unresolved", "no_match"


def write_fallback_svg(headshot_dir: Path) -> None:
    fallback_path = headshot_dir / "fallback.svg"
    if fallback_path.exists():
        return
    fallback_path.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="760" viewBox="0 0 1040 760">
  <rect width="1040" height="760" fill="#061113"/>
  <rect x="22" y="22" width="996" height="716" rx="42" fill="#081b1f" stroke="#00e5ff" stroke-width="10" opacity="0.85"/>
  <circle cx="520" cy="295" r="128" fill="#123139" stroke="#9beeff" stroke-width="8" opacity="0.9"/>
  <path d="M275 655c35-126 129-196 245-196s210 70 245 196" fill="#123139" stroke="#9beeff" stroke-width="8" opacity="0.9"/>
  <text x="520" y="705" text-anchor="middle" font-family="monospace" font-size="44" fill="#00e5ff">NO_HEADSHOT</text>
</svg>
""",
        encoding="utf-8",
    )


def download_headshot(person_id: str, output_path: Path, overwrite: bool, sleep_seconds: float) -> bool:
    if output_path.exists() and not overwrite:
        return True

    url = NBA_HEADSHOT_URL_TEMPLATE.format(person_id=person_id)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code != 200:
            return False
        content_type = response.headers.get("content-type", "").lower()
        if "image" not in content_type and not response.content.startswith(b"\x89PNG"):
            return False
        if len(response.content) < 1000:
            return False
        output_path.write_bytes(response.content)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        return True
    except requests.RequestException:
        return False


def build_headshot_rows(player_names: list[str], nba_by_key: dict[str, dict]) -> list[HeadshotRow]:
    rows: list[HeadshotRow] = []
    for player_name in player_names:
        person_id, nba_full_name, match_status, match_method = resolve_player(player_name, nba_by_key)
        if person_id:
            headshot_url = NBA_HEADSHOT_URL_TEMPLATE.format(person_id=person_id)
            local_path = f"/headshots/{person_id}.png"
            notes = ""
        else:
            headshot_url = ""
            local_path = FALLBACK_WEB_PATH
            notes = "needs_manual_id"

        rows.append(
            HeadshotRow(
                player_name=player_name,
                normalized_player_name=compact_player_key(player_name),
                nba_person_id=person_id,
                nba_full_name=nba_full_name,
                headshot_url=headshot_url,
                local_headshot_path=local_path,
                match_status=match_status,
                match_method=match_method,
                notes=notes,
            )
        )
    return rows


def write_mapping_csv(rows: list[HeadshotRow], output_csv_path: Path) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(HeadshotRow.__dataclass_fields__.keys())
    with output_csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build player_headshots.csv and cache NBA headshot images.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--headshot-dir", type=Path, default=None)
    parser.add_argument("--extra-csv", type=Path, action="append", default=[])
    parser.add_argument("--no-download", action="store_true", help="Only write the CSV mapping; do not download images.")
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_csv_path = args.output_csv or project_root / "backend" / "data" / "player_headshots.csv"
    headshot_dir = args.headshot_dir or project_root / "frontend" / "public" / "headshots"
    headshot_dir.mkdir(parents=True, exist_ok=True)
    write_fallback_svg(headshot_dir)

    nba_by_key, duplicate_keys = load_nba_index()
    player_names = collect_player_names(project_root, args.extra_csv)
    rows = build_headshot_rows(player_names, nba_by_key)

    if not args.no_download:
        downloaded = 0
        failed_downloads = 0
        for row in rows:
            if not row.nba_person_id:
                continue
            image_path = headshot_dir / f"{row.nba_person_id}.png"
            if download_headshot(row.nba_person_id, image_path, args.overwrite_images, args.sleep_seconds):
                downloaded += 1
            else:
                failed_downloads += 1
        print(f"Downloaded/found images: {downloaded}")
        print(f"Failed image downloads: {failed_downloads}")

    write_mapping_csv(rows, output_csv_path)

    matched_count = sum(row.match_status == "matched" for row in rows)
    unresolved_rows = [row for row in rows if row.match_status != "matched"]

    print(f"Wrote: {output_csv_path}")
    print(f"Players found in source CSVs: {len(rows)}")
    print(f"Matched to NBA IDs: {matched_count}")
    print(f"Unresolved: {len(unresolved_rows)}")

    if unresolved_rows:
        print("\nUnresolved names to manually check:")
        for row in unresolved_rows[:200]:
            print(f"  - {row.player_name} ({row.match_method})")
        if len(unresolved_rows) > 200:
            print(f"  ... {len(unresolved_rows) - 200} more")

    if duplicate_keys:
        print(f"\nNBA API duplicate normalized-name keys detected: {len(duplicate_keys)}")


if __name__ == "__main__":
    main()
