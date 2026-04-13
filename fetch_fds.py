from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from FDSAPI import API_URL, CERT, FootprintDescriptors

TARGET_METROS = ["ATL", "BOS", "DEN", "DFW", "IAD", "LAX", "MIA", "ORD", "PHL", "PIT", "SEA", "SJC"]
DEFAULT_QUARTER = "2026Q1"
DEFAULT_SEARCH_TERM = "w80"
OUTPUT_DIR = Path(__file__).resolve().parent / "FDS2"
KNEE_OUTPUT_DIR = Path(__file__).resolve().parent / "KneeData"


def fetch_stdspace_text(client: FootprintDescriptors, metro: str, quarter: str, search_term: str) -> str | None:
    result = client.get_stdspace_for_maprule(
        metro=metro,
        quarter=quarter,
        search_term=search_term,
    )
    if not result:
        return None

    stdspace = result[0].get("stdspace")
    if not stdspace:
        return None

    return stdspace


def save_stdspace(output_dir: Path, metro: str, stdspace_text: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{metro.lower()}.txt"
    output_path.write_text(stdspace_text, encoding="utf-8")
    return output_path


def fetch_and_save_all_metros(
    metros: Iterable[str],
    quarter: str = DEFAULT_QUARTER,
    search_term: str = DEFAULT_SEARCH_TERM,
) -> tuple[list[Path], list[str]]:
    client = FootprintDescriptors(
        api_url=API_URL,
        certificate=str(Path(CERT).expanduser()),
    )

    saved_paths: list[Path] = []
    missing_metros: list[str] = []

    for metro in metros:
        print(f"Fetching stdspace for {metro} (quarter={quarter}, search={search_term})")
        stdspace_text = fetch_stdspace_text(client, metro, quarter, search_term)
        if stdspace_text is None:
            missing_metros.append(metro)
            print(f"No stdspace returned for {metro}")
            continue

        output_path = save_stdspace(OUTPUT_DIR, metro, stdspace_text)
        saved_paths.append(output_path)
        print(f"Saved {metro} footprint descriptor to {output_path}")

    return saved_paths, missing_metros


def fetch_knee(
    client: FootprintDescriptors,
    metro: str,
    quarter: str,
    search_term: str,
) -> dict | None:
    """Fetch the knee data for ``metro`` by first resolving the network and
    maprule via the stdspace endpoint, then querying the knee endpoint."""

    result = client.get_stdspace_for_maprule(
        metro=metro,
        quarter=quarter,
        search_term=search_term,
    )
    if not result:
        return None

    network = result[0].get("network")
    maprule = result[0].get("maprule")
    if not network or not maprule:
        print(f"  Missing network/maprule in stdspace response for {metro}")
        return None

    knee = client.get_knee_for_maprule(
        metro=metro,
        quarter=quarter,
        network=network,
        maprule=maprule,
    )
    return knee


def save_knee(output_dir: Path, metro: str, knee_data: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{metro.lower()}.json"
    output_path.write_text(json.dumps(knee_data, indent=2), encoding="utf-8")
    return output_path


def fetch_and_save_all_knees(
    metros: Iterable[str],
    quarter: str = DEFAULT_QUARTER,
    search_term: str = DEFAULT_SEARCH_TERM,
) -> tuple[list[Path], list[str]]:
    """Fetch knee data for every metro and save each result as a JSON file
    in ``KneeData/``."""

    client = FootprintDescriptors(
        api_url=API_URL,
        certificate=str(Path(CERT).expanduser()),
    )

    saved_paths: list[Path] = []
    missing_metros: list[str] = []

    for metro in metros:
        print(f"Fetching knee for {metro} (quarter={quarter}, search={search_term})")
        knee_data = fetch_knee(client, metro, quarter, search_term)
        if knee_data is None:
            missing_metros.append(metro)
            print(f"  No knee data returned for {metro}")
            continue

        output_path = save_knee(KNEE_OUTPUT_DIR, metro, knee_data)
        saved_paths.append(output_path)
        print(f"  Saved {metro} knee data to {output_path}")

    return saved_paths, missing_metros


def main() -> None:
    #saved_paths, missing_metros = fetch_and_save_all_metros(TARGET_METROS)

    #print(f"\nSaved {len(saved_paths)} footprint descriptors to {OUTPUT_DIR}")
    #if missing_metros:
    #    print(f"Missing metros: {missing_metros}")

    #print()
    saved_knee_paths, missing_knee_metros = fetch_and_save_all_knees(TARGET_METROS)

    print(f"\nSaved {len(saved_knee_paths)} knee files to {KNEE_OUTPUT_DIR}")
    if missing_knee_metros:
        print(f"Missing knee metros: {missing_knee_metros}")


if __name__ == "__main__":
    main()
