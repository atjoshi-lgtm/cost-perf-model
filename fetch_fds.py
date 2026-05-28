from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Iterable

from FDSAPI import API_URL, CERT, FootprintDescriptors


def load_target_metros() -> list[str]:
    fds_metros_file = Path(__file__).resolve().parent / "PERF" / "fds_metros.json"
    if not fds_metros_file.exists():
        return ["ATL", "BOS", "DEN", "DFW", "IAD", "LAX", "MIA", "ORD", "PHL", "PIT", "SEA", "SJC"]
    with open(fds_metros_file, "r") as f:
        data = json.load(f)
    return [item["metro"] for item in data if "_" not in item["metro"]]


TARGET_METROS = load_target_metros()
DEFAULT_QUARTER = "2026Q1"
DEFAULT_BUCKET = "AkamaiHD"
OUTPUT_DIR = Path(__file__).resolve().parent / f"FD_{DEFAULT_BUCKET}"
KNEE_OUTPUT_DIR = Path(__file__).resolve().parent / f"KneeData_{DEFAULT_BUCKET}"


def fetch_stdspace_text(client: FootprintDescriptors, metro: str, quarter: str, bucket: str) -> str | None:
    result = client.get_stdspace_for_bucket(
        metro=metro,
        quarter=quarter,
        bucket=bucket,
    )
    if not result:
        return None

    stdspace = result.get("stdspace")
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
    bucket: str = DEFAULT_BUCKET,
) -> tuple[list[Path], list[str]]:
    client = FootprintDescriptors(
        api_url=API_URL,
        certificate=str(Path(CERT).expanduser()),
    )

    saved_paths: list[Path] = []
    missing_metros: list[str] = []

    for metro in metros:
        print(f"Fetching stdspace for {metro} (quarter={quarter}, bucket={bucket})")
        stdspace_text = fetch_stdspace_text(client, metro, quarter, bucket)
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
    bucket: str,
) -> dict | None:
    """Fetch the knee data for ``metro`` by querying the bucket endpoint."""
    return client.get_knee_for_bucket(metro=metro, quarter=quarter, bucket=bucket)


def save_knee(output_dir: Path, metro: str, knee_data: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{metro.lower()}.json"
    output_path.write_text(json.dumps(knee_data, indent=2), encoding="utf-8")
    return output_path


def fetch_and_save_all_knees(
    metros: Iterable[str],
    quarter: str = DEFAULT_QUARTER,
    bucket: str = DEFAULT_BUCKET,
) -> tuple[list[Path], list[str]]:
    """Fetch knee data for every metro and save each result as a JSON file
    in the KneeData directory."""

    client = FootprintDescriptors(
        api_url=API_URL,
        certificate=str(Path(CERT).expanduser()),
    )

    saved_paths: list[Path] = []
    missing_metros: list[str] = []

    for metro in metros:
        print(f"Fetching knee for {metro} (quarter={quarter}, bucket={bucket})")
        knee_data = fetch_knee(client, metro, quarter, bucket)
        if knee_data is None:
            missing_metros.append(metro)
            print(f"  No knee data returned for {metro}")
            continue

        output_path = save_knee(KNEE_OUTPUT_DIR, metro, knee_data)
        saved_paths.append(output_path)
        print(f"  Saved {metro} knee data to {output_path}")

    return saved_paths, missing_metros


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str, default="w80")
    args = parser.parse_args()

    global OUTPUT_DIR
    global KNEE_OUTPUT_DIR
    global DEFAULT_BUCKET
    DEFAULT_BUCKET = args.bucket
    OUTPUT_DIR = Path(__file__).resolve().parent / f"FDS_{args.bucket}"
    KNEE_OUTPUT_DIR = Path(__file__).resolve().parent / f"KneeData_{args.bucket}"

    saved_paths, missing_metros = fetch_and_save_all_metros(TARGET_METROS, bucket=DEFAULT_BUCKET)

    print(f"\nSaved {len(saved_paths)} footprint descriptors to {OUTPUT_DIR}")
    if missing_metros:
        print(f"Missing metros: {missing_metros}")

    print()
    saved_knee_paths, missing_knee_metros = fetch_and_save_all_knees(TARGET_METROS, bucket=DEFAULT_BUCKET)

    print(f"\nSaved {len(saved_knee_paths)} knee files to {KNEE_OUTPUT_DIR}")
    if missing_knee_metros:
        print(f"Missing knee metros: {missing_knee_metros}")


if __name__ == "__main__":
    main()
