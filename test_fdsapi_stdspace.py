from __future__ import annotations

import argparse
from pathlib import Path

from FDSAPI import API_URL, CERT, FootprintDescriptors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the get_stdspace_for_maprule API call from FDSAPI.py",
    )
    parser.add_argument("--metro", required=True, help="Metro name, e.g. Dallas")
    parser.add_argument("--quarter", required=True, help="Quarter, e.g. 1Q26")
    parser.add_argument(
        "--search-term",
        required=True,
        help="Search term passed to the maprule search endpoint",
    )
    parser.add_argument(
        "--api-url",
        default=API_URL,
        help=f"FDS API base URL (default: {API_URL})",
    )
    parser.add_argument(
        "--cert",
        default=CERT,
        help=f"Client certificate path (default: {CERT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cert_path = str(Path(args.cert).expanduser())

    client = FootprintDescriptors(api_url=args.api_url, certificate=cert_path)
    result = client.get_stdspace_for_maprule(
        metro=args.metro,
        quarter=args.quarter,
        search_term=args.search_term,
    )

    # Convert result to json and return "stdspace" value
    return result[0]["stdspace"]

if __name__ == "__main__":
    main()
