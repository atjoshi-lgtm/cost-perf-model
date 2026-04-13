from __future__ import annotations

from pathlib import Path
import math

from fds import FootprintDescriptor

_BASE_DIR = Path(__file__).resolve().parent
_FDS2_DIR = _BASE_DIR / "FDS2"

FDS_EXCEPTIONS = {"LGA": "EWR_LGA"}
TEST_METRO = "BOS"
BUCKET_SIZE_MB = 10 * 1024


def _fds_path_for_metro(metro: str) -> Path:
    metro_key = FDS_EXCEPTIONS.get(metro, metro).lower()
    return _FDS2_DIR / f"{metro_key}.txt"


def _read_fd_text_for_metro(metro: str) -> str:
    fd_path = _fds_path_for_metro(metro)
    if not fd_path.exists():
        raise FileNotFoundError(f"No Footprint Descriptor file found for metro {metro}: {fd_path}")
    return fd_path.read_text(encoding="utf-8")


def get_smoothed_fd_for_metro(metro: str, bucket_size_mb: int = BUCKET_SIZE_MB) -> FootprintDescriptor:
    fd_text = _read_fd_text_for_metro(metro)
    return FootprintDescriptor.from_text(fd_text).smooth_by_cache_bucket(bucket_size_mb)


def verify_descriptor(metro: str) -> None:
    descriptor = get_smoothed_fd_for_metro(metro)
    points = list(descriptor._points_sorted_by_cache)

    if not points:
        raise ValueError(f"Descriptor for {metro} has no points after smoothing")

    duplicate_hitrates: dict[float, list[float]] = {}
    previous_cache = -math.inf
    previous_hitrate = -math.inf
    seen_hitrates: dict[float, float] = {}

    for point in points:
        cache_space = float(point.cache_space)
        hitrate = float(point.hitrate)

        if cache_space <= previous_cache:
            raise AssertionError(
                f"Cache space is not strictly increasing for {metro}: {cache_space} after {previous_cache}"
            )

        if hitrate < previous_hitrate:
            raise AssertionError(
                f"Hitrate decreased for {metro}: {hitrate} after {previous_hitrate} at disk {cache_space}"
            )

        if hitrate in seen_hitrates:
            duplicate_hitrates.setdefault(hitrate, [seen_hitrates[hitrate]]).append(cache_space)
        else:
            seen_hitrates[hitrate] = cache_space

        previous_cache = cache_space
        previous_hitrate = hitrate

    print(f"Verified smoothed descriptor for {metro}")
    print(f"Points checked: {len(points)}")
    print(f"Min disk: {points[0].cache_space:.2f} MB")
    print(f"Max disk: {points[-1].cache_space:.2f} MB")
    print(f"Min hitrate: {points[0].hitrate:.4f}%")
    print(f"Max hitrate: {points[-1].hitrate:.4f}%")

    if duplicate_hitrates:
        print("Found duplicate hitrates across different disk points:")
        for hitrate, cache_spaces in sorted(duplicate_hitrates.items()):
            formatted_spaces = ", ".join(f"{space:.2f}" for space in cache_spaces)
            print(f"  hitrate={hitrate:.4f}% at disk_mb=[{formatted_spaces}]")
    else:
        print("All hitrates are unique across disk points.")

if __name__ == "__main__":
    all = ["ATL", "BOS", "DEN", "DFW", "IAD", "LAX", "MIA", "ORD", "PHL", "PIT", "SEA", "SJC"]

    for metro in all:
        verify_descriptor(metro)
