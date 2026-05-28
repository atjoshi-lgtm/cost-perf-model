from __future__ import annotations

from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import DefaultDict

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "gradient_descent_metro_log.txt"
OUTPUT_DIR = BASE_DIR / "OtherBigFoot_APAC"
METRICS = [
    "disk_mb",
    "hitrate",
    "cost",
    "perf_penalty",
    "p50",
    "p95",
    "cost_gradient",
    "perf_gradient",
    "overall_gradient",
]
METROS_PER_PLOT = 5
MARKERS = [
    "o",
    "s",
    "^",
    "D",
    "v",
    "P",
    "X",
    "<",
    ">",
    "*",
    "h",
    "H",
    "p",
    "8",
    "d",
]


SeriesMap = dict[str, dict[str, list[float]]]


def parse_metro_areas(file_path: Path) -> tuple[dict[str, dict], dict[str, str], dict[str, str]]:
    airport_info: dict[str, dict] = {}
    metro_to_airport: dict[str, str] = {}
    airport_to_metro: dict[str, str] = {}

    with file_path.open("r", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) != 8:
                continue

            parts = [part.strip().strip('"') for part in parts]
            identifier, metro_area, latitude, longitude, airport_code, country, state, max_distance = parts

            if country != "US":
                continue

            airport_info[airport_code] = {
                "id": identifier,
                "metro_area": metro_area,
                "latitude": float(latitude),
                "longitude": float(longitude),
                "country": country,
                "state": state,
                "max_distance_from_center": float(max_distance),
            }
            metro_to_airport[metro_area] = airport_code
            airport_to_metro[airport_code] = metro_area

    return airport_info, metro_to_airport, airport_to_metro


def parse_log_file(log_path: Path) -> SeriesMap:
    metro_series: DefaultDict[str, DefaultDict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    current_iteration: int | None = None

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("iteration="):
                current_iteration = int(line.split("=", 1)[1])
                continue

            if current_iteration is None:
                raise ValueError("Found metro data before any iteration header")

            parts = [part.strip() for part in line.split(",")]
            metro = parts[0]
            values: dict[str, float] = {}
            for part in parts[1:]:
                key, value = part.split("=", 1)
                values[key.strip()] = float(value.strip())

            metro_series[metro]["iteration"].append(float(current_iteration))
            for metric in METRICS:
                if metric not in values:
                    raise ValueError(f"Missing metric '{metric}' for metro '{metro}' at iteration {current_iteration}")
                metro_series[metro][metric].append(values[metric])

    return {metro: dict(series) for metro, series in metro_series.items()}


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def sanitize_filename(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def marker_for_metro(metro: str) -> str:
    return MARKERS[sum(ord(char) for char in metro) % len(MARKERS)]


def display_name_for_metro(metro: str, airport_to_metro: dict[str, str]) -> str:
    return airport_to_metro.get(metro, metro)


def plot_metric_group(
    metro_series: SeriesMap,
    airport_to_metro: dict[str, str],
    metric: str,
    metros: list[str],
    group_index: int,
    group_count: int,
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(14, 8))

    for metro in metros:
        iterations = metro_series[metro]["iteration"]
        values = metro_series[metro][metric]
        ax.plot(
            iterations,
            values,
            marker=marker_for_metro(metro),
            linewidth=1.8,
            markersize=4.5,
            label=display_name_for_metro(metro, airport_to_metro),
        )

    metro_names = ", ".join(display_name_for_metro(metro, airport_to_metro) for metro in metros)
    ax.set_title(f"{metric} over iterations — metros {group_index + 1}/{group_count}\n{metro_names}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)

    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))

    filename = f"{sanitize_filename(metric)}_group_{group_index + 1:02d}.png"
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    _, _, airport_to_metro = parse_metro_areas(BASE_DIR / "PERF" / "metro_areas.csv")
    metro_series = parse_log_file(LOG_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metros = sorted(metro_series)
    metro_groups = chunked(metros, METROS_PER_PLOT)
    total_expected = len(METRICS) * len(metro_groups)
    created_files: list[Path] = []

    for metric in METRICS:
        for group_index, metro_group in enumerate(metro_groups):
            created_files.append(
                plot_metric_group(
                    metro_series=metro_series,
                    airport_to_metro=airport_to_metro,
                    metric=metric,
                    metros=metro_group,
                    group_index=group_index,
                    group_count=len(metro_groups),
                    output_dir=OUTPUT_DIR,
                )
            )

    print(
        f"Parsed {len(metros)} metros across {len(metro_groups)} groups and created "
        f"{len(created_files)} plot files in {OUTPUT_DIR} (expected {total_expected})."
    )


if __name__ == "__main__":
    main()
