from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "gradient_descent_metro_log.txt"
OUTPUT_DIR = BASE_DIR / "US"
METRICS = ["disk_mb", "hitrate", "cost", "perf_penalty", "p50", "p95"]


def parse_log_file(log_path: Path) -> dict[str, dict[str, list[float]]]:
    metro_series: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    current_iteration: int | None = None

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("iteration="):
                current_iteration = int(line.split("=", 1)[1])
                continue

            parts = [part.strip() for part in line.split(",")]
            metro = parts[0]
            values: dict[str, float] = {}
            for part in parts[1:]:
                key, value = part.split("=", 1)
                values[key.strip()] = float(value.strip())

            if current_iteration is None:
                raise ValueError("Found metro data before any iteration header")

            metro_series[metro]["iteration"].append(float(current_iteration))
            for metric in METRICS:
                metro_series[metro][metric].append(values.get(metric, 0.0))

    return metro_series


def plot_metric_for_metro(metro: str, iterations: list[float], values: list[float], metric: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(iterations, values, marker="o")
    plt.title(f"{metric} over iterations for {metro}")
    plt.xlabel("Iteration")
    plt.ylabel(metric)
    plt.grid(True)
    plt.tight_layout()

    output_path = output_dir / f"{metro}_{metric}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def main() -> None:
    metro_series = parse_log_file(LOG_PATH)
    for metro, series in metro_series.items():
        iterations = series["iteration"]
        for metric in METRICS:
            plot_metric_for_metro(metro, iterations, series[metric], metric, OUTPUT_DIR)


if __name__ == "__main__":
    main()
