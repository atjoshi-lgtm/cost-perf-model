from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "gradient_descent_summary_log.txt"
OUTPUT_PATH = BASE_DIR / "US_run.png"


def parse_summary_log(log_path: Path) -> dict[str, list[float]]:
    series = {
        "iteration": [],
        "total_cost": [],
        "total_perf_penalty": [],
        "objective": [],
    }

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            values: dict[str, float] = {}
            for part in line.split(","):
                key, value = part.strip().split("=", 1)
                if key == "iteration":
                    values[key] = int(value)
                else:
                    values[key] = float(value)

            series["iteration"].append(values["iteration"])
            series["total_cost"].append(values["total_cost"])
            series["total_perf_penalty"].append(values["total_perf_penalty"])
            series["objective"].append(values["objective"])

    return series


def plot_summary(series: dict[str, list[float]], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    iterations = series["iteration"]

    plt.plot(iterations, series["total_cost"], marker="o", label="total_cost")
    plt.plot(iterations, series["total_perf_penalty"], marker="o", label="total_perf_penalty")
    plt.plot(iterations, series["objective"], marker="o", label="objective")

    plt.title("US run summary over iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def main() -> None:
    series = parse_summary_log(LOG_PATH)
    plot_summary(series, OUTPUT_PATH)


if __name__ == "__main__":
    main()
