"""Run the already-generated model result files into one summary CSV.

This script is intentionally lightweight: the expensive model scripts should be
run individually, and their metrics can then be combined here.
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="model_outputs")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    files = sorted(results_dir.glob("*metrics.csv")) + sorted(
        results_dir.glob("*results.csv")
    )
    if not files:
        raise FileNotFoundError(
            f"No result CSV files found in {results_dir}. Run model scripts first."
        )
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame.insert(0, "source_file", path.name)
        frames.append(frame)
    summary = pd.concat(frames, ignore_index=True, sort=False)
    output = results_dir / "combined_model_summary.csv"
    summary.to_csv(output, index=False)
    print(f"Saved {len(summary)} rows to {output}")


if __name__ == "__main__":
    main()
