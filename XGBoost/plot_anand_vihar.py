"""Plot XGBoost actual-versus-predicted values from the Anand Vihar timeline CSV."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="model_outputs/Anand_Vihar_Full_Timeline_XGBoost.csv")
    parser.add_argument("--output", default="model_outputs/anand_vihar_xgboost_actual_vs_predicted.png")
    parser.add_argument("--start", help="Optional plot start time, e.g. 2026-01-01")
    parser.add_argument("--end", help="Optional plot end time, e.g. 2026-06-30")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Prediction CSV not found: {input_path}. Run train_localized_anand_vihar.py first."
        )
    df = pd.read_csv(input_path)
    df["timeline"] = pd.to_datetime(df["timeline"])
    if args.start:
        df = df[df["timeline"] >= pd.Timestamp(args.start)]
    if args.end:
        df = df[df["timeline"] <= pd.Timestamp(args.end)]
    df = df.sort_values("timeline")
    if df.empty:
        raise ValueError("No rows remain after applying the requested date range.")

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axes[0].plot(df["timeline"], df["actual_1h"], label="Actual PM2.5 (t+1h)", linewidth=1.2)
    axes[0].plot(df["timeline"], df["predicted_1h"], label="XGBoost Predicted (t+1h)", linewidth=1.0, alpha=0.85)
    axes[0].set_title("Anand Vihar: Actual vs Predicted PM2.5 (t+1 hour)")
    axes[0].set_ylabel("PM2.5")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].plot(df["timeline"], df["actual_24h"], label="Actual PM2.5 (t+24h)", linewidth=1.2)
    axes[1].plot(df["timeline"], df["predicted_24h"], label="XGBoost Predicted (t+24h)", linewidth=1.0, alpha=0.85)
    axes[1].set_title("Anand Vihar: Actual vs Predicted PM2.5 (t+24 hours)")
    axes[1].set_ylabel("PM2.5")
    axes[1].set_xlabel("Time")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output.resolve()}")


if __name__ == "__main__":
    main()
