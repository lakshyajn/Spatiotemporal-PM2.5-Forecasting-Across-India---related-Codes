"""Residual diagnostics for saved localized XGBoost predictions.

Residual = actual - predicted. Stacking residual plots belong to the separate
stacking implementation; this file currently analyzes XGBoost's own Anand Vihar CSV.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="model_outputs/Anand_Vihar_Full_Timeline_XGBoost.csv")
    parser.add_argument("--output-dir", default="model_outputs")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction CSV not found: {path}. Run train_localized_anand_vihar.py first."
        )
    df = pd.read_csv(path)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for horizon in ("1h", "24h"):
        actual_col, pred_col = f"actual_{horizon}", f"predicted_{horizon}"
        missing = [c for c in (actual_col, pred_col) if c not in df.columns]
        if missing:
            raise KeyError(f"CSV is missing columns required for {horizon}: {missing}")
        actual = pd.to_numeric(df[actual_col], errors="coerce")
        predicted = pd.to_numeric(df[pred_col], errors="coerce")
        mask = actual.notna() & predicted.notna() & np.isfinite(actual) & np.isfinite(predicted)
        actual = actual[mask].to_numpy(dtype=float)
        predicted = predicted[mask].to_numpy(dtype=float)
        residual = actual - predicted
        all_metrics.append({
            "horizon": horizon, "rows": int(len(actual)),
            "mean_residual": float(np.mean(residual)) if len(residual) else float("nan"),
            "mae": float(np.mean(np.abs(residual))) if len(residual) else float("nan"),
            "rmse": float(np.sqrt(np.mean(residual ** 2))) if len(residual) else float("nan"),
        })

        fig, axes = plt.subplots(2, 1, figsize=(9, 8))
        axes[0].hist(residual, bins=60, edgecolor="black", alpha=0.75)
        axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
        axes[0].set_title(f"XGBoost Residual Distribution ({horizon})")
        axes[0].set_xlabel("Residual: actual - predicted")
        axes[0].set_ylabel("Frequency")
        axes[0].grid(True, linestyle="--", alpha=0.25)

        axes[1].scatter(actual, residual, s=8, alpha=0.25)
        axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
        axes[1].set_title(f"Actual PM2.5 vs Residuals ({horizon})")
        axes[1].set_xlabel("Actual PM2.5")
        axes[1].set_ylabel("Residual")
        axes[1].grid(True, linestyle="--", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / f"xgboost_residual_analysis_{horizon}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    result = pd.DataFrame(all_metrics)
    result.to_csv(out / "xgboost_residual_metrics.csv", index=False)
    print(result.to_string(index=False))
    print(f"Saved residual plots and metrics under: {out.resolve()}")


if __name__ == "__main__":
    main()
