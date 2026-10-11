# XGBoost Experiment Instructions

This guide explains how to set up the environment, prepare the dataset, run the XGBoost experiments, and locate their outputs.

## 1. Environment Setup

Python 3.10 or newer is recommended. Run these commands from the repository root:

```bash
python -m venv .venv
```

Activate the environment.

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install the XGBoost dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r XGBoost/requirements.txt
```

The notebook's training configuration uses CUDA by default. A compatible GPU and XGBoost installation are required for GPU execution. To use the CPU, pass `--device cpu` to the training script. CPU execution may be slower and can produce small numerical differences compared with a GPU run.

## 2. Dataset Setup

The default dataset path is:

```text
data/dataset_v4_filled.parquet
```

The dataset is not included in the repository. Refer to [`data/README.md`](../data/README.md) for dataset access and placement instructions. The dataset folder is intended to contain the instructions, while the actual data file remains excluded from version control.

The prepared Parquet dataset should contain a `timestamp` column, the forecast targets `target_1h` and `target_24h`, and the numeric feature columns required by the experiments. The main overall forecasting experiment selects numeric features while excluding timestamp/station metadata and future-target columns. Seasonal and ablation experiments use the feature list defined in the shared utility code.

A different dataset location can be provided using the `--data` argument where supported. For example:

```bash
python XGBoost/train_seasonal.py --data "/path/to/dataset_v4_filled.parquet" --experiment overall
```

## 3. Run the Experiments

All commands below should be run from the repository root.

### Overall Forecasting: One Hour and 24 Hours Ahead

```bash
python XGBoost/train_seasonal.py --experiment overall
```

This runs the main monthly walk-forward evaluation for `target_1h` and `target_24h`. For each month, the training set includes earlier months and days 1–24 of the current month; days 25 onward are used for testing. The output summarizes the monthly-fold metrics.

To select the execution device or dataset path:

```bash
python XGBoost/train_seasonal.py --experiment overall --device cpu --data "data/dataset_v4_filled.parquet"
```

### Seasonal Evaluation

```bash
python XGBoost/train_seasonal.py --experiment seasonal
```

This runs separate Winter, Summer, and Monsoon evaluations using the pruned feature list from the notebook. Within each month, days 1–20 are used for training and days 21 onward for testing. The season definitions are Winter (November–February), Summer (March–June), and Monsoon (July–October).

To run both the overall and seasonal experiments:

```bash
python XGBoost/train_seasonal.py --experiment both
```

### Meteorological-Feature Ablation

```bash
python XGBoost/train_no_meteorological.py
```

This compares the full pruned feature set with the same set after removing the configured meteorological variables. Both forecast targets are evaluated using the notebook's monthly day-20 split.

### Historical-Feature Ablation

```bash
python XGBoost/train_no_historical.py
```

This compares the pruned feature set with a reduced set excluding the historical PM2.5/lag/rolling/trend group configured in the notebook. The original feature-removal list also excludes `pm10`. Both forecast targets are evaluated.

### Localized Anand Vihar Predictions

```bash
python XGBoost/train_localized_anand_vihar.py
```

This generates localized predictions for the Anand Vihar station. The default output is:

```text
model_outputs/Anand_Vihar_Full_Timeline_XGBoost.csv
```

To specify a different dataset path, output path, or device:

```bash
python XGBoost/train_localized_anand_vihar.py \
  --data "data/dataset_v4_filled.parquet" \
  --output "model_outputs/Anand_Vihar_Full_Timeline_XGBoost.csv" \
  --device cpu
```

**Evaluation note:** this script follows the original notebook's visualization workflow: it trains using the available labelled Anand Vihar rows and predicts along the same timeline. These predictions are useful for plotting, but they are not an independent held-out test evaluation.

### Plot Localized Predictions

```bash
python XGBoost/plot_anand_vihar.py
```

By default, this reads `model_outputs/Anand_Vihar_Full_Timeline_XGBoost.csv` and saves an actual-versus-predicted plot to `model_outputs/anand_vihar_xgboost_actual_vs_predicted.png`.

Optional input/output paths and a plotting date range can be supplied:

```bash
python XGBoost/plot_anand_vihar.py \
  --input "model_outputs/Anand_Vihar_Full_Timeline_XGBoost.csv" \
  --output "model_outputs/anand_vihar_xgboost_actual_vs_predicted.png" \
  --start 2026-01-01 --end 2026-06-30
```

### XGBoost Residual Diagnostics

```bash
python XGBoost/residual_analysis.py
```

This reads the localized Anand Vihar prediction CSV and generates residual diagnostics for the XGBoost predictions. Because these predictions come from the full-timeline fitting workflow, the residuals are in-sample diagnostics and must not be interpreted as held-out test errors. Stacking and stacked-model residual analysis are maintained separately.

To use a different input CSV or output directory:

```bash
python XGBoost/residual_analysis.py \
  --input "model_outputs/Anand_Vihar_Full_Timeline_XGBoost.csv" \
  --output-dir "model_outputs"
```

## 4. Generated Outputs

The scripts create `model_outputs/` when needed. Depending on the experiment, outputs include:

* `overall_forecast_metrics.csv` and `overall_forecast_fold_metrics.csv`
* `seasonal_metrics.csv` and `seasonal_fold_metrics.csv`
* `meteorological_ablation_metrics.csv` and `meteorological_ablation_fold_metrics.csv`
* `historical_ablation_metrics.csv` and `historical_ablation_fold_metrics.csv`
* `Anand_Vihar_Full_Timeline_XGBoost.csv`
* Actual-versus-predicted plots and XGBoost residual diagnostic outputs

Generated model outputs and the dataset should remain local unless selected result artifacts are intentionally published.

## 5. Reference Results and Reproducibility

The saved overall walk-forward output in the original notebook matches the report's Table 4.1:

| Forecast target |    MAE |   RMSE |    R² |
| --------------- | -----: | -----: | ----: |
| `target_1h`     |  8.636 | 23.667 | 0.787 |
| `target_24h`    | 20.551 | 42.573 | 0.410 |

The seasonal and feature-ablation experiments use a different feature configuration and split procedure from the primary overall evaluation. Some values in the report differ slightly from the stored notebook outputs. Scripts calculate metrics from the dataset at run time; they do not insert report values as constants. For publication-quality reproducibility, retain the generated CSV metrics for each run and compare them with the corresponding report table.

Stacking code and stacked-model residual figures are not included in this directory and are maintained separately.

## 6. Troubleshooting

### Dataset Not Found

Confirm that `data/dataset_v4_filled.parquet` exists, or provide the full path through `--data`.

### Missing Column Error

Check that the Parquet file is the prepared forecasting dataset and includes the target columns and feature fields required by the chosen experiment.

### CUDA/Device Error

Try `--device cpu` on training scripts that support the option. CPU execution can take longer on the full dataset.

### High Memory Use or Long Runtime

The dataset contains millions of rows. Start by checking the script and dependencies on a small development sample if needed, but use the intended full dataset and notebook configuration for final result reproduction. Do not interpret results from a sample as the final report metrics.
