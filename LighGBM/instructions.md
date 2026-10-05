# PM2.5 Forecasting with LightGBM

This project evaluates several LightGBM models for hourly PM2.5 forecasting.
The experiments compare historical PM2.5 lag features, meteorological and fire
features, seasonal/monthly splits, city-specific training, feature selection,
and one-hour and 24-hour forecasting horizons.

The project expects a Parquet dataset named `dataset_v4.parquet`. The dataset
is not included in this repository because it is large. Place it in the project
folder or pass its full path with `--data`.

## Project files

| File | Experiment |
| --- | --- |
| `common_lightgbm.py` | Shared loading, preprocessing, feature selection, training, metrics, and plotting helpers |
| `lag_model.py` | PM2.5 lag features with chronological `TimeSeriesSplit` validation |
| `no_lag_model.py` | No-lag model with meteorological, pollutant, fire, and station features |
| `no_lag_monthly_split_model.py` | Train on days 1-22 and test on days 23-31 of each month |
| `seasonal_split_model.py` | Seasonal train/test evaluation using Indian seasonal month groupings |
| `lag_no_meteorological_features_model.py` | Lag model with meteorological columns removed |
| `city_wise_lag_model.py` | Compare all-station training against one selected city |
| `top_k_feature_model.py` | Compare all lag features against the most important Top-K features |
| `t_plus_1h_model.py` | One-hour-ahead prediction using `target_1h` |
| `t_plus_24h_model.py` | 24-hour-ahead prediction using `target_24h` |
| `model_summary.py` | Combine generated CSV result files |
| `lightGBM_all_models.ipynb` | Original combined notebook containing all experiments |

## 1. Create the environment

From the project folder, create and activate a virtual environment.

### Windows PowerShell

```powershell
cd "E:\Academics\BTP"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
cd /path/to/BTP
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.10 or newer is recommended. Use the same environment when running
the scripts or opening the notebook.

## 2. Prepare the dataset

Place the dataset at:

```text
E:\Academics\BTP\dataset_v4.parquet
```

Alternatively, pass a different path to every script:

```powershell
python lag_model.py --data "D:\datasets\dataset_v4.parquet"
```

The dataset must contain at least:

- `timestamp`: date/time column
- `station`: monitoring station identifier
- `pm25`: current PM2.5 value

The feature-rich and seasonal models also expect columns such as
`temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `surface_pressure`,
`precipitation`, `cloud_cover`, `shortwave_radiation`,
`boundary_layer_height`, `wind_speed`, `wind_u`, `wind_v`, `fire_count`,
`frp_mean`, and `is_winter`.

The horizon scripts additionally require `target_1h` or `target_24h`.
Lag columns used by the lag experiments include `pm25_lag1`, `pm25_lag3`,
`pm25_lag6`, `pm25_lag12`, `pm25_lag24`, `pm25_lag48`, `pm25_lag72`,
`pm25_rmean24`, `pm25_rmax24`, `pm25_rmin24`, `pm25_rstd24`, and
`pm25_trend6h`.

## 3. Run an experiment

All scripts write plots and CSV files to `model_outputs` by default.

```powershell
python lag_model.py --data "dataset_v4.parquet"
python no_lag_model.py --data "dataset_v4.parquet"
python no_lag_monthly_split_model.py --data "dataset_v4.parquet"
python seasonal_split_model.py --data "dataset_v4.parquet"
python lag_no_meteorological_features_model.py --data "dataset_v4.parquet"
python city_wise_lag_model.py --data "dataset_v4.parquet" --city delhi
python top_k_feature_model.py --data "dataset_v4.parquet" --top-k 10
python t_plus_1h_model.py --data "dataset_v4.parquet"
python t_plus_24h_model.py --data "dataset_v4.parquet"
```

To use another output folder:

```powershell
python lag_model.py `
  --data "D:\datasets\dataset_v4.parquet" `
  --output-dir "results\lag"
```

Each `TimeSeriesSplit` experiment uses five chronological folds. The models
use LightGBM early stopping and report RMSE, MAE, R2, fold metrics, and feature
importance. PM2.5 predictions are clipped to zero in the no-lag,
no-meteorological, monthly, seasonal, and horizon experiments. The city-wise
comparison preserves the original lag experiment's unclipped predictions.

## 4. Run all experiments

For a complete run, execute the commands one at a time. The dataset contains
millions of rows, so training can require substantial RAM, CPU time, and disk
space for generated plots and metrics.

```powershell
$data = "dataset_v4.parquet"
python lag_model.py --data $data
python no_lag_model.py --data $data
python no_lag_monthly_split_model.py --data $data
python seasonal_split_model.py --data $data
python lag_no_meteorological_features_model.py --data $data
python city_wise_lag_model.py --data $data --city delhi
python top_k_feature_model.py --data $data --top-k 10
python t_plus_1h_model.py --data $data
python t_plus_24h_model.py --data $data
```

For a quick development check, use a smaller copy/sample of the dataset.
Do not use a sampled dataset for final paper results.

## 5. Combine result files

After running the experiments:

```powershell
python model_summary.py --results-dir model_outputs
```

This creates `model_outputs\combined_model_summary.csv`. Monthly, seasonal,
city-wise, and feature-importance outputs are also saved in the selected
output directory.

## 6. Run the notebook

The original notebook is useful for inspecting all experiments in one place:

```powershell
jupyter notebook lightGBM_all_models.ipynb
```

If using Google Colab, upload the notebook and mount Google Drive. Update the
dataset path in the notebook's `DATA_CANDIDATES` list before running the cells.

## Reproducibility notes

- The scripts sort records by `timestamp` before modelling.
- Chronological splits are used instead of random train/test splits.
- The LightGBM random seed is set to `42`.
- Missing meteorological and pollutant values are interpolated where available.
- Fire-related missing values are filled with zero.
- Model outputs are generated locally and are not committed as source data.

## Common problems

### `FileNotFoundError: No dataset found`

Pass the correct Parquet path explicitly:

```powershell
python lag_model.py --data "C:\full\path\dataset_v4.parquet"
```

### `ImportError` for `pyarrow`

Activate the virtual environment and reinstall:

```powershell
pip install -r requirements.txt
```

### Missing required column

Check the Parquet schema and confirm that the required columns for the chosen
experiment are present. The horizon scripts require their matching target
column, while seasonal/engineered-feature scripts require meteorological and
calendar columns.

### Training is too slow or uses too much memory

Start with a smaller development dataset, close other memory-heavy programs,
or reduce the dataset before training. Keep the full dataset and the default
settings for final reported results.
