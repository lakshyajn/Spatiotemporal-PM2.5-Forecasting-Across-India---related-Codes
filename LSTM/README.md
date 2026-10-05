
# LSTM PM2.5 Forecasting – Reproducibility Code

Independent scripts for the LSTM experiments reported in the paper.

## Requirements

- Python 3.9+
- torch, numpy, pandas, scikit-learn, pyarrow, matplotlib, tqdm

## Dataset

Set the parquet path inside `lstm/common.py`:

DATASET_PATH = "path/to/dataset.parquet"

## Files

| File                               | Result                                  |
| ---------------------------------- | --------------------------------------- |
| `train_seasonal.py`              | Seasonal LSTM (winter, summer, monsoon) |
| `train_no_meteorology.py`        | LSTM without meteorological features    |
| `train_no_historical.py`         | LSTM without historical PM2.5           |
| `train_localized_anand_vihar.py` | Single-station LSTM                     |
| `plot_anand_vihar.py`            | Actual vs predicted at Anand Vihar      |
| `residual_analysis.py`           | Residual histograms and scatter         |

## Run

python train_seasonal.py
python train_no_meteorology.py
python train_no_historical.py
python train_localized_anand_vihar.py
python plot_anand_vihar.py
python residual_analysis.py
