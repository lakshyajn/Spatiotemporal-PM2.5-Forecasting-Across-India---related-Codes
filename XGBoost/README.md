# XGBoost PM2.5 Forecasting

This directory contains the XGBoost experiments for the project "Spatiotemporal PM2.5 Forecasting Across India: A Multi-Model Machine Learning and Ensemble Approach."

The shared configuration, feature selection, data loading, and evaluation utilities are provided in `common.py`. Individual scripts implement the forecasting, seasonal, feature-ablation, and localized prediction experiments.

## Files

| File                               | Purpose                                                                                                                  |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `common.py`                      | Shared XGBoost parameters, exact feature lists, data loading, and the notebook's monthly walk-forward helper             |
| `train_seasonal.py`              | Primary `t+1h` / `t+24h` monthly walk-forward evaluation (notebook cell 21) and season-specific evaluation (cell 23) |
| `train_no_meteorological.py`     | Meteorological-feature ablation from notebook cell 23                                                                    |
| `train_no_historical.py`         | Historical PM2.5/lag-feature ablation from notebook cell 23                                                              |
| `train_localized_anand_vihar.py` | Local Anand Vihar full-timeline prediction export from notebook cell 30                                                  |
| `plot_anand_vihar.py`            | Plot actual versus predicted values from the localized output CSV                                                        |
| `residual_analysis.py`           | Residual diagnostics for XGBoost localized predictions; stacking residuals are handled separately                        |
| `requirements.txt`               | Python dependencies                                                                                                      |
| `instructions.md`                | Setup and commands                                                                                                       |

## Dataset

## Dataset

The experiments use the prepared dataset `data/dataset_v4_filled.parquet`.

The dataset is not stored in the Git repository. Instructions for obtaining and placing the dataset are provided in [`data/README.md`](../data/README.md).

A different dataset location can be specified through the `--data` argument supported by the relevant training scripts.

## Quick start

Run the commands from the repository root after installing the dependencies and preparing the dataset.

```bash
python -m pip install -r XGBoost/requirements.txt
python XGBoost/train_seasonal.py --experiment both
python XGBoost/train_no_meteorological.py
python XGBoost/train_no_historical.py
python XGBoost/train_localized_anand_vihar.py
python XGBoost/plot_anand_vihar.py
python XGBoost/residual_analysis.py
```

Heavy model runs can take substantial time for the full multi-million-row dataset. Results are saved to `model_outputs/` by default. Use `--device cpu` if CUDA is unavailable; this can change speed and can produce small numerical differences versus the notebook's CUDA run.

## Reproducibility

The scripts preserve the corresponding notebook experiments' model parameters, feature configurations, prediction clipping, and data-splitting procedures. 

The saved overall forecasting results match Table 4.1 of the project report. Seasonal and feature-ablation experiments use separate feature configurations and evaluation procedures. Some results in the report differ slightly from the saved notebook outputs. The scripts calculate metrics from the experimental data without hard-coding the reported values. 

The localized Anand Vihar experiment reproduces the notebook's full-timeline prediction workflow. Since the model is trained on the labelled observations used for this visualization, its residuals are not independent test-set errors. 

Stacking and stacked-model residual analysis are documented separately.
