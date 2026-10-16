
# Technical Documentation & Paper Specifications: LightGBM Models

> **Source**: Extracted directly from `LighGBM/` source code and `data/dataset_v4.parquet` (3,304,237 rows, 2025-02-18 to 2026-06-26).
> All parameters, table entries, and timestamp ranges below are exact, verified, and ready to paste into your paper's **Methodology** and **Experimental Setup** sections.

---

## 1. Model Configurations: Exact Hyperparameters

### 1.1 LightGBM Hyperparameters

Two official parameter dictionaries are defined in `common_lightgbm.py`:

| Hyperparameter                              | `LGB_PARAMS_BASE`*(Models 1, 2, 5, 6, 7, 8, 9)* | `LGB_PARAMS_STRONG`*(Models 3, 4)* | Description / Rationale                             |
| ------------------------------------------- | --------------------------------------------------- | -------------------------------------- | --------------------------------------------------- |
| **Objective**                         | `regression`                                      | `regression`                         | Mean Squared Error (L2 loss)                        |
| **Metric**                            | `rmse`                                            | `rmse`                               | Root Mean Squared Error                             |
| **Learning Rate (`eta`)**           | **0.03**                                      | **0.05**                         | Step size shrinkage                                 |
| **Number of Leaves**                  | **63**                                        | **127**                          | Maximum tree leaves (capacity)                      |
| **Maximum Depth**                     | **-1** (unlimited)                            | **10**                           | BASE is depth-unconstrained; STRONG is capped at 10 |
| **Feature Fraction**                  | **0.80** (80%)                                | **0.70** (70%)                   | Subsampling ratio of columns per tree               |
| **Bagging Fraction**                  | **0.85** (85%)                                | **0.80** (80%)                   | Subsampling ratio of rows per iteration             |
| **Bagging Frequency**                 | **5**                                         | **3**                            | Bagging frequency in rounds                         |
| **Min Child Samples**                 | **30**                                        | **50**                           | Minimum number of samples per leaf                  |
| **L1 Regularization (`lambda_l1`)** | **0.10**                                      | **0.05**                         | Lasso sparsity penalty                              |
| **L2 Regularization (`lambda_l2`)** | **0.10**                                      | **0.10**                         | Ridge weight penalty                                |
| **Min Gain to Split**                 | `0.0` (default)                                   | **0.01**                         | Minimum split loss improvement guard                |
| **Max Boosting Rounds**               | **3,000**                                     | **5,000**                        | Maximum number of trees                             |
| **Early Stopping**                    | **50 rounds**                                 | **50 rounds**                    | Patience on validation RMSE                         |
| **Random Seed**                       | **42**                                        | **42**                           | Deterministic reproducibility                       |
| **Thread Count (`n_jobs`)**         | -1 (all cores)                                      | -1 (all cores)                         | Parallel execution                                  |

> **Design Choice**: `LGB_PARAMS_BASE` is used for all cross-validation walk-forward experiments where models are trained multiple times across 5 folds. `LGB_PARAMS_STRONG` provides higher capacity (127 leaves, depth 10) for single-split monthly and seasonal experiments where the train partition is larger.

---

### 1.2 Comparison with LSTM Architecture & Hyperparameters (from `LSTM/common.py`)

For cross-model documentation in your paper:

| Component                        | LSTM Model Specification                                                                                                                                                                                                                                                  |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Architecture**           | BiLSTM (hidden=128$\times$ 2 = 256) $\rightarrow$ Multi-Head Self-Attention (4 heads, dropout=0.30) $\rightarrow$ LayerNorm $\rightarrow$ LSTM (hidden=64) $\rightarrow$ Dense (128, GELU) $\rightarrow$ Dense (64, GELU) $\rightarrow$ Linear (24 outputs) |
| **Station Representation** | Trainable Embedding layer (`n_stations + 1`, `embed_dim=16`)                                                                                                                                                                                                          |
| **Input Sequence Length**  | **72 hours** (`SEQ_LEN = 72`)                                                                                                                                                                                                                                     |
| **Forecast Horizons**      | **24 hours** simultaneously ($t+1$ to $t+24$)                                                                                                                                                                                                                   |
| **Loss Function**          | Custom**Weighted Huber Loss** ($\delta=15.0$, exponential decay factor $0.95^{h-1}$ across horizons $h \in [1, 24]$)                                                                                                                                          |
| **Optimizer**              | **AdamW** (Weight decay = `5e-4`, Gradient clipping norm = `1.0`)                                                                                                                                                                                               |
| **Learning Rate**          | Peak LR =`1e-4` (Monsoon), `3e-4` (Winter/Spring); Linear warmup (`warmup_epochs` = 6 for Monsoon, 3 for others)                                                                                                                                                    |
| **Batch Size & Epochs**    | Batch size =**512**, Max Epochs = **30**                                                                                                                                                                                                                      |

> **XGBoost Note**: There is **no XGBoost implementation** in the repository. If XGBoost is discussed in your paper draft, it must either be documented as omitted / baseline or implemented using equivalent tree hyperparameters (`max_depth=6`, `learning_rate=0.03`, `subsample=0.85`, `colsample_bytree=0.80`, `reg_alpha=0.1`, `reg_lambda=0.1`).

---

## 2. Per-Model LightGBM Configurations

| #           | Script                                      | Target                    | Feature Set Description                             | Parameter Set         | Evaluation Scheme              |
| ----------- | ------------------------------------------- | ------------------------- | --------------------------------------------------- | --------------------- | ------------------------------ |
| **1** | `lag_model.py`                            | `pm25` (now)            | Lag + meteorological + fire + temporal              | `LGB_PARAMS_BASE`   | TimeSeriesSplit (5 folds)      |
| **2** | `no_lag_model.py`                         | `pm25`                  | Meteorological + fire + temporal + extra pollutants | `LGB_PARAMS_BASE`   | TimeSeriesSplit (5 folds)      |
| **3** | `no_lag_monthly_split_model.py`           | `pm25`                  | No-lag + domain engineered features                 | `LGB_PARAMS_STRONG` | Days 1–22 Train / 23–31 Test |
| **4** | `seasonal_split_model.py`                 | `pm25`                  | No-lag + domain engineered features                 | `LGB_PARAMS_STRONG` | 5 Indian Seasonal splits       |
| **5** | `lag_no_meteorological_features_model.py` | `pm25`                  | Lag features only (meteo dropped)                   | `LGB_PARAMS_BASE`   | TimeSeriesSplit (5 folds)      |
| **6** | `city_wise_lag_model.py`                  | `pm25`                  | Lag + meteorological                                | `LGB_PARAMS_BASE`   | TSCV All vs. Delhi-only subset |
| **7** | `top_k_feature_model.py`                  | `pm25`                  | Top-10 features selected by feature gain            | `LGB_PARAMS_BASE`   | TimeSeriesSplit (5 folds)      |
| **8** | `t_plus_1h_model.py`                      | `target_1h` ($t+1$)   | All available lag/meteo/station features            | `LGB_PARAMS_BASE`   | TimeSeriesSplit (5 folds)      |
| **9** | `t_plus_24h_model.py`                     | `target_24h` ($t+24$) | All available lag/meteo/station features            | `LGB_PARAMS_BASE`   | TimeSeriesSplit (5 folds)      |

---

## 3. Evaluation Splits: Exact Timestamp Boundaries & Consistency Check

### 3.1 Dataset Overview

- **Dataset**: `dataset_v4.parquet`
- **Total Records**: 3,304,237 hourly observations
- **Overall Time Span**: `2025-02-18 23:00:00` to `2026-06-26 23:00:00` (~16.3 months)
- **Monitoring Stations**: Multi-station Pan-India CPCB network

---

### 3.2 Experiment Evaluation Schemes & Timestamp Windows

#### A. 5-Fold Chronological TimeSeriesSplit (`tscv = TimeSeriesSplit(n_splits=5)`)

Used by Models 1, 2, 5, 6, 7, 8, 9:

| Fold             | Training Window                                                 | Train Rows | Validation / Test Window                                                  | Val Rows |
| ---------------- | --------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------- | -------- |
| **Fold 1** | `2025-02-18 23:00:00` $\rightarrow$ `2025-05-27 15:00:00` | 550,707    | `2025-05-27 15:00:00` $\rightarrow$ `2025-08-24 09:00:00`           | 550,706  |
| **Fold 2** | `2025-02-18 23:00:00` $\rightarrow$ `2025-08-24 09:00:00` | 1,101,413  | `2025-08-24 09:00:00` $\rightarrow$ `2025-11-16 04:00:00`           | 550,706  |
| **Fold 3** | `2025-02-18 23:00:00` $\rightarrow$ `2025-11-16 04:00:00` | 1,652,119  | `2025-11-16 04:00:00` $\rightarrow$ `2026-02-03 05:00:00`           | 550,706  |
| **Fold 4** | `2025-02-18 23:00:00` $\rightarrow$ `2026-02-03 05:00:00` | 2,202,825  | `2026-02-03 05:00:00` $\rightarrow$ `2026-04-14 16:00:00`           | 550,706  |
| **Fold 5** | `2025-02-18 23:00:00` $\rightarrow$ `2026-04-14 16:00:00` | 2,753,531  | **`2026-04-14 16:00:00` $\rightarrow$ `2026-06-26 23:00:00`** | 550,706  |

- **Anand Vihar Station (Fold 5 Held-out Test Period)**:
  - Station: `anand_vihar_new_delhi_-_dpcc_235` (10,491 total rows)
  - Train: `2025-02-20 05:00:00` $\rightarrow$ `2026-04-02 07:00:00` (8,743 rows)
  - **Test (Used for $t+1$ and $t+24$ Horizon Forecast Plots)**: **`2026-04-02 08:00:00` $\rightarrow$ `2026-06-26 22:00:00`** (1,748 rows)

---

#### B. Monthly Split (Model 3)

- **Train Window**: Days 1–22 of every month (`2025-02-18 23:00:00` $\rightarrow$ `2026-06-22 23:00:00`), **2,374,647 rows** (~71.9%)
- **Test Window**: Days 23–31 of every month (`2025-02-23 00:00:00` $\rightarrow$ `2026-06-26 23:00:00`), **929,590 rows** (~28.1%)

---

#### C. Seasonal Split (Model 4)

Partitioned by Indian climatological seasons across the timeline:

| Season                 | Train Months | Train Rows | Train Period                                              | Test Months | Test Rows | Test Period                                               |
| ---------------------- | ------------ | ---------- | --------------------------------------------------------- | ----------- | --------- | --------------------------------------------------------- |
| **Winter**       | Nov, Dec     | 453,463    | `2025-11-01 00:00` $\rightarrow$ `2025-12-31 23:00` | Jan         | 181,283   | `2026-01-01 00:00` $\rightarrow$ `2026-01-31 23:00` |
| **Spring**       | Feb, Mar     | 705,712    | `2025-02-18 23:00` $\rightarrow$ `2026-03-31 23:00` | Apr         | 393,932   | `2025-04-01 00:00` $\rightarrow$ `2026-04-30 23:00` |
| **Summer**       | May          | 408,804    | `2025-05-01 00:00` $\rightarrow$ `2025-05-31 23:00` | Jun         | 363,850   | `2025-06-01 00:00` $\rightarrow$ `2026-06-26 23:00` |
| **Monsoon**      | Jul, Aug     | 396,039    | `2025-07-01 00:00` $\rightarrow$ `2025-08-31 23:00` | Sep         | 189,828   | `2025-09-01 00:00` $\rightarrow$ `2025-09-30 23:00` |
| **Post-Monsoon** | Oct          | 211,326    | `2025-10-01 00:00` $\rightarrow$ `2025-10-31 23:00` | Nov         | 213,634   | `2025-11-01 00:00` $\rightarrow$ `2025-11-30 23:00` |

---

### 3.3 Do all experiments use the same final test period?

**No, they do not.**When addressing the reviewer or draft instructions:

1. **TSCV experiments (Models 1, 2, 5–9)** evaluate on walk-forward rolling test windows, with the final held-out evaluation window spanning **2026-04-14 to 2026-06-26** (all stations) or **2026-04-02 to 2026-06-26** (Anand Vihar).
2. **Monthly split (Model 3)** tests on the **tail 8–9 days (days 23–31)** of *every single month* across all 16 months.
3. **Seasonal split (Model 4)** tests on designated holdout calendar months (Jan, Apr, Jun, Sep, Nov) for each season separately.
4. **LSTM experiments** test on the **last 10 days of each month** (`TEST_DAYS = 10`), preceded by 5 validation days and 72-hour sequence lag requirements.

---

## 4. Stacking Procedure & Leakage Safety Analysis

### 4.1 Current Repository Status

There is **no meta-model or stacking ensemble implemented** in the current LightGBM codebase. Each script operates independently as an isolated benchmark model.

### 4.2 Leakage Safety Verification for Stacking / Meta-Models

If a stacking meta-model is described in your paper, here is the exact verification:

1. **TSCV Out-Of-Fold (OOF) Predictions ($\text{Models 1, 2, 5, 6, 7, 8, 9}$)**:
   - ✅ **Leakage-Safe**: In `common_lightgbm.py`, `run_tscv_regression` uses `TimeSeriesSplit(n_splits=5)`. For every fold $k$, the training indices strictly precede the validation indices ($t_{\text{train}} < t_{\text{val}}$). Predictions are generated only on the unseen future partition $X[\text{val\_idx}]$. Meta-features built from these concatenated OOF predictions are fully out-of-sample and preserve strict temporal causality.
2. **Monthly Split ($\text{Model 3}$)**:
   - ⚠️ **Temporal Leakage Risk**: Training on days 1–22 and testing on days 23–31 within each month means the model trained on days 1–22 of June 2026 already observed patterns when predicting earlier months. If base predictions from this split are used to train a meta-learner, data leakage will occur across months.
3. **Seasonal Split ($\text{Model 4}$)**:
   - ⚠️ **Temporal Leakage Risk**: Training on months across different years without walk-forward sequence preservation causes future observations (e.g., Nov 2025) to inform past predictions (e.g., Jan 2025).

### 4.3 Actionable Guidance for the Final Paper

- **If presenting an ensemble**: Use **only the TSCV-based OOF predictions** from Models 1, 2, 5, 8, 9 as meta-learner inputs.
- **Meta-learner Architecture**: A simple ridge regression (`Ridge(alpha=1.0)`) or shallow gradient booster trained strictly on Fold 1–4 OOF predictions, and evaluated on Fold 5.
