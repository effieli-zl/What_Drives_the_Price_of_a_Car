# What Drives the Price of a Car?

Regression analysis of ~426K used-car listings to identify the key drivers of price, following the **CRISP-DM** framework.

- **Deliverable:** `Assignment_11.1_What_Drives_the_Price_of_a_Car.ipynb`
- **Audience:** a used-car dealership fine-tuning its inventory
- **Task type:** supervised regression (target = `price`)

---

## Table of Contents

- [Project Structure](#project-structure)
- [Data](#data)
- [How to Run](#how-to-run)
- [CRISP-DM Workflow](#crisp-dm-workflow)
  - [1. Business Understanding](#1-business-understanding)
  - [2. Data Understanding](#2-data-understanding)
  - [3. Data Preparation](#3-data-preparation)
  - [4. Modeling](#4-modeling)
  - [5. Evaluation](#5-evaluation)
  - [6. Deployment](#6-deployment)
- [Key Findings](#key-findings)
- [Limitations & Next Steps](#limitations--next-steps)

---

## Project Structure

```
What_Drives_the_Price_of_a_Car/
├── Assignment_11.1_What_Drives_the_Price_of_a_Car.ipynb   # main analysis
├── data/
│   └── vehicles.csv        # raw dataset (~426K rows)
├── images/                 # supporting images
└── README.md
```

---

## Data

- **Source:** Kaggle used-car dataset (426,880 rows; subset of an original ~3M-row set)
- **Format:** single CSV, 18 columns
- **Target:** `price`
- **Feature groups:**
  - **Numeric:** `year`, `odometer`
  - **Categorical:** `manufacturer`, `model`, `condition`, `cylinders`, `fuel`, `title_status`, `transmission`, `drive`, `size`, `type`, `paint_color`, `region`, `state`
  - **Dropped as non-predictive:** `id`, `VIN`
- **Known quality issues:**
  - Heavy missingness — `size` (~72%), `condition`/`cylinders` (~40%), `VIN`/`drive` (~30%)
  - Extreme outliers in `price` (up to $3.7B) and `odometer` (up to 10M mi)
  - `price = 0` placeholder rows (not real salvage prices)

---

## How to Run

**Requirements**

- Python 3.9+
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `plotly`

**Steps**

1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn plotly
   ```
2. Ensure `data/vehicles.csv` is present.
3. Open the notebook and run all cells top to bottom:
   ```bash
   jupyter notebook Assignment_11.1_What_Drives_the_Price_of_a_Car.ipynb
   ```

> **Note:** the two Sequential Feature Selector cells are the slow step (several minutes on the full data). All later cells depend on variables defined earlier, so run in order.

---

## CRISP-DM Workflow

### 1. Business Understanding

- **Goal:** identify what makes a used car more or less expensive so the dealership can stock what consumers value.
- **Reframed as data problem:** regression — find which features are associated with `price` and how.
- **Success = ** clear driver identification + non-technical recommendations for inventory.

### 2. Data Understanding

- Loaded data; reviewed `info()`, `describe()`, missing-value counts, distributions.
- Flagged outliers and missingness (see [Data](#data)).

### 3. Data Preparation

- **Outlier trim:** keep `price > 0` and `odometer > 0`, both capped at the 99th percentile → ~383K rows.
- **Dropped columns:**
  - `id`, `VIN` — no analytical value
  - `region`, `model` — too high-cardinality
  - `size` — ~72% missing
- **Encoding:**
  - `condition` → ordinal (`salvage`=0 … `new`=5)
  - `cylinders` → numeric via regex extract
  - `manufacturer` → top-15 buckets (+ "Other"); `state` → top-25 buckets (+ "Other")
  - Remaining categoricals → one-hot (`drop_first=True`)
- **Missing values:** mode for low-missing fields; `"missing"` category for high-missing fields; median for numerics.
- **Two modeling frames:**
  - `vehicles_simplified` — 4 numeric features
  - `vehicles_cleaned` — full one-hot feature set (~83 features)

### 4. Modeling

- **Models tried:** Linear, Polynomial (degree search 1–5), Ridge + Sequential Feature Selection, Lasso — with and without polynomial terms.
- **Validation:** 5-fold cross-validation; `GridSearchCV` for polynomial degree and Lasso `alpha`.
- **Metric:** RMSE (dollars; interpretable), MSE used internally by GridSearch.
- **Key modeling lesson:** polynomial degree-3 predicts marginally better but produces uninterpretable, multicollinear interaction terms → switched to non-poly models for inference.

### 5. Evaluation

Unified comparison — every model scored on the **same 80/20 hold-out split**:

| Model | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|
| Lasso — all features + poly deg-3 | $8,484 | $5,890 | **0.60** |
| Lasso — all features (no poly) | $8,746 | $6,198 | 0.57 |
| Ridge — 15 selected features (no poly) | $9,008 | $6,429 | 0.55 |
| Polynomial deg-3 — 4 numeric | $9,376 | $6,535 | 0.51 |
| Linear — 4 numeric | $10,306 | $7,527 | 0.41 |
| Baseline (predict mean price) | $13,385 | $11,015 | 0.00 |

- **R²** = share of price variance explained (0 = no better than guessing the mean).
- Best predictor (poly Lasso) beats the interpretable model by only ~3% RMSE → interpretability chosen.
- Two independent methods (Ridge+SFS, Lasso) agree on the same signed driver list.

### 6. Deployment

- Non-technical report to dealers with ranked drivers + 5 inventory recommendations.
- Framed as **directional guidance**, not a per-VIN pricing tool.

---

## Key Findings

**Raises value** (biggest first)

- Newer model **year** — largest positive driver
- Lower **mileage**
- **Diesel** fuel (gas is cheapest)
- **Pickups & trucks** — body-style premium
- More **cylinders** (V6/V8)

**Lowers value** (biggest first)

- High **odometer** — largest negative driver
- **Front-wheel drive** (vs 4WD/RWD)
- **Rebuilt title** (vs clean)
- **Sedans**

**Two dominant levers:** mileage and model year outweigh everything else combined.

**Inventory recommendations**

1. Price & acquire on **year + mileage first**.
2. Stock **trucks/pickups** for margin; sedans for volume.
3. Prefer **clean titles**; discount rebuilt at buy.
4. Anchor the high end with **diesel / large-engine** units; gas 4-cylinders for value tier.
5. Watch **drivetrain** by region (4WD/RWD hold value better).

---

## Limitations & Next Steps

**Limitations**

- Best model explains ~60% of variance; ~$8,500 typical error on a ~$15k median car.
- Not a per-VIN price calculator.
- Accuracy capped by dropped high-cardinality fields (`model`, `region`) and heavy missingness.

**Next steps**

- Engineer `vehicle_age` and `miles_per_year`.
- Recover `model` / `trim` via target (mean) encoding instead of dropping.
- Revisit `price = 0` and placeholder handling.
- Build a per-VIN estimator once `model`/`trim` are incorporated.
