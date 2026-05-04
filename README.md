# Thesis code — Market efficiency (DXY & EEM)

Bachelor thesis: **Testing Market Efficiency: Evidence from the US Dollar Index and Emerging Markets**.

This repository supports the empirical part of the thesis: **two parallel pipelines** for **US Dollar Index (DXY)** and **iShares MSCI Emerging Markets ETF (EEM)**. Both arms use daily data from 2005 through the fixed sample end in `config.py`, build aligned feature matrices (lags, cross-asset returns, rolling volatility), and compare:

- **Weak-form** style benchmarks (e.g. random-walk-like ARIMA(0,0,0)) with **two-step ARIMA/ARIMAX + GARCH**, **joint** AR / ARX + GARCH (`arch`),
- **Ridge / Lasso** on a common 15-feature set,
- **Random Forest, XGBoost, SVR** with time-series–safe cross-validation,

including rolling **out-of-sample** forecasts, Diebold–Mariano and related diagnostics, and thesis-style tables/figures. Root notebook **`05_appendix.ipynb`** regenerates **appendices B–D** (model diagnostics, coefficients, data tables) plus extra replication outputs.

---

## Repository layout

| Path | Role |
|------|------|
| `DXY_part/` | DXY target + 9 cross-asset predictors; `notebooks/01–03`, `src/`, `data/processed/`. |
| `EMCI_part/` | EEM target + 9 predictors; same notebook numbering pattern (`*_em` where needed). |
| `04_comparison.ipynb` | Side-by-side comparative charts for the thesis (run from **repo root**). |
| `05_appendix.ipynb` | Appendix tables/figures for print + supplementary grids/diagnostics (**repo root**). |
| `reports/tables/` | CSV exports from `05_appendix` (and related). |
| `reports/figures/` | PDF figures from `04`, `05`, and exploration notebooks as configured. |
| `requirements.txt` | Single dependency list for the whole project. |

---

## Environment (Python)

**Important:** install packages into the **same** interpreter that your Jupyter kernel uses (see [Reproduce](#how-to-reproduce-end-to-end)).

### Option A — `venv` (Windows example)

```powershell
cd "path\to\Diplom - copy"
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Option B — Conda

```powershell
conda create -n thesis-dxy-eem python=3.11 -y
conda activate thesis-dxy-eem
cd "path\to\Diplom - copy"
python -m pip install -r requirements.txt
```

After `conda activate`, prefer **`python -m pip`** so packages are not installed into the wrong environment.

Subfolders `DXY_part/requirements.txt` and `EMCI_part/requirements.txt` reference the root file (`-r ../requirements.txt`).

**Main libraries:** `pandas`, `numpy`, `statsmodels`, `arch`, `pmdarima`, `scikit-learn`, `xgboost`, `yfinance`, `matplotlib`, `seaborn`, etc. (see `requirements.txt`).

---

## How to reproduce end-to-end

1. **Kernel:** In the notebook, run `import sys; print(sys.executable)` and ensure it matches the environment where you installed dependencies.

2. **DXY pipeline**
   - `DXY_part/notebooks/01_data_exploration.ipynb` — download/clean data, EDA; writes `data/processed/dxy_dataset_features.csv` & `dxy_dataset_target.csv`.
   - `DXY_part/notebooks/02_model_fitting.ipynb` — ARIMA/GARCH/Ridge/Lasso, rolling OOS, prediction CSVs.
   - `DXY_part/notebooks/03_ML_evaluation.ipynb` — RF, XGBoost, SVR, OOS, tests.

3. **EEM pipeline**
   - `EMCI_part/notebooks/01_data_exploration_em.ipynb` → `02_model_fitting_em.ipynb` → `03_ML_evaluation_em.ipynb`.

4. **`04_comparison.ipynb`** — run with **working directory = repository root** (paths to `DXY_part/`, `EMCI_part/`, `reports/`).

5. **`05_appendix.ipynb`** — run from **repository root**. Heavy `arch` MLE refits: a full run may take **several minutes**.

**Minimal re-run:** If `data/processed/*.csv` is already present, you can often skip **01** and run **02–03** only, unless you need fresh downloads.

**Data:** Notebook **01** uses **Yahoo Finance** (`yfinance`); rerun requires internet. Sample end dates are set in `DXY_part/config.py` and `EMCI_part/config.py`.

---

## Troubleshooting

- **`ModuleNotFoundError`:** `python -m pip install -r requirements.txt` with the **same** Python as the kernel.
- **04 / 05 paths:** Open the project from the **repo root** or set the notebook cwd there so `DXY_part` / `EMCI_part` resolve.

---

## License / citation

Use and cite in line with your university’s thesis rules.
