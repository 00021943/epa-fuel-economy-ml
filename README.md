# Vehicle Fuel Economy — MLDA Coursework (6COSC017C-n)

Student ID: `00021943`

This project predicts the combined fuel economy (MPG) of cars with a combustion engine from their
technical characteristics (engine size, cylinders, transmission, drive type, fuel type, model year
and others). The goal is to estimate the fuel economy of cars from new model years before their
official EPA label is available, so the models are tested on the newest model years (2024–2027)
instead of a random sample.

Main steps (all in `notebooks/coursework.ipynb`):
- join of the EPA vehicle and emissions files, EDA and data cleaning;
- removal of leakage columns (`city08`, `highway08`, CO2 and other values calculated from the
  measured fuel consumption); electric and hydrogen cars are excluded (their MPG is an MPG-equivalent);
- time-based split (train 1984–2023, test 2024–2027) and `GroupKFold` cross-validation by base model;
- three tuned models (Ridge, Random Forest, Gradient Boosting) and a mean baseline; the final model
  (Random Forest) is chosen with a check on 2021–2023, before the test set is used.

## How this work differs from existing Kaggle notebooks
Kaggle has an older copy of this data (EPA, 1984–2017) with 19 public notebooks
(https://www.kaggle.com/datasets/epa/fuel-economy/code). Most of them only explore the data; three fit a model:
- [Predictive Modeling for MPG Estimation](https://www.kaggle.com/code/ffaraz/predictive-modeling-for-mpg-estimation):
  mixes electric and combustion-engine cars, random split, 4 features (no engine size), R² only.
- [2007 and 2008 FE using SimpleReg & MARS](https://www.kaggle.com/code/aissas/2007-and-2008-fe-using-simplereg-mars):
  trains on 2007 and tests on 199 cars from 2008, with one feature (engine size).
- [Fuel Economy Economy Analysis](https://www.kaggle.com/code/nnennaifeanyi/fuel-economy-economy-analysis):
  R linear regressions for annual fuel cost and the fuel economy score, R² on the training data only.

This project uses the official 2026 release, only combustion-engine cars, 13 features, a test set of four
future model years, cross-validation grouped by base model, three tuned models against two baselines
(mean and a look-up of the same car) and MAE, RMSE and R² for every model. Details are in the first cell
of the notebook.

## Dataset
US EPA and U.S. Department of Energy fuel economy data, downloaded from
https://www.fueleconomy.gov/feg/download.shtml on 7 October 2026 (records updated up to September 2026).
U.S. government data, public domain. Not downloaded from Kaggle.

- `data/raw/vehicles.csv`: one row per car configuration (50,407 rows, 84 columns).
- `data/raw/emissions.csv`: emission scores, joined to `vehicles.csv` on `id`.

Column documentation: https://www.fueleconomy.gov/feg/ws/index.shtml

## Project structure
```
data/raw/          original dataset (not modified)
notebooks/         Jupyter notebook with the full analysis
app/               Streamlit multi-page application
models/            final_model.joblib, the final Random Forest pipeline used by the app, refitted on all
                   years 1984–2027 (section E.8); the test results are for the same model trained on 1984–2023
pyproject.toml     dependencies
uv.lock            locked dependency versions
requirements.txt   dependencies for pip users
```

## Setup
With [uv](https://docs.astral.sh/uv/) (exact locked versions):
```bash
uv sync
```
Or with pip:
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

## Run the notebook
```bash
uv run jupyter notebook notebooks/coursework.ipynb
```
Run all cells from top to bottom (about 5 minutes, most of it is the model tuning in section D).
The notebook reads the data from `../data/raw/` and writes the final model to `../models/`.

## Run the app
```bash
uv run streamlit run app/Home.py
```
Deployed app: not deployed yet (the link will be added here).

## License
MIT — see `LICENSE`.
