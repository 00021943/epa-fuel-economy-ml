# Vehicle Fuel Economy — MLDA Coursework (6COSC017C-n)

Student ID: `00021943`

<!-- TODO: one short paragraph — what problem you solve and why it matters -->

## Dataset
US EPA Fuel Economy Data — https://www.fueleconomy.gov/feg/download.shtml
File: `data/raw/vehicles.csv` (downloaded <DATE>).
Column documentation: https://www.fueleconomy.gov/feg/ws/index.shtml

## Project structure
```
data/raw/          original dataset (not modified)
notebooks/         Jupyter notebook with the full analysis
app/               Streamlit multi-page application
models/            saved trained models
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

## Run the app
```bash
uv run streamlit run app/Home.py
```
Deployed app: <!-- TODO: Streamlit Cloud link -->

## License
MIT — see `LICENSE`.
