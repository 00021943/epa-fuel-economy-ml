"""Shared code for all pages: paths, data preparation, model loading, look-up."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Paths are resolved from this file, so the app works from any working directory
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "models" / "final_model.joblib"
EVALUATION_PATH = ROOT / "models" / "evaluation.json"

TARGET = "comb08"
NUM_FEATURES = ["year", "displ", "cylinders", "gears"]
BIN_FEATURES = ["turbo", "supercharger"]
CAT_FEATURES = ["make", "fuelType1", "drive_group", "vehicle_type", "trans_type", "fuel_tech", "start_stop"]
FEATURES = NUM_FEATURES + BIN_FEATURES + CAT_FEATURES
# Features used to find an identical car (all features except the year)
LOOKUP_FEATURES = [f for f in FEATURES if f != "year"]
HYBRID_TECHS = ["Hybrid", "Plug-in Hybrid"]

DRIVE_MAP = {
    "Front-Wheel Drive": "FWD",
    "Rear-Wheel Drive": "RWD",
    "2-Wheel Drive": "RWD",
    "All-Wheel Drive": "4WD/AWD",
    "4-Wheel Drive": "4WD/AWD",
    "4-Wheel or All-Wheel Drive": "4WD/AWD",
    "Part-time 4-Wheel Drive": "4WD/AWD",
}

# Raw columns kept next to the features (used on the Preprocessing page and for look-up)
KEPT_RAW_COLUMNS = ["id", "model", "baseModel", "drive", "VClass", "trany", "tCharger",
                    "sCharger", "atvType", "startStop"]


def get_vehicle_type(vclass: str) -> str:
    """Group the detailed EPA vehicle class into 6 simple types."""
    vclass = vclass.lower()
    if "pickup" in vclass:
        return "Pickup"
    if "sport utility" in vclass:
        return "SUV"
    if "van" in vclass:
        return "Van"
    if "special purpose" in vclass:
        return "Special Purpose"
    if "wagon" in vclass:
        return "Station Wagon"
    return "Car"


def get_trans_type(trany: str) -> str:
    """Manual, CVT or Automatic from the EPA transmission text."""
    if trany.startswith("Manual"):
        return "Manual"
    if "variable" in trany or "AV" in trany:
        return "CVT"
    return "Automatic"


def _join_emissions(vehicles: pd.DataFrame) -> pd.DataFrame:
    """Left join of emission scores, aggregated to one row per car id."""
    emissions = pd.read_csv(RAW_DIR / "emissions.csv")
    emissions.loc[emissions["score"] < 1, "score"] = None  # impossible scores
    per_car = emissions.groupby("id").agg(
        emission_score=("score", "mean"),
        smartway_score=("smartwayScore", "max"),
    )
    return vehicles.merge(per_car, on="id", how="left", validate="one_to_one")


def _fix_wrong_values(data: pd.DataFrame) -> pd.DataFrame:
    """Fill the engine of Subaru RX Turbo 1985 and the cylinders of the Mazda RX-7."""
    data = data.copy()
    subaru = (data["make"] == "Subaru") & (data["model"] == "RX Turbo") & (data["year"] == 1985)
    data.loc[subaru, ["displ", "cylinders"]] = [1.8, 4]
    mazda_rotary = (data["make"] == "Mazda") & data["model"].str.startswith("RX-")
    data.loc[mazda_rotary & data["cylinders"].isna(), "cylinders"] = 2
    return data


def _add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering (notebook section C.5)."""
    data = data.copy()
    data["drive_group"] = data["drive"].map(DRIVE_MAP).fillna("Unknown")
    data["vehicle_type"] = data["VClass"].map(get_vehicle_type)
    data["trans_type"] = data["trany"].map(get_trans_type)
    data["gears"] = data["trany"].str.extract(r"(\d+)")[0].astype(float)
    data.loc[data["trans_type"] == "CVT", "gears"] = np.nan  # CVT has no fixed gears
    data["turbo"] = data["tCharger"].notna().astype(int)
    data["supercharger"] = (data["sCharger"] == "S").astype(int)
    data["fuel_tech"] = data["atvType"].fillna("Standard")
    data["start_stop"] = data["startStop"].fillna("Unknown")
    return data


@st.cache_data(show_spinner="Preparing the data ...")
def load_prepared_data() -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Reproduce the notebook data preparation. Returns the data and the row count after each step."""
    vehicles = pd.read_csv(RAW_DIR / "vehicles.csv", low_memory=False)
    steps = [("Raw vehicles.csv", len(vehicles))]

    data = _join_emissions(vehicles)
    steps.append(("After joining emissions (one row per car)", len(data)))

    # Remove electric, hydrogen and fuel-cell cars (empty fuelType1)
    no_engine = data["fuelType1"].isin(["Electricity", "Hydrogen"]) | data["fuelType1"].isna()
    data = data[~no_engine]
    steps.append(("Combustion-engine cars only", len(data)))

    # Rows that are identical in every column except id
    data = data[~data.drop(columns="id").duplicated()]
    steps.append(("Duplicates removed", len(data)))

    data = _fix_wrong_values(data)
    data = data.dropna(subset=["trany"])
    steps.append(("Missing transmission removed", len(data)))

    data = _add_features(data)
    columns = [TARGET] + FEATURES + KEPT_RAW_COLUMNS
    return data[columns].reset_index(drop=True), steps


@st.cache_resource(show_spinner="Loading the model ...")
def load_model() -> Pipeline:
    """The final scikit-learn pipeline (preprocessing + Random Forest)."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_evaluation() -> dict:
    """Results saved by the notebook."""
    with open(EVALUATION_PATH, encoding="utf-8") as file:
        return json.load(file)


def make_preprocessor() -> ColumnTransformer:
    """Same preprocessing as notebook section C.8."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUM_FEATURES),
        ("bin", "passthrough", BIN_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False),
         CAT_FEATURES),
    ])


def table_to_frame(table: dict) -> pd.DataFrame:
    """Convert {model: {column: value}} from evaluation.json to a DataFrame (one row per model)."""
    return pd.DataFrame(table).T


def find_identical_cars(data: pd.DataFrame, model_name: str, car: dict) -> pd.DataFrame:
    """All rows (all years) with the same model name and the same 12 features (year excluded)."""
    mask = data["model"] == model_name
    for feature in LOOKUP_FEATURES:
        value = car[feature]
        if pd.isna(value):
            mask &= data[feature].isna()  # CVT: gears is empty on both sides
        else:
            mask &= data[feature] == value
    return data[mask]


def lookup_epa_value(data: pd.DataFrame, model_name: str, car: dict) -> tuple[float, int] | None:
    """EPA MPG of an identical car in the latest year where it appears, or None."""
    found = find_identical_cars(data, model_name, car)
    if found.empty:
        return None
    latest_year = int(found["year"].max())
    mpg = float(found.loc[found["year"] == latest_year, TARGET].median())
    return mpg, latest_year
