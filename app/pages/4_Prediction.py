import pandas as pd
import streamlit as st

from utils import (HYBRID_TECHS, TARGET, load_evaluation, load_model, load_prepared_data,
                   lookup_epa_value)

st.set_page_config(page_title="Prediction", layout="wide")
st.title("Prediction")

data, _ = load_prepared_data()
model = load_model()
evaluation = load_evaluation()
NEW_MODEL = "New model / not in the list"
LAST_TRAIN_YEAR = 2027
DISCLAIMER = "This is an estimate of the EPA label, not an official EPA value."

st.write("Enter the details of a car. " + DISCLAIMER)

# Make and model name are outside the form, so the model list updates when the make changes
makes = sorted(data["make"].unique())
make = st.selectbox("Make", makes, index=makes.index("Toyota"))
names = sorted(data.loc[data["make"] == make, "model"].unique())
model_name = st.selectbox("Model name", [NEW_MODEL] + names)


def options(column: str) -> list:
    return sorted(data[column].unique())


with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    year = c1.number_input("Model year", 2024, 2030, 2025)
    displ = c1.number_input("Engine size (litres)", 0.6, 8.4, 2.0, step=0.1)
    cylinders = c1.selectbox("Cylinders", [int(c) for c in options("cylinders")], index=2)
    trans_type = c2.selectbox("Transmission", options("trans_type"), index=0)
    gears = c2.number_input("Number of gears (ignored for CVT)", 1, 10, 8)
    drive_group = c2.selectbox("Drive", [d for d in options("drive_group") if d != "Unknown"])
    vehicle_type = c3.selectbox("Vehicle type", options("vehicle_type"), index=options("vehicle_type").index("Car"))
    fuel_tech = c3.selectbox("Fuel technology", options("fuel_tech"), index=options("fuel_tech").index("Standard"))
    fuel_type1 = c3.selectbox("Fuel type", options("fuelType1"), index=options("fuelType1").index("Regular Gasoline"))
    d1, d2, d3 = st.columns(3)
    turbo = d1.checkbox("Turbo")
    supercharger = d2.checkbox("Supercharger")
    start_stop = d3.checkbox("Start-stop system", value=True)
    submitted = st.form_submit_button("Predict MPG")


def build_car() -> dict:
    """The 13 model features from the form values."""
    return {
        "year": min(int(year), LAST_TRAIN_YEAR), "displ": float(displ), "cylinders": float(cylinders),
        "gears": float("nan") if trans_type == "CVT" else float(gears),
        "turbo": int(turbo), "supercharger": int(supercharger), "make": make, "fuelType1": fuel_type1,
        "drive_group": drive_group, "vehicle_type": vehicle_type, "trans_type": trans_type,
        "fuel_tech": fuel_tech, "start_stop": "Y" if start_stop else "N",
    }


def similar_cars(car: dict) -> pd.DataFrame:
    """Up to 5 cars with the same fuel technology and vehicle type and the closest engine size."""
    pool = data[(data["fuel_tech"] == car["fuel_tech"]) & (data["vehicle_type"] == car["vehicle_type"])].copy()
    pool["engine size gap"] = (pool["displ"] - car["displ"]).abs()
    pool = pool.sort_values(["engine size gap", "year"], ascending=[True, False])
    columns = ["make", "model", "displ", "cylinders", "trans_type", "drive_group", TARGET]
    return pool.drop_duplicates(columns).head(5)[["year"] + columns]


def show_warnings(car: dict) -> None:
    mae = evaluation["lookup"]["model_mae_not_matched"]
    st.caption(f"Typical error for a car that is not in the data: about ±{mae:.1f} MPG.")
    if car["fuel_tech"] in HYBRID_TECHS:
        tech_mae = evaluation["test_mae_per_fuel_tech"][evaluation["final_model"]][car["fuel_tech"]]
        st.warning(f"{car['fuel_tech']} cars are harder to predict: the typical test error is "
                   f"{tech_mae:.1f} MPG. Treat this value with extra care.")
    if int(year) > LAST_TRAIN_YEAR:
        st.info(f"The data ends in {LAST_TRAIN_YEAR}. The model treats year {int(year)} like {LAST_TRAIN_YEAR}, "
                "because tree models do not extrapolate.")


if submitted:
    car = build_car()
    predicted = float(model.predict(pd.DataFrame([car]))[0])
    found = None if model_name == NEW_MODEL else lookup_epa_value(data, model_name, car)
    if found:
        st.success(f"EPA value of an identical car: **{found[0]:.0f} MPG** (model year {found[1]}).")
        st.metric("Model prediction (for comparison)", f"{predicted:.1f} MPG")
    else:
        st.success(f"Estimated combined MPG: **{predicted:.1f}**")
        show_warnings(car)
    st.caption(DISCLAIMER)
    st.subheader("Similar cars in the data")
    st.dataframe(similar_cars(car), hide_index=True, width="stretch")
