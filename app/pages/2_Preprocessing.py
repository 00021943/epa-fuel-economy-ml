import altair as alt
import pandas as pd
import streamlit as st

from utils import TARGET, load_model, load_prepared_data

st.set_page_config(page_title="Preprocessing", layout="wide")
st.title("Preprocessing")

data, steps = load_prepared_data()
model = load_model()

STEP_REASONS = {
    "Raw vehicles.csv": "Original EPA table, one row per car configuration.",
    "After joining emissions (one row per car)": "Emission scores were averaged per car id first, so the join "
                                                  "does not create duplicate rows (checked with one-to-one).",
    "Combustion-engine cars only": "Electric, hydrogen and fuel-cell cars have no engine size and an "
                                   "MPG-equivalent that is not comparable.",
    "Duplicates removed": "Rows identical in every column except the id give some cars extra weight.",
    "Missing transmission removed": "Only 2 rows had no transmission, too few to impute.",
}

st.header("Rows after each step")
step_table = pd.DataFrame(steps, columns=["Step", "Rows"])
step_table["Reason"] = step_table["Step"].map(STEP_REASONS)
st.dataframe(step_table, hide_index=True, width="stretch")
st.caption(f"The final modelling table has {len(data):,} cars.")

st.header("Data corrections")
st.markdown(
    "- **Subaru RX Turbo 1985** (2 rows): engine size and cylinders were empty. They were set to the "
    "real values, 1.8 litres and 4 cylinders.\n"
    "- **Mazda RX-7 1986** (1 row): cylinders were empty. All other RX-7 and RX-8 rows have 2 (rotary "
    "engine with 2 rotors), so 2 was used.\n"
    "- **City MPG higher than highway MPG** (565 cars): almost all are hybrids, where this is correct, "
    "so the values were kept."
)

st.header("Missing values")
missing = pd.DataFrame([
    ("trany (transmission)", steps[-2][1] - steps[-1][1], "Rows removed"),
    ("drive", int(data["drive"].isna().sum()), "Own category 'Unknown' (all from 1984)"),
    ("tCharger", int(data["tCharger"].isna().sum()), "Empty means no turbo, so turbo = 0"),
    ("sCharger", int(data["sCharger"].isna().sum()), "Empty means no supercharger, so supercharger = 0"),
    ("atvType", int(data["atvType"].isna().sum()), "Empty means no special technology: 'Standard'"),
    ("startStop", int(data["startStop"].isna().sum()), "Not reported before 2013: 'Unknown'"),
    ("gears (CVT cars)", int(data["gears"].isna().sum()), "CVT has no fixed gears: median imputation in the pipeline"),
], columns=["Column", "Missing rows", "Handling"])
st.dataframe(missing, hide_index=True, width="stretch")

st.header("Removed columns (data leakage and others)")
st.markdown(
    "- **Leakage**: `city08`, `highway08` and all their versions, `co2`, `co2TailpipeGpm`, `barrels08`, "
    "`fuelCost08`, `youSaveSpend`, `feScore`, `ghgScore`, `smartway_score`. They are calculated from the "
    "measured fuel consumption, so they would give the answer to the model (correlation 0.99-1.00 with the target).\n"
    "- **Electric-part columns** (`range`, `phevComb`, `charge240`, ...): meaningful for about 1% of the cars only.\n"
    "- **Identifiers and dates** (`id`, `model`, `baseModel`, `engId`, `createdOn`, `modifiedOn`): "
    "they do not describe the technology, and `model` would only let the model memorise names.\n"
    "- **Raw columns replaced by engineered features**: `drive`, `VClass`, `trany`, `tCharger`, `sCharger`, "
    "`atvType`, `startStop`.\n"
    "- **`emission_score`**: about half is missing and users of the app would not know it."
)

st.header("Feature engineering explorer")
engineered = ["drive_group", "vehicle_type", "trans_type", "gears", "turbo", "supercharger", "fuel_tech", "start_stop"]
feature = st.selectbox("Engineered feature", engineered)
groups = data.groupby(data[feature].map(str))[TARGET].agg(["count", "median"]).reset_index()
groups.columns = ["group", "Cars", "Median MPG"]
left, right = st.columns(2)
left.dataframe(groups, hide_index=True, width="stretch")
chart = alt.Chart(groups).mark_bar().encode(
    alt.X("group:N", title=feature, sort=None), alt.Y("Median MPG:Q"), tooltip=["group", "Cars", "Median MPG"])
right.altair_chart(chart, width="stretch")
st.caption("The groups have clearly different median MPG, so each engineered feature carries information.")

st.header("Encoding and scaling")
n_columns = len(model.named_steps["prep"].get_feature_names_out())
st.markdown(
    "- **Numeric** (`year`, `displ`, `cylinders`, `gears`): missing values filled with the median, then standardised.\n"
    "- **Binary** (`turbo`, `supercharger`): kept as 0/1.\n"
    "- **Categorical** (7 columns): one-hot encoding. Categories with fewer than 20 training cars "
    "(mostly rare makes) go into one 'infrequent' column, and an unknown category does not raise an error."
)
st.metric("Columns after preprocessing (final model)", n_columns)
