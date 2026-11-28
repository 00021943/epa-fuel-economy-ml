import altair as alt
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

from utils import FEATURES, TARGET, load_evaluation, load_prepared_data, make_preprocessor, table_to_frame

st.set_page_config(page_title="Model Training", layout="wide")
st.title("Model Training")

evaluation = load_evaluation()
data, _ = load_prepared_data()
MAX_TRAIN_ROWS = 15_000
TRAIN_LAST_YEAR = 2020
VALID_YEARS = (2021, 2023)

st.header("Models")
st.markdown(
    "- **Baseline**: always predicts the mean MPG of the training data. Every real model must beat it.\n"
    "- **Ridge regression**: a linear model with a penalty (`alpha`) that keeps the coefficients stable. "
    "Fast and easy to explain, but it cannot learn curved effects such as engine size vs MPG.\n"
    "- **Random Forest**: many deep decision trees trained on random samples, and their predictions are "
    "averaged. It learns non-linear effects and interactions on its own.\n"
    "- **Gradient Boosting** (`HistGradientBoostingRegressor`): small trees built one after another, each "
    "fixing the errors of the previous ones. Fast on this data size."
)

st.header("Cross-validation results")
cv_table = table_to_frame(evaluation["cv"])
st.dataframe(cv_table.round(3), width="stretch")
cv_long = cv_table["CV MAE"].rename("CV MAE").reset_index().rename(columns={"index": "Model"})
st.altair_chart(alt.Chart(cv_long).mark_bar().encode(
    alt.X("Model:N", sort=None, title=None), alt.Y("CV MAE:Q", title="CV MAE (MPG)")), width="stretch")
st.caption("Both tree models have a much lower error than Ridge, and all three models are far better than "
           "the baseline (5-fold cross-validation, grouped by base model).")

st.header("Tuning results")
ridge = pd.DataFrame(evaluation["ridge_search"])
col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Ridge: alpha vs CV MAE")
    st.altair_chart(alt.Chart(ridge).mark_line(point=True).encode(
        alt.X("alpha:Q", scale=alt.Scale(type="symlog"), title="alpha"),
        alt.Y("cv_mae:Q", scale=alt.Scale(zero=False), title="CV MAE (MPG)")), width="stretch")
    st.caption("The penalty hardly matters until it becomes very strong.")
with col_right:
    st.subheader("Random Forest grid")
    st.dataframe(pd.DataFrame(evaluation["rf_search"]).sort_values("CV MAE").round(3),
                 hide_index=True, width="stretch", height=320)
st.subheader("Gradient Boosting grid")
st.dataframe(pd.DataFrame(evaluation["hgb_search"]).sort_values("CV MAE").round(3),
             hide_index=True, width="stretch", height=320)

st.header("Validation on 2021-2023")
valid = table_to_frame(evaluation["validation_2021_2023"])
st.dataframe(valid.round(3), width="stretch")
per_year = (table_to_frame(evaluation["validation_mae_per_year"]).rename_axis("Model")
            .reset_index().melt("Model", var_name="Year", value_name="MAE"))
st.altair_chart(alt.Chart(per_year).mark_line(point=True).encode(
    alt.X("Year:O"), alt.Y("MAE:Q", title="MAE (MPG)"), alt.Color("Model:N")), width="stretch")
st.caption("For the three trained models the error grows for newer years, because they have to predict cars "
           "further from the training data.")
st.info(f"**Final-model rule:** the model with the lowest MAE on 2021-2023 is chosen. "
        f"That is **{evaluation['final_model']}** "
        f"(MAE {valid.loc[evaluation['final_model'], 'MAE']:.3f} MPG).")


def train_and_score(name: str, params: dict) -> dict:
    """Train on a sample of 1984-2020 and score on 2021-2023."""
    train = data[data["year"] <= TRAIN_LAST_YEAR]
    train = train.sample(min(MAX_TRAIN_ROWS, len(train)), random_state=42)
    valid_data = data[data["year"].between(*VALID_YEARS)]
    estimators = {
        "Ridge": lambda: Ridge(alpha=params["alpha"]),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=42, n_jobs=-1),
        "Gradient Boosting": lambda: HistGradientBoostingRegressor(
            max_iter=params["max_iter"], learning_rate=params["learning_rate"],
            early_stopping=False, random_state=42),
    }
    pipe = Pipeline([("prep", make_preprocessor()), ("model", estimators[name]())])
    pipe.fit(train[FEATURES], train[TARGET])
    pred = pipe.predict(valid_data[FEATURES])
    truth = valid_data[TARGET]
    return {"MAE": mean_absolute_error(truth, pred), "RMSE": root_mean_squared_error(truth, pred),
            "R2": r2_score(truth, pred)}


st.header("Train a model yourself")
st.write(f"The model is trained on at most {MAX_TRAIN_ROWS:,} cars from 1984-{TRAIN_LAST_YEAR} and "
         f"validated on all cars from {VALID_YEARS[0]}-{VALID_YEARS[1]}.")
with st.form("train_form"):
    name = st.selectbox("Model", ["Ridge", "Random Forest", "Gradient Boosting"])
    c1, c2 = st.columns(2)
    alpha = c1.number_input("Ridge: alpha", 0.0001, 1000.0, 10.0, format="%.4f")
    n_estimators = c1.slider("Random Forest: number of trees", 10, 100, 50, step=10)
    max_depth = c2.slider("Random Forest: max depth", 5, 30, 20)
    max_iter = c1.slider("Gradient Boosting: iterations", 50, 300, 150, step=50)
    learning_rate = c2.select_slider("Gradient Boosting: learning rate", [0.02, 0.05, 0.1, 0.2], value=0.1)
    submitted = st.form_submit_button("Train")

if submitted:
    params = {"alpha": alpha, "n_estimators": n_estimators, "max_depth": max_depth,
              "max_iter": max_iter, "learning_rate": learning_rate}
    with st.spinner("Training ..."):
        mine = train_and_score(name, params)
    tuned = evaluation["validation_2021_2023"][name]
    result = pd.DataFrame({
        "Your model": [mine["MAE"], mine["RMSE"], mine["R2"]],
        "Notebook (tuned, full data)": [tuned["MAE"], tuned["RMSE"], tuned["R2"]],
    }, index=["MAE", "RMSE", "R2"])
    st.dataframe(result.round(3), width="stretch")
    st.caption("The notebook models use all training years up to 2023 and tuned settings, so they are usually better.")
