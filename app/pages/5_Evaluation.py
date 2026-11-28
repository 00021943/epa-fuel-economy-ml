import altair as alt
import pandas as pd
import streamlit as st

from utils import load_evaluation, table_to_frame

st.set_page_config(page_title="Evaluation", layout="wide")
st.title("Evaluation")

ev = load_evaluation()
final = ev["final_model"]
st.write(f"The models were tested once on the newest cars: model years {ev['test_years'][0]}-{ev['test_years'][1]} "
         f"({ev['n_test']:,} cars), after training on {ev['train_years'][0]}-{ev['train_years'][1]}.")

st.header("Test metrics")
metrics = table_to_frame(ev["test"]).rename_axis("Model")
metrics.index = [f"{m} (final)" if m == final else m for m in metrics.index]
st.dataframe(metrics.round(3), width="stretch")
st.caption("The final model keeps an R² of about 0.94 on future years; Ridge is clearly worse.")

st.header("CV vs validation vs test MAE")
mae = pd.DataFrame({
    "CV": table_to_frame(ev["cv"])["CV MAE"],
    "2021-2023": table_to_frame(ev["validation_2021_2023"])["MAE"],
    "Test": table_to_frame(ev["test"])["MAE"],
}).rename_axis("Model").reset_index().melt("Model", var_name="Set", value_name="MAE")
st.altair_chart(alt.Chart(mae).mark_bar().encode(
    alt.X("Model:N", sort=None, title=None), alt.Y("MAE:Q", title="MAE (MPG)"),
    alt.Color("Set:N"), alt.XOffset("Set:N")), width="stretch")
st.caption("For every model the test error is higher than the cross-validation error, because the test cars are "
           "newer than all training cars. The 2021-2023 check is much closer to the test error than CV.")

left, right = st.columns(2)
with left:
    st.header("Test MAE per year")
    per_year = (table_to_frame(ev["test_mae_per_year"]).rename_axis("Model").reset_index()
                .melt("Model", var_name="Year", value_name="MAE"))
    st.altair_chart(alt.Chart(per_year).mark_line(point=True).encode(
        alt.X("Year:O"), alt.Y("MAE:Q", title="MAE (MPG)"), alt.Color("Model:N")), width="stretch")
    st.caption("The error of the tree models grows with every year after the training data, "
               "because trees treat a new year like the last training year.")
with right:
    st.header("Test MAE per fuel technology")
    per_tech = (table_to_frame(ev["test_mae_per_fuel_tech"]).rename_axis("Model").reset_index()
                .melt("Model", var_name="Fuel technology", value_name="MAE"))
    st.altair_chart(alt.Chart(per_tech).mark_bar().encode(
        alt.X("Fuel technology:N", title=None), alt.Y("MAE:Q", title="MAE (MPG)"),
        alt.Color("Model:N"), alt.XOffset("Model:N")), width="stretch")
    st.caption("Hybrids and plug-in hybrids have the largest errors of the final model.")

st.header("Predicted vs true MPG")
pred = pd.DataFrame(ev["test_predictions"])
pred["group"] = pred["fuel_tech"].isin(["Hybrid", "Plug-in Hybrid"]).map({True: "Hybrid / plug-in", False: "Other"})
limit = float(max(pred["true"].max(), pred["predicted"].max()))
points = alt.Chart(pred).mark_circle(size=20, opacity=0.5).encode(
    alt.X("true:Q", title="True MPG"), alt.Y("predicted:Q", title="Predicted MPG"),
    alt.Color("group:N", title="Cars"), tooltip=["year", "fuel_tech", "true", "predicted"])
diagonal = alt.Chart(pd.DataFrame({"x": [0, limit], "y": [0, limit]})).mark_line(color="gray", strokeDash=[4, 4]).encode(
    x="x:Q", y="y:Q")
st.altair_chart(points + diagonal, width="stretch")
st.caption("Points close to the dashed diagonal are good predictions; the largest misses are hybrids.")

st.header("Random Forest vs Gradient Boosting")
boot = ev["bootstrap_rf_minus_gb"]
includes_zero = boot["low"] <= 0 <= boot["high"]
st.write(f"Bootstrap over {boot['base_models']} base models, difference in MAE (RF minus GB): "
         f"mean {boot['mean']:.3f} MPG, 95% interval [{boot['low']:.3f}, {boot['high']:.3f}].")
st.caption("The interval includes 0, so the two models are not clearly different on the test set."
           if includes_zero else "The interval does not include 0, so the difference is significant.")

st.header("Look-up compared with the model")
lk = ev["lookup"]
cols = st.columns(4)
cols[0].metric("Cars found in older data", f"{lk['matched']:,}")
cols[1].metric("Look-up MAE (found)", f"{lk['lookup_mae_matched']:.2f}")
cols[2].metric("Model MAE (found)", f"{lk['model_mae_matched']:.2f}")
cols[3].metric("Model MAE (not found)", f"{lk['model_mae_not_matched']:.2f}")
st.caption(f"For {lk['matched']:,} test cars an identical older car exists and its MPG is a better answer than the model; "
           f"the other {lk['not_matched']:,} cars need the model. Combined MAE: {lk['combined_mae_all']:.2f} "
           f"(model alone: {lk['model_mae_all']:.2f}).")

st.header("Permutation importance")
imp = pd.Series(ev["permutation_importance"], name="MAE increase").rename_axis("Feature").reset_index()
st.altair_chart(alt.Chart(imp).mark_bar().encode(
    alt.X("MAE increase:Q", title="Increase in MAE when shuffled (MPG)"),
    alt.Y("Feature:N", sort="-x", title=None)), width="stretch")
st.caption("Engine size and fuel technology matter most. Year is exactly 0: every test car is newer than "
           "all training cars, so shuffling the year changes no tree prediction.")
