import streamlit as st

from utils import load_evaluation, load_prepared_data

st.set_page_config(page_title="Vehicle Fuel Economy", layout="wide")

evaluation = load_evaluation()
data, _ = load_prepared_data()
final = evaluation["final_model"]
test = evaluation["test"][final]

st.title("Vehicle Fuel Economy")
st.subheader("Can we predict the combined fuel economy (MPG) of a car with a combustion engine "
             "from its technical characteristics?")
st.write("The model uses engine size, number of cylinders, transmission, drive type, fuel type and "
         "model year to estimate the EPA combined MPG (`comb08`). It was trained on the model years "
         f"{evaluation['train_years'][0]}-{evaluation['train_years'][1]} and tested once on the newest years "
         f"{evaluation['test_years'][0]}-{evaluation['test_years'][1]}. The model used in this app is the same "
         "model refitted on all years, so it also knows the newest cars.")

# Key numbers, all taken from the saved evaluation (two rows, so the values are not cut off)
row1 = st.columns(3)
row1[0].metric("Cars modelled", f"{len(data):,}")
row1[1].metric("Training years", f"{evaluation['train_years'][0]}-{evaluation['train_years'][1]}")
row1[2].metric("Test years", f"{evaluation['test_years'][0]}-{evaluation['test_years'][1]}")
row2 = st.columns(3)
row2[0].metric("Final model", final)
row2[1].metric("Test MAE", f"{test['MAE']:.2f} MPG")
row2[2].metric("Test R²", f"{test['R2']:.3f}")
st.caption(f"For new or changed cars (not found in the older data) the error is larger: "
           f"MAE {evaluation['lookup']['model_mae_not_matched']:.2f} MPG.")

st.subheader("How to use the app")
st.markdown(
    "- **Data Exploration**: an interactive dashboard with filters and charts of the data.\n"
    "- **Preprocessing**: how the raw data became the modelling data, step by step.\n"
    "- **Model Training**: the models, their tuning results, and a form to train a model yourself.\n"
    "- **Prediction**: enter the details of a car and get its estimated MPG.\n"
    "- **Evaluation**: how well the final model works on the test years 2024-2027."
)

st.warning("The results are **estimates of the EPA label, not official EPA values**. "
           "The app covers combustion-engine cars only (petrol, diesel, natural gas and hybrids); "
           "electric and hydrogen cars are not included.")

st.caption("Data source: [fueleconomy.gov](https://www.fueleconomy.gov/feg/ws/index.shtml) "
           "(U.S. EPA and Department of Energy, public domain). Student ID 00021943.")
