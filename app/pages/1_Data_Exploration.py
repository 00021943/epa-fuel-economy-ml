import altair as alt
import pandas as pd
import streamlit as st

from utils import HYBRID_TECHS, TARGET, load_prepared_data

st.set_page_config(page_title="Data Exploration", layout="wide")
st.title("Data Exploration")

data, _ = load_prepared_data()
MAX_SCATTER_POINTS = 5000
CATEGORY_COLUMNS = ["drive_group", "vehicle_type", "trans_type", "fuel_tech", "fuelType1"]
NUMERIC_COLUMNS = ["year", "displ", "cylinders", "gears", TARGET]


def apply_filters(frame: pd.DataFrame) -> pd.DataFrame:
    """Sidebar filters; returns the filtered data."""
    st.sidebar.header("Filters")
    year_min, year_max = int(frame["year"].min()), int(frame["year"].max())
    years = st.sidebar.slider("Model year", year_min, year_max, (year_min, year_max))
    all_techs = sorted(frame["fuel_tech"].unique())
    all_types = sorted(frame["vehicle_type"].unique())
    techs = st.sidebar.multiselect("Fuel technology", all_techs, default=all_techs)
    types = st.sidebar.multiselect("Vehicle type", all_types, default=all_types)
    makes = st.sidebar.multiselect("Make (optional, empty = all)", sorted(frame["make"].unique()))
    mask = frame["year"].between(*years) & frame["fuel_tech"].isin(techs) & frame["vehicle_type"].isin(types)
    if makes:
        mask &= frame["make"].isin(makes)
    return frame[mask]


def show_kpis(frame: pd.DataFrame) -> None:
    cols = st.columns(4)
    cols[0].metric("Cars", f"{len(frame):,}")
    cols[1].metric("Median MPG", f"{frame[TARGET].median():.1f}")
    cols[2].metric("Mean engine size", f"{frame['displ'].mean():.2f} L")
    cols[3].metric("Hybrids and plug-in hybrids", f"{frame['fuel_tech'].isin(HYBRID_TECHS).mean():.1%}")


def histogram(frame: pd.DataFrame) -> None:
    chart = alt.Chart(frame[[TARGET]]).mark_bar().encode(
        alt.X(f"{TARGET}:Q", bin=alt.Bin(maxbins=50), title="Combined MPG"),
        alt.Y("count()", title="Cars"),
    )
    st.altair_chart(chart, width="stretch")
    st.caption("Most cars have a combined MPG between 15 and 30, with a long tail of very efficient hybrids.")


def mpg_by_year(frame: pd.DataFrame) -> None:
    split = st.checkbox("Split by fuel technology")
    keys = ["year", "fuel_tech"] if split else ["year"]
    grouped = frame.groupby(keys, as_index=False)[TARGET].mean()
    encodings = {"x": alt.X("year:O", title="Model year"), "y": alt.Y(f"{TARGET}:Q", title="Average MPG")}
    if split:
        encodings["color"] = alt.Color("fuel_tech:N", title="Fuel technology")
    st.altair_chart(alt.Chart(grouped).mark_line(point=True).encode(**encodings), width="stretch")
    st.caption("Average fuel economy stayed flat for years and has risen since about 2010.")


def engine_scatter(frame: pd.DataFrame) -> None:
    sample = frame.sample(min(MAX_SCATTER_POINTS, len(frame)), random_state=42)
    chart = alt.Chart(sample[["displ", TARGET, "fuel_tech", "make", "model", "year"]]).mark_circle(
        size=25, opacity=0.5).encode(
        alt.X("displ:Q", title="Engine size (litres)"),
        alt.Y(f"{TARGET}:Q", title="Combined MPG"),
        alt.Color("fuel_tech:N", title="Fuel technology"),
        tooltip=["make", "model", "year", "displ", TARGET],
    )
    st.altair_chart(chart, width="stretch")
    st.caption(f"Bigger engines use more fuel (sample of at most {MAX_SCATTER_POINTS:,} cars); "
               "hybrids reach higher MPG at the same engine size.")


def median_by_category(frame: pd.DataFrame) -> None:
    column = st.selectbox("Group by", CATEGORY_COLUMNS)
    grouped = frame.groupby(column, as_index=False)[TARGET].median()
    chart = alt.Chart(grouped).mark_bar().encode(
        alt.X(f"{TARGET}:Q", title="Median MPG"),
        alt.Y(f"{column}:N", sort="-x", title=None),
    )
    st.altair_chart(chart, width="stretch")
    st.caption("The median MPG differs clearly between groups, so these columns help the model.")


def correlation_heatmap(frame: pd.DataFrame) -> None:
    corr = frame[NUMERIC_COLUMNS].corr().stack().rename("correlation").reset_index()
    corr.columns = ["x", "y", "correlation"]
    base = alt.Chart(corr).encode(alt.X("x:N", title=None), alt.Y("y:N", title=None))
    heat = base.mark_rect().encode(alt.Color("correlation:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])))
    text = base.mark_text().encode(alt.Text("correlation:Q", format=".2f"))
    st.altair_chart(heat + text, width="stretch")
    st.caption("Engine size and cylinders are strongly negatively correlated with MPG, "
               "and strongly correlated with each other.")


filtered = apply_filters(data)
if filtered.empty:
    st.warning("No cars match the filters. Please widen the selection.")
    st.stop()

show_kpis(filtered)

left, right = st.columns(2)
with left:
    st.subheader("MPG distribution")
    histogram(filtered)
    st.subheader("Engine size vs MPG")
    engine_scatter(filtered)
with right:
    st.subheader("Average MPG by year")
    mpg_by_year(filtered)
    st.subheader("Median MPG by category")
    median_by_category(filtered)

st.subheader("Correlation of numeric columns")
correlation_heatmap(filtered)

st.subheader("Filtered data")
preview_columns = ["year", "make", "model", "displ", "cylinders", "trans_type", "drive_group",
                   "vehicle_type", "fuel_tech", TARGET]
st.dataframe(filtered[preview_columns].head(500), width="stretch")
st.download_button("Download filtered data (CSV)", filtered[preview_columns].to_csv(index=False),
                   file_name="vehicles_filtered.csv", mime="text/csv")
