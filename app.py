import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os
from pathlib import Path
from src.artifacts import ensure_artifact

# ============================
# Config
# ============================
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")

PROJECT_ROOT = Path(__file__).resolve().parent
HOLDOUT_ENGINEERED_PATH = ensure_artifact(
    "data/processed/feature_engineered_holdout.csv", PROJECT_ROOT
)
HOLDOUT_META_PATH = ensure_artifact(
    "data/processed/cleaning_holdout.csv", PROJECT_ROOT
)


def prediction_payload(df: pd.DataFrame) -> list[dict]:
    """Convert dashboard rows into strict JSON-safe prediction records."""
    sanitized = df.replace([float("inf"), float("-inf")], pd.NA)
    sanitized = sanitized.astype(object).where(pd.notna(sanitized), None)
    return sanitized.to_dict(orient="records")


# ============================
# Data loading
# ============================
@st.cache_data
def load_data():
    fe = pd.read_csv(HOLDOUT_ENGINEERED_PATH)
    meta = pd.read_csv(HOLDOUT_META_PATH, parse_dates=["date"])[["date", "city_full"]]

    if len(fe) != len(meta):
        st.warning(
            ":material/warning: Engineered and metadata holdout lengths differ. "
            "Aligning by index."
        )
        min_len = min(len(fe), len(meta))
        fe = fe.iloc[:min_len].copy()
        meta = meta.iloc[:min_len].copy()

    disp = pd.DataFrame(index=fe.index)
    disp["date"] = meta["date"]
    disp["region"] = meta["city_full"]
    disp["year"] = disp["date"].dt.year
    disp["month"] = disp["date"].dt.month
    disp["actual_price"] = fe["price"]

    return fe, disp

fe_df, disp_df = load_data()

# ============================
# UI
# ============================
st.title(":material/real_estate_agent: Housing Price Prediction — Holdout Explorer")

years = sorted(disp_df["year"].unique())
months = list(range(1, 13))
regions = ["All"] + sorted(disp_df["region"].dropna().unique())

col1, col2, col3 = st.columns(3)
with col1:
    year = st.selectbox(":material/calendar_month: Select Year", years, index=0)
with col2:
    month = st.selectbox(":material/event: Select Month", months, index=0)
with col3:
    region = st.selectbox(":material/location_on: Select Region", regions, index=0)

if st.button("Show Predictions", icon=":material/rocket_launch:"):
    mask = (disp_df["year"] == year) & (disp_df["month"] == month)
    if region != "All":
        mask &= (disp_df["region"] == region)

    idx = disp_df.index[mask]

    if len(idx) == 0:
        st.warning(":material/search_off: No data found for these filters.")
    else:
        st.write(
            f":material/calendar_month: Running predictions for "
            f"**{year}-{month:02d}** | Region: **{region}**"
        )

        payload = prediction_payload(fe_df.loc[idx])

        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            out = resp.json()
            preds = out.get("predictions", [])
            actuals = out.get("actuals", None)

            view = disp_df.loc[idx, ["date", "region", "actual_price"]].copy()
            view = view.sort_values("date")
            view["prediction"] = pd.Series(preds, index=view.index).astype(float)

            if actuals is not None and len(actuals) == len(view):
                view["actual_price"] = pd.Series(actuals, index=view.index).astype(float)

            # Metrics
            mae = (view["prediction"] - view["actual_price"]).abs().mean()
            rmse = ((view["prediction"] - view["actual_price"]) ** 2).mean() ** 0.5
            avg_pct_error = ((view["prediction"] - view["actual_price"]).abs() / view["actual_price"]).mean() * 100

            st.subheader(":material/compare_arrows: Predictions vs Actuals")
            st.dataframe(
                view[["date", "region", "actual_price", "prediction"]].reset_index(drop=True),
                use_container_width=True
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("MAE", f"{mae:,.0f}")
            with c2:
                st.metric("RMSE", f"{rmse:,.0f}")
            with c3:
                st.metric("Avg % Error", f"{avg_pct_error:.2f}%")

            # ============================
            # Yearly Trend Chart
            # ============================
            if region == "All":
                yearly_data = disp_df[disp_df["year"] == year].copy()
                idx_all = yearly_data.index
                payload_all = prediction_payload(fe_df.loc[idx_all])

                resp_all = requests.post(API_URL, json=payload_all, timeout=60)
                resp_all.raise_for_status()
                preds_all = resp_all.json().get("predictions", [])

                yearly_data["prediction"] = pd.Series(preds_all, index=yearly_data.index).astype(float)

            else:
                yearly_data = disp_df[(disp_df["year"] == year) & (disp_df["region"] == region)].copy()
                idx_region = yearly_data.index
                payload_region = prediction_payload(fe_df.loc[idx_region])

                resp_region = requests.post(API_URL, json=payload_region, timeout=60)
                resp_region.raise_for_status()
                preds_region = resp_region.json().get("predictions", [])

                yearly_data["prediction"] = pd.Series(preds_region, index=yearly_data.index).astype(float)

            # Aggregate by month
            monthly_avg = yearly_data.groupby("month")[["actual_price", "prediction"]].mean().reset_index()

            # Highlight selected month
            monthly_avg["highlight"] = monthly_avg["month"].apply(lambda m: "Selected" if m == month else "Other")

            fig = px.line(
                monthly_avg,
                x="month",
                y=["actual_price", "prediction"],
                markers=True,
                labels={"value": "Price", "month": "Month"},
                title=f"Yearly Trend — {year}{'' if region=='All' else f' — {region}'}"
            )

            # Add highlight with background shading
            highlight_month = month
            fig.add_vrect(
                x0=highlight_month - 0.5,
                x1=highlight_month + 0.5,
                fillcolor="red",
                opacity=0.1,
                layer="below",
                line_width=0,
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f":material/error: API call failed: {e}")
            st.exception(e)

else:
    st.info(
        ":material/tune: Choose filters and click **Show Predictions** to compute."
    )
