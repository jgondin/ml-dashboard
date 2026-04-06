# Install: pip install streamlit pandas shap matplotlib plotly pyarrow xgboost scikit-learn

import pickle
import matplotlib
matplotlib.use("Agg")  # non-interactive backend required for Streamlit

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PARQUET_PATH   = "data/predictions"
EXPLAINER_PATH = "models/shap_explainer.pkl"
MODEL_PATH     = "models/xgboost_model.json"

FEATURE_COLS = [
    "login_failures",
    "geo_anomaly_score",
    "patch_lag_days",
    "failed_auths",
    "unusual_process_count",
    "data_exfil_bytes",
]

st.set_page_config(page_title="Exposure - Machine Learning Based Sensitive Device Prediction", layout="wide")


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str, explainer_path: str):
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    with open(explainer_path, "rb") as f:
        explainer = pickle.load(f)
    return model, explainer


@st.cache_data
def load_data(data_path: str, model_path: str, explainer_path: str) -> pd.DataFrame:
    # ── Load inference parquet ────────────────────────────────────────────────
    df = pd.read_parquet(data_path)
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # ── Score with model + compute SHAP ───────────────────────────────────────
    model, explainer = load_model(model_path, explainer_path)
    X = df[FEATURE_COLS].values.astype(float)

    df["prediction_score"] = model.predict_proba(X)[:, 1]

    shap_values = explainer.shap_values(X)
    # shap_values is a list [class0, class1] for binary classifiers
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    df["shap_base_value"] = float(explainer.expected_value[1]
                                  if isinstance(explainer.expected_value, np.ndarray)
                                  else explainer.expected_value)
    for i, feat in enumerate(FEATURE_COLS):
        df[f"shap_{feat}"] = sv[:, i]

    return df


if not Path(EXPLAINER_PATH).exists() or not Path(MODEL_PATH).exists():
    st.error(
        "No trained model found. Run `python train.py` first to train and save the model."
    )
    st.stop()

df_all = load_data(PARQUET_PATH, MODEL_PATH, EXPLAINER_PATH)

shap_cols    = [f"shap_{f}" for f in FEATURE_COLS]
feature_cols = FEATURE_COLS


# ── Sidebar Filters ───────────────────────────────────────────────────────────
st.sidebar.title("Filters")

min_d, max_d = df_all["date"].min().date(), df_all["date"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
tenants = st.sidebar.multiselect("Tenant", sorted(df_all["tenant_id"].unique()),
                                  default=sorted(df_all["tenant_id"].unique()))
countries = st.sidebar.multiselect("Country", sorted(df_all["country"].unique()),
                                    default=sorted(df_all["country"].unique()))
threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.5, 0.01)

start, end = (date_range[0], date_range[1]) if len(date_range) == 2 else (date_range[0], date_range[0])
df_base = df_all[
    (df_all["date"].dt.date >= start) & (df_all["date"].dt.date <= end) &
    (df_all["tenant_id"].isin(tenants)) &
    (df_all["country"].isin(countries))
].copy()

df = df_base[df_base["prediction_score"] >= threshold].copy()

st.sidebar.metric("Filtered records", len(df_base))

# ── Main Layout ───────────────────────────────────────────────────────────────
st.title("Exposure - Machine Learning Based Sensitive Device Prediction")
tab_overview, tab_device, tab_shap_global, tab_shap_device = st.tabs(
    ["Overview", "Device Drill-down", "SHAP Global", "SHAP per Device"]
)


# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab_overview:
    if df_base.empty:
        st.warning("No data matches current filters.")
    else:
        sensitive_aps = set(df_base.loc[df_base["prediction_score"] >= threshold, "ap_serial"])
        all_aps = set(df_base["ap_serial"])
        total_devices = len(all_aps)
        sensitive_devices = len(sensitive_aps)
        non_sensitive_devices = len(all_aps - sensitive_aps)
        pct_sensitive = (sensitive_devices / total_devices * 100) if total_devices > 0 else 0.0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Devices",         total_devices)
        col2.metric("Sensitive Devices",     sensitive_devices)
        col3.metric("Non-Sensitive Devices", non_sensitive_devices)
        col4.metric("% Sensitive",           f"{pct_sensitive:.1f}%")

        if df.empty:
            st.info("No devices above threshold — adjust the slider to see charts.")
        else:
            total_devices_per_day = df_base.groupby("date")["ap_serial"].nunique()
            daily = (df.groupby("date")
                       .agg(sensitive_count=("ap_serial", "nunique"), avg_score=("prediction_score", "mean"))
                       .reset_index())
            daily["pct_sensitive"] = (daily["sensitive_count"] / daily["date"].map(total_devices_per_day) * 100).round(1)
            st.markdown("Percentage of devices predicted as sensitive each day. Bar color reflects the average threat score for that day — darker red means higher average risk.")
            st.plotly_chart(
                px.bar(daily, x="date", y="pct_sensitive", color="avg_score",
                       color_continuous_scale="RdYlGn_r",
                       title="% Devices Predicted as Sensitive per Day",
                       labels={"pct_sensitive": "% Sensitive Devices", "avg_score": "Avg Sensitivity Score"}),
                use_container_width=True,
            )
            st.markdown("Distribution of threat scores across tenants for devices above the threshold. The box shows the interquartile range; the line is the median. Wider spreads indicate more variability in risk within that tenant.")
            st.plotly_chart(
                px.box(df, x="tenant_id", y="prediction_score", color="tenant_id",
                       title="Sensitivity Score Distribution by Tenant (Sensitive Devices Only)",
                       labels={"tenant_id": "Tenant", "prediction_score": "Sensitivity Score"}),
                use_container_width=True,
            )
            st.markdown("Same score distribution broken down by country. Use this to spot whether threat activity is concentrated in a particular region.")
            st.plotly_chart(
                px.box(df, x="country", y="prediction_score", color="country",
                       title="Sensitivity Score Distribution by Country (Sensitive Devices Only)",
                       labels={"country": "Country", "prediction_score": "Sensitivity Score"}),
                use_container_width=True,
            )
            st.subheader("Top 20 Highest-Risk Devices")
            st.markdown("Devices with the highest individual prediction scores within the selected filters. Use this to prioritise which devices to investigate first.")
            st.dataframe(
                df.nlargest(20, "prediction_score")[["date", "tenant_id", "ap_serial", "prediction_score"]]
                  .rename(columns={"date": "Date", "tenant_id": "Tenant",
                                   "ap_serial": "AP Serial", "prediction_score": "Sensitivity Score"})
                  .reset_index(drop=True),
                use_container_width=True,
            )


# ── Tab 2: Device Drill-down ──────────────────────────────────────────────────
with tab_device:
    devices = sorted(df["ap_serial"].unique()) if not df.empty else []
    if not devices:
        st.warning("No devices match current filters.")
    else:
        sel_device = st.selectbox("Select Device", devices, key="dd_dev")
        dev_df = df[df["ap_serial"] == sel_device].sort_values("date")
        st.markdown("Track how the sensitivity score for a single device evolves over time. A rising trend may indicate increasing exposure or deteriorating security posture.")
        st.plotly_chart(
            px.line(dev_df, x="date", y="prediction_score", markers=True,
                    title=f"Sensitivity Score Over Time — {sel_device}",
                    labels={"date": "Date", "prediction_score": "Sensitivity Score"}),
            use_container_width=True,
        )
        st.markdown("Full record for this device across the selected date range, including all raw feature values and SHAP contributions used by the model.")
        st.dataframe(
            dev_df.rename(columns={"date": "Date", "tenant_id": "Tenant",
                                   "ap_serial": "AP Serial", "prediction_score": "Sensitivity Score"})
                  .reset_index(drop=True),
            use_container_width=True,
        )


# ── Tab 3: SHAP Global ────────────────────────────────────────────────────────
with tab_shap_global:
    if len(df) < 2:
        st.warning("Need at least 2 records for global SHAP plots.")
    else:
        mean_shap = df[shap_cols].abs().mean()
        mean_shap.index = feature_cols
        mean_shap = mean_shap.sort_values()

        st.markdown("Average absolute SHAP value per feature across all sensitive devices. Longer bars indicate features that have the most influence on the model's predictions — these are the key drivers of exposure risk.")
        st.plotly_chart(
            px.bar(x=mean_shap.values, y=mean_shap.index, orientation="h",
                   title="Mean |SHAP| — Feature Importance",
                   labels={"x": "Mean |SHAP Value|", "y": "Feature"}),
            use_container_width=True,
        )

        st.subheader("SHAP Beeswarm Plot")
        st.markdown("Each dot represents one device-day observation. Dots to the right (positive SHAP) push the score higher; dots to the left lower it. Color shows the feature value — red means high, blue means low. This reveals both the direction and magnitude of each feature's impact across the entire dataset.")
        explanation = shap.Explanation(
            values=df[shap_cols].values.astype(float),
            base_values=df["shap_base_value"].values.astype(float),
            data=df[feature_cols].values.astype(float),
            feature_names=feature_cols,
        )
        shap.plots.beeswarm(explanation, show=False, max_display=len(feature_cols))
        fig_bee = plt.gcf()
        st.pyplot(fig_bee, bbox_inches="tight")
        plt.close(fig_bee)

        st.subheader("SHAP Violin Plot")
        st.markdown("Shows the distribution of SHAP values per feature as violins. A wide violin means the feature's impact varies greatly across devices; a narrow one means it contributes consistently. Color indicates the feature value — red means high, blue means low.")
        shap.summary_plot(
            df[shap_cols].values.astype(float),
            df[feature_cols].values.astype(float),
            feature_names=feature_cols,
            plot_type="violin",
            show=False,
        )
        fig_violin = plt.gcf()
        st.pyplot(fig_violin, bbox_inches="tight")
        plt.close(fig_violin)


# ── Tab 4: SHAP per Device ────────────────────────────────────────────────────
with tab_shap_device:
    devices4 = sorted(df["ap_serial"].unique()) if not df.empty else []
    if not devices4:
        st.warning("No devices match current filters.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            sel_dev4 = st.selectbox("Select Device", devices4, key="shap_dev")
        with col2:
            dev_dates = sorted(df[df["ap_serial"] == sel_dev4]["date"].dt.date.unique())
            sel_date4 = st.selectbox("Select Date", dev_dates, key="shap_date")

        row_df = df[(df["ap_serial"] == sel_dev4) & (df["date"].dt.date == sel_date4)]
        if row_df.empty:
            st.warning("No record found for this device/date combination.")
        else:
            row = row_df.iloc[0]
            st.metric("Sensitivity Score", f"{row['prediction_score']:.4f}")
            st.markdown("This waterfall chart breaks down exactly why the model assigned this score to the selected device on this date. Each bar shows how much a specific feature pushed the score up (red) or down (blue) from the baseline. The final value on the right is the resulting prediction score.")
            explanation_single = shap.Explanation(
                values=row[shap_cols].values.astype(float),
                base_values=float(row["shap_base_value"]),
                data=row[feature_cols].values.astype(float),
                feature_names=feature_cols,
            )
            shap.plots.waterfall(explanation_single, show=False)
            fig_wf = plt.gcf()
            st.pyplot(fig_wf, bbox_inches="tight")
            plt.close(fig_wf)
