# Install: pip install streamlit pandas shap matplotlib plotly pyarrow

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
PARQUET_PATH = "data/predictions"

st.set_page_config(page_title="Exposure - Machine Learning Based Sensitive Device Prediction", layout="wide")


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    if Path(path).exists():
        df = pd.read_parquet(path)
        # PySpark hive-partitioned datasets encode the partition column (date=...)
        # as a string; convert it back to datetime if needed.
        if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])
        return df

    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    tenants = ["TenantA", "TenantB", "TenantC"]
    tenant_country = {"TenantA": "US", "TenantB": "FR", "TenantC": "FR"}
    features = ["login_failures", "geo_anomaly_score", "patch_lag_days",
                "failed_auths", "unusual_process_count", "data_exfil_bytes"]
    sensitive_set = {
        "TenantA": {f"TenantA_ap_{i:03d}" for i in range(1, 5)},
        "TenantB": {f"TenantB_ap_{i:03d}" for i in range(1, 4)},
        "TenantC": {f"TenantC_ap_{i:03d}" for i in range(1, 4)},
    }

    rows = []
    for tenant in tenants:
        for i in range(1, 13):
            device = f"{tenant}_ap_{i:03d}"
            is_sensitive = device in sensitive_set[tenant]
            for date in dates:
                score = float(rng.beta(30, 5) if is_sensitive else rng.beta(2, 20))
                shap_vals = rng.normal(0, 0.06, len(features))
                row = {
                    "date": date,
                    "tenant_id": tenant,
                    "country": tenant_country[tenant],
                    "ap_serial": device,
                    "prediction_score": score,
                    "shap_base_value": 0.15,
                }
                for feat, sv in zip(features, shap_vals):
                    row[f"shap_{feat}"] = float(sv)
                    row[feat] = float(abs(rng.normal(sv * 10, 1)))
                rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


df_all = load_data(PARQUET_PATH)

shap_cols = [c for c in df_all.columns if c.startswith("shap_") and c != "shap_base_value"]
feature_cols = [c.replace("shap_", "") for c in shap_cols]


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

        latest_date = df_base["date"].max()
        sensitive_latest = df_base[
            (df_base["date"] == latest_date) & (df_base["prediction_score"] >= threshold)
        ]["ap_serial"].nunique()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sensitive", sensitive_latest, help=f"Devices above threshold on {latest_date.date()}")
        col2.metric("Total Sensitive Devices", sensitive_devices)
        col3.metric("Total Non-Sensitive Devices", non_sensitive_devices)
        col4.metric("% Sensitive Devices", f"{pct_sensitive:.1f}%")

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
                       labels={"pct_sensitive": "% Sensitive Devices", "avg_score": "Avg Score"}),
                use_container_width=True,
            )
            st.markdown("Distribution of threat scores across tenants for devices above the threshold. The box shows the interquartile range; the line is the median. Wider spreads indicate more variability in risk within that tenant.")
            st.plotly_chart(
                px.box(df, x="tenant_id", y="prediction_score", color="tenant_id",
                       title="Score Distribution by Tenant",
                       labels={"tenant_id": "Tenant", "prediction_score": "Score"}),
                use_container_width=True,
            )
            st.markdown("Same score distribution broken down by country. Use this to spot whether threat activity is concentrated in a particular region.")
            st.plotly_chart(
                px.box(df, x="country", y="prediction_score", color="country",
                       title="Score Distribution by Country",
                       labels={"country": "Country", "prediction_score": "Score"}),
                use_container_width=True,
            )
            st.subheader("Top 20 Highest-Risk Devices")
            st.markdown("Devices with the highest individual prediction scores within the selected filters. Use this to prioritise which devices to investigate first.")
            st.dataframe(
                df.nlargest(20, "prediction_score")[["date", "tenant_id", "ap_serial", "prediction_score"]]
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
                    title=f"Score Over Time — {sel_device}"),
            use_container_width=True,
        )
        st.markdown("Full record for this device across the selected date range, including all raw feature values and SHAP contributions used by the model.")
        st.dataframe(dev_df.reset_index(drop=True), use_container_width=True)


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
                   labels={"x": "Mean |SHAP|", "y": "Feature"}),
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
            st.metric("Prediction Score", f"{row['prediction_score']:.4f}")
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
