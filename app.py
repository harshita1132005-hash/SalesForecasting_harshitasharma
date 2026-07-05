"""
Sales Forecasting & Demand Intelligence Dashboard
Superstore Sales — Internship Final Project (Task 7)

Run locally with:  streamlit run app.py
Deploy free at:     https://share.streamlit.io  (Streamlit Community Cloud)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBRegressor

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# Data loading (cached so it only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_superstore.csv", parse_dates=["Order Date"])
    monthly = df.set_index("Order Date")["Sales"].resample("MS").sum()
    monthly.index.freq = "MS"
    model_comparison = pd.read_csv("model_comparison.csv")
    anomaly_df = pd.read_csv("anomaly_results.csv", parse_dates=["Week"])
    cluster_df = pd.read_csv("cluster_results.csv")
    return df, monthly, model_comparison, anomaly_df, cluster_df

df, monthly, model_comparison, anomaly_df, cluster_df = load_data()


def season_num(mo):
    if mo in [12, 1, 2]:
        return 0
    if mo in [3, 4, 5]:
        return 1
    if mo in [6, 7, 8]:
        return 2
    return 3


@st.cache_data
def xgb_forecast(series_values, _series_index, steps):
    series = pd.Series(series_values, index=pd.to_datetime(_series_index))
    d = series.reset_index()
    d.columns = ["Date", "Sales"]
    d["Lag1"] = d["Sales"].shift(1)
    d["Lag2"] = d["Sales"].shift(2)
    d["Lag3"] = d["Sales"].shift(3)
    d["RollingMean3"] = d["Sales"].shift(1).rolling(3).mean()
    d["Month"] = d["Date"].dt.month
    d["Quarter"] = d["Date"].dt.quarter
    d["Season"] = d["Month"].apply(season_num)
    d = d.dropna().reset_index(drop=True)
    feats = ["Lag1", "Lag2", "Lag3", "RollingMean3", "Month", "Quarter", "Season"]

    # Backtest on last 3 months for MAE/RMSE display
    train_d, test_d = d.iloc[:-3], d.iloc[-3:]
    mdl_bt = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    mdl_bt.fit(train_d[feats], train_d["Sales"])
    bt_preds = mdl_bt.predict(test_d[feats])
    mae = np.mean(np.abs(test_d["Sales"].values - bt_preds))
    rmse = np.sqrt(np.mean((test_d["Sales"].values - bt_preds) ** 2))

    # Full-data model for genuine future forecast
    mdl = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    mdl.fit(d[feats], d["Sales"])

    hist = list(series.values)
    last_date = series.index[-1]
    preds, future_dates = [], []
    for i in range(steps):
        nd = last_date + pd.DateOffset(months=i + 1)
        lag1, lag2, lag3 = hist[-1], hist[-2], hist[-3]
        roll3 = np.mean(hist[-3:])
        mo = nd.month
        q = (mo - 1) // 3 + 1
        sea = season_num(mo)
        Xp = pd.DataFrame([[lag1, lag2, lag3, roll3, mo, q, sea]], columns=feats)
        p = mdl.predict(Xp)[0]
        preds.append(p)
        hist.append(p)
        future_dates.append(nd)

    return future_dates, preds, mae, rmse


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("📊 Sales Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Demand Segments"],
)

# ---------------------------------------------------------------------------
# PAGE 1 — Sales Overview
# ---------------------------------------------------------------------------
if page == "Sales Overview":
    st.title("Sales Overview Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        yearly = df.groupby("Year")["Sales"].sum().reset_index()
        fig = px.bar(yearly, x="Year", y="Sales", title="Total Sales by Year",
                     color_discrete_sequence=["#2563eb"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        m = monthly.reset_index()
        m.columns = ["Date", "Sales"]
        fig = px.line(m, x="Date", y="Sales", title="Monthly Sales Trend", markers=True)
        fig.update_traces(line_color="#16a34a")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sales by Region & Category")
    c1, c2 = st.columns(2)
    with c1:
        regions = st.multiselect("Filter Region(s)", df["Region"].unique(),
                                  default=list(df["Region"].unique()))
    with c2:
        cats = st.multiselect("Filter Category(ies)", df["Category"].unique(),
                               default=list(df["Category"].unique()))

    filtered = df[df["Region"].isin(regions) & df["Category"].isin(cats)]
    grouped = filtered.groupby(["Region", "Category"])["Sales"].sum().reset_index()
    fig = px.bar(grouped, x="Region", y="Sales", color="Category", barmode="group",
                 title="Sales by Region & Category")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE 2 — Forecast Explorer
# ---------------------------------------------------------------------------
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")
    st.caption("Forecasts generated with XGBoost — the best-performing model from Task 3's comparison.")

    col1, col2 = st.columns(2)
    with col1:
        dim_type = st.selectbox("Select dimension", ["Category", "Region"])
    with col2:
        if dim_type == "Category":
            options = sorted(df["Category"].unique())
        else:
            options = sorted(df["Region"].unique())
        dim_value = st.selectbox(f"Select {dim_type}", options)

    horizon = st.slider("Forecast horizon (months ahead)", 1, 3, 3)

    sub_df = df[df[dim_type] == dim_value]
    series = sub_df.set_index("Order Date")["Sales"].resample("MS").sum()

    with st.spinner("Training XGBoost model and generating forecast..."):
        future_dates, preds, mae, rmse = xgb_forecast(series.values, series.index, steps=3)

    preds = preds[:horizon]
    future_dates = future_dates[:horizon]

    hist_recent = series.iloc[-12:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_recent.index, y=hist_recent.values,
                              mode="lines+markers", name="Historical", line=dict(color="#2563eb")))
    fig.add_trace(go.Scatter(x=[hist_recent.index[-1]] + future_dates,
                              y=[hist_recent.values[-1]] + preds,
                              mode="lines+markers", name="Forecast", line=dict(color="#dc2626", dash="dash")))
    fig.update_layout(title=f"{horizon}-Month Forecast: {dim_value} ({dim_type})",
                       xaxis_title="Date", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("Model MAE (3-month backtest)", f"${mae:,.0f}")
    m2.metric("Model RMSE (3-month backtest)", f"${rmse:,.0f}")

    st.subheader("Forecast values")
    fc_table = pd.DataFrame({"Month": [d.strftime("%b %Y") for d in future_dates],
                              "Forecasted Sales": [f"${p:,.0f}" for p in preds]})
    st.table(fc_table)

# ---------------------------------------------------------------------------
# PAGE 3 — Anomaly Report
# ---------------------------------------------------------------------------
elif page == "Anomaly Report":
    st.title("Anomaly Report")
    st.caption("Weekly sales anomalies detected via Isolation Forest and Z-Score (±2σ) methods.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=anomaly_df["Week"], y=anomaly_df["Sales"],
                              mode="lines", name="Weekly Sales", line=dict(color="#2563eb")))
    iso_pts = anomaly_df[anomaly_df["IsoAnomaly"]]
    z_pts = anomaly_df[anomaly_df["ZAnomaly"]]
    fig.add_trace(go.Scatter(x=iso_pts["Week"], y=iso_pts["Sales"], mode="markers",
                              name="Isolation Forest Anomaly",
                              marker=dict(color="#dc2626", size=10)))
    fig.add_trace(go.Scatter(x=z_pts["Week"], y=z_pts["Sales"], mode="markers",
                              name="Z-Score Anomaly",
                              marker=dict(color="#d97706", size=10, symbol="x")))
    fig.update_layout(title="Weekly Sales with Detected Anomalies", xaxis_title="Week", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected anomaly dates")
    anomalies = anomaly_df[anomaly_df["IsoAnomaly"] | anomaly_df["ZAnomaly"]].copy()
    anomalies["Detected By"] = anomalies.apply(
        lambda r: ", ".join(filter(None, [
            "Isolation Forest" if r["IsoAnomaly"] else None,
            "Z-Score" if r["ZAnomaly"] else None,
        ])), axis=1)
    display_cols = anomalies[["Week", "Sales", "Detected By"]].sort_values("Sales", ascending=False)
    display_cols["Sales"] = display_cols["Sales"].map(lambda x: f"${x:,.0f}")
    st.dataframe(display_cols, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# PAGE 4 — Product Demand Segments
# ---------------------------------------------------------------------------
elif page == "Demand Segments":
    st.title("Product Demand Segments")
    st.caption("K-Means clustering (k=4) on sub-category demand behavior, visualized via PCA.")

    fig = px.scatter(cluster_df, x="PCA1", y="PCA2", color="ClusterLabel",
                      text="Sub-Category", size="TotalSales", size_max=40,
                      title="Product Sub-Category Demand Clusters")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sub-categories by cluster")
    for label in cluster_df["ClusterLabel"].unique():
        with st.expander(f"📦 {label}"):
            sub = cluster_df[cluster_df["ClusterLabel"] == label][
                ["Sub-Category", "TotalSales", "GrowthRate", "Volatility", "AvgOrderValue"]
            ].round(1)
            st.dataframe(sub, use_container_width=True, hide_index=True)

st.sidebar.markdown("---")
st.sidebar.caption("Superstore Sales Forecasting & Demand Intelligence System")
