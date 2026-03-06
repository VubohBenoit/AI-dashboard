"""
src/components/pages.py
────────────────────────
Individual dashboard page renderers called from app.py.
Each page function receives pre-computed data and renders
Streamlit components + Plotly charts.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.services import charts as ch
from src.utils.helpers import (
    fmt_currency, fmt_pct, fmt_delta,
    generate_sales_insight, generate_segment_insight,
    generate_product_insight, compute_trend,
)


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED STYLE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _kpi_card(label: str, value: str, delta: str = None, delta_color: str = "normal"):
    """Renders a single metric tile."""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def page_overview(sales_df: pd.DataFrame, kpis: dict):
    st.markdown("## 📊 Business Overview")
    st.caption("High-level revenue metrics across all categories and regions.")

    # ── KPI Row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        delta_str, delta_col = fmt_delta(kpis["mom_change"])
        _kpi_card("Total Revenue", fmt_currency(kpis["total_revenue"]),
                  delta_str, delta_col)
    with c2:
        _kpi_card("Units Sold", f"{kpis['total_units']:,}")
    with c3:
        _kpi_card("Avg Margin", fmt_pct(kpis["avg_margin"]))
    with c4:
        _kpi_card("Total Orders", f"{kpis['total_orders']:,}")

    st.divider()

    # ── Granularity selector ──
    gran = st.radio("Aggregate by", ["D", "W", "ME"],
                    format_func=lambda x: {"D":"Day","W":"Week","ME":"Month"}[x],
                    horizontal=True, index=1)

    st.plotly_chart(ch.revenue_over_time(sales_df, gran),
                    use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(ch.revenue_by_category(sales_df), use_container_width=True)
    with col_r:
        st.plotly_chart(ch.channel_donut(sales_df), use_container_width=True)

    st.plotly_chart(ch.regional_heatmap(sales_df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — SALES FORECAST
# ══════════════════════════════════════════════════════════════════════════════

def page_forecast(sales_df: pd.DataFrame, forecaster, forecast_df: pd.DataFrame):
    st.markdown("## 🔮 Sales Forecasting")
    st.caption("Gradient Boosting model trained on 2 years of daily sales data.")

    # AI Insight
    with st.expander("🤖 AI Insight", expanded=True):
        st.markdown(generate_sales_insight(sales_df, forecast_df))

    st.plotly_chart(ch.forecast_chart(sales_df, forecast_df),
                    use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### 📅 90-Day Forecast Summary")
        summary = pd.DataFrame({
            "Metric":  ["Total Projected Revenue", "Peak Daily Revenue",
                        "Min Daily Revenue", "Average Daily"],
            "Value":   [
                fmt_currency(forecast_df["yhat"].sum()),
                fmt_currency(forecast_df["yhat"].max()),
                fmt_currency(forecast_df["yhat"].min()),
                fmt_currency(forecast_df["yhat"].mean()),
            ]
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

    with col_r:
        st.markdown("#### 🧠 Feature Importance")
        fi = forecaster.feature_importance().head(6)
        fig = go.Figure(go.Bar(
            x=fi["importance"], y=fi["feature"],
            orientation="h",
            marker=dict(color="#6366f1", line=dict(width=0)),
        ))
        fig.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#1a1d2e",
            font=dict(color="#e2e8f0", size=11),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#2a2d3e"),
            yaxis=dict(gridcolor="#2a2d3e"),
            height=260,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(ch.margin_waterfall(sales_df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def page_customers(customers_df: pd.DataFrame, segmenter, segmented_df: pd.DataFrame):
    st.markdown("## 👥 Customer Segmentation")
    st.caption(f"KMeans clustering on RFM features — {segmenter.best_k} segments auto-detected.")

    summary = segmenter.cluster_summary(segmented_df)

    # AI Insight
    with st.expander("🤖 AI Insight", expanded=True):
        st.markdown(generate_segment_insight(summary))

    # Segment cards
    cols = st.columns(min(segmenter.best_k, 4))
    for i, row in summary.iterrows():
        with cols[i % len(cols)]:
            info = segmenter.SEGMENT_LABELS.get(row["cluster"], ("Other","#94a3b8","•"))
            st.markdown(f"""
            <div style="background:#1a1d2e;border:1px solid #2a2d3e;border-radius:10px;
                        padding:14px;margin-bottom:8px;text-align:center">
                <div style="font-size:22px">{info[2]}</div>
                <div style="color:{info[1]};font-weight:700;font-size:13px">{row['segment_label']}</div>
                <div style="color:#94a3b8;font-size:11px">{row['count']} customers</div>
                <div style="color:#e2e8f0;font-size:12px;margin-top:4px">
                    Avg spend: <b>${row['avg_spend']:,.0f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.plotly_chart(ch.segment_scatter(segmented_df), use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(ch.churn_histogram(segmented_df), use_container_width=True)
    with col_r:
        st.markdown("#### 🔴 High Churn Risk Customers")
        high_risk = (segmented_df[segmented_df["churn_risk"] > 0.65]
                     .nlargest(10, "total_spend")
                     [["customer_id","segment_label","total_spend",
                       "days_since_last_purchase","churn_risk"]]
                     .rename(columns={
                         "customer_id": "ID",
                         "segment_label": "Segment",
                         "total_spend": "Spend ($)",
                         "days_since_last_purchase": "Days Inactive",
                         "churn_risk": "Risk",
                     }))
        high_risk["Spend ($)"] = high_risk["Spend ($)"].apply(lambda x: f"${x:,.0f}")
        high_risk["Risk"]      = high_risk["Risk"].apply(lambda x: f"{x:.0%}")
        st.dataframe(high_risk, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — PRODUCT ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def page_products(products_df: pd.DataFrame, recommender):
    st.markdown("## 📦 Product Analytics")
    st.caption("Performance metrics, rankings, and content-based recommendations.")

    top10 = recommender.top_performers(10)

    with st.expander("🤖 AI Insight", expanded=True):
        st.markdown(generate_product_insight(top10, products_df))

    st.plotly_chart(ch.product_bubble(products_df), use_container_width=True)

    col_l, col_r = st.columns([1.2, 0.8])
    with col_l:
        st.markdown("#### 🏆 Top 10 Products")
        display = top10[["name","category","price","units_sold","rating","return_rate"]].copy()
        display["price"]       = display["price"].apply(lambda x: f"${x:.2f}")
        display["return_rate"] = display["return_rate"].apply(lambda x: f"{x:.1%}")
        display["units_sold"]  = display["units_sold"].apply(lambda x: f"{x:,}")
        st.dataframe(display, hide_index=True, use_container_width=True)

    with col_r:
        st.markdown("#### 🔍 Product Recommender")
        product_names = products_df["product_id"].tolist()
        selected_id   = st.selectbox("Select a product", product_names,
                                     format_func=lambda pid: (
                                         f"{pid} — "
                                         f"{products_df[products_df['product_id']==pid]['name'].values[0]}"
                                     ))
        if selected_id:
            try:
                recs = recommender.recommend(selected_id, top_n=5)
                for _, r in recs.iterrows():
                    st.markdown(
                        f"**{r['name']}** `{r['category']}` "
                        f"${r['price']:.2f} ⭐{r['rating']} "
                        f"sim={r['similarity']:.2f}"
                    )
            except Exception as e:
                st.warning(str(e))

    # Low stock alert
    low_stock = products_df[products_df["stock_level"] < 30].sort_values("stock_level")
    if len(low_stock):
        st.warning(f"⚠️ {len(low_stock)} products below 30 units in stock")
        st.dataframe(low_stock[["product_id","name","category","stock_level","units_sold"]],
                     hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def page_anomalies(sales_df: pd.DataFrame, anomaly_df: pd.DataFrame):
    st.markdown("## 🔍 Anomaly Detection")
    st.caption("IsolationForest detects unusual sales days based on revenue, volume, and transaction count.")

    n_anomalies = anomaly_df["is_anomaly"].sum()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Days Analyzed", f"{len(anomaly_df):,}")
    with col2:
        st.metric("Anomalies Found", str(n_anomalies),
                  delta=f"{n_anomalies/len(anomaly_df)*100:.1f}% of days")
    with col3:
        avg_anomaly_rev = anomaly_df[anomaly_df["is_anomaly"]]["revenue"].mean()
        st.metric("Avg Anomaly Revenue", fmt_currency(avg_anomaly_rev))

    st.plotly_chart(ch.anomaly_chart(anomaly_df), use_container_width=True)

    st.markdown("#### 📋 Anomalous Days Detail")
    detail = (anomaly_df[anomaly_df["is_anomaly"]]
              .sort_values("anomaly_score")
              [["date","revenue","units","transactions","anomaly_score"]]
              .rename(columns={"anomaly_score": "outlier_score"}))
    detail["revenue"]      = detail["revenue"].apply(fmt_currency)
    detail["outlier_score"]= detail["outlier_score"].apply(lambda x: f"{x:.3f}")
    st.dataframe(detail, hide_index=True, use_container_width=True)
