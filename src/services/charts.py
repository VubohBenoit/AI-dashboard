"""
src/services/charts.py
───────────────────────
All Plotly chart factories for the dashboard.
Each function receives processed data and returns a go.Figure.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── Design tokens ─────────────────────────────────────────────────────────────
PALETTE = {
    "bg":        "#0f1117",
    "surface":   "#1a1d2e",
    "border":    "#2a2d3e",
    "text":      "#e2e8f0",
    "muted":     "#64748b",
    "accent1":   "#6366f1",   # indigo
    "accent2":   "#06b6d4",   # cyan
    "accent3":   "#10b981",   # emerald
    "accent4":   "#f59e0b",   # amber
    "danger":    "#ef4444",
}

CAT_COLORS = [PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"],
              PALETTE["accent4"], "#8b5cf6", "#ec4899"]

BASE_LAYOUT = dict(
    paper_bgcolor = PALETTE["bg"],
    plot_bgcolor  = PALETTE["surface"],
    font          = dict(family="'DM Mono', monospace", color=PALETTE["text"], size=12),
    margin        = dict(l=16, r=16, t=40, b=16),
    xaxis         = dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"],
                         tickfont=dict(size=10)),
    yaxis         = dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"],
                         tickfont=dict(size=10)),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor=PALETTE["border"],
                         font=dict(size=11)),
    hoverlabel    = dict(bgcolor=PALETTE["surface"], bordercolor=PALETTE["border"],
                         font=dict(color=PALETTE["text"])),
)


def _apply_base(fig: go.Figure, title: str = "") -> go.Figure:
    layout = dict(**BASE_LAYOUT)
    if title:
        layout["title"] = dict(text=title, font=dict(size=14, color=PALETTE["text"]),
                                x=0.02, xanchor="left")
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  KPI CARDS  (returns dict, not a figure)
# ══════════════════════════════════════════════════════════════════════════════

def compute_kpis(sales_df: pd.DataFrame) -> dict:
    """Returns KPI metrics dict for the summary cards."""
    total_rev     = sales_df["revenue"].sum()
    total_units   = sales_df["units"].sum()
    avg_margin    = sales_df["profit_margin"].mean()
    total_orders  = len(sales_df)

    # Month-over-month change
    today         = sales_df["date"].max()
    this_month    = sales_df[sales_df["date"] >= today - pd.Timedelta(days=30)]["revenue"].sum()
    last_month    = sales_df[(sales_df["date"] >= today - pd.Timedelta(days=60)) &
                             (sales_df["date"] <  today - pd.Timedelta(days=30))]["revenue"].sum()
    mom_change    = (this_month - last_month) / (last_month + 1) * 100

    return {
        "total_revenue":  round(total_rev, 0),
        "total_units":    int(total_units),
        "avg_margin":     round(avg_margin * 100, 1),
        "total_orders":   total_orders,
        "mom_change":     round(mom_change, 1),
        "this_month_rev": round(this_month, 0),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  REVENUE OVER TIME
# ══════════════════════════════════════════════════════════════════════════════

def revenue_over_time(sales_df: pd.DataFrame, granularity: str = "W") -> go.Figure:
    """Area chart of revenue aggregated by week or month."""
    label_map = {"D": "day", "W": "week", "ME": "month"}
    agg = (sales_df.groupby(pd.Grouper(key="date", freq=granularity))["revenue"]
                   .sum()
                   .reset_index())

    fig = go.Figure()

    # Fill area
    fig.add_trace(go.Scatter(
        x=agg["date"], y=agg["revenue"],
        mode="lines",
        line=dict(color=PALETTE["accent1"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.12)",
        name="Revenue",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>$%{y:,.0f}<extra></extra>",
    ))

    # 8-period moving average
    agg["ma"] = agg["revenue"].rolling(8, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=agg["date"], y=agg["ma"],
        mode="lines",
        line=dict(color=PALETTE["accent2"], width=1.5, dash="dot"),
        name="8-period MA",
        hovertemplate="<b>MA</b> $%{y:,.0f}<extra></extra>",
    ))

    return _apply_base(fig, "Revenue Over Time")


# ══════════════════════════════════════════════════════════════════════════════
#  SALES FORECAST
# ══════════════════════════════════════════════════════════════════════════════

def forecast_chart(sales_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    """Combines historical daily revenue with forecast + confidence band."""
    historical = (sales_df.groupby("date")["revenue"]
                          .sum()
                          .reset_index()
                          .tail(120))   # last 120 days

    fig = go.Figure()

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["ds"], forecast_df["ds"].iloc[::-1]]),
        y=pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"].iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(6,182,212,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI",
        hoverinfo="skip",
    ))

    # Historical
    fig.add_trace(go.Scatter(
        x=historical["date"], y=historical["revenue"],
        mode="lines",
        line=dict(color=PALETTE["accent1"], width=2),
        name="Historical",
        hovertemplate="<b>%{x|%b %d}</b><br>$%{y:,.0f}<extra></extra>",
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"], y=forecast_df["yhat"],
        mode="lines",
        line=dict(color=PALETTE["accent2"], width=2.5, dash="dash"),
        name="Forecast",
        hovertemplate="<b>%{x|%b %d}</b><br>$%{y:,.0f}<extra></extra>",
    ))

    # Divider line — convert Timestamp to ms (int) for Plotly compatibility
    split = historical["date"].max()
    split_ms = int(pd.Timestamp(split).timestamp() * 1000)
    fig.add_vline(x=split_ms, line_dash="dot", line_color=PALETTE["muted"],
                  annotation_text="Today", annotation_font_color=PALETTE["muted"])

    return _apply_base(fig, "Sales Forecast (90 Days)")


# ══════════════════════════════════════════════════════════════════════════════
#  REVENUE BY CATEGORY
# ══════════════════════════════════════════════════════════════════════════════

def revenue_by_category(sales_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of total revenue per category."""
    agg = (sales_df.groupby("category")["revenue"]
                   .sum()
                   .sort_values()
                   .reset_index())

    fig = go.Figure(go.Bar(
        x=agg["revenue"], y=agg["category"],
        orientation="h",
        marker=dict(
            color=CAT_COLORS[:len(agg)],
            line=dict(width=0),
        ),
        hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>",
    ))
    return _apply_base(fig, "Revenue by Category")


# ══════════════════════════════════════════════════════════════════════════════
#  CHANNEL BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════

def channel_donut(sales_df: pd.DataFrame) -> go.Figure:
    """Donut chart showing revenue split by sales channel."""
    agg = sales_df.groupby("channel")["revenue"].sum().reset_index()

    fig = go.Figure(go.Pie(
        labels=agg["channel"],
        values=agg["revenue"],
        hole=0.60,
        marker=dict(colors=CAT_COLORS, line=dict(color=PALETTE["bg"], width=3)),
        hovertemplate="<b>%{label}</b><br>$%{value:,.0f} (%{percent})<extra></extra>",
        textfont=dict(color=PALETTE["text"]),
    ))
    fig.update_layout(
        showlegend=True,
        annotations=[dict(text="Channels", x=0.5, y=0.5, showarrow=False,
                          font=dict(size=13, color=PALETTE["muted"]))],
    )
    return _apply_base(fig, "Revenue by Channel")


# ══════════════════════════════════════════════════════════════════════════════
#  REGIONAL HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def regional_heatmap(sales_df: pd.DataFrame) -> go.Figure:
    """Heatmap of revenue by region × category."""
    pivot = (sales_df.groupby(["region","category"])["revenue"]
                     .sum()
                     .unstack(fill_value=0))

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0,"#1a1d2e"],[0.5,"#6366f1"],[1,"#06b6d4"]],
        hovertemplate="<b>%{y} × %{x}</b><br>$%{z:,.0f}<extra></extra>",
        showscale=True,
        colorbar=dict(tickfont=dict(color=PALETTE["text"]), thickness=12),
    ))
    return _apply_base(fig, "Revenue: Region × Category")


# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOMER SEGMENTS SCATTER
# ══════════════════════════════════════════════════════════════════════════════

def segment_scatter(segmented_df: pd.DataFrame) -> go.Figure:
    """2D PCA scatter of customer clusters."""
    fig = go.Figure()
    for label in segmented_df["segment_label"].unique():
        sub   = segmented_df[segmented_df["segment_label"] == label]
        color = sub["segment_color"].iloc[0]
        fig.add_trace(go.Scatter(
            x=sub["pca_x"], y=sub["pca_y"],
            mode="markers",
            name=label,
            marker=dict(color=color, size=5, opacity=0.75,
                        line=dict(width=0)),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Spend: $%{customdata[0]:,.0f}<br>"
                "Churn risk: %{customdata[1]:.0%}<extra></extra>"
            ),
            text=sub["customer_id"],
            customdata=sub[["total_spend","churn_risk"]].values,
        ))
    return _apply_base(fig, "Customer Segments (PCA)")


# ══════════════════════════════════════════════════════════════════════════════
#  CHURN RISK HISTOGRAM
# ══════════════════════════════════════════════════════════════════════════════

def churn_histogram(customers_df: pd.DataFrame) -> go.Figure:
    """Distribution of churn risk scores."""
    fig = go.Figure(go.Histogram(
        x=customers_df["churn_risk"],
        nbinsx=30,
        marker=dict(color=PALETTE["accent1"], opacity=0.85,
                    line=dict(width=0)),
        hovertemplate="Risk: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0.5, line_dash="dot", line_color=PALETTE["danger"],
                  annotation_text="High risk threshold",
                  annotation_font_color=PALETTE["danger"])
    return _apply_base(fig, "Churn Risk Distribution")


# ══════════════════════════════════════════════════════════════════════════════
#  PRODUCT PERFORMANCE BUBBLE
# ══════════════════════════════════════════════════════════════════════════════

def product_bubble(products_df: pd.DataFrame) -> go.Figure:
    """Bubble chart: price vs rating, size=units_sold, color=category."""
    cat_map = {c: CAT_COLORS[i % len(CAT_COLORS)]
               for i, c in enumerate(products_df["category"].unique())}

    fig = go.Figure()
    for cat in products_df["category"].unique():
        sub = products_df[products_df["category"] == cat]
        fig.add_trace(go.Scatter(
            x=sub["price"],
            y=sub["rating"],
            mode="markers",
            name=cat,
            marker=dict(
                size=sub["units_sold"] / sub["units_sold"].max() * 40 + 6,
                color=cat_map[cat],
                opacity=0.75,
                line=dict(width=0),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Price: $%{x:.2f}<br>"
                "Rating: %{y:.1f}<br>"
                "Units: %{customdata:,d}<extra></extra>"
            ),
            text=sub["name"],
            customdata=sub["units_sold"],
        ))
    return _apply_base(fig, "Products: Price vs Rating (size = units sold)")


# ══════════════════════════════════════════════════════════════════════════════
#  ANOMALY CHART
# ══════════════════════════════════════════════════════════════════════════════

def anomaly_chart(anomaly_df: pd.DataFrame) -> go.Figure:
    """Scatter overlay marking anomalous sales days in red."""
    normal  = anomaly_df[~anomaly_df["is_anomaly"]]
    outlier = anomaly_df[anomaly_df["is_anomaly"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal["date"], y=normal["revenue"],
        mode="markers",
        marker=dict(color=PALETTE["accent1"], size=4, opacity=0.6),
        name="Normal",
        hovertemplate="<b>%{x|%b %d}</b><br>$%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=outlier["date"], y=outlier["revenue"],
        mode="markers",
        marker=dict(color=PALETTE["danger"], size=9, symbol="x",
                    line=dict(width=2, color=PALETTE["danger"])),
        name="Anomaly",
        hovertemplate="<b>⚠ %{x|%b %d}</b><br>$%{y:,.0f}<extra></extra>",
    ))
    return _apply_base(fig, "Revenue Anomaly Detection")


# ══════════════════════════════════════════════════════════════════════════════
#  MARGIN WATERFALL
# ══════════════════════════════════════════════════════════════════════════════

def margin_waterfall(sales_df: pd.DataFrame) -> go.Figure:
    """Waterfall chart showing average profit margin by category."""
    agg = (sales_df.groupby("category")["profit_margin"]
                   .mean()
                   .sort_values(ascending=False)
                   .reset_index())

    colors = [PALETTE["accent3"] if v >= agg["profit_margin"].mean()
              else PALETTE["accent4"]
              for v in agg["profit_margin"]]

    fig = go.Figure(go.Bar(
        x=agg["category"],
        y=(agg["profit_margin"] * 100).round(1),
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>Margin: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=agg["profit_margin"].mean() * 100,
                  line_dash="dot", line_color=PALETTE["muted"],
                  annotation_text="Average",
                  annotation_font_color=PALETTE["muted"])
    return _apply_base(fig, "Profit Margin by Category (%)")