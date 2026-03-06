"""
src/utils/helpers.py
─────────────────────
Utility functions: formatting, AI-generated text insights,
caching wrappers, and metric helpers.
"""

import pandas as pd
import numpy as np
import streamlit as st


# ── Number formatting ─────────────────────────────────────────────────────────

def fmt_currency(value: float, compact: bool = True) -> str:
    if compact:
        if abs(value) >= 1_000_000:
            return f"${value/1_000_000:.2f}M"
        if abs(value) >= 1_000:
            return f"${value/1_000:.1f}K"
    return f"${value:,.2f}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}%"


def fmt_delta(value: float) -> str:
    arrow = "▲" if value >= 0 else "▼"
    color = "normal" if value >= 0 else "inverse"
    return f"{arrow} {abs(value):.1f}%", color


# ── Trend helpers ─────────────────────────────────────────────────────────────

def compute_trend(series: pd.Series, periods: int = 30) -> float:
    """Returns the percentage change over the last `periods` data points."""
    if len(series) < periods * 2:
        return 0.0
    recent = series.iloc[-periods:].mean()
    prior  = series.iloc[-periods*2:-periods].mean()
    return (recent - prior) / (prior + 1e-9) * 100


def top_n_by(df: pd.DataFrame, col: str, group: str, n: int = 5) -> pd.DataFrame:
    return (df.groupby(group)[col]
              .sum()
              .nlargest(n)
              .reset_index()
              .rename(columns={col: f"total_{col}"}))


# ── AI Insight Generator ──────────────────────────────────────────────────────

def generate_sales_insight(sales_df: pd.DataFrame, forecast_df: pd.DataFrame) -> str:
    """
    Generates a plain-English summary of current sales performance
    and forecast outlook — no external LLM required, rule-based NLG.
    """
    today        = sales_df["date"].max()
    last_30      = sales_df[sales_df["date"] >= today - pd.Timedelta(days=30)]
    prev_30      = sales_df[(sales_df["date"] >= today - pd.Timedelta(days=60)) &
                            (sales_df["date"] <  today - pd.Timedelta(days=30))]

    rev_now   = last_30["revenue"].sum()
    rev_prev  = prev_30["revenue"].sum()
    mom_pct   = (rev_now - rev_prev) / (rev_prev + 1) * 100

    top_cat   = sales_df.groupby("category")["revenue"].sum().idxmax()
    top_ch    = sales_df.groupby("channel")["revenue"].sum().idxmax()
    avg_margin = sales_df["profit_margin"].mean() * 100

    fc_total  = forecast_df["yhat"].sum()
    fc_growth = (forecast_df["yhat"].iloc[-1] - forecast_df["yhat"].iloc[0]) / \
                (forecast_df["yhat"].iloc[0] + 1) * 100

    trend_word = "**growing** 📈" if mom_pct > 0 else "**declining** 📉"
    fc_word    = "upward" if fc_growth > 0 else "downward"

    return f"""
### 🤖 AI Sales Insight

Revenue is {trend_word} **{abs(mom_pct):.1f}%** month-over-month.
The strongest category is **{top_cat}**, and the most effective channel is **{top_ch}**.
Average profit margin stands at **{avg_margin:.1f}%**.

**90-day forecast:** The model projects an {fc_word} trend,
with estimated revenue of **{fmt_currency(fc_total)}** over the next 90 days
({'+'if fc_growth>0 else ''}{fc_growth:.1f}% trajectory from today).

*Confidence intervals widen further out — treat long-range forecasts as directional only.*
"""


def generate_segment_insight(summary_df: pd.DataFrame) -> str:
    """Plain-English summary of customer segment composition."""
    total       = summary_df["count"].sum()
    top_seg     = summary_df.loc[summary_df["avg_spend"].idxmax(), "segment_label"]
    risky_row   = summary_df[summary_df["segment_label"].str.contains("Risk|Lost", case=False)]
    risky_pct   = risky_row["count"].sum() / total * 100 if len(risky_row) else 0

    return f"""
### 🤖 AI Segment Insight

**{summary_df.shape[0]} distinct customer clusters** identified from RFM analysis.
The highest-value segment is **{top_seg}** by average lifetime spend.
Customers classified as "At Risk" or "Lost" represent **{risky_pct:.1f}%** of the base —
consider targeted re-engagement campaigns for this group.

*Clusters are derived from spend, frequency, recency, and churn-risk features using KMeans.*
"""


def generate_product_insight(top_df: pd.DataFrame, products_df: pd.DataFrame) -> str:
    """Plain-English summary of product performance."""
    best        = top_df.iloc[0]
    avg_rating  = products_df["rating"].mean()
    low_stock   = (products_df["stock_level"] < 20).sum()
    high_return = products_df.nlargest(1, "return_rate").iloc[0]

    return f"""
### 🤖 AI Product Insight

Top performer: **{best['name']}** ({best['category']}) with a composite score of **{best['score']:,.0f}**.
Average catalogue rating is **{avg_rating:.1f}/5.0**.
**{low_stock} products** have dangerously low stock (< 20 units) — reorder recommended.
Highest return rate: **{high_return['name']}** at {high_return['return_rate']*100:.1f}% —
investigate quality or listing accuracy.
"""
