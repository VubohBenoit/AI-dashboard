"""
src/data/generator.py
─────────────────────
Generates realistic synthetic business data for the dashboard.
Covers sales, customers, and products across 2 years.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ── Seed for reproducibility ──────────────────────────────────────────────────
RNG = np.random.default_rng(42)

CATEGORIES   = ["Electronics", "Clothing", "Food & Beverage", "Home & Garden", "Sports"]
REGIONS      = ["North", "South", "East", "West", "Central"]
CHANNELS     = ["Online", "In-Store", "Mobile App", "Partner"]
SEGMENTS     = ["Enterprise", "SMB", "Consumer", "Government"]


# ── Sales Data ────────────────────────────────────────────────────────────────

def generate_sales_data(n_days: int = 730) -> pd.DataFrame:
    """
    Daily sales records with realistic seasonality, trend, and noise.

    Returns columns:
        date, category, region, channel, revenue, units, profit_margin, discount
    """
    dates = pd.date_range(end=datetime.today(), periods=n_days, freq="D")
    records = []

    for date in dates:
        # Number of transactions per day varies
        n_transactions = RNG.integers(15, 45)

        for _ in range(n_transactions):
            category = RNG.choice(CATEGORIES)
            region   = RNG.choice(REGIONS)
            channel  = RNG.choice(CHANNELS)

            # Base revenue with category-specific multiplier
            base = {"Electronics": 350, "Clothing": 85, "Food & Beverage": 45,
                    "Home & Garden": 120, "Sports": 95}[category]

            # Seasonality: peak in Nov-Dec, dip in Jan-Feb
            month_factor = 1 + 0.4 * np.sin((date.month - 3) * np.pi / 6)
            # Long-term upward trend
            day_idx      = (date - dates[0]).days
            trend_factor = 1 + 0.0003 * day_idx
            # Random noise
            noise        = RNG.normal(1.0, 0.18)

            revenue      = max(10, base * month_factor * trend_factor * noise)
            units        = max(1, int(revenue / base * RNG.integers(1, 6)))
            margin       = RNG.uniform(0.12, 0.48)
            discount     = RNG.choice([0, 0, 0, 0.05, 0.10, 0.15, 0.20],
                                      p=[0.5, 0.15, 0.1, 0.1, 0.07, 0.05, 0.03])

            records.append({
                "date":          date,
                "category":      category,
                "region":        region,
                "channel":       channel,
                "revenue":       round(revenue, 2),
                "units":         units,
                "profit_margin": round(margin, 3),
                "discount":      discount,
            })

    return pd.DataFrame(records)


# ── Customer Data ─────────────────────────────────────────────────────────────

def generate_customer_data(n_customers: int = 800) -> pd.DataFrame:
    """
    Customer profiles with RFM (Recency, Frequency, Monetary) attributes
    and behavioural features for segmentation.

    Returns columns:
        customer_id, segment, region, age, total_spend, purchase_frequency,
        days_since_last_purchase, avg_order_value, churn_risk, lifetime_months
    """
    ids = [f"C{str(i).zfill(4)}" for i in range(1, n_customers + 1)]

    # Segment distributions
    seg_probs = [0.15, 0.35, 0.42, 0.08]   # Enterprise, SMB, Consumer, Government

    ages              = RNG.integers(22, 72, n_customers)
    lifetime_months   = RNG.integers(1, 60, n_customers)
    purchase_freq     = RNG.integers(1, 52, n_customers)
    avg_order         = RNG.exponential(180, n_customers) + 30
    total_spend       = avg_order * purchase_freq * (lifetime_months / 12)
    recency           = RNG.integers(1, 365, n_customers)
    churn_risk        = np.clip(recency / 365 - purchase_freq / 100 + RNG.normal(0, 0.1, n_customers), 0, 1)

    return pd.DataFrame({
        "customer_id":               ids,
        "segment":                   RNG.choice(SEGMENTS, n_customers, p=seg_probs),
        "region":                    RNG.choice(REGIONS, n_customers),
        "age":                       ages,
        "total_spend":               np.round(total_spend, 2),
        "purchase_frequency":        purchase_freq,
        "days_since_last_purchase":  recency,
        "avg_order_value":           np.round(avg_order, 2),
        "churn_risk":                np.round(churn_risk, 3),
        "lifetime_months":           lifetime_months,
    })


# ── Product Data ──────────────────────────────────────────────────────────────

def generate_product_data(n_products: int = 120) -> pd.DataFrame:
    """
    Product catalogue with performance metrics.

    Returns columns:
        product_id, name, category, price, cost, stock_level,
        units_sold, rating, return_rate
    """
    adjectives = ["Pro", "Ultra", "Smart", "Premium", "Classic", "Eco", "Lite", "Max"]
    nouns = {
        "Electronics":     ["Headphones", "Tablet", "Speaker", "Camera", "Watch", "Laptop"],
        "Clothing":        ["Jacket", "Sneakers", "T-Shirt", "Jeans", "Hoodie", "Dress"],
        "Food & Beverage": ["Protein Bar", "Coffee Blend", "Energy Drink", "Snack Pack", "Tea Set"],
        "Home & Garden":   ["Lamp", "Planter", "Rug", "Cushion", "Tool Kit", "Mirror"],
        "Sports":          ["Yoga Mat", "Dumbbells", "Backpack", "Water Bottle", "Resistance Band"],
    }

    records = []
    for i in range(1, n_products + 1):
        cat   = RNG.choice(CATEGORIES)
        adj   = RNG.choice(adjectives)
        noun  = RNG.choice(nouns[cat])
        price = round(float(RNG.uniform(15, 600)), 2)
        cost  = round(price * RNG.uniform(0.35, 0.70), 2)

        records.append({
            "product_id":   f"P{str(i).zfill(3)}",
            "name":         f"{adj} {noun}",
            "category":     cat,
            "price":        price,
            "cost":         cost,
            "stock_level":  int(RNG.integers(0, 500)),
            "units_sold":   int(RNG.integers(10, 2000)),
            "rating":       round(float(RNG.uniform(2.8, 5.0)), 1),
            "return_rate":  round(float(RNG.uniform(0.01, 0.18)), 3),
        })

    return pd.DataFrame(records)


# ── Convenience loader ────────────────────────────────────────────────────────

def load_all_data() -> dict[str, pd.DataFrame]:
    """Returns a dict with keys: 'sales', 'customers', 'products'."""
    return {
        "sales":     generate_sales_data(),
        "customers": generate_customer_data(),
        "products":  generate_product_data(),
    }
