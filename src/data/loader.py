"""
src/data/loader.py
───────────────────
Charge des données réelles depuis CSV, Excel, ou SQL.
Mappe automatiquement les colonnes vers le format interne du dashboard.

Format interne attendu
──────────────────────
sales     : date, category, region, channel, revenue, units, profit_margin, discount
customers : customer_id, segment, region, age, total_spend, purchase_frequency,
            days_since_last_purchase, avg_order_value, churn_risk, lifetime_months
products  : product_id, name, category, price, cost, stock_level,
            units_sold, rating, return_rate
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
#  COLUMN MAPPING — adapte tes noms de colonnes ici
# ══════════════════════════════════════════════════════════════════════════════

# Modifie ce dict si tes colonnes ont des noms différents.
# Format : "nom_dans_ton_fichier" -> "nom_interne_dashboard"

SALES_COLUMN_MAP = {
    # Dates
    "order_date":        "date",
    "transaction_date":  "date",
    "sale_date":         "date",
    "Date":              "date",
    "DATE":              "date",

    # Revenue
    "amount":            "revenue",
    "total":             "revenue",
    "total_amount":      "revenue",
    "sales":             "revenue",
    "Revenue":           "revenue",
    "price":             "revenue",
    "total_price":       "revenue",

    # Units
    "quantity":          "units",
    "qty":               "units",
    "Quantity":          "units",
    "units_sold":        "units",

    # Category
    "product_category":  "category",
    "Category":          "category",
    "dept":              "category",
    "department":        "category",

    # Region
    "Region":            "region",
    "location":          "region",
    "store":             "region",
    "city":              "region",
    "country":           "region",

    # Channel
    "Channel":           "channel",
    "source":            "channel",
    "sales_channel":     "channel",
    "platform":          "channel",

    # Margin
    "margin":            "profit_margin",
    "gross_margin":      "profit_margin",
    "profit":            "profit_margin",

    # Discount
    "Discount":          "discount",
    "discount_rate":     "discount",
    "promo":             "discount",
}

CUSTOMER_COLUMN_MAP = {
    "id":                    "customer_id",
    "client_id":             "customer_id",
    "user_id":               "customer_id",
    "CustomerID":            "customer_id",

    "total_revenue":         "total_spend",
    "lifetime_value":        "total_spend",
    "clv":                   "total_spend",
    "ltv":                   "total_spend",

    "orders":                "purchase_frequency",
    "order_count":           "purchase_frequency",
    "num_orders":            "purchase_frequency",

    "recency":               "days_since_last_purchase",
    "days_inactive":         "days_since_last_purchase",
    "last_purchase_days":    "days_since_last_purchase",

    "avg_basket":            "avg_order_value",
    "average_order":         "avg_order_value",
    "aov":                   "avg_order_value",
}

PRODUCT_COLUMN_MAP = {
    "id":            "product_id",
    "sku":           "product_id",
    "SKU":           "product_id",
    "ProductID":     "product_id",

    "product_name":  "name",
    "title":         "name",
    "Name":          "name",

    "unit_price":    "price",
    "Price":         "price",
    "selling_price": "price",

    "unit_cost":     "cost",
    "Cost":          "cost",
    "cogs":          "cost",

    "stock":         "stock_level",
    "inventory":     "stock_level",
    "qty_on_hand":   "stock_level",

    "sold":          "units_sold",
    "total_sold":    "units_sold",
    "sales_qty":     "units_sold",

    "stars":         "rating",
    "Rating":        "rating",
    "avg_rating":    "rating",

    "returns":       "return_rate",
    "return_pct":    "return_rate",
    "refund_rate":   "return_rate",
}


# ══════════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _apply_column_map(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Renomme les colonnes selon le mapping, ignore les colonnes absentes."""
    rename = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=rename)


def _fill_missing_columns(df: pd.DataFrame, defaults: dict) -> pd.DataFrame:
    """Ajoute les colonnes manquantes avec des valeurs par défaut."""
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


def _read_file(path: str) -> pd.DataFrame:
    """Lit un fichier CSV ou Excel automatiquement."""
    p = Path(path)
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif p.suffix.lower() == ".csv":
        # Essaie différents séparateurs
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(path, sep=sep)
                if len(df.columns) > 1:
                    return df
            except Exception:
                continue
        return pd.read_csv(path)
    else:
        raise ValueError(f"Format non supporté : {p.suffix}. Utilise .csv ou .xlsx")


# ══════════════════════════════════════════════════════════════════════════════
#  SALES LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_sales_from_file(path: str) -> pd.DataFrame:
    """
    Charge un fichier CSV/Excel de transactions et le normalise.

    Colonnes minimales requises dans ton fichier :
        - Une colonne de date  (ex: date, order_date, transaction_date)
        - Une colonne montant  (ex: revenue, amount, total)

    Toutes les autres colonnes sont optionnelles (valeurs par défaut appliquées).
    """
    df = _read_file(path)
    df = _apply_column_map(df, SALES_COLUMN_MAP)

    # Colonne date obligatoire
    if "date" not in df.columns:
        raise ValueError(
            "Colonne 'date' introuvable. Renomme ta colonne de date en 'date' "
            "ou ajoute-la dans SALES_COLUMN_MAP dans loader.py"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Colonne revenue obligatoire
    if "revenue" not in df.columns:
        raise ValueError(
            "Colonne 'revenue' introuvable. Renomme ta colonne de montant en 'revenue' "
            "ou ajoute-la dans SALES_COLUMN_MAP dans loader.py"
        )

    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)

    # Colonnes optionnelles avec valeurs par défaut
    defaults = {
        "units":         1,
        "category":      "Unknown",
        "region":        "Unknown",
        "channel":       "Unknown",
        "profit_margin": 0.25,
        "discount":      0.0,
    }
    df = _fill_missing_columns(df, defaults)
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(1).astype(int)

    return df[["date","category","region","channel","revenue","units","profit_margin","discount"]]


# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOMER LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_customers_from_file(path: str) -> pd.DataFrame:
    """
    Charge un fichier CSV/Excel de clients.

    Si tu n'as pas de fichier clients, le dashboard peut dériver les métriques
    RFM directement depuis les données de ventes via `derive_customers_from_sales()`.
    """
    df = _read_file(path)
    df = _apply_column_map(df, CUSTOMER_COLUMN_MAP)

    if "customer_id" not in df.columns:
        df["customer_id"] = [f"C{str(i).zfill(4)}" for i in range(len(df))]

    defaults = {
        "segment":                    "Unknown",
        "region":                     "Unknown",
        "age":                        35,
        "total_spend":                0,
        "purchase_frequency":         1,
        "days_since_last_purchase":   90,
        "avg_order_value":            0,
        "churn_risk":                 0.3,
        "lifetime_months":            12,
    }
    df = _fill_missing_columns(df, defaults)

    # Normalise churn_risk entre 0 et 1
    if df["churn_risk"].max() > 1:
        df["churn_risk"] = df["churn_risk"] / 100

    return df[[
        "customer_id","segment","region","age","total_spend",
        "purchase_frequency","days_since_last_purchase",
        "avg_order_value","churn_risk","lifetime_months"
    ]]


# ══════════════════════════════════════════════════════════════════════════════
#  PRODUCT LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_products_from_file(path: str) -> pd.DataFrame:
    """Charge un fichier CSV/Excel de produits."""
    df = _read_file(path)
    df = _apply_column_map(df, PRODUCT_COLUMN_MAP)

    if "product_id" not in df.columns:
        df["product_id"] = [f"P{str(i).zfill(3)}" for i in range(len(df))]
    if "name" not in df.columns:
        df["name"] = df["product_id"]

    defaults = {
        "category":    "Unknown",
        "price":       0.0,
        "cost":        0.0,
        "stock_level": 100,
        "units_sold":  0,
        "rating":      3.0,
        "return_rate": 0.05,
    }
    df = _fill_missing_columns(df, defaults)

    return df[["product_id","name","category","price","cost",
               "stock_level","units_sold","rating","return_rate"]]


# ══════════════════════════════════════════════════════════════════════════════
#  DERIVE CUSTOMERS FROM SALES (si pas de fichier clients)
# ══════════════════════════════════════════════════════════════════════════════

def derive_customers_from_sales(sales_df: pd.DataFrame,
                                  customer_col: str = "customer_id") -> pd.DataFrame:
    """
    Dérive les métriques RFM directement depuis les transactions.
    Utile si tu as une colonne customer_id dans tes ventes mais pas de fichier clients séparé.

    Paramètre :
        customer_col : nom de la colonne identifiant le client dans sales_df
    """
    if customer_col not in sales_df.columns:
        raise ValueError(
            f"Colonne '{customer_col}' introuvable dans les données de ventes. "
            "Ajoute une colonne d'identifiant client."
        )

    today = sales_df["date"].max()

    rfm = sales_df.groupby(customer_col).agg(
        total_spend              = ("revenue", "sum"),
        purchase_frequency       = ("revenue", "count"),
        last_purchase            = ("date",    "max"),
        avg_order_value          = ("revenue", "mean"),
    ).reset_index().rename(columns={customer_col: "customer_id"})

    rfm["days_since_last_purchase"] = (today - rfm["last_purchase"]).dt.days
    rfm["lifetime_months"] = (
        (today - sales_df.groupby(customer_col)["date"].min()).dt.days / 30
    ).values.astype(int)

    # Score de churn simple basé sur la récence
    max_recency = rfm["days_since_last_purchase"].max()
    rfm["churn_risk"] = (rfm["days_since_last_purchase"] / (max_recency + 1)).round(3)

    rfm["segment"] = "Customer"
    rfm["region"]  = "Unknown"
    rfm["age"]     = 35

    return rfm[[
        "customer_id","segment","region","age","total_spend",
        "purchase_frequency","days_since_last_purchase",
        "avg_order_value","churn_risk","lifetime_months"
    ]]


# ══════════════════════════════════════════════════════════════════════════════
#  SQL LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_sales_from_sql(db_path: str, query: str) -> pd.DataFrame:
    """
    Charge les ventes depuis une base SQLite.

    Exemple :
        df = load_sales_from_sql(
            "ma_base.db",
            "SELECT order_date as date, amount as revenue, category FROM orders"
        )
    """
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql_query(query, conn)
    conn.close()
    return load_sales_from_file.__wrapped__(df) if hasattr(load_sales_from_file, "__wrapped__") else df


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UPLOAD HELPER
# ══════════════════════════════════════════════════════════════════════════════

def load_from_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Charge un fichier uploadé via st.file_uploader() de Streamlit.
    Usage :
        file = st.file_uploader("Upload")
        df   = load_from_uploaded_file(file)
    """
    import io
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        content = uploaded_file.read().decode("utf-8", errors="replace")
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(io.StringIO(content), sep=sep)
                if len(df.columns) > 1:
                    return df
            except Exception:
                continue
        return pd.read_csv(io.StringIO(content))
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Formats supportés : .csv, .xlsx, .xls")