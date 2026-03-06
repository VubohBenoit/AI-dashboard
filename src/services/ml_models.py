"""
src/services/ml_models.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import warnings

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  SALES FORECASTER
# ══════════════════════════════════════════════════════════════════════════════

class SalesForecaster:

    FEATURE_COLS = ["day_of_week", "month", "day_of_year", "week", "trend",
                    "lag_7", "lag_30", "rolling_7", "rolling_30"]

    def __init__(self):
        self.model  = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                learning_rate=0.05, random_state=42)
        self.scaler = StandardScaler()
        self.fitted = False

    def _make_row_features(self, date, trend_idx, hist):
        lag_7   = float(hist[-7])  if len(hist) >= 7  else float(np.mean(hist))
        lag_30  = float(hist[-30]) if len(hist) >= 30 else float(np.mean(hist))
        roll_7  = float(np.mean(hist[-7:]))
        roll_30 = float(np.mean(hist[-30:]))
        return [
            date.dayofweek,
            date.month,
            date.dayofyear,
            int(date.isocalendar()[1]),
            trend_idx,
            lag_7,
            lag_30,
            roll_7,
            roll_30,
        ]

    def fit(self, sales_df: pd.DataFrame) -> "SalesForecaster":
        daily = (sales_df.groupby("date")["revenue"]
                         .sum()
                         .reset_index()
                         .rename(columns={"date": "ds", "revenue": "y"})
                         .sort_values("ds")
                         .reset_index(drop=True))

        self._history    = daily["y"].tolist()
        self._last_date  = daily["ds"].iloc[-1]
        self._last_trend = len(daily) - 1

        rows = []
        for i, row in daily.iterrows():
            hist_so_far = self._history[:i+1]
            rows.append(self._make_row_features(row["ds"], i, hist_so_far))

        X = np.array(rows)
        y = daily["y"].values
        self.model.fit(self.scaler.fit_transform(X), y)
        self.fitted = True
        return self

    def predict(self, periods: int = 90) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Call fit() first.")

        hist  = list(self._history)
        preds = []
        dates = pd.date_range(start=self._last_date + pd.Timedelta(days=1), periods=periods)

        for i, date in enumerate(dates):
            trend_idx = self._last_trend + i + 1
            features  = self._make_row_features(date, trend_idx, hist)
            X         = np.array(features, dtype=float).reshape(1, -1)
            # Safety: replace any NaN just in case
            X         = np.nan_to_num(X, nan=np.nanmean(X))
            yp        = float(self.model.predict(self.scaler.transform(X))[0])
            yp        = max(0.0, yp)
            preds.append(yp)
            hist.append(yp)

        std   = float(np.std(self._history[-30:])) * 0.5
        lower = np.maximum(0, np.array(preds) - 1.96 * std)
        upper = np.array(preds) + 1.96 * std

        return pd.DataFrame({
            "ds":         dates,
            "yhat":       np.round(preds, 2),
            "yhat_lower": np.round(lower, 2),
            "yhat_upper": np.round(upper, 2),
        })

    def feature_importance(self) -> pd.DataFrame:
        imp = self.model.feature_importances_
        return (pd.DataFrame({"feature": self.FEATURE_COLS, "importance": imp})
                  .sort_values("importance", ascending=False))


# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOMER SEGMENTER
# ══════════════════════════════════════════════════════════════════════════════

class CustomerSegmenter:

    FEATURES = ["total_spend", "purchase_frequency", "days_since_last_purchase",
                "avg_order_value", "churn_risk", "lifetime_months"]

    SEGMENT_LABELS = {
        0: ("Champions",      "#10b981", "💎"),
        1: ("Loyal",          "#3b82f6", "⭐"),
        2: ("At Risk",        "#f59e0b", "⚠️"),
        3: ("Lost",           "#ef4444", "❌"),
        4: ("New Customers",  "#8b5cf6", "🆕"),
        5: ("High Potential", "#06b6d4", "🚀"),
    }

    def __init__(self):
        self.scaler = StandardScaler()
        self.model  = None
        self.pca    = PCA(n_components=2)
        self.fitted = False
        self.best_k = 4

    def fit(self, customers_df: pd.DataFrame) -> "CustomerSegmenter":
        X = self.scaler.fit_transform(customers_df[self.FEATURES])
        best_score, best_k = -1, 4
        for k in range(3, 7):
            km     = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            score  = silhouette_score(X, labels)
            if score > best_score:
                best_score, best_k = score, k
        self.best_k = best_k
        self.model  = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        self.model.fit(X)
        self.pca.fit(X)
        self.fitted = True
        return self

    def predict(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        df = customers_df.copy()
        X  = self.scaler.transform(df[self.FEATURES])
        df["cluster"]       = self.model.predict(X)
        df["segment_label"] = df["cluster"].map(
            lambda c: self.SEGMENT_LABELS.get(c, ("Other","#94a3b8","•"))[0])
        df["segment_color"] = df["cluster"].map(
            lambda c: self.SEGMENT_LABELS.get(c, ("Other","#94a3b8","•"))[1])
        coords        = self.pca.transform(X)
        df["pca_x"]   = coords[:, 0]
        df["pca_y"]   = coords[:, 1]
        return df

    def cluster_summary(self, segmented_df: pd.DataFrame) -> pd.DataFrame:
        return (segmented_df.groupby(["cluster", "segment_label"])
                            .agg(count          = ("customer_id", "count"),
                                 avg_spend      = ("total_spend",            "mean"),
                                 avg_frequency  = ("purchase_frequency",     "mean"),
                                 avg_churn_risk = ("churn_risk",             "mean"),
                                 avg_recency    = ("days_since_last_purchase","mean"))
                            .reset_index()
                            .round(2))


# ══════════════════════════════════════════════════════════════════════════════
#  PRODUCT RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════

class ProductRecommender:

    FEATURES = ["price", "cost", "units_sold", "rating", "return_rate", "stock_level"]

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.matrix = None
        self.df     = None
        self.fitted = False

    def fit(self, products_df: pd.DataFrame) -> "ProductRecommender":
        self.df     = products_df.copy().reset_index(drop=True)
        self.matrix = self.scaler.fit_transform(self.df[self.FEATURES])
        self.fitted = True
        return self

    def recommend(self, product_id: str, top_n: int = 5) -> pd.DataFrame:
        idx = self.df[self.df["product_id"] == product_id].index
        if len(idx) == 0:
            raise ValueError(f"Product '{product_id}' not found.")
        idx   = idx[0]
        query = self.matrix[idx]
        scores = [(i, 1 - cosine(query, self.matrix[i]))
                  for i in range(len(self.matrix)) if i != idx]
        scores.sort(key=lambda x: x[1], reverse=True)
        top = [s[0] for s in scores[:top_n]]
        result = self.df.iloc[top][["product_id","name","category","price","rating","units_sold"]].copy()
        result["similarity"] = [round(scores[i][1], 3) for i in range(top_n)]
        return result

    def top_performers(self, n: int = 10) -> pd.DataFrame:
        df = self.df.copy()
        df["score"] = df["units_sold"] * df["rating"] / (df["return_rate"] + 0.01)
        return df.nlargest(n, "score")[["product_id","name","category",
                                        "price","units_sold","rating","return_rate","score"]]


# ══════════════════════════════════════════════════════════════════════════════
#  ANOMALY DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class AnomalyDetector:

    def __init__(self, contamination: float = 0.05):
        self.model  = IsolationForest(contamination=contamination,
                                      random_state=42, n_estimators=100)
        self.fitted = False

    def fit_predict(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        daily = (sales_df.groupby("date")
                         .agg(revenue=("revenue","sum"),
                              units=("units","sum"),
                              transactions=("revenue","count"))
                         .reset_index())
        X = daily[["revenue","units","transactions"]].values
        daily["anomaly"]       = self.model.fit_predict(X)
        daily["anomaly_score"] = self.model.score_samples(X)
        daily["is_anomaly"]    = daily["anomaly"] == -1
        self.fitted = True
        return daily
