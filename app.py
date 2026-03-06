"""
app.py
───────
AI Business Analytics Dashboard — entry point.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="AI Business Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Mono', monospace !important; }
[data-testid="stSidebar"] { background: #0d1017 !important; border-right: 1px solid #1e2130 !important; }
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }
[data-testid="metric-container"] { background: #1a1d2e; border: 1px solid #2a2d3e; border-radius: 10px; padding: 14px 18px !important; }
[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="stDataFrame"] { border: 1px solid #2a2d3e; border-radius: 8px; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }
hr { border-color: #2a2d3e !important; }
[data-testid="stPlotlyChart"] { background: #1a1d2e; border: 1px solid #2a2d3e; border-radius: 10px; padding: 4px; }
[data-testid="stExpander"] { background: #1a1d2e; border: 1px solid #2a2d3e !important; border-radius: 10px; }
[data-testid="stSpinner"] > div { border-top-color: #6366f1 !important; }
</style>
""", unsafe_allow_html=True)

from src.data.generator import load_all_data
from src.data.loader import load_from_uploaded_file, derive_customers_from_sales
from src.services.ml_models import (
    SalesForecaster, CustomerSegmenter, ProductRecommender, AnomalyDetector,
)
from src.services.charts import compute_kpis
from src.components.pages import (
    page_overview, page_forecast, page_customers, page_products, page_anomalies,
)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_sales(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError("Colonne 'date' manquante après mapping.")
    if "revenue" not in df.columns:
        raise ValueError("Colonne 'revenue' manquante après mapping.")
    df = df.copy()
    df["date"]    = pd.to_datetime(df["date"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)
    df = df.dropna(subset=["date"])
    for col, val in [("units",1), ("category","Unknown"), ("region","Unknown"),
                     ("channel","Unknown"), ("profit_margin",0.25), ("discount",0.0)]:
        if col not in df.columns:
            df[col] = val
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(1).astype(int)
    return df[["date","category","region","channel","revenue","units","profit_margin","discount"]]


# ══════════════════════════════════════════════════════════════════════════════
#  UPLOAD PAGE
# ══════════════════════════════════════════════════════════════════════════════

def page_upload_data():
    st.markdown("## 📂 Charger vos données réelles")
    st.caption("Uploadez un CSV ou Excel pour analyser vos propres données.")

    if st.session_state.get("use_real_data"):
        st.success("✅ Données réelles actives — navigue vers les autres pages.")
        if st.button("🔄 Revenir aux données synthétiques"):
            for k in ["real_sales","real_customers","real_products","use_real_data"]:
                st.session_state.pop(k, None)
            st.rerun()
        return

    st.info("**Colonnes minimales requises :** une colonne `date` + une colonne montant (`revenue`, `amount`, `total`...)")

    sales_file = st.file_uploader("📊 Upload fichier de ventes (CSV ou Excel)",
                                   type=["csv","xlsx","xls"])

    if not sales_file:
        st.divider()
        st.markdown("#### 📋 Template CSV attendu")
        example = pd.DataFrame({
            "date":     ["2024-01-15","2024-01-16","2024-01-17"],
            "revenue":  [1250.00, 340.50, 890.00],
            "category": ["Electronics","Clothing","Sports"],
            "region":   ["North","South","East"],
            "channel":  ["Online","In-Store","Mobile App"],
            "units":    [2, 1, 3],
        })
        st.dataframe(example, hide_index=True, use_container_width=True)
        st.download_button("⬇️ Télécharger ce template", example.to_csv(index=False),
                           "template_ventes.csv", "text/csv")
        return

    try:
        raw = load_from_uploaded_file(sales_file)
        st.success(f"✓ {len(raw):,} lignes · {len(raw.columns)} colonnes")
        with st.expander("👁 Aperçu"):
            st.dataframe(raw.head(5), hide_index=True, use_container_width=True)
    except Exception as e:
        st.error(f"Impossible de lire le fichier : {e}")
        return

    # ── Mapping colonnes ──
    st.markdown("#### 🔧 Mapper vos colonnes")
    opts = ["(aucune)"] + list(raw.columns)

    def auto(candidates):
        for c in candidates:
            if c in raw.columns:
                return opts.index(c)
        return 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        date_col = st.selectbox("📅 DATE *", opts,
            index=auto(["date","order_date","transaction_date","Date","DATE","created_at"]))
    with c2:
        rev_col = st.selectbox("💰 REVENUE *", opts,
            index=auto(["revenue","amount","total","price","Revenue","Amount","sales"]))
    with c3:
        cat_col = st.selectbox("🏷 CATÉGORIE", opts,
            index=auto(["category","Category","dept","product_category","type"]))
    with c4:
        reg_col = st.selectbox("📍 RÉGION", opts,
            index=auto(["region","Region","location","city","store","country"]))
    with c5:
        ch_col = st.selectbox("📡 CANAL", opts,
            index=auto(["channel","Channel","source","platform","sales_channel"]))

    st.markdown("#### 👥 ID Client *(optionnel — pour segmentation)*")
    cust_opts = ["(aucune)"] + list(raw.columns)
    cust_col = st.selectbox("Colonne client", cust_opts,
        index=auto(["customer_id","client_id","user_id","CustomerID","email"]))

    st.divider()
    if st.button("✅ Lancer l'analyse", type="primary", use_container_width=True):
        with st.spinner("Normalisation en cours..."):
            try:
                df     = raw.copy()
                rename = {}
                if date_col != "(aucune)": rename[date_col] = "date"
                if rev_col  != "(aucune)": rename[rev_col]  = "revenue"
                if cat_col  != "(aucune)": rename[cat_col]  = "category"
                if reg_col  != "(aucune)": rename[reg_col]  = "region"
                if ch_col   != "(aucune)": rename[ch_col]   = "channel"
                if cust_col != "(aucune)": rename[cust_col] = "customer_id"
                df = df.rename(columns=rename)

                sales_df = _normalize_sales(df)
                st.session_state["real_sales"] = sales_df

                if "customer_id" in df.columns:
                    df["date"]    = pd.to_datetime(df["date"], errors="coerce")
                    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)
                    customers_df  = derive_customers_from_sales(df, "customer_id")
                    st.session_state["real_customers"] = customers_df

                st.session_state["use_real_data"] = True
                st.success(f"✅ {len(sales_df):,} transactions chargées !")
                st.balloons()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Erreur : {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  DATA & MODELS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="⚙️ Chargement des données synthétiques…")
def get_synthetic_data():
    return load_all_data()


def get_active_data() -> dict:
    if st.session_state.get("use_real_data") and "real_sales" in st.session_state:
        synth = get_synthetic_data()
        return {
            "sales":     st.session_state["real_sales"],
            "customers": st.session_state.get("real_customers", synth["customers"]),
            "products":  st.session_state.get("real_products",  synth["products"]),
        }
    return get_synthetic_data()


@st.cache_resource(show_spinner="🧠 Entraînement des modèles ML…")
def train_models(sales_hash: int, customer_hash: int, product_hash: int) -> dict:
    data        = get_active_data()
    forecaster  = SalesForecaster().fit(data["sales"])
    forecast    = forecaster.predict(periods=90)
    segmenter   = CustomerSegmenter().fit(data["customers"])
    segmented   = segmenter.predict(data["customers"])
    recommender = ProductRecommender().fit(data["products"])
    detector    = AnomalyDetector(contamination=0.05)
    anomalies   = detector.fit_predict(data["sales"])
    kpis        = compute_kpis(data["sales"])
    return dict(forecaster=forecaster, forecast=forecast, segmenter=segmenter,
                segmented=segmented, recommender=recommender,
                anomalies=anomalies, kpis=kpis)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(kpis: dict) -> str:
    with st.sidebar:
        st.markdown("""
        <div style="padding:6px 0 18px">
            <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:800;
                        color:#e2e8f0;letter-spacing:-0.03em">📊 AI Dashboard</div>
            <div style="font-size:10px;color:#4a5568;letter-spacing:0.1em;margin-top:2px">
                BUSINESS INTELLIGENCE</div>
        </div>""", unsafe_allow_html=True)

        if st.session_state.get("use_real_data"):
            st.markdown('<div style="background:rgba(16,185,129,0.15);border:1px solid '
                        'rgba(16,185,129,0.4);border-radius:6px;padding:6px 10px;font-size:11px;'
                        'color:#10b981;margin-bottom:12px">● Données réelles actives</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:rgba(99,102,241,0.1);border:1px solid '
                        'rgba(99,102,241,0.3);border-radius:6px;padding:6px 10px;font-size:11px;'
                        'color:#6366f1;margin-bottom:12px">○ Mode démo</div>',
                        unsafe_allow_html=True)

        page = st.radio("Navigate", [
            "🏠 Overview", "🔮 Sales Forecast", "👥 Customers",
            "📦 Products", "🔍 Anomalies", "📂 Charger mes données"
        ], label_visibility="collapsed")

        st.divider()
        st.markdown('<div style="font-size:10px;color:#4a5568;letter-spacing:0.1em">MÉTRIQUES</div>',
                    unsafe_allow_html=True)
        st.metric("Revenue (30j)", f"${kpis['this_month_rev']:,.0f}",
                  f"{kpis['mom_change']:+.1f}% MoM")
        st.metric("Marge moy.", f"{kpis['avg_margin']:.1f}%")
        st.divider()
        st.markdown('<div style="font-size:10px;color:#2a3040;text-align:center">'
                    'Scikit-learn · Plotly · Streamlit</div>', unsafe_allow_html=True)
    return page


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    data          = get_active_data()
    sales_hash    = hash(str(data["sales"].shape) + str(data["sales"]["date"].max()))
    customer_hash = hash(str(data["customers"].shape))
    product_hash  = hash(str(data["products"].shape))
    models        = train_models(sales_hash, customer_hash, product_hash)
    page          = render_sidebar(models["kpis"])

    if   page == "🏠 Overview":           page_overview(data["sales"], models["kpis"])
    elif page == "🔮 Sales Forecast":     page_forecast(data["sales"], models["forecaster"], models["forecast"])
    elif page == "👥 Customers":          page_customers(data["customers"], models["segmenter"], models["segmented"])
    elif page == "📦 Products":           page_products(data["products"], models["recommender"])
    elif page == "🔍 Anomalies":          page_anomalies(data["sales"], models["anomalies"])
    elif page == "📂 Charger mes données": page_upload_data()


if __name__ == "__main__":
    main()
