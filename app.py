# app.py (version complète corrigée et design pro)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import glob
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# =========================
# Page config + CSS
# =========================
st.set_page_config(page_title="CashGAB - Dashboard", layout="wide")
st.markdown(
    """
    <style>
    body {background-color: #f5f7fa;}
    /* Container spacing */
    .main .block-container { padding: 1.5rem 2rem 2rem 2rem; }
    /* KPI card */
    .kpi-card { background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%); border-radius: 12px; padding: 20px; box-shadow: 0 6px 18px rgba(30,58,138,0.06); border: 1px solid rgba(30,58,138,0.08); }
    .kpi-title { color:#556677; font-size:14px; margin-bottom:6px; font-weight:600; }
    .kpi-value { color:#0b5394; font-size:30px; font-weight:800; }
    .kpi-sub { color:#8899a6; font-size:12px; }
    /* status badges */
    .badge-crit { background:#d32f2f; color:white; padding:6px 12px; border-radius:12px; font-weight:700; }
    .badge-alert { background:#f9a825; color:black; padding:6px 12px; border-radius:12px; font-weight:700; }
    .badge-norm { background:#2e7d32; color:white; padding:6px 12px; border-radius:12px; font-weight:700; }
    /* small text */
    .muted { color:#7f8b95; font-size:13px; }
    h2 { color:#0b5394; font-weight:700; }
    h3 { color:#0b5394; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Load data function
# =========================
@st.cache_data
def load_data(path="df_weekly_clean.csv"):
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", sep=",", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
    except Exception as e:
        st.error(f"Erreur lecture CSV: {e}")
        return pd.DataFrame()
    if df.empty:
        st.error("CSV vide ou non trouvé.")
        return pd.DataFrame()
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    else:
        st.error("Colonne 'ds' absente.")
        return pd.DataFrame()
    if "num_gab" in df.columns:
        df["num_gab"] = df["num_gab"].astype(str)
    if "total_montant" not in df.columns:
        st.error("Colonne 'total_montant' absente.")
        return pd.DataFrame()
    df["week"] = df["ds"].dt.isocalendar().week
    df["year"] = df["ds"].dt.year
    if "y" not in df.columns:
        df["y"] = df["total_montant"]
    return df

df = load_data()
if df.empty:
    st.stop()

# =========================
# Load LSTM models
# =========================
@st.cache_data
def load_lstm_models(pattern="lstm_gab_*.h5"):
    models, scalers = {}, {}
    for model_file in glob.glob(pattern):
        gab_id = model_file.split("_")[-1].replace(".h5","")
        scaler_file = f"scaler_gab_{gab_id}.save"
        try:
            models[gab_id] = load_model(model_file, compile=False)
            scalers[gab_id] = joblib.load(scaler_file)
        except Exception:
            continue
    return models, scalers

lstm_models, lstm_scalers = load_lstm_models()

# =========================
# Sidebar - filters & nav
# =========================
st.sidebar.image("https://www.albaridbank.ma/themes/baridbank/logo.png", width=220)
st.sidebar.title("CashGAB")
tab = st.sidebar.radio("Navigation", ["Tableau de bord analytique", "Prévisions LSTM 20 GAB"])

# Filters
st.sidebar.markdown("---")
regions = ["Toutes"] + sorted(df["region"].dropna().unique().tolist()) if "region" in df.columns else ["Toutes"]
region_filter = st.sidebar.selectbox("Région", regions)
df_region = df[df["region"] == region_filter] if (region_filter != "Toutes") else df.copy()
agences = ["Toutes"] + sorted(df_region["agence"].dropna().unique().tolist()) if "agence" in df_region.columns else ["Toutes"]
agence_filter = st.sidebar.selectbox("Agence", agences)
st.sidebar.markdown("Période (TDB)")
date_min = df["ds"].min().date()
date_max = df["ds"].max().date()
date_range = st.sidebar.date_input("Date début / fin", [date_min, date_max])
st.sidebar.markdown("---")
user_seuil = st.sidebar.number_input("Seuil critique global (MAD) — facultatif (0 = auto)", value=0, step=10000)

# Apply filters
df_filtered = df.copy()
if region_filter != "Toutes":
    df_filtered = df_filtered[df_filtered["region"] == region_filter]
if agence_filter != "Toutes":
    df_filtered = df_filtered[df_filtered["agence"] == agence_filter]
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
df_filtered = df_filtered[(df_filtered["ds"] >= start_date) & (df_filtered["ds"] <= end_date)]

# Latest per GAB
df_latest = df_filtered.loc[df_filtered.groupby("num_gab")["ds"].idxmax()].copy() if not df_filtered.empty else pd.DataFrame()

# Seuil critique
df_avg_gab = df_filtered.groupby("num_gab")["total_montant"].mean().to_dict() if not df_filtered.empty else {}
def get_seuil_for_gab(gab_id):
    return user_seuil if user_seuil>0 else df_avg_gab.get(gab_id, 100000)
if not df_latest.empty:
    df_latest["seuil_critique"] = df_latest["num_gab"].apply(get_seuil_for_gab)
    def classify(row):
        s,v=row["seuil_critique"], row["total_montant"]
        return "Critique" if v<s else ("Alerte" if v<2*s else "Normal")
    df_latest["status"] = df_latest.apply(classify, axis=1)

# =========================
# Main dashboard
# =========================
if tab == "Tableau de bord analytique":
    st.title("CashGAB — Tableau de bord analytique")

    # ---------- KPIs ----------
    montant_total = df_filtered["total_montant"].sum() if not df_filtered.empty else 0
    nombre_ops = int(df_filtered["total_nombre"].sum()) if "total_nombre" in df_filtered.columns and not df_filtered.empty else int(df_filtered.shape[0])
    nb_gabs = df_filtered["num_gab"].nunique() if not df_filtered.empty else 0
    pct_crit = (df_latest["status"].eq("Critique").sum()/df_latest.shape[0]*100) if not df_latest.empty else 0

    k1,k2,k3,k4 = st.columns(4)
    k1.markdown(f'<div class="kpi-card"><div class="kpi-title">Montant total retraits</div><div class="kpi-value">{montant_total/1_000_000:,.2f} M MAD</div><div class="kpi-sub">Période sélectionnée</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card"><div class="kpi-title">Nombre d\'opérations</div><div class="kpi-value">{nombre_ops:,}</div><div class="kpi-sub">Période sélectionnée</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card"><div class="kpi-title">Nombre GAB</div><div class="kpi-value">{nb_gabs}</div><div class="kpi-sub">Filtrés</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi-card"><div class="kpi-title">GAB critiques (%)</div><div class="kpi-value">{pct_crit:.1f}%</div><div class="kpi-sub">Dernière semaine</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # =========================
    # Evolution des retraits
    # =========================
    st.subheader("Évolution des retraits")
    if df_filtered.empty:
        st.info("Aucune donnée pour les filtres sélectionnés.")
    else:
        df_evol = df_filtered.groupby("ds")["total_montant"].sum().reset_index().sort_values("ds")
        df_evol["MA4"] = df_evol["total_montant"].rolling(4).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_evol["ds"], y=df_evol["total_montant"], mode="lines+markers", name="Total retraits"))
        fig.add_trace(go.Scatter(x=df_evol["ds"], y=df_evol["MA4"], mode="lines", name="MA(4)", line=dict(dash="dash", color="orange")))

        fig.update_layout(
            title="Évolution hebdomadaire des retraits avec MA(4)",
            xaxis_title="Date",
            yaxis_title="Montant (MAD)",
            template="plotly_white",
            hovermode="x unified",
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=56, label="2M", step="day", stepmode="backward"),
                        dict(count=182, label="6M", step="day", stepmode="backward"),
                        dict(step="all", label="Tout")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    # =========================
    # Répartition régions
    # =========================
    st.subheader("Répartition des retraits par région")
    if "region" in df_filtered.columns:
        df_reg = df_filtered.groupby("region")["total_montant"].sum().reset_index()
        fig_reg = go.Figure(data=[go.Pie(labels=df_reg["region"], values=df_reg["total_montant"], hole=0.3)])
        fig_reg.update_layout(template="plotly_white", title="Répartition par région")
        st.plotly_chart(fig_reg, use_container_width=True)

# =========================
# Prévisions LSTM
# =========================
elif tab == "Prévisions LSTM 20 GAB":
    st.title("Prévisions LSTM des GAB")
    gab_list = sorted(lstm_models.keys())
    selected_gab = st.selectbox("Sélectionner un GAB", gab_list)
    periods = st.number_input("Nombre de semaines à prévoir", value=4, step=1)

    if selected_gab in lstm_models:
        model = lstm_models[selected_gab]
        scaler = lstm_scalers[selected_gab]

        # Préparer dernière séquence
        df_gab = df_filtered[df_filtered["num_gab"]==selected_gab].sort_values("ds")
        y_values = df_gab["total_montant"].values.reshape(-1,1)
        scaled = scaler.transform(y_values)
        last_seq = scaled[-4:].reshape(1,4,1)

        # Prédictions
        preds_scaled = []
        seq = last_seq.copy()
        for _ in range(periods):
            pred = model.predict(seq, verbose=0)
            preds_scaled.append(pred[0][0])
            seq = np.append(seq[:,1:,:], [[pred]], axis=1)
        preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
        future_dates = pd.date_range(df_gab["ds"].max()+pd.Timedelta(days=7), periods=periods, freq="W")

        # Plot
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=df_gab["ds"], y=df_gab["total_montant"], mode="lines+markers", name="Historique"))
        fig_pred.add_trace(go.Scatter(x=future_dates, y=preds, mode="lines+markers", name="Prévision", line=dict(color="orange", dash="dash")))
        fig_pred.update_layout(title=f"Prévisions LSTM - GAB {selected_gab}", template="plotly_white", xaxis_title="Date", yaxis_title="Montant (MAD)")
        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.warning("Pas de modèle LSTM disponible pour ce GAB.")
