import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import glob
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from io import StringIO, BytesIO

# ========================================
# Configuration page & style
# ========================================
st.set_page_config(page_title="CashGAB : Dashboard GAB", layout="wide")

st.markdown("""
<style>
.main .block-container { padding: 1.2rem 2rem 2rem 2rem; }
.kpi-card { background: #ffffff; border-radius: 10px; padding: 14px; box-shadow: 0 6px 18px rgba(30,58,138,0.06); border: 1px solid rgba(30,58,138,0.08); height: 110px; }
.kpi-title { color: #566270; font-size: 13px; margin-bottom: 6px; }
.kpi-value { color: #0b5394; font-size: 26px; font-weight: 700; }
.kpi-sub { color: #8b99a6; font-size: 12px; }
.badge-crit { background:#fdecea; color:#d32f2f; padding:6px 10px; border-radius:12px; font-weight:700; }
.badge-alert { background:#fff8e1; color:#f9a825; padding:6px 10px; border-radius:12px; font-weight:700; }
.badge-norm { background:#e8f5e9; color:#2e7d32; padding:6px 10px; border-radius:12px; font-weight:700; }
.muted { color:#7f8b95; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ========================================
# Charger CSV
# ========================================
@st.cache_data
def load_data_safe(file_path="df_weekly_clean.csv"):
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        content = content.decode("utf-8", errors="ignore")
        # Supprimer BOM ou caractères invisibles
        content = content.lstrip('\ufeff').lstrip('´╗┐')
        df = pd.read_csv(StringIO(content), sep=",", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
    except Exception as e:
        st.error(f"Erreur CSV: {e}")
        return pd.DataFrame()

    if df.empty:
        st.error("Le CSV est vide ou mal formaté.")
        return pd.DataFrame()

    # Colonnes essentielles
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    else:
        st.error("La colonne 'ds' est absente du CSV.")
        return pd.DataFrame()

    if "num_gab" in df.columns:
        df["num_gab"] = df["num_gab"].astype(str)

    if "y" not in df.columns and "total_montant" in df.columns:
        df["y"] = df["total_montant"]

    df["week_day"] = df["ds"].dt.dayofweek
    df["week"] = df["ds"].dt.isocalendar().week
    df["year"] = df["ds"].dt.year

    return df
    df = load_data_safe()
if df.empty:
    st.stop()

# ========================================
# Charger modèles LSTM + scalers
# ========================================
@st.cache_data
def load_lstm_models():
    models, scalers = {}, {}
    for model_file in glob.glob("lstm_gab_*.h5"):
        gab_id = model_file.split("_")[-1].replace(".h5","")
        scaler_file = f"scaler_gab_{gab_id}.save"
        try:
            models[gab_id] = load_model(model_file, compile=False)
            scalers[gab_id] = joblib.load(scaler_file)
        except Exception:
            continue
    return models, scalers

lstm_models, lstm_scalers = load_lstm_models()

# ========================================
# Sidebar & navigation
# ========================================
st.sidebar.image("https://www.albaridbank.ma/themes/baridbank/logo.png", width=250)
st.sidebar.title("CashGAB")
st.sidebar.markdown("Solution de gestion proactive des GABs")
tab = st.sidebar.radio("Navigation", ["Tableau de bord analytique", "Prévisions LSTM 20 GAB"])

# Filtres globaux
regions = df["region"].dropna().unique() if "region" in df.columns else []
region_filter = st.sidebar.selectbox("Région", ["Toutes"] + sorted(regions.tolist()) if len(regions)>0 else ["Toutes"])
df_region = df[df["region"] == region_filter] if region_filter != "Toutes" else df.copy()
agences = df_region["agence"].dropna().unique() if "agence" in df_region.columns else []
agence_filter = st.sidebar.selectbox("Agence", ["Toutes"] + sorted(agences.tolist()) if len(agences)>0 else ["Toutes"])

# Filtre date
date_min = df["ds"].min()
date_max = df["ds"].max()
st.sidebar.markdown("Période (TDB)")
date_debut = st.sidebar.date_input("Date début", date_min)
date_fin = st.sidebar.date_input("Date fin", date_max)
df_filtered = df.copy()
if region_filter != "Toutes":
    df_filtered = df_filtered[df_filtered["region"] == region_filter]
if agence_filter != "Toutes":
    df_filtered = df_filtered[df_filtered["agence"] == agence_filter]
df_filtered = df_filtered[(df_filtered["ds"] >= pd.to_datetime(date_debut)) & (df_filtered["ds"] <= pd.to_datetime(date_fin))]

# Seuil critique
seuil_input = st.sidebar.number_input("Seuil critique (MAD)", value=0, step=10000)

# ========================================
# Tableau de bord analytique
# ========================================
if tab == "Tableau de bord analytique":
    st.title("CashGAB — Tableau de bord analytique")
    montant_total = df_filtered["total_montant"].sum() if "total_montant" in df_filtered.columns else 0
    nombre_operations = df_filtered["total_nombre"].sum() if "total_nombre" in df_filtered.columns else 0
    nb_gabs = df_filtered["num_gab"].nunique() if "num_gab" in df_filtered.columns else 0

    # Evolution retraits
    st.markdown("### Evolution des retraits")
    if not df_filtered.empty:
        df_evol = df_filtered.groupby("ds")["total_montant"].sum().reset_index()
        fig_evol = go.Figure()
        fig_evol.add_trace(go.Scatter(x=df_evol["ds"], y=df_evol["total_montant"]/1000,
                                      mode="lines+markers", name="Total retraits hebdo"))
        fig_evol.update_layout(title="Evolution hebdomadaire des retraits (K MAD)",
                               xaxis_title="Date", yaxis_title="Montant retiré (K MAD)")
        st.plotly_chart(fig_evol, use_container_width=True)

    # Répartition régionale
    st.markdown("### Répartition régionale & évolution")
    if not df_filtered.empty:
        df_region_pie = df_filtered.groupby("region")["total_montant"].mean().reset_index().sort_values("total_montant", ascending=False)
        if not df_region_pie.empty:
            fig_pie = go.Figure(go.Bar(
                x=df_region_pie["region"],
                y=df_region_pie["total_montant"]/1000,
                text=(df_region_pie["total_montant"]/1000).round(1),
                textposition='auto',
                marker_color='rgb(26, 118, 255)'
            ))
            fig_pie.update_layout(title="Montant moyen hebdo par région (K MAD)",
                                  xaxis_title="Région", yaxis_title="Montant moyen (K MAD)")
            st.plotly_chart(fig_pie, use_container_width=True)

    # KPIs et alertes
    df_latest = df_filtered.loc[df_filtered.groupby('num_gab')['ds'].idxmax()].copy() if not df_filtered.empty else pd.DataFrame()
    if not df_latest.empty:
        if seuil_input == 0:
            df_avg = df_filtered.groupby("num_gab")["total_montant"].mean().reset_index()
            df_latest = df_latest.merge(df_avg, on="num_gab", suffixes=('', '_avg'))
            df_latest["seuil_critique"] = df_latest["total_montant_avg"]
        else:
            df_latest["seuil_critique"] = seuil_input

        nb_critique = df_latest[df_latest["total_montant"] < df_latest["seuil_critique"]]["num_gab"].nunique()
        dispo_proxy = (df_latest.shape[0] - nb_critique) / df_latest.shape[0] * 100

        k1, k2, k3, k4 = st.columns([2,2,2,2])
        k1.markdown(f"""<div class="kpi-card"><div class="kpi-title">Montant total retraits</div><div class="kpi-value">{montant_total/1_000_000:,.2f} M MAD</div><div class="kpi-sub">Période filtrée</div></div>""", unsafe_allow_html=True)
        k2.markdown(f"""<div class="kpi-card"><div class="kpi-title">Nombre total opérations</div><div class="kpi-value">{nombre_operations:,.0f}</div><div class="kpi-sub">Période filtrée</div></div>""", unsafe_allow_html=True)
        k3.markdown(f"""<div class="kpi-card"><div class="kpi-title">Nombre GAB (réseau)</div><div class="kpi-value">{nb_gabs}</div><div class="kpi-sub">Filtré</div></div>""", unsafe_allow_html=True)
        k4.markdown(f"""<div class="kpi-card"><div class="kpi-title">Disponibilité (proxy)</div><div class="kpi-value">{dispo_proxy:.0f}%</div><div class="kpi-sub">GAB au-dessus du seuil</div></div>""", unsafe_allow_html=True)

# ========================================
# Prévisions LSTM
# ========================================
if tab == "Prévisions LSTM 20 GAB":
    st.title("Prévisions LSTM - 20 GAB")
    st.sidebar.header("Paramètres de simulation")
    gab_options = [gab for gab in sorted(df["num_gab"].unique()) if gab in lstm_models]

    if not gab_options:
        st.warning("Aucun GAB disponible avec modèles LSTM.")
    else:
        gab_selected = st.sidebar.selectbox("Sélectionner un GAB", gab_options)
        period_forecast = st.sidebar.selectbox("Période de prévision", [1,2,4,6])
        variation = st.sidebar.slider("Facteur de variation (%)", -50,50,0)

        df_gab = df[df["num_gab"] == gab_selected].sort_values("ds")
        if len(df_gab) < 52:
            st.warning("Pas assez de données pour la prévision LSTM (min 52 semaines).")
        else:
            try:
                n_steps = 4
                scaler = lstm_scalers[gab_selected]
                model = lstm_models[gab_selected]

                y_scaled = scaler.transform(df_gab[['y']].values)
                X = np.array([y_scaled[i:i+n_steps] for i in range(len(y_scaled)-n_steps)]).reshape(-1,n_steps,1)
                y_pred_scaled = model.predict(X, verbose=0)
                y_pred = scaler.inverse_transform(y_pred_scaled)
                y_true = df_gab['y'].values[n_steps:]
                dates = df_gab['ds'][n_steps:]

                last_seq = y_scaled[-n_steps:].reshape(1,n_steps,1)
                future_preds = []
                future_dates = [df_gab["ds"].max() + pd.Timedelta(weeks=i+1) for i in range(period_forecast)]
                for _ in range(period_forecast):
                    pred_scaled = model.predict(last_seq, verbose=0)
                    pred = scaler.inverse_transform(pred_scaled)[0,0]*(1+variation/100)
                    future_preds.append(pred/1000)
                    last_seq = np.concatenate([last_seq[:,1:,:], pred_scaled.reshape(1,1,1)], axis=1)

                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=dates, y=y_true/1000, mode="lines+markers", name="Montant réel"))
                fig_pred.add_trace(go.Scatter(x=dates, y=y_pred.flatten()/1000, mode="lines+markers", name="Prédiction LSTM"))
                fig_pred.add_trace(go.Scatter(x=future_dates, y=future_preds, mode="lines+markers", name=f"Prévisions ajustées ({variation}%)"))
                fig_pred.update_layout(title=f"Prévision LSTM GAB {gab_selected}", xaxis_title="Date", yaxis_title="Montant retiré (K MAD)")
                st.plotly_chart(fig_pred, use_container_width=True)

            except Exception as e:
                st.error(f"Erreur lors des prévisions: {e}")
