import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import glob
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import io

# ========================================
# Configuration de la page
# ========================================
st.set_page_config(page_title="Dashboard GAB - Optimisation Cash", layout="wide")

# ========================================
# Chargement des données
# ========================================
@st.cache_data
def load_data():
    df = pd.read_csv("df_weekly_clean.csv", parse_dates=["ds"])
    df["num_gab"] = df["num_gab"].astype(str)
    df["week_day"] = df["ds"].dt.dayofweek
    df["week"] = df["ds"].dt.isocalendar().week
    df["year"] = df["ds"].dt.year
    return df

df = load_data()

# ========================================
# Chargement silencieux des modèles LSTM
# ========================================
@st.cache_data
def load_lstm_models():
    models, scalers = {}, {}
    for model_file in glob.glob("lstm_gab_*.h5"):
        gab_id = model_file.split("_")[-1].replace(".h5", "")
        try:
            models[gab_id] = load_model(model_file, compile=False)
            scalers[gab_id] = joblib.load(f"scaler_gab_{gab_id}.save")
        except Exception as e:
            st.warning(f"Erreur chargement modèle {gab_id}: {e}")
    return models, scalers

lstm_models, lstm_scalers = load_lstm_models()

# ========================================
# Navigation
# ========================================
tab = st.sidebar.radio("Navigation", ["Tableau de bord analytique", "Prévisions LSTM 20 GAB"])

# ========================================
# Onglet 1 : Tableau de bord analytique
# ========================================
if tab == "Tableau de bord analytique":
    st.title("📊 Tableau de bord analytique - Gestion du Cash GAB")

    # Filtres
    st.sidebar.header("Filtres")
    regions = df["region"].dropna().unique()
    region = st.sidebar.selectbox("Région", ["Toutes"] + sorted(regions.tolist()))
    if region != "Toutes":
        agences = df[df["region"] == region]["agence"].dropna().unique()
    else:
        agences = df["agence"].dropna().unique()
    agence = st.sidebar.selectbox("Agence", ["Toutes"] + sorted(agences.tolist()))
    if agence != "Toutes":
        gabs = df[df["agence"] == agence]["num_gab"].dropna().unique()
    else:
        gabs = df["num_gab"].dropna().unique()
    gab = st.sidebar.selectbox("GAB", ["Tous"] + sorted(gabs.tolist()))

    # Dates
    date_min, date_max = df["ds"].min(), df["ds"].max()
    date_debut = st.sidebar.date_input("Date début", date_min)
    date_fin = st.sidebar.date_input("Date fin", date_max)

    # Filtrage
    df_filtered = df.copy()
    if region != "Toutes":
        df_filtered = df_filtered[df_filtered["region"] == region]
    if agence != "Toutes":
        df_filtered = df_filtered[df_filtered["agence"] == agence]
    if gab != "Tous":
        df_filtered = df_filtered[df_filtered["num_gab"] == gab]
    df_filtered = df_filtered[(df_filtered["ds"] >= pd.to_datetime(date_debut)) & (df_filtered["ds"] <= pd.to_datetime(date_fin))]

    # Classification des GABs selon les seuils
    seuil_haut = df_filtered["total_montant"].quantile(0.9)
    seuil_bas = df_filtered["total_montant"].quantile(0.1)
    df_status = df_filtered.groupby("num_gab")["total_montant"].mean().reset_index()
    df_status["statut"] = df_status["total_montant"].apply(lambda x: 
        "Critique" if x >= seuil_haut else ("Alerte" if x <= seuil_bas else "Normal"))

    nb_critique = (df_status["statut"] == "Critique").sum()
    nb_alerte = (df_status["statut"] == "Alerte").sum()
    nb_normal = (df_status["statut"] == "Normal").sum()

    # KPIs
    st.subheader("📈 Indicateurs de performance clés (KPI)")
    volume_moyen = df_filtered["total_montant"].mean() / 1000
    ecart_type = df_filtered["total_montant"].std() / 1000
    taux_evol = df_filtered["total_montant"].pct_change().mean() * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Volume moyen hebdo", f"{volume_moyen:,.0f} KDH")
    col2.metric("Volatilité hebdo", f"{ecart_type:,.1f} KDH")
    col3.metric("Évolution moyenne", f"{taux_evol:.1f} %")
    col4.metric("GABs critiques", f"{nb_critique}")
    col5.metric("GABs en alerte", f"{nb_alerte}")

    if nb_critique > 0:
        st.warning(f"⚠️ {nb_critique} GAB(s) présentent un risque de rupture de cash (niveau critique).")

    # Graphique d'évolution
    st.subheader("📉 Évolution des retraits (avec moyenne mobile)")
    df_plot = df_filtered.groupby("ds")["total_montant"].sum().reset_index()
    df_plot["moyenne_mobile"] = df_plot["total_montant"].rolling(window=4).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot["ds"], y=df_plot["total_montant"]/1000, mode="lines", name="Retraits (KDH)"))
    fig.add_trace(go.Scatter(x=df_plot["ds"], y=df_plot["moyenne_mobile"]/1000, mode="lines", name="Tendance (moy. mobile)", line=dict(dash="dash")))
    fig.update_layout(title="Évolution des retraits hebdomadaires", xaxis_title="Date", yaxis_title="Montant (KDH)")
    st.plotly_chart(fig, use_container_width=True)

    # Export graphique
    buffer = io.BytesIO()
    fig.write_image(buffer, format="png")
    st.download_button("📥 Télécharger le graphique (PNG)", buffer.getvalue(), file_name="evolution_retraits.png")

# ========================================
# Onglet 2 : Prévisions LSTM
# ========================================
elif tab == "Prévisions LSTM 20 GAB":
    st.title("🤖 Prévisions LSTM - Anticipation des besoins en cash")

    gab_options = [gab for gab in sorted(df["num_gab"].unique()) if gab in lstm_models]
    gab_selected = st.selectbox("Sélectionner un GAB", gab_options)

    df_gab = df[df["num_gab"] == gab_selected].sort_values("ds")
    if len(df_gab) < 52:
        st.warning("Pas assez de données pour effectuer une prévision.")
    else:
        try:
            model = lstm_models[gab_selected]
            scaler = lstm_scalers[gab_selected]
            n_steps = 4

            y_scaled = scaler.transform(df_gab[['y']].values)
            X = np.array([y_scaled[i:i+n_steps] for i in range(len(y_scaled)-n_steps)]).reshape(-1, n_steps, 1)
            y_pred = scaler.inverse_transform(model.predict(X))
            y_true = df_gab['y'].values[n_steps:]
            dates = df_gab['ds'][n_steps:]

            # Prévisions futures
            future_steps = 6
            last_seq = y_scaled[-n_steps:].reshape(1, n_steps, 1)
            preds, future_dates = [], []
            for i in range(future_steps):
                pred_scaled = model.predict(last_seq)
                pred = scaler.inverse_transform(pred_scaled)[0, 0]
                preds.append(pred/1000)
                last_seq = np.concatenate([last_seq[:,1:,:], pred_scaled.reshape(1,1,1)], axis=1)
                future_dates.append(df_gab["ds"].max() + pd.Timedelta(weeks=i+1))

            # Graphique
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=dates, y=y_true/1000, mode="lines+markers", name="Réel"))
            fig_pred.add_trace(go.Scatter(x=dates, y=y_pred.flatten()/1000, mode="lines+markers", name="Prédit"))
            fig_pred.add_trace(go.Scatter(x=future_dates, y=preds, mode="lines+markers", name="Prévision future"))
            fig_pred.update_layout(title=f"Prévisions LSTM - GAB {gab_selected}", xaxis_title="Date", yaxis_title="Montant (KDH)")
            st.plotly_chart(fig_pred, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur lors des prévisions : {e}")
