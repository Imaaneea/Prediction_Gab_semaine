import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# ========================================
# Configuration
# ========================================
st.set_page_config(page_title="Dashboard GAB", layout="wide")

# ========================================
# Chargement des données
# ========================================
@st.cache_data
def load_data():
    df = pd.read_csv("df_weekly_clean.csv", parse_dates=["ds"])
    df["lib_gab"] = df["lib_gab"].astype(str)
    df["week_day"] = df["ds"].dt.dayofweek
    return df

@st.cache_data
def load_subset():
    df_subset = pd.read_csv("df_subset.csv", parse_dates=["ds"])
    df_subset["lib_gab"] = df_subset["lib_gab"].astype(str)
    return df_subset

df = load_data()
df_subset = load_subset()

# ========================================
# Chargement des modèles LSTM
# ========================================
@st.cache_data
def load_lstm_models():
    models = {}
    scalers = {}
    for model_file in glob.glob("lstm_gab_*.h5"):
        gab_id = model_file.split("_")[-1].replace(".h5","")
        scaler_file = f"scaler_gab_{gab_id}.save"
        try:
            models[gab_id] = load_model(model_file, compile=False)
            scalers[gab_id] = joblib.load(scaler_file)
        except Exception as e:
            st.warning(f"Impossible de charger {gab_id}: {e}")
    return models, scalers

lstm_models, lstm_scalers = load_lstm_models()

# ========================================
# Onglets
# ========================================
tab = st.sidebar.radio("Navigation", ["Tableau de bord analytique", "Prévisions LSTM 20 GAB"])

# ========================================
# Onglet 1 : Tableau de bord analytique
# ========================================
if tab == "Tableau de bord analytique":
    st.title("Tableau de bord analytique - GAB")

    # Sidebar filtres
    st.sidebar.header("Filtres")
    regions = df["region"].dropna().unique()
    region = st.sidebar.selectbox("Région", ["Toutes"] + sorted(regions.tolist()))

    if region != "Toutes":
        agences = df[df["region"] == region]["agence"].dropna().unique()
    else:
        agences = df["agence"].dropna().unique()
    agence = st.sidebar.selectbox("Agence", ["Toutes"] + sorted(agences.tolist()))

    if agence != "Toutes":
        gabs = df[df["agence"] == agence]["lib_gab"].dropna().unique()
    else:
        gabs = df["lib_gab"].dropna().unique()
    gab = st.sidebar.selectbox("GAB", ["Tous"] + sorted(gabs.tolist()))

    # Filtre de dates
    date_min = df["ds"].min()
    date_max = df["ds"].max()
    date_debut = st.sidebar.date_input("Date début", date_min)
    date_fin = st.sidebar.date_input("Date fin", date_max)

    # Appliquer filtres
    df_filtered = df.copy()
    if region != "Toutes":
        df_filtered = df_filtered[df_filtered["region"] == region]
    if agence != "Toutes":
        df_filtered = df_filtered[df_filtered["agence"] == agence]
    if gab != "Tous":
        df_filtered = df_filtered[df_filtered["lib_gab"] == gab]
    df_filtered = df_filtered[(df_filtered["ds"] >= pd.to_datetime(date_debut)) &
                              (df_filtered["ds"] <= pd.to_datetime(date_fin))]

    # KPIs
    total_retrait = df_filtered["total_montant"].sum() / 1000
    total_operations = df_filtered["total_nombre"].sum()
    nb_gab = df_filtered["lib_gab"].nunique()
    mean_retrait = df_filtered["total_montant"].mean()
    std_retrait = df_filtered["total_montant"].std()
    weekend_sum = df_filtered[df_filtered["week_day"]>=5]["total_montant"].sum()
    part_weekend = weekend_sum / df_filtered["total_montant"].sum() * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Montant total retiré (K)", f"{total_retrait:,.0f} K")
    col2.metric("Nombre total d'opérations", f"{total_operations:,.0f}")
    col3.metric("Nombre de GAB", nb_gab)
    col4.metric("Montant moyen / semaine", f"{mean_retrait:,.0f}")
    col5.metric("Écart-type retraits", f"{std_retrait:,.0f}")
    st.write(f"Part des retraits pendant le week-end : {part_weekend:.2f}%")

    # Graphiques interactifs
    st.subheader("Évolution hebdomadaire des retraits")
    fig = px.line(df_filtered, x="ds", y="total_montant", color="lib_gab",
                  labels={"ds":"Semaine", "total_montant":"Montant retiré"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Boxplot des retraits par GAB")
    fig2 = px.box(df_filtered, x="lib_gab", y="total_montant", points="all",
                  labels={"lib_gab":"GAB", "total_montant":"Montant retiré"})
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Histogramme du nombre de retraits par GAB")
    ops_per_gab = df_filtered.groupby("lib_gab")["total_nombre"].sum().reset_index()
    fig3 = px.bar(ops_per_gab, x="lib_gab", y="total_nombre",
                  labels={"lib_gab":"GAB", "total_nombre":"Nombre d'opérations"})
    st.plotly_chart(fig3, use_container_width=True)

    # Tableau filtré
    st.subheader("Données filtrées")
    st.dataframe(df_filtered.sort_values("ds", ascending=False))
    st.download_button("Télécharger CSV filtré", df_filtered.to_csv(index=False), "data_filtered.csv", "text/csv")

# ========================================
# Onglet 2 : Prévisions LSTM 20 GAB
# ========================================
if tab == "Prévisions LSTM 20 GAB":
    st.title("Prévisions LSTM - 20 GAB")
    gab_options = sorted(list(lstm_models.keys()))
    gab_selected = st.selectbox("Sélectionner un GAB", gab_options)
    gab_selected = str(gab_selected)

    # Utiliser df_subset pour données historiques
    if gab_selected not in df_subset["lib_gab"].unique():
        st.warning(f"Aucune donnée historique trouvée pour le GAB {gab_selected}")
    else:
        df_gab = df_subset[df_subset["lib_gab"] == gab_selected].sort_values("ds")
        if len(df_gab) < 52:
            st.warning("Pas assez de données pour effectuer une prévision LSTM (minimum 52 semaines).")
        else:
            st.subheader(f"Visualisation des données et prévisions pour {gab_selected}")
            scaler = lstm_scalers[gab_selected]
            model = lstm_models[gab_selected]

            data = df_gab["total_montant"].values.reshape(-1,1)
            data_scaled = scaler.transform(data)
            pred_scaled = model.predict(data_scaled, verbose=0)
            pred = scaler.inverse_transform(pred_scaled)

            # Graphique
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=df_gab["ds"], y=df_gab["total_montant"],
                                          mode="lines+markers", name="Montant réel"))
            fig_pred.add_trace(go.Scatter(x=df_gab["ds"], y=pred.flatten(),
                                          mode="lines+markers", name="Montant prédit LSTM"))
            fig_pred.update_layout(xaxis_title="Date", yaxis_title="Montant retiré")
            st.plotly_chart(fig_pred, use_container_width=True)

            # Export
            df_pred = pd.DataFrame({
                "ds": df_gab["ds"],
                "total_montant_reel": df_gab["total_montant"],
                "total_montant_pred": pred.flatten()
            })
            st.download_button("Télécharger prévisions CSV",
                               df_pred.to_csv(index=False),
                               f"pred_{gab_selected}.csv",
                               "text/csv")
