import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# ========================================
# Configuration de la page
# ========================================
st.set_page_config(page_title="Dashboard GAB", layout="wide")

# ========================================
# Charger les données
# ========================================
@st.cache_data
def load_data():
    df = pd.read_csv("df_weekly_clean.csv", parse_dates=["ds"])
    return df

df = load_data()

# ========================================
# Charger les modèles LSTM et scalers des 20 GAB
# ========================================
@st.cache_data
def load_lstm_models():
    models = {}
    scalers = {}
    for model_file in glob.glob("lstm_gab_*.h5"):
        gab_id = model_file.split("_")[-1].replace(".h5","")
        scaler_file = f"scaler_gab_{gab_id}.save"
        models[gab_id] = load_model(model_file)
        scalers[gab_id] = joblib.load(scaler_file)
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
    
    # Filtres de dates
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
    
    # KPIs globaux
    total_retrait = df_filtered["total_montant"].sum() / 1000  # en K
    total_operations = df_filtered["total_nombre"].sum() / 1000  # en K
    nb_gab = df_filtered["num_gab"].nunique()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Montant total retiré (K)", f"{total_retrait:,.1f} K")
    col2.metric("Nombre d'opérations (K)", f"{total_operations:,.1f} K")
    col3.metric("Nombre de GAB", nb_gab)
    
    # Graphiques
    st.subheader("Évolution hebdomadaire des retraits et opérations")
    fig, ax1 = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=df_filtered, x="ds", y="total_montant", ax=ax1, label="Montant retiré")
    ax1.set_ylabel("Montant retiré")
    ax1.tick_params(axis='x', rotation=45)
    ax2 = ax1.twinx()
    sns.lineplot(data=df_filtered, x="ds", y="total_nombre", ax=ax2, color="orange", label="Nombre d'opérations")
    ax2.set_ylabel("Nombre d'opérations")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
    st.pyplot(fig)
    
    # Tableau interactif
    st.subheader("Données filtrées")
    st.dataframe(df_filtered.sort_values("ds", ascending=False))
    st.download_button("Télécharger CSV filtré", df_filtered.to_csv(index=False), "data_filtered.csv", "text/csv")

# ========================================
# Onglet 2 : Prévisions LSTM 20 GAB
# ========================================
if tab == "Prévisions LSTM 20 GAB":
    st.title("Prévisions LSTM - 20 GAB")
    
    # Sélection du GAB
    gab_options = sorted(list(lstm_models.keys()))
    gab_selected = st.selectbox("Sélectionner un GAB", gab_options)
    
    df_gab = df[df["lib_gab"] == gab_selected].sort_values("ds")
    
    if len(df_gab) < 52:
        st.warning("Pas assez de données pour effectuer une prévision LSTM (minimum 52 semaines).")
    else:
        st.subheader(f"Visualisation des données et prévisions pour {gab_selected}")
        
        # Normalisation + prédiction
        scaler = lstm_scalers[gab_selected]
        model = lstm_models[gab_selected]
        
        # On prend uniquement la colonne total_montant pour prédiction
        data = df_gab["total_montant"].values.reshape(-1,1)
        data_scaled = scaler.transform(data)
        
        # Prévision LSTM (on fait un simple fit prédictif pour visualisation)
        # Ici, on suppose que le modèle prédit à partir de la série entière
        pred_scaled = model.predict(data_scaled, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)
        
        # Graphique
        fig2, ax = plt.subplots(figsize=(12,5))
        ax.plot(df_gab["ds"], df_gab["total_montant"], label="Montant réel", color="blue")
        ax.plot(df_gab["ds"], pred.flatten(), label="Montant prédit LSTM", color="red")
        ax.set_xlabel("Date")
        ax.set_ylabel("Montant retiré")
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        st.pyplot(fig2)
        
        # Export des prévisions
        df_pred = pd.DataFrame({
            "ds": df_gab["ds"],
            "total_montant_reel": df_gab["total_montant"],
            "total_montant_pred": pred.flatten()
        })
        st.download_button("Télécharger prévisions CSV", df_pred.to_csv(index=False), f"pred_{gab_selected}.csv", "text/csv")
