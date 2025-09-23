import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

# ==============================
# Charger les données
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("df_weekly_clean.csv", parse_dates=["ds"])
    return df

df = load_data()

# ==============================
# Sidebar - Filtres
# ==============================
st.sidebar.header("Filtres")

regions = df["region"].dropna().unique()
region_selected = st.sidebar.selectbox("Région", ["Toutes"] + sorted(regions.tolist()))

agences = df[df["region"] == region_selected]["agence"].dropna().unique() if region_selected != "Toutes" else df["agence"].dropna().unique()
agence_selected = st.sidebar.selectbox("Agence", ["Toutes"] + sorted(agences.tolist()))

gabs = df[df["agence"] == agence_selected]["lib_gab"].dropna().unique() if agence_selected != "Toutes" else df["lib_gab"].dropna().unique()
gab_selected = st.sidebar.selectbox("GAB", ["Tous"] + sorted(gabs.tolist()))

date_min = df["ds"].min()
date_max = df["ds"].max()
date_debut = st.sidebar.date_input("Date de début", date_min)
date_fin = st.sidebar.date_input("Date de fin", date_max)

# ==============================
# Filtrage
# ==============================
df_filtered = df.copy()

if region_selected != "Toutes":
    df_filtered = df_filtered[df_filtered["region"] == region_selected]

if agence_selected != "Toutes":
    df_filtered = df_filtered[df_filtered["agence"] == agence_selected]

if gab_selected != "Tous":
    df_filtered = df_filtered[df_filtered["lib_gab"] == gab_selected]

df_filtered = df_filtered[(df_filtered["ds"] >= pd.to_datetime(date_debut)) & 
                          (df_filtered["ds"] <= pd.to_datetime(date_fin))]

# ==============================
# KPIs
# ==============================
st.title("📊 Tableau de bord GAB")

if not df_filtered.empty:
    col1, col2, col3 = st.columns(3)
    with col1:
        total_retrait = df_filtered["total_montant"].sum()
        st.metric("Montant total retiré", f"{total_retrait:,.0f} MAD")
    with col2:
        total_op = df_filtered["total_nombre"].sum()
        st.metric("Nombre d'opérations", f"{total_op:,}")
    with col3:
        nb_gab = df_filtered["num_gab"].nunique()
        st.metric("Nombre de GAB", nb_gab)

    # ==============================
    # Graphiques
    # ==============================
    st.subheader("📈 Évolution hebdomadaire des retraits")
    fig1 = px.line(df_filtered, x="ds", y="total_montant", color="lib_gab",
                   title="Évolution hebdomadaire des retraits")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📊 Évolution hebdomadaire du nombre d’opérations")
    fig2 = px.line(df_filtered, x="ds", y="total_nombre", color="lib_gab",
                   title="Évolution hebdomadaire du nombre d’opérations")
    st.plotly_chart(fig2, use_container_width=True)

    # ==============================
    # Prédictions LSTM
    # ==============================
    st.subheader("🤖 Prévision LSTM pour un GAB")

    if gab_selected != "Tous":
        gab_num = df[df["lib_gab"] == gab_selected]["num_gab"].iloc[0]  # retrouver le numéro pour charger le modèle
        model_path = f"lstm_gab_{gab_num}.h5"
        scaler_path = f"scaler_gab_{gab_num}.save"

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            data_gab = df[df["num_gab"] == gab_num][["ds", "total_montant"]].sort_values("ds")

            if len(data_gab) >= 52:
                # Charger scaler et modèle
                scaler = joblib.load(scaler_path)
                model = load_model(model_path)

                values = data_gab["total_montant"].values.reshape(-1, 1)
                scaled_values = scaler.transform(values)

                X_input = scaled_values[-52:].reshape(1, 52, 1)
                y_pred_scaled = model.predict(X_input)
                y_pred = scaler.inverse_transform(y_pred_scaled)

                # Préparer DataFrame résultat
                next_date = data_gab["ds"].max() + timedelta(weeks=1)
                forecast_df = pd.DataFrame({"ds": [next_date], "Prévision retrait": [y_pred[0, 0]]})

                fig3 = px.line(data_gab, x="ds", y="total_montant", title=f"Prévision des retraits - {gab_selected}")
                fig3.add_scatter(x=forecast_df["ds"], y=forecast_df["Prévision retrait"], mode="markers+lines", name="Prévision")
                st.plotly_chart(fig3, use_container_width=True)

                st.success(f"Prévision pour {next_date.date()} : {y_pred[0,0]:,.0f} MAD")

            else:
                st.warning("Pas assez de données pour effectuer une prévision LSTM (minimum 52 semaines).")
        else:
            st.info("⚠️ Aucun modèle LSTM disponible pour ce GAB.")
else:
    st.warning("Aucune donnée disponible pour les filtres sélectionnés.")
