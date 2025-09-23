# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from tensorflow.keras.models import load_model
import os

# =========================
# Configuration de la page
# =========================
st.set_page_config(page_title="Prévision des retraits GAB", layout="wide")
st.title("Prévision des retraits GAB avec LSTM")
st.write("Application basée sur les modèles LSTM pour prédire les retraits hebdomadaires des GAB.")

st.sidebar.image(
    "https://www.albaridbank.ma/themes/baridbank/logo.png",
    use_container_width=True
)

# =========================
# Charger les données
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("df_weekly_clean.csv", parse_dates=['ds'])
    return df

df = load_data()

# =========================
# Sidebar - Filtres
# =========================
st.sidebar.header("Filtres")

# Région
region_list = sorted(df['region'].dropna().unique())
selected_region = st.sidebar.selectbox("Région :", ["Toutes"] + region_list)

# Filtrer par région pour les autres listes
df_region = df if selected_region == "Toutes" else df[df['region'] == selected_region]

# Agence
agence_list = sorted(df_region['agence'].dropna().unique())
selected_agence = st.sidebar.selectbox("Agence :", ["Toutes"] + agence_list)

# Filtrer par agence pour la liste des GAB
df_agence = df_region if selected_agence == "Toutes" else df_region[df_region['agence'] == selected_agence]

# GAB
gab_list = sorted(df_agence['lib_gab'].dropna().unique())
selected_gab = st.sidebar.selectbox("GAB :", ["Tous"] + gab_list)

# Dates
date_min = df['ds'].min()
date_max = df['ds'].max()
start_date = st.sidebar.date_input("Date début :", date_min)
end_date = st.sidebar.date_input("Date fin :", date_max)

# Filtrer le dataframe selon les choix
df_filtered = df.copy()
if selected_region != "Toutes":
    df_filtered = df_filtered[df_filtered['region'] == selected_region]
if selected_agence != "Toutes":
    df_filtered = df_filtered[df_filtered['agence'] == selected_agence]
if selected_gab != "Tous":
    df_filtered = df_filtered[df_filtered['lib_gab'] == selected_gab]
df_filtered = df_filtered[(df_filtered['ds'] >= pd.to_datetime(start_date)) & 
                          (df_filtered['ds'] <= pd.to_datetime(end_date))]

# =========================
# KPIs dynamiques
# =========================
total_montant = df_filtered['total_montant'].sum()
total_nombre = df_filtered['total_nombre'].sum()
total_gabs = df_filtered['num_gab'].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("Montant total retiré", f"{total_montant:,.0f} MAD")
col2.metric("Nombre total de retraits", f"{total_nombre:,}")
col3.metric("Nombre de GAB", f"{total_gabs:,}")

# =========================
# Graphiques
# =========================
# Top 10 GAB par montant
st.subheader("Top 10 des GAB par montant")
top10 = df_filtered.groupby('lib_gab')['total_montant'].sum().sort_values(ascending=False).head(10).reset_index()
top10['total_montant'] = top10['total_montant'].apply(lambda x: f"{x:,.0f} MAD")
st.dataframe(top10)

# Evolution hebdomadaire
st.subheader("Évolution hebdomadaire des retraits")
df_evo = df_filtered.groupby('ds')['total_montant'].sum().reset_index()
fig = px.line(df_evo, x='ds', y='total_montant', markers=True)
fig.update_layout(xaxis_title="Semaine", yaxis_title="Montant retrait (MAD)")
st.plotly_chart(fig, use_container_width=True)

# =========================
# LSTM - Prévisions
# =========================
@st.cache_resource
def load_lstm_gab_model(model_file, scaler_file):
    model = load_model(model_file, compile=False)
    scaler = joblib.load(scaler_file)
    return model, scaler

# Vérifier si un GAB est sélectionné
if selected_gab != "Tous" and not df_filtered.empty:
    gab_num = df_filtered['num_gab'].iloc[0]
    model_file = f"lstm_gab_{gab_num}.h5"
    scaler_file = f"scaler_gab_{gab_num}.save"

    if os.path.exists(model_file) and os.path.exists(scaler_file):
        model, scaler = load_lstm_gab_model(model_file, scaler_file)
        
        df_filtered = df_filtered.sort_values("ds")
        y_values = df_filtered['total_montant'].values.reshape(-1,1)
        n_steps = 52
        forecast_periods = 12

        if len(y_values) >= n_steps:
            y_scaled = scaler.transform(y_values)
            last_seq = y_scaled[-n_steps:].reshape(1, n_steps, 1)
            preds_future_scaled = []

            for _ in range(forecast_periods):
                yhat_scaled = model.predict(last_seq, verbose=0)[0,0]
                preds_future_scaled.append(yhat_scaled)
                last_seq = np.append(last_seq[:,1:,:], [[[yhat_scaled]]], axis=1)

            preds_future = scaler.inverse_transform(np.array(preds_future_scaled).reshape(-1,1)).flatten()
            future_dates = pd.date_range(start=df_filtered['ds'].max() + pd.Timedelta(weeks=1),
                                         periods=forecast_periods, freq='W-MON')
            
            df_plot = pd.concat([
                pd.DataFrame({'ds': df_filtered['ds'], 'valeur': df_filtered['total_montant'], 'type': 'Historique'}),
                pd.DataFrame({'ds': future_dates, 'valeur': preds_future, 'type': 'Prévision'})
            ])
            
            st.subheader(f"Prévision des retraits pour {selected_gab}")
            fig = px.line(df_plot, x='ds', y='valeur', color='type', markers=True)
            fig.update_layout(xaxis_title="Semaine", yaxis_title="Montant retrait (MAD)")
            st.plotly_chart(fig, use_container_width=True)

            df_download = pd.DataFrame({'ds': future_dates, 'yhat': preds_future, 'lib_gab': selected_gab})
            csv = df_download.to_csv(index=False)
            st.download_button(
                label="Télécharger les prévisions",
                data=csv,
                file_name=f"forecast_gab_{selected_gab}.csv",
                mime='text/csv'
            )
        else:
            st.warning(f"Pas assez de données pour effectuer une prévision LSTM (minimum {n_steps} semaines).")
    else:
        st.warning(f"Modèle ou scaler non trouvé pour le GAB {gab_num}.")
