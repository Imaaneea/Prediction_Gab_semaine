# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras import losses
import seaborn as sns
from datetime import datetime

# =========================
# Config de la page
# =========================
st.set_page_config(
    page_title="Prévision des retraits GAB",
    layout="wide"
)

# Logo
st.sidebar.image("https://www.albaridbank.ma/themes/baridbank/logo.png", use_column_width=True)

# =========================
# Charger les données
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("df_subset.csv", parse_dates=['ds'])
    return df

df = load_data()

# =========================
# Sidebar filtres
# =========================
st.sidebar.header("Filtres")
regions = df['region'].unique()
selected_region = st.sidebar.selectbox("Région", np.append(["Toutes"], regions))

agences = df[df['region']==selected_region]['agence'].unique() if selected_region != "Toutes" else df['agence'].unique()
selected_agence = st.sidebar.selectbox("Agence", np.append(["Toutes"], agences))

gabs = df[df['agence']==selected_agence]['lib_gab'].unique() if selected_agence != "Toutes" else df['lib_gab'].unique()
selected_gab = st.sidebar.selectbox("GAB", np.append(["Tous"], gabs))

# Sélecteurs séparés pour période
min_date = df['ds'].min()
max_date = df['ds'].max()
start_date = st.sidebar.date_input("Date de début", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("Date de fin", max_date, min_value=min_date, max_value=max_date)

# Filtrer le dataframe selon les sélections
df_filtered = df.copy()
if selected_region != "Toutes":
    df_filtered = df_filtered[df_filtered['region']==selected_region]
if selected_agence != "Toutes":
    df_filtered = df_filtered[df_filtered['agence']==selected_agence]
if selected_gab != "Tous":
    df_filtered = df_filtered[df_filtered['lib_gab']==selected_gab]

df_filtered = df_filtered[(df_filtered['ds'] >= pd.to_datetime(start_date)) & 
                          (df_filtered['ds'] <= pd.to_datetime(end_date))]

# =========================
# KPIs
# =========================
total_montant = df_filtered['total_montant'].sum()
total_nombre = df_filtered['total_nombre'].sum()
num_gabs_region = df_filtered.groupby('region')['num_gab'].nunique().sum()

col1, col2, col3 = st.columns(3)
col1.metric("Montant total retiré", f"{total_montant:,.0f} MAD")
col2.metric("Nombre total de retraits", f"{total_nombre:,.0f}")
col3.metric("Nombre de GAB (région)", f"{num_gabs_region}")

# =========================
# Top 10 des GAB par montant
# =========================
st.subheader("Top 10 des GAB par montant")
top10 = df_filtered.groupby('lib_gab')['total_montant'].sum().sort_values(ascending=False).head(10)
st.table(top10.reset_index().rename(columns={'lib_gab':'GAB','total_montant':'Montant total'}))

# =========================
# Historique par semaine pour GAB sélectionné
# =========================
st.subheader("Évolution hebdomadaire des retraits")

if selected_gab != "Tous":
    df_gab = df_filtered[df_filtered['lib_gab']==selected_gab].sort_values('ds')
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_gab['ds'], df_gab['total_montant'], marker='o', label='Montant retrait')
    ax.set_xlabel("Semaine")
    ax.set_ylabel("Montant retrait")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
else:
    st.write("Sélectionnez un GAB pour afficher l'évolution hebdomadaire.")

# =========================
# Charger modèle et scaler
# =========================
@st.cache_resource
def load_lstm_model():
    model = load_model("lstm_gab_model.h5", custom_objects={'mse': losses.MeanSquaredError})
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_lstm_model()

# =========================
# Prévisions LSTM pour le GAB sélectionné
# =========================
if selected_gab != "Tous":
    n_steps = 4  # semaines utilisées pour l'entraînement
    forecast_periods = 12

    y_values = df_gab['y'].values.reshape(-1,1)
    y_scaled = scaler.transform(y_values)
    last_seq = y_scaled[-n_steps:].reshape(1,n_steps,1)
    preds_future_scaled = []

    for _ in range(forecast_periods):
        yhat_scaled = model.predict(last_seq, verbose=0)[0,0]
        preds_future_scaled.append(yhat_scaled)
        last_seq = np.append(last_seq[:,1:,:], [[[yhat_scaled]]], axis=1)

    preds_future = scaler.inverse_transform(np.array(preds_future_scaled).reshape(-1,1)).flatten()
    future_dates = pd.date_range(start=df_gab['ds'].max() + pd.Timedelta(weeks=1), periods=forecast_periods, freq='W-MON')
    df_future = pd.DataFrame({'ds': future_dates, 'yhat': preds_future})

    st.subheader("Prévision des retraits (prochaines semaines)")
    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(df_gab['ds'], df_gab['y'], marker='o', label='Historique')
    ax2.plot(df_future['ds'], df_future['yhat'], marker='x', label='Prévision LSTM')
    ax2.set_xlabel("Semaine")
    ax2.set_ylabel("Montant retrait")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    # Télécharger les résultats
    st.subheader("Télécharger les prévisions")
    df_download = df_future.copy()
    df_download['lib_gab'] = selected_gab
    csv = df_download.to_csv(index=False)
    st.download_button(
        label="Télécharger CSV",
        data=csv,
        file_name=f"forecast_gab_{selected_gab}.csv",
        mime='text/csv'
    )
