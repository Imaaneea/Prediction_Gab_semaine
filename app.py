# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras import losses

# =========================
# Interface et logo
# =========================
st.set_page_config(page_title="Prévision des retraits GAB", layout="wide")
st.image("https://www.albaridbank.ma/themes/baridbank/logo.png", use_container_width=True)
st.title("Prévision des retraits GAB avec LSTM")
st.write("Application basée sur le modèle LSTM pour prédire les retraits hebdomadaires des GAB.")

# =========================
# Charger les données
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("df_subset.csv", parse_dates=['ds'])
    return df

df = load_data()

# =========================
# Barre latérale (filtres)
# =========================
st.sidebar.header("Filtres")

# Région
regions = sorted(df['region'].unique())
selected_region = st.sidebar.selectbox("Sélectionnez la région :", regions)

# Agence filtrée par région
agences = sorted(df[df['region']==selected_region]['agence'].unique())
selected_agence = st.sidebar.selectbox("Sélectionnez l'agence :", agences)

# GAB filtré par agence
gabs = sorted(df[df['agence']==selected_agence]['lib_gab'].unique())
selected_gab = st.sidebar.selectbox("Sélectionnez le GAB :", gabs)

# Période
date_min = df['ds'].min()
date_max = df['ds'].max()
start_date = st.sidebar.date_input("Date début", date_min)
end_date = st.sidebar.date_input("Date fin", date_max)

# Filtrer df selon sélection
df_filtered = df[(df['region']==selected_region) &
                 (df['agence']==selected_agence) &
                 (df['lib_gab']==selected_gab) &
                 (df['ds']>=pd.to_datetime(start_date)) &
                 (df['ds']<=pd.to_datetime(end_date))].sort_values('ds')

# =========================
# KPI
# =========================
st.subheader("Tableau de bord")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Montant total retiré", f"{df_filtered['total_montant'].sum():,.0f} MAD")
with col2:
    st.metric("Nombre total de retraits", f"{df_filtered['total_nombre'].sum():,.0f}")
with col3:
    st.metric("Nombre de GAB (région)", df[df['region']==selected_region]['num_gab'].nunique())

# =========================
# Top 10 GAB par montant
# =========================
st.subheader("Top 10 des GAB par montant")
top10 = df[df['region']==selected_region].groupby('lib_gab')['total_montant'].sum().sort_values(ascending=False).head(10).reset_index()
st.table(top10.style.format({"total_montant":"{:.0f}"}))

# =========================
# Évolution hebdomadaire
# =========================
st.subheader(f"Évolution hebdomadaire du GAB : {selected_gab}")
fig_evo = px.line(df_filtered, x='ds', y='total_montant', markers=True, labels={'ds':'Semaine','total_montant':'Montant retrait'})
st.plotly_chart(fig_evo, use_container_width=True)

# =========================
# Charger modèle LSTM + scaler
# =========================
@st.cache_resource
def load_lstm_model():
    model = load_model("lstm_gab_model.h5", compile=False)  # modèle sans compilation
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_lstm_model()

# =========================
# Prévisions LSTM
# =========================
st.subheader("Prévision LSTM pour les prochaines semaines")
n_steps = 4  # nombre de semaines utilisées pour prédiction
forecast_periods = 12

if len(df_filtered) >= n_steps:
    y_values = df_filtered['total_montant'].values.reshape(-1,1)
    y_scaled = scaler.transform(y_values)

    last_seq = y_scaled[-n_steps:].reshape(1,n_steps,1)
    preds_future_scaled = []

    for _ in range(forecast_periods):
        yhat_scaled = model.predict(last_seq, verbose=0)[0,0]
        preds_future_scaled.append(yhat_scaled)
        last_seq = np.append(last_seq[:,1:,:], [[[yhat_scaled]]], axis=1)

    preds_future = scaler.inverse_transform(np.array(preds_future_scaled).reshape(-1,1)).flatten()
    future_dates = pd.date_range(start=df_filtered['ds'].max() + pd.Timedelta(weeks=1), periods=forecast_periods, freq='W-MON')

    df_future = pd.DataFrame({'ds': future_dates, 'Prévision LSTM': preds_future})

    fig_pred = px.line(df_future, x='ds', y='Prévision LSTM', markers=True)
    st.plotly_chart(fig_pred, use_container_width=True)

    # Télécharger prévisions
    st.download_button(
        label="Télécharger prévisions CSV",
        data=df_future.to_csv(index=False),
        file_name=f"forecast_gab_{selected_gab}.csv",
        mime='text/csv'
    )
else:
    st.warning(f"Pas assez de données pour le GAB {selected_gab} (minimum {n_steps} semaines requises).")
