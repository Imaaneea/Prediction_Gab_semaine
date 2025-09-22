# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras import losses

# =========================
# Logo et titre
# =========================
st.image("https://www.albaridbank.ma/themes/baridbank/logo.png", width=200)
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
# Sidebar - Filtres
# =========================
st.sidebar.header("Filtres")
gab_list = df['lib_gab'].unique()
selected_gab = st.sidebar.selectbox("GAB :", gab_list)

agence_list = df['agence'].unique()
selected_agence = st.sidebar.multiselect("Agence :", agence_list, default=agence_list)

region_list = df['region'].unique()
selected_region = st.sidebar.multiselect("Région :", region_list, default=region_list)

date_min = df['ds'].min()
date_max = df['ds'].max()
start_date = st.sidebar.date_input("Date début :", date_min)
end_date = st.sidebar.date_input("Date fin :", date_max)

# Filtrer les données
df_filtered = df[
    (df['lib_gab'] == selected_gab) &
    (df['agence'].isin(selected_agence)) &
    (df['region'].isin(selected_region)) &
    (df['ds'] >= pd.to_datetime(start_date)) &
    (df['ds'] <= pd.to_datetime(end_date))
]

# =========================
# KPI
# =========================
st.subheader("Tableau de bord")
col1, col2, col3 = st.columns(3)

total_montant = df_filtered['total_montant'].sum()
total_nombre = df_filtered['total_nombre'].sum()
total_gabs_region = df_filtered.groupby('region')['num_gab'].nunique().sum()

col1.metric("Montant total retiré", f"{total_montant:,.0f} MAD")
col2.metric("Nombre total de retraits", f"{total_nombre:,.0f}")
col3.metric("Nombre de GAB (région)", f"{total_gabs_region}")

# Top 10 des GAB par montant
st.subheader("Top 10 des GAB par montant")
top10 = df_filtered.groupby('lib_gab')['total_montant'].sum().sort_values(ascending=False).head(10).reset_index()
st.dataframe(top10)

# =========================
# Historique retraits
# =========================
st.subheader("Évolution hebdomadaire des retraits")
fig = px.line(df_filtered, x='ds', y='total_montant', markers=True, title=f"Historique des retraits - {selected_gab}")
st.plotly_chart(fig, use_container_width=True)

# =========================
# Charger modèle et scaler LSTM
# =========================
@st.cache_resource
def load_lstm_model():
    model = load_model("lstm_gab_model.h5", custom_objects={'mse': losses.MeanSquaredError})
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_lstm_model()
except Exception as e:
    st.warning("Le modèle LSTM n'a pas pu être chargé.")
    model = None
    scaler = None

# =========================
# Prévisions LSTM
# =========================
if model is not None and len(df_filtered) >= 4:
    n_steps = 4  # nombre de semaines utilisées pour l'entraînement
    forecast_periods = 12

    y_values = df_filtered.sort_values('ds')['y'].values.reshape(-1,1)
    y_scaled = scaler.transform(y_values)

    last_seq = y_scaled[-n_steps:].reshape(1,n_steps,1)
    preds_future_scaled = []

    for _ in range(forecast_periods):
        yhat_scaled = model.predict(last_seq, verbose=0)[0,0]
        preds_future_scaled.append(yhat_scaled)
        last_seq = np.append(last_seq[:,1:,:], [[[yhat_scaled]]], axis=1)

    preds_future = scaler.inverse_transform(np.array(preds_future_scaled).reshape(-1,1)).flatten()
    future_dates = pd.date_range(start=df_filtered['ds'].max() + pd.Timedelta(weeks=1),
                                 periods=forecast_periods, freq='W-MON')
    df_future = pd.DataFrame({'ds': future_dates, 'Prévision LSTM': preds_future})

    st.subheader("Prévision des retraits (prochaines semaines)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_filtered['ds'], y=df_filtered['total_montant'],
                              mode='lines+markers', name='Historique'))
    fig2.add_trace(go.Scatter(x=df_future['ds'], y=df_future['Prévision LSTM'],
                              mode='lines+markers', name='Prévision'))
    st.plotly_chart(fig2, use_container_width=True)

    # Télécharger les prévisions
    st.subheader("Télécharger les prévisions")
    df_download = df_future.copy()
    df_download['GAB'] = selected_gab
    csv = df_download.to_csv(index=False)
    st.download_button(
        label="Télécharger CSV",
        data=csv,
        file_name=f"forecast_gab_{selected_gab}.csv",
        mime='text/csv'
    )
