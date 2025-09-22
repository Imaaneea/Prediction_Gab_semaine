# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras import losses
from datetime import datetime

st.set_page_config(page_title="Prévision GAB LSTM", layout="wide")

# =========================
# Bandeau Logo et titre
# =========================
st.markdown(
    """
    <div style="display:flex; align-items:center;">
        <img src="https://www.albaridbank.ma/themes/baridbank/logo.png" width="150">
        <h2 style="margin-left:20px;">Prévision des retraits GAB avec LSTM</h2>
    </div>
    """, unsafe_allow_html=True
)
st.write("Application interactive pour analyser et prévoir les retraits hebdomadaires des GAB.")

# =========================
# Charger les données
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("df_subset.csv", parse_dates=['ds'])
    df['month'] = df['ds'].dt.month
    df['week'] = df['ds'].dt.isocalendar().week
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

start_date, end_date = st.sidebar.date_input(
    "Période",
    [df['ds'].min(), df['ds'].max()],
    min_value=df['ds'].min(),
    max_value=df['ds'].max()
)

# Filtrer
df_filtered = df.copy()
if selected_region != "Toutes":
    df_filtered = df_filtered[df_filtered['region']==selected_region]
if selected_agence != "Toutes":
    df_filtered = df_filtered[df_filtered['agence']==selected_agence]
if selected_gab != "Tous":
    df_filtered = df_filtered[df_filtered['lib_gab']==selected_gab]
df_filtered = df_filtered[(df_filtered['ds'] >= pd.to_datetime(start_date)) & (df_filtered['ds'] <= pd.to_datetime(end_date))]

# =========================
# KPI avec style
# =========================
st.subheader("Tableau de bord")
col1, col2, col3 = st.columns(3)

def format_number(x):
    return f"{x:,.0f}"

col1.metric("💰 Montant total retiré (MAD)", format_number(df_filtered['total_montant'].sum()))
col2.metric("🔢 Nombre total de retraits", format_number(df_filtered['total_nombre'].sum()))
col3.metric("🏧 Nombre de GAB", df_filtered['num_gab'].nunique())

# =========================
# Top 10 des GAB par montant
# =========================
st.subheader("Top 10 des GAB par montant")
top_gabs = df_filtered.groupby('lib_gab')['total_montant'].sum().sort_values(ascending=False).head(10)
st.bar_chart(top_gabs)

# Nombre de GAB par région
st.subheader("Nombre de GAB par région")
gabs_region = df_filtered.groupby('region')['num_gab'].nunique()
st.bar_chart(gabs_region)

# =========================
# Historique et évolution
# =========================
st.subheader("Historique et évolution des retraits")

if selected_gab != "Tous":
    df_gab = df_filtered[df_filtered['lib_gab']==selected_gab].sort_values('ds')
else:
    df_gab = df_filtered.groupby('ds')['y'].sum().reset_index()

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(df_gab['ds'], df_gab['y'], marker='o', color="#1f77b4", label='Montant retrait')
ax.set_xlabel("Semaine")
ax.set_ylabel("Montant retrait")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format_number(x)))
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
st.pyplot(fig)

# =========================
# Prévisions LSTM si un GAB est sélectionné
# =========================
if selected_gab != "Tous":
    @st.cache_resource
    def load_lstm_model():
        model = load_model("lstm_gab_model.h5", custom_objects={'mse': losses.MeanSquaredError})
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler

    model, scaler = load_lstm_model()

    n_steps = 4
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
    ax2.plot(df_gab['ds'], df_gab['y'], marker='o', color="#1f77b4", label='Historique')
    ax2.plot(df_future['ds'], df_future['yhat'], marker='x', color="#ff7f0e", label='Prévision LSTM')
    ax2.set_xlabel("Semaine")
    ax2.set_ylabel("Montant retrait")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format_number(x)))
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("Télécharger les prévisions")
    df_download = df_future.copy()
    df_download['num_gab'] = df_gab['num_gab'].iloc[0]
    csv = df_download.to_csv(index=False)
    st.download_button(
        label="Télécharger CSV",
        data=csv,
        file_name=f"forecast_gab_{df_gab['num_gab'].iloc[0]}.csv",
        mime='text/csv'
    )
