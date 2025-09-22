# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
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
# Sélection du GAB et période
# =========================
gab_list = df['num_gab'].unique()
selected_gab = st.selectbox("Sélectionnez un GAB :", gab_list)

st.subheader("Filtrer par période")
start_date = st.date_input("Date de début", df['ds'].min())
end_date = st.date_input("Date de fin", df['ds'].max())

# Filtrer le GAB sélectionné par période
df_gab = df[df['num_gab'] == selected_gab].sort_values('ds')
df_gab_period = df_gab[(df_gab['ds'] >= pd.to_datetime(start_date)) & (df_gab['ds'] <= pd.to_datetime(end_date))]

# =========================
# Tableau de bord des indicateurs
# =========================
st.subheader("Tableau de bord")

total_amount = df_gab_period['y'].sum()
total_retraits = df_gab_period['y'].count()
st.metric("Montant total retiré (MAD)", f"{total_amount:,.0f}")
st.metric("Nombre total de retraits", f"{total_retraits:,.0f}")

# Nombre de GAB par région
st.write("Nombre de GAB par région :")
st.dataframe(df.groupby('region')['num_gab'].nunique().sort_values(ascending=False))

# =========================
# Top 10 des GAB par montant
# =========================
st.subheader("Top 10 des GAB par montant total retiré")
top10_gab = df.groupby('num_gab').agg({'y':'sum','lib_gab':'first'}).sort_values('y', ascending=False).head(10)
st.dataframe(top10_gab[['lib_gab','y']])

# =========================
# Historique et évolution hebdomadaire
# =========================
st.subheader(f"Évolution hebdomadaire des retraits pour le GAB {selected_gab}")
if not df_gab_period.empty:
    df_weekly = df_gab_period.groupby('week')['y'].sum().reset_index()
    fig1, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(df_weekly['week'], df_weekly['y'], marker='o')
    ax1.set_xlabel("Semaine")
    ax1.set_ylabel("Montant retiré")
    ax1.grid(True)
    st.pyplot(fig1)
else:
    st.write("Aucune donnée pour la période sélectionnée.")

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
# Prévisions LSTM
# =========================
st.subheader("Prévision des retraits (prochaines semaines)")

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

fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(df_gab['ds'], df_gab['y'], marker='o', label='Historique')
ax2.plot(df_future['ds'], df_future['yhat'], marker='x', label='Prévision LSTM')
ax2.set_xlabel("Semaine")
ax2.set_ylabel("Montant retrait")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

# =========================
# Télécharger les résultats
# =========================
st.subheader("Télécharger les prévisions")
df_download = df_future.copy()
df_download['num_gab'] = selected_gab
csv = df_download.to_csv(index=False)
st.download_button(
    label="Télécharger CSV",
    data=csv,
    file_name=f"forecast_gab_{selected_gab}.csv",
    mime='text/csv'
)
