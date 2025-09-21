# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model

# =========================
# Titre de l'application
# =========================
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

# Liste des GAB disponibles
gab_list = df['num_gab'].unique()
selected_gab = st.selectbox("Sélectionnez un GAB :", gab_list)

# Filtrer les données pour le GAB sélectionné
df_gab = df[df['num_gab'] == selected_gab].sort_values('ds')

# =========================
# Afficher l'historique
# =========================
st.subheader("Historique des retraits")
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(df_gab['ds'], df_gab['y'], marker='o', label='Réel')
ax.set_xlabel("Semaine")
ax.set_ylabel("Montant retrait")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# =========================
# Charger modèle et scaler
# =========================
@st.cache_resource
def load_lstm_model():
    model = load_model("lstm_gab_model.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_lstm_model()

# =========================
# Prévisions LSTM
# =========================
n_steps = 4  # doit correspondre au nombre de semaines utilisées pour l'entraînement
forecast_periods = 12

# Préparer les données pour LSTM
y_values = df_gab['y'].values.reshape(-1,1)
y_scaled = scaler.transform(y_values)

# Dernière séquence pour prédiction future
last_seq = y_scaled[-n_steps:].reshape(1,n_steps,1)
preds_future_scaled = []

for _ in range(forecast_periods):
    yhat_scaled = model.predict(last_seq, verbose=0)[0,0]
    preds_future_scaled.append(yhat_scaled)
    last_seq = np.append(last_seq[:,1:,:], [[[yhat_scaled]]], axis=1)

# Re-transformer en valeurs originales
preds_future = scaler.inverse_transform(np.array(preds_future_scaled).reshape(-1,1)).flatten()
future_dates = pd.date_range(start=df_gab['ds'].max() + pd.Timedelta(weeks=1), periods=forecast_periods, freq='W-MON')

df_future = pd.DataFrame({'ds': future_dates, 'yhat': preds_future})

# =========================
# Afficher la prévision
# =========================
st.subheader("Prévision des retraits (prochaines semaines)")
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
