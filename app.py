# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# Filtrer par GAB, Région ou Agence
# =========================
st.sidebar.header("Filtres")
regions = ["Toutes"] + list(df['region'].unique())
selected_region = st.sidebar.selectbox("Sélectionnez une région :", regions)

if selected_region != "Toutes":
    df_region = df[df['region'] == selected_region]
else:
    df_region = df.copy()

agences = ["Toutes"] + list(df_region['agence'].unique())
selected_agence = st.sidebar.selectbox("Sélectionnez une agence :", agences)

if selected_agence != "Toutes":
    df_filtered = df_region[df_region['agence'] == selected_agence]
else:
    df_filtered = df_region.copy()

gab_list = df_filtered['num_gab'].unique()
selected_gab = st.sidebar.selectbox("Sélectionnez un GAB :", gab_list)

df_gab = df_filtered[df_filtered['num_gab'] == selected_gab].sort_values('ds')

# =========================
# Affichage tableau de bord indicateurs
# =========================
st.subheader("Tableau de bord")
total_montant = df_gab['total_montant'].sum()
total_nombre = df_gab['total_nombre'].sum()
st.metric("Montant total retiré", f"{total_montant:,.0f} MAD")
st.metric("Nombre total de retraits", f"{total_nombre:,}")

# =========================
# Historique des retraits
# =========================
st.subheader("Historique des retraits")
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(df_gab['ds'], df_gab['y'], marker='o', label='Réel', color='blue')
ax.set_xlabel("Semaine")
ax.set_ylabel("Montant retrait")
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
st.pyplot(fig)

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
n_steps = 4  # doit correspondre au nombre de semaines utilisées pour l'entraînement
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

# =========================
# Afficher la prévision
# =========================
st.subheader("Prévision des retraits (prochaines semaines)")
fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(df_gab['ds'], df_gab['y'], marker='o', label='Historique', color='blue')
ax2.plot(df_future['ds'], df_future['yhat'], marker='x', label='Prévision LSTM', color='orange')
ax2.set_xlabel("Semaine")
ax2.set_ylabel("Montant retrait")
ax2.grid(True, linestyle='--', alpha=0.6)
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
