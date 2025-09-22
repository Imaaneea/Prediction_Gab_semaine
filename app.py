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
# Config page
# =========================
st.set_page_config(page_title="Prévision retraits GAB", layout="wide")

# =========================
# Logo et titre
# =========================
st.image("https://www.albaridbank.ma/themes/baridbank/logo.png", width=200)
st.title("Prévision des retraits GAB avec LSTM")
st.markdown("Application pour prédire les retraits hebdomadaires des GAB et visualiser les indicateurs par région et agence.")

# =========================
# Charger les données
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("df_subset.csv", parse_dates=['ds'])
    return df

df = load_data()

# =========================
# Tableau de bord global
# =========================
st.subheader("Tableau de bord global")

total_retraits = df['total_montant'].sum()
total_transactions = df['total_nombre'].sum()
avg_retrait_par_gab = df.groupby('num_gab')['total_montant'].mean().mean()
top_gabs = df.groupby('num_gab')['total_montant'].sum().sort_values(ascending=False).head(5)

col1, col2, col3 = st.columns(3)
col1.metric("Total des retraits (MAD)", f"{total_retraits:,.0f}")
col2.metric("Total des transactions", f"{total_transactions:,}")
col3.metric("Moyenne retrait par GAB (MAD)", f"{avg_retrait_par_gab:,.2f}")

st.subheader("Top 5 GAB par montant total")
st.bar_chart(top_gabs)

# =========================
# Indicateurs par région et agence
# =========================
st.subheader("Indicateurs par région et agence")
region_summary = df.groupby('region')['total_montant'].sum().sort_values(ascending=False)
agence_summary = df.groupby('agence')['total_montant'].sum().sort_values(ascending=False)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Montant total par région")
    st.bar_chart(region_summary)
with col2:
    st.markdown("### Montant total par agence")
    st.bar_chart(agence_summary)

# =========================
# Evolution hebdomadaire
# =========================
st.subheader("Evolution hebdomadaire des retraits")
df_weekly = df.groupby('ds')['total_montant'].sum().reset_index()
fig, ax = plt.subplots(figsize=(10,4))
sns.lineplot(data=df_weekly, x='ds', y='total_montant', marker='o', ax=ax)
ax.set_xlabel("Semaine")
ax.set_ylabel("Total des retraits (MAD)")
ax.grid(True)
st.pyplot(fig)

# =========================
# Choix du GAB
# =========================
gab_list = df['num_gab'].unique()
selected_gab = st.selectbox("Sélectionnez un GAB :", gab_list)
df_gab = df[df['num_gab'] == selected_gab].sort_values('ds')

# =========================
# Historique et prévision du GAB
# =========================
st.subheader(f"Historique et prévisions pour le GAB {selected_gab}")
fig2, ax2 = plt.subplots(figsize=(10,4))
sns.lineplot(data=df_gab, x='ds', y='y', marker='o', ax=ax2, label='Historique')
ax2.set_xlabel("Semaine")
ax2.set_ylabel("Montant retrait")
ax2.grid(True)
st.pyplot(fig2)

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

fig3, ax3 = plt.subplots(figsize=(10,4))
sns.lineplot(x=df_gab['ds'], y=df_gab['y'], marker='o', label='Historique', ax=ax3)
sns.lineplot(x=df_future['ds'], y=df_future['yhat'], marker='x', label='Prévision', ax=ax3)
ax3.set_xlabel("Semaine")
ax3.set_ylabel("Montant retrait")
ax3.grid(True)
ax3.legend()
st.pyplot(fig3)

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
