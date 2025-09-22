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
# Titre de l'application
# =========================
st.set_page_config(page_title="Prévision retraits GAB", layout="wide")
st.title("Prévision des retraits GAB avec LSTM")
st.write("Application basée sur le modèle LSTM pour prédire les retraits hebdomadaires des GAB.")

# =========================
# Sidebar avec logo et filtres
# =========================
st.sidebar.image(
    "https://www.albaridbank.ma/themes/baridbank/logo.png",
    use_container_width=True
)

# Charger les données
@st.cache_data
def load_data():
    df = pd.read_csv("df_subset.csv", parse_dates=['ds'])
    return df

df = load_data()

# Filtres
regions = ["Toutes"] + list(df['region'].unique())
selected_region = st.sidebar.selectbox("Sélectionnez une région :", regions)

agences = ["Toutes"]
if selected_region != "Toutes":
    agences += list(df[df['region']==selected_region]['agence'].unique())
else:
    agences += list(df['agence'].unique())
selected_agence = st.sidebar.selectbox("Sélectionnez une agence :", agences)

gabs = ["Tous"] + list(df['lib_gab'].unique())
selected_gab = st.sidebar.selectbox("Sélectionnez un GAB :", gabs)

# Filtre période
date_min = df['ds'].min()
date_max = df['ds'].max()
start_date = st.sidebar.date_input("Date début :", date_min, min_value=date_min, max_value=date_max)
end_date = st.sidebar.date_input("Date fin :", date_max, min_value=date_min, max_value=date_max)

# Filtrer les données
df_filtered = df[(df['ds'] >= pd.to_datetime(start_date)) & (df['ds'] <= pd.to_datetime(end_date))]
if selected_region != "Toutes":
    df_filtered = df_filtered[df_filtered['region']==selected_region]
if selected_agence != "Toutes":
    df_filtered = df_filtered[df_filtered['agence']==selected_agence]
if selected_gab != "Tous":
    df_filtered = df_filtered[df_filtered['lib_gab']==selected_gab]

# =========================
# Tableau de bord
# =========================
st.subheader("Tableau de bord")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Montant total retiré", f"{df_filtered['total_montant'].sum():,.0f} MAD")
with col2:
    st.metric("Nombre total de retraits", f"{df_filtered['total_nombre'].sum():,.0f}")
with col3:
    st.metric("Nombre de GAB (région)", df_filtered['num_gab'].nunique())

# Top 10 GAB par montant
st.subheader("Top 10 des GAB par montant")
top_gabs = df_filtered.groupby(['lib_gab'])['total_montant'].sum().sort_values(ascending=False).head(10).reset_index()
st.dataframe(top_gabs.style.format({"total_montant": "{:,.2f}"}))

# =========================
# Évolution hebdomadaire
# =========================
st.subheader("Évolution hebdomadaire des retraits")
if selected_gab != "Tous":
    df_gab = df_filtered[df_filtered['lib_gab']==selected_gab].sort_values('ds')
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_gab['ds'], df_gab['total_montant'], marker='o', label='Montant retiré')
    ax.set_xlabel("Semaine")
    ax.set_ylabel("Montant retrait")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Sélectionnez un GAB pour afficher l'évolution hebdomadaire.")

# =========================
# Charger modèle et scaler
# =========================
@st.cache_resource
def load_lstm_model():
    try:
        model = load_model("lstm_gab_model.h5", custom_objects={'mse': losses.MeanSquaredError})
    except Exception as e:
        st.error(f"Erreur chargement modèle LSTM : {e}")
        model = None
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error(f"Erreur chargement scaler : {e}")
        scaler = None
    return model, scaler

model, scaler = load_lstm_model()

# =========================
# Prévisions LSTM pour le GAB sélectionné
# =========================
if selected_gab != "Tous" and model is not None and scaler is not None:
    n_steps = 4
    forecast_periods = 12
    y_values = df_gab['total_montant'].values.reshape(-1,1)
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
    ax2.plot(df_gab['ds'], df_gab['total_montant'], marker='o', label='Historique')
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
