# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from tensorflow.keras.models import load_model
import os

# =========================
# Titre et logo
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
    csv_path = os.path.join(os.path.dirname(__file__), "df_weekly_clean.csv")
    if not os.path.exists(csv_path):
        st.error(f"Le fichier df_weekly_clean.csv n'a pas été trouvé dans {csv_path}.")
        return pd.DataFrame()
    df = pd.read_csv(csv_path, parse_dates=['ds'])
    return df

df = load_data()
if df.empty:
    st.stop()

# =========================
# Sidebar - Filtres pour KPIs
# =========================
st.sidebar.header("Filtres")
region_list = sorted(df['region'].dropna().unique())
selected_region = st.sidebar.selectbox("Région :", region_list)

agence_list = sorted(df[df['region'] == selected_region]['agence'].dropna().unique())
selected_agence = st.sidebar.selectbox("Agence :", agence_list)

df_filtered = df[
    (df['region'] == selected_region) &
    (df['agence'] == selected_agence)
]

# =========================
# Fonctions de formatage
# =========================
def format_montant_k(val):
    return f"{val/1_000:,.0f} K MAD".replace(",", " ")

def format_nombre(val):
    return f"{val:,.0f}".replace(",", " ")

# =========================
# KPIs par agence
# =========================
st.subheader(f"KPIs pour l'agence {selected_agence}")
total_montant = df_filtered['total_montant'].sum()
total_nombre = df_filtered['total_nombre'].sum()
total_gabs_agence = df_filtered['num_gab'].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("Montant total retiré", format_montant_k(total_montant))
col2.metric("Nombre total de retraits", format_nombre(total_nombre))
col3.metric("Nombre de GAB", format_nombre(total_gabs_agence))

# =========================
# Evolution historique par GAB
# =========================
st.subheader("Évolution historique des retraits par GAB")
gab_list = sorted(df_filtered['lib_gab'].dropna().unique())
selected_gab = st.selectbox("Sélectionner un GAB :", gab_list)

df_gab = df_filtered[df_filtered['lib_gab'] == selected_gab].sort_values("ds")
fig = px.line(df_gab, x='ds', y='total_montant', markers=True)
fig.update_layout(xaxis_title="Semaine", yaxis_title="Montant retrait (MAD)")
st.plotly_chart(fig, use_container_width=True)

# =========================
# SECTION LSTM - Prévisions
# =========================
st.subheader(f"Prévisions LSTM pour {selected_gab}")
n_steps = 52
forecast_periods = 12

gab_num = df_gab['num_gab'].iloc[0]
model_file = f"lstm_gab_{gab_num}.h5"
scaler_file = f"scaler_gab_{gab_num}.save"

@st.cache_resource
def load_lstm_gab_model(model_file, scaler_file):
    model = load_model(model_file, compile=False)
    scaler = joblib.load(scaler_file)
    return model, scaler

if os.path.exists(model_file) and os.path.exists(scaler_file):
    model, scaler = load_lstm_gab_model(model_file, scaler_file)
    y_values = df_gab['total_montant'].values.reshape(-1,1)

    if len(y_values) >= n_steps:
        y_scaled = scaler.transform(y_values)
        last_seq = y_scaled[-n_steps:].reshape(1, n_steps, 1)
        preds_future_scaled = []

        for _ in range(forecast_periods):
            yhat_scaled = model.predict(last_seq, verbose=0)[0,0]
            preds_future_scaled.append(yhat_scaled)
            last_seq = np.append(last_seq[:,1:,:], [[[yhat_scaled]]], axis=1)

        preds_future = scaler.inverse_transform(np.array(preds_future_scaled).reshape(-1,1)).flatten()
        future_dates = pd.date_range(start=df_gab['ds'].max() + pd.Timedelta(weeks=1),
                                     periods=forecast_periods, freq='W-MON')

        df_plot = pd.concat([
            pd.DataFrame({'ds': df_gab['ds'], 'valeur': df_gab['total_montant'], 'type': 'Historique'}),
            pd.DataFrame({'ds': future_dates, 'valeur': preds_future, 'type': 'Prévision'})
        ])
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
    st.info(f"Modèle ou scaler non trouvé pour le GAB {gab_num}.")
