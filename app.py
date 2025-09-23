# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from tensorflow.keras.models import load_model

# =========================
# Titre et logo
# =========================
st.set_page_config(page_title="Prévision des retraits GAB", layout="wide")
st.title("Prévision des retraits GAB avec LSTM")
st.write("Application basée sur le modèle LSTM pour prédire les retraits hebdomadaires des GAB.")

# Sidebar avec logo et filtres
st.sidebar.image(
    "https://www.albaridbank.ma/themes/baridbank/logo.png",
    use_container_width=True
)

# =========================
# Charger les données
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("df_subset.csv", parse_dates=['ds'])
    return df

df = load_data()

# =========================
# Sidebar - Filtres dépendants
# =========================
st.sidebar.header("Filtres")

region_list = sorted(df['region'].dropna().unique())
selected_region = st.sidebar.selectbox("Région :", region_list)

agence_list = sorted(df[df['region'] == selected_region]['agence'].dropna().unique())
selected_agence = st.sidebar.selectbox("Agence :", agence_list)

gab_list = sorted(df[(df['region'] == selected_region) & (df['agence'] == selected_agence)]['lib_gab'].dropna().unique())
selected_gab = st.sidebar.selectbox("GAB :", gab_list)

date_min = df['ds'].min()
date_max = df['ds'].max()
start_date = st.sidebar.date_input("Date début :", date_min)
end_date = st.sidebar.date_input("Date fin :", date_max)

df_filtered = df[
    (df['region'] == selected_region) &
    (df['agence'] == selected_agence) &
    (df['lib_gab'] == selected_gab) &
    (df['ds'] >= pd.to_datetime(start_date)) &
    (df['ds'] <= pd.to_datetime(end_date))
]

# =========================
# Fonction de formatage K MAD
# =========================
def format_montant_k(val):
    return f"{val/1_000:,.0f} K MAD".replace(",", " ")

def format_nombre(val):
    return f"{val:,.0f}".replace(",", " ")

# =========================
# KPIs
# =========================
total_montant = df_filtered['total_montant'].sum()
total_nombre = df_filtered['total_nombre'].sum()
total_gabs_region = df[df['region'] == selected_region]['num_gab'].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("Montant total retiré", format_montant_k(total_montant))
col2.metric("Nombre total de retraits", format_nombre(total_nombre))
col3.metric(f"Nombre de GAB ({selected_region})", format_nombre(total_gabs_region))

# =========================
# Top 10 des GAB par montant
# =========================
st.subheader("Top 10 des GAB par montant")
top10 = df[df['region'] == selected_region].groupby('lib_gab')['total_montant'].sum().sort_values(ascending=False).head(10).reset_index()
top10['total_montant'] = top10['total_montant'].apply(format_montant_k)
st.dataframe(top10)

# =========================
# Evolution hebdomadaire
# =========================
st.subheader(f"Évolution hebdomadaire des retraits pour {selected_gab}")
df_evo = df_filtered.groupby('ds')['y'].sum().reset_index()
fig = px.line(df_evo, x='ds', y='y', markers=True)
fig.update_layout(xaxis_title="Semaine", yaxis_title="Montant retrait (K MAD)")
st.plotly_chart(fig, use_container_width=True)

# =========================
# Charger modèle et scaler
# =========================
@st.cache_resource
def load_lstm_model():
    model = load_model("lstm_gab_model.h5", compile=False)
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_lstm_model()

# =========================
# Prévisions LSTM
# =========================
n_steps = 26  # corrigé
forecast_periods = 12

df_filtered = df_filtered.sort_values("ds")
y_values = df_filtered['y'].values.reshape(-1,1)

if len(y_values) >= n_steps:
    y_scaled = scaler.transform(y_values)
    last_seq = y_scaled[-n_steps:].reshape(1,n_steps,1)
    preds_future_scaled = []

    for _ in range(forecast_periods):
        yhat_scaled = model.predict(last_seq, verbose=0)[0,0]
        preds_future_scaled.append(yhat_scaled)
        last_seq = np.append(last_seq[:,1:,:], [[[yhat_scaled]]], axis=1)

    preds_future = scaler.inverse_transform(np.array(preds_future_scaled).reshape(-1,1)).flatten()
    future_dates = pd.date_range(start=df_filtered['ds'].max() + pd.Timedelta(weeks=1), periods=forecast_periods, freq='W-MON')
    df_future = pd.DataFrame({'ds': future_dates, 'yhat': preds_future})

    df_future['yhat'] = df_future['yhat'].apply(format_montant_k)

    st.subheader("Prévision des retraits (prochaines semaines)")
    fig2 = px.line(
        pd.concat([
            pd.DataFrame({'ds': df_filtered['ds'], 'valeur': df_filtered['y'], 'type': 'Historique'}),
            pd.DataFrame({'ds': future_dates, 'valeur': preds_future, 'type': 'Prévision'})
        ]),
        x="ds", y="valeur", color="type", markers=True
    )
    fig2.update_layout(xaxis_title="Semaine", yaxis_title="Montant retrait (K MAD)")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Télécharger les prévisions")
    df_download = pd.DataFrame({'ds': future_dates, 'yhat': preds_future, 'lib_gab': selected_gab})
    csv = df_download.to_csv(index=False)
    st.download_button(
        label="Télécharger CSV",
        data=csv,
        file_name=f"forecast_gab_{selected_gab}.csv",
        mime='text/csv'
    )
else:
    st.warning(f"Pas assez de données pour effectuer une prévision LSTM (minimum {n_steps} semaines).")
