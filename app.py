# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras import losses

# =========================
# Titre et logo
# =========================
st.set_page_config(page_title="Prévision des retraits GAB", layout="wide")
st.image("https://www.albaridbank.ma/themes/baridbank/logo.png", use_column_width=True)
st.title("Prévision des retraits GAB avec LSTM")
st.write("Application basée sur le modèle LSTM pour prédire les retraits hebdomadaires des GAB.")

# Sidebar avec logo et filtres
# =========================
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

# Filtre Région
region_list = df['region'].unique()
selected_region = st.sidebar.selectbox("Région :", region_list)

# Filtre Agence dépendant de la région
agence_list = df[df['region'] == selected_region]['agence'].unique()
selected_agence = st.sidebar.selectbox("Agence :", agence_list)

# Filtre GAB dépendant de l'agence
gab_list = df[(df['region'] == selected_region) & (df['agence'] == selected_agence)]['lib_gab'].unique()
selected_gab = st.sidebar.selectbox("GAB :", gab_list)

# Filtre période
date_min = df['ds'].min()
date_max = df['ds'].max()
start_date = st.sidebar.date_input("Date début :", date_min)
end_date = st.sidebar.date_input("Date fin :", date_max)

# Filtrer le DataFrame
df_filtered = df[
    (df['region'] == selected_region) &
    (df['agence'] == selected_agence) &
    (df['lib_gab'] == selected_gab) &
    (df['ds'] >= pd.to_datetime(start_date)) &
    (df['ds'] <= pd.to_datetime(end_date))
]

# =========================
# KPIs
# =========================
total_montant = df_filtered['total_montant'].sum()
total_nombre = df_filtered['total_nombre'].sum()
total_gabs_region = df[df['region'] == selected_region]['num_gab'].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("Montant total retiré", f"{total_montant:,.2f} MAD")
col2.metric("Nombre total de retraits", f"{total_nombre:,.0f}")
col3.metric(f"Nombre de GAB ({selected_region})", total_gabs_region)

# =========================
# Top 10 des GAB par montant
# =========================
st.subheader("Top 10 des GAB par montant")
top10 = df[df['region'] == selected_region].groupby('lib_gab')['total_montant'].sum().sort_values(ascending=False).head(10).reset_index()
st.dataframe(top10)

# =========================
# Evolution hebdomadaire
# =========================
st.subheader(f"Évolution hebdomadaire des retraits pour {selected_gab}")
df_evo = df_filtered.groupby('ds')['y'].sum().reset_index()
fig = px.line(df_evo, x='ds', y='y', markers=True)
fig.update_layout(xaxis_title="Semaine", yaxis_title="Montant retrait")
st.plotly_chart(fig, use_container_width=True)

# =========================
# Charger modèle et scaler
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
n_steps = 4  # nombre de semaines pour l'entrée du modèle
forecast_periods = 12

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

    st.subheader("Prévision des retraits (prochaines semaines)")
    fig2 = px.line(df_future, x='ds', y='yhat', markers=True)
    fig2.add_scatter(x=df_filtered['ds'], y=df_filtered['y'], mode='lines+markers', name='Historique')
    fig2.update_layout(xaxis_title="Semaine", yaxis_title="Montant retrait")
    st.plotly_chart(fig2, use_container_width=True)

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
else:
    st.warning(f"Pas assez de données pour effectuer une prévision LSTM (minimum {n_steps} semaines).")
