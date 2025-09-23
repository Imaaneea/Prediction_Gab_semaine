import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ===============================
# Charger les données
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("df_weekly_clean.csv", parse_dates=["ds"])
    df["ds"] = pd.to_datetime(df["ds"])
    return df

df = load_data()

st.set_page_config(page_title="Prévision retraits GAB", layout="wide")

# ===============================
# Filtres latéraux
# ===============================
st.sidebar.header("Filtres")

regions = df["region"].dropna().unique()
region = st.sidebar.selectbox("Région :", options=["Toutes"] + list(regions))

if region != "Toutes":
    agences = df[df["region"] == region]["agence"].dropna().unique()
else:
    agences = df["agence"].dropna().unique()
agence = st.sidebar.selectbox("Agence :", options=["Toutes"] + list(agences))

if agence != "Toutes":
    gabs = df[df["agence"] == agence]["num_gab"].dropna().unique()
else:
    gabs = df["num_gab"].dropna().unique()
gab = st.sidebar.selectbox("GAB :", options=["Tous"] + list(gabs))

min_date = df["ds"].min().date()
max_date = df["ds"].max().date()
date_debut, date_fin = st.sidebar.date_input(
    "Période :", [min_date, max_date], min_value=min_date, max_value=max_date
)

# ===============================
# Filtrage des données
# ===============================
df_filtered = df.copy()

if region != "Toutes":
    df_filtered = df_filtered[df_filtered["region"] == region]
if agence != "Toutes":
    df_filtered = df_filtered[df_filtered["agence"] == agence]
if gab != "Tous":
    df_filtered = df_filtered[df_filtered["num_gab"] == gab]

df_filtered = df_filtered[
    (df_filtered["ds"].dt.date >= date_debut) & (df_filtered["ds"].dt.date <= date_fin)
]

# Debug min/max date
# st.write("Min date filtrée :", df_filtered["ds"].min())
# st.write("Max date filtrée :", df_filtered["ds"].max())

# ===============================
# KPIs
# ===============================
total_retrait = df_filtered["retrait"].sum()
total_op = df_filtered["operation"].sum()
nb_gab = df_filtered["num_gab"].nunique()

st.title("📊 Prévision des retraits GAB")
st.markdown("Application basée sur les modèles LSTM pour prédire les retraits hebdomadaires des GAB.")

col1, col2, col3 = st.columns(3)
col1.metric("Retraits globaux", f"{total_retrait:,.0f} MAD")
col2.metric("Nombre d'opérations", f"{total_op:,}")
col3.metric("Nombre de GAB", nb_gab)

# ===============================
# Graphique évolution hebdomadaire
# ===============================
st.subheader("📈 Évolution hebdomadaire des retraits")

if df_filtered.empty:
    st.warning("Aucune donnée disponible pour ce filtre.")
else:
    fig = px.line(df_filtered, x="ds", y="retrait",
                  title="Évolution hebdomadaire des retraits")
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# Prévisions LSTM
# ===============================
st.subheader("🤖 Prévisions avec LSTM")

if gab != "Tous":
    df_gab = df_filtered[df_filtered["num_gab"] == gab].copy()
    df_gab = df_gab.sort_values("ds")

    if len(df_gab) < 52:
        st.warning("Pas assez de données pour effectuer une prévision LSTM (minimum 52 semaines).")
    else:
        # Préparer les données
        values = df_gab["retrait"].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(values)

        X, y = [], []
        window = 12
        for i in range(window, len(scaled)):
            X.append(scaled[i - window:i, 0])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Modèle LSTM
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=10, batch_size=8, verbose=0)

        # Prévisions
        preds = []
        last_window = scaled[-window:]
        current_input = last_window.reshape(1, window, 1)

        for _ in range(12):  # 12 semaines de prévision
            pred = model.predict(current_input, verbose=0)
            preds.append(pred[0, 0])
            current_input = np.append(current_input[:, 1:, :], [[pred]], axis=1)

        preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

        future_dates = pd.date_range(df_gab["ds"].max() + pd.Timedelta(weeks=1),
                                     periods=12, freq="W")
        df_preds = pd.DataFrame({"ds": future_dates, "retrait": preds.flatten()})

        # Graphe
        fig2 = px.line(df_gab, x="ds", y="retrait", title=f"Prévision des retraits pour le GAB {gab}")
        fig2.add_scatter(x=df_preds["ds"], y=df_preds["retrait"], mode="lines+markers", name="Prévisions")
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Veuillez sélectionner un GAB spécifique pour afficher les prévisions.")
