import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# ========================================
# Configuration de la page
# ========================================
st.set_page_config(page_title="Dashboard GAB", layout="wide")

# ========================================
# Chargement des données
# ========================================
@st.cache_data
def load_data():
    all_files = glob.glob("data/*.csv")
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    return df

df = load_data()

# ========================================
# Préparation des données
# ========================================
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

# ========================================
# Chargement des modèles
# ========================================
@st.cache_resource
def load_models():
    model = load_model("models/lstm_model.h5")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_models()

# ========================================
# Barre latérale (filtres)
# ========================================
st.sidebar.header("Filtres")

# Filtres Région et Agence si disponibles
regions = df['region'].dropna().unique().tolist() if 'region' in df.columns else []
selected_region = st.sidebar.selectbox("Région", ["Toutes"] + regions)

if selected_region != "Toutes" and 'region' in df.columns:
    df = df[df['region'] == selected_region]

agences = df['agence'].dropna().unique().tolist() if 'agence' in df.columns else []
selected_agence = st.sidebar.selectbox("Agence", ["Toutes"] + agences)

if selected_agence != "Toutes" and 'agence' in df.columns:
    df = df[df['agence'] == selected_agence]

# Sélection du GAB
gab_list = df['gab_id'].unique().tolist()
selected_gab = st.sidebar.selectbox("GAB", gab_list)

# ========================================
# Section KPI
# ========================================
col1, col2, col3 = st.columns(3)

nb_gabs = df['gab_id'].nunique()
total_retraits = df['retrait'].sum()
total_ops = df['nb_operations'].sum() if 'nb_operations' in df.columns else None

col1.metric("🏧 Nombre de GABs", nb_gabs)
col2.metric("💰 Total des retraits (MAD)", f"{total_retraits:,.0f}")
if total_ops is not None:
    col3.metric("🔢 Nombre total d’opérations", f"{total_ops:,.0f}")
else:
    col3.metric("🔢 Nombre total d’opérations", "Non disponible")

# ========================================
# Données filtrées pour le GAB sélectionné
# ========================================
df_gab = df[df['gab_id'] == selected_gab]

# ========================================
# Graphique 1 : évolution des retraits avec moyenne mobile
# ========================================
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_gab['date'], y=df_gab['retrait'],
    mode='lines+markers', name='Retraits'
))
fig.add_trace(go.Scatter(
    x=df_gab['date'],
    y=df_gab['retrait'].rolling(window=4).mean(),
    mode='lines', name='Moyenne mobile (4 sem.)'
))
fig.update_layout(
    title=f"📉 Évolution des retraits du GAB {selected_gab}",
    xaxis_title="Date",
    yaxis_title="Montant des retraits (MAD)",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# ========================================
# Prédiction LSTM
# ========================================
def predict_lstm(df_gab):
    if len(df_gab) < 10:
        return np.nan
    values = df_gab['retrait'].values[-10:].reshape(-1, 1)
    scaled = scaler.transform(values)
    X = np.array([scaled])
    y_pred = model.predict(X)
    y_pred_inv = scaler.inverse_transform(y_pred)
    return y_pred_inv[0][0]

pred = predict_lstm(df_gab)

if not np.isnan(pred):
    st.markdown(f"### 🔮 Prévision des retraits semaine suivante : **{pred:,.0f} MAD**")
else:
    st.info("Pas assez de données pour générer une prévision.")

# ========================================
# Risque de rupture de cash
# ========================================
seuil_rupture = 500000
gabs_risque = df.groupby('gab_id')['retrait'].mean()
gabs_critique = gabs_risque[gabs_risque > seuil_rupture]
nb_risque = len(gabs_critique)

st.warning(f"⚠️ {nb_risque} GAB(s) présentent un risque de rupture de cash (niveau critique).")

# ========================================
# Carte des GABs (si coordonnées disponibles)
# ========================================
if {'latitude', 'longitude'}.issubset(df.columns):
    fig_map = px.scatter_mapbox(
        df, lat="latitude", lon="longitude",
        color="retrait", size="retrait",
        hover_name="gab_id",
        hover_data=['agence', 'region'] if 'region' in df.columns else None,
        zoom=5, mapbox_style="carto-positron",
        title="📍 Répartition géographique des GAB"
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("Les coordonnées géographiques ne sont pas disponibles pour afficher la carte.")

# ========================================
# Footer
# ========================================
st.markdown("---")
st.markdown("💡 *Dashboard de prévision et d’analyse des retraits GAB – Projet PFE 2025*")
