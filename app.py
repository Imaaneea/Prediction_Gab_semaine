import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================
# Configuration de la page
# ========================================
st.set_page_config(page_title="Dashboard GAB", layout="wide")

# ========================================
# Charger les données
# ========================================
@st.cache_data
def load_data():
    df = pd.read_csv("df_weekly_clean.csv", parse_dates=["ds"])
    return df

df = load_data()

# ========================================
# Sidebar - Filtres
# ========================================
st.sidebar.header("Filtres")

regions = df["region"].dropna().unique()
region = st.sidebar.selectbox("Région", ["Toutes"] + sorted(regions.tolist()))

if region != "Toutes":
    agences = df[df["region"] == region]["agence"].dropna().unique()
else:
    agences = df["agence"].dropna().unique()
agence = st.sidebar.selectbox("Agence", ["Toutes"] + sorted(agences.tolist()))

if agence != "Toutes":
    gabs = df[df["agence"] == agence]["lib_gab"].dropna().unique()
else:
    gabs = df["lib_gab"].dropna().unique()
gab = st.sidebar.selectbox("GAB", ["Tous"] + sorted(gabs.tolist()))

# Filtres de dates
date_min = df["ds"].min()
date_max = df["ds"].max()
date_debut = st.sidebar.date_input("Date début", date_min)
date_fin = st.sidebar.date_input("Date fin", date_max)

# ========================================
# Appliquer les filtres
# ========================================
df_filtered = df.copy()

if region != "Toutes":
    df_filtered = df_filtered[df_filtered["region"] == region]
if agence != "Toutes":
    df_filtered = df_filtered[df_filtered["agence"] == agence]
if gab != "Tous":
    df_filtered = df_filtered[df_filtered["lib_gab"] == gab]

df_filtered = df_filtered[(df_filtered["ds"] >= pd.to_datetime(date_debut)) &
                          (df_filtered["ds"] <= pd.to_datetime(date_fin))]

# ========================================
# KPIs globaux
# ========================================
total_retrait = df_filtered["total_montant"].sum() / 1000  # en K
total_operations = df_filtered["total_nombre"].sum() / 1000  # en K
nb_gab = df_filtered["num_gab"].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("Montant total retiré (K)", f"{total_retrait:,.1f} K")
col2.metric("Nombre d'opérations (K)", f"{total_operations:,.1f} K")
col3.metric("Nombre de GAB", nb_gab)

# ========================================
# Graphiques
# ========================================
st.subheader("Évolution hebdomadaire des retraits et opérations")

fig, ax1 = plt.subplots(figsize=(12, 5))

sns.lineplot(data=df_filtered, x="ds", y="total_montant", ax=ax1, label="Montant retiré")
ax1.set_ylabel("Montant retiré")
ax1.tick_params(axis='x', rotation=45)

ax2 = ax1.twinx()
sns.lineplot(data=df_filtered, x="ds", y="total_nombre", ax=ax2, color="orange", label="Nombre d'opérations")
ax2.set_ylabel("Nombre d'opérations")

# Gestion des légendes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

st.pyplot(fig)

# ========================================
# Section prévisions LSTM (placeholder)
# ========================================
st.subheader("Prévisions LSTM (par GAB)")

if gab != "Tous":
    df_gab = df_filtered[df_filtered["lib_gab"] == gab]
    if len(df_gab) < 52:
        st.warning("Pas assez de données pour effectuer une prévision LSTM (minimum 52 semaines).")
    else:
        st.info("Ici, tu pourras intégrer ton modèle LSTM pour la prévision.")
else:
    st.info("Sélectionnez un GAB pour voir les prévisions LSTM.")
