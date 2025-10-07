import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import glob
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Analyse des retraits GAB", layout="wide")

st.title("💳 Tableau de bord d’analyse des retraits GAB")

# =====================
# Chargement des données
# =====================
@st.cache_data
def load_data():
    all_files = glob.glob("data/*.csv")
    df_list = []
    for f in all_files:
        temp = pd.read_csv(f)
        df_list.append(temp)
    df = pd.concat(df_list, ignore_index=True)
    df["ds"] = pd.to_datetime(df["ds"])
    return df

df = load_data()

# =====================
# Filtres
# =====================
st.sidebar.header("🎛️ Filtres")
regions = st.sidebar.multiselect("Région", sorted(df["region"].dropna().unique()), [])
agences = st.sidebar.multiselect("Agence", sorted(df["agence"].dropna().unique()), [])
date_min, date_max = st.sidebar.date_input("Période", [df["ds"].min(), df["ds"].max()])
user_seuil = st.sidebar.number_input("Seuil critique personnalisé (MAD, facultatif)", value=0, step=10000)

# Application des filtres
df_filtered = df.copy()
if regions:
    df_filtered = df_filtered[df_filtered["region"].isin(regions)]
if agences:
    df_filtered = df_filtered[df_filtered["agence"].isin(agences)]
df_filtered = df_filtered[(df_filtered["ds"] >= pd.to_datetime(date_min)) & (df_filtered["ds"] <= pd.to_datetime(date_max))]

# =====================
# Seuil critique dynamique
# =====================
df_avg_gab = df_filtered.groupby("num_gab")["total_montant"].mean().to_dict()

def get_seuil(gab_id):
    if user_seuil > 0:
        return user_seuil
    else:
        return df_avg_gab.get(gab_id, 100000)

df_latest = df_filtered.loc[df_filtered.groupby("num_gab")["ds"].idxmax()].copy()
df_latest["seuil_critique"] = df_latest["num_gab"].apply(get_seuil)

def classify_gab(row):
    s = row["seuil_critique"]
    if row["total_montant"] < s:
        return "Critique"
    elif row["total_montant"] < 2 * s:
        return "Alerte"
    else:
        return "Normal"

df_latest["status"] = df_latest.apply(classify_gab, axis=1)

# =====================
# Section 1 : KPIs
# =====================
st.markdown("## 📊 Indicateurs clés")

total_retraits = df_filtered["total_montant"].sum() / 1_000
nb_ops = len(df_filtered)
nb_gab = df_filtered["num_gab"].nunique()
pct_critique = (df_latest["status"].eq("Critique").sum() / len(df_latest)) * 100 if len(df_latest) > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Montant total (K MAD)", f"{total_retraits:,.0f}")
col2.metric("Nombre d’opérations", f"{nb_ops:,}")
col3.metric("Nombre de GABs", f"{nb_gab}")
col4.metric("GABs critiques (%)", f"{pct_critique:.1f}%")

# =====================
# Section 2 : Évolution des retraits
# =====================
st.markdown("## 📈 Évolution des retraits")
evol = df_filtered.groupby("ds")["total_montant"].sum().reset_index()
fig_evol = go.Figure()
fig_evol.add_trace(go.Scatter(x=evol["ds"], y=evol["total_montant"]/1000,
                              mode="lines+markers", name="Montants retirés"))
fig_evol.update_layout(title="Évolution hebdomadaire des retraits (K MAD)",
                       xaxis_title="Date", yaxis_title="Montant (K MAD)")
st.plotly_chart(fig_evol, use_container_width=True)

# =====================
# Section 3 : Répartition régionale & évolution
# =====================
st.markdown("## 🌍 Répartition régionale & évolution")
df_region = df_filtered.groupby("region")["total_montant"].sum().reset_index().sort_values("total_montant", ascending=False)
fig_region = go.Figure(go.Bar(
    x=df_region["region"],
    y=df_region["total_montant"]/1000,
    text=(df_region["total_montant"]/1000).round(0),
    textposition="auto",
    marker_color="lightskyblue"
))
fig_region.update_layout(title="Montants totaux retirés par région (K MAD)",
                         xaxis_title="Région", yaxis_title="Montant (K MAD)")
st.plotly_chart(fig_region, use_container_width=True)

# =====================
# Section 4 : Alertes / Fiches réseau
# =====================
st.markdown("## 🚨 Alertes récentes et fiches réseau")

def status_label(total, seuil):
    if total < seuil:
        return ("Critique", "red-status")
    elif total < 2*seuil:
        return ("Alerte", "orange-status")
    else:
        return ("Normal", "green-status")

st.markdown("""
<style>
.red-status {color: white; background-color: #e74c3c; padding: 3px 10px; border-radius: 10px;}
.orange-status {color: white; background-color: #f39c12; padding: 3px 10px; border-radius: 10px;}
.green-status {color: white; background-color: #27ae60; padding: 3px 10px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

cols_to_show = ["num_gab", "agence", "region", "total_montant", "seuil_critique"]
display_df = df_latest[cols_to_show].copy().reset_index(drop=True)

display_df["status_html"] = display_df.apply(
    lambda x: f'<span class="{status_label(x["total_montant"], x["seuil_critique"])[1]}">'
              f'{status_label(x["total_montant"], x["seuil_critique"])[0]}</span>',
    axis=1
)

for i, row in display_df.iterrows():
    with st.expander(f"💡 GAB {row['num_gab']} — {row['agence']} ({row['region']})"):
        st.markdown(f"""
        - **Montant retiré (K MAD)** : {row['total_montant']/1000:.2f}  
        - **Seuil critique (MAD)** : {row['seuil_critique']:,.0f}  
        - **Statut** : {row['status_html']}
        """, unsafe_allow_html=True)
