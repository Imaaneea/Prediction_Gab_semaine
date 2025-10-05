import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import glob
from tensorflow.keras.models import load_model
import joblib
import folium
from streamlit_folium import st_folium
from datetime import timedelta

# ========================================
# Configuration de la page
# ========================================
st.set_page_config(page_title="CashStream - Dashboard GAB", layout="wide")

# ========================================
# Chargement des données
# ========================================
@st.cache_data
def load_data():
    df = pd.read_csv("df_weekly_clean.csv", parse_dates=["ds"])
    df["num_gab"] = df["num_gab"].astype(str)
    df["week_day"] = df["ds"].dt.dayofweek
    df["week"] = df["ds"].dt.isocalendar().week
    df["year"] = df["ds"].dt.year
    # Statut fictif par seuils pour la démo
    df["status"] = pd.cut(df["total_montant"], bins=[-np.inf, 50000, 150000, np.inf],
                          labels=["Critique", "Alerte", "Normal"])
    # Coordonnées fictives pour la carte (à remplacer par les vraies)
    df["lat"] = 34 + np.random.rand(len(df))/10
    df["lon"] = -6 + np.random.rand(len(df))/10
    return df

df = load_data()

# ========================================
# Chargement silencieux des modèles LSTM
# ========================================
@st.cache_data
def load_lstm_models():
    models = {}
    scalers = {}
    for model_file in glob.glob("lstm_gab_*.h5"):
        gab_id = model_file.split("_")[-1].replace(".h5", "")
        scaler_file = f"scaler_gab_{gab_id}.save"
        try:
            models[gab_id] = load_model(model_file, compile=False)
            scalers[gab_id] = joblib.load(scaler_file)
        except Exception as e:
            st.warning(f"Impossible de charger LSTM pour {gab_id}: {e}")
    return models, scalers

lstm_models, lstm_scalers = load_lstm_models()

# ========================================
# Onglets
# ========================================
tab = st.sidebar.radio("Navigation", ["Tableau de bord réseau", "Prévisions LSTM multi-GAB"])

# ========================================
# Onglet 1 : Tableau de bord réseau
# ========================================
if tab == "Tableau de bord réseau":
    st.title("Tableau de bord réseau - GAB et agences")

    # Sidebar filtres
    st.sidebar.header("Filtres")
    regions = df["region"].dropna().unique()
    region = st.sidebar.selectbox("Région", ["Toutes"] + sorted(regions.tolist()))
    
    if region != "Toutes":
        agences = df[df["region"] == region]["agence"].dropna().unique()
    else:
        agences = df["agence"].dropna().unique()
    agence = st.sidebar.selectbox("Agence", ["Toutes"] + sorted(agences.tolist()))

    df_filtered = df.copy()
    if region != "Toutes":
        df_filtered = df_filtered[df_filtered["region"] == region]
    if agence != "Toutes":
        df_filtered = df_filtered[df_filtered["agence"] == agence]

    # KPIs
    st.subheader("KPIs réseau")
    cash_total = df_filtered["total_montant"].sum()
    nb_gab_actifs = df_filtered["num_gab"].nunique()
    nb_critique = df_filtered[df_filtered["status"]=="Critique"]["num_gab"].nunique()
    nb_alerte = df_filtered[df_filtered["status"]=="Alerte"]["num_gab"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cash total (KDH)", f"{cash_total/1000:,.0f}")
    col2.metric("GAB actifs", f"{nb_gab_actifs}")
    col3.metric("GAB critique", f"{nb_critique}")
    col4.metric("GAB alerte", f"{nb_alerte}")

    # Carte interactive
    st.subheader("Carte interactive du réseau")
    m = folium.Map(location=[34, -6], zoom_start=6)
    status_colors = {"Normal":"green", "Alerte":"orange", "Critique":"red"}
    for _, row in df_filtered.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color=status_colors.get(row["status"], "blue"),
            fill=True,
            fill_color=status_colors.get(row["status"], "blue"),
            popup=f"GAB: {row['num_gab']}<br>Agence: {row['agence']}<br>Cash: {row['total_montant']}"
        ).add_to(m)
    st_folium(m, width=700)

# ========================================
# Onglet 2 : Prévisions LSTM multi-GAB
# ========================================
if tab == "Prévisions LSTM multi-GAB":
    st.title("Prévisions LSTM multi-GAB")

    gab_options = [gab for gab in sorted(df["num_gab"].unique()) if gab in lstm_models]
    if not gab_options:
        st.warning("Aucun GAB disponible avec modèles LSTM.")
    else:
        gab_selected = st.multiselect("Sélectionner un ou plusieurs GABs", gab_options, default=gab_options[:3])
        future_steps = st.slider("Nombre de semaines futures", 1, 12, 4)

        fig_pred = go.Figure()
        combined_csv = []

        for gab in gab_selected:
            df_gab = df[df["num_gab"] == gab].sort_values("ds")
            scaler = lstm_scalers[gab]
            model = lstm_models[gab]

            y_scaled = scaler.transform(df_gab[['y']].values)
            X = []
            n_steps = 4
            for i in range(len(y_scaled) - n_steps):
                X.append(y_scaled[i:i+n_steps])
            X = np.array(X).reshape(-1, n_steps, 1)

            y_pred_scaled = model.predict(X, verbose=0)
            y_pred = scaler.inverse_transform(y_pred_scaled)
            y_true = df_gab['y'].values[n_steps:]
            dates = df_gab['ds'][n_steps:]

            # Prévisions futures
            last_seq = y_scaled[-n_steps:].reshape(1, n_steps, 1)
            future_preds = []
            future_dates = [df_gab["ds"].max() + timedelta(weeks=i+1) for i in range(future_steps)]
            for _ in range(future_steps):
                pred_scaled = model.predict(last_seq, verbose=0)
                pred = scaler.inverse_transform(pred_scaled)[0,0]
                future_preds.append(pred)
                last_seq = np.concatenate([last_seq[:,1:,:], pred_scaled.reshape(1,1,1)], axis=1)

            # Graphique
            fig_pred.add_trace(go.Scatter(x=dates, y=y_true/1000, mode="lines+markers", name=f"{gab} réel"))
            fig_pred.add_trace(go.Scatter(x=dates, y=y_pred.flatten()/1000, mode="lines+markers", name=f"{gab} prédiction"))
            fig_pred.add_trace(go.Scatter(x=future_dates, y=np.array(future_preds)/1000, mode="lines+markers", name=f"{gab} futur"))

            # CSV combiné
            combined_csv.append(pd.DataFrame({
                "num_gab": gab,
                "ds": list(dates) + future_dates,
                "y_true_kdh": list(y_true/1000) + [None]*future_steps,
                "y_pred_kdh": list(y_pred.flatten()/1000) + list(np.array(future_preds)/1000)
            }))

        fig_pred.update_layout(xaxis_title="Date", yaxis_title="Montant retiré (KDH)")
        st.plotly_chart(fig_pred, use_container_width=True)

        # Téléchargement CSV combiné
        df_csv_combined = pd.concat(combined_csv, ignore_index=True)
        st.download_button(
            label="Télécharger prévisions multi-GAB CSV",
            data=df_csv_combined.to_csv(index=False),
            file_name="predictions_multi_gab.csv",
            mime="text/csv"
        )
