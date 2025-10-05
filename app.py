import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import os

# ========================================
# Configuration de la page
# ========================================
st.set_page_config(page_title="Dashboard GAB", layout="wide")

# ========================================
# Chargement des données
# ========================================
@st.cache_data
def load_data():
    df = pd.read_csv("df_weekly_clean.csv", parse_dates=["ds"])
    df["num_gab"] = df["num_gab"].astype(str).str.strip()
    df["week_day"] = df["ds"].dt.dayofweek
    df["week"] = df["ds"].dt.isocalendar().week
    df["year"] = df["ds"].dt.year
    return df

df = load_data()

# ========================================
# Chargement des modèles LSTM
# ========================================
@st.cache_data
def load_lstm_models():
    models = {}
    scalers = {}

    st.write("Debug fichiers modèles et scalers")
    st.write(f"Répertoire courant: {os.getcwd()}")

    model_files = glob.glob("lstm_gab_*.h5")
    scaler_files = glob.glob("scaler_gab_*.save")
    
    st.write("Modèles LSTM trouvés:\n", model_files)
    st.write("Scalers trouvés:\n", scaler_files)

    # Extraire l'ID du GAB pour chaque fichier
    model_ids = [os.path.basename(f).split("_")[-1].replace(".h5","") for f in model_files]
    scaler_ids = [os.path.basename(f).split("_")[-1].replace(".save","") for f in scaler_files]

    # Détection des GAB ayant à la fois modèle et scaler
    gab_common = list(set(model_ids) & set(scaler_ids))
    st.write("GAB détectés avec modèles et scalers:\n", gab_common)

    # Charger modèles et scalers
    for gab_id in gab_common:
        try:
            models[gab_id] = load_model(f"lstm_gab_{gab_id}.h5", compile=False)
            scalers[gab_id] = joblib.load(f"scaler_gab_{gab_id}.save")
        except Exception as e:
            st.warning(f"Impossible de charger LSTM pour {gab_id}: {e}")

    return models, scalers

lstm_models, lstm_scalers = load_lstm_models()

# ========================================
# Onglets
# ========================================
tab = st.sidebar.radio("Navigation", ["Tableau de bord analytique", "Prévisions LSTM 20 GAB"])

# ========================================
# Onglet 1 : Tableau de bord analytique
# ========================================
if tab == "Tableau de bord analytique":
    st.title("Tableau de bord analytique - GAB")

    # Sidebar filtres
    st.sidebar.header("Filtres")
    regions = df["region"].dropna().unique()
    region = st.sidebar.selectbox("Région", ["Toutes"] + sorted(regions.tolist()))

    if region != "Toutes":
        agences = df[df["region"] == region]["agence"].dropna().unique()
    else:
        agences = df["agence"].dropna().unique()
    agence = st.sidebar.selectbox("Agence", ["Toutes"] + sorted(agences.tolist()))

    if agence != "Toutes":
        gabs = df[df["agence"] == agence]["num_gab"].dropna().unique()
    else:
        gabs = df["num_gab"].dropna().unique()
    gab = st.sidebar.selectbox("GAB", ["Tous"] + sorted(gabs.tolist()))

    # Filtre de dates
    date_min = df["ds"].min()
    date_max = df["ds"].max()
    date_debut = st.sidebar.date_input("Date début", date_min)
    date_fin = st.sidebar.date_input("Date fin", date_max)

    # Appliquer filtres
    df_filtered = df.copy()
    if region != "Toutes":
        df_filtered = df_filtered[df_filtered["region"] == region]
    if agence != "Toutes":
        df_filtered = df_filtered[df_filtered["agence"] == agence]
    if gab != "Tous":
        df_filtered = df_filtered[df_filtered["num_gab"] == gab]
    df_filtered = df_filtered[(df_filtered["ds"] >= pd.to_datetime(date_debut)) &
                              (df_filtered["ds"] <= pd.to_datetime(date_fin))]

    # KPIs principaux
    st.subheader("KPIs principaux")
    volume_moyen_semaine = df_filtered.groupby("week")["total_montant"].mean().mean()
    nombre_operations = df_filtered["total_nombre"].sum()
    nombre_gab_actifs = df_filtered["num_gab"].nunique()
    ecart_type_retraits = df_filtered["total_montant"].std()
    part_weekend = df_filtered[df_filtered["week_day"]>=5]["total_montant"].sum() / df_filtered["total_montant"].sum() * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Volume moyen hebdo", f"{volume_moyen_semaine/1000:,.0f} KDH")
    col2.metric("Nombre total d'opérations", f"{nombre_operations/1000:,.0f} KDH")
    col3.metric("Nombre de GAB actifs", f"{nombre_gab_actifs}")
    col4.metric("Écart-type des retraits", f"{ecart_type_retraits/1000:,.0f} KDH")
    col5.metric("Part des retraits week-end", f"{part_weekend:.1f} %")

    # Camembert
    st.subheader("Répartition des retraits hebdo par région (par année)")
    years = sorted(df_filtered["year"].unique())
    selected_year = st.selectbox("Sélectionner l'année", years, key="year_pie")
    df_year = df_filtered[df_filtered["year"] == selected_year]
    df_pie = df_year.groupby("region")["total_montant"].mean().reset_index()
    df_pie["total_montant_kdh"] = df_pie["total_montant"] / 1000
    fig_pie = px.pie(df_pie, names="region", values="total_montant_kdh",
                     title=f"Montant moyen hebdo par région en {selected_year}")
    st.plotly_chart(fig_pie, use_container_width=True)

    # Graphique évolution
    st.subheader("Évolution des retraits")
    level_options = ["Global"] + sorted(df_filtered["region"].unique()) + sorted(df_filtered["num_gab"].unique())
    selected_level = st.selectbox("Sélectionner le niveau", level_options, key="evol_level")

    if selected_level == "Global":
        df_plot = df_filtered.groupby("ds")["total_montant"].sum().reset_index()
        df_plot["total_montant_kdh"] = df_plot["total_montant"] / 1000
        title = "Évolution des retraits globaux"
    elif selected_level in df_filtered["region"].unique():
        df_plot = df_filtered[df_filtered["region"] == selected_level].groupby("ds")["total_montant"].sum().reset_index()
        df_plot["total_montant_kdh"] = df_plot["total_montant"] / 1000
        title = f"Évolution des retraits - Région {selected_level}"
    else:
        df_plot = df_filtered[df_filtered["num_gab"] == selected_level].groupby("ds")["total_montant"].sum().reset_index()
        df_plot["total_montant_kdh"] = df_plot["total_montant"] / 1000
        title = f"Évolution des retraits - GAB {selected_level}"

    fig_line = px.line(df_plot, x="ds", y="total_montant_kdh", title=title,
                       labels={"ds":"Semaine", "total_montant_kdh":"Montant retiré (KDH)"})
    st.plotly_chart(fig_line, use_container_width=True)

# ========================================
# Onglet 2 : Prévisions LSTM 20 GAB
# ========================================
if tab == "Prévisions LSTM 20 GAB":
    st.title("Prévisions LSTM - 20 GAB")

    # Transformer les numéros GAB en string pour correspondre aux modèles
    df["num_gab"] = df["num_gab"].astype(str)
    lstm_models_str = {str(k): v for k, v in lstm_models.items()}
    lstm_scalers_str = {str(k): v for k, v in lstm_scalers.items()}

    # Liste des GAB disponibles avec modèles
    gab_options = [gab for gab in sorted(df["num_gab"].unique()) if gab in lstm_models_str]

    if not gab_options:
        st.warning("Aucun GAB disponible avec modèles LSTM.")
    else:
        gab_selected = st.selectbox("Sélectionner un GAB", gab_options)
        df_gab = df[df["num_gab"] == gab_selected].sort_values("ds")

        if len(df_gab) < 52:
            st.warning("Pas assez de données pour effectuer une prévision LSTM (minimum 52 semaines).")
        else:
            st.subheader(f"Visualisation des données et prévisions pour GAB {gab_selected}")

            try:
                # ====== Préparation des données ======
                feature_col = ['y']  # correspond à ton entraînement
                scaler = lstm_scalers_str[gab_selected]
                model = lstm_models_str[gab_selected]

                # Normalisation des données
                data_scaled = scaler.transform(df_gab[feature_col].values.reshape(-1,1))

                # Séquences pour LSTM
                n_steps = 4
                X = []
                for i in range(len(data_scaled)-n_steps):
                    X.append(data_scaled[i:i+n_steps])
                X = np.array(X)
                X = X.reshape((X.shape[0], X.shape[1], 1))

                # Prédictions sur toutes les séquences
                y_pred_scaled = model.predict(X, verbose=0)
                y_pred = scaler.inverse_transform(y_pred_scaled)

                # Ajuster y_true
                y_true = df_gab['y'].values[n_steps:]

                # ====== Prévisions futures ======
                last_sequence = data_scaled[-n_steps:].reshape(1, n_steps, 1)
                forecast_steps = 4
                future_preds = []
                for _ in range(forecast_steps):
                    pred_scaled = model.predict(last_sequence, verbose=0)
                    pred = scaler.inverse_transform(pred_scaled)[0,0]
                    future_preds.append(pred)
                    last_sequence = np.concatenate([last_sequence[:,1:,:], pred_scaled.reshape(1,1,1)], axis=1)

                # Dates futures
                last_date = df_gab["ds"].max()
                future_dates = [last_date + pd.Timedelta(weeks=i+1) for i in range(forecast_steps)]

                # ====== DataFrame pour le graphe ======
                df_pred = pd.DataFrame({
                    "ds": list(df_gab["ds"][n_steps:]) + future_dates,
                    "Montant réel": list(y_true/1000) + [None]*forecast_steps,
                    "Montant prédit LSTM": list(y_pred.flatten()/1000) + [fp/1000 for fp in future_preds]
                })

                # ====== Graphique ======
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=df_pred["ds"], y=df_pred["Montant réel"],
                    mode="lines+markers", name="Montant réel (KDH)"
                ))
                fig_pred.add_trace(go.Scatter(
                    x=df_pred["ds"], y=df_pred["Montant prédit LSTM"],
                    mode="lines+markers", name="Montant prédit LSTM (KDH)"
                ))
                fig_pred.update_layout(
                    title=f"GAB {gab_selected} - Prédictions LSTM",
                    xaxis_title="Date",
                    yaxis_title="Montant retiré (KDH)"
                )
                st.plotly_chart(fig_pred, use_container_width=True)

                # ====== Téléchargement CSV ======
                st.download_button(
                    label="Télécharger prévisions CSV",
                    data=df_pred.to_csv(index=False),
                    file_name=f"pred_{gab_selected}.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Erreur lors de la génération des prévisions: {e}")
