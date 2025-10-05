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
    df = pd.read_csv("df_weekly_clean.csv", parse_dates=["ds"])
    df["num_gab"] = pd.to_numeric(df["num_gab"], errors="coerce")
    df["week_day"] = df["ds"].dt.dayofweek
    df["week"] = df["ds"].dt.isocalendar().week
    df["year"] = df["ds"].dt.year
    return df

@st.cache_data
def load_subset():
    df_subset = pd.read_csv("df_subset.csv", parse_dates=["ds"])
    df_subset["num_gab"] = pd.to_numeric(df_subset["num_gab"], errors="coerce")
    return df_subset

df = load_data()
df_subset = load_subset()

# ========================================
# Chargement des modèles LSTM
# ========================================
@st.cache_data
def load_lstm_models():
    models = {}
    scalers = {}
    for model_file in glob.glob("lstm_gab_*.h5"):
        gab_id = model_file.split("_")[-1].replace(".h5","")
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
tab = st.sidebar.radio("Navigation", ["Tableau de bord analytique", "Prévisions LSTM 20 GAB"])

# ========================================
# Onglet 1 : Tableau de bord analytique
# ========================================
if tab == "Tableau de bord analytique":
    st.title("Tableau de bord analytique - GAB")

    # =====================
    # Sidebar filtres
    # =====================
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

    # =====================
    # Appliquer filtres
    # =====================
    df_filtered = df.copy()
    if region != "Toutes":
        df_filtered = df_filtered[df_filtered["region"] == region]
    if agence != "Toutes":
        df_filtered = df_filtered[df_filtered["agence"] == agence]
    if gab != "Tous":
        df_filtered = df_filtered[df_filtered["num_gab"] == gab]
    df_filtered = df_filtered[(df_filtered["ds"] >= pd.to_datetime(date_debut)) &
                              (df_filtered["ds"] <= pd.to_datetime(date_fin))]

    # =====================
    # KPIs principaux
    # =====================
    st.subheader("KPIs principaux")
    volume_moyen_semaine = df_filtered.groupby("week")["total_montant"].mean().mean()
    nombre_operations = df_filtered["total_nombre"].sum()
    nombre_gab_actifs = df_filtered["num_gab"].nunique()
    ecart_type_retraits = df_filtered["total_montant"].std()
    part_weekend = df_filtered[df_filtered["week_day"]>=5]["total_montant"].sum() / df_filtered["total_montant"].sum() * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Volume moyen hebdo", f"{volume_moyen_semaine:,.0f} DH")
    col2.metric("Nombre total d'opérations", f"{nombre_operations:,.0f}")
    col3.metric("Nombre de GAB actifs", f"{nombre_gab_actifs}")
    col4.metric("Écart-type des retraits", f"{ecart_type_retraits:,.0f} DH")
    col5.metric("Part des retraits week-end", f"{part_weekend:.1f} %")

    # =====================
    # Camembert - Montant moyen hebdo par région et année
    # =====================
    st.subheader("Répartition des retraits hebdo par région (par année)")
    years = sorted(df_filtered["year"].unique())
    selected_year = st.selectbox("Sélectionner l'année", years, key="year_pie")

    df_year = df_filtered[df_filtered["year"] == selected_year]
    df_pie = df_year.groupby("region")["total_montant"].mean().reset_index()
    df_pie.rename(columns={"total_montant":"Montant moyen hebdo"}, inplace=True)
    fig_pie = px.pie(df_pie, names="region", values="Montant moyen hebdo",
                     title=f"Montant moyen hebdo par région en {selected_year}")
    st.plotly_chart(fig_pie, use_container_width=True)

    # =====================
    # Graphique d'évolution des retraits
    # =====================
    st.subheader("Évolution des retraits")
    level_options = ["Global"] + sorted(df_filtered["region"].unique()) + sorted(df_filtered["num_gab"].unique())
    selected_level = st.selectbox("Sélectionner le niveau", level_options, key="evol_level")

    if selected_level == "Global":
        df_plot = df_filtered.groupby("ds")["total_montant"].sum().reset_index()
        title = "Évolution des retraits globaux"
    elif selected_level in df_filtered["region"].unique():
        df_plot = df_filtered[df_filtered["region"] == selected_level].groupby("ds")["total_montant"].sum().reset_index()
        title = f"Évolution des retraits - Région {selected_level}"
    else:
        df_plot = df_filtered[df_filtered["num_gab"] == selected_level].groupby("ds")["total_montant"].sum().reset_index()
        title = f"Évolution des retraits - GAB {selected_level}"

    fig_line = px.line(df_plot, x="ds", y="total_montant", title=title,
                       labels={"ds":"Semaine", "total_montant":"Montant retiré"})
    st.plotly_chart(fig_line, use_container_width=True)

# ========================================
# Onglet 2 : Prévisions LSTM 20 GAB
# ========================================
if tab == "Prévisions LSTM 20 GAB":
    st.title("Prévisions LSTM - 20 GAB")

    # S'assurer que num_gab est string pour matcher avec les modèles
    df_subset["num_gab"] = df_subset["num_gab"].astype(str)
    lstm_models_str = {str(k): v for k, v in lstm_models.items()}
    lstm_scalers_str = {str(k): v for k, v in lstm_scalers.items()}

    # Seuls les GAB pour lesquels un modèle et scaler existent
    gab_options = [gab for gab in sorted(df_subset["num_gab"].unique()) if gab in lstm_models_str]

    if not gab_options:
        st.warning("Aucun GAB disponible avec modèles LSTM.")
    else:
        gab_selected = st.selectbox("Sélectionner un GAB", gab_options)

        # Récupérer les données historiques depuis df_subset
        df_gab = df_subset[df_subset["num_gab"] == gab_selected].sort_values("ds").copy()
        df_gab['ds'] = pd.to_datetime(df_gab['ds'])

        if len(df_gab) < 52:
            st.warning("Pas assez de données pour effectuer une prévision LSTM (minimum 52 semaines).")
        else:
            st.subheader(f"Visualisation des données et prévisions pour GAB {gab_selected}")

            # Re-créer toutes les features comme à l'entraînement
            df_gab['weekday'] = df_gab['ds'].dt.weekday
            df_gab['month'] = df_gab['ds'].dt.month
            df_gab['week_of_month'] = ((df_gab['week'] - 1) % 4) + 1
            df_gab['is_weekend'] = (df_gab['weekday'] >= 5).astype(int)
            df_gab['is_month_end'] = (df_gab['ds'].dt.day >= 25).astype(int)
            df_gab['rolling_mean_4'] = df_gab['total_montant'].rolling(4, min_periods=1).mean()
            df_gab['diff_last_week'] = df_gab['total_montant'].diff().fillna(0)

            features = ['total_montant','week_of_month','month','is_weekend','is_month_end','rolling_mean_4','diff_last_week']

            # Charger scaler et modèle
            scaler = lstm_scalers_str[gab_selected]
            model = lstm_models_str[gab_selected]

            # Normalisation
            data_scaled = scaler.transform(df_gab[features].values)

            # Prévisions futures 4 semaines
            sequence_length = 5
            forecast_steps = 4
            last_sequence = data_scaled[-sequence_length:].reshape(1, sequence_length, len(features))
            future_preds = []
            future_dates = []

            for i in range(forecast_steps):
                pred_scaled = model.predict(last_sequence, verbose=0)
                pred = scaler.inverse_transform(
                    np.hstack([pred_scaled, np.zeros((1, len(features)-1))])
                )[0, 0]
                future_preds.append(pred)

                # Créer la prochaine séquence
                last_features = last_sequence[0, -1, 1:]  # features hors total_montant
                next_scaled = np.hstack([pred_scaled, last_features]).reshape(1,1,len(features))
                last_sequence = np.concatenate([last_sequence[:,1:,:], next_scaled], axis=1)

                # Dates futures (1 semaine après la dernière date connue)
                next_date = df_gab["ds"].iloc[-1] + pd.Timedelta(weeks=i+1)
                future_dates.append(next_date)

            # DataFrame pour affichage et export
            df_pred = pd.DataFrame({
                "ds": future_dates,
                "total_montant_pred": future_preds
            })

            # Graphique
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=df_gab["ds"], y=df_gab["total_montant"],
                                          mode="lines+markers", name="Montant réel"))
            fig_pred.add_trace(go.Scatter(x=df_pred["ds"], y=df_pred["total_montant_pred"],
                                          mode="lines+markers", name="Prévisions 4 semaines"))
            fig_pred.update_layout(xaxis_title="Date", yaxis_title="Montant retiré")
            st.plotly_chart(fig_pred, use_container_width=True)

            # Bouton téléchargement CSV
            st.download_button(
                label="Télécharger prévisions CSV",
                data=df_pred.to_csv(index=False),
                file_name=f"forecast_4weeks_{gab_selected}.csv",
                mime="text/csv"
            )
