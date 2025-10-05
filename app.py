import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import glob
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# ========================================
# CONFIGURATION DE LA PAGE
# ========================================
st.set_page_config(page_title="CashStream - Optimisation du Cash GAB", layout="wide")

st.markdown("<h1 style='text-align: center; color: #004aad;'>💰 CashStream – Tableau de Bord Intelligent</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Solution intelligente pour la gestion proactive du cash et la prévision des besoins GAB.</p>", unsafe_allow_html=True)

# ========================================
# CHARGEMENT DES DONNÉES
# ========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("df_weekly_clean.csv", parse_dates=["ds"])
        df["num_gab"] = df["num_gab"].astype(str)
        df["week_day"] = df["ds"].dt.dayofweek
        df["week"] = df["ds"].dt.isocalendar().week
        df["year"] = df["ds"].dt.year

        try:
            df_gabs = pd.read_csv("df_gabs.csv")
            if "num_gab" in df_gabs.columns:
                df = df.merge(df_gabs, on="num_gab", how="left")
        except FileNotFoundError:
            st.warning("⚠️ Le fichier df_gabs.csv n’a pas été trouvé. Les infos d’agence/région ne seront pas affichées.")

        return df
    except FileNotFoundError:
        st.error("🚨 Fichier df_weekly_clean.csv introuvable dans le dossier du projet.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return pd.DataFrame()

df = load_data()

# ========================================
# CHARGEMENT DES MODÈLES LSTM
# ========================================
@st.cache_data
def load_lstm_models():
    models, scalers = {}, {}
    for model_file in glob.glob("lstm_gab_*.h5"):
        gab_id = model_file.split("_")[-1].replace(".h5", "")
        scaler_file = f"scaler_gab_{gab_id}.save"
        try:
            models[gab_id] = load_model(model_file, compile=False)
            scalers[gab_id] = joblib.load(scaler_file)
        except Exception as e:
            st.warning(f"Impossible de charger le modèle LSTM pour {gab_id}: {e}")
    return models, scalers

lstm_models, lstm_scalers = load_lstm_models()

# ========================================
# NAVIGATION
# ========================================
tab = st.sidebar.radio("📊 Navigation", ["Tableau de bord analytique", "Prévisions LSTM 20 GAB"])

# ========================================
# TABLEAU DE BORD ANALYTIQUE
# ========================================
if tab == "Tableau de bord analytique":
    st.subheader("📈 Analyse des retraits GAB")

    # ---- FILTRES ----
    st.sidebar.header("🧭 Filtres")
    regions = df["region"].dropna().unique() if "region" in df.columns else []
    region = st.sidebar.selectbox("Région", ["Toutes"] + sorted(regions.tolist()))

    if region != "Toutes" and "agence" in df.columns:
        agences = df[df["region"] == region]["agence"].dropna().unique()
    else:
        agences = df["agence"].dropna().unique() if "agence" in df.columns else []
    agence = st.sidebar.selectbox("Agence", ["Toutes"] + sorted(agences.tolist()))

    if agence != "Toutes":
        gabs = df[df["agence"] == agence]["num_gab"].dropna().unique()
    else:
        gabs = df["num_gab"].dropna().unique()
    gab = st.sidebar.selectbox("GAB", ["Tous"] + sorted(gabs.tolist()))

    # ---- DATES ----
    date_min, date_max = df["ds"].min(), df["ds"].max()
    date_debut = st.sidebar.date_input("Date début", date_min)
    date_fin = st.sidebar.date_input("Date fin", date_max)

    # ---- APPLICATION DES FILTRES ----
    df_filtered = df.copy()
    if region != "Toutes":
        df_filtered = df_filtered[df_filtered["region"] == region]
    if agence != "Toutes":
        df_filtered = df_filtered[df_filtered["agence"] == agence]
    if gab != "Tous":
        df_filtered = df_filtered[df_filtered["num_gab"] == gab]
    df_filtered = df_filtered[(df_filtered["ds"] >= pd.to_datetime(date_debut)) &
                              (df_filtered["ds"] <= pd.to_datetime(date_fin))]

    if df_filtered.empty:
        st.warning("Aucune donnée disponible pour ces filtres.")
        st.stop()

    # ---- KPI ----
    st.subheader("🔹 Indicateurs de Performance")
    total_montant = df_filtered["total_montant"].sum() / 1000
    total_operations = df_filtered["total_nombre"].sum()
    nombre_gab = df_filtered["num_gab"].nunique()
    volume_moyen = df_filtered.groupby("week")["total_montant"].mean().mean() / 1000
    ecart_type = df_filtered["total_montant"].std() / 1000
    part_weekend = df_filtered[df_filtered["week_day"] >= 5]["total_montant"].sum() / df_filtered["total_montant"].sum() * 100

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("💸 Montant total retiré", f"{total_montant:,.0f} KDH")
    col2.metric("🏧 Nombre de GAB actifs", nombre_gab)
    col3.metric("🔢 Nombre total d’opérations", f"{total_operations:,}")
    col4.metric("📊 Volume moyen hebdo", f"{volume_moyen:,.0f} KDH")
    col5.metric("📉 Écart-type des retraits", f"{ecart_type:,.0f} KDH")
    col6.metric("📅 Part retraits week-end", f"{part_weekend:.1f}%")

    # ---- GRAPHIQUE EVOLUTION ----
    st.subheader("📊 Évolution des retraits")
    level_options = ["Global"] + sorted(df_filtered["region"].unique()) + sorted(df_filtered["num_gab"].unique())
    selected_level = st.selectbox("Niveau d’analyse", level_options, key="evol_level")

    if selected_level == "Global":
        df_plot = df_filtered.groupby("ds")["total_montant"].sum().reset_index()
        title = "Évolution des retraits globaux"
    elif selected_level in df_filtered["region"].unique():
        df_plot = df_filtered[df_filtered["region"] == selected_level].groupby("ds")["total_montant"].sum().reset_index()
        title = f"Évolution des retraits – Région {selected_level}"
    else:
        df_plot = df_filtered[df_filtered["num_gab"] == selected_level].groupby("ds")["total_montant"].sum().reset_index()
        title = f"Évolution des retraits – GAB {selected_level}"

    df_plot["total_montant_kdh"] = df_plot["total_montant"] / 1000
    df_plot["moyenne_mobile"] = df_plot["total_montant_kdh"].rolling(window=4).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot["ds"], y=df_plot["total_montant_kdh"], mode="lines+markers", name="Montant retiré (KDH)"))
    fig.add_trace(go.Scatter(x=df_plot["ds"], y=df_plot["moyenne_mobile"], mode="lines", name="Moyenne mobile (4 sem)", line=dict(dash="dash")))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Montant retiré (KDH)")
    st.plotly_chart(fig, use_container_width=True)

# ========================================
# PRÉVISIONS LSTM
# ========================================
if tab == "Prévisions LSTM 20 GAB":
    st.title("🤖 Prévisions hebdomadaires - Modèle LSTM")

    gab_options = [gab for gab in sorted(df["num_gab"].unique()) if gab in lstm_models]
    if not gab_options:
        st.warning("Aucun modèle LSTM disponible.")
    else:
        gab_selected = st.selectbox("Sélectionner un GAB", gab_options)
        df_gab = df[df["num_gab"] == gab_selected].sort_values("ds")

        if len(df_gab) < 52:
            st.warning("Pas assez de données pour ce GAB.")
        else:
            try:
                n_steps = 4
                scaler = lstm_scalers[gab_selected]
                model = lstm_models[gab_selected]
                y_scaled = scaler.transform(df_gab[['y']].values)
                X = np.array([y_scaled[i:i+n_steps] for i in range(len(y_scaled)-n_steps)]).reshape(-1, n_steps, 1)

                y_pred_scaled = model.predict(X, verbose=0)
                y_pred = scaler.inverse_transform(y_pred_scaled)
                y_true = df_gab['y'].values[n_steps:]
                dates = df_gab['ds'][n_steps:]

                future_preds, last_seq = [], y_scaled[-n_steps:].reshape(1, n_steps, 1)
                for _ in range(6):
                    pred_scaled = model.predict(last_seq, verbose=0)
                    pred = scaler.inverse_transform(pred_scaled)[0, 0]
                    future_preds.append(pred/1000)
                    last_seq = np.concatenate([last_seq[:,1:,:], pred_scaled.reshape(1,1,1)], axis=1)
                future_dates = [df_gab["ds"].max() + pd.Timedelta(weeks=i+1) for i in range(6)]

                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=dates, y=y_true/1000, mode="lines+markers", name="Réel (KDH)"))
                fig_pred.add_trace(go.Scatter(x=dates, y=y_pred.flatten()/1000, mode="lines+markers", name="Prédiction (KDH)"))
                fig_pred.add_trace(go.Scatter(x=future_dates, y=future_preds, mode="lines+markers", name="Prévision future (KDH)"))
                fig_pred.update_layout(title=f"Prévision des retraits - GAB {gab_selected}", xaxis_title="Date", yaxis_title="Montant (KDH)")
                st.plotly_chart(fig_pred, use_container_width=True)

                df_csv = pd.DataFrame({"ds": list(dates) + future_dates, "y_true_kdh": list(y_true/1000) + [None]*6, "y_pred_kdh": list(y_pred.flatten()/1000) + future_preds})
                st.download_button("📥 Télécharger les prévisions", data=df_csv.to_csv(index=False), file_name=f"pred_{gab_selected}.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Erreur : {e}")
