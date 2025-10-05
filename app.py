import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import glob
from tensorflow.keras.models import load_model
import joblib

# ========================================
# Configuration
# ========================================
st.set_page_config(page_title="Dashboard GAB - Cash Management", layout="wide")

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
    return df

df = load_data()

# ========================================
# Chargement des modèles LSTM
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
        except:
            continue
    return models, scalers

lstm_models, lstm_scalers = load_lstm_models()

# ========================================
# Onglets
# ========================================
tab = st.sidebar.radio("Navigation", ["Tableau de bord analytique", "Prévisions LSTM"])

# ========================================
# Onglet 1 : Tableau de bord analytique
# ========================================
if tab == "Tableau de bord analytique":
    st.title("Tableau de bord analytique - GAB")

    # Filtres
    regions = df["region"].dropna().unique()
    region = st.sidebar.selectbox("Région", ["Toutes"] + sorted(regions.tolist()))
    agences = df[df["region"] == region]["agence"].dropna().unique() if region!="Toutes" else df["agence"].dropna().unique()
    agence = st.sidebar.selectbox("Agence", ["Toutes"] + sorted(agences.tolist()))
    gabs = df[df["agence"] == agence]["num_gab"].dropna().unique() if agence!="Toutes" else df["num_gab"].dropna().unique()
    gab = st.sidebar.selectbox("GAB", ["Tous"] + sorted(gabs.tolist()))
    date_min, date_max = df["ds"].min(), df["ds"].max()
    date_debut = st.sidebar.date_input("Date début", date_min)
    date_fin = st.sidebar.date_input("Date fin", date_max)

    df_filtered = df.copy()
    if region != "Toutes": df_filtered = df_filtered[df_filtered["region"] == region]
    if agence != "Toutes": df_filtered = df_filtered[df_filtered["agence"] == agence]
    if gab != "Tous": df_filtered = df_filtered[df_filtered["num_gab"] == gab]
    df_filtered = df_filtered[(df_filtered["ds"]>=pd.to_datetime(date_debut)) & (df_filtered["ds"]<=pd.to_datetime(date_fin))]

    # KPIs
    st.subheader("KPIs principaux")
    volume_moyen = df_filtered.groupby("week")["total_montant"].mean().mean()
    nb_operations = df_filtered["total_nombre"].sum()
    nb_gab_actifs = df_filtered["num_gab"].nunique()
    ecart_type = df_filtered["total_montant"].std()
    part_weekend = df_filtered[df_filtered["week_day"]>=5]["total_montant"].sum()/df_filtered["total_montant"].sum()*100
    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric("Volume moyen hebdo", f"{volume_moyen/1000:,.0f} KDH")
    col2.metric("Nombre opérations", f"{nb_operations/1000:,.0f} KDH")
    col3.metric("GAB actifs", f"{nb_gab_actifs}")
    col4.metric("Écart-type retraits", f"{ecart_type/1000:,.0f} KDH")
    col5.metric("Part week-end", f"{part_weekend:.1f} %")

    # Graphique évolution
    st.subheader("Évolution des retraits")
    level_options = ["Global"] + sorted(df_filtered["region"].unique()) + sorted(df_filtered["num_gab"].unique())
    selected_level = st.selectbox("Niveau", level_options)
    if selected_level=="Global":
        df_plot = df_filtered.groupby("ds")["total_montant"].sum().reset_index()
    elif selected_level in df_filtered["region"].unique():
        df_plot = df_filtered[df_filtered["region"]==selected_level].groupby("ds")["total_montant"].sum().reset_index()
    else:
        df_plot = df_filtered[df_filtered["num_gab"]==selected_level].groupby("ds")["total_montant"].sum().reset_index()
    df_plot["total_montant_kdh"] = df_plot["total_montant"]/1000
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df_plot["ds"], y=df_plot["total_montant_kdh"], mode="lines+markers", name="Montant retiré (KDH)"))
    fig_line.update_layout(xaxis_title="Date", yaxis_title="Montant retiré (KDH)", title="Évolution")
    st.plotly_chart(fig_line,use_container_width=True)

# ========================================
# Onglet 2 : Prévisions LSTM multi-GAB
# ========================================
if tab == "Prévisions LSTM":
    st.title("Prévisions LSTM multi-GAB")
    gab_options = [gab for gab in sorted(df["num_gab"].unique()) if gab in lstm_models]
    gab_selected_list = st.multiselect("Sélectionner GABs", gab_options)
    future_weeks = st.slider("Semaines futures", 1, 12, 4)

    if gab_selected_list:
        fig_pred = go.Figure()
        df_all_preds = pd.DataFrame()

        for gab_selected in gab_selected_list:
            df_gab = df[df["num_gab"]==gab_selected].sort_values("ds")
            if len(df_gab)<52: 
                st.warning(f"{gab_selected}: pas assez de données")
                continue

            # LSTM
            n_steps = 4
            scaler = lstm_scalers[gab_selected]
            model = lstm_models[gab_selected]
            y_scaled = scaler.transform(df_gab[['y']].values)
            X = np.array([y_scaled[i:i+n_steps] for i in range(len(y_scaled)-n_steps)]).reshape(-1,n_steps,1)
            y_pred_scaled = model.predict(X,verbose=0)
            y_pred = scaler.inverse_transform(y_pred_scaled)
            dates = df_gab['ds'][n_steps:]
            y_true = df_gab['y'].values[n_steps:]

            # Prévisions futures
            last_seq = y_scaled[-n_steps:].reshape(1,n_steps,1)
            future_preds = []
            future_dates = [df_gab['ds'].max()+pd.Timedelta(weeks=i+1) for i in range(future_weeks)]
            for _ in range(future_weeks):
                pred_scaled = model.predict(last_seq, verbose=0)
                pred = scaler.inverse_transform(pred_scaled)[0,0]
                future_preds.append(pred/1000)
                last_seq = np.concatenate([last_seq[:,1:,:], pred_scaled.reshape(1,1,1)],axis=1)

            # Graphique
            fig_pred.add_trace(go.Scatter(x=dates, y=y_true/1000, mode="lines+markers", name=f"{gab_selected} Réel"))
            fig_pred.add_trace(go.Scatter(x=dates, y=y_pred.flatten()/1000, mode="lines+markers", name=f"{gab_selected} LSTM"))
            fig_pred.add_trace(go.Scatter(x=future_dates, y=future_preds, mode="lines+markers", name=f"{gab_selected} Futur"))

            # DataFrame export
            df_gab_csv = pd.DataFrame({
                "ds": list(dates)+future_dates,
                "gab": gab_selected,
                "y_true_kdh": list(y_true/1000)+[None]*future_weeks,
                "y_pred_kdh": list(y_pred.flatten()/1000)+future_preds
            })
            df_all_preds = pd.concat([df_all_preds, df_gab_csv])

            # Alerte seuil
            seuil = y_true.mean() + 2*y_true.std()
            if any(np.array(future_preds)*1000 > seuil):
                st.warning(f"{gab_selected}: Prévision dépasse seuil critique ({seuil:,.0f} DH)")

        fig_pred.update_layout(xaxis_title="Date", yaxis_title="Montant retiré (KDH)", title="Prévisions LSTM multi-GAB")
        st.plotly_chart(fig_pred,use_container_width=True)

        # Téléchargement CSV
        st.download_button(
            label="Télécharger prévisions CSV",
            data=df_all_preds.to_csv(index=False),
            file_name="multi_gab_predictions.csv",
            mime="text/csv"
        )
