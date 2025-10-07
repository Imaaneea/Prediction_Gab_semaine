import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import glob
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# ========================================
# 1. Configuration de la page et Style CSS
# ========================================
st.set_page_config(page_title="Dashboard de Trésorerie", layout="wide")

# Injection de CSS pour un design moderne inspiré des maquettes
st.markdown("""
<style>
    /* --- Général --- */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* --- Barre latérale --- */
    .st-emotion-cache-16txtl3 {
        padding: 1rem 1rem;
    }
    /* --- Cartes KPI --- */
    .kpi-card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        border: 1px solid #E0E0E0;
        height: 120px; /* Hauteur fixe pour l'alignement */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .kpi-title { font-size: 14px; color: #555555; margin-bottom: 10px; }
    .kpi-value { font-size: 28px; font-weight: bold; color: #1E3A8A; }
    .kpi-desc { font-size: 12px; color: #888888; }

    /* --- Statuts dans le tableau --- */
    .status-critique { background-color: #FFCDD2; color: #C62828; padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold; }
    .status-alerte { background-color: #FFF9C4; color: #F57F17; padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold; }
    .status-normal { background-color: #C8E6C9; color: #2E7D32; padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold; }
    
    /* --- Panneau de détails --- */
    .detail-pane {
        background-color: #FFFFFF;
        border-left: 1px solid #E0E0E0;
        padding: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# 2. Chargement des données (Votre code original)
# ========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("df_weekly_clean.csv", parse_dates=["ds"])
    except FileNotFoundError:
        st.error("ERREUR : Le fichier 'df_weekly_clean.csv' est introuvable. Veuillez vous assurer qu'il est dans le même dossier que l'application.")
        st.stop()

    df["num_gab"] = df["num_gab"].astype(str)
    df["week_day"] = df["ds"].dt.dayofweek
    df["week"] = df["ds"].dt.isocalendar().week
    df["year"] = df["ds"].dt.year
    # Assurez-vous que la colonne 'y' existe pour les prédictions
    if 'y' not in df.columns and 'total_montant' in df.columns:
        df['y'] = df['total_montant']
    return df

df = load_data()

# ========================================
# 3. Chargement des modèles LSTM (Votre code original)
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
        except Exception:
            pass # Ignore les erreurs de chargement en silence
    return models, scalers

lstm_models, lstm_scalers = load_lstm_models()

# ========================================
# 4. Barre de navigation latérale
# ========================================
st.sidebar.title("CashGAB")
st.sidebar.markdown("---")
tab = st.sidebar.radio("Navigation", 
    ["📊 Tableau de bord", "📈 Prévisions", "⚙️ Planification"], 
    captions=["Vue d'ensemble du réseau", "Simulation et prévision", "Gestion des transferts"])

# Initialisation de l'état de session pour le GAB sélectionné
if 'selected_gab' not in st.session_state:
    st.session_state.selected_gab = None

# ========================================
# 5. Onglet : Tableau de bord (Entièrement dynamique et interactif)
# ========================================
if tab == "📊 Tableau de bord":
    st.title("Dashboard de Trésorerie")
    st.markdown("Accueil > Tableau de bord > **Vue d'ensemble**")

    # --- Préparation des données pour l'affichage ---
    df_latest = df.loc[df.groupby('num_gab')['ds'].idxmax()].copy()

    # --- Calculs dynamiques des KPIs ---
    cash_disponible_total = df_latest['total_montant'].sum()
    seuil_critique = 100000
    agences_a_risque = df_latest[df_latest['total_montant'] < seuil_critique].shape[0]
    transferts_a_prevoir = df_latest[df_latest['total_montant'] < seuil_critique]['total_montant'].sum()
    dispo_reseau = (df_latest.shape[0] - agences_a_risque) / df_latest.shape[0] * 100 if not df_latest.empty else 100

    st.info(f"⚠️ **La disponibilité réseau est de {dispo_reseau:.0f}%.** {agences_a_risque} GAB(s) présentent un risque critique.")

    # --- Affichage des KPIs dynamiques ---
    kpi_cols = st.columns(4)
    kpi_cols[0].markdown(f'<div class="kpi-card"><div class="kpi-title">💵 Cash Disponible</div><div class="kpi-value">{cash_disponible_total/1000000:.2f}M MAD</div><div class="kpi-desc">Total sur le réseau</div></div>', unsafe_allow_html=True)
    kpi_cols[1].markdown(f'<div class="kpi-card"><div class="kpi-title">🏢 GABs à Risque</div><div class="kpi-value">{agences_a_risque}</div><div class="kpi-desc">Cash sous le seuil critique</div></div>', unsafe_allow_html=True)
    kpi_cols[2].markdown(f'<div class="kpi-card"><div class="kpi-title">🚚 Transferts à Prévoir</div><div class="kpi-value">{transferts_a_prevoir/1000:.0f}K MAD</div><div class="kpi-desc">Montant total à couvrir</div></div>', unsafe_allow_html=True)
    kpi_cols[3].markdown(f'<div class="kpi-card"><div class="kpi-title">🌐 Disponibilité Réseau</div><div class="kpi-value">{dispo_reseau:.0f}%</div><div class="kpi-desc">GABs opérationnels</div></div>', unsafe_allow_html=True)
    
    st.markdown("  
", unsafe_allow_html=True)

    # --- Tableau principal et Panneau de détails ---
    main_col, detail_col = st.columns([0.6, 0.4])
    with main_col:
        st.subheader("État du réseau")

        def get_status_html(cash):
            if cash < seuil_critique: return '<span class="status-critique">Critique</span>'
            elif seuil_critique <= cash < 200000: return '<span class="status-alerte">Alerte</span>'
            else: return '<span class="status-normal">Normal</span>'

        df_latest['État'] = df_latest['total_montant'].apply(get_status_html)
        
        # Affichage du tableau ligne par ligne pour la cliquabilité
        header_cols = st.columns((1, 2, 2, 2, 1))
        header_cols[0].markdown("**ID GAB**")
        header_cols[1].markdown("**Agence**")
        header_cols[2].markdown("**Cash Disponible**")
        header_cols[3].markdown("**Région**")
        header_cols[4].markdown("**État**")

        for _, row in df_latest.iterrows():
            row_cols = st.columns((1, 2, 2, 2, 1))
            row_cols[0].write(row['num_gab'])
            row_cols[1].write(row['agence'])
            row_cols[2].write(f"{row['total_montant']/1000:,.0f} K MAD")
            row_cols[3].write(row['region'])
            row_cols[4].markdown(row['État'], unsafe_allow_html=True)
            # Le bouton invisible couvre la ligne pour la rendre cliquable
            if row_cols[0].button(" ", key=f"btn_{row['num_gab']}"):
                st.session_state.selected_gab = row['num_gab']
                st.rerun()

    with detail_col:
        st.markdown('<div class="detail-pane">', unsafe_allow_html=True)
        if st.session_state.selected_gab is None:
            st.info("Cliquez sur une ligne du tableau pour afficher les détails.")
        else:
            gab_id = st.session_state.selected_gab
            gab_data = df_latest[df_latest['num_gab'] == gab_id].iloc[0]
            
            st.subheader(f"Détails : {gab_data['agence']}")
            st.markdown(f"**ID GAB :** {gab_id} | **Région :** {gab_data['region']}")
            
            st.metric("Cash disponible actuel", f"{gab_data['total_montant']/1000:,.0f} K MAD")
            
            st.markdown("---")
            st.markdown("**Évolution du Cash Disponible**")
            
            history_df = df[df['num_gab'] == gab_id].sort_values('ds')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history_df['ds'], y=history_df['total_montant'], mode='lines', fill='tozeroy'))
            fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0), yaxis_title="Montant (MAD)")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Historique des Transferts (Exemple)**")
            # Données fictives pour l'historique
            transfer_data = {
                'Date': pd.to_datetime(['2025-05-20', '2025-05-12']),
                'Montant (MAD)': [150000, 200000],
                'Type': ['Entrant', 'Entrant']
            }
            st.dataframe(pd.DataFrame(transfer_data), use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ========================================
# 6. Onglet : Prévisions (Votre code original)
# ========================================
elif tab == "📈 Prévisions":
    st.title("Prévisions et Simulation LSTM")
    st.sidebar.header("Paramètres de simulation")
    
    gab_options = [gab for gab in sorted(df["num_gab"].unique()) if gab in lstm_models]
    if not gab_options:
        st.warning("Aucun modèle LSTM n'a été chargé. Vérifiez la présence des fichiers .h5 et .save.")
    else:
        gab_selected = st.sidebar.selectbox("Sélectionner un GAB", gab_options)
        period_forecast = st.sidebar.selectbox("Période de prévision (semaines)", [1, 2, 4, 6])
        variation = st.sidebar.slider("Facteur de variation (%)", -50, 50, 0)
        
        df_gab = df[df["num_gab"] == gab_selected].sort_values("ds")

        if len(df_gab) < 52:
            st.warning("Pas assez de données pour ce GAB (minimum 52 semaines requises).")
        else:
            st.subheader(f"Prévisions pour le GAB {gab_selected}")
            try:
                n_steps = 4
                scaler = lstm_scalers[gab_selected]
                model = lstm_models[gab_selected]

                y_scaled = scaler.transform(df_gab[['y']].values)
                
                # Prévisions futures
                last_sequence = y_scaled[-n_steps:].reshape(1, n_steps, 1)
                future_preds_adjusted = []
                future_dates = [df_gab["ds"].max() + pd.Timedelta(weeks=i+1) for i in range(period_forecast)]

                current_sequence = last_sequence
                for _ in range(period_forecast):
                    pred_scaled = model.predict(current_sequence, verbose=0)
                    pred_adjusted = scaler.inverse_transform(pred_scaled)[0, 0] * (1 + variation/100)
                    future_preds_adjusted.append(pred_adjusted)
                    current_sequence = np.append(current_sequence[:, 1:, :], pred_scaled.reshape(1, 1, 1), axis=1)

                # Graphique
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=df_gab['ds'], y=df_gab['y'], mode="lines", name="Montant réel"))
                fig_pred.add_trace(go.Scatter(x=future_dates, y=future_preds_adjusted, mode="lines+markers", name=f"Prévisions ajustées ({variation}%)", line=dict(color='red', dash='dash')))
                fig_pred.update_layout(title=f"Prévision LSTM pour GAB {gab_selected}", xaxis_title="Date", yaxis_title="Montant retiré (MAD)")
                st.plotly_chart(fig_pred, use_container_width=True)

            except Exception as e:
                st.error(f"Erreur lors de la génération des prévisions : {e}")

# ========================================
# 7. Onglet : Planification (Placeholder)
# ========================================
elif tab == "⚙️ Planification":
    st.title("Planification des Transferts")
    st.info("Cette section est en cours de construction. Elle contiendra le calendrier et la gestion des plans de transfert.")
    # Vous pouvez utiliser une image de la maquette comme aperçu
    st.image("https://i.imgur.com/3g6gL01.png", caption="Aperçu de la vue de planification." )

