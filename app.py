import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import glob
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# ========================================
# Configuration page & style
# ========================================
st.set_page_config(page_title="CashGAB : Dashboard GAB", layout="wide")

# CSS pour un design moderne (clair / bleu)
st.markdown(
    """
    <style>
    /* Container */
    .main .block-container { padding: 1.2rem 2rem 2rem 2rem; }

    /* KPI cards */
    .kpi-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 14px;
        box-shadow: 0 6px 18px rgba(30,58,138,0.06);
        border: 1px solid rgba(30,58,138,0.08);
        height: 110px;
    }
    .kpi-title { color: #566270; font-size: 13px; margin-bottom: 6px; }
    .kpi-value { color: #0b5394; font-size: 26px; font-weight: 700; }
    .kpi-sub { color: #8b99a6; font-size: 12px; }

    /* status badges */
    .badge-crit { background:#fdecea; color:#d32f2f; padding:6px 10px; border-radius:12px; font-weight:700; }
    .badge-alert { background:#fff8e1; color:#f9a825; padding:6px 10px; border-radius:12px; font-weight:700; }
    .badge-norm { background:#e8f5e9; color:#2e7d32; padding:6px 10px; border-radius:12px; font-weight:700; }

    /* small helpers */
    .muted { color:#7f8b95; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========================================
# Load data
# ========================================
@st.cache_data
def load_data():
    df = pd.read_csv("df_weekly_clean.csv", parse_dates=["ds"])
    # Force types & fallback
    if "num_gab" in df.columns:
        df["num_gab"] = df["num_gab"].astype(str)
    df["week_day"] = df["ds"].dt.dayofweek
    df["week"] = df["ds"].dt.isocalendar().week
    df["year"] = df["ds"].dt.year
    # Ensure 'y' exists (used by models/training)
    if "y" not in df.columns and "total_montant" in df.columns:
        df["y"] = df["total_montant"]
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Erreur lors du chargement des données : {e}")
    st.stop()

# ========================================
# Load LSTM models + scalers (silent)
# ========================================
@st.cache_data
def load_lstm_models():
    models, scalers = {}, {}
    for model_file in glob.glob("lstm_gab_*.h5"):
        gab_id = model_file.split("_")[-1].replace(".h5","")
        scaler_file = f"scaler_gab_{gab_id}.save"
        try:
            models[gab_id] = load_model(model_file, compile=False)
            scalers[gab_id] = joblib.load(scaler_file)
        except Exception:
            # ignore errors on loading (warn later)
            continue
    return models, scalers

lstm_models, lstm_scalers = load_lstm_models()

# ========================================
# Sidebar & Navigation
# ========================================
st.sidebar.title("CashGAB")
st.sidebar.markdown("Solution de gestion proactive du cash")
tab = st.sidebar.radio("Navigation", ["Tableau de bord analytique", "Prévisions LSTM 20 GAB"])

# Global filters (applied on tab1)
st.sidebar.markdown("---")
st.sidebar.header("Filtres globaux (TDB)")
regions = df["region"].dropna().unique() if "region" in df.columns else []
region_filter = st.sidebar.selectbox("Région", ["Toutes"] + sorted(regions.tolist()) if len(regions)>0 else ["Toutes"])
if region_filter != "Toutes":
    df_region = df[df["region"] == region_filter]
else:
    df_region = df.copy()

agences = df_region["agence"].dropna().unique() if "agence" in df_region.columns else []
agence_filter = st.sidebar.selectbox("Agence", ["Toutes"] + sorted(agences.tolist()) if len(agences)>0 else ["Toutes"])

# Date filter boundaries
date_min = df["ds"].min()
date_max = df["ds"].max()
st.sidebar.markdown("Période (TDB)")
date_debut = st.sidebar.date_input("Date début", date_min)
date_fin = st.sidebar.date_input("Date fin", date_max)

# Apply filters to a working df for TDB
df_filtered = df.copy()
if region_filter != "Toutes":
    df_filtered = df_filtered[df_filtered["region"] == region_filter]
if agence_filter != "Toutes":
    df_filtered = df_filtered[df_filtered["agence"] == agence_filter]
df_filtered = df_filtered[(df_filtered["ds"] >= pd.to_datetime(date_debut)) & (df_filtered["ds"] <= pd.to_datetime(date_fin))]

# ========================================
# Tab 1 : Tableau de bord analytique (amélioré)
# ========================================
if tab == "Tableau de bord analytique":
    st.title("Tableau de bord analytique — CashGAB")
    st.markdown("Vue d’ensemble du réseau et KPIs interactifs")

    # KPI calculations linked to df_filtered (respect filters)
    # Montant total retraits (somme)
    montant_total = df_filtered["total_montant"].sum() if "total_montant" in df_filtered.columns else 0
    # Nombre total opérations (sum of total_nombre)
    nombre_operations = df_filtered["total_nombre"].sum() if "total_nombre" in df_filtered.columns else 0
    # Nombre GAB réseau (unique num_gab)
    nb_gabs = df_filtered["num_gab"].nunique() if "num_gab" in df_filtered.columns else 0
    # Disponibilité proxy: percent of GAB with montants above a chosen threshold
    # Allow user to set critical threshold
    st.sidebar.markdown("---")
    seuil_critique = st.sidebar.number_input("Seuil critique (MAD)", value=100000, step=10000)
    # Define statuses by latest available week per GAB
    df_latest = df_filtered.loc[df_filtered.groupby('num_gab')['ds'].idxmax()].copy() if ("num_gab" in df_filtered.columns and not df_filtered.empty) else pd.DataFrame()
    nb_critique = 0
    nb_alerte = 0
    dispo_proxy = 100.0
    if not df_latest.empty:
        nb_critique = df_latest[df_latest["total_montant"] < seuil_critique]["num_gab"].nunique()
        nb_alerte = df_latest[(df_latest["total_montant"] >= seuil_critique) & (df_latest["total_montant"] < 2*seuil_critique)]["num_gab"].nunique()
        dispo_proxy = (df_latest.shape[0] - nb_critique) / df_latest.shape[0] * 100

    # KPI display (styled)
    k1, k2, k3, k4 = st.columns([2,2,2,2])
    k1.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Montant total retraits</div>
            <div class="kpi-value">{montant_total/1_000_000:,.2f} M MAD</div>
            <div class="kpi-sub">Période filtrée</div>
        </div>
    """, unsafe_allow_html=True)
    k2.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Nombre total opérations</div>
            <div class="kpi-value">{nombre_operations:,.0f}</div>
            <div class="kpi-sub">Période filtrée</div>
        </div>
    """, unsafe_allow_html=True)
    k3.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Nombre GAB (réseau)</div>
            <div class="kpi-value">{nb_gabs}</div>
            <div class="kpi-sub">Filtré</div>
        </div>
    """, unsafe_allow_html=True)
    k4.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Disponibilité (proxy)</div>
            <div class="kpi-value">{dispo_proxy:.0f}%</div>
            <div class="kpi-sub">GAB au-dessus du seuil</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### Répartition régionale & évolution")
    # Pie per region (average weekly sum)
    if not df_filtered.empty:
        df_region_pie = df_filtered.groupby("region")["total_montant"].mean().reset_index().sort_values("total_montant", ascending=False)
        if not df_region_pie.empty:
            fig_pie = go.Figure(go.Pie(labels=df_region_pie["region"], values=(df_region_pie["total_montant"]/1000).round(2)))
            fig_pie.update_layout(margin=dict(l=0,r=0,t=30,b=0), title="Montant moyen hebdo par région (K MAD)")
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Aucune donnée pour la période / filtres sélectionnés.")

    # Evolution timeseries for selected level
    st.markdown("#### Évolution des retraits")
    level_options = ["Global"]
    if "region" in df_filtered.columns:
        level_options += sorted(df_filtered["region"].dropna().unique().tolist())
    if "num_gab" in df_filtered.columns:
        level_options += sorted(df_filtered["num_gab"].dropna().unique().tolist())

    selected_level = st.selectbox("Niveau", level_options, index=0)
    if selected_level == "Global":
        df_plot = df_filtered.groupby("ds")["total_montant"].sum().reset_index()
    elif selected_level in df_filtered["region"].unique():
        df_plot = df_filtered[df_filtered["region"] == selected_level].groupby("ds")["total_montant"].sum().reset_index()
    else:
        df_plot = df_filtered[df_filtered["num_gab"] == selected_level].groupby("ds")["total_montant"].sum().reset_index()

    if not df_plot.empty:
        df_plot["total_montant_kdh"] = df_plot["total_montant"]/1000
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=df_plot["ds"], y=df_plot["total_montant_kdh"], mode="lines+markers", name="Montant retiré (K MAD)"))
        # 7-point moving average
        if len(df_plot) >= 3:
            df_plot["ma7"] = df_plot["total_montant_kdh"].rolling(3, min_periods=1).mean()
            fig_line.add_trace(go.Scatter(x=df_plot["ds"], y=df_plot["ma7"], mode="lines", name="MA (3)", line=dict(dash="dash")))
        fig_line.update_layout(height=420, xaxis_title="Date", yaxis_title="Montant (K MAD)")
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Pas de séries temporelles pour cette sélection.")

    # Table of GABs with status (critical / alert / normal) using df_latest
    st.markdown("### Fiches réseau (aperçu des GABs)")
    if not df_latest.empty:
        def status_label(val):
            if val < seuil_critique:
                return ("Critique", "badge-crit")
            if val < 2*seuil_critique:
                return ("Alerte", "badge-alert")
            return ("Normal", "badge-norm")

        # Display simple table: id, agence (if exists), region (if exists), cash, status
        cols_to_show = ["num_gab"]
        if "agence" in df_latest.columns: cols_to_show.append("agence")
        if "region" in df_latest.columns: cols_to_show.append("region")
        cols_to_show += ["total_montant"]

        # Build display
        st.write("Cliquez sur une ligne pour plus de détails (sélection simulée).")
        # We'll show a lightweight dataframe and a status badge column
        display_df = df_latest[cols_to_show].copy()
        display_df = display_df.reset_index(drop=True)
        display_df["status"] = display_df["total_montant"].apply(lambda x: status_label(x)[0])
        display_df["status_html"] = display_df["total_montant"].apply(lambda x: f'<span class="{status_label(x)[1]}">{status_label(x)[0]}</span>')
        # Render as table with unsafe HTML for status
        # Use .to_html for formatting but keep simple
        for _, row in display_df.iterrows():
            cols = st.columns([1, 2, 1, 1, 1])
            cols[0].write(row["num_gab"])
            if "agence" in row.index:
                cols[1].write(row.get("agence","-"))
            if "region" in row.index:
                cols[2].write(row.get("region","-"))
            cols[3].write(f"{row['total_montant']/1000:,.0f} K MAD")
            cols[4].markdown(row["status_html"], unsafe_allow_html=True)

    else:
        st.info("Aucune fiche réseau disponible pour la sélection.")

# ========================================
# Tab 2 : Prévisions LSTM (inchangé fonctionnellement, amélioré UI)
# ========================================
if tab == "Prévisions LSTM 20 GAB":
    st.title("Prévisions LSTM - CashGAB")
    st.markdown("Simulation et prévisions LSTM par GAB (ajustable)")

    # Params in sidebar (for this tab only)
    st.sidebar.markdown("---")
    st.sidebar.header("Paramètres prédiction")
    gab_options = [gab for gab in sorted(df["num_gab"].unique()) if gab in lstm_models]
    if not gab_options:
        st.sidebar.info("Aucun modèle LSTM chargé.")
        st.warning("Aucun GAB disponible avec modèles LSTM. Vérifiez les fichiers .h5 et .save.")
    else:
        gab_selected = st.sidebar.selectbox("Sélectionner un GAB", gab_options)
        period_forecast = st.sidebar.selectbox("Période (semaines)", [1,2,4,6], index=2)
        variation = st.sidebar.slider("Facteur de variation (%)", -50, 50, 0)

        df_gab = df[df["num_gab"] == gab_selected].sort_values("ds")
        if len(df_gab) < 52:
            st.warning("Pas assez de données pour effectuer une prévision LSTM (minimum 52 semaines).")
        else:
            st.subheader(f"Prévisions pour GAB {gab_selected}")

            try:
                # Preparation same as original
                n_steps = 4
                scaler = lstm_scalers[gab_selected]
                model = lstm_models[gab_selected]

                y_scaled = scaler.transform(df_gab[['y']].values)
                X = []
                for i in range(len(y_scaled) - n_steps):
                    X.append(y_scaled[i:i+n_steps])
                X = np.array(X).reshape(-1, n_steps, 1)

                # Predictions on history
                y_pred_scaled = model.predict(X, verbose=0)
                y_pred = scaler.inverse_transform(y_pred_scaled)

                y_true = df_gab['y'].values[n_steps:]
                dates = df_gab['ds'][n_steps:]

                # Future forecasts adjusted by variation
                last_sequence = y_scaled[-n_steps:].reshape(1, n_steps, 1)
                future_preds = []
                future_dates = [df_gab["ds"].max() + pd.Timedelta(weeks=i+1) for i in range(period_forecast)]

                for _ in range(period_forecast):
                    pred_scaled = model.predict(last_sequence, verbose=0)
                    pred = scaler.inverse_transform(pred_scaled)[0, 0]
                    pred_adjusted = pred * (1 + variation/100)
                    future_preds.append(pred_adjusted/1000)  # K in MAD
                    last_sequence = np.concatenate([last_sequence[:,1:,:], pred_scaled.reshape(1,1,1)], axis=1)

                # Plot: actual, historical predictions, future adjusted
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=dates, y=y_true/1000, mode="lines+markers", name="Montant réel (K MAD)"))
                fig_pred.add_trace(go.Scatter(x=dates, y=y_pred.flatten()/1000, mode="lines+markers", name="Prédiction historique (K MAD)"))
                fig_pred.add_trace(go.Scatter(x=future_dates, y=future_preds, mode="lines+markers", name=f"Prévisions futures ajustées ({variation}%)"))

                fig_pred.update_layout(title=f"Prévision LSTM GAB {gab_selected}", xaxis_title="Date", yaxis_title="Montant (K MAD)",
                                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_pred, use_container_width=True)

                # CSV download
                df_csv = pd.DataFrame({
                    "ds": list(dates) + future_dates,
                    "y_true_kdh": list(y_true/1000) + [None]*period_forecast,
                    "y_pred_kdh": list(y_pred.flatten()/1000) + future_preds
                })
                st.download_button("Télécharger prévisions CSV", df_csv.to_csv(index=False), file_name=f"pred_{gab_selected}.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Erreur lors de la génération des prévisions: {e}")
