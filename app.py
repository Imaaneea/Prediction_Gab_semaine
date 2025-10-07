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

st.markdown(
    """
    <style>
    .main .block-container { padding: 1.2rem 2rem 2rem 2rem; }
    .kpi-card { background: #ffffff; border-radius: 10px; padding: 14px; box-shadow: 0 6px 18px rgba(30,58,138,0.06); border: 1px solid rgba(30,58,138,0.08); height: 110px; }
    .kpi-title { color: #566270; font-size: 13px; margin-bottom: 6px; }
    .kpi-value { color: #0b5394; font-size: 26px; font-weight: 700; }
    .kpi-sub { color: #8b99a6; font-size: 12px; }
    .badge-crit { background:#fdecea; color:#d32f2f; padding:6px 10px; border-radius:12px; font-weight:700; }
    .badge-alert { background:#fff8e1; color:#f9a825; padding:6px 10px; border-radius:12px; font-weight:700; }
    .badge-norm { background:#e8f5e9; color:#2e7d32; padding:6px 10px; border-radius:12px; font-weight:700; }
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
    try:
        df = pd.read_csv(
            "df_weekly_clean.csv",
            encoding="utf-8-sig",
            sep=",",
            on_bad_lines="skip"
        )
        df.columns = df.columns.str.strip()
    except Exception as e:
        st.error(f"Erreur lors de la lecture du CSV : {e}")
        return pd.DataFrame()

    if df.empty:
        st.error("Le fichier CSV est vide.")
        return pd.DataFrame()

    if "ds" in df.columns: 
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    else:
        st.error("La colonne 'ds' est absente du CSV.")
        return pd.DataFrame()

    if "num_gab" in df.columns:
        df["num_gab"] = df["num_gab"].astype(str)

    df["week_day"] = df["ds"].dt.dayofweek
    df["week"] = df["ds"].dt.isocalendar().week
    df["year"] = df["ds"].dt.year

    if "y" not in df.columns and "total_montant" in df.columns:
        df["y"] = df["total_montant"]

    return df

df = load_data()
if df.empty:
    st.stop()

# ========================================
# Load LSTM models + scalers
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
            continue
    return models, scalers

lstm_models, lstm_scalers = load_lstm_models()

# ========================================
# Sidebar & Navigation
# ========================================
st.sidebar.image("https://www.albaridbank.ma/themes/baridbank/logo.png", width=250)
st.sidebar.title("CashGAB")
st.sidebar.markdown("Solution de gestion proactive des GABs")
tab = st.sidebar.radio("Navigation", ["Tableau de bord analytique", "Prévisions LSTM 20 GAB"])

# Global filters
st.sidebar.markdown("---")
regions = df["region"].dropna().unique() if "region" in df.columns else []
region_filter = st.sidebar.selectbox("Région", ["Toutes"] + sorted(regions.tolist()) if len(regions)>0 else ["Toutes"])
df_region = df[df["region"] == region_filter] if region_filter != "Toutes" else df.copy()
agences = df_region["agence"].dropna().unique() if "agence" in df_region.columns else []
agence_filter = st.sidebar.selectbox("Agence", ["Toutes"] + sorted(agences.tolist()) if len(agences)>0 else ["Toutes"])

# Date filter
date_min = df["ds"].min()
date_max = df["ds"].max()
st.sidebar.markdown("Période (TDB)")
date_debut = st.sidebar.date_input("Date début", date_min)
date_fin = st.sidebar.date_input("Date fin", date_max)

df_filtered = df.copy()
if region_filter != "Toutes":
    df_filtered = df_filtered[df_filtered["region"] == region_filter]
if agence_filter != "Toutes":
    df_filtered = df_filtered[df_filtered["agence"] == agence_filter]
df_filtered = df_filtered[(df_filtered["ds"] >= pd.to_datetime(date_debut)) & (df_filtered["ds"] <= pd.to_datetime(date_fin))]

# ========================================
# Tableau de bord analytique
# ========================================
if tab == "Tableau de bord analytique":
    st.title("CashGAB — Tableau de bord analytique")
    st.markdown("Vue d’ensemble du réseau et KPIs interactifs")

    montant_total = df_filtered["total_montant"].sum() if "total_montant" in df_filtered.columns else 0
    nombre_operations = df_filtered["total_nombre"].sum() if "total_nombre" in df_filtered.columns else 0
    nb_gabs = df_filtered["num_gab"].nunique() if "num_gab" in df_filtered.columns else 0

    # --- Seuil critique dynamique ---
    df_avg_gab = df_filtered.groupby("num_gab")["total_montant"].mean().to_dict()
    user_seuil = st.sidebar.number_input(
        "Seuil critique personnalisé (MAD, facultatif)", 
        value=0, step=10000,
        help="Si >0, ce seuil remplacera la moyenne historique pour tous les GAB"
    )

    def get_seuil(gab_id):
        return user_seuil if user_seuil > 0 else df_avg_gab.get(gab_id, 100_000)

    df_latest = df_filtered.loc[df_filtered.groupby('num_gab')['ds'].idxmax()].copy()
    df_latest["seuil_critique"] = df_latest["num_gab"].apply(get_seuil)

    def classify_gab(row):
        s = row["seuil_critique"]
        if row["total_montant"] < s:
            return "Critique"
        elif row["total_montant"] < 2*s:
            return "Alerte"
        else:
            return "Normal"

    df_latest["status"] = df_latest.apply(classify_gab, axis=1)

    # --- KPI ---
    nb_critique = df_latest[df_latest["status"]=="Critique"]["num_gab"].nunique()
    nb_alerte = df_latest[df_latest["status"]=="Alerte"]["num_gab"].nunique()
    dispo_proxy = (df_latest.shape[0] - nb_critique) / df_latest.shape[0] * 100 if not df_latest.empty else 100.0

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

    # --- Alertes récentes (bar chart par région) ---
    st.markdown("### Alertes récentes (dernier état des GABs)")
    if not df_latest.empty:
        alert_counts_region = df_latest.groupby(["region","status"])["num_gab"].count().reset_index()
        fig_alert = go.Figure()
        for status, color in zip(["Critique","Alerte","Normal"], ['#d32f2f','#f9a825','#2e7d32']):
            df_s = alert_counts_region[alert_counts_region["status"]==status]
            fig_alert.add_trace(go.Bar(
                x=df_s["region"], y=df_s["num_gab"],
                name=status, marker_color=color
            ))
        fig_alert.update_layout(barmode='stack', title="Répartition des GAB par statut et région",
                                xaxis_title="Région", yaxis_title="Nombre de GAB")
        st.plotly_chart(fig_alert, use_container_width=True)
    else:
        st.info("Aucune alerte disponible pour la période sélectionnée.")

    # --- Evolution des retraits ---
    st.markdown("### Evolution des retraits (analyse détaillée)")
    if not df_filtered.empty:
        df_evol = df_filtered.groupby("ds")["total_montant"].sum().reset_index()
        fig_evol = go.Figure()
        fig_evol.add_trace(go.Scatter(
            x=df_evol["ds"], y=df_evol["total_montant"]/1000,
            mode="lines+markers",
            name="Total retraits hebdomadaire"
        ))
        fig_evol.update_layout(
            title="Evolution hebdomadaire des retraits (K MAD)",
            xaxis_title="Date",
            yaxis_title="Montant retiré (K MAD)"
        )
        st.plotly_chart(fig_evol, use_container_width=True)
    else:
        st.info("Pas de données pour l'évolution des retraits sur la période sélectionnée.")

    # --- Fiches réseau ---
    st.markdown("### Fiches réseau (aperçu des GABs)")
    if not df_latest.empty:
        def status_label(val):
            if val=="Critique":
                return "badge-crit"
            elif val=="Alerte":
                return "badge-alert"
            else:
                return "badge-norm"

        cols_to_show = ["num_gab"]
        if "agence" in df_latest.columns:
            cols_to_show.append("agence")
        if "region" in df_latest.columns:
            cols_to_show.append("region")
        cols_to_show.append("total_montant")

        display_df = df_latest[cols_to_show].copy().reset_index(drop=True)
        display_df["status_html"] = df_latest["status"].apply(lambda x: f'<span class="{status_label(x)}">{x}</span>')

        st.write("Cliquez sur une ligne pour plus de détails (sélection simulée).")
        for _, row in display_df.iterrows():
            n_cols = 2
            if "agence" in row.index: n_cols+=1
            if "region" in row.index: n_cols+=1
            n_cols+=1
            cols = st.columns(n_cols)
            col_idx = 0
            cols[col_idx].write(row["num_gab"]); col_idx+=1
            if "agence" in row.index:
                cols[col_idx].write(row.get("agence","-")); col_idx+=1
            if "region" in row.index:
                cols[col_idx].write(row.get("region","-")); col_idx+=1
            cols[col_idx].write(f"{row['total_montant']/1000:,.0f} K MAD"); col_idx+=1
            cols[col_idx].markdown(row["status_html"], unsafe_allow_html=True)
    else:
        st.info("Aucune fiche réseau disponible pour la sélection.")

    # --- Répartition régionale (bar chart) ---
    st.markdown("### Répartition régionale & évolution")
    if not df_filtered.empty:
        df_region_avg = df_filtered.groupby("region")["total_montant"].mean().reset_index().sort_values("total_montant", ascending=False)
        fig_region = go.Figure(go.Bar(
            x=df_region_avg["region"],
            y=df_region_avg["total_montant"]/1000,
            text=(df_region_avg["total_montant"]/1000).round(0),
            textposition="auto",
            marker_color='lightskyblue'
        ))
        fig_region.update_layout(title="Montant moyen hebdo par région (K MAD)", xaxis_title="Région", yaxis_title="Montant moyen")
        st.plotly_chart(fig_region, use_container_width=True)
    else:
        st.info("Aucune donnée pour la période / filtres sélectionnés.")

# ========================================
# Prévisions LSTM
# ========================================
if tab == "Prévisions LSTM 20 GAB":
    st.title("Prévisions LSTM - 20 GAB")
    st.sidebar.header("Paramètres de simulation")
    gab_options = [gab for gab in sorted(df["num_gab"].unique()) if gab in lstm_models]

    if not gab_options:
        st.warning("Aucun GAB disponible avec modèles LSTM.")
    else:
        gab_selected = st.sidebar.selectbox("Sélectionner un GAB", gab_options)
        period_forecast = st.sidebar.selectbox("Période de prévision", [1,2,4,6])
        variation = st.sidebar.slider("Facteur de variation (%)", -50,50,0)

        df_gab = df[df["num_gab"] == gab_selected].sort_values("ds")
        if len(df_gab) < 52:
            st.warning("Pas assez de données pour la prévision LSTM (min 52 semaines).")
        else:
            st.subheader(f"Prévisions pour GAB {gab_selected}")

            try:
                n_steps = 4
                scaler = lstm_scalers[gab_selected]
                model = lstm_models[gab_selected]

                y_scaled = scaler.transform(df_gab[['y']].values)
                X = []
                for i in range(len(y_scaled) - n_steps):
                    X.append(y_scaled[i:i+n_steps])
                X = np.array(X).reshape(-1, n_steps, 1)

                y_pred_scaled = model.predict(X, verbose=0)
                y_pred = scaler.inverse_transform(y_pred_scaled)
                y_true = df_gab['y'].values[n_steps:]
                dates = df_gab['ds'][n_steps:]

                last_sequence = y_scaled[-n_steps:].reshape(1, n_steps, 1)
                future_preds = []
                future_dates = [df_gab["ds"].max() + pd.Timedelta(weeks=i+1) for i in range(period_forecast)]

                for _ in range(period_forecast):
                    pred_scaled = model.predict(last_sequence, verbose=0)
                    pred = scaler.inverse_transform(pred_scaled)[0,0]
                    pred_adjusted = pred * (1 + variation/100)
                    future_preds.append(pred_adjusted/1000)
                    last_sequence = np.concatenate([last_sequence[:,1:,:], pred_scaled.reshape(1,1,1)], axis=1)

                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=dates, y=y_true/1000, mode="lines+markers", name="Montant réel"))
                fig_pred.add_trace(go.Scatter(x=dates, y=y_pred.flatten()/1000, mode="lines+markers", name="Prédiction LSTM"))
                fig_pred.add_trace(go.Scatter(x=future_dates, y=future_preds, mode="lines+markers", name=f"Prévisions ajustées ({variation}%)"))
                fig_pred.update_layout(title=f"Prévision LSTM GAB {gab_selected}", xaxis_title="Date", yaxis_title="Montant retiré (K MAD)")
                st.plotly_chart(fig_pred, use_container_width=True)

                df_csv = pd.DataFrame({
                    "ds": list(dates)+future_dates,
                    "y_true_kdh": list(y_true/1000)+[None]*period_forecast,
                    "y_pred_kdh": list(y_pred.flatten()/1000)+future_preds
                })
                st.download_button("Télécharger CSV", df_csv.to_csv(index=False), file_name=f"pred_{gab_selected}.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Erreur lors des prévisions: {e}")
