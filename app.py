# app.py — Dashboard CashGAB (complet : KPIs, Graphes, Alertes, Prévisions LSTM)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import glob
import os

# optional tensorflow imports guarded
try:
    from tensorflow.keras.models import load_model
    import joblib
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# -----------------------
# Page config + CSS
# -----------------------
st.set_page_config(page_title="CashGAB - Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');
        body { font-family: 'Poppins', sans-serif; background: #f7fbff; }
        .kpi-card { background: linear-gradient(180deg,#ffffff 0%,#f7fbff 100%); border-radius:12px; padding:14px; box-shadow:0 6px 18px rgba(30,58,138,0.06); border:1px solid rgba(30,58,138,0.08); }
        .kpi-title { color:#556677; font-size:13px; margin-bottom:6px; }
        .kpi-value { color:#0b5394; font-size:26px; font-weight:700; }
        .kpi-sub { color:#8899a6; font-size:12px; }
        .badge-crit { background:#d32f2f; color:white; padding:6px 10px; border-radius:12px; font-weight:700; }
        .badge-alert { background:#f9a825; color:black; padding:6px 10px; border-radius:12px; font-weight:700; }
        .badge-norm { background:#2e7d32; color:white; padding:6px 10px; border-radius:12px; font-weight:700; }
        .section-title { font-size:18px; color:#102a43; margin:8px 0 12px 0; font-weight:600; }
        .small-muted { color:#7f8b95; font-size:13px; }
        /* card animation + hover */
        .card { transition: transform .18s ease, box-shadow .18s ease; }
        .card:hover { transform: translateY(-6px); box-shadow: 0 10px 30px rgba(0,0,0,0.08); }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Load data (support multiple file layouts)
# -----------------------
@st.cache_data
def load_data():
    # Prefer named master file if exists
    if os.path.exists("df_weekly_clean.csv"):
        try:
            df = pd.read_csv("df_weekly_clean.csv", encoding="utf-8-sig", sep=",", on_bad_lines="skip")
        except Exception:
            df = pd.read_csv("df_weekly_clean.csv", on_bad_lines="skip")
    else:
        files = glob.glob("data/*.csv")
        if not files:
            # fallback to any csv in root
            files = glob.glob("*.csv")
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_csv(f, on_bad_lines="skip") for f in files], ignore_index=True)

    # normalize column names (strip)
    df.columns = df.columns.str.strip()

    # find date column -> unify to 'ds'
    if "ds" not in df.columns:
        if "date" in df.columns:
            df["ds"] = pd.to_datetime(df["date"], errors="coerce")
        elif "Date" in df.columns:
            df["ds"] = pd.to_datetime(df["Date"], errors="coerce")
        else:
            # try common name
            for c in df.columns:
                if "date" in c.lower():
                    df["ds"] = pd.to_datetime(df[c], errors="coerce")
                    break

    # unify GAB id column to 'num_gab'
    if "num_gab" not in df.columns:
        if "GAB" in df.columns:
            df["num_gab"] = df["GAB"].astype(str)
        elif "gab" in df.columns:
            df["num_gab"] = df["gab"].astype(str)
        else:
            # try to detect numeric code column
            for c in df.columns:
                if c.lower().startswith("gab") or c.lower().startswith("num"):
                    df["num_gab"] = df[c].astype(str)
                    break

    # find total amount column
    amount_cols = ["total_montant", "Montant_retraits", "Montant", "total_amount", "montant"]
    found_amt = None
    for c in amount_cols:
        if c in df.columns:
            found_amt = c
            break
    if found_amt is None:
        # fallback: pick numeric column with large values
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            # choose column with largest mean
            means = {c: df[c].abs().mean() for c in numeric_cols}
            found_amt = max(means, key=means.get)
    # assign standardized column
    if found_amt is not None:
        df["total_montant"] = df[found_amt].astype(float)
    else:
        # can't continue without amount
        return pd.DataFrame()

    # unify operations count
    if "total_nombre" not in df.columns:
        if "nb_operations" in df.columns:
            df["total_nombre"] = df["nb_operations"]
        elif "total_transactions" in df.columns:
            df["total_nombre"] = df["total_transactions"]
        else:
            # set default
            df["total_nombre"] = 0

    # ensure ds exists
    if "ds" not in df.columns or df["ds"].isna().all():
        return pd.DataFrame()

    # derived columns
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"])
    df["week"] = df["ds"].dt.isocalendar().week
    df["year"] = df["ds"].dt.year
    if "y" not in df.columns:
        df["y"] = df["total_montant"]

    # ensure region/agence columns exist (create if missing)
    if "region" not in df.columns:
        df["region"] = "Inconnue"
    if "agence" not in df.columns:
        df["agence"] = "Inconnue"

    # convert num_gab to string
    if "num_gab" in df.columns:
        df["num_gab"] = df["num_gab"].astype(str)
    else:
        df["num_gab"] = df.index.astype(str)

    return df

df = load_data()
if df.empty:
    st.error("Aucune donnée trouvée ou format du CSV non supporté. Vérifie les fichiers CSV et les noms de colonnes (date, total_montant, num_gab).")
    st.stop()

# -----------------------
# Load LSTM models + scalers (if TF available)
# -----------------------
@st.cache_data
def load_lstm_models(pattern="lstm_gab_*.h5"):
    models, scalers = {}, {}
    if not TF_AVAILABLE:
        return models, scalers
    for model_file in glob.glob(pattern):
        # extract gab id suffix (assumes pattern lstm_gab_<id>.h5)
        basename = os.path.basename(model_file)
        parts = basename.split("_")
        gab_id = parts[-1].replace(".h5", "")
        scaler_file = f"scaler_gab_{gab_id}.save"
        try:
            models[gab_id] = load_model(model_file, compile=False)
            if os.path.exists(scaler_file):
                scalers[gab_id] = joblib.load(scaler_file)
            else:
                scalers[gab_id] = None
        except Exception:
            # skip model if error
            continue
    return models, scalers

lstm_models, lstm_scalers = load_lstm_models()

# -----------------------
# Sidebar / Navigation / Filters
# -----------------------
st.sidebar.image("https://www.albaridbank.ma/themes/baridbank/logo.png", width=220)
st.sidebar.title("CashGAB")
tab = st.sidebar.radio("Navigation", ["Tableau de bord analytique", "Prévisions LSTM 20 GAB"])

# Filters: region, agence, date range
st.sidebar.markdown("---")
regions = ["Toutes"] + sorted(df["region"].dropna().unique().tolist())
region_filter = st.sidebar.selectbox("Région", regions)
df_region = df[df["region"] == region_filter] if region_filter != "Toutes" else df.copy()
agences = ["Toutes"] + sorted(df_region["agence"].dropna().unique().tolist())
agence_filter = st.sidebar.selectbox("Agence", agences)

st.sidebar.markdown("Période (TDB)")
date_min = df["ds"].min().date()
date_max = df["ds"].max().date()
date_range = st.sidebar.date_input("Date début / fin", [date_min, date_max])
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

# Seuil critique global optional
st.sidebar.markdown("---")
user_seuil = st.sidebar.number_input("Seuil critique global (MAD) — facultatif (0 = auto)", value=0, step=10000)

# Apply filters
df_filtered = df.copy()
if region_filter != "Toutes":
    df_filtered = df_filtered[df_filtered["region"] == region_filter]
if agence_filter != "Toutes":
    df_filtered = df_filtered[df_filtered["agence"] == agence_filter]
df_filtered = df_filtered[(df_filtered["ds"] >= start_date) & (df_filtered["ds"] <= end_date)]

# -----------------------
# Compute per-GAB last state and thresholds
# -----------------------
if (not df_filtered.empty) and ("num_gab" in df_filtered.columns):
    # last row per GAB in filtered period
    idx = df_filtered.groupby("num_gab")["ds"].idxmax()
    df_latest = df_filtered.loc[idx].copy().reset_index(drop=True)
else:
    df_latest = pd.DataFrame()

# compute per-gab average (used as default threshold)
if not df_filtered.empty:
    df_avg_gab = df_filtered.groupby("num_gab")["total_montant"].mean().to_dict()
else:
    df_avg_gab = {}

def get_seuil_for_gab(gab_id):
    if user_seuil and user_seuil > 0:
        return user_seuil
    return df_avg_gab.get(gab_id, 100000)

if not df_latest.empty:
    df_latest["seuil_critique"] = df_latest["num_gab"].apply(get_seuil_for_gab)
    def classify(row):
        s = row["seuil_critique"]
        v = float(row["total_montant"])
        if v < s:
            return "Critique"
        elif v < 2*s:
            return "Alerte"
        else:
            return "Normal"
    df_latest["status"] = df_latest.apply(classify, axis=1)

# -----------------------
# TAB: Tableau de bord analytique
# -----------------------
if tab == "Tableau de bord analytique":
    st.title("CashGAB — Tableau de bord analytique")

    # KPIs (top)
    montant_total = df_filtered["total_montant"].sum() if not df_filtered.empty else 0
    nombre_ops = int(df_filtered["total_nombre"].sum()) if "total_nombre" in df_filtered.columns and not df_filtered.empty else int(df_filtered.shape[0])
    nb_gabs = df_filtered["num_gab"].nunique() if not df_filtered.empty else 0
    pct_crit = (df_latest["status"].eq("Critique").sum() / df_latest.shape[0] * 100) if not df_latest.empty else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="kpi-card"><div class="kpi-title">Montant total retraits</div><div class="kpi-value">{montant_total/1_000_000:,.2f} M MAD</div><div class="kpi-sub">Période sélectionnée</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card"><div class="kpi-title">Nombre d\'opérations</div><div class="kpi-value">{nombre_ops:,}</div><div class="kpi-sub">Période sélectionnée</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card"><div class="kpi-title">Nombre GAB</div><div class="kpi-value">{nb_gabs}</div><div class="kpi-sub">Filtrés</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi-card"><div class="kpi-title">GAB critiques (%)</div><div class="kpi-value">{pct_crit:.1f}%</div><div class="kpi-sub">Basé sur dernière semaine</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Evolution des retraits (graph sophistiqué)
    st.subheader("Évolution des retraits")
    if df_filtered.empty:
        st.info("Aucune donnée pour les filtres sélectionnés.")
    else:
        df_evol = df_filtered.groupby("ds")["total_montant"].sum().reset_index().sort_values("ds")
        df_evol["ma4"] = df_evol["total_montant"].rolling(window=4, min_periods=1).mean()
        df_evol["pct_change"] = df_evol["total_montant"].pct_change().fillna(0) * 100
        last = df_evol.iloc[-1]
        prev = df_evol.iloc[-2] if len(df_evol) > 1 else last
        delta_pct = ((last["total_montant"] - prev["total_montant"]) / prev["total_montant"] * 100) if prev["total_montant"] != 0 else 0

        # small KPIs for chart
        ca, cb, cc = st.columns(3)
        ca.metric("Dernière semaine (K MAD)", f"{last['total_montant']/1000:,.1f}", f"{delta_pct:+.1f}% vs sem. préc.")
        cb.metric("MA4 (K MAD)", f"{last['ma4']/1000:,.1f}")
        cc.metric("Variation semaine", f"{last['pct_change']:.1f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_evol["ds"], y=df_evol["total_montant"]/1000, mode="lines+markers", name="Total retraits",
                                 hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Montant: %{y:,.0f} K MAD<extra></extra>"))
        fig.add_trace(go.Scatter(x=df_evol["ds"], y=df_evol["ma4"]/1000, mode="lines", name="MA(4 semaines)", line=dict(dash="dash")))
        anomalies = df_evol[df_evol["total_montant"] < 0.7 * df_evol["ma4"]]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(x=anomalies["ds"], y=anomalies["total_montant"]/1000, mode="markers", name="Anomalies (<<MA4)",
                                     marker=dict(color="red", size=9, symbol="x")))

        # stable rangeselector + rangeslider
        fig.update_layout(
            title="Evolution hebdomadaire des retraits avec MA(4)",
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=8, label="2M", step="week", stepmode="backward"),
                        dict(count=26, label="6M", step="week", stepmode="backward"),
                        dict(step="all", label="Tout")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            ),
            yaxis_title="Montant (K MAD)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Répartition régionale & évolution
    st.subheader("Répartition régionale & évolution")
    if df_filtered.empty:
        st.info("Aucune donnée pour la sélection.")
    else:
        df_region_tot = df_filtered.groupby("region").agg(total_sum=("total_montant", "sum"), mean_hb=("total_montant", "mean")).reset_index()
        metric = st.selectbox("Metric affichée", ["Montant total", "Montant moyen hebdo"], index=0)
        metric_col = "total_sum" if metric == "Montant total" else "mean_hb"
        df_region_tot = df_region_tot.sort_values(metric_col, ascending=True)
        fig_r = go.Figure()
        fig_r.add_trace(go.Bar(x=df_region_tot[metric_col]/1000, y=df_region_tot["region"], orientation="h",
                               marker=dict(color=df_region_tot[metric_col], colorscale="Blues", showscale=False),
                               text=(df_region_tot[metric_col]/1000).round(0), textposition="auto",
                               hovertemplate="<b>%{y}</b><br>" + metric + ": %{x:,.0f} K MAD<br>Part: %{customdata:.1f}%<extra></extra>",
                               customdata=(df_region_tot[metric_col] / df_region_tot[metric_col].sum() * 100).round(2)))
        region_choice = st.selectbox("Afficher la tendance hebdo pour la région:", ["Aucune"] + df_region_tot["region"].tolist())
        if region_choice != "Aucune":
            df_reg_week = df_filtered[df_filtered["region"] == region_choice].groupby("ds")["total_montant"].sum().reset_index().sort_values("ds")
            if not df_reg_week.empty:
                # overlay: simple line (visual trick)
                fig_r.add_trace(go.Scatter(x=df_reg_week["total_montant"]/1000, y=[region_choice]*len(df_reg_week), mode="lines+markers",
                                           line=dict(color="firebrick", width=2), name=f"Tendance {region_choice}",
                                           hovertemplate="%{x:.0f} K MAD (%{y})<extra></extra>"))
        fig_r.update_layout(title=f"Répartition par région — {metric}", xaxis_title="K MAD", yaxis_title="", height=520)
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("---")

    # Alertes récentes & fiches réseau (clickable)
    st.subheader("Alertes récentes et fiches réseau")
    if df_latest.empty:
        st.info("Aucune donnée GAB pour la sélection.")
    else:
        cols_show = ["num_gab"]
        if "agence" in df_latest.columns: cols_show.append("agence")
        if "region" in df_latest.columns: cols_show.append("region")
        cols_show += ["total_montant", "seuil_critique", "status"]
        display_df = df_latest[cols_show].copy().reset_index(drop=True)
        display_df["status_rank"] = display_df["status"].map({"Critique": 0, "Alerte": 1, "Normal": 2}).fillna(3)
        display_df = display_df.sort_values(["status_rank", "total_montant"])

        st.markdown("Filtrer par statut :")
        stat_filter = st.multiselect("Statuts", options=["Critique", "Alerte", "Normal"], default=["Critique", "Alerte", "Normal"])
        df_to_show = display_df[display_df["status"].isin(stat_filter)]

        for _, row in df_to_show.iterrows():
            gabb = row["num_gab"]
            title = f"GAB {gabb} — {row.get('agence','-')} — {row.get('region','-')}"
            with st.expander(title, expanded=False):
                st.write(f"**Montant retiré (K MAD)**: {row['total_montant']/1000:,.1f}")
                st.write(f"**Seuil critique (MAD)**: {row['seuil_critique']:,.0f}")
                # status badge
                st.markdown(
                    f"<span class='badge-crit'>{'Critique'}</span>" if row["status"] == "Critique" else
                    (f"<span class='badge-alert'>{'Alerte'}</span>" if row["status"] == "Alerte" else f"<span class='badge-norm'>{'Normal'}</span>"),
                    unsafe_allow_html=True
                )
                # recent weekly series for this GAB
                df_gab = df_filtered[df_filtered["num_gab"] == gabb].sort_values("ds")
                if not df_gab.empty:
                    fig_g = go.Figure()
                    fig_g.add_trace(go.Bar(x=df_gab["ds"], y=df_gab["total_montant"]/1000, name="Hebdo (K MAD)"))
                    fig_g.add_trace(go.Scatter(x=df_gab["ds"], y=df_gab["total_montant"].rolling(4, min_periods=1).mean()/1000,
                                               mode="lines", name="MA4"))
                    fig_g.update_layout(title=f"Historique hebdo GAB {gabb}", xaxis_title="Date", yaxis_title="K MAD")
                    st.plotly_chart(fig_g, use_container_width=True)
                else:
                    st.info("Pas d'historique pour ce GAB.")

# -----------------------
# TAB: Prévisions LSTM 20 GAB
# -----------------------
if tab == "Prévisions LSTM 20 GAB":
    st.title("Prévisions LSTM - 20 GAB")
    st.sidebar.header("Paramètres de simulation")
    if not TF_AVAILABLE:
        st.error("TensorFlow / joblib non disponibles sur cet environnement — impossibilité d'exécuter les modèles LSTM.")
    else:
        gab_options = [gab for gab in sorted(df["num_gab"].unique()) if gab in lstm_models]
        if not gab_options:
            st.warning("Aucun GAB disponible avec modèles LSTM (vérifie fichiers lstm_gab_*.h5 et scaler_gab_<id>.save).")
        else:
            gab_selected = st.sidebar.selectbox("Sélectionner un GAB", gab_options)
            period_forecast = st.sidebar.selectbox("Période de prévision (semaines)", [1,2,4,6])
            variation = st.sidebar.slider("Facteur de variation (%)", -50,50,0)

            df_gab = df[df["num_gab"] == gab_selected].sort_values("ds")
            if len(df_gab) < 52:
                st.warning("Pas assez de données pour la prévision LSTM (min 52 semaines).")
            else:
                try:
                    n_steps = 4
                    scaler = lstm_scalers.get(gab_selected)
                    model = lstm_models.get(gab_selected)
                    if model is None:
                        st.error("Modèle LSTM manquant pour ce GAB.")
                    elif scaler is None:
                        st.error("Scaler manquant pour ce GAB.")
                    else:
                        # prepare training sequences
                        y_vals = df_gab[['y']].values.astype(float)
                        y_scaled = scaler.transform(y_vals)
                        X = []
                        for i in range(len(y_scaled) - n_steps):
                            X.append(y_scaled[i:i+n_steps])
                        X = np.array(X).reshape(-1, n_steps, 1)

                        # predicted on historical sequences (model->pred)
                        y_pred_scaled = model.predict(X, verbose=0)
                        y_pred = scaler.inverse_transform(y_pred_scaled)

                        # actual (aligned)
                        y_true = df_gab['y'].values[n_steps:]
                        dates_hist = df_gab['ds'].values[n_steps:]

                        # iterative future forecast
                        last_seq = y_scaled[-n_steps:].reshape(1, n_steps, 1)
                        future_preds = []
                        future_dates = [df_gab["ds"].max() + pd.Timedelta(weeks=i+1) for i in range(period_forecast)]
                        for _ in range(period_forecast):
                            pred_scaled = model.predict(last_seq, verbose=0)
                            pred = scaler.inverse_transform(pred_scaled)[0,0]
                            pred_adj = pred * (1 + variation/100)
                            future_preds.append(pred_adj/1000)
                            # update last_seq with scaled pred
                            last_seq = np.concatenate([last_seq[:,1:,:], pred_scaled.reshape(1,1,1)], axis=1)

                        # plot
                        fig_pred = go.Figure()
                        fig_pred.add_trace(go.Scatter(x=dates_hist, y=y_true/1000, mode="lines+markers", name="Montant réel"))
                        fig_pred.add_trace(go.Scatter(x=dates_hist, y=y_pred.flatten()/1000, mode="lines+markers", name="Prédiction LSTM (in-sample)"))
                        fig_pred.add_trace(go.Scatter(x=future_dates, y=future_preds, mode="lines+markers", name=f"Prévisions ajustées ({variation}%)"))
                        fig_pred.update_layout(title=f"Prévision LSTM GAB {gab_selected}", xaxis_title="Date", yaxis_title="Montant retiré (K MAD)")
                        st.plotly_chart(fig_pred, use_container_width=True)

                        # download csv
                        df_csv = pd.DataFrame({
                            "ds": list(dates_hist) + future_dates,
                            "y_true_k": list(y_true/1000) + [None]*period_forecast,
                            "y_pred_k": list(y_pred.flatten()/1000) + future_preds
                        })
                        st.download_button("Télécharger CSV", df_csv.to_csv(index=False), file_name=f"pred_{gab_selected}.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Erreur lors des prévisions: {e}")
