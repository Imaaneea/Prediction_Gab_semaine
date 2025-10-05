import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import glob
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from datetime import timedelta

# ========================================
# Configuration de la page
# ========================================
st.set_page_config(page_title="Dashboard GAB", layout="wide")

# ========================================
# Chargement des données (base)
# ========================================
@st.cache_data
def load_weekly():
    try:
        df = pd.read_csv("df_weekly_clean.csv", parse_dates=["ds"])
        df["num_gab"] = df["num_gab"].astype(str)
        df["week_day"] = df["ds"].dt.dayofweek
        df["week"] = df["ds"].dt.isocalendar().week
        df["year"] = df["ds"].dt.year
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement de df_weekly_clean.csv : {e}")
        return pd.DataFrame()

# Chargement optionnel df_gabs (infos réseau / coordonnées)
@st.cache_data
def load_gabs_info():
    try:
        dfg = pd.read_csv("df_gabs.csv")
        if "num_gab" in dfg.columns:
            dfg["num_gab"] = dfg["num_gab"].astype(str)
        return dfg
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

df = load_weekly()
df_gabs_info = load_gabs_info()  # contient éventuellement lat/lon, agence, region, etc.

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
        except Exception as e:
            st.warning(f"Impossible de charger modèle {gab_id}: {e}")
            continue
        try:
            scalers[gab_id] = joblib.load(scaler_file)
        except Exception as e:
            # scaler missing -> still keep model but warn
            st.warning(f"Scaler manquant pour {gab_id}: {e}")
    return models, scalers

lstm_models, lstm_scalers = load_lstm_models()

# ========================================
# Navigation (on garde tes 2 onglets + on ajoute "Réseau")
# ========================================
tab = st.sidebar.radio("Navigation", ["Tableau de bord analytique", "Prévisions LSTM 20 GAB", "Réseau & Supervision"])

# -------------------------
# Onglet 1 : Tableau de bord analytique (base conservée + KPIs filtrés)
# -------------------------
if tab == "Tableau de bord analytique":
    st.title("Tableau de bord analytique - GAB")

    # Sidebar filtres (robustes si colonnes manquantes)
    st.sidebar.header("Filtres")
    regions = df["region"].dropna().unique() if "region" in df.columns else []
    region = st.sidebar.selectbox("Région", ["Toutes"] + sorted(regions.tolist())) if len(regions) > 0 else "Toutes"

    if region != "Toutes" and "agence" in df.columns:
        agences = df[df["region"] == region]["agence"].dropna().unique()
    else:
        agences = df["agence"].dropna().unique() if "agence" in df.columns else []
    agence = st.sidebar.selectbox("Agence", ["Toutes"] + sorted(agences.tolist())) if len(agences) > 0 else "Toutes"

    if agence != "Toutes" and "agence" in df.columns:
        gabs = df[df["agence"] == agence]["num_gab"].dropna().unique()
    else:
        gabs = df["num_gab"].dropna().unique()
    gab = st.sidebar.selectbox("GAB", ["Tous"] + sorted(gabs.tolist())) if len(gabs) > 0 else "Tous"

    # Date filters (safe defaults)
    date_min = df["ds"].min() if not df.empty else pd.to_datetime("2020-01-01")
    date_max = df["ds"].max() if not df.empty else pd.to_datetime("2025-01-01")
    date_debut = st.sidebar.date_input("Date début", date_min)
    date_fin = st.sidebar.date_input("Date fin", date_max)

    # Apply filters
    df_filtered = df.copy()
    if region != "Toutes" and "region" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["region"] == region]
    if agence != "Toutes" and "agence" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["agence"] == agence]
    if gab != "Tous":
        df_filtered = df_filtered[df_filtered["num_gab"] == gab]
    df_filtered = df_filtered[(df_filtered["ds"] >= pd.to_datetime(date_debut)) &
                              (df_filtered["ds"] <= pd.to_datetime(date_fin))]

    if df_filtered.empty:
        st.warning("Aucune donnée après application des filtres.")
        st.stop()

    # KPIs dynamiques (demandés)
    nb_gab = df_filtered["num_gab"].nunique()
    montant_total = df_filtered["total_montant"].sum()
    nombre_operations = df_filtered["total_nombre"].sum() if "total_nombre" in df_filtered.columns else None

    st.subheader("KPIs principaux (filtrés)")
    if nombre_operations is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nombre de GAB", f"{nb_gab}")
        c2.metric("Montant total retraits", f"{montant_total/1000:,.0f} KDH")
        c3.metric("Nombre total opérations", f"{nombre_operations:,.0f}")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Nombre de GAB", f"{nb_gab}")
        c2.metric("Montant total retraits", f"{montant_total/1000:,.0f} KDH")

    # autres KPIs existants (conservés)
    volume_moyen_semaine = df_filtered.groupby("week")["total_montant"].mean().mean()
    ecart_type_retraits = df_filtered["total_montant"].std()
    part_weekend = (df_filtered[df_filtered["week_day"] >= 5]["total_montant"].sum() / df_filtered["total_montant"].sum() * 100) if df_filtered["total_montant"].sum() > 0 else 0

    c5, c6 = st.columns(2)
    c5.metric("Volume moyen hebdo", f"{volume_moyen_semaine/1000:,.0f} KDH")
    c6.metric("Part retraits weekend", f"{part_weekend:.1f} %")

    # Evolution plot (conserved)
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

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df_plot["ds"], y=df_plot["total_montant_kdh"], mode="lines+markers", name="Montant retiré (KDH)"))
    fig_line.update_layout(title=title, xaxis_title="Date", yaxis_title="Montant retiré (KDH)")
    st.plotly_chart(fig_line, use_container_width=True)

# -------------------------
# Onglet 2 : Prévisions LSTM 20 GAB (inchangé)
# -------------------------
if tab == "Prévisions LSTM 20 GAB":
    st.title("Prévisions LSTM - 20 GAB")

    # GAB disponibles avec modèles
    gab_options = [gab for gab in sorted(df["num_gab"].unique()) if gab in lstm_models]
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
                # Préparation
                n_steps = 4
                scaler = lstm_scalers[gab_selected]
                model = lstm_models[gab_selected]

                y_scaled = scaler.transform(df_gab[['y']].values)
                X = []
                for i in range(len(y_scaled) - n_steps):
                    X.append(y_scaled[i:i+n_steps])
                X = np.array(X).reshape(-1, n_steps, 1)

                # Prédictions sur toutes les semaines
                y_pred_scaled = model.predict(X, verbose=0)
                y_pred = scaler.inverse_transform(y_pred_scaled)

                y_true = df_gab['y'].values[n_steps:]
                dates = df_gab['ds'][n_steps:]

                # Prévisions futures (paramétrable fixe ici 6)
                last_sequence = y_scaled[-n_steps:].reshape(1, n_steps, 1)
                future_preds = []
                future_steps = 6
                future_dates = [df_gab["ds"].max() + pd.Timedelta(weeks=i+1) for i in range(future_steps)]

                for _ in range(future_steps):
                    pred_scaled = model.predict(last_sequence, verbose=0)
                    pred = scaler.inverse_transform(pred_scaled)[0, 0]
                    future_preds.append(pred/1000)  # KDH
                    last_sequence = np.concatenate([last_sequence[:,1:,:], pred_scaled.reshape(1,1,1)], axis=1)

                # Graphique final
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=dates, y=y_true/1000, mode="lines+markers", name="Montant réel (KDH)"))
                fig_pred.add_trace(go.Scatter(x=dates, y=y_pred.flatten()/1000, mode="lines+markers", name="Prédiction LSTM (KDH)"))
                fig_pred.add_trace(go.Scatter(x=future_dates, y=future_preds, mode="lines+markers", name="Prévisions futures (KDH)"))

                fig_pred.update_layout(xaxis_title="Date", yaxis_title="Montant retiré (KDH)")
                st.plotly_chart(fig_pred, use_container_width=True)

                # Téléchargement CSV
                df_csv = pd.DataFrame({
                    "ds": list(dates) + future_dates,
                    "y_true_kdh": list(y_true/1000) + [None]*future_steps,
                    "y_pred_kdh": list(y_pred.flatten()/1000) + future_preds
                })
                st.download_button(
                    label="Télécharger prévisions CSV",
                    data=df_csv.to_csv(index=False),
                    file_name=f"pred_{gab_selected}.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Erreur lors de la génération des prévisions: {e}")

# -------------------------
# Onglet 3 : Réseau & Supervision (nouveau)
# -------------------------
if tab == "Réseau & Supervision":
    st.title("Réseau & Supervision - Carte, KPIs, Alertes et Simulation")

    # Préparer table d'inventaire réseau à partir de df_gabs_info si dispo, sinon extraire depuis df
    if not df_gabs_info.empty:
        network = df_gabs_info.copy()
        # Try to ensure columns consistency
        if "num_gab" in network.columns:
            network["num_gab"] = network["num_gab"].astype(str)
    else:
        # Build minimal network from df
        network = df.groupby("num_gab").agg({
            "num_gab": "first",
            "total_montant": "mean"
        }).rename(columns={"total_montant": "mean_total"}).reset_index(drop=True)
        # ensure num_gab as column if groupby changed structure
        if "num_gab" not in network.columns and len(network.columns) > 0:
            network["num_gab"] = df["num_gab"].unique()

    # KPI réseau (globales)
    st.subheader("KPIs réseau (vue globale)")
    total_cash = df.groupby("num_gab")["total_montant"].mean().sum()  # somme moyenne par GAB
    gabs_total = df["num_gab"].nunique()
    # compute availability proxy: fraction of GABs with avg weekly > small threshold
    availability = (df.groupby("num_gab")["total_montant"].mean() > 1000).mean() * 100

    k1, k2, k3 = st.columns(3)
    k1.metric("Montant total (somme moy.)", f"{total_cash/1000:,.0f} KDH")
    k2.metric("Nombre GAB (réseau)", f"{gabs_total}")
    k3.metric("Disponibilité (proxy)", f"{availability:.0f} %")

    st.markdown("**Filtres réseaux & recherche**")
    # filters by region/agence/status/search text
    region_net = st.selectbox("Filtrer par région (réseau)", ["Toutes"] + sorted(df["region"].dropna().unique()) if "region" in df.columns else ["Toutes"])
    agence_net = st.selectbox("Filtrer par agence (réseau)", ["Toutes"] + sorted(df["agence"].dropna().unique()) if "agence" in df.columns else ["Toutes"])
    search_text = st.text_input("Recherche (num_gab / libellé)")

    # Seuils configurables (pour alerte / critique)
    st.markdown("**Seuils d'alerte (configurables)**")
    seuil_alerte = st.number_input("Seuil Alerte (montant moyen hebdo)", value=50000, step=1000)
    seuil_critique = st.number_input("Seuil Critique (montant moyen hebdo)", value=150000, step=1000)
    if seuil_critique < seuil_alerte:
        st.error("Le seuil critique doit être supérieur au seuil alerte.")
    else:
        # compute avg weekly per GAB
        gab_avg = df.groupby("num_gab")["total_montant"].mean().reset_index().rename(columns={"total_montant":"avg_week"})
        gab_avg["num_gab"] = gab_avg["num_gab"].astype(str)
        # join network info if available
        if not df_gabs_info.empty and "num_gab" in df_gabs_info.columns:
            gab_table = gab_avg.merge(df_gabs_info.drop_duplicates(subset=["num_gab"]), on="num_gab", how="left")
        else:
            gab_table = gab_avg.copy()

        # derive status
        def status_from_avg(x):
            if x >= seuil_critique:
                return "Critique"
            if x <= seuil_alerte:
                return "Inactif" if x == 0 else "Alerte"
            return "Normal"

        gab_table["status"] = gab_table["avg_week"].apply(status_from_avg)

        # apply network filters and search
        display_table = gab_table.copy()
        if region_net != "Toutes" and "region" in display_table.columns:
            display_table = display_table[display_table["region"] == region_net]
        if agence_net != "Toutes" and "agence" in display_table.columns:
            display_table = display_table[display_table["agence"] == agence_net]
        if search_text:
            display_table = display_table[display_table["num_gab"].str.contains(search_text, na=False) |
                                          display_table.get("lib_gab", "").astype(str).str.contains(search_text, na=False)]

        # counts by status
        counts = display_table["status"].value_counts().to_dict()
        st.markdown(f"✅ Normals: {counts.get('Normal',0)}  •  ⚠️ Alerte: {counts.get('Alerte',0)}  •  🔴 Critique: {counts.get('Critique',0)}  •  ⛔ Inactif: {counts.get('Inactif',0)}")

        # Map (if coords available)
        if ("lat" in gab_table.columns and "lon" in gab_table.columns) or ("latitude" in gab_table.columns and "longitude" in gab_table.columns):
            lat_col = "lat" if "lat" in gab_table.columns else "latitude"
            lon_col = "lon" if "lon" in gab_table.columns else "longitude"
            map_df = display_table.dropna(subset=[lat_col, lon_col])
            if not map_df.empty:
                # color map
                color_map = {"Normal":"green","Alerte":"orange","Critique":"red","Inactif":"gray"}
                fig_map = px.scatter_mapbox(
                    map_df,
                    lat=lat_col, lon=lon_col,
                    hover_name="num_gab",
                    hover_data=["avg_week","status","agence"] if "agence" in map_df.columns else ["avg_week","status"],
                    color="status",
                    size=map_df["avg_week"].fillna(0),
                    color_discrete_map=color_map,
                    zoom=5,
                    mapbox_style="open-street-map",
                    title="Carte réseau (code couleur par statut)"
                )
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info("Aucune coordonnée disponible pour la sélection actuelle.")
        else:
            st.info("Pas de coordonnées disponibles (lat/lon) pour afficher la carte. Ajoute ces colonnes dans df_gabs.csv si possible.")

        # affichage tableau synthèse
        st.subheader("Fiches réseau (aperçu)")
        show_table = display_table[["num_gab","avg_week","status"] + ([c for c in ["agence","region","lib_gab"] if c in display_table.columns])]
        st.dataframe(show_table.sort_values("avg_week", ascending=False).reset_index(drop=True))

        # Export CSV (statuts + métriques)
        csv_bytes = show_table.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger statut réseau (CSV)", data=csv_bytes, file_name="reseau_status.csv", mime="text/csv")

        # -----------------
        # Simulation simple : comparer prévisions LSTM (si dispo) vs threshold
        # -----------------
        st.subheader("Simulation: détection points à risque (prévisions)")
        selected_for_sim = st.multiselect("Choisir des GAB pour simulation (multi)", options=display_table["num_gab"].unique(), default=display_table["num_gab"].unique()[:5])

        simulation_results = []
        future_weeks = st.slider("Semaines futures à simuler", 1, 12, 4)
        if st.button("Lancer la simulation"):
            for g in selected_for_sim:
                if g in lstm_models and g in lstm_scalers:
                    try:
                        # prepare df_gab
                        df_g = df[df["num_gab"] == g].sort_values("ds")
                        n_steps = 4
                        scaler = lstm_scalers[g]
                        model = lstm_models[g]
                        y_scaled = scaler.transform(df_g[['y']].values)
                        last_seq = y_scaled[-n_steps:].reshape(1,n_steps,1)
                        preds = []
                        for _ in range(future_weeks):
                            p_scaled = model.predict(last_seq, verbose=0)
                            p = scaler.inverse_transform(p_scaled)[0,0]
                            preds.append(p)
                            last_seq = np.concatenate([last_seq[:,1:,:], p_scaled.reshape(1,1,1)], axis=1)
                        avg_pred = np.mean(preds)
                        status_pred = "Critique" if avg_pred >= seuil_critique else ("Alerte" if avg_pred <= seuil_alerte else "Normal")
                        simulation_results.append({"num_gab":g, "avg_pred":avg_pred, "status_pred":status_pred})
                    except Exception as e:
                        simulation_results.append({"num_gab":g, "error":str(e)})
                else:
                    simulation_results.append({"num_gab":g, "note":"Modèle/scaler absent"})

            sim_df = pd.DataFrame(simulation_results)
            st.table(sim_df)
            st.download_button("Télécharger résultats simulation", data=sim_df.to_csv(index=False).encode("utf-8"), file_name="simulation_reseau.csv", mime="text/csv")

