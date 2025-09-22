# =========================
# Sidebar - Filtres dépendants
# =========================
st.sidebar.header("Filtres")

# Filtre Région
region_list = df['region'].unique()
selected_region = st.sidebar.selectbox("Région :", region_list)

# Filtre Agence dépendant de la région
agence_list = df[df['region'] == selected_region]['agence'].unique()
selected_agence = st.sidebar.selectbox("Agence :", agence_list)

# Filtre GAB dépendant de l'agence
gab_list = df[(df['region'] == selected_region) & (df['agence'] == selected_agence)]['lib_gab'].unique()
selected_gab = st.sidebar.selectbox("GAB :", gab_list)

# Filtre période
date_min = df['ds'].min()
date_max = df['ds'].max()
start_date = st.sidebar.date_input("Date début :", date_min)
end_date = st.sidebar.date_input("Date fin :", date_max)
