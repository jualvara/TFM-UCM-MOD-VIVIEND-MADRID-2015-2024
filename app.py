import streamlit as st
import pandas as pd
import joblib
import json
import pydeck as pdk
import altair as alt

# -------------------------------
# Cargar modelo cacheado
# -------------------------------
@st.cache_resource
def load_model():
    with st.spinner("Cargando modelo... ⏳"):
        return joblib.load("modelo_precio.pkl")

model = load_model()

# -------------------------------
# Cargar catálogo de distritos/barrios
# -------------------------------
@st.cache_data
def load_catalogo():
    return pd.read_csv("catalogo_distritos_barrios.csv")

catalogo = load_catalogo()

# -------------------------------
# Cargar columnas esperadas
# -------------------------------
with open("columns.json", "r") as f:
    expected_cols = json.load(f)

# -------------------------------
# Interfaz principal
# -------------------------------
st.title("Predicción del Precio de Vivienda en Madrid 🏠")
st.markdown("Aplicación de prueba usando modelo entrenado (TFM 2013–2024).")

# =========================================================
# 1️⃣ Calculadora de precio de vivienda
# =========================================================
st.header("1️⃣ Calculadora de precio de vivienda")

# Selector de distrito
distrito = st.selectbox("Distrito", catalogo["DISTRITO_x"].unique())

# Selector dinámico de barrio
barrios_distrito = catalogo[catalogo["DISTRITO_x"] == distrito]["BARRIO"].unique()
barrio = st.selectbox("Barrio", barrios_distrito)

# Tipo de vivienda
tipo_vivienda = st.selectbox("Tipo de vivienda", ["Piso", "Chalet", "Estudio", "Otro"])

# Variables numéricas
superficie = st.number_input("Superficie (m²)", 20, 300, 80)
antiguedad = st.slider("Antigüedad (años)", 0, 100, 30)
renta = st.number_input("Renta media distrital (€)", 500, 6000, 2500)
paro = st.slider("Tasa de paro (%)", 0, 40, 10)
zonas_verdes = st.number_input("Zonas verdes por habitante (m²)", 0, 100, 20)

# Construir DataFrame con todas las columnas esperadas
data = pd.DataFrame(columns=expected_cols)
data.loc[0] = 0  # inicializar en cero
data.loc[0, "DISTRITO_x"] = distrito
data.loc[0, "BARRIO"] = barrio
data.loc[0, "TIPO_VIVIENDA"] = tipo_vivienda
data.loc[0, "superficie"] = superficie
data.loc[0, "antiguedad"] = antiguedad
data.loc[0, "renta"] = renta
data.loc[0, "paro"] = paro
data.loc[0, "zonas_verdes"] = zonas_verdes

# Predicción
if st.button("Predecir Precio"):
    with st.spinner("Calculando predicción... 🔮"):
        pred = model.predict(data)[0]
    st.success(f"El precio estimado es: **{pred:,.0f} €/m²**")

# =========================================================
# 2️⃣ Mapa interactivo de distritos
# =========================================================
st.header("2️⃣ Mapa interactivo de distritos de Madrid")

# Cargar coordenadas de distritos
coords = pd.read_csv("distritos_madrid_coords.csv")

# Capa de puntos
layer = pdk.Layer(
    "ScatterplotLayer",
    coords,
    get_position=["Longitud", "Latitud"],
    get_radius=400,
    get_color=[255, 0, 0, 160],
    pickable=True
)

# Vista inicial del mapa
view_state = pdk.ViewState(latitude=40.4168, longitude=-3.7038, zoom=11)

# Renderizar mapa
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{Distrito}"}))

with st.expander("📋 Ver datos de distritos"):
    st.dataframe(coords)


# -------------------------------
# 3️⃣ Comparador de distritos
# -------------------------------
st.header("3️⃣ Comparador de distritos")

# Para demo: añadimos variables ficticias al CSV de distritos
# 👉 Sustituye esto por datos reales si los tienes en tu catálogo
distritos_data = coords.copy()
distritos_data["Precio_m2"] = [4950, 3600, 2800, 3900, 4300, 5200, 5100, 4100, 4800, 3400,
                               4000, 3700, 4200, 2500, 3000, 3100, 3300, 3500, 2900, 2700, 3200]
distritos_data["Renta_media"] = [32000, 28000, 24000, 27000, 30000, 42000, 41000, 31000, 38000, 25000,
                                 33000, 29000, 34000, 20000, 23000, 22000, 26000, 27000, 21000, 20500, 22500]
distritos_data["Paro"] = [9.0, 11.2, 15.0, 10.5, 8.5, 6.2, 6.5, 12.1, 7.8, 14.5,
                          10.0, 11.0, 9.5, 16.0, 13.5, 12.8, 11.7, 10.2, 15.5, 16.2, 14.0]

# Selección múltiple
seleccion = st.multiselect("Selecciona distritos para comparar:", distritos_data["Distrito"].unique())

if seleccion:
    df_sel = distritos_data[distritos_data["Distrito"].isin(seleccion)]

    st.subheader("📊 Tabla comparativa")
    st.dataframe(df_sel[["Distrito", "Precio_m2", "Renta_media", "Paro"]])

    # Transformar a formato largo para Altair
    df_melt = df_sel.melt(id_vars="Distrito",
                          value_vars=["Precio_m2", "Renta_media", "Paro"],
                          var_name="Indicador",
                          value_name="Valor")

    st.subheader("📈 Comparación gráfica")
    chart = (
        alt.Chart(df_melt)
        .mark_bar()
        .encode(
            x=alt.X("Distrito:N", title="Distrito"),
            y=alt.Y("Valor:Q", title="Valor"),
            color="Indicador:N",
            tooltip=["Distrito", "Indicador", "Valor"]
        )
        .properties(width=600, height=400)
    )

    st.altair_chart(chart, use_container_width=True)

else:
    st.info("Selecciona al menos un distrito para ver la comparación.")