import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_precio.pkl")

# Cargar catálogo de distritos y barrios
@st.cache_data
def cargar_catalogo():
    return pd.read_csv("catalogo_distritos_barrios.csv")

# Interfaz
st.title("Predicción del Precio de Vivienda (€/m²) en Madrid")

modelo = cargar_modelo()
catalogo = cargar_catalogo()

# Selectbox de distrito
distrito = st.selectbox("Selecciona un distrito", catalogo["DISTRITO_x"].unique())

# Filtrar barrios por distrito seleccionado
barrios_filtrados = catalogo[catalogo["DISTRITO_x"] == distrito]["BARRIO"].unique()
barrio = st.selectbox("Selecciona un barrio", barrios_filtrados)

# Tipo de vivienda
tipo_vivienda = st.selectbox("Tipo de vivienda", ["NUEVA", "SEGUNDA MANO"])

st.markdown("---")

# Entradas adicionales (puedes personalizar)
st.subheader("Introduce otras variables:")
transacciones = st.number_input("Transacciones", min_value=0, value=50)
renta_neta_persona = st.number_input("Renta neta por persona (€)", min_value=0, value=20000)
renta_neta_hogar = st.number_input("Renta neta por hogar (€)", min_value=0, value=40000)
viviendas_turisticas = st.number_input("Viviendas turísticas reales", min_value=0, value=100)
paro = st.number_input("Tasa de paro (%)", min_value=0.0, value=7.0)
seguridad = st.slider("Percepción de seguridad (1-10)", min_value=1.0, max_value=10.0, value=7.5)
satisfaccion = st.slider("Satisfacción barrio (1-10)", min_value=1.0, max_value=10.0, value=7.0)

# Botón para predecir
if st.button("Predecir precio por m²"):
    # Crear input como DataFrame con mismas columnas que en el entrenamiento
    entrada = pd.DataFrame([{
        "DISTRITO_x": distrito,
        "BARRIO": barrio,
        "TIPO_VIVIENDA": tipo_vivienda,
        "TRANSACCIONES_x": transacciones,
        "RENTA_NETA_PERSONA_x": renta_neta_persona,
        "RENTA_NETA_HOGAR_x": renta_neta_hogar,
        "VIVIENDAS_TURISTICAS_REAL_x": viviendas_turisticas,
        "Tasa absoluta de paro registrado (febrero)": paro,
        "Percepción de seguridad en el barrio (media) (Robusto 1-10)": seguridad,
        "Satisfacción de la vida en el barrio (media) (Robusto 1-10)": satisfaccion
    }])

    # Predecir
    prediccion = modelo.predict(entrada)[0]
    st.success(f"🔍 Precio estimado: **{prediccion:,.2f} €/m²**")
