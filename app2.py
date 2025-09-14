import streamlit as st
import pandas as pd
import joblib
import sys
import numpy as np
import sklearn
import openpyxl
import json

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
# Interfaz Streamlit
# -------------------------------
st.title("Predicción del Precio de Vivienda en Madrid 🏠")
st.markdown("Aplicación de prueba usando modelo entrenado (TFM 2013–2024).")

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

# -------------------------------
# Construir DataFrame con todas las columnas esperadas
# -------------------------------
data = pd.DataFrame(columns=expected_cols)
data.loc[0] = 0  # inicializar en cero

# Rellenar con inputs del usuario
data.loc[0, "DISTRITO_x"] = distrito
data.loc[0, "BARRIO"] = barrio
data.loc[0, "TIPO_VIVIENDA"] = tipo_vivienda
data.loc[0, "superficie"] = superficie
data.loc[0, "antiguedad"] = antiguedad
data.loc[0, "renta"] = renta
data.loc[0, "paro"] = paro
data.loc[0, "zonas_verdes"] = zonas_verdes

# -------------------------------
# Predicción
# -------------------------------
if st.button("Predecir Precio"):
    with st.spinner("Calculando predicción... 🔮"):
        pred = model.predict(data)[0]
    st.success(f"El precio estimado es: **{pred:,.0f} €/m²**")
