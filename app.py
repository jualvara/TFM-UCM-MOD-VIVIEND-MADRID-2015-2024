import streamlit as st
import pandas as pd
import joblib

# Cargar modelo entrenado
model = joblib.load("modelo_precio.pkl")

# Cargar catálogo ligero
catalogo = pd.read_csv("catalogo_distritos_barrios.csv")

st.title("Predicción del Precio de Vivienda en Madrid 🏠")
st.markdown("Aplicación de prueba usando modelo entrenado (TFM 2013–2024).")

# Selector de distrito
distrito = st.selectbox("Distrito", catalogo["DISTRITO_x"].unique())

# Filtrar barrios dinámicamente
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

# Construir DataFrame de entrada
data = pd.DataFrame({
    "DISTRITO_x": [distrito],
    "BARRIO": [barrio],
    "TIPO_VIVIENDA": [tipo_vivienda],
    "superficie": [superficie],
    "antiguedad": [antiguedad],
    "renta": [renta],
    "paro": [paro],
    "zonas_verdes": [zonas_verdes]
})

# Predicción
if st.button("Predecir Precio"):
    pred = model.predict(data)[0]
    st.success(f"El precio estimado es: **{pred:,.0f} €/m²**")

