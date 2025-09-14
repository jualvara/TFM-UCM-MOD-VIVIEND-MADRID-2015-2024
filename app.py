# Auto instalación de pandas si no está (solo para debugging en entornos con problemas)
try:
    import pandas as pd
except ImportError:
    import os
    os.system("pip install pandas")
    import pandas as pd




import streamlit as st
import joblib
import numpy as np
import os
import subprocess


MODELO_PATH = "modelo_precio.pkl"
CATALOGO_PATH = "catalogo_distritos_barrios.csv"

# ─────────────────────────────────────────────
# 1. Ejecutar entrenamiento.py si no existe el modelo
if not os.path.exists(MODELO_PATH):
    st.warning("🔄 Entrenando modelo por primera vez...")
    result = subprocess.run(["python", "entrenamiento.py"], capture_output=True, text=True)
    if result.returncode != 0:
        st.error("❌ Error al entrenar el modelo:")
        st.code(result.stderr)
        st.stop()
    else:
        st.success("✅ Modelo entrenado correctamente.")

# ─────────────────────────────────────────────
# 2. Cargar modelo y catálogo

@st.cache_resource
def cargar_modelo():
    return joblib.load(MODELO_PATH)

@st.cache_data
def cargar_catalogo():
    return pd.read_csv(CATALOGO_PATH)

st.title("🏡 Predicción del Precio de Vivienda en Madrid")

modelo = cargar_modelo()
catalogo = cargar_catalogo()

# ─────────────────────────────────────────────
# 3. Selectores dinámicos

distrito = st.selectbox("Selecciona un distrito", catalogo["DISTRITO_x"].unique())
barrios_filtrados = catalogo[catalogo["DISTRITO_x"] == distrito]["BARRIO"].unique()
barrio = st.selectbox("Selecciona un barrio", barrios_filtrados)
tipo_vivienda = st.selectbox("Tipo de vivienda", ["NUEVA", "SEGUNDA MANO"])

st.markdown("---")

# ─────────────────────────────────────────────
# 4. Entradas adicionales

st.subheader("📊 Introduce variables adicionales:")
transacciones = st.number_input("Transacciones", min_value=0, value=50)
renta_neta_persona = st.number_input("Renta neta por persona (€)", min_value=0, value=20000)
renta_neta_hogar = st.number_input("Renta neta por hogar (€)", min_value=0, value=40000)
viviendas_turisticas = st.number_input("Viviendas turísticas reales", min_value=0, value=100)
paro = st.number_input("Tasa de paro (%)", min_value=0.0, value=7.0)
seguridad = st.slider("Percepción de seguridad (1-10)", min_value=1.0, max_value=10.0, value=7.5)
satisfaccion = st.slider("Satisfacción en el barrio (1-10)", min_value=1.0, max_value=10.0, value=7.0)

# ─────────────────────────────────────────────
# 5. Predicción

if st.button("📈 Predecir precio por m²"):
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

    prediccion = modelo.predict(entrada)[0]
    st.success(f"💶 Precio estimado: **{prediccion:,.2f} €/m²**")

# ─────────────────────────────────────────────
# 6. Botón para descargar modelo generado

if os.path.exists(MODELO_PATH):
    with open(MODELO_PATH, "rb") as f:
        st.download_button(
            label="⬇️ Descargar modelo entrenado (.pkl)",
            data=f,
            file_name="modelo_precio.pkl",
            mime="application/octet-stream"
        )
