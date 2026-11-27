import os

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Configuración inicial de la página web
st.set_page_config(
    page_title="Predictor de Rendimiento Agrícola", page_icon="🌾", layout="centered"
)

st.title("🌾 Aplicativo Predictivo de Rendimiento Agrícola")
st.markdown(
    "Herramienta basada en **Machine Learning (Random Forest optimizado)**."
    " Ingrese la ubicación y el cultivo; el sistema estimará el rinde a partir"
    " de los patrones climáticos históricos de la zona."
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "dataset_agro_clima_con_ratio_perdida.csv")

FEATURES_NUM = [
    "anio",
    "lluvia_total_ciclo_mm",
    "lluvia_periodo_critico_mm",
    "temp_media_ciclo",
    "dias_ola_calor",
    "ratio_perdida_area",
]
FEATURES_CAT = ["cultivo", "provincia", "grupo_cultivo"]


@st.cache_resource
def entrenar_modelo_campeon():
    """Entrena el modelo campeón con TODO el histórico disponible.

    En el notebook se entrena con campañas <= 2018 y se evalúa con 2019-2024.
    Para la app (uso productivo) se reentrena con todas las campañas, de modo
    que el modelo incorpore el nivel tecnológico más reciente.
    """
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df_model = df[df["rendimiento_kgxha"] > 0].copy()

    X = df_model[FEATURES_NUM + FEATURES_CAT]
    y = df_model["rendimiento_kgxha"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURES_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_CAT),
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=10,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model_pipeline.fit(X, y)
    return model_pipeline, df, int(df_model["anio"].max())


with st.spinner("Cargando y entrenando el modelo de Machine Learning..."):
    pipeline, df_base, ultimo_anio = entrenar_modelo_campeon()

# --- ENTRADAS DEL USUARIO ---
st.sidebar.header("Ubicación y Cultivo")

cultivos_disponibles = sorted(df_base["cultivo"].unique())
cultivo_sel = st.sidebar.selectbox("Seleccione el Cultivo", cultivos_disponibles)

grupo_cultivo_sel = df_base.loc[df_base["cultivo"] == cultivo_sel, "grupo_cultivo"].values[0]
st.sidebar.info(f"**Grupo Agronómico:** {grupo_cultivo_sel.capitalize()}")

provincias_disponibles = sorted(df_base["provincia"].unique())
provincia_sel = st.sidebar.selectbox("Provincia", provincias_disponibles)

departamentos_disponibles = sorted(
    df_base[df_base["provincia"] == provincia_sel]["departamento"].unique()
)
departamento_sel = st.sidebar.selectbox("Departamento / Partido", departamentos_disponibles)

anio_sel = st.sidebar.slider(
    "Año de Campaña Objetivo", min_value=2019, max_value=2030, value=2026, step=1
)
if anio_sel > ultimo_anio:
    st.sidebar.caption(
        f"Nota: el modelo (Random Forest) no extrapola la tendencia más allá de"
        f" {ultimo_anio}; para años posteriores usa el nivel tecnológico de {ultimo_anio}."
    )

st.sidebar.subheader("Escenario Climático Zonal")
escenario_climatico = st.sidebar.selectbox(
    "Condición Hídrica Esperada",
    ["Promedio Histórico (Normal)", "Año Seco (Déficit)", "Año Húmedo (Niño)"],
)

# La pérdida de área se conoce recién al cierre de la campaña: en la app se
# ingresa como un supuesto explícito del escenario, no como un dato observado.
pct_perdida = st.sidebar.slider(
    "Pérdida de superficie esperada (%)", min_value=0, max_value=60, value=5, step=1,
    help="Porcentaje de la superficie sembrada que se espera no cosechar.",
)

# --- IMPUTACIÓN AUTOMÁTICA DEL CLIMA SEGÚN EL HISTÓRICO ZONAL ---
df_zonal = df_base[
    (df_base["provincia"] == provincia_sel)
    & (df_base["departamento"] == departamento_sel)
    & (df_base["cultivo"] == cultivo_sel)
]
if df_zonal.empty:
    # Respaldo: mediana nacional del cultivo si no hay datos para ese departamento
    df_zonal = df_base[df_base["cultivo"] == cultivo_sel]

base_lluvia_tot = df_zonal["lluvia_total_ciclo_mm"].median()
base_lluvia_crit = df_zonal["lluvia_periodo_critico_mm"].median()
base_temp = df_zonal["temp_media_ciclo"].median()
base_calor = int(df_zonal["dias_ola_calor"].median())

# Ajuste según el escenario seleccionado (factores fijos, supuestos del modelo)
if escenario_climatico == "Año Seco (Déficit)":
    base_lluvia_tot *= 0.75
    base_lluvia_crit *= 0.70
    base_calor += 5
elif escenario_climatico == "Año Húmedo (Niño)":
    base_lluvia_tot *= 1.25
    base_lluvia_crit *= 1.20
    base_calor = max(0, base_calor - 2)

# --- PREDICCIÓN ---
if st.button("Estimar Rendimiento de la Región"):
    input_data = pd.DataFrame({
        "anio": [anio_sel],
        "lluvia_total_ciclo_mm": [base_lluvia_tot],
        "lluvia_periodo_critico_mm": [base_lluvia_crit],
        "temp_media_ciclo": [base_temp],
        "dias_ola_calor": [base_calor],
        "ratio_perdida_area": [pct_perdida / 100],
        "cultivo": [cultivo_sel],
        "provincia": [provincia_sel],
        "grupo_cultivo": [grupo_cultivo_sel],
    })

    prediccion_rinde = pipeline.predict(input_data)[0]

    st.markdown("---")
    st.subheader("Resultado de la Predicción")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Rendimiento Estimado", value=f"{prediccion_rinde:,.0f} kg/ha")

    with col2:
        st.metric(label="Potencial Productivo", value=f"{(prediccion_rinde / 1000):.2f} t/ha")

    st.success(
        f"Para **{cultivo_sel}** en **{departamento_sel} ({provincia_sel})**, con un"
        f" escenario **{escenario_climatico}**, {pct_perdida}% de pérdida de superficie"
        f" esperada y campaña **{anio_sel}**, el modelo estima un rinde de"
        f" **{prediccion_rinde:,.0f} kg/ha** (referencia histórica zonal:"
        f" {base_lluvia_crit:.0f} mm de lluvia en período crítico)."
    )
    st.caption(
        "Error medio de referencia del modelo en campañas no vistas (2019–2024):"
        " MAE ≈ 676 kg/ha."
    )
