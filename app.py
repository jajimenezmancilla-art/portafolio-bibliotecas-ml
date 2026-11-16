"""App de Streamlit: predicción de demanda de préstamos (Biblioteca Pública de Seattle).

Ejecutar desde la raíz del proyecto:
    streamlit run app.py

Toda la lógica de carga, modelos y evaluación está en src/; aquí solo se presenta.
"""
import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from src.modelos import (  # noqa: E402
    HORIZONTE, evaluar, random_forest, regresion_historico, regresion_ventana,
)
from src.red_neuronal import red_neuronal  # noqa: E402
from src.serie_mensual import cargar_serie_mensual  # noqa: E402

# Mismos modelos y configuraciones que el notebook 02
MODELOS = {
    "Regresión lineal (histórico completo)": regresion_historico,
    "Regresión lineal (ventana 36 meses)": partial(regresion_ventana, ventana=36),
    "Regresión lineal (ventana 48 meses, ajustada)": partial(regresion_ventana, ventana=48),
    "Random Forest (min_samples_leaf=2)": partial(random_forest, min_samples_leaf=2),
    "Random Forest (min_samples_leaf=5, ajustado)": partial(random_forest, min_samples_leaf=5),
    "Red neuronal (capas 64-32, 300 épocas)": partial(red_neuronal, unidades=(64, 32), epocas=300),
}

# Periodos de anomalía que se sombrean en el gráfico (inicio y fin inclusivos)
ANOMALIAS = [
    ("2020-04-01", "2020-08-31", "#d62728", "Pandemia (2020)"),
    ("2024-05-01", "2024-07-31", "#ff7f0e", "Ataque de ransomware (2024)"),
]


@st.cache_data(show_spinner="Entrenando modelos (solo la primera vez)...")
def calcular_resultados():
    """Carga la serie y evalúa todos los modelos. Se guarda en caché entre interacciones."""
    serie = cargar_serie_mensual(str(RAIZ / "data" / "totales_mensuales.csv"))
    resultados = {nombre: evaluar(serie, predictor) for nombre, predictor in MODELOS.items()}
    return serie, resultados


st.set_page_config(page_title="Demanda de préstamos bibliotecarios", layout="wide")
st.title("Predicción de demanda de préstamos bibliotecarios")
st.write(
    "Predicción del total mensual de préstamos de la Biblioteca Pública de Seattle, "
    "con datos abiertos de 2015 a 2026."
)

serie, resultados = calcular_resultados()

# 1. Serie completa con anomalías sombreadas
st.subheader("Serie mensual de préstamos")
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(serie["fecha"], serie["total_checkouts"], color="#1f4e79", label="Total mensual")
for inicio, fin, color, etiqueta in ANOMALIAS:
    ax.axvspan(pd.Timestamp(inicio), pd.Timestamp(fin), color=color, alpha=0.2, label=etiqueta)
ax.set_xlabel("Fecha")
ax.set_ylabel("Total de préstamos")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower left")
st.pyplot(fig)
plt.close(fig)
st.caption(
    "Sombreado: las dos caídas anómalas. **Pandemia (abr–ago 2020):** cierre de las "
    "sedes desde mediados de marzo de 2020. **Ataque de ransomware (may–jul 2024):** "
    "el ataque se detectó el 25 de mayo de 2024 y dejó fuera de servicio catálogos y "
    "préstamos durante semanas. Ver el README para las fuentes."
)

# 2. Selector de modelo: real vs. predicho en el holdout y métricas
st.subheader("Modelo: real frente a predicho en los últimos 12 meses")
modelo = st.selectbox("Elige un modelo", list(MODELOS))
r = resultados[modelo]

col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE holdout", f"{r['MAE holdout']:,.0f}")
col2.metric("RMSE holdout", f"{r['RMSE holdout']:,.0f}")
col3.metric("MAE ventana móvil", f"{r['MAE móvil']:,.0f}")
col4.metric("RMSE ventana móvil", f"{r['RMSE móvil']:,.0f}")
st.caption("MAE y RMSE en préstamos por mes. Menor es mejor.")

test = serie.iloc[-HORIZONTE:]
fig, ax = plt.subplots(figsize=(12, 4.5))
ultimos = serie.iloc[-36:]
ax.plot(ultimos["fecha"], ultimos["total_checkouts"], color="black", label="Real")
ax.plot(test["fecha"], r["pred_holdout"], "--", color="#d62728", linewidth=2, label="Predicho")
ax.axvline(test["fecha"].iloc[0], color="gray", linestyle=":", label="Inicio del holdout")
ax.set_xlabel("Fecha")
ax.set_ylabel("Total de préstamos")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower left")
st.pyplot(fig)
plt.close(fig)

# 3. Tabla comparativa de todos los modelos
st.subheader("Comparación de modelos")
tabla = pd.DataFrame({
    nombre: {
        "MAE holdout": res["MAE holdout"],
        "RMSE holdout": res["RMSE holdout"],
        "MAE ventana móvil": res["MAE móvil"],
        "RMSE ventana móvil": res["RMSE móvil"],
    }
    for nombre, res in resultados.items()
}).T.round(0).astype(int)
st.dataframe(tabla, width="stretch")
st.caption(
    f"Holdout: últimos {HORIZONTE} meses. Ventana móvil: últimos 24 meses, con el modelo "
    "re-entrenado antes de cada mes. Los modelos ajustados se eligieron con validación "
    "cruzada sobre meses anteriores al holdout."
)
