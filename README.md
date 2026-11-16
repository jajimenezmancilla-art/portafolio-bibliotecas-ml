# Predicción de demanda y minería de patrones en bibliotecas

Proyecto de portafolio que aplica machine learning y minería de datos
a datos abiertos de circulación bibliotecaria (Seattle Public Library).
Predice el total mensual de préstamos y detecta caídas anómalas en la serie.

## Cómo correrlo
1. `python -m venv venv`
2. `venv\Scripts\activate` (Windows)
3. `pip install -r requirements.txt`
4. `python src/descargar_datos.py` (genera `data/checkouts_seattle.csv` y `data/totales_mensuales.csv`)
5. `jupyter lab`, y ejecutar en orden:
   - `notebooks/01_explorar_datos.ipynb`: serie, anomalías y tres modelos base.
   - `notebooks/02_modelos_y_red_neuronal.ipynb`: red neuronal, ajuste de hiperparámetros y tabla final.

## Estructura
- `src/descargar_datos.py`: descarga desde la API Socrata de data.seattle.gov.
- `src/serie_mensual.py`: carga de la serie mensual y detección de anomalías.
- `src/modelos.py`: los modelos base, la evaluación (holdout y ventana móvil) y la validación cruzada.
- `src/red_neuronal.py`: red densa en Keras.

## App interactiva (Streamlit)

Desde la raíz del proyecto, con el venv activado:

```
streamlit run app.py
```

La app muestra la serie mensual completa con las dos caídas anómalas
sombreadas, un selector para comparar los seis modelos (real frente a
predicho en los últimos 12 meses, con MAE y RMSE) y la tabla comparativa
de holdout y ventana móvil. Los modelos se entrenan una sola vez por
proceso: la primera carga tarda unos 6 minutos por la red neuronal y después
los cambios de selector son inmediatos.

[CAPTURA AQUÍ]

## Hallazgo 1: dos caídas anómalas en la serie mensual

La serie de préstamos totales (enero 2015 a agosto 2026) tiene una tendencia
al alza y dos episodios de caída. Un mes se marca como anómalo si queda más de
un 20% por debajo de la mediana móvil centrada de 13 meses.

| Episodio | Meses marcados | Caída máxima | Causa |
|---|---|---|---|
| Pandemia de COVID-19 | abril a julio 2020 | −26% (abril) | Cierre de las sedes desde mediados de marzo de 2020, con servicio limitado durante meses ([informe de levy de la biblioteca](https://www.spl.org/Seattle-Public-Library/documents/about-us/levy/reports/2020_Q3_Library-Levy-Report.pdf)) |
| Ransomware | junio y julio 2024 | −48% (junio) | Ataque de ransomware detectado el 25 de mayo de 2024. Dejó fuera de servicio catálogos, préstamos y computadoras públicas durante semanas ([Washington State Standard](https://washingtonstatestandard.com/2024/05/28/ransomware-attack-knocks-seattle-public-librarys-computer-system-offline/), [KUOW](https://www.kuow.org/governing/2024-05-29/hoping-to-use-the-computer-at-a-seattle-library-think-again)) |

Ambas caídas tienen una causa externa identificada, así que se tratan como
eventos y no como ruido. Mayo de 2024 también cayó (−18%, bajo el umbral),
consistente con que el ataque empezó el 25 de mayo.

## Hallazgo 2: comparación de modelos

Todos los modelos predicen el total mensual de préstamos y se evalúan con el
mismo criterio:
- **Holdout:** últimos 12 meses (sep 2025 a ago 2026), entrenando con lo anterior.
- **Ventana móvil:** últimos 24 meses, re-entrenando antes de cada mes.

Los hiperparámetros de los modelos ajustados se eligieron con **validación
cruzada de series de tiempo** sobre dos ventanas de 12 meses *anteriores* al
holdout. El holdout no se usó para elegir nada.

### Los cuatro modelos (configuración inicial)

| Modelo | MAE holdout | RMSE holdout | MAE ventana móvil | RMSE ventana móvil |
|---|---|---|---|---|
| Regresión lineal, histórico completo | 135,961 | 141,891 | 111,779 | 117,118 |
| Regresión lineal, últimos 36 meses | 61,057 | 71,372 | **47,226** | 55,248 |
| Random Forest (`min_samples_leaf=2`) | 104,419 | 127,186 | 84,457 | 103,948 |
| Red neuronal (capas 64-32, 300 épocas) | **47,698** | 68,663 | 58,894 | 78,022 |

### Versiones ajustadas por validación cruzada

| Modelo | MAE holdout | RMSE holdout | MAE ventana móvil | RMSE ventana móvil |
|---|---|---|---|---|
| Regresión lineal, ventana de 48 meses | **45,976** | **55,676** | **40,057** | **47,685** |
| Random Forest (`min_samples_leaf=5`) | 96,231 | 114,068 | 78,246 | 95,718 |

MAE = error absoluto medio, en préstamos por mes (menor es mejor).
RMSE = raíz del error cuadrático medio, que penaliza más los errores grandes.

### Qué ganó y por qué

- **Entre los cuatro modelos pedidos no hay un ganador en las dos
  evaluaciones.** La red neuronal tiene el menor error en el holdout
  (47,698), pero en la ventana móvil la regresión de 36 meses es mejor
  (47,226 frente a 58,894). Con un solo año de prueba no se puede decir cuál
  generaliza mejor.
- **El mejor modelo en ambas evaluaciones es la regresión lineal con
  ventana de 48 meses**, que ajustó la validación cruzada. Supera a la de 36 meses en ambas
  evaluaciones, y a la de todo el histórico por mucho.
- **Por qué la regresión gana:** la serie tiene una tendencia casi lineal en
  los últimos años. Una línea sobre pocos meses recientes la sigue bien, y los
  años 2015 a 2019 (niveles más bajos) y las caídas de 2020 y 2024 no la
  arrastran. La red neuronal tiene miles de parámetros frente a unos 100
  ejemplos de entrenamiento; eso probablemente explica su peor desempeño en
  la ventana móvil, aunque no se probó directamente.
- **Por qué el Random Forest pierde:** no puede predecir valores fuera de los
  que ya vio, y usa el año anterior como referencia, así que va detrás del
  crecimiento. Todos los modelos subestiman la serie, pero el Random Forest
  lo hace más.

## Limitaciones

- **Un solo año de prueba.** El holdout no contiene caídas, así que no mide
  qué tan bien predice un choque como los de 2020 o 2024.
- **Pocos datos.** La serie tiene 140 meses. Con esa cantidad, las diferencias
  entre modelos cercanos (por ejemplo, ventana de 36 frente a 48 meses) pueden
  ser ruido. La validación cruzada solo tiene dos pliegues.
- **Una sola semilla para la red neuronal.** No se midió la variación del
  resultado entre entrenamientos.
- **Las caídas están en el entrenamiento.** Los modelos con histórico completo
  aprenden de esos meses atípicos; no se excluyeron.
- **La descarga falló en este entorno por un error de certificado SSL** al
  usar `requests` de Python. La descarga de datos que se usó es la del
  15 de noviembre de 2026; `descargar_datos.py` no se volvió a ejecutar.
