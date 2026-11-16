# Predicción de demanda y minería de patrones en bibliotecas

Proyecto de portafolio que aplica machine learning y minería de datos
a datos abiertos de circulación bibliotecaria (Seattle Public Library).
Tiene dos partes:
- **Proyecto A:** predice el total mensual de préstamos de la biblioteca de Seattle y detecta caídas anómalas en la serie.
- **Proyecto B:** busca patrones de lectura (reglas de asociación, agrupamiento y árbol de decisión) en el dataset abierto Goodbooks-10k.

## Cómo correrlo
1. `python -m venv venv`
2. `venv\Scripts\activate` (Windows)
3. `pip install -r requirements.txt`
4. `python src/descargar_datos.py` (genera `data/checkouts_seattle.csv` y `data/totales_mensuales.csv`)
5. `jupyter lab`, y ejecutar en orden:
   - `notebooks/01_explorar_datos.ipynb`: serie, anomalías y tres modelos base.
   - `notebooks/02_modelos_y_red_neuronal.ipynb`: red neuronal, ajuste de hiperparámetros y tabla final.
6. Para el Proyecto B: `python src/descargar_goodbooks.py` (descarga unos 93 MB a `data/goodbooks_*.csv`) y luego ejecutar `notebooks/03_mineria_patrones.ipynb`.

## Estructura
- `src/descargar_datos.py`: descarga desde la API Socrata de data.seattle.gov.
- `src/serie_mensual.py`: carga de la serie mensual y detección de anomalías.
- `src/modelos.py`: los modelos base, la evaluación (holdout y ventana móvil) y la validación cruzada.
- `src/red_neuronal.py`: red densa en Keras.
- `src/descargar_goodbooks.py`: descarga Goodbooks-10k desde GitHub (Proyecto B).
- `src/mineria.py`: géneros por libro, cestas, Apriori propio, reglas entre libros, vectores por lectores y atributos para el árbol (Proyecto B).

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
- **Los datos son una foto del 15 de noviembre de 2026.** El dataset de Seattle
  se actualiza cada mes, así que volver a ejecutar `descargar_datos.py` puede
  cambiar las cifras. (El error de certificado SSL que impedía la descarga con
  `requests` se resolvió con `pip_system_certs`, incluido en `requirements.txt`.)

## Proyecto B: minería de patrones de lectura

Datos: [Goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k), con 10.000
libros, unos 6 millones de valoraciones de 53.424 usuarios y las etiquetas que
los lectores les pusieron. Todo el análisis está en
`notebooks/03_mineria_patrones.ipynb`.

Goodbooks no trae una clasificación oficial, así que el primer paso es de control
de vocabulario: 120 etiquetas libres (`sci-fi`, `scifi`, `science-fiction`...) se
agrupan a mano en 16 géneros, y cada libro recibe como género principal el que
concentra más usos de sus etiquetas.

### Hallazgo 3: qué se lee junto

Reglas de asociación con un Apriori escrito a mano (da los mismos 78 conjuntos
frecuentes que la librería `mlxtend`; el notebook repite esa comprobación solo si
`mlxtend` está instalada, y no está en `requirements.txt`). La cesta de cada usuario son los géneros
que pesan 10% o más entre los libros que valoró con 4 o 5 estrellas.

| Regla | Soporte | Confianza | Lift |
|---|---|---|---|
| ficción literaria → ficción histórica | 10,9% | 63,4% | 2,22 |
| paranormal → juvenil | 6,7% | 55,8% | 1,88 |
| terror → misterio y thriller | 3,5% | 63,2% | 1,83 |
| ciencia ficción → fantasía | 10,0% | 68,1% | 1,52 |
| misterio y thriller → fantasía | 11,8% | 34,1% | 0,76 |

Lift mayor que 1: aparecen juntos más de lo esperable por azar. Menor que 1: sus
públicos se cruzan poco.

A nivel de títulos (pares que comparte al menos el 2% de los usuarios), los pares
de mayor lift son tomos de una misma saga. Entre autores distintos aparecen
afinidades como *El nombre del viento* con *El imperio final* (lift 7,6) o *Dune*
con *Fundación* (lift 5,7). De los 200 pares de autores distintos con más lift,
el 65,5% comparte género principal; el resto cruza géneros, como *Big Little Lies*
(thriller) con *El ruiseñor* (ficción histórica).

### Hallazgo 4: los lectores recuperan buena parte de la clasificación por género

Cada libro se representa solo por quiénes lo leyeron (matriz libros × usuarios
reducida a 50 dimensiones) y se agrupa con K-means. Las etiquetas no intervienen.

| Grupos (k) | Pureza | ARI | NMI | Silueta |
|---|---|---|---|---|
| 8 | 0,482 | 0,275 | 0,435 | 0,133 |
| 16 | 0,553 | 0,286 | 0,480 | 0,174 |
| 30 | 0,633 | 0,238 | 0,489 | 0,231 |

Con 16 grupos, el 55% de los libros cae en un grupo donde su género es el
mayoritario (asignar a todos el género más frecuente daría 17%). Cinco grupos
son casi puros: no ficción de divulgación (98%), manga (97%), infantil ilustrado
(96%), cómics (94%) y romance (93%). Otros mezclan géneros con un mismo público, como el de
thriller y ficción histórica reciente. La silueta es baja: los grupos no están
bien separados y k = 16 se eligió para comparar con los 16 géneros, no por ser óptimo.

### Hallazgo 5: el género casi no se puede predecir sin información de contenido

Un árbol de decisión que usa año de publicación, nota media, popularidad y
distribución de estrellas acierta el género del **34,2%** de los libros de prueba
(línea base: 16,8%; F1 macro: 0,27). Solo reconoce bien los clásicos (F1 0,65),
por el año de publicación. Humor y terror no los acierta nunca.

### Limitaciones del Proyecto B

- **La clasificación de referencia es propia.** Los 16 géneros salen de un
  diccionario hecho a mano sobre etiquetas de lectores; no es Dewey ni LCC y no
  está validada. Hay asignaciones discutibles (*El alquimista* queda en no ficción).
- **Sesgo del dataset.** Son los 10.000 libros más populares de Goodreads, casi
  todos en inglés, y usuarios muy activos. No representa a los usuarios de una biblioteca.
- **Valoraciones, no préstamos.** El Proyecto A usa préstamos reales de Seattle;
  este usa valoraciones porque los datos de Seattle no identifican al usuario.
- **Umbrales elegidos a mano** (10% para la cesta, soportes de 3% y 2%, k = 16):
  no se midió cuánto cambian los resultados al moverlos.
- **La pureza del agrupamiento y la exactitud del árbol no son comparables:** la
  primera se mide sobre los mismos libros agrupados y la segunda sobre un conjunto de prueba.

