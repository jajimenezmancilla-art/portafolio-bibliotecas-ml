# Predicción de demanda y minería de patrones en bibliotecas

Proyecto de portafolio que aplica machine learning y minería de datos
a datos abiertos de circulación bibliotecaria (Seattle Public Library).

## Cómo correrlo
1. `python -m venv venv`
2. `venv\Scripts\activate` (Windows)
3. `pip install -r requirements.txt`
4. `python src/descargar_datos.py`
5. `jupyter lab` y ejecutar `notebooks/01_explorar_datos.ipynb`

## Hallazgo 1: dos caídas anómalas en la serie mensual

La serie de préstamos totales por mes (2015 a agosto 2026) es estable, con
una tendencia al alza, salvo dos episodios claros. Un mes se marca como
anómalo si queda más de un 20% por debajo de la mediana móvil de 13 meses.

| Episodio | Meses marcados | Caída máxima | Causa |
|---|---|---|---|
| Pandemia de COVID-19 | abril a julio 2020 | −26% (abril) | Cierre de las sedes desde mediados de marzo de 2020, con servicio limitado durante meses ([informe de levy de la biblioteca](https://www.spl.org/Seattle-Public-Library/documents/about-us/levy/reports/2020_Q3_Library-Levy-Report.pdf)) |
| Ransomware | junio y julio 2024 | −48% (junio) | Ataque de ransomware detectado el 25 de mayo de 2024. Dejó fuera de servicio los catálogos, préstamos y computadoras públicas durante semanas ([Washington State Standard](https://washingtonstatestandard.com/2024/05/28/ransomware-attack-knocks-seattle-public-librarys-computer-system-offline/), [KUOW](https://www.kuow.org/governing/2024-05-29/hoping-to-use-the-computer-at-a-seattle-library-think-again)) |

Ambas caídas tienen una causa externa identificada, así que se tratan como
eventos y no como ruido. El ataque de 2024 también explica la caída mayor de
mayo (−18%, por debajo del umbral), que empezó a finales de ese mes.

## Hallazgo 2: comparación de tres modelos

Los tres modelos predicen el total mensual de préstamos. Se evalúan de dos
formas: un **holdout** de los últimos 12 meses (sep 2025 a ago 2026), y una
**ventana móvil** de los últimos 24 meses, re-entrenando antes de cada mes.

| Modelo | MAE (holdout 12 m) | RMSE (holdout 12 m) | MAE (ventana móvil 24 m) | RMSE (ventana móvil 24 m) |
|---|---|---|---|---|
| Regresión lineal, histórico completo | 135,961 | 141,891 | 111,779 | 117,118 |
| **Regresión lineal, últimos 36 meses** | **61,057** | **71,372** | **47,226** | **55,248** |
| Random Forest (mes y valor del año anterior) | 104,419 | 127,186 | 84,457 | 103,948 |

MAE = error absoluto medio (en préstamos por mes; menor es mejor).
RMSE = raíz del error cuadrático medio (penaliza más los errores grandes).

**Ganador: la regresión lineal con solo los últimos 36 meses.** Gana en las
dos evaluaciones, con un MAE de 61 mil préstamos frente a 104 mil del Random
Forest y 136 mil de la regresión con todo el histórico.

Por qué: la regresión con todo el histórico intenta trazar una sola línea
sobre 11 años. Los años 2015 a 2019 tienen niveles más bajos, y las caídas de
2020 y 2024 tiran de la línea hacia abajo, así que subestima el nivel actual.
La ventana de 36 meses solo ve el nivel y la tendencia recientes. El Random
Forest no puede predecir valores fuera de los que ya vio, y usa el año
anterior como referencia, así que va detrás del crecimiento.

Limitaciones: el holdout cubre un año sin caídas, así que no mide qué pasa
en un choque como los de 2020 o 2024. La serie tiene 140 meses, pocos para
modelos más complejos.
