# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proyecto

Proyecto de portafolio (en español) que aplica machine learning y minería de datos a datos abiertos de bibliotecas y lectura. Tiene dos partes terminadas, con sus resultados resumidos en [README.md](README.md):

- **Proyecto A (predicción de demanda):** predice el total mensual de préstamos de la Seattle Public Library y detecta caídas anómalas en la serie. Notebooks 01 y 02, y la app de Streamlit ([app.py](app.py)).
- **Proyecto B (minería de patrones):** reglas de asociación, agrupamiento y árbol de decisión sobre el dataset abierto Goodbooks-10k. Notebook 03.

El código, nombres de variables, comentarios, mensajes y documentación se escriben en español; mantener esa convención.

## Comandos (Windows)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/descargar_datos.py       # Proyecto A: data/checkouts_seattle.csv y data/totales_mensuales.csv
python src/descargar_goodbooks.py   # Proyecto B: unos 93 MB en data/goodbooks_*.csv
jupyter lab                         # notebooks 01, 02 y 03, en ese orden
streamlit run app.py                # app del Proyecto A; la primera carga tarda unos 6 minutos
```

Para ejecutar un notebook de principio a fin sin abrir Jupyter:

```powershell
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_mineria_patrones.ipynb
```

No hay tests, linter ni build configurados. La comprobación es volver a ejecutar los notebooks y comparar las cifras con las del README.

## Arquitectura y datos

- Los scripts de `src/` usan la ruta relativa `data/...`, así que se ejecutan desde la raíz del repo. Los notebooks viven en `notebooks/`, añaden `..` a `sys.path` para importar `src.*` y leen de `../data`.
- `data/*.csv` está en `.gitignore`: los datos se regeneran con los scripts de descarga, no se versionan.

Proyecto A:
- [src/descargar_datos.py](src/descargar_datos.py) consulta la API Socrata de data.seattle.gov (dataset `tmmm-ytt6`, "Checkouts by Title"): una muestra de 50.000 registros desde 2022 y los totales mensuales desde 2015. La API devuelve todos los campos como strings; convertir tipos antes de analizar.
- [src/serie_mensual.py](src/serie_mensual.py): carga de la serie mensual y detección de anomalías.
- [src/modelos.py](src/modelos.py): modelos base, evaluación (holdout y ventana móvil) y validación cruzada de series de tiempo.
- [src/red_neuronal.py](src/red_neuronal.py): red densa en Keras.

Proyecto B:
- [src/descargar_goodbooks.py](src/descargar_goodbooks.py) baja los cuatro CSV de Goodbooks-10k desde GitHub.
- [src/mineria.py](src/mineria.py): diccionario `GENEROS` (120 etiquetas agrupadas a mano en 16 géneros), género principal por libro, cestas por usuario, Apriori propio, reglas entre pares de libros, vectores de libros por lectores (SVD) y atributos para el árbol.
- [notebooks/03_mineria_patrones.ipynb](notebooks/03_mineria_patrones.ipynb) compara el Apriori propio con `mlxtend` solo si está instalada; no está en `requirements.txt` y el notebook omite esa celda si falta.

## Entorno

- Stack: pandas 3, numpy, scikit-learn 1.9, scipy, matplotlib, seaborn, TensorFlow/Keras, Streamlit, Jupyter. `requirements.txt` es un `pip freeze` completo del venv; si se añade una dependencia, regenerarlo con `pip freeze > requirements.txt`.
- `pip_system_certs` está instalado para que `requests` use los certificados del sistema; sin él las descargas fallan por SSL en esta máquina.
- Las cifras del README y las "lecturas" en markdown de los notebooks están escritas a mano. Si cambia una versión de librería o los datos, volver a ejecutar y actualizar ambos: el K-means del notebook 03 (k = 16) dio resultados algo distintos entre scikit-learn 1.7 y 1.9.
