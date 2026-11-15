# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proyecto

Proyecto de portafolio (en español) que aplica machine learning y minería de datos a datos abiertos de circulación bibliotecaria de la Seattle Public Library: predicción de demanda y minería de patrones. Está en etapa inicial: por ahora solo existe el script de descarga de datos; `notebooks/` está vacío.

El código, nombres de variables, comentarios, mensajes y documentación se escriben en español; mantener esa convención.

## Comandos (Windows)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/descargar_datos.py      # descarga datos a data/checkouts_seattle.csv
jupyter lab                         # análisis exploratorio en notebooks/
```

No hay tests, linter ni build configurados.

## Arquitectura y datos

- [src/descargar_datos.py](src/descargar_datos.py) consulta la API Socrata de data.seattle.gov (dataset `tmmm-ytt6`, "Checkouts by Title") con `$limit=50000` y `$where=checkoutyear >= 2022`, y guarda el resultado como CSV. Usa la ruta relativa `data/...`, así que debe ejecutarse desde la raíz del repo.
- La API Socrata devuelve todos los campos como strings en JSON; convertir tipos (p. ej. `checkouts`, `checkoutyear`, `checkoutmonth`) antes de analizar.
- `data/*.csv` está en `.gitignore`: los datos se regeneran con el script, no se versionan.
- Stack: pandas, numpy, scikit-learn, matplotlib, seaborn, Jupyter. `requirements.txt` es un `pip freeze` completo del venv; si se añade una dependencia, regenerarlo con `pip freeze > requirements.txt`.
