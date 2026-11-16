"""Descarga el dataset abierto Goodbooks-10k (Proyecto B, minería de patrones).

Fuente: https://github.com/zygmuntz/goodbooks-10k
Son 10.000 libros, unos 6 millones de valoraciones de 53.000 usuarios y las
etiquetas (tags) que los lectores les pusieron. Los archivos quedan en data/
con el prefijo goodbooks_; como el resto de data/*.csv, no se versionan.

Debe ejecutarse desde la raíz del repo: python src/descargar_goodbooks.py
"""
import requests

BASE = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/"
ARCHIVOS = ["books.csv", "ratings.csv", "tags.csv", "book_tags.csv"]


def descargar_goodbooks():
    """Descarga los cuatro CSV de Goodbooks-10k a data/goodbooks_<nombre>."""
    for nombre in ARCHIVOS:
        respuesta = requests.get(BASE + nombre, timeout=300)
        respuesta.raise_for_status()
        destino = "data/goodbooks_" + nombre
        with open(destino, "wb") as f:
            f.write(respuesta.content)
        print(f"{destino}: {len(respuesta.content) / 1e6:.1f} MB")


if __name__ == "__main__":
    descargar_goodbooks()
