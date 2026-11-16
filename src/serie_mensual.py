"""Carga y preparación de la serie mensual de préstamos totales.

Lo usan los notebooks en notebooks/; se importa desde allí con sys.path.
"""
import numpy as np
import pandas as pd


def cargar_serie_mensual(ruta="data/totales_mensuales.csv"):
    """Lee totales_mensuales.csv y devuelve la serie ordenada por fecha.

    Columnas que agrega:
    - fecha: primer día del mes (datetime).
    - t: índice de tiempo 0..n-1 (para la regresión lineal).
    - mes: número de mes (1-12), usado como variable de calendario.
    - lag12: préstamos del mismo mes del año anterior (NaN en los primeros 12 meses).
    """
    df = pd.read_csv(ruta)
    df["fecha"] = pd.to_datetime(
        df["checkoutyear"].astype(str) + "-" + df["checkoutmonth"].astype(str),
        format="%Y-%m",
    )
    df = df.sort_values("fecha").reset_index(drop=True)
    df["t"] = np.arange(len(df))
    df["mes"] = df["fecha"].dt.month
    df["lag12"] = df["total_checkouts"].shift(12)
    return df


def marcar_anomalias(df, ventana=13, umbral=-0.20):
    """Marca los meses cuyo total queda más de `umbral` por debajo de la mediana móvil centrada.

    Se usa la mediana (y no la media) para que los propios meses anómalos
    no arrastren la referencia.
    """
    df = df.copy()
    mediana = df["total_checkouts"].rolling(
        ventana, center=True, min_periods=ventana // 2 + 1
    ).median()
    df["desv_vs_mediana"] = df["total_checkouts"] / mediana - 1
    df["anomalia"] = df["desv_vs_mediana"] < umbral
    return df
