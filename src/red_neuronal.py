"""Red neuronal densa (Keras) para el total mensual de préstamos.

Entrada: los 12 valores del año anterior completo (rezagos 12 a 23 meses)
más el mes del año en forma seno/coseno. Con esa entrada, cualquier mes
de un horizonte de 12 meses se predice sin usar valores del tramo de
prueba: son conocidos desde el entrenamiento.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

REZAGO_INICIAL = 12   # primer rezago usado (mismo mes del año anterior)
LARGO_VENTANA = 12    # cuántos rezagos (un año completo)
PRIMER_INDICE = REZAGO_INICIAL + LARGO_VENTANA - 1  # primera fila con ventana completa


def _entrada(valores, meses, i):
    """Vector de entrada para la fila i (usa solo filas anteriores a i - 12)."""
    ventana = valores[i - PRIMER_INDICE:i - REZAGO_INICIAL + 1]
    angulo = 2 * np.pi * meses[i] / 12
    return np.concatenate([ventana, [np.sin(angulo), np.cos(angulo)]])


def crear_modelo(unidades=(32, 16), tasa_aprendizaje=1e-3):
    """Red densa: dos capas ocultas con ReLU y una salida lineal."""
    modelo = keras.Sequential([
        keras.Input(shape=(LARGO_VENTANA + 2,)),
        keras.layers.Dense(unidades[0], activation="relu"),
        keras.layers.Dense(unidades[1], activation="relu"),
        keras.layers.Dense(1),
    ])
    modelo.compile(
        optimizer=keras.optimizers.Adam(tasa_aprendizaje),
        loss="mse",
    )
    return modelo


def red_neuronal(train, test, unidades=(32, 16), epocas=300, semilla=42):
    """Entrena la red con `train` y predice `test`. Mismo contrato que los demás predictores."""
    historia = pd.concat([train, test])
    valores = historia["total_checkouts"].to_numpy(dtype=float)
    meses = historia["mes"].to_numpy()
    n = len(train)

    # Escalar con el máximo del entrenamiento para que la red trabaje en rango ~[0, 1]
    escala = valores[:n].max()
    valores_s = valores / escala

    X = np.array([_entrada(valores_s, meses, i) for i in range(PRIMER_INDICE, n)])
    y = valores_s[PRIMER_INDICE:n]
    X_test = np.array([_entrada(valores_s, meses, i) for i in range(n, n + len(test))])

    tf.keras.utils.set_random_seed(semilla)
    modelo = crear_modelo(unidades)
    modelo.fit(X, y, epochs=epocas, batch_size=16, verbose=0)
    return modelo.predict(X_test, verbose=0).ravel() * escala
