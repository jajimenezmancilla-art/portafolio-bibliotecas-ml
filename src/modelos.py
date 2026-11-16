"""Modelos de predicción del total mensual de préstamos y su evaluación.

Todos los modelos usan la misma interfaz: una función `predictor(train, test)`
que recibe el tramo de entrenamiento y el de prueba (filas de la serie de
`serie_mensual.cargar_serie_mensual`) y devuelve las predicciones para `test`.
Así la evaluación es idéntica para todos.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

HORIZONTE = 12   # meses de prueba del holdout
N_MOVIL = 24     # meses de la evaluación en ventana móvil
VENTANA = 36     # meses usados por la regresión de ventana reciente


def regresion_historico(train, test):
    """Regresión lineal sobre el índice de tiempo, con todo el histórico."""
    reg = LinearRegression().fit(train[["t"]], train["total_checkouts"])
    return reg.predict(test[["t"]])


def regresion_ventana(train, test, ventana=VENTANA):
    """Regresión lineal sobre el índice de tiempo, solo con los últimos `ventana` meses."""
    reciente = train.iloc[-ventana:]
    reg = LinearRegression().fit(reciente[["t"]], reciente["total_checkouts"])
    return reg.predict(test[["t"]])


def random_forest(train, test, min_samples_leaf=2):
    """Random Forest con variables de calendario (`mes`) y el valor del año anterior (`lag12`)."""
    vars_rf = ["mes", "lag12"]
    con_lag = train.dropna(subset=vars_rf)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=min_samples_leaf, random_state=42)
    rf.fit(con_lag[vars_rf], con_lag["total_checkouts"])
    return rf.predict(test[vars_rf])


def evaluar(serie, predictor):
    """Evalúa `predictor` con el holdout (últimos HORIZONTE meses) y con la ventana móvil.

    Devuelve un diccionario con MAE y RMSE de cada evaluación y las
    predicciones del holdout.
    """
    train_h, test_h = serie.iloc[:-HORIZONTE], serie.iloc[-HORIZONTE:]
    pred_h = predictor(train_h, test_h)
    real_h = test_h["total_checkouts"].to_numpy()

    reales_m, pred_m = [], []
    for i in range(len(serie) - N_MOVIL, len(serie)):
        pred_m.append(predictor(serie.iloc[:i], serie.iloc[[i]])[0])
        reales_m.append(serie["total_checkouts"].iloc[i])

    return {
        "MAE holdout": mean_absolute_error(real_h, pred_h),
        "RMSE holdout": root_mean_squared_error(real_h, pred_h),
        "MAE móvil": mean_absolute_error(reales_m, pred_m),
        "RMSE móvil": root_mean_squared_error(reales_m, pred_m),
        "pred_holdout": pred_h,
    }


def validar_series_temporales(serie, predictor, n_folds=2):
    """Validación cruzada para series de tiempo, sin tocar el holdout.

    Cada pliegue usa como prueba una ventana de HORIZONTE meses justo antes
    del holdout, y como entrenamiento todo lo anterior a esa ventana.
    Devuelve el MAE medio de los pliegues. Se usa para elegir hiperparámetros.
    """
    maes = []
    for j in range(1, n_folds + 1):
        fin_train = len(serie) - HORIZONTE * (j + 1)
        train = serie.iloc[:fin_train]
        prueba = serie.iloc[fin_train:fin_train + HORIZONTE]
        pred = predictor(train, prueba)
        maes.append(mean_absolute_error(prueba["total_checkouts"], pred))
    return float(np.mean(maes))


def tabla_comparativa(serie, predictores):
    """Tabla con MAE y RMSE (holdout y ventana móvil) para cada predictor.

    `predictores` es un diccionario {nombre: función predictor}.
    """
    filas = {}
    for nombre, predictor in predictores.items():
        r = evaluar(serie, predictor)
        filas[nombre] = {k: v for k, v in r.items() if k != "pred_holdout"}
    return pd.DataFrame(filas).T.round(0).astype(int)
