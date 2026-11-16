import requests
import pandas as pd

url = "https://data.seattle.gov/resource/tmmm-ytt6.json"
parametros = {
    "$limit": 50000,
    "$where": "checkoutyear >= 2022"
}

respuesta = requests.get(url, params=parametros)
respuesta.raise_for_status()

datos = respuesta.json()
df = pd.DataFrame(datos)

df.to_csv("data/checkouts_seattle.csv", index=False)
print("Listo. Filas descargadas:", len(df))
print(df.head())


def descargar_totales_mensuales():
    url = "https://data.seattle.gov/resource/tmmm-ytt6.json"
    parametros = {
        "$select": "checkoutyear, checkoutmonth, sum(checkouts) as total_checkouts",
        "$where": "checkoutyear >= 2015",
        "$group": "checkoutyear, checkoutmonth",
        "$order": "checkoutyear, checkoutmonth",
        "$limit": 500
    }
    respuesta = requests.get(url, params=parametros)
    respuesta.raise_for_status()
    df_totales = pd.DataFrame(respuesta.json())
    df_totales.to_csv("data/totales_mensuales.csv", index=False)
    print("Totales mensuales guardados:", df_totales.shape)
    print(df_totales.head())

if __name__ == "__main__":
    descargar_totales_mensuales()