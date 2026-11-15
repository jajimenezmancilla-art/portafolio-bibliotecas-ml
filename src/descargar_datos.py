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