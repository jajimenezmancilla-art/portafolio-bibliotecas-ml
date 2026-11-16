"""Minería de patrones sobre Goodbooks-10k (Proyecto B).

Lo usa notebooks/03_mineria_patrones.ipynb; se importa con sys.path.
Contiene: la asignación de género a cada libro a partir de sus etiquetas,
las cestas de lectura, un Apriori propio para reglas de asociación,
reglas entre pares de libros, el agrupamiento por lectores y el árbol
de decisión.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse

# Cada género agrupa las etiquetas (tags) de los lectores que significan lo mismo.
# Se dejan fuera las etiquetas mixtas (p. ej. "sci-fi-fantasy") y la genérica "fiction".
GENEROS = {
    "fantasía": ["fantasy", "high-fantasy", "epic-fantasy", "fantasy-fiction", "magic"],
    "ciencia ficción": ["science-fiction", "sci-fi", "scifi", "sf"],
    "juvenil": ["young-adult", "ya", "ya-fiction", "young-adult-fiction", "teen",
                "ya-books", "ya-lit", "teen-fiction", "youngadult"],
    "clásicos": ["classics", "classic", "classic-literature", "classics-to-read",
                 "classic-fiction", "classic-lit", "modern-classics"],
    "romance": ["romance", "contemporary-romance", "adult-romance", "chick-lit",
                "chicklit", "chic-lit", "erotica", "new-adult"],
    "misterio y thriller": ["mystery", "thriller", "suspense", "crime", "mysteries",
                            "thrillers", "mystery-thriller", "mystery-suspense",
                            "crime-fiction", "detective", "mystery-crime",
                            "crime-mystery", "suspense-thriller", "crime-thriller",
                            "murder-mystery", "thriller-mystery", "thriller-suspense",
                            "mystery-thrillers"],
    "no ficción": ["non-fiction", "nonfiction", "non-fic", "history", "biography",
                   "memoir", "memoirs", "biographies", "autobiography",
                   "biography-memoir", "memoir-biography", "science", "psychology",
                   "self-help", "self-improvement", "philosophy", "religion",
                   "spirituality", "business", "politics", "essays"],
    "ficción histórica": ["historical-fiction", "historical", "fiction-historical"],
    "paranormal": ["paranormal", "urban-fantasy", "vampires", "vampire",
                   "paranormal-romance", "supernatural", "pnr", "paranormal-fantasy"],
    "terror": ["horror"],
    "infantil": ["childrens", "children", "children-s", "kids", "children-s-books",
                 "childrens-books", "kids-books", "juvenile", "middle-grade",
                 "picture-books", "children-s-literature", "children-s-lit",
                 "childrens-lit", "kid-lit", "children-books", "kid-books",
                 "chapter-books"],
    "cómics": ["graphic-novels", "graphic-novel", "comics", "comic", "manga", "graphic"],
    "distopía": ["dystopian", "dystopia"],
    "humor": ["humor", "humour", "funny", "comedy", "humorous"],
    "ficción literaria": ["literary-fiction", "literary", "literature",
                          "contemporary-fiction", "general-fiction"],
    "poesía": ["poetry"],
}


def cargar_goodbooks(carpeta="data"):
    """Lee los cuatro CSV de Goodbooks-10k y los devuelve en un diccionario."""
    return {
        "libros": pd.read_csv(f"{carpeta}/goodbooks_books.csv"),
        "valoraciones": pd.read_csv(f"{carpeta}/goodbooks_ratings.csv"),
        "etiquetas": pd.read_csv(f"{carpeta}/goodbooks_tags.csv"),
        "etiquetas_libro": pd.read_csv(f"{carpeta}/goodbooks_book_tags.csv"),
    }


def generos_por_libro(datos, umbral=0.15):
    """Asigna géneros a cada libro según el peso de sus etiquetas.

    Devuelve una tabla indexada por book_id con una columna por género
    (proporción de los usos de etiquetas de género que caen en ese género),
    más `genero_principal` (el de mayor proporción) y `generos` (lista de
    los que alcanzan `umbral`). Los libros sin etiquetas de género se omiten.
    """
    tag_a_genero = {tag: g for g, tags in GENEROS.items() for tag in tags}
    et = datos["etiquetas_libro"].merge(datos["etiquetas"], on="tag_id")
    et = et[et["count"] > 0]
    et["genero"] = et["tag_name"].map(tag_a_genero)
    et = et.dropna(subset=["genero"])
    usos = et.pivot_table(index="goodreads_book_id", columns="genero",
                          values="count", aggfunc="sum", fill_value=0)
    prop = usos.div(usos.sum(axis=1), axis=0)
    ids = datos["libros"].set_index("goodreads_book_id")["book_id"]
    prop.index = prop.index.map(ids)
    prop.index.name = "book_id"
    prop = prop.sort_index()
    columnas = list(prop.columns)
    prop["genero_principal"] = prop[columnas].idxmax(axis=1)
    prop["generos"] = [
        [g for g in columnas if fila[g] >= umbral] for _, fila in prop[columnas].iterrows()
    ]
    return prop


def cestas_de_generos(datos, generos, nota_minima=4, peso_minimo=0.10):
    """Arma la "cesta" de géneros de cada usuario.

    Un género entra en la cesta del usuario si al menos `peso_minimo` de los
    libros que valoró con `nota_minima` o más tienen ese género principal.
    Devuelve una tabla booleana usuarios x géneros.
    """
    gustos = datos["valoraciones"]
    gustos = gustos[gustos["rating"] >= nota_minima]
    gustos = gustos.merge(generos[["genero_principal"]], left_on="book_id", right_index=True)
    conteo = pd.crosstab(gustos["user_id"], gustos["genero_principal"])
    peso = conteo.div(conteo.sum(axis=1), axis=0)
    return peso >= peso_minimo


def apriori(cestas, soporte_minimo=0.05, tamano_maximo=3):
    """Apriori propio: devuelve los conjuntos frecuentes y su soporte.

    `cestas` es una tabla booleana transacciones x ítems. Va por niveles:
    solo arma candidatos de tamaño k con conjuntos frecuentes de tamaño k-1
    (si un conjunto no es frecuente, ningún conjunto que lo contenga lo es).
    """
    matriz = cestas.to_numpy(dtype=bool)
    items = list(cestas.columns)
    frecuentes = {}
    nivel = []
    for i in range(len(items)):
        soporte = matriz[:, i].mean()
        if soporte >= soporte_minimo:
            frecuentes[frozenset([i])] = soporte
            nivel.append(frozenset([i]))
    k = 2
    while nivel and k <= tamano_maximo:
        candidatos = {a | b for a, b in combinations(nivel, 2) if len(a | b) == k}
        nivel = []
        for candidato in candidatos:
            # Poda: todos sus subconjuntos de tamaño k-1 deben ser frecuentes.
            if any(frozenset(s) not in frecuentes for s in combinations(candidato, k - 1)):
                continue
            soporte = matriz[:, sorted(candidato)].all(axis=1).mean()
            if soporte >= soporte_minimo:
                frecuentes[candidato] = soporte
                nivel.append(candidato)
        k += 1
    return {frozenset(items[i] for i in c): s for c, s in frecuentes.items()}


def reglas_de_asociacion(frecuentes, confianza_minima=0.3):
    """Genera las reglas antecedente -> consecuente (un solo ítem) con sus métricas.

    - soporte: proporción de cestas que contienen antecedente y consecuente.
    - confianza: de las cestas con el antecedente, cuántas tienen el consecuente.
    - lift: confianza dividida por la frecuencia del consecuente. Mayor que 1
      significa que aparecen juntos más de lo esperable por azar.
    """
    filas = []
    for conjunto, soporte in frecuentes.items():
        if len(conjunto) < 2:
            continue
        for consecuente in conjunto:
            antecedente = conjunto - {consecuente}
            confianza = soporte / frecuentes[antecedente]
            if confianza < confianza_minima:
                continue
            filas.append({
                "antecedente": " + ".join(sorted(antecedente)),
                "consecuente": consecuente,
                "soporte": soporte,
                "confianza": confianza,
                "lift": confianza / frecuentes[frozenset([consecuente])],
            })
    columnas = ["antecedente", "consecuente", "soporte", "confianza", "lift"]
    return pd.DataFrame(filas, columns=columnas).sort_values("lift", ascending=False).reset_index(drop=True)


def reglas_entre_libros(datos, nota_minima=4, soporte_minimo=0.02):
    """Reglas de asociación entre pares de libros: a quién le gustó A, ¿le gustó B?

    La cesta de cada usuario son los libros que valoró con `nota_minima` o más.
    Solo se consideran pares donde ambos libros y el par alcanzan `soporte_minimo`.
    Devuelve una fila por par (A, B) con soporte, confianza en ambos sentidos,
    lift y si comparten autor.
    """
    gustos = datos["valoraciones"]
    gustos = gustos[gustos["rating"] >= nota_minima]
    n_usuarios = gustos["user_id"].nunique()
    soporte_libro = gustos.groupby("book_id").size() / n_usuarios
    frecuentes = soporte_libro[soporte_libro >= soporte_minimo].index
    gustos = gustos[gustos["book_id"].isin(frecuentes)]
    usuarios = gustos["user_id"].astype("category").cat.codes.to_numpy()
    libros = gustos["book_id"].astype("category")
    matriz = sparse.csr_matrix(
        (np.ones(len(gustos), dtype=np.int32), (usuarios, libros.cat.codes.to_numpy()))
    )
    juntos = (matriz.T @ matriz).toarray() / n_usuarios
    ids = libros.cat.categories.to_numpy()
    soporte = soporte_libro.loc[ids].to_numpy()
    i, j = np.triu_indices(len(ids), k=1)
    pares = pd.DataFrame({
        "libro_a": ids[i], "libro_b": ids[j], "soporte": juntos[i, j],
        "confianza_a_b": juntos[i, j] / soporte[i],
        "confianza_b_a": juntos[i, j] / soporte[j],
        "lift": juntos[i, j] / (soporte[i] * soporte[j]),
    })
    pares = pares[pares["soporte"] >= soporte_minimo]
    info = datos["libros"].set_index("book_id")[["title", "authors"]]
    pares = pares.join(info.add_suffix("_a"), on="libro_a").join(info.add_suffix("_b"), on="libro_b")
    autores_a = pares["authors_a"].str.split(", ").map(set)
    autores_b = pares["authors_b"].str.split(", ").map(set)
    pares["mismo_autor"] = [bool(a & b) for a, b in zip(autores_a, autores_b)]
    return pares.sort_values("lift", ascending=False).reset_index(drop=True)


def vectores_por_lectores(datos, componentes=50, semilla=42):
    """Representa cada libro por quiénes lo leyeron (no por sus etiquetas).

    Arma la matriz libros x usuarios (1 si el usuario valoró el libro), la
    normaliza y la reduce con SVD truncado a `componentes` dimensiones.
    Devuelve (vectores, book_ids).
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    val = datos["valoraciones"]
    libros = val["book_id"].astype("category")
    usuarios = val["user_id"].astype("category")
    matriz = sparse.csr_matrix(
        (np.ones(len(val), dtype=np.float32), (libros.cat.codes.to_numpy(), usuarios.cat.codes.to_numpy()))
    )
    matriz = normalize(matriz)
    vectores = TruncatedSVD(n_components=componentes, random_state=semilla).fit_transform(matriz)
    return normalize(vectores), libros.cat.categories.to_numpy()


def atributos_para_arbol(datos):
    """Atributos del libro que no dependen de las etiquetas, para el árbol de decisión."""
    libros = datos["libros"].set_index("book_id")
    total = libros[["ratings_1", "ratings_2", "ratings_3", "ratings_4", "ratings_5"]].sum(axis=1)
    return pd.DataFrame({
        "año_publicación": libros["original_publication_year"],
        "nota_media": libros["average_rating"],
        "log_valoraciones": np.log10(libros["work_ratings_count"]),
        "log_ediciones": np.log10(libros["books_count"]),
        "reseñas_por_valoración": libros["work_text_reviews_count"] / libros["work_ratings_count"],
        "proporción_5_estrellas": libros["ratings_5"] / total,
        "proporción_1_estrella": libros["ratings_1"] / total,
        "en_inglés": libros["language_code"].fillna("").str.startswith("en").astype(int),
    })
