import pandas as pd
import numpy as np
import umap.umap_ as umap
import re
from transformers import BertTokenizer, BertModel
import torch


# MODELOS (TOKENIZADOR Y EMBEDDER)
model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

# FUNCION PARA OBTENER EMBEDDING A PARTIR DE TOKEN
def get_multilingual_token_embedding(token_id):
    #token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
	    #print(f"El token '{token}' no pertenece al vocabulario de multilingual BERT.")
        return None
    embedding_vector = model.embeddings.word_embeddings.weight[token_id]
    #print(f"Token: '{token}' | ID: {token_id}")
    #print(f"Embedding shape: {embedding_vector.shape}")
    return embedding_vector

def procesar_texto(texto, instancia_id, agregar_emb=False):

    # Arma una lista con las palabras o puntuaciones (¿?.,) junto con los indices de inicio y fin en el texto
    matches = list(re.finditer(r"\w+('\w+)?|[¿?,\.]", texto))

    # Extrae posiciones de puntuaciones y palabras a partir de "matches"
    # Emprolija la lista anterior y divide en dos, una para palabras y otra para puntucaiones
    # pos_puntuacion = [('puntuacion', indice_inicial, indice_final), ...]
    # pos_palabras = [('palabra', indice_inicial, indice_final), ...]
    pos_puntuacion = []
    pos_palabras = []
    for m in matches:
        tok = m.group()
        start, end = m.start(), m.end()
        if tok in "¿?,.":
            pos_puntuacion.append((tok, start, end))
        else:
            pos_palabras.append((tok, start, end))

    # Unimos pos_puntuacion y pos_palabras
    # Arma una lista de tuplas, donde en cada tupla se encuentra una palabra del texto junto con
    # los indices y su puntuacion inicial y final.
    # puntuacion_palabra = [('palabra', indice_inicial, indice_final, 'punt_inicial', 'punt_final'), ...]
    puntuacion_palabra = []
    for i, (palabra, start, end) in enumerate(pos_palabras):
        punt_ini = ""
        punt_fin = ""
        for p, p_start, p_end in pos_puntuacion:
            # Puntuación inicial: ¿ justo antes del token
            if p_end == start and p == "¿":
                punt_ini = "¿"
            # Puntuación final: ? o . justo después del token
            elif p_start == end and p in ".?":
                punt_fin = p
            # Coma como puntuación final si está justo después
            elif p_start == end and p == ",":
                punt_fin = ","
        puntuacion_palabra.append((palabra, start, end, punt_ini, punt_fin))

    # Armamos las filas del csv etiquetando cada token
    filas = []

    for palabra, start, end, punt_ini, punt_fin in puntuacion_palabra:
        # Capitalización
        if palabra.islower():
            cap = 0
        elif palabra.isupper():
            cap = 3
        elif palabra[0].isupper() and palabra[1:].islower():
            cap = 1
        else:
            cap = 2

        # Tokenización
        encoding = tokenizer(palabra.lower(), add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
        token_ids = encoding["input_ids"]
        token_texts = tokenizer.convert_ids_to_tokens(token_ids)

        # Armamos el diccionario de etiquetas para cada token ("filas")
        for i, (tid, tok) in enumerate(zip(token_ids, token_texts)):
            filas.append({
                "instancia_id": instancia_id,
                "token_id": tid,
                "token": tok,
                "punt_inicial": punt_ini if i == 0 else "",
                "punt_final": punt_fin if i == len(token_ids) - 1 else "",
                "capitalización": cap
            })

    # Agregar índice de oración (distancia desde la última puntuación final)
    indice_oracion = 0
    for fila in filas:
        fila["indice_oracion"] = indice_oracion
        if fila["punt_final"] in [".", "?"]:
            indice_oracion = 0  # reinicia después de puntuación final
        else:
            indice_oracion += 1

    # Agregar columnas i_punt_inicial, i_punt_final e i_puntuacion
    # Estas columnas seran usadas como etiquetas para entrenar el modelo
    # En el caso de predecir puntuacion inicial y final por separado, se usan
    # i_punt_inicial y i_punt_final
    # Si se predicen con una misma etiqueta, se usa i_puntuacion

    # Mapeo de puntuaciones a índices
    punct_to_index = {"": 0, "¿": 1, "?": 2, ".": 3, ",": 4}
    for fila in filas:
        fila["i_punt_inicial"] = punct_to_index.get(fila["punt_inicial"], 0) 
        fila["i_punt_final"] = punct_to_index.get(fila["punt_final"], 0) 
        fila["i_puntuacion"] = punct_to_index.get(fila["punt_inicial"], 0) + punct_to_index.get(fila["punt_final"], 0)

    # AGREGAR EMBEDDING
    if agregar_emb:
        # Calculamos embeddings de cada token
        embeddings = []
        for fila in filas:
            tensor = get_multilingual_token_embedding(fila["token_id"])
            embedding = tensor.detach().numpy()
            embeddings.append(embedding)

        # Agregar el embedding con dimension original (768)
        for i, fila in enumerate(filas):
            fila["embedding"] = embeddings[i]

    return filas


# Leer el CSV (debe tener columna "texto")
df_entrada = pd.read_csv("dataset/textos.csv") 

# Lista donde acumularemos todas las instancias
todas_las_instancias = []

instancia_id = 1
# Iterar por cada texto (fila/instancia) en el CSV
for idx, fila in df_entrada.iterrows():
    texto = fila["texto"]
    instancia_id = idx + 1 

    filas = procesar_texto(texto, instancia_id, agregar_emb=True)

    # Agregar al acumulador
    todas_las_instancias.extend(filas)


# Agregamos embeddings con dimension reducida 
agregar_emb_red = True
dim = 5
if agregar_emb_red == True:
    # Aplicar UMAP sobre los embedding de todos los textos
    # np.vstak apila todos los embeddings (arma matriz)
    embeddings = [instancia["embedding"] for instancia in todas_las_instancias]
    X = np.vstack(embeddings)
    umap_model = umap.UMAP(n_components=dim, random_state=42)
    X_umap = umap_model.fit_transform(X)
    for i, fila in enumerate(todas_las_instancias):
        fila["embedding_red"] = X_umap[i]

df_final = pd.DataFrame(todas_las_instancias)

# Guardar como CSV
df_final.to_csv("tokens_etiquetados.csv", index=False)

print(df_final)
