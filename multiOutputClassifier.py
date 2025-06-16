#%% 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re 
import joblib
import pandas as pd

# Cargar el CSV
path = "./tokens_etiquetados/tokens_etiquetados_or_fin1000_dim_152.csv"
save_model = False
save_results = True 

# Obtener nombre del archivo
if save_results or save_model: 
    match = re.search(r"fin(.*?)\.csv", path)
    if match:
        save_name = match.group(1)  
    else:
        save_name = input("No se encontró coincidencia. Ingresá manualmente el nombre para guardar: ")

df_token = pd.read_csv(path) 

# Tus features
columnas_drop = ['instancia_id', 'idx_palabra', 'token', 'punt_inicial', 'punt_final', 
                'capitalización', 'i_punt_inicial', 'i_punt_final']
X = df_token.drop(columns=columnas_drop)

# Todas las etiquetas que querés predecir juntas:
Y = df_token[['capitalización', 'i_punt_inicial', 'i_punt_final']]

# División train-test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#%% Modelo multitarea
# Modelo multitarea

model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

#%%

# Predecir todas las columnas a la vez
y_pred = model.predict(X_test)

# Evaluar por separado cada tarea
for i, col in enumerate(Y.columns):
    print(f"\n--- Resultados para {col} ---")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))

#%% Guardar todos los errores

# Crear lista para guardar todos los errores
errores_lista = []

# Iterar por cada tarea
for i, col in enumerate(Y.columns):
    y_true_col = y_test.iloc[:, i]
    y_pred_col = y_pred[:, i]
    
    for idx, true_val, pred_val in zip(y_test.index, y_true_col, y_pred_col):
        if true_val != pred_val:
            errores_lista.append({
                'indice_original': idx,
                'tarea': col,
                'y_true': true_val,
                'y_pred': pred_val, 
                'idx_palabra': df_token.loc[idx, 'idx_palabra'],
                'token': df_token.loc[idx, 'token']
            })

# Convertir a DataFrame
errores_df = pd.DataFrame(errores_lista)

# Guardar CSV
errores_df.to_csv('./salidas/errores_MultiOutputClassifier.csv', index=False)

# %% Observar errores
# Observar errores

# Cargar datos
nombre_csv_original = './datasets/oraciones_final.csv'
df = pd.read_csv(nombre_csv_original) 
errores_df = pd.read_csv('./salidas/errores_MultiOutputClassifier.csv')

# Crear estructura para almacenar errores por instancia_id
errores_por_instancia = {}

# Agrupar errores por instancia_id
for _, error_row in errores_df.iterrows():
    idx_original = error_row['indice_original'] 
    tarea = error_row['tarea']
    y_true = error_row['y_true']
    y_pred = error_row['y_pred']
    idx_palabra = error_row['idx_palabra']
    token = error_row['token']
    
    # Obtener el instancia_id correspondiente
    instancia_id = df_token.loc[idx_original, 'instancia_id'] - 1 

    if instancia_id not in errores_por_instancia:
        errores_por_instancia[instancia_id] = []

    errores_por_instancia[instancia_id].append({
        'tarea': tarea,
        'y_true': y_true,
        'y_pred': y_pred, 
        'idx_palabra': idx_palabra, 
        'token': token        
    })

# Recorremos cada instancia con errores
punct_to_index = {"": 0, "¿": 1, "?": 2, ".": 3, ",": 4}
index_to_punct = {0: '', 1: '¿', 2: '?', 3: '.', 4: ','}

#%%

for instancia_id, errores in errores_por_instancia.items():
    print(f"\n🔹 Instancia ID: {instancia_id}")
    
    # Obtener la oración original del dataset original
    oracion_original = df.loc[instancia_id]['texto'] 

    print(f"📝 Oración original: {oracion_original}")
    
    # Inicializar lista para reconstruir oración predicha palabra por palabra
    palabras = oracion_original.split()  # suponemos separación por espacios
    palabras_predichas = palabras.copy()

    # Insertar predicciones erróneas en su posición correspondiente
    for error in errores:
        idx_palabra = int(error['idx_palabra'])
        y_pred = error['y_pred']
        tarea = error['tarea'] 

        if tarea == 'capitalización': 
            if y_pred == 0:  # minúscula
                palabras_predichas[idx_palabra] = palabras_predichas[idx_palabra].lower()
            elif y_pred == 1:  # capitalizada
                palabra = palabras_predichas[idx_palabra]
                palabras_predichas[idx_palabra] = palabra[0].upper() + palabra[1:].lower()
            elif y_pred == 2:  # mixto
                palabras_predichas[idx_palabra] = f"[{palabras[idx_palabra]} → AlgunasMayúsculas]"
            else:  # mayúscula
                palabras_predichas[idx_palabra] = palabras_predichas[idx_palabra].upper()


        elif tarea == 'i_punt_final':
            palabras_predichas[idx_palabra] += index_to_punct.get(int(y_pred), '')
        
        else:  # i_punt_inicio u otras
            palabras_predichas[idx_palabra] = index_to_punct.get(int(y_pred), '') + palabras_predichas[idx_palabra]

    # Reconstruir oración con anotaciones
    oracion_anotada = ' '.join(palabras_predichas)
    print(f"🔍 Oración con errores señalados: {oracion_anotada}")

    # Opcional: también podrías imprimir cada error detallado
    for error in errores:
        print(f"  ⚠️ Error en tarea: {error['tarea']} | palabra idx {error['idx_palabra']} | y_true: {error['y_true']} | y_pred: {error['y_pred']}")

# %%
