#%%
import re 
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Cargar el CSV
path = "./tokens_etiquetados/tokens_etiquetados_or_fin1000_dim_15.csv"
save_model = False
save_results = True 

# Obtener nombre del archivo
if save_results or save_model: 
    match = re.search(r"fin(.*?)\.csv", path)
    if match:
        save_name = match.group(1)  
    else:
        save_name = input("No se encontró coincidencia. Ingresá manualmente el nombre para guardar: ")

df = pd.read_csv(path) 


# instancia_id,token_id,token,punt_inicial,punt_final,capitalización,es_inicio_instancia,es_fin_instancia,es_primer_token,es_ultimo_token,i_punt_inicial,i_punt_final,dim_red_...
# Features comunes
#features = ['token_id', 'dim_red_0', 'dim_red_1', 'dim_red_2', 'dim_red_3', 'dim_red_4', 'dim_red_5', 'dim_red_6', 'dim_red_7', 'dim_red_8',
#            'dim_red_9', 'dim_red_10', 'dim_red_11', 'dim_red_12', 'dim_red_13', 'dim_red_14', 'es_inicio_instancia', 'es_fin_instancia',
#            'es_primer_token', 'es_ultimo_token'] 
#X = df[features]

columnas_drop = ['instancia_id', 'token', 'punt_inicial', 'punt_final', 
                'capitalización', 'i_punt_inicial', 'i_punt_final']
X = df.drop(columns=columnas_drop)



def entrenar_y_evaluar(X, y, nombre_modelo):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, digits=2)
    matrix = confusion_matrix(y_test, y_pred)

    print(f"\n--- Resultados para {nombre_modelo} ---")
    print(report)
    print(matrix)

        # Obtener índices originales del conjunto de test
    indices_test = X_test.index
    
    # Crear DataFrame con los errores
    errores_df = pd.DataFrame({
        'indice_original': indices_test,
        'y_true': y_test,
        'y_pred': y_pred
    })
    
    # Filtrar solo los errores (donde y_true != y_pred)
    errores_df = errores_df[errores_df['y_true'] != errores_df['y_pred']]
    
    # Guardar a CSV con el nombre del modelo
    errores_df.to_csv(f'./salidas/errores_{nombre_modelo}.csv', index=False)

    return model, (report, matrix)

#%% Modelos separados
# Modelos separados

### 1. Modelo para capitalización
model_cap, (report_cap, matrix_cap) = entrenar_y_evaluar(X, df['capitalización'], "random_forest_capitalizacion")

### 2. Modelo para puntuación inicial
model_ini, (report_ini, matrix_ini) = entrenar_y_evaluar(X, df['i_punt_inicial'], "random_forest_i_punt_inicial")

### 3. Modelo para puntuación final
model_fin, (report_fin, matrix_fin) = entrenar_y_evaluar(X, df['i_punt_final'], "random_forest_i_punt_final")

if save_model: 
    joblib.dump(model_cap, f"random_forest_capitalizacion_{save_name}.joblib")
    joblib.dump(model_ini, f"random_forest_i_punt_inicial_{save_name}.joblib")
    joblib.dump(model_fin, f"random_forest_i_punt_final_{save_name}.joblib")

    print("\n✅ Modelos entrenados y guardados exitosamente.")

if save_results: 
    # Guardar reporte
    results_path = f"./salidas/random_forest_{save_name}_reporte.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"--- Resultados para capitalización  ---\n\n")
        f.write("== Classification Report ==\n")
        f.write(report_cap + "\n\n")
        f.write("== Confusion Matrix ==\n")
        f.write(str(matrix_cap))
        f.write(f"--- Resultados para puntuación inicial  ---\n\n")
        f.write("== Classification Report ==\n")
        f.write(report_ini + "\n\n")
        f.write("== Confusion Matrix ==\n")
        f.write(str(matrix_ini))
        f.write(f"--- Resultados para puntuación final  ---\n\n")
        f.write("== Classification Report ==\n")
        f.write(report_fin + "\n\n")
        f.write("== Confusion Matrix ==\n")
        f.write(str(matrix_fin))

    print(f"📝 Reporte guardado como: {results_path}")


'''
### 1. Modelo para capitalización
y_cap = df['capitalización']

X_train_cap, X_test_cap, y_train_cap, y_test_cap = train_test_split(X, y_cap, test_size=0.2, random_state=42)
model_cap = RandomForestClassifier(n_estimators=100, random_state=42)
model_cap.fit(X_train_cap, y_train_cap)
y_pred_cap = model_cap.predict(X_test_cap)
print("\n--- Resultados para capitalización ---")
print(classification_report(y_test_cap, y_pred_cap))
print(confusion_matrix(y_test_cap, y_pred_cap))

### 2. Modelo para puntuación inicial
y_ini = df['i_punt_inicial']

X_train_ini, X_test_ini, y_train_ini, y_test_ini = train_test_split(X, y_ini, test_size=0.2, random_state=42)
model_ini = RandomForestClassifier(n_estimators=100, random_state=42)
model_ini.fit(X_train_ini, y_train_ini)
y_pred_ini = model_ini.predict(X_test_ini)
print("\n--- Resultados para i_punt_inicial ---")
print(classification_report(y_test_ini, y_pred_ini))
print(confusion_matrix(y_test_ini, y_pred_ini))

### 3. Modelo para puntuación final
y_fin = df['i_punt_final']

X_train_fin, X_test_fin, y_train_fin, y_test_fin = train_test_split(X, y_fin, test_size=0.2, random_state=42)
model_fin = RandomForestClassifier(n_estimators=100, random_state=42)
model_fin.fit(X_train_fin, y_train_fin)
y_pred_fin = model_fin.predict(X_test_fin)
print("\n--- Resultados para i_punt_final ---")
print(classification_report(y_test_fin, y_pred_fin))
print(confusion_matrix(y_test_fin, y_pred_fin))

if save_model: 
    joblib.dump(model_cap, f"random_forest_capitalizacion_{save_name}.joblib")
    joblib.dump(model_ini, f"random_forest_i_punt_inicial_{save_name}.joblib")
    joblib.dump(model_fin, f"random_forest_i_punt_final_{save_name}.joblib")

    print("\n✅ Modelos entrenados y guardados exitosamente.")


'''


# %%
