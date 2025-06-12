import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
#import joblib

# Cargar el CSV
df = pd.read_csv("tokens_etiquetados_or_fin1000_dim_15.csv") 

# Features comunes
features = ['token_id', 'dim_red_0', 'dim_red_1', 'dim_red_2', 'dim_red_3', 'dim_red_4', 'dim_red_5', 'dim_red_6', 'dim_red_7', 'dim_red_8',
            'dim_red_9', 'dim_red_10', 'dim_red_11', 'dim_red_12', 'dim_red_13', 'dim_red_14', 'es_inicio_instancia', 'es_fin_instancia',
            'es_primer_token', 'es_ultimo_token'] 
X = df[features]

### 1. Modelo para capitalización
y_cap = df['capitalización']

X_train_cap, X_test_cap, y_train_cap, y_test_cap = train_test_split(X, y_cap, test_size=0.2, random_state=42)
model_cap = RandomForestClassifier(n_estimators=100, random_state=42)
model_cap.fit(X_train_cap, y_train_cap)
y_pred_cap = model_cap.predict(X_test_cap)
print("\n--- Resultados para capitalización ---")
print(classification_report(y_test_cap, y_pred_cap))
print(confusion_matrix(y_test_cap, y_pred_cap))
#joblib.dump(model_cap, "random_forest_capitalizacion.joblib")

### 2. Modelo para puntuación inicial
y_ini = df['i_punt_inicial']

X_train_ini, X_test_ini, y_train_ini, y_test_ini = train_test_split(X, y_ini, test_size=0.2, random_state=42)
model_ini = RandomForestClassifier(n_estimators=100, random_state=42)
model_ini.fit(X_train_ini, y_train_ini)
y_pred_ini = model_ini.predict(X_test_ini)
print("\n--- Resultados para i_punt_inicial ---")
print(classification_report(y_test_ini, y_pred_ini))
print(confusion_matrix(y_test_ini, y_pred_ini))
#joblib.dump(model_ini, "random_forest_i_punt_inicial.joblib")

### 3. Modelo para puntuación final
y_fin = df['i_punt_final']

X_train_fin, X_test_fin, y_train_fin, y_test_fin = train_test_split(X, y_fin, test_size=0.2, random_state=42)
model_fin = RandomForestClassifier(n_estimators=100, random_state=42)
model_fin.fit(X_train_fin, y_train_fin)
y_pred_fin = model_fin.predict(X_test_fin)
print("\n--- Resultados para i_punt_final ---")
print(classification_report(y_test_fin, y_pred_fin))
print(confusion_matrix(y_test_fin, y_pred_fin))
#joblib.dump(model_fin, "random_forest_i_punt_final.joblib")

print("\n✅ Modelos entrenados y guardados exitosamente.")

