"""
Script para entrenar y guardar modelos de Machine Learning
Modelos: Regresión Logística y Red Neuronal Artificial
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json

# Cargar dataset
print("Cargando dataset...")
df = pd.read_excel('dataset.xlsx')

# Separar características y variable objetivo
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalizar datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nDataset cargado:")
print(f"- Total registros: {len(df)}")
print(f"- Características: {X.shape[1]}")
print(f"- Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")
print(f"- Clases: {sorted(y.unique())}")

# ===========================
# MODELO 1: REGRESIÓN LOGÍSTICA
# ===========================
print("\n" + "="*50)
print("ENTRENANDO REGRESIÓN LOGÍSTICA")
print("="*50)

log_reg = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
log_reg.fit(X_train_scaled, y_train)

y_pred_lr = log_reg.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

print(f"\nAccuracy: {accuracy_lr:.4f}")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_lr))
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred_lr))

# ===========================
# MODELO 2: RED NEURONAL ARTIFICIAL
# ===========================
print("\n" + "="*50)
print("ENTRENANDO RED NEURONAL ARTIFICIAL")
print("="*50)

mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=1000,
    random_state=42,
    activation='relu',
    solver='adam',
    early_stopping=True,
    validation_fraction=0.2
)
mlp.fit(X_train_scaled, y_train)

y_pred_mlp = mlp.predict(X_test_scaled)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

print(f"\nAccuracy: {accuracy_mlp:.4f}")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_mlp))
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred_mlp))

# ===========================
# GUARDAR MODELOS
# ===========================
print("\n" + "="*50)
print("GUARDANDO MODELOS")
print("="*50)

# Guardar modelos
joblib.dump(log_reg, 'models/logistic_regression.pkl')
joblib.dump(mlp, 'models/neural_network.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Guardar nombres de características
feature_names = X.columns.tolist()
with open('models/feature_names.json', 'w') as f:
    json.dump(feature_names, f)

# Guardar métricas
metrics = {
    'logistic_regression': {
        'accuracy': float(accuracy_lr),
        'confusion_matrix': confusion_matrix(y_test, y_pred_lr).tolist(),
        'classification_report': classification_report(y_test, y_pred_lr, output_dict=True)
    },
    'neural_network': {
        'accuracy': float(accuracy_mlp),
        'confusion_matrix': confusion_matrix(y_test, y_pred_mlp).tolist(),
        'classification_report': classification_report(y_test, y_pred_mlp, output_dict=True)
    }
}

with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n✓ Modelos guardados en ./models/")
print("  - logistic_regression.pkl")
print("  - neural_network.pkl")
print("  - scaler.pkl")
print("  - feature_names.json")
print("  - metrics.json")

print("\n" + "="*50)
print("ENTRENAMIENTO COMPLETADO")
print("="*50)
