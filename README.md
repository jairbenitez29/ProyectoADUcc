# 🏥 MediPredict AI - Sistema de Diagnóstico Médico

Sistema inteligente de diagnóstico médico utilizando Machine Learning con dos modelos de clasificación: **Regresión Logística** y **Red Neuronal Artificial**.

## 📋 Características

- ✅ **Predicción Individual**: Diagnóstico para un paciente individual
- ✅ **Predicción por Lotes**: Procesamiento masivo de múltiples pacientes
- ✅ **Métricas Detalladas**: Matriz de confusión, accuracy, precision, recall, F1-score
- ✅ **Visualizaciones Interactivas**: Gráficos con Plotly
- ✅ **Interfaz Moderna**: Diseño UI profesional con Streamlit
- ✅ **Comparación de Modelos**: Evaluación de Regresión Logística vs Red Neuronal

## 🚀 Demo en Vivo

**Aplicación desplegada en Streamlit Cloud:**
👉 [https://your-app.streamlit.app](https://your-app.streamlit.app)

## 📊 Modelos de Machine Learning

### 1. Regresión Logística
- Modelo lineal para clasificación multiclase
- **Accuracy**: 70.6%
- Rápido y eficiente

### 2. Red Neuronal Artificial (MLP)
- Multi-Layer Perceptron con 2 capas ocultas (100, 50 neuronas)
- **Accuracy**: 76.5%
- Mayor capacidad de aprendizaje de patrones complejos

## 🛠️ Instalación y Uso Local

### Requisitos Previos
- Python 3.8+
- pip

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/medipredict-ai.git
cd medipredict-ai
```

### Paso 2: Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Paso 3: Entrenar los Modelos

```bash
python train_models.py
```

Este script:
- Carga el dataset
- Entrena ambos modelos (Regresión Logística y Red Neuronal)
- Guarda los modelos entrenados en `models/`
- Genera métricas de evaluación

### Paso 4: Ejecutar la Aplicación

```bash
streamlit run app.py
```

La aplicación estará disponible en `http://localhost:8501`

## 📁 Estructura del Proyecto

```
medipredict-ai/
│
├── app.py                      # Aplicación principal Streamlit
├── train_models.py             # Script de entrenamiento de modelos
├── dataset.xlsx                # Dataset de entrenamiento
├── requirements.txt            # Dependencias Python
├── README.md                   # Este archivo
│
├── .streamlit/
│   └── config.toml            # Configuración de Streamlit
│
└── models/                     # Modelos entrenados (generados)
    ├── logistic_regression.pkl
    ├── neural_network.pkl
    ├── scaler.pkl
    ├── feature_names.json
    └── metrics.json
```

## ☁️ Deployment en Streamlit Cloud

### Paso 1: Preparar el Repositorio

1. Crea un repositorio en GitHub
2. Asegúrate de incluir todos los archivos necesarios:
   - `app.py`
   - `requirements.txt`
   - `train_models.py`
   - `dataset.xlsx`
   - Carpeta `models/` con los modelos entrenados

### Paso 2: Entrenar Modelos Localmente

⚠️ **IMPORTANTE**: Debes entrenar los modelos localmente antes de subir a GitHub:

```bash
python train_models.py
```

Esto generará la carpeta `models/` con todos los archivos necesarios.

### Paso 3: Subir a GitHub

```bash
git init
git add .
git commit -m "Initial commit - MediPredict AI"
git branch -M main
git remote add origin https://github.com/tu-usuario/medipredict-ai.git
git push -u origin main
```

### Paso 4: Configurar Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Inicia sesión con tu cuenta de GitHub
3. Haz clic en "New app"
4. Selecciona:
   - **Repository**: tu-usuario/medipredict-ai
   - **Branch**: main
   - **Main file path**: app.py
5. Haz clic en "Deploy"

¡Listo! Tu aplicación estará disponible en `https://tu-usuario-medipredict-ai.streamlit.app`

## 📊 Características del Dataset

- **Total de registros**: 81 pacientes
- **Características**: 55 variables
  - Datos demográficos (edad, género, ocupación, origen)
  - Síntomas clínicos (fiebre, dolor de cabeza, mareos, etc.)
  - Resultados de laboratorio (hematocrito, hemoglobina, enzimas, etc.)
- **Variable objetivo**: diagnosis (3 clases)

## 🔧 Uso de la Aplicación

### Predicción Individual

1. Selecciona el modelo en el sidebar
2. Ve a la pestaña "🔍 Predicción Individual"
3. Completa los datos del paciente
4. Haz clic en "Realizar Predicción"
5. Visualiza el diagnóstico y las probabilidades

### Predicción por Lotes

1. Selecciona el modelo en el sidebar
2. Ve a la pestaña "📊 Predicción por Lotes"
3. Carga un archivo CSV o Excel con los datos
4. Haz clic en "Procesar Lote"
5. Descarga los resultados

**Formato del archivo**:
- Debe contener las mismas 55 columnas del dataset de entrenamiento
- Opcionalmente puede incluir la columna `diagnosis` para evaluación

### Métricas de Modelos

1. Ve a la pestaña "📈 Métricas de Modelos"
2. Compara el rendimiento de ambos modelos
3. Visualiza matrices de confusión y reportes de clasificación

## 📦 Dependencias Principales

- `streamlit` - Framework de aplicaciones web
- `pandas` - Manipulación de datos
- `numpy` - Operaciones numéricas
- `scikit-learn` - Modelos de Machine Learning
- `plotly` - Visualizaciones interactivas
- `openpyxl` - Lectura de archivos Excel

## 🎨 Personalización

### Cambiar el Tema

Edita `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#667eea"  # Color principal
backgroundColor = "#ffffff"  # Fondo
secondaryBackgroundColor = "#f0f2f6"  # Fondo secundario
textColor = "#262730"  # Color de texto
```

### Modificar Modelos

Edita `train_models.py` para cambiar hiperparámetros:

```python
# Regresión Logística
log_reg = LogisticRegression(
    max_iter=1000,
    C=1.0,  # Regularización
    solver='lbfgs'
)

# Red Neuronal
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Capas ocultas
    activation='relu',
    learning_rate_init=0.001
)
```

## 📄 Licencia

Este proyecto fue desarrollado como parte del Proyecto Final de Análisis de Datos 2025.

## 👥 Autor

- **Nombre**: [Tu Nombre]
- **Curso**: Análisis de Datos
- **Año**: 2025

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📞 Soporte

Si tienes alguna pregunta o problema:
- Abre un [Issue](https://github.com/tu-usuario/medipredict-ai/issues)
- Contacta al autor

---

**MediPredict AI** - Sistema de Diagnóstico Médico con Machine Learning 🏥✨
