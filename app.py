"""
Sistema de Diagnóstico Médico con Machine Learning
Aplicación Streamlit para predicción individual y por lotes
Proyecto Final - Análisis de Datos 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Configuración de la página
st.set_page_config(
    page_title="SistemaPredict - Diagnóstico Médico ML",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para diseño profesional
st.markdown("""
<style>
    /* Imports */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

    /* Global */
    * {
        font-family: 'Poppins', sans-serif;
    }

    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        text-align: center;
        color: #5a6c7d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Info Box */
    .info-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
    }

    .info-section h3 {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: white;
    }

    .info-section p {
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1rem;
        color: rgba(255, 255, 255, 0.95);
    }

    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-top: 1.5rem;
    }

    .info-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .info-card h4 {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: white;
    }

    .info-card p {
        font-size: 0.9rem;
        margin: 0;
        color: rgba(255, 255, 255, 0.9);
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 1rem;
        opacity: 0.95;
        font-weight: 500;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        font-size: 1rem;
    }

    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }

    /* Success Box */
    .success-box {
        padding: 2rem;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 12px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
    }

    /* Warning Box */
    .warning-box {
        padding: 1.5rem;
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    .warning-box p {
        margin: 0;
        color: #2c3e50;
        font-size: 0.95rem;
    }

    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }

    /* Model Selector */
    .model-selector-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
    }

    /* Feature Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 2rem 0;
    }

    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        text-align: center;
    }

    .feature-icon {
        font-size: 2.5rem;
        color: #667eea;
        margin-bottom: 1rem;
    }

    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }

    .feature-desc {
        font-size: 0.9rem;
        color: #6c757d;
        line-height: 1.5;
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }

    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

# Cargar modelos y datos
@st.cache_resource
def load_models():
    """Cargar modelos entrenados"""
    try:
        models_dir = Path("models")
        return {
            'logistic': joblib.load(models_dir / "logistic_regression.pkl"),
            'neural': joblib.load(models_dir / "neural_network.pkl"),
            'scaler': joblib.load(models_dir / "scaler.pkl")
        }
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None

@st.cache_data
def load_feature_names():
    """Cargar nombres de características"""
    try:
        with open("models/feature_names.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error cargando características: {e}")
        return []

@st.cache_data
def load_metrics():
    """Cargar métricas de los modelos"""
    try:
        with open("models/metrics.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error cargando métricas: {e}")
        return {}

# Mapeo de diagnósticos
DIAGNOSIS_MAP = {
    1: "Diagnóstico Tipo 1",
    2: "Diagnóstico Tipo 2",
    3: "Diagnóstico Tipo 3"
}

# Función para formatear nombres de características
def format_feature_name(name):
    """Convertir nombre de característica a formato legible"""
    return name.replace('_', ' ').title()

# Función para crear gráfico de probabilidades
def plot_probabilities(probabilities, prediction):
    """Crear gráfico de barras de probabilidades"""
    classes = [f"Clase {i+1}" for i in range(len(probabilities))]
    colors = ['#667eea' if i == prediction-1 else '#cbd5e0' for i in range(len(probabilities))]

    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probabilities,
            marker_color=colors,
            text=[f'{p*100:.1f}%' for p in probabilities],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Probabilidades por Clase",
        xaxis_title="Clase de Diagnóstico",
        yaxis_title="Probabilidad",
        yaxis_range=[0, 1],
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins, sans-serif")
    )

    return fig

# Función para crear matriz de confusión
def plot_confusion_matrix(cm, title="Matriz de Confusión"):
    """Crear visualización de matriz de confusión"""
    labels = [f"Clase {i+1}" for i in range(len(cm))]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16, "family": "Poppins"},
        hoverongaps=False,
        showscale=True
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Predicción",
        yaxis_title="Valor Real",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins, sans-serif")
    )

    return fig

# Header
st.markdown('<h1 class="main-header">SistemaPredict</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistema de Diagnóstico Médico Inteligente con Machine Learning</p>', unsafe_allow_html=True)

# Cargar modelos
models = load_models()
feature_names = load_feature_names()
metrics_data = load_metrics()

if not models or not feature_names:
    st.error("Error: No se pudieron cargar los modelos. Asegúrese de ejecutar train_models.py primero.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### Configuración del Sistema")
    st.markdown("---")

    # Selector de página
    page = st.radio(
        "Navegación",
        ["Inicio", "Predicción Individual", "Predicción por Lotes", "Métricas y Evaluación"],
        label_visibility="visible"
    )

    st.markdown("---")

    # Selector de modelo
    st.markdown("### Modelo de Machine Learning")
    model_type = st.selectbox(
        "Seleccionar Modelo",
        ["logistic", "neural"],
        format_func=lambda x: "Regresión Logística" if x == "logistic" else "Red Neuronal Artificial"
    )

    # Información del modelo
    if metrics_data:
        model_metrics = metrics_data.get("logistic_regression" if model_type == "logistic" else "neural_network", {})
        accuracy = model_metrics.get("accuracy", 0)

        st.markdown("---")
        st.metric("Accuracy del Modelo", f"{accuracy*100:.1f}%", delta=None)

        st.markdown("---")
        st.markdown("### Información")
        st.info(
            "**Dataset:** 81 pacientes\n\n"
            "**Características:** 55 variables\n\n"
            "**Clases:** 3 tipos de diagnóstico"
        )

# ==========================================
# PÁGINA: INICIO - DOCUMENTACIÓN
# ==========================================
if page == "Inicio":
    # Sección Principal de Información
    st.markdown("""
    <div class="info-section">
        <h3>¿Qué es SistemaPredict?</h3>
        <p>
            SistemaPredict es una aplicación web de diagnóstico médico asistido por inteligencia artificial,
            desarrollada como parte del Proyecto Final de Análisis de Datos 2025. El sistema utiliza algoritmos
            de Machine Learning para predecir diagnósticos médicos basándose en datos clínicos y de laboratorio
            de pacientes.
        </p>
        <p>
            El sistema permite a profesionales de la salud realizar predicciones tanto individuales como masivas,
            proporcionando herramientas de análisis y visualización para la toma de decisiones informadas.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Características Principales
    st.markdown("## Características Principales")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon"><i class="fas fa-user-md"></i></div>
            <div class="feature-title">Predicción Individual</div>
            <div class="feature-desc">
                Diagnóstico personalizado ingresando datos demográficos, síntomas y resultados de laboratorio
                de un paciente específico.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon"><i class="fas fa-database"></i></div>
            <div class="feature-title">Procesamiento por Lotes</div>
            <div class="feature-desc">
                Análisis masivo de múltiples pacientes mediante carga de archivos CSV/Excel con
                generación automática de reportes.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon"><i class="fas fa-chart-line"></i></div>
            <div class="feature-title">Métricas Detalladas</div>
            <div class="feature-desc">
                Evaluación completa con matrices de confusión, accuracy, precision, recall y F1-score
                para ambos modelos.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Metodología
    st.markdown("## Metodología y Tecnología")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Modelos de Machine Learning")
        st.markdown("""
        **1. Regresión Logística**
        - Modelo lineal para clasificación multiclase
        - Interpretable y eficiente computacionalmente
        - Accuracy: 70.6%
        - Ideal para análisis rápidos

        **2. Red Neuronal Artificial (MLP)**
        - Multi-Layer Perceptron con arquitectura de 2 capas ocultas
        - Capas: 100 y 50 neuronas
        - Accuracy: 76.5%
        - Mayor capacidad de aprendizaje de patrones complejos
        """)

    with col2:
        st.markdown("### Dataset y Características")
        st.markdown("""
        **Datos del Entrenamiento:**
        - 81 pacientes con diagnósticos confirmados
        - 55 características por paciente
        - 3 clases de diagnóstico
        - División: 80% entrenamiento, 20% prueba

        **Variables Incluidas:**
        - Datos demográficos (edad, género, ocupación, origen)
        - Síntomas clínicos (fiebre, dolor, mareos, etc.)
        - Exámenes de laboratorio (hemograma completo, química sanguínea)
        - Días de hospitalización
        """)

    # Cómo Funciona
    st.markdown("## ¿Cómo Funciona el Sistema?")

    st.markdown("""
    <div class="info-grid">
        <div class="info-card">
            <h4>1. Recopilación de Datos</h4>
            <p>
                El sistema recibe información del paciente: datos demográficos, síntomas reportados
                y resultados de exámenes de laboratorio.
            </p>
        </div>
        <div class="info-card">
            <h4>2. Preprocesamiento</h4>
            <p>
                Los datos se normalizan utilizando StandardScaler para asegurar que todas las
                características tengan la misma escala.
            </p>
        </div>
        <div class="info-card">
            <h4>3. Predicción</h4>
            <p>
                El modelo seleccionado (Regresión Logística o Red Neuronal) procesa los datos
                y genera una predicción con probabilidades.
            </p>
        </div>
        <div class="info-card">
            <h4>4. Interpretación</h4>
            <p>
                El sistema presenta el diagnóstico predicho junto con el nivel de confianza
                y las probabilidades para cada clase.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tecnologías Utilizadas
    st.markdown("## Stack Tecnológico")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Frontend & Deployment**
        - Streamlit (Framework web)
        - Plotly (Visualizaciones)
        - HTML/CSS personalizado
        - Streamlit Cloud (Hosting)
        """)

    with col2:
        st.markdown("""
        **Machine Learning**
        - Scikit-learn (Modelos)
        - Pandas (Procesamiento)
        - NumPy (Cálculos numéricos)
        - Joblib (Serialización)
        """)

    with col3:
        st.markdown("""
        **Control de Versiones**
        - Git (Control de versiones)
        - GitHub (Repositorio)
        - Python 3.8+ (Lenguaje base)
        """)

    # Limitaciones y Consideraciones
    st.markdown("## Consideraciones Importantes")

    st.warning("""
    **Nota Importante:** Este sistema es una herramienta de apoyo diagnóstico y NO reemplaza
    el criterio médico profesional. Las predicciones deben ser interpretadas por personal
    de salud calificado y consideradas junto con otros factores clínicos relevantes.
    """)

    st.info("""
    **Alcance del Proyecto:** Sistema desarrollado con fines educativos como parte del curso
    de Análisis de Datos. El modelo fue entrenado con un dataset específico y su aplicación
    en entornos clínicos reales requeriría validación adicional y aprobación regulatoria.
    """)

# ==========================================
# PÁGINA: PREDICCIÓN INDIVIDUAL
# ==========================================
elif page == "Predicción Individual":
    st.markdown('<div class="section-header"><i class="fas fa-user-md"></i> Predicción Individual</div>', unsafe_allow_html=True)
    st.markdown("Ingrese los datos completos del paciente para obtener un diagnóstico predictivo.")

    st.markdown("""
    <div class="warning-box">
        <p><strong>Instrucciones:</strong> Complete todos los campos con los datos del paciente. Los valores por defecto son 0 para campos binarios (Sí/No). Asegúrese de ingresar los datos de exámenes de laboratorio con la precisión adecuada.</p>
    </div>
    """, unsafe_allow_html=True)

    # Crear formulario en columnas
    with st.form("prediction_form"):
        st.markdown("### Datos del Paciente")

        # Organizar campos en 3 columnas
        cols = st.columns(3)
        feature_values = {}

        for idx, feature in enumerate(feature_names):
            col = cols[idx % 3]
            with col:
                # Determinar tipo de input
                if feature in ['male', 'female'] or 'fever' in feature or 'headache' in feature or 'dizziness' in feature:
                    # Campo binario (0 o 1)
                    feature_values[feature] = st.number_input(
                        format_feature_name(feature),
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=1.0,
                        key=feature,
                        help="0 = No, 1 = Sí"
                    )
                else:
                    # Campo numérico general
                    feature_values[feature] = st.number_input(
                        format_feature_name(feature),
                        value=0.0,
                        step=0.01,
                        format="%.2f",
                        key=feature
                    )

        # Botón de predicción
        submitted = st.form_submit_button("Realizar Predicción", use_container_width=True)

        if submitted:
            try:
                # Preparar datos
                X = np.array([[feature_values[feat] for feat in feature_names]])
                X_scaled = models['scaler'].transform(X)

                # Seleccionar modelo
                model = models[model_type]

                # Realizar predicción
                prediction = int(model.predict(X_scaled)[0])
                probabilities = model.predict_proba(X_scaled)[0]
                confidence = max(probabilities)

                # Mostrar resultados
                st.markdown("---")
                st.markdown('<div class="section-header"><i class="fas fa-check-circle"></i> Resultado del Diagnóstico</div>', unsafe_allow_html=True)

                # Resultado principal
                diagnosis = DIAGNOSIS_MAP.get(prediction, "Desconocido")
                st.markdown(f'<div class="success-box">{diagnosis}</div>', unsafe_allow_html=True)

                # Métricas
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Predicción", f"Clase {prediction}")

                with col2:
                    st.metric("Confianza", f"{confidence*100:.1f}%")

                with col3:
                    st.metric("Modelo", "R. Logística" if model_type == "logistic" else "Red Neuronal")

                # Gráfico de probabilidades
                st.plotly_chart(plot_probabilities(probabilities, prediction), use_container_width=True)

                # Tabla de probabilidades
                prob_df = pd.DataFrame({
                    'Clase de Diagnóstico': [f'Clase {i+1}' for i in range(len(probabilities))],
                    'Probabilidad': [f'{p*100:.2f}%' for p in probabilities]
                })

                st.markdown("#### Detalle de Probabilidades")
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Error al realizar la predicción: {e}")

# ==========================================
# PÁGINA: PREDICCIÓN POR LOTES
# ==========================================
elif page == "Predicción por Lotes":
    st.markdown('<div class="section-header"><i class="fas fa-database"></i> Predicción por Lotes</div>', unsafe_allow_html=True)
    st.markdown("Cargue un archivo CSV o Excel con datos de múltiples pacientes para realizar predicciones masivas.")

    st.markdown("""
    <div class="warning-box">
        <p><strong>Requisitos del archivo:</strong> El archivo debe contener las mismas 55 columnas del dataset de entrenamiento. Si incluye la columna "diagnosis", el sistema calculará métricas de evaluación automáticamente.</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload file
    uploaded_file = st.file_uploader(
        "Seleccione un archivo",
        type=['csv', 'xlsx', 'xls'],
        help="Formatos soportados: CSV, XLSX, XLS (máximo 200MB)"
    )

    if uploaded_file is not None:
        try:
            # Leer archivo
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"Archivo cargado correctamente: {uploaded_file.name}")
            st.metric("Total de Registros", len(df))

            # Mostrar preview
            with st.expander("Vista Previa de Datos", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)

            # Botón para procesar
            if st.button("Procesar Lote Completo", use_container_width=True):
                with st.spinner("Procesando predicciones..."):
                    try:
                        # Validar columnas
                        missing_cols = set(feature_names) - set(df.columns)
                        if missing_cols:
                            st.error(f"Error: Faltan las siguientes columnas en el archivo: {missing_cols}")
                            st.stop()

                        # Preparar datos
                        X = df[feature_names].values
                        X_scaled = models['scaler'].transform(X)

                        # Seleccionar modelo
                        model = models[model_type]

                        # Realizar predicciones
                        predictions = model.predict(X_scaled)
                        probabilities = model.predict_proba(X_scaled)

                        # Agregar predicciones al DataFrame
                        df['Predicción'] = predictions
                        df['Confianza'] = [max(p) for p in probabilities]

                        st.markdown("---")
                        st.markdown('<div class="section-header"><i class="fas fa-check-circle"></i> Resultados del Procesamiento</div>', unsafe_allow_html=True)

                        # Métricas generales
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Muestras Procesadas", len(predictions))

                        with col2:
                            st.metric("Confianza Promedio", f"{np.mean([max(p) for p in probabilities])*100:.1f}%")

                        with col3:
                            st.metric("Modelo Utilizado", "R. Logística" if model_type == "logistic" else "Red Neuronal")

                        # Si existe columna diagnosis, calcular métricas
                        if 'diagnosis' in df.columns:
                            y_true = df['diagnosis'].values
                            y_pred = predictions

                            accuracy = accuracy_score(y_true, y_pred)
                            cm = confusion_matrix(y_true, y_pred)
                            class_report = classification_report(y_true, y_pred, output_dict=True)

                            st.markdown("---")
                            st.markdown('<div class="section-header"><i class="fas fa-chart-bar"></i> Métricas de Evaluación</div>', unsafe_allow_html=True)

                            # Accuracy
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{accuracy*100:.1f}%</div><div class="metric-label">Accuracy Global</div></div>', unsafe_allow_html=True)

                            # Matriz de confusión
                            st.markdown("### Matriz de Confusión")
                            st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)

                            # Reporte de clasificación
                            st.markdown("### Reporte de Clasificación Detallado")

                            report_df = pd.DataFrame(class_report).transpose()
                            report_df = report_df[report_df.index.str.isdigit()]
                            report_df.index = [f'Clase {i}' for i in report_df.index]
                            report_df = report_df[['precision', 'recall', 'f1-score', 'support']]
                            report_df.columns = ['Precisión', 'Recall', 'F1-Score', 'Soporte']

                            st.dataframe(
                                report_df.style.format({
                                    'Precisión': '{:.2%}',
                                    'Recall': '{:.2%}',
                                    'F1-Score': '{:.2%}',
                                    'Soporte': '{:.0f}'
                                }),
                                use_container_width=True
                            )

                        # Mostrar resultados
                        st.markdown("---")
                        st.markdown("### Tabla de Resultados Completos")
                        st.dataframe(df, use_container_width=True)

                        # Descargar resultados
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Descargar Resultados (CSV)",
                            data=csv,
                            file_name="predicciones_resultados.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    except Exception as e:
                        st.error(f"Error al procesar el lote: {e}")

        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")

# ==========================================
# PÁGINA: MÉTRICAS DE MODELOS
# ==========================================
elif page == "Métricas y Evaluación":
    st.markdown('<div class="section-header"><i class="fas fa-chart-line"></i> Métricas y Evaluación de Modelos</div>', unsafe_allow_html=True)
    st.markdown("Comparación detallada del rendimiento de los modelos de Machine Learning entrenados.")

    if not metrics_data:
        st.warning("No hay métricas disponibles para mostrar.")
        st.stop()

    # Comparación de Accuracy
    st.markdown("## Comparación de Accuracy")

    col1, col2 = st.columns(2)

    lr_accuracy = metrics_data['logistic_regression']['accuracy']
    nn_accuracy = metrics_data['neural_network']['accuracy']

    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{lr_accuracy*100:.1f}%</div><div class="metric-label">Regresión Logística</div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{nn_accuracy*100:.1f}%</div><div class="metric-label">Red Neuronal Artificial</div></div>', unsafe_allow_html=True)

    # Gráfico de comparación
    fig_comparison = go.Figure(data=[
        go.Bar(
            x=['Regresión Logística', 'Red Neuronal Artificial'],
            y=[lr_accuracy, nn_accuracy],
            marker_color=['#667eea', '#764ba2'],
            text=[f'{lr_accuracy*100:.1f}%', f'{nn_accuracy*100:.1f}%'],
            textposition='auto',
        )
    ])

    fig_comparison.update_layout(
        title="Comparación de Accuracy entre Modelos",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 1],
        height=400,
        showlegend=False,
        font=dict(family="Poppins, sans-serif")
    )

    st.plotly_chart(fig_comparison, use_container_width=True)

    # Métricas detalladas por modelo
    st.markdown("---")

    for model_name, display_name in [
        ('logistic_regression', 'Regresión Logística'),
        ('neural_network', 'Red Neuronal Artificial')
    ]:
        with st.expander(f"{display_name} - Métricas Detalladas", expanded=True):
            model_data = metrics_data[model_name]

            # Matriz de confusión
            cm = np.array(model_data['confusion_matrix'])
            st.plotly_chart(
                plot_confusion_matrix(cm, f"Matriz de Confusión - {display_name}"),
                use_container_width=True
            )

            # Reporte de clasificación
            st.markdown("#### Reporte de Clasificación")

            report = model_data['classification_report']
            report_df = pd.DataFrame(report).transpose()
            report_df = report_df[report_df.index.str.isdigit()]
            report_df.index = [f'Clase {i}' for i in report_df.index]
            report_df = report_df[['precision', 'recall', 'f1-score', 'support']]
            report_df.columns = ['Precisión', 'Recall', 'F1-Score', 'Soporte']

            st.dataframe(
                report_df.style.format({
                    'Precisión': '{:.2%}',
                    'Recall': '{:.2%}',
                    'F1-Score': '{:.2%}',
                    'Soporte': '{:.0f}'
                }),
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 2rem 0; font-size: 0.9rem;">
    <p><strong>SistemaPredict</strong> - Sistema de Diagnóstico Médico con Machine Learning</p>
    <p>Proyecto Final - Análisis de Datos 2025</p>
    <p>Desarrollado con Streamlit • Scikit-learn • Plotly • Python</p>
</div>
""", unsafe_allow_html=True)
