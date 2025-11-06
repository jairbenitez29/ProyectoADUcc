"""
Sistema de Diagnóstico Médico con Machine Learning
Aplicación Streamlit para predicción individual y por lotes
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
    page_title="MediPredict AI - Sistema de Diagnóstico",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar el diseño
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #718096;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    .success-box {
        padding: 1.5rem;
        background: #48bb78;
        color: white;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background: #edf2f7;
        border-left: 4px solid #667eea;
        border-radius: 4px;
        margin: 1rem 0;
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
        xaxis_title="Clase",
        yaxis_title="Probabilidad",
        yaxis_range=[0, 1],
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
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
        textfont={"size": 16},
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
    )

    return fig

# Header
st.markdown('<h1 class="main-header">🏥 MediPredict AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistema Inteligente de Diagnóstico Médico con Machine Learning</p>', unsafe_allow_html=True)

# Cargar modelos
models = load_models()
feature_names = load_feature_names()
metrics_data = load_metrics()

if not models or not feature_names:
    st.error("⚠️ Error: No se pudieron cargar los modelos. Asegúrese de ejecutar train_models.py primero.")
    st.stop()

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/000000/hospital-3.png", width=80)
st.sidebar.title("⚙️ Configuración")

# Selector de página
page = st.sidebar.radio(
    "Navegación",
    ["🔍 Predicción Individual", "📊 Predicción por Lotes", "📈 Métricas de Modelos"],
    label_visibility="collapsed"
)

# Selector de modelo
st.sidebar.markdown("---")
st.sidebar.subheader("Seleccionar Modelo")
model_type = st.sidebar.selectbox(
    "Tipo de Modelo",
    ["logistic", "neural"],
    format_func=lambda x: "📊 Regresión Logística" if x == "logistic" else "🧠 Red Neuronal Artificial"
)

# Información del modelo
if metrics_data:
    model_metrics = metrics_data.get("logistic_regression" if model_type == "logistic" else "neural_network", {})
    accuracy = model_metrics.get("accuracy", 0)

    st.sidebar.markdown("---")
    st.sidebar.metric("Accuracy del Modelo", f"{accuracy*100:.1f}%")

# ==========================================
# PÁGINA: PREDICCIÓN INDIVIDUAL
# ==========================================
if page == "🔍 Predicción Individual":
    st.header("🔍 Predicción Individual")
    st.markdown("Ingrese los datos del paciente para obtener un diagnóstico")

    st.markdown('<div class="info-box">💡 <strong>Sugerencia:</strong> Complete todos los campos con los datos del paciente. Los valores por defecto son 0 para campos binarios.</div>', unsafe_allow_html=True)

    # Crear formulario en columnas
    with st.form("prediction_form"):
        st.subheader("📋 Datos del Paciente")

        # Organizar campos en 3 columnas
        cols = st.columns(3)
        feature_values = {}

        for idx, feature in enumerate(feature_names):
            col = cols[idx % 3]
            with col:
                # Determinar tipo de input
                if feature in ['male', 'female'] or feature.startswith('fever') or feature.startswith('head'):
                    # Campo binario (0 o 1)
                    feature_values[feature] = st.number_input(
                        format_feature_name(feature),
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=1.0,
                        key=feature
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
        submitted = st.form_submit_button("🔮 Realizar Predicción", use_container_width=True)

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
                st.subheader("✅ Resultado del Diagnóstico")

                # Resultado principal
                diagnosis = DIAGNOSIS_MAP.get(prediction, "Desconocido")
                st.markdown(f'<div class="success-box">🎯 {diagnosis}</div>', unsafe_allow_html=True)

                # Métricas
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Predicción", f"Clase {prediction}")

                with col2:
                    st.metric("Confianza", f"{confidence*100:.1f}%")

                with col3:
                    st.metric("Modelo Usado", "R. Logística" if model_type == "logistic" else "Red Neuronal")

                # Gráfico de probabilidades
                st.plotly_chart(plot_probabilities(probabilities, prediction), use_container_width=True)

                # Tabla de probabilidades
                prob_df = pd.DataFrame({
                    'Clase': [f'Clase {i+1}' for i in range(len(probabilities))],
                    'Probabilidad': [f'{p*100:.2f}%' for p in probabilities]
                })

                st.markdown("#### 📊 Detalle de Probabilidades")
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"❌ Error al realizar la predicción: {e}")

# ==========================================
# PÁGINA: PREDICCIÓN POR LOTES
# ==========================================
elif page == "📊 Predicción por Lotes":
    st.header("📊 Predicción por Lotes")
    st.markdown("Cargue un archivo CSV o Excel con múltiples pacientes para realizar predicciones masivas")

    st.markdown('<div class="info-box">💡 <strong>Requisito:</strong> El archivo debe contener las mismas columnas que el dataset de entrenamiento. Si incluye la columna "diagnosis", se mostrarán métricas de evaluación.</div>', unsafe_allow_html=True)

    # Upload file
    uploaded_file = st.file_uploader(
        "Seleccione un archivo CSV o Excel",
        type=['csv', 'xlsx', 'xls'],
        help="Formatos soportados: CSV, XLSX, XLS"
    )

    if uploaded_file is not None:
        try:
            # Leer archivo
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"✅ Archivo cargado: {uploaded_file.name}")
            st.metric("Total de Registros", len(df))

            # Mostrar preview
            with st.expander("👀 Vista Previa de Datos", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)

            # Botón para procesar
            if st.button("🚀 Procesar Lote", use_container_width=True):
                with st.spinner("Procesando predicciones..."):
                    try:
                        # Validar columnas
                        missing_cols = set(feature_names) - set(df.columns)
                        if missing_cols:
                            st.error(f"❌ Faltan las siguientes columnas: {missing_cols}")
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
                        st.subheader("✅ Resultados del Procesamiento")

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
                            st.subheader("📊 Métricas de Evaluación")

                            # Accuracy
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{accuracy*100:.1f}%</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)

                            # Matriz de confusión
                            st.markdown("#### Matriz de Confusión")
                            st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)

                            # Reporte de clasificación
                            st.markdown("#### 📋 Reporte de Clasificación")

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
                        st.markdown("#### 📄 Resultados Completos")
                        st.dataframe(df, use_container_width=True)

                        # Descargar resultados
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Descargar Resultados (CSV)",
                            data=csv,
                            file_name="predicciones_resultados.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    except Exception as e:
                        st.error(f"❌ Error al procesar el lote: {e}")

        except Exception as e:
            st.error(f"❌ Error al leer el archivo: {e}")

# ==========================================
# PÁGINA: MÉTRICAS DE MODELOS
# ==========================================
elif page == "📈 Métricas de Modelos":
    st.header("📈 Métricas de los Modelos")
    st.markdown("Comparación del rendimiento de los modelos de Machine Learning")

    if not metrics_data:
        st.warning("⚠️ No hay métricas disponibles")
        st.stop()

    # Comparación de Accuracy
    st.subheader("🎯 Comparación de Accuracy")

    col1, col2 = st.columns(2)

    lr_accuracy = metrics_data['logistic_regression']['accuracy']
    nn_accuracy = metrics_data['neural_network']['accuracy']

    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{lr_accuracy*100:.1f}%</div><div class="metric-label">📊 Regresión Logística</div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{nn_accuracy*100:.1f}%</div><div class="metric-label">🧠 Red Neuronal</div></div>', unsafe_allow_html=True)

    # Gráfico de comparación
    fig_comparison = go.Figure(data=[
        go.Bar(
            x=['Regresión Logística', 'Red Neuronal'],
            y=[lr_accuracy, nn_accuracy],
            marker_color=['#667eea', '#764ba2'],
            text=[f'{lr_accuracy*100:.1f}%', f'{nn_accuracy*100:.1f}%'],
            textposition='auto',
        )
    ])

    fig_comparison.update_layout(
        title="Comparación de Accuracy",
        yaxis_title="Accuracy",
        yaxis_range=[0, 1],
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig_comparison, use_container_width=True)

    # Métricas detalladas por modelo
    st.markdown("---")

    for model_name, display_name, icon in [
        ('logistic_regression', 'Regresión Logística', '📊'),
        ('neural_network', 'Red Neuronal Artificial', '🧠')
    ]:
        with st.expander(f"{icon} {display_name}", expanded=True):
            model_data = metrics_data[model_name]

            # Matriz de confusión
            cm = np.array(model_data['confusion_matrix'])
            st.plotly_chart(
                plot_confusion_matrix(cm, f"Matriz de Confusión - {display_name}"),
                use_container_width=True
            )

            # Reporte de clasificación
            st.markdown("#### 📋 Reporte de Clasificación")

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
<div style="text-align: center; color: #718096; padding: 2rem 0;">
    <p><strong>MediPredict AI</strong> - Sistema de Diagnóstico Médico con Machine Learning</p>
    <p>Proyecto Final - Análisis de Datos 2025</p>
    <p>Desarrollado con Streamlit, Scikit-learn y Plotly</p>
</div>
""", unsafe_allow_html=True)
