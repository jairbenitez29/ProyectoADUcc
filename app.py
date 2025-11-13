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
from datetime import datetime
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend para no mostrar ventanas
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #1a1a1a;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #1a1a1a;
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #1a1a1a;
    }

    [data-testid="stSidebar"] .stSelectbox label {
        color: #1a1a1a;
    }

    [data-testid="stSidebar"] hr {
        border-color: #dc2626;
    }

    /* Main content */
    .block-container {
        padding-top: 2rem;
        max-width: 100%;
    }

    /* Page Header */
    .page-header {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem 0;
    }

    .page-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }

    .page-subtitle {
        font-size: 1.1rem;
        color: #666;
        font-weight: 400;
    }

    /* Info Section (Homepage) */
    .info-section {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(220, 38, 38, 0.25);
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

    /* Feature Cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        text-align: center;
        border: 1px solid #f0f0f0;
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        box-shadow: 0 4px 16px rgba(220, 38, 38, 0.15);
        border-color: #dc2626;
        transform: translateY(-3px);
    }

    .feature-icon {
        font-size: 2.5rem;
        color: #dc2626;
        margin-bottom: 1rem;
    }

    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }

    .feature-desc {
        font-size: 0.9rem;
        color: #666;
        line-height: 1.5;
    }

    /* Cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        border: 1px solid #f0f0f0;
    }

    .card-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f0f0;
    }

    .card-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a1a1a;
    }

    .card-icon {
        font-size: 1.5rem;
        color: #dc2626;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(220, 38, 38, 0.25);
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(220, 38, 38, 0.35);
    }

    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        line-height: 1;
    }

    .metric-label {
        font-size: 1rem;
        opacity: 0.95;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Success Box */
    .success-box {
        padding: 2.5rem;
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        color: white;
        border-radius: 16px;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin: 2rem 0;
        box-shadow: 0 8px 30px rgba(220, 38, 38, 0.3);
        letter-spacing: -0.5px;
    }

    /* Info Box */
    .info-box {
        padding: 1.5rem;
        background: #f8f9fa;
        border-left: 4px solid #dc2626;
        border-radius: 8px;
        margin: 1.5rem 0;
    }

    .info-box p {
        margin: 0;
        color: #2c3e50;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        font-weight: 600;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.3);
        transition: all 0.3s ease;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stButton>button:hover {
        box-shadow: 0 6px 25px rgba(220, 38, 38, 0.4);
        transform: translateY(-2px);
        background: linear-gradient(135deg, #b91c1c 0%, #991b1b 100%);
    }

    /* Forms */
    .stNumberInput input {
        border: 2px solid #e5e5e5;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 0.95rem;
        transition: all 0.3s ease;
    }

    .stNumberInput input:focus {
        border-color: #dc2626;
        box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.1);
    }

    /* Select Box */
    .stSelectbox > div > div {
        border: 2px solid #e5e5e5;
        border-radius: 8px;
        background: white;
    }

    .stSelectbox > div > div:focus-within {
        border-color: #dc2626;
        box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.1);
    }

    /* File Uploader */
    .stFileUploader {
        border: 2px dashed #dc2626;
        border-radius: 12px;
        padding: 2rem;
        background: #fef2f2;
        transition: all 0.3s ease;
    }

    .stFileUploader:hover {
        background: #fee2e2;
        border-color: #b91c1c;
    }

    /* Metrics (Streamlit native) */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a1a;
    }

    [data-testid="stMetricLabel"] {
        color: #666;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }

    /* Dataframes */
    .stDataFrame {
        border: 1px solid #e5e5e5;
        border-radius: 8px;
        overflow: hidden;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
        font-weight: 600;
        color: #1a1a1a;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }

    ::-webkit-scrollbar-thumb {
        background: #dc2626;
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #b91c1c;
    }

    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #dc2626 !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
        border-bottom: 2px solid #e5e5e5;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
        color: #666;
        border-radius: 8px 8px 0 0;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f8f9fa;
        color: #1a1a1a;
    }

    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #dc2626;
        border-bottom: 3px solid #dc2626;
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
    1: "Dengue",
    2: "Malaria",
    3: "Leptospirosis"
}

# Función para formatear nombres de características
def format_feature_name(name):
    """Convertir nombre de característica a formato legible"""
    return name.replace('_', ' ').title()

# Función para crear gráfico de probabilidades
def plot_probabilities(probabilities, prediction):
    """Crear gráfico de barras de probabilidades"""
    classes = [DIAGNOSIS_MAP.get(i+1, f"Clase {i+1}") for i in range(len(probabilities))]
    colors = ['#dc2626' if i == prediction-1 else '#e5e5e5' for i in range(len(probabilities))]

    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probabilities,
            marker_color=colors,
            text=[f'{p*100:.1f}%' for p in probabilities],
            textposition='auto',
            textfont=dict(size=14, color='#1a1a1a', family='Inter', weight='bold')
        )
    ])

    fig.update_layout(
        title="Probabilidades de Diagnóstico",
        xaxis_title="Clase de Diagnóstico",
        yaxis_title="Probabilidad",
        yaxis_range=[0, 1],
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color='#1a1a1a'),
        title_font=dict(size=18, color='#1a1a1a', family='Inter', weight='bold')
    )

    return fig

# Función para crear matriz de confusión
def plot_confusion_matrix(cm, title="Matriz de Confusión"):
    """Crear visualización de matriz de confusión"""
    labels = [DIAGNOSIS_MAP.get(i+1, f"Clase {i+1}") for i in range(len(cm))]

    # Crear colorscale personalizada: blanco a rojo
    colorscale = [
        [0, '#ffffff'],
        [0.5, '#fecaca'],
        [1, '#dc2626']
    ]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=colorscale,
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16, "family": "Inter", "color": "#1a1a1a"},
        hoverongaps=False,
        showscale=True,
        colorbar=dict(title="Cantidad", title_font=dict(color='#1a1a1a'))
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Predicción",
        yaxis_title="Valor Real",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color='#1a1a1a'),
        title_font=dict(size=18, color='#1a1a1a', family='Inter', weight='bold')
    )

    return fig

# Función para generar reporte PDF
def generate_pdf_report(df, predictions, probabilities, model_name, y_true=None):
    """Genera un reporte PDF completo con los resultados de predicción"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)
    story = []
    styles = getSampleStyleSheet()

    # Estilos personalizados
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#dc2626'),
        spaceAfter=30,
        alignment=TA_CENTER
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=12,
        spaceBefore=12
    )

    # Título
    story.append(Paragraph("Reporte de Predicción - SistemaPredict", title_style))
    story.append(Paragraph(f"Diagnóstico de Dengue, Malaria y Leptospirosis", styles['Normal']))
    story.append(Paragraph(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    story.append(Paragraph(f"Modelo: {model_name}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    # Métricas generales
    story.append(Paragraph("Resumen General", heading_style))
    total_processed = len(predictions)
    avg_confidence = np.mean([max(p) for p in probabilities]) * 100

    # Contar predicciones por enfermedad
    unique, counts = np.unique(predictions, return_counts=True)
    disease_counts = dict(zip(unique, counts))

    summary_data = [
        ['Métrica', 'Valor'],
        ['Total de pacientes procesados', str(total_processed)],
        ['Confianza promedio', f'{avg_confidence:.2f}%'],
        ['Casos de Dengue predichos', str(disease_counts.get(1, 0))],
        ['Casos de Malaria predichos', str(disease_counts.get(2, 0))],
        ['Casos de Leptospirosis predichos', str(disease_counts.get(3, 0))]
    ]

    if y_true is not None:
        accuracy = accuracy_score(y_true, predictions) * 100
        summary_data.append(['Accuracy del modelo', f'{accuracy:.2f}%'])

    summary_table = Table(summary_data, colWidths=[4*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc2626')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))

    # Matriz de confusión (si hay datos reales)
    if y_true is not None:
        story.append(Paragraph("Matriz de Confusión", heading_style))
        cm = confusion_matrix(y_true, predictions)

        # Crear matriz de confusión con matplotlib
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap='Reds')
        ax.figure.colorbar(im, ax=ax)

        # Etiquetas
        tick_labels = [DIAGNOSIS_MAP.get(i+1, f'Clase {i+1}') for i in range(len(cm))]
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=tick_labels,
               yticklabels=tick_labels,
               xlabel='Predicción',
               ylabel='Valor Real')

        # Rotar etiquetas
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Agregar texto en las celdas
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=14, fontweight='bold')

        fig.tight_layout()

        # Guardar figura a buffer
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)

        # Agregar imagen al PDF
        img = Image(img_buffer, width=5*inch, height=4*inch)
        story.append(img)
        story.append(Spacer(1, 0.3*inch))

        # Reporte de clasificación
        story.append(Paragraph("Reporte de Clasificación por Enfermedad", heading_style))
        class_report = classification_report(y_true, predictions, output_dict=True)

        report_data = [['Enfermedad', 'Precision', 'Recall', 'F1-Score', 'Soporte']]
        for key in ['1', '2', '3']:
            if key in class_report:
                disease_name = DIAGNOSIS_MAP.get(int(key), f'Clase {key}')
                report_data.append([
                    disease_name,
                    f"{class_report[key]['precision']:.2%}",
                    f"{class_report[key]['recall']:.2%}",
                    f"{class_report[key]['f1-score']:.2%}",
                    f"{int(class_report[key]['support'])}"
                ])

        report_table = Table(report_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1*inch])
        report_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc2626')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
        ]))
        story.append(report_table)
        story.append(Spacer(1, 0.3*inch))

    # Distribución de predicciones
    story.append(PageBreak())
    story.append(Paragraph("Distribución de Predicciones", heading_style))

    # Crear gráfico de barras con matplotlib
    fig, ax = plt.subplots(figsize=(7, 4))
    diseases = [DIAGNOSIS_MAP.get(i+1, f'Clase {i+1}') for i in range(3)]
    counts_list = [disease_counts.get(i+1, 0) for i in range(3)]
    colors_bars = ['#dc2626', '#991b1b', '#7f1d1d']

    bars = ax.bar(diseases, counts_list, color=colors_bars)
    ax.set_ylabel('Cantidad de Casos', fontsize=11)
    ax.set_xlabel('Enfermedad', fontsize=11)
    ax.set_title('Distribución de Diagnósticos Predichos', fontsize=13, fontweight='bold')

    # Agregar valores en las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.xticks(rotation=15, ha='right')
    fig.tight_layout()

    # Guardar figura
    img_buffer2 = BytesIO()
    plt.savefig(img_buffer2, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    img_buffer2.seek(0)

    img2 = Image(img_buffer2, width=5.5*inch, height=3.5*inch)
    story.append(img2)
    story.append(Spacer(1, 0.3*inch))

    # Tabla de resultados (primeros 20)
    story.append(Paragraph("Muestra de Resultados Detallados (primeros 20 casos)", heading_style))

    # Crear tabla con resultados
    results_data = [['#', 'Predicción', 'Confianza']]
    for i in range(min(20, len(predictions))):
        pred = predictions[i]
        conf = max(probabilities[i]) * 100
        disease_name = DIAGNOSIS_MAP.get(pred, f'Clase {pred}')
        results_data.append([
            str(i+1),
            disease_name,
            f'{conf:.1f}%'
        ])

    results_table = Table(results_data, colWidths=[0.7*inch, 2.5*inch, 1.5*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc2626')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
    ]))
    story.append(results_table)

    # Nota final
    story.append(Spacer(1, 0.4*inch))
    note = Paragraph(
        "<b>Nota:</b> Este reporte fue generado automáticamente por SistemaPredict. "
        "Los resultados deben ser interpretados por personal de salud calificado.",
        styles['Normal']
    )
    story.append(note)

    # Generar PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# Función para generar reporte PDF individual
def generate_individual_pdf_report(patient_info, prediction, probabilities, model_name, feature_values):
    """Genera un reporte PDF para un paciente individual"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)
    story = []
    styles = getSampleStyleSheet()

    # Estilos personalizados
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#dc2626'),
        spaceAfter=20,
        alignment=TA_CENTER
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=10,
        spaceBefore=12
    )

    # Título
    story.append(Paragraph("Reporte de Diagnóstico Individual", title_style))
    story.append(Paragraph("SistemaPredict - Diagnóstico de Dengue, Malaria y Leptospirosis", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    # Información del paciente
    story.append(Paragraph("Información del Paciente", heading_style))
    patient_data = [
        ['Campo', 'Valor'],
        ['Nombre Completo', patient_info['name']],
        ['Documento de Identidad', patient_info['id']],
        ['Fecha de Consulta', patient_info['date']],
        ['Edad', f"{int(feature_values.get('age', 0))} años"],
        ['Género', 'Masculino' if feature_values.get('male', 0) == 1 else 'Femenino'],
        ['Modelo Utilizado', model_name]
    ]

    if patient_info.get('notes'):
        patient_data.append(['Observaciones', patient_info['notes']])

    patient_table = Table(patient_data, colWidths=[2.5*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc2626')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.3*inch))

    # Diagnóstico
    story.append(Paragraph("Diagnóstico Predicho", heading_style))
    diagnosis = DIAGNOSIS_MAP.get(prediction, "Desconocido")
    confidence = max(probabilities) * 100

    diagnosis_data = [
        ['Enfermedad Predicha', diagnosis],
        ['Nivel de Confianza', f'{confidence:.2f}%']
    ]

    diagnosis_table = Table(diagnosis_data, colWidths=[2.5*inch, 4*inch])
    diagnosis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fee2e2')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#991b1b'))
    ]))
    story.append(diagnosis_table)
    story.append(Spacer(1, 0.3*inch))

    # Probabilidades detalladas
    story.append(Paragraph("Probabilidades Detalladas", heading_style))

    prob_data = [['Enfermedad', 'Probabilidad', 'Predicción']]
    for i in range(len(probabilities)):
        disease_id = i + 1
        disease_name = DIAGNOSIS_MAP.get(disease_id, f'Clase {disease_id}')
        prob = probabilities[i] * 100
        is_pred = '✓' if disease_id == prediction else ''
        prob_data.append([disease_name, f'{prob:.2f}%', is_pred])

    # Ordenar por probabilidad (mantener header en su lugar)
    header = prob_data[0]
    rows = sorted(prob_data[1:], key=lambda x: float(x[1].replace('%', '')), reverse=True)
    prob_data = [header] + rows

    prob_table = Table(prob_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc2626')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 0.4*inch))

    # Gráfico de barras
    fig, ax = plt.subplots(figsize=(6, 3.5))
    diseases = [DIAGNOSIS_MAP.get(i+1, f'Clase {i+1}') for i in range(3)]
    probs = [probabilities[i] * 100 for i in range(3)]
    colors_bars = ['#dc2626' if i+1 == prediction else '#e5e5e5' for i in range(3)]

    bars = ax.bar(diseases, probs, color=colors_bars)
    ax.set_ylabel('Probabilidad (%)', fontsize=10)
    ax.set_title('Distribución de Probabilidades', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 100])

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xticks(rotation=15, ha='right')
    fig.tight_layout()

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    img_buffer.seek(0)

    img = Image(img_buffer, width=5*inch, height=3*inch)
    story.append(img)
    story.append(Spacer(1, 0.3*inch))

    # Nota final
    note = Paragraph(
        "<b>Nota:</b> Este diagnóstico fue generado por un modelo de Machine Learning "
        "y debe ser interpretado por personal de salud calificado. No reemplaza el criterio médico profesional.",
        styles['Normal']
    )
    story.append(note)

    story.append(Spacer(1, 0.3*inch))
    footer = Paragraph(
        f"Reporte generado el {datetime.now().strftime('%d/%m/%Y a las %H:%M')} por SistemaPredict",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    )
    story.append(footer)

    # Generar PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# Inicializar session state
if 'model_type' not in st.session_state:
    st.session_state.model_type = 'neural'
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'inicio'

# Cargar modelos
models = load_models()
feature_names = load_feature_names()
metrics_data = load_metrics()

if not models or not feature_names:
    st.error("Error: No se pudieron cargar los modelos. Asegúrese de ejecutar train_models.py primero.")
    st.stop()

# ==================================
# SIDEBAR DE NAVEGACIÓN (STREAMLIT NATIVO)
# ==================================
with st.sidebar:
    st.title("🏥 SistemaPredict")
    st.markdown("### Navegación")
    st.markdown("---")

    page = st.radio(
        "Selecciona una página:",
        ["Inicio", "Predicción Individual", "Predicción por Lotes", "Métricas de Modelos"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### Configuración")

    model_type = st.selectbox(
        "Modelo de ML",
        ["neural", "logistic"],
        index=0 if st.session_state.model_type == "neural" else 1,
        format_func=lambda x: "Red Neuronal Artificial" if x == "neural" else "Regresión Logística"
    )
    st.session_state.model_type = model_type

    if metrics_data:
        model_metrics = metrics_data.get("logistic_regression" if model_type == "logistic" else "neural_network", {})
        accuracy = model_metrics.get("accuracy", 0)
        st.metric("Accuracy", f"{accuracy*100:.1f}%")

    st.markdown("---")
    st.info(
        "**Dataset:** 81 pacientes\n\n"
        "**Características:** 55 variables\n\n"
        "**Enfermedades:** Dengue, Malaria, Leptospirosis"
    )

# Mapear la selección a current_page
page_mapping = {
    "Inicio": "inicio",
    "Predicción Individual": "individual",
    "Predicción por Lotes": "lotes",
    "Métricas de Modelos": "metricas"
}
st.session_state.current_page = page_mapping[page]

# ==========================================
# PÁGINA: INICIO - DOCUMENTACIÓN
# ==========================================
if st.session_state.current_page == 'inicio':
    # Sección Principal de Información
    st.markdown("""
    <div class="info-section">
        <h3>¿Qué es SistemaPredict?</h3>
        <p>
            SistemaPredict es una aplicación web de diagnóstico médico asistido por inteligencia artificial,
            desarrollada como parte del Proyecto Final de Análisis de Datos 2025. El sistema utiliza algoritmos
            de Machine Learning para identificar y clasificar <strong>tres enfermedades endémicas en Colombia:
            Dengue, Malaria y Leptospirosis</strong>, basándose en datos clínicos y de laboratorio de pacientes.
        </p>
        <p>
            Este proyecto utiliza un dataset real de una región endémica de Colombia, permitiendo a profesionales
            de la salud realizar predicciones tanto individuales como masivas, proporcionando herramientas de
            análisis y visualización para la toma de decisiones informadas en el diagnóstico diferencial de estas
            tres enfermedades.
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
                Clasificación de Dengue, Malaria o Leptospirosis ingresando datos demográficos,
                síntomas y resultados de laboratorio de un paciente específico.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon"><i class="fas fa-database"></i></div>
            <div class="feature-title">Procesamiento por Lotes</div>
            <div class="feature-desc">
                Análisis masivo de múltiples pacientes para diagnóstico diferencial mediante carga
                de archivos CSV/Excel con generación automática de reportes.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon"><i class="fas fa-chart-line"></i></div>
            <div class="feature-title">Métricas Detalladas</div>
            <div class="feature-desc">
                Evaluación completa del rendimiento para cada enfermedad con matrices de confusión,
                accuracy, precision, recall y F1-score para ambos modelos.
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
        - Dataset de región endémica en Colombia
        - 81 pacientes con diagnósticos confirmados
        - 55 características clínicas y de laboratorio
        - **3 enfermedades:** Dengue, Malaria y Leptospirosis
        - División: 80% entrenamiento, 20% prueba

        **Variables Incluidas:**
        - Datos demográficos (edad, género, ocupación, origen)
        - Síntomas clínicos (fiebre, dolor, mareos, ictericia, etc.)
        - Exámenes de laboratorio (hemograma completo, química sanguínea)
        - Días de hospitalización y temperatura corporal
        """)

    # Cómo Funciona
    st.markdown("## ¿Cómo Funciona el Sistema?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **1. Recopilación de Datos**

        El sistema recibe información del paciente: datos demográficos, síntomas reportados
        y resultados de exámenes de laboratorio relevantes para Dengue, Malaria y Leptospirosis.

        **2. Preprocesamiento**

        Los datos se normalizan utilizando StandardScaler para asegurar que todas las
        características tengan la misma escala y peso en el modelo.
        """)

    with col2:
        st.markdown("""
        **3. Predicción**

        El modelo seleccionado (Regresión Logística o Red Neuronal) procesa los datos
        y genera una predicción indicando si es Dengue, Malaria o Leptospirosis.

        **4. Interpretación**

        El sistema presenta la enfermedad identificada junto con el nivel de confianza
        y las probabilidades para cada una de las tres enfermedades.
        """)

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

    # Consideraciones
    st.markdown("## Consideraciones Importantes")

    st.warning("""
    **Nota Importante:** Este sistema es una herramienta de apoyo para el diagnóstico diferencial
    de Dengue, Malaria y Leptospirosis y NO reemplaza el criterio médico profesional. Las
    predicciones deben ser interpretadas por personal de salud calificado y consideradas junto
    con otros factores clínicos, epidemiológicos y de laboratorio confirmatorio relevantes.
    """)

    st.info("""
    **Alcance del Proyecto:** Sistema desarrollado con fines educativos como parte del curso
    de Análisis de Datos 2025. El modelo fue entrenado con un dataset real de una región endémica
    en Colombia con 81 casos confirmados de Dengue, Malaria y Leptospirosis. Su aplicación en
    entornos clínicos reales requeriría validación adicional y aprobación regulatoria.
    """)

# ==========================================
# PÁGINA: PREDICCIÓN INDIVIDUAL
# ==========================================
elif st.session_state.current_page == 'individual':
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Predicción Individual</div>
        <div class="page-subtitle">Diagnóstico diferencial de Dengue, Malaria y Leptospirosis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <p><strong>Instrucciones:</strong> Complete los datos clínicos y de laboratorio del paciente. Los campos binarios aceptan 0 (No) o 1 (Sí). El sistema identificará si el cuadro clínico corresponde a Dengue, Malaria o Leptospirosis.</p>
    </div>
    """, unsafe_allow_html=True)

    # Definir límites razonables basados en rangos clínicos
    # Formato: (min, max, step, help_text, default_value)
    FIELD_LIMITS = {
        'age': (0.0, 100.0, 1.0, "Edad máxima: 100 años", 0.0),
        'hospitalization_days': (0.0, 365.0, 1.0, "Días de hospitalización", 0.0),
        'body_temperature': (30.0, 45.0, 0.1, "Temperatura corporal en °C", 36.5),
        'hematocrit': (0.0, 60.0, 0.1, "Porcentaje (%)", 0.0),
        'hemoglobin': (0.0, 25.0, 0.1, "g/dL", 0.0),
        'red_blood_cells': (0.0, 10000000.0, 1000.0, "Células/μL", 0.0),
        'white_blood_cells': (0.0, 50000.0, 100.0, "Células/μL", 0.0),
        'neutrophils': (0.0, 100.0, 0.1, "Porcentaje (%)", 0.0),
        'eosinophils': (0.0, 100.0, 0.1, "Porcentaje (%)", 0.0),
        'basophils': (0.0, 10.0, 0.01, "Porcentaje (%)", 0.0),
        'monocytes': (0.0, 100.0, 0.1, "Porcentaje (%)", 0.0),
        'lymphocytes': (0.0, 100.0, 0.1, "Porcentaje (%)", 0.0),
        'platelets': (0.0, 1000000.0, 1000.0, "Células/μL", 0.0),
        'AST (SGOT)': (0.0, 1000.0, 1.0, "U/L", 0.0),
        'ALT (SGPT)': (0.0, 1500.0, 1.0, "U/L", 0.0),
        'ALP (alkaline_phosphatase)': (0.0, 500.0, 1.0, "U/L", 0.0),
        'total_bilirubin': (0.0, 20.0, 0.01, "mg/dL", 0.0),
        'direct_bilirubin': (0.0, 15.0, 0.01, "mg/dL", 0.0),
        'indirect_bilirubin': (0.0, 10.0, 0.01, "mg/dL", 0.0),
        'total_proteins': (0.0, 12.0, 0.1, "g/dL", 0.0),
        'albumin': (0.0, 6.0, 0.01, "g/dL", 0.0),
        'creatinine': (0.0, 15.0, 0.01, "mg/dL", 0.0),
        'urea': (0.0, 300.0, 0.1, "mg/dL", 0.0)
    }

    # Lista de campos binarios
    BINARY_FIELDS = [
        'male', 'female', 'urban_origin', 'rural_origin', 'homemaker',
        'student', 'professional', 'merchant', 'agriculture_livestock',
        'various_jobs', 'unemployed', 'fever', 'headache', 'dizziness',
        'loss_of_appetite', 'weakness', 'myalgias', 'arthralgias',
        'eye_pain', 'hemorrhages', 'vomiting', 'abdominal_pain',
        'chills', 'hemoptysis', 'edema', 'jaundice', 'bruises',
        'petechiae', 'rash', 'diarrhea', 'respiratory_difficulty', 'itching'
    ]

    # Formulario
    with st.form("prediction_form"):
        st.markdown('<div class="card"><div class="card-header"><i class="fas fa-user-injured card-icon"></i><span class="card-title">Información del Paciente</span></div></div>', unsafe_allow_html=True)

        # Campos de identificación del paciente
        st.markdown("#### Datos de Identificación")
        id_col1, id_col2, id_col3 = st.columns(3)

        with id_col1:
            patient_name = st.text_input(
                "Nombre Completo del Paciente *",
                placeholder="Ej: Juan Pérez García",
                help="Nombre completo del paciente para identificación"
            )

        with id_col2:
            patient_id = st.text_input(
                "Documento de Identidad *",
                placeholder="Ej: 1234567890",
                help="Número de documento de identidad (CC, TI, etc.)"
            )

        with id_col3:
            consultation_date = st.date_input(
                "Fecha de Consulta *",
                value=datetime.now().date(),
                help="Fecha de la consulta médica"
            )

        patient_notes = st.text_area(
            "Observaciones Clínicas (Opcional)",
            placeholder="Síntomas adicionales, antecedentes relevantes, etc.",
            help="Información adicional que pueda ser relevante para el diagnóstico"
        )

        st.markdown("---")
        st.markdown("#### Datos Clínicos y de Laboratorio")

        cols = st.columns(3)
        feature_values = {}

        for idx, feature in enumerate(feature_names):
            col = cols[idx % 3]
            with col:
                # Campos binarios (0 o 1)
                if feature in BINARY_FIELDS:
                    feature_values[feature] = st.number_input(
                        format_feature_name(feature),
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=1.0,
                        key=feature,
                        help="0 = No, 1 = Sí"
                    )
                # Campos con límites específicos
                elif feature in FIELD_LIMITS:
                    min_val, max_val, step_val, help_text, default_val = FIELD_LIMITS[feature]
                    feature_values[feature] = st.number_input(
                        format_feature_name(feature),
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step_val,
                        key=feature,
                        help=help_text
                    )
                # Campos numéricos generales
                else:
                    feature_values[feature] = st.number_input(
                        format_feature_name(feature),
                        min_value=0.0,
                        max_value=1000000.0,
                        value=0.0,
                        step=0.01,
                        format="%.2f",
                        key=feature
                    )

        submitted = st.form_submit_button("Realizar Predicción")

        if submitted:
            try:
                # Validar campos obligatorios
                if not patient_name or not patient_id:
                    st.error("⚠️ Por favor complete los campos obligatorios: Nombre del Paciente y Documento de Identidad")
                    st.stop()

                X = np.array([[feature_values[feat] for feat in feature_names]])
                X_scaled = models['scaler'].transform(X)
                model = models[st.session_state.model_type]

                prediction = int(model.predict(X_scaled)[0])
                probabilities = model.predict_proba(X_scaled)[0]
                confidence = max(probabilities)

                st.markdown("---")
                st.markdown("## 📋 Resultados del Diagnóstico")

                # Mostrar información del paciente
                st.markdown("### Información del Paciente")
                patient_info_col1, patient_info_col2, patient_info_col3 = st.columns(3)

                with patient_info_col1:
                    st.markdown(f"""
                    **Nombre:** {patient_name}
                    **Documento:** {patient_id}
                    """)

                with patient_info_col2:
                    st.markdown(f"""
                    **Fecha de Consulta:** {consultation_date.strftime('%d/%m/%Y')}
                    **Modelo Usado:** {'Red Neuronal' if st.session_state.model_type == 'neural' else 'Regresión Logística'}
                    """)

                with patient_info_col3:
                    st.markdown(f"""
                    **Edad:** {int(feature_values.get('age', 0))} años
                    **Género:** {'Masculino' if feature_values.get('male', 0) == 1 else 'Femenino'}
                    """)

                if patient_notes:
                    st.markdown(f"**Observaciones:** {patient_notes}")

                st.markdown("---")
                st.markdown("### Diagnóstico Predicho")

                diagnosis = DIAGNOSIS_MAP.get(prediction, "Desconocido")
                st.markdown(f'<div class="success-box">{diagnosis}</div>', unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicción", DIAGNOSIS_MAP.get(prediction, "Desconocido"))
                with col2:
                    st.metric("Confianza", f"{confidence*100:.1f}%")
                with col3:
                    st.metric("Modelo", "Red Neuronal" if st.session_state.model_type == "neural" else "R. Logística")

                st.plotly_chart(plot_probabilities(probabilities, prediction), use_container_width=True)

                # Crear tabla de probabilidades con mejor formato
                st.markdown("### Tabla de Probabilidades Detallada")

                prob_data = []
                for i in range(len(probabilities)):
                    disease_id = i + 1
                    disease_name = DIAGNOSIS_MAP.get(disease_id, f'Clase {disease_id}')
                    prob = probabilities[i] * 100
                    is_prediction = (disease_id == prediction)

                    prob_data.append({
                        'Enfermedad': disease_name,
                        'Probabilidad (%)': prob,
                        'Es Predicción': '✓' if is_prediction else ''
                    })

                # Ordenar por probabilidad descendente
                prob_df = pd.DataFrame(prob_data).sort_values('Probabilidad (%)', ascending=False)

                # Aplicar estilos a la tabla
                def style_probability_table(df):
                    def highlight_prediction(row):
                        if row['Es Predicción'] == '✓':
                            return ['background-color: #fee2e2; font-weight: bold'] * len(row)
                        return [''] * len(row)

                    def color_probability(val):
                        if isinstance(val, (int, float)):
                            if val >= 70:
                                color = '#dc2626'  # Rojo fuerte - alta probabilidad
                            elif val >= 40:
                                color = '#f97316'  # Naranja - probabilidad media
                            else:
                                color = '#6b7280'  # Gris - baja probabilidad
                            return f'color: {color}; font-weight: bold'
                        return ''

                    styled = df.style.apply(highlight_prediction, axis=1)
                    styled = styled.map(color_probability, subset=['Probabilidad (%)'])
                    styled = styled.format({'Probabilidad (%)': '{:.2f}%'})

                    return styled

                st.dataframe(
                    style_probability_table(prob_df),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Enfermedad": st.column_config.TextColumn(
                            "Enfermedad",
                            width="medium",
                        ),
                        "Probabilidad (%)": st.column_config.ProgressColumn(
                            "Probabilidad",
                            format="%.2f%%",
                            min_value=0,
                            max_value=100,
                        ),
                        "Es Predicción": st.column_config.TextColumn(
                            "Predicción",
                            width="small",
                        )
                    }
                )

                # Botón para descargar reporte PDF individual
                st.markdown("---")
                st.markdown("### 📄 Descargar Reporte")

                # Preparar información del paciente para el PDF
                patient_info_dict = {
                    'name': patient_name,
                    'id': patient_id,
                    'date': consultation_date.strftime('%d/%m/%Y'),
                    'notes': patient_notes if patient_notes else ''
                }

                model_name = "Red Neuronal Artificial" if st.session_state.model_type == "neural" else "Regresión Logística"

                # Generar PDF
                pdf_buffer = generate_individual_pdf_report(
                    patient_info=patient_info_dict,
                    prediction=prediction,
                    probabilities=probabilities,
                    model_name=model_name,
                    feature_values=feature_values
                )

                # Botón de descarga
                st.download_button(
                    label="📥 Descargar Reporte Completo (PDF)",
                    data=pdf_buffer,
                    file_name=f"diagnostico_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )

                st.info("💡 **Tip:** Descargue el reporte PDF para mantener un registro completo del diagnóstico del paciente.")

            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# PÁGINA: PREDICCIÓN POR LOTES
# ==========================================
elif st.session_state.current_page == 'lotes':
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Predicción por Lotes</div>
        <div class="page-subtitle">Clasificación masiva de casos de Dengue, Malaria y Leptospirosis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <p><strong>Requisitos:</strong> Archivo CSV o Excel con las mismas 55 columnas del entrenamiento. El sistema clasificará cada caso como Dengue, Malaria o Leptospirosis.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Seleccione un archivo",
        type=['csv', 'xlsx', 'xls'],
        help="Formatos: CSV, XLSX, XLS (máximo 200MB)"
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"✅ Archivo cargado: {uploaded_file.name}")
            st.metric("Total de Registros", len(df))

            with st.expander("Vista Previa de Datos", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)

            if st.button("Procesar Lote Completo"):
                with st.spinner("Procesando..."):
                    try:
                        missing_cols = set(feature_names) - set(df.columns)
                        if missing_cols:
                            st.error(f"❌ Faltan columnas: {missing_cols}")
                            st.stop()

                        X = df[feature_names].values
                        X_scaled = models['scaler'].transform(X)
                        model = models[st.session_state.model_type]

                        predictions = model.predict(X_scaled)
                        probabilities = model.predict_proba(X_scaled)

                        df['Predicción'] = predictions
                        df['Confianza'] = [max(p) for p in probabilities]

                        st.markdown("---")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Procesadas", len(predictions))
                        with col2:
                            st.metric("Confianza Media", f"{np.mean([max(p) for p in probabilities])*100:.1f}%")
                        with col3:
                            st.metric("Modelo", "Red Neuronal" if st.session_state.model_type == "neural" else "R. Logística")

                        if 'diagnosis' in df.columns:
                            y_true = df['diagnosis'].values
                            accuracy = accuracy_score(y_true, predictions)
                            cm = confusion_matrix(y_true, predictions)
                            class_report = classification_report(y_true, predictions, output_dict=True)

                            st.markdown("---")
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{accuracy*100:.1f}%</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)

                            st.markdown("### Matriz de Confusión")
                            st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)

                            st.markdown("### Reporte de Clasificación")
                            report_df = pd.DataFrame(class_report).transpose()
                            report_df = report_df[report_df.index.str.isdigit()]
                            report_df.index = [DIAGNOSIS_MAP.get(int(i), f'Clase {i}') for i in report_df.index]
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

                        st.markdown("---")
                        st.markdown("### Resultados Completos")
                        st.dataframe(df, use_container_width=True)

                        # Generar reporte PDF
                        model_name = "Red Neuronal Artificial" if st.session_state.model_type == "neural" else "Regresión Logística"
                        y_true_pdf = df['diagnosis'].values if 'diagnosis' in df.columns else None

                        pdf_buffer = generate_pdf_report(
                            df=df,
                            predictions=predictions,
                            probabilities=probabilities,
                            model_name=model_name,
                            y_true=y_true_pdf
                        )

                        st.download_button(
                            label="📄 Descargar Reporte Completo (PDF)",
                            data=pdf_buffer,
                            file_name=f"reporte_prediccion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        except Exception as e:
            st.error(f"❌ Error al leer archivo: {e}")

# ==========================================
# PÁGINA: MÉTRICAS DE MODELOS
# ==========================================
elif st.session_state.current_page == 'metricas':
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Métricas de Modelos</div>
        <div class="page-subtitle">Evaluación del rendimiento en la clasificación de Dengue, Malaria y Leptospirosis</div>
    </div>
    """, unsafe_allow_html=True)

    if not metrics_data:
        st.warning("⚠️ No hay métricas disponibles")
        st.stop()

    st.markdown("## Comparación de Accuracy")

    col1, col2 = st.columns(2)

    lr_accuracy = metrics_data['logistic_regression']['accuracy']
    nn_accuracy = metrics_data['neural_network']['accuracy']

    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{lr_accuracy*100:.1f}%</div><div class="metric-label">Regresión Logística</div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{nn_accuracy*100:.1f}%</div><div class="metric-label">Red Neuronal</div></div>', unsafe_allow_html=True)

    # Gráfico comparativo
    fig_comparison = go.Figure(data=[
        go.Bar(
            x=['Regresión Logística', 'Red Neuronal'],
            y=[lr_accuracy, nn_accuracy],
            marker_color=['#666', '#dc2626'],
            text=[f'{lr_accuracy*100:.1f}%', f'{nn_accuracy*100:.1f}%'],
            textposition='auto',
            textfont=dict(size=14, color='white', family='Inter', weight='bold')
        )
    ])

    fig_comparison.update_layout(
        title="Comparación de Accuracy",
        yaxis_title="Accuracy",
        yaxis_range=[0, 1],
        height=400,
        showlegend=False,
        font=dict(family="Inter, sans-serif", color='#1a1a1a')
    )

    st.plotly_chart(fig_comparison, use_container_width=True)

    st.markdown("---")

    for model_name, display_name in [
        ('logistic_regression', 'Regresión Logística'),
        ('neural_network', 'Red Neuronal Artificial')
    ]:
        with st.expander(f"{display_name} - Métricas Detalladas", expanded=True):
            model_data = metrics_data[model_name]

            cm = np.array(model_data['confusion_matrix'])
            st.plotly_chart(
                plot_confusion_matrix(cm, f"Matriz de Confusión - {display_name}"),
                use_container_width=True
            )

            st.markdown("#### Reporte de Clasificación")
            report = model_data['classification_report']
            report_df = pd.DataFrame(report).transpose()
            report_df = report_df[report_df.index.str.isdigit()]
            report_df.index = [DIAGNOSIS_MAP.get(int(i), f'Clase {i}') for i in report_df.index]
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
<div style="text-align: center; color: #666; padding: 2rem 0; font-size: 0.9rem;">
    <p><strong style="color: #1a1a1a;">SistemaPredict</strong> - Sistema de Diagnóstico Médico con Machine Learning</p>
    <p>Proyecto Final - Análisis de Datos 2025</p>
</div>
""", unsafe_allow_html=True)
