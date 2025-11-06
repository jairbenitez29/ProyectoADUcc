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
    initial_sidebar_state="collapsed"
)

# Versión: 2.0 - Menú integrado en header

# CSS personalizado para diseño profesional con colores rojo, blanco y negro
st.markdown("""
<style>
    /* Imports */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Ocultar botones de navegación completamente */
    button[kind="primary"] {
        position: absolute !important;
        left: -10000px !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        pointer-events: all !important; /* Mantener clickeable */
        visibility: hidden !important;
    }

    /* Ocultar el contenedor de los botones de navegación */
    div[data-testid="stHorizontalBlock"]:first-of-type,
    div[data-testid="column"]:has(button[key*="nav_btn"]) {
        position: absolute !important;
        left: -10000px !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        opacity: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Remove default padding */
    .block-container {
        padding-top: 7rem !important;
        max-width: 100% !important;
    }

    /* Sticky Navbar */
    .navbar {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        width: 100% !important;
        z-index: 9999 !important;
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
        padding: 0 !important;
        margin: 0 !important;
        display: block !important;
        visibility: visible !important;
    }

    .navbar-container {
        max-width: 1400px;
        margin: 0 auto;
        display: flex !important;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 2rem;
    }

    .navbar-brand {
        display: flex !important;
        align-items: center;
        gap: 0.75rem;
        flex: 1;
        justify-content: center;
    }

    .brand-logo {
        font-size: 1.8rem;
        color: #dc2626 !important;
        font-weight: 800;
    }

    .brand-text {
        font-size: 1.5rem;
        font-weight: 700;
        color: white !important;
        letter-spacing: -0.5px;
    }

    /* Botón Hamburguesa */
    .hamburger-btn {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        width: 45px;
        height: 45px;
        background: transparent;
        border: 2px solid rgba(220, 38, 38, 0.5);
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .hamburger-btn:hover {
        background: rgba(220, 38, 38, 0.1);
        border-color: #dc2626;
    }

    .hamburger-line {
        width: 25px;
        height: 3px;
        background: #dc2626;
        margin: 3px 0;
        transition: all 0.3s ease;
        border-radius: 2px;
    }

    /* Sidebar Menu */
    .sidebar-menu {
        position: fixed;
        top: 0;
        left: -320px;
        width: 320px;
        height: 100vh;
        background: linear-gradient(180deg, #1a1a1a 0%, #2d2d2d 100%);
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.5);
        z-index: 1000;
        transition: left 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        overflow-y: auto;
        padding: 2rem 0;
    }

    .sidebar-menu.open {
        left: 0;
    }

    .sidebar-header {
        padding: 0 1.5rem 1.5rem 1.5rem;
        border-bottom: 2px solid rgba(220, 38, 38, 0.3);
        margin-bottom: 1rem;
    }

    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .sidebar-icon {
        color: #dc2626;
        font-size: 1.8rem;
    }

    .sidebar-close {
        position: absolute;
        top: 1.5rem;
        right: 1.5rem;
        width: 35px;
        height: 35px;
        background: rgba(220, 38, 38, 0.1);
        border: 2px solid rgba(220, 38, 38, 0.3);
        border-radius: 6px;
        color: #dc2626;
        font-size: 1.2rem;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
    }

    .sidebar-close:hover {
        background: #dc2626;
        color: white;
        transform: rotate(90deg);
    }

    .sidebar-nav {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        padding: 1rem;
    }

    .sidebar-nav-item {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.5rem;
        color: #e0e0e0;
        background: transparent;
        border: 2px solid transparent;
        border-radius: 10px;
        text-decoration: none;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .sidebar-nav-item:hover {
        background: rgba(220, 38, 38, 0.1);
        border-color: rgba(220, 38, 38, 0.3);
        color: white;
        transform: translateX(5px);
    }

    .sidebar-nav-item.active {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        border-color: #dc2626;
        color: white;
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.4);
    }

    .sidebar-nav-item i {
        font-size: 1.3rem;
        width: 25px;
        text-align: center;
    }

    /* Overlay oscuro cuando el menú está abierto */
    .sidebar-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(0, 0, 0, 0.6);
        z-index: 999;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
    }

    .sidebar-overlay.show {
        opacity: 1;
        visibility: visible;
    }

    /* Timer visual en el sidebar */
    .sidebar-timer {
        padding: 0.75rem 1.5rem;
        margin: 1rem 1.5rem;
        background: rgba(220, 38, 38, 0.1);
        border: 1px solid rgba(220, 38, 38, 0.3);
        border-radius: 8px;
        color: #e0e0e0;
        font-size: 0.85rem;
        text-align: center;
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
    labels = [f"Clase {i+1}" for i in range(len(cm))]

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
        colorbar=dict(title="Cantidad", titlefont=dict(color='#1a1a1a'))
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

# Selector de modelo (sin usar el sidebar de Streamlit)
# Mantener funcionalidad pero sin mostrar el sidebar nativo

# Navbar con Menú Lateral
model_display = "Red Neuronal" if st.session_state.model_type == "neural" else "Regresión Logística"

# Botones ocultos para manejar la navegación (NO SE VERÁN)
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("nav_inicio", key="nav_btn_inicio", type="primary"):
        st.session_state.current_page = 'inicio'
with col2:
    if st.button("nav_individual", key="nav_btn_individual", type="primary"):
        st.session_state.current_page = 'individual'
with col3:
    if st.button("nav_lotes", key="nav_btn_lotes", type="primary"):
        st.session_state.current_page = 'lotes'
with col4:
    if st.button("nav_metricas", key="nav_btn_metricas", type="primary"):
        st.session_state.current_page = 'metricas'

# Crear el navbar y sidebar HTML
current_page = st.session_state.current_page
inicio_active = "active" if current_page == "inicio" else ""
individual_active = "active" if current_page == "individual" else ""
lotes_active = "active" if current_page == "lotes" else ""
metricas_active = "active" if current_page == "metricas" else ""

# Renderizar todo el HTML necesario
st.markdown(f"""
<div class="sidebar-overlay" id="sidebarOverlay"></div>

<div class="sidebar-menu" id="sidebarMenu">
    <div class="sidebar-close" id="sidebarClose">
        <i class="fas fa-times"></i>
    </div>
    <div class="sidebar-header">
        <div class="sidebar-title">
            <i class="fas fa-heartbeat sidebar-icon"></i>
            <span>SistemaPredict</span>
        </div>
    </div>
    <div class="sidebar-timer" id="sidebarTimer">
        Se cerrará automáticamente en <span id="timerCount">5</span>s
    </div>
    <nav class="sidebar-nav">
        <div class="sidebar-nav-item {inicio_active}" data-page="inicio">
            <i class="fas fa-home"></i>
            <span>Inicio</span>
        </div>
        <div class="sidebar-nav-item {individual_active}" data-page="individual">
            <i class="fas fa-user-md"></i>
            <span>Predicción Individual</span>
        </div>
        <div class="sidebar-nav-item {lotes_active}" data-page="lotes">
            <i class="fas fa-database"></i>
            <span>Predicción por Lotes</span>
        </div>
        <div class="sidebar-nav-item {metricas_active}" data-page="metricas">
            <i class="fas fa-chart-line"></i>
            <span>Métricas de Modelos</span>
        </div>
    </nav>
    <div style="padding: 0 1.5rem; margin-top: 2rem;">
        <div style="padding: 1rem; background: rgba(220, 38, 38, 0.1); border: 1px solid rgba(220, 38, 38, 0.3); border-radius: 8px;">
            <p style="color: #e0e0e0; font-size: 0.85rem; margin: 0 0 0.5rem 0; font-weight: 600;">Modelo Activo:</p>
            <p style="color: #dc2626; font-size: 1rem; margin: 0; font-weight: 700;">{model_display}</p>
        </div>
    </div>
</div>

<div class="navbar">
    <div class="navbar-container">
        <div class="hamburger-btn" id="hamburgerBtn">
            <div class="hamburger-line"></div>
            <div class="hamburger-line"></div>
            <div class="hamburger-line"></div>
        </div>
        <div class="navbar-brand">
            <div class="brand-logo"><i class="fas fa-heartbeat"></i></div>
            <div class="brand-text">SistemaPredict</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# JavaScript para funcionalidad del menú
st.markdown("""
<script>
    let autoCloseTimer = null;
    let countdownTimer = null;
    let timeRemaining = 5;

    const sidebar = document.getElementById('sidebarMenu');
    const overlay = document.getElementById('sidebarOverlay');
    const hamburger = document.getElementById('hamburgerBtn');
    const closeBtn = document.getElementById('sidebarClose');
    const timerDisplay = document.getElementById('timerCount');
    const navItems = document.querySelectorAll('.sidebar-nav-item');

    function openSidebar() {
        sidebar.classList.add('open');
        overlay.classList.add('show');
        startAutoClose();
    }

    function closeSidebar() {
        sidebar.classList.remove('open');
        overlay.classList.remove('show');
        clearTimers();
        resetTimer();
    }

    function startAutoClose() {
        timeRemaining = 5;
        updateTimerDisplay();
        countdownTimer = setInterval(() => {
            timeRemaining--;
            updateTimerDisplay();
            if (timeRemaining <= 0) {
                closeSidebar();
            }
        }, 1000);
    }

    function updateTimerDisplay() {
        if (timerDisplay) {
            timerDisplay.textContent = timeRemaining;
        }
    }

    function clearTimers() {
        if (autoCloseTimer) clearTimeout(autoCloseTimer);
        if (countdownTimer) clearInterval(countdownTimer);
    }

    function resetTimer() {
        timeRemaining = 5;
        updateTimerDisplay();
    }

    hamburger.addEventListener('click', openSidebar);
    closeBtn.addEventListener('click', closeSidebar);
    overlay.addEventListener('click', closeSidebar);

    const pageMapping = {
        'inicio': 0,
        'individual': 1,
        'lotes': 2,
        'metricas': 3
    };

    navItems.forEach(item => {
        item.addEventListener('click', function() {
            const page = this.getAttribute('data-page');
            const index = pageMapping[page];
            const buttons = document.querySelectorAll('button[kind="primary"]');
            if (buttons[index]) {
                buttons[index].click();
                closeSidebar();
            }
        });
    });

    sidebar.addEventListener('mouseenter', () => {
        clearTimers();
        resetTimer();
        startAutoClose();
    });
</script>
""", unsafe_allow_html=True)

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

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **1. Recopilación de Datos**

        El sistema recibe información del paciente: datos demográficos, síntomas reportados
        y resultados de exámenes de laboratorio.

        **2. Preprocesamiento**

        Los datos se normalizan utilizando StandardScaler para asegurar que todas las
        características tengan la misma escala.
        """)

    with col2:
        st.markdown("""
        **3. Predicción**

        El modelo seleccionado (Regresión Logística o Red Neuronal) procesa los datos
        y genera una predicción con probabilidades.

        **4. Interpretación**

        El sistema presenta el diagnóstico predicho junto con el nivel de confianza
        y las probabilidades para cada clase.
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
elif st.session_state.current_page == 'individual':
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Predicción Individual</div>
        <div class="page-subtitle">Diagnóstico personalizado para un paciente</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <p><strong>Instrucciones:</strong> Complete los datos del paciente. Los campos binarios aceptan 0 (No) o 1 (Sí).</p>
    </div>
    """, unsafe_allow_html=True)

    # Formulario
    with st.form("prediction_form"):
        st.markdown('<div class="card"><div class="card-header"><i class="fas fa-user-injured card-icon"></i><span class="card-title">Datos del Paciente</span></div></div>', unsafe_allow_html=True)

        cols = st.columns(3)
        feature_values = {}

        for idx, feature in enumerate(feature_names):
            col = cols[idx % 3]
            with col:
                if feature in ['male', 'female'] or any(x in feature for x in ['fever', 'headache', 'dizziness', 'weakness', 'vomiting']):
                    feature_values[feature] = st.number_input(
                        format_feature_name(feature),
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=1.0,
                        key=feature,
                        help="0 = No, 1 = Sí"
                    )
                elif 'age' in feature.lower() or 'edad' in feature.lower():
                    # Límite de edad hasta 80 años
                    feature_values[feature] = st.number_input(
                        format_feature_name(feature),
                        min_value=0.0,
                        max_value=80.0,
                        value=0.0,
                        step=1.0,
                        key=feature,
                        help="Edad máxima: 80 años"
                    )
                else:
                    feature_values[feature] = st.number_input(
                        format_feature_name(feature),
                        value=0.0,
                        step=0.01,
                        format="%.2f",
                        key=feature
                    )

        submitted = st.form_submit_button("Realizar Predicción")

        if submitted:
            try:
                X = np.array([[feature_values[feat] for feat in feature_names]])
                X_scaled = models['scaler'].transform(X)
                model = models[st.session_state.model_type]

                prediction = int(model.predict(X_scaled)[0])
                probabilities = model.predict_proba(X_scaled)[0]
                confidence = max(probabilities)

                st.markdown("---")

                diagnosis = DIAGNOSIS_MAP.get(prediction, "Desconocido")
                st.markdown(f'<div class="success-box">{diagnosis}</div>', unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicción", f"Clase {prediction}")
                with col2:
                    st.metric("Confianza", f"{confidence*100:.1f}%")
                with col3:
                    st.metric("Modelo", "Red Neuronal" if st.session_state.model_type == "neural" else "R. Logística")

                st.plotly_chart(plot_probabilities(probabilities, prediction), use_container_width=True)

                prob_df = pd.DataFrame({
                    'Clase': [f'Clase {i+1}' for i in range(len(probabilities))],
                    'Probabilidad': [f'{p*100:.2f}%' for p in probabilities]
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# PÁGINA: PREDICCIÓN POR LOTES
# ==========================================
elif st.session_state.current_page == 'lotes':
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Predicción por Lotes</div>
        <div class="page-subtitle">Procesamiento masivo de múltiples pacientes</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <p><strong>Requisitos:</strong> Archivo CSV o Excel con las mismas 55 columnas del entrenamiento.</p>
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

                        st.markdown("---")
                        st.markdown("### Resultados Completos")
                        st.dataframe(df, use_container_width=True)

                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Descargar Resultados (CSV)",
                            data=csv,
                            file_name="resultados_prediccion.csv",
                            mime="text/csv",
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
        <div class="page-subtitle">Comparación y evaluación del rendimiento</div>
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
<div style="text-align: center; color: #666; padding: 2rem 0; font-size: 0.9rem;">
    <p><strong style="color: #1a1a1a;">SistemaPredict</strong> - Sistema de Diagnóstico Médico con Machine Learning</p>
    <p>Proyecto Final - Análisis de Datos 2025</p>
</div>
""", unsafe_allow_html=True)
