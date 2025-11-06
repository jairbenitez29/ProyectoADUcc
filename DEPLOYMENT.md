# 📤 Guía de Deployment a Streamlit Cloud

Esta guía te ayudará paso a paso para subir tu proyecto a GitHub y desplegarlo en Streamlit Cloud.

## ✅ Pre-requisitos

Antes de comenzar, asegúrate de tener:
- ✅ Una cuenta de GitHub ([crear cuenta](https://github.com/join))
- ✅ Git instalado en tu computadora
- ✅ Los modelos ya entrenados (carpeta `models/` con archivos .pkl y .json)

## 📝 Paso 1: Preparar el Proyecto

### 1.1 Verificar que los modelos estén entrenados

```bash
# Entrenar los modelos (si aún no lo has hecho)
python train_models.py
```

Esto creará la carpeta `models/` con:
- `logistic_regression.pkl`
- `neural_network.pkl`
- `scaler.pkl`
- `feature_names.json`
- `metrics.json`

### 1.2 Verificar archivos necesarios

Asegúrate de tener estos archivos en tu proyecto:
- ✅ `app.py` - Aplicación principal
- ✅ `requirements.txt` - Dependencias
- ✅ `dataset.xlsx` - Dataset
- ✅ `models/` - Carpeta con modelos entrenados
- ✅ `.streamlit/config.toml` - Configuración
- ✅ `.gitignore` - Archivos a ignorar
- ✅ `README.md` - Documentación

## 🚀 Paso 2: Subir a GitHub

### 2.1 Crear un nuevo repositorio en GitHub

1. Ve a [github.com](https://github.com)
2. Haz clic en el botón **"+"** en la esquina superior derecha
3. Selecciona **"New repository"**
4. Configuración:
   - **Repository name**: `medipredict-ai` (o el nombre que prefieras)
   - **Description**: "Sistema de Diagnóstico Médico con Machine Learning"
   - **Public** o **Private** (recomiendo Public)
   - **NO** marques "Add a README file" (ya lo tenemos)
5. Haz clic en **"Create repository"**

### 2.2 Configurar Git en tu proyecto

Abre la terminal en la carpeta del proyecto y ejecuta:

```bash
# Inicializar repositorio git
git init

# Añadir todos los archivos
git add .

# Crear el primer commit
git commit -m "Initial commit - MediPredict AI"

# Cambiar a la rama main
git branch -M main

# Conectar con GitHub (REEMPLAZA con tu URL)
git remote add origin https://github.com/TU-USUARIO/medipredict-ai.git

# Subir los archivos
git push -u origin main
```

**⚠️ IMPORTANTE**: Reemplaza `TU-USUARIO` con tu nombre de usuario de GitHub.

### 2.3 Verificar que todo se subió correctamente

Ve a tu repositorio en GitHub y verifica que todos los archivos estén ahí, especialmente:
- La carpeta `models/` con los 5 archivos
- `app.py`
- `requirements.txt`
- `dataset.xlsx`

## ☁️ Paso 3: Deployment en Streamlit Cloud

### 3.1 Crear cuenta en Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Haz clic en **"Sign in"**
3. Selecciona **"Continue with GitHub"**
4. Autoriza Streamlit Cloud para acceder a tu GitHub

### 3.2 Desplegar la aplicación

1. Una vez dentro, haz clic en **"New app"**

2. Configuración del deployment:
   - **Repository**: Selecciona `TU-USUARIO/medipredict-ai`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL** (opcional): Personaliza la URL de tu app

3. Haz clic en **"Deploy!"**

### 3.3 Esperar el deployment

- Streamlit Cloud instalará las dependencias (puede tomar 2-5 minutos)
- Verás los logs en tiempo real
- Cuando termine, tu app estará disponible en: `https://tu-usuario-medipredict-ai.streamlit.app`

## 🎉 ¡Listo!

Tu aplicación ahora está desplegada y accesible desde cualquier lugar del mundo.

## 🔧 Actualizar la Aplicación

Cuando hagas cambios en tu código:

```bash
# Añadir cambios
git add .

# Crear commit
git commit -m "Descripción de los cambios"

# Subir a GitHub
git push
```

Streamlit Cloud detectará los cambios automáticamente y re-desplegará tu app.

## ❗ Solución de Problemas

### Problema 1: "ModuleNotFoundError"

**Solución**: Verifica que `requirements.txt` tenga todas las dependencias.

### Problema 2: "FileNotFoundError: models/"

**Solución**: Asegúrate de:
1. Haber ejecutado `python train_models.py` localmente
2. Haber subido la carpeta `models/` a GitHub
3. Verificar que la carpeta exista en tu repositorio de GitHub

### Problema 3: La app no carga los modelos

**Solución**:
1. Verifica en GitHub que los archivos `.pkl` estén en `models/`
2. Los archivos `.pkl` deben ser menores a 100MB (límite de GitHub)
3. Si son más grandes, considera usar Git LFS

### Problema 4: Errores de versión de dependencias

**Solución**: Actualiza `requirements.txt` con las versiones que funcionan localmente:

```bash
pip freeze > requirements.txt
```

## 📊 Monitoreo

En Streamlit Cloud puedes:
- ✅ Ver logs en tiempo real
- ✅ Ver métricas de uso
- ✅ Re-desplegar manualmente
- ✅ Ver analytics de la aplicación

## 🔒 Configuración Avanzada

### Secrets Management

Si necesitas variables de entorno secretas:

1. En Streamlit Cloud, ve a tu app
2. Haz clic en **"Settings"** → **"Secrets"**
3. Agrega tus secretos en formato TOML:

```toml
# Ejemplo
[database]
host = "localhost"
user = "admin"
password = "secret"
```

4. Accede en tu código:

```python
import streamlit as st
db_host = st.secrets["database"]["host"]
```

## 🎨 Personalización

### Cambiar URL de la app

1. Ve a tu app en Streamlit Cloud
2. Haz clic en **"Settings"** → **"General"**
3. Cambia el **"App URL"**
4. Guarda los cambios

### Configurar dominio personalizado

Solo disponible en el plan Business de Streamlit Cloud.

## 📞 Ayuda y Soporte

- 📖 Documentación oficial: [docs.streamlit.io](https://docs.streamlit.io)
- 💬 Foro de la comunidad: [discuss.streamlit.io](https://discuss.streamlit.io)
- 🐛 Reportar bugs: [github.com/streamlit/streamlit](https://github.com/streamlit/streamlit/issues)

---

¡Felicidades! 🎉 Ahora tu aplicación de Machine Learning está disponible para todo el mundo.
