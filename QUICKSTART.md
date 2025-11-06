# 🚀 Inicio Rápido - MediPredict AI

## 📦 Comandos Esenciales

### 1️⃣ Subir a GitHub (PRIMERO DEBES CREAR EL REPO EN GITHUB.COM)

```bash
# Añadir todos los archivos
git add .

# Crear commit inicial
git commit -m "Initial commit - MediPredict AI"

# Cambiar a rama main
git branch -M main

# Conectar con tu repositorio (CAMBIA LA URL)
git remote add origin https://github.com/TU-USUARIO/medipredict-ai.git

# Subir archivos
git push -u origin main
```

### 2️⃣ Desplegar en Streamlit Cloud

1. Ve a: https://share.streamlit.io
2. Sign in con GitHub
3. Click "New app"
4. Selecciona tu repositorio
5. Main file: `app.py`
6. Click "Deploy!"

## ✅ Checklist Pre-Deployment

- [ ] Los modelos están entrenados (`models/` existe con 5 archivos)
- [ ] Repositorio creado en GitHub
- [ ] Git está inicializado (`git init` ejecutado)
- [ ] Archivos añadidos al commit (`git add .`)
- [ ] Remote configurado (`git remote add origin`)

## 🔍 Verificar Modelos

```bash
# Verificar que existan los modelos
ls -la models/

# Debería mostrar:
# - logistic_regression.pkl
# - neural_network.pkl
# - scaler.pkl
# - feature_names.json
# - metrics.json
```

## 🧪 Probar Localmente Primero

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar app
streamlit run app.py
```

## 📝 Comandos Git Útiles

```bash
# Ver estado
git status

# Ver cambios
git diff

# Ver historial
git log --oneline

# Actualizar después de cambios
git add .
git commit -m "Mensaje descriptivo"
git push
```

## ❗ Troubleshooting Rápido

### Error: "models not found"
```bash
python train_models.py
```

### Error: "remote origin already exists"
```bash
git remote remove origin
git remote add origin TU-URL-AQUI
```

### Error: "failed to push"
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

## 🎯 URLs Importantes

- **GitHub**: https://github.com
- **Streamlit Cloud**: https://share.streamlit.io
- **Documentación Streamlit**: https://docs.streamlit.io

## 💡 Tip Pro

Después de desplegar, comparte tu app con:
```
https://tu-usuario-medipredict-ai.streamlit.app
```

---

📖 Para más detalles, consulta `DEPLOYMENT.md`
