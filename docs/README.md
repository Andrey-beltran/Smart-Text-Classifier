# 📚 Documentación - Smart Text Classifier

## 🎯 Bienvenido a la Documentación

Esta documentación proporciona una guía completa para usar, desplegar y extender el **Smart Text Classifier**. El proyecto combina inteligencia artificial avanzada con una interfaz web interactiva para el análisis de sentimientos en múltiples idiomas.

## 📋 Tabla de Contenidos

### **🚀 Inicio Rápido**
- [README.md](README.md) - Documentación principal del proyecto
- [Instalación](README.md#instalación) - Guía de instalación paso a paso
- [Uso Básico](README.md#uso) - Ejemplos de uso inmediato

### **🏗️ Arquitectura y Desarrollo**
- [Arquitectura del Sistema](architecture.md) - Diseño y componentes del sistema
- [Referencia de API](api_reference.md) - Documentación completa de la API
- [Ejemplos de Uso](examples.md) - Casos de uso y ejemplos prácticos

### **🚀 Deployment y Producción**
- [Guías de Deployment](deployment.md) - Despliegue en diferentes entornos
- [Configuración de Producción](deployment.md#configuración-de-producción) - Configuración para producción
- [Monitoreo y Logging](deployment.md#monitoreo-y-logging) - Monitoreo del sistema

### **📊 Características Principales**

#### **Análisis de Sentimientos**
- ✅ **Individual**: Clasificación de texto único
- ✅ **Por Lotes**: Procesamiento masivo de textos
- ✅ **CSV**: Importación y análisis de archivos
- ✅ **Multilingüe**: Soporte para 6 idiomas

#### **Interfaz Web**
- ✅ **Streamlit**: Dashboard interactivo
- ✅ **Visualizaciones**: Gráficos dinámicos con Plotly
- ✅ **Métricas**: Estadísticas detalladas
- ✅ **Descarga**: Exportación de resultados

#### **Tecnologías**
- ✅ **Python 3.11**: Lenguaje principal
- ✅ **Streamlit**: Framework web
- ✅ **Hugging Face**: Modelos de IA
- ✅ **Plotly**: Visualizaciones interactivas

## 🎯 Casos de Uso

### **1. Análisis de Reseñas de Productos**
```python
from model.predict_simple import classify_batch

reseñas = [
    "I love this product!",
    "This is terrible!",
    "It's okay, nothing special."
]

resultados = classify_batch(reseñas)
for reseña, (sentimiento, confianza) in zip(reseñas, resultados):
    print(f"{reseña} -> {sentimiento} ({confianza:.3f})")
```

### **2. Análisis de Comentarios en Redes Sociales**
```python
from multilingual import classify_text_multilingual, detect_language

texto = "Me encanta este producto!"
idioma = detect_language(texto)
sentimiento, confianza = classify_text_multilingual(texto, idioma)
print(f"Idioma: {idioma}, Sentimiento: {sentimiento}")
```

### **3. Dashboard Interactivo**
```bash
# Ejecutar aplicación
py -3.11 -m streamlit run app_simple.py

# Acceder en navegador
# http://localhost:8501
```

## 🛠️ Instalación Rápida

### **Requisitos**
- Python 3.11 o superior
- pip (gestor de paquetes Python)
- Git (para clonar el repositorio)

### **Instalación**
```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/smart-text-classifier.git
cd smart-text-classifier

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
py -3.11 -m streamlit run app_simple.py
```

## 📊 Ejemplos de Uso

### **Análisis Individual**
```python
from model.predict_simple import classify_text

texto = "I love this product!"
sentimiento, confianza = classify_text(texto)
print(f"Sentimiento: {sentimiento}, Confianza: {confianza:.3f}")
```

### **Análisis por Lotes**
```python
from model.predict_simple import classify_batch

textos = ["I love this!", "This is terrible!", "It's okay."]
resultados = classify_batch(textos)
for texto, (sent, conf) in zip(textos, resultados):
    print(f"{texto} -> {sent} ({conf:.3f})")
```

### **Análisis Multilingüe**
```python
from multilingual import classify_text_multilingual

texto_es = "Me encanta este producto!"
sentimiento, confianza = classify_text_multilingual(texto_es, "es")
print(f"Sentimiento: {sentimiento}, Confianza: {confianza:.3f}")
```

## 🚀 Deployment

### **Local**
```bash
py -3.11 -m streamlit run app_simple.py
```

### **Streamlit Cloud**
1. Subir código a GitHub
2. Conectar con Streamlit Cloud
3. Deploy automático

### **Docker**
```bash
docker build -t smart-text-classifier .
docker run -p 8501:8501 smart-text-classifier
```

### **Heroku**
```bash
heroku create smart-text-classifier
git push heroku main
```

## 🔧 Configuración

### **Variables de Entorno**
```bash
MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
CONFIDENCE_THRESHOLD=0.7
MAX_LENGTH=128
BATCH_SIZE=32
LOG_LEVEL=INFO
```

### **Archivo de Configuración**
```toml
# .streamlit/config.toml
[server]
port = 8501
headless = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
```

## 📈 Rendimiento

### **Métricas Típicas**
- **Velocidad**: 500-1000 textos/segundo
- **Memoria**: 100-200MB RAM
- **Precisión**: 85-95% (dependiendo del modelo)
- **Latencia**: 1-5ms por texto

### **Optimizaciones**
- Caching de modelos
- Procesamiento por lotes
- Lazy loading
- Connection pooling

## 🐛 Troubleshooting

### **Problemas Comunes**

#### **Error de PyTorch**
```bash
# Solución: Usar modelo simple
py -3.11 -m streamlit run app_simple.py
```

#### **Error de Puerto**
```bash
# Solución: Cambiar puerto
streamlit run app_simple.py --server.port 8502
```

#### **Error de Dependencias**
```bash
# Solución: Reinstalar dependencias
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

## 🤝 Contribuciones

### **Cómo Contribuir**
1. Fork del repositorio
2. Crear rama feature
3. Commit cambios
4. Push a la rama
5. Crear Pull Request

### **Áreas de Contribución**
- 🐛 Bug fixes
- ✨ Nuevas funcionalidades
- 📚 Documentación
- 🧪 Tests
- 🌍 Idiomas

## 📞 Soporte

### **Obtener Ayuda**
- 📧 **Email**: tu-email@ejemplo.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/tu-usuario/smart-text-classifier/issues)
- 💬 **Discusiones**: [GitHub Discussions](https://github.com/tu-usuario/smart-text-classifier/discussions)

### **Recursos Adicionales**
- 📚 **Documentación**: [docs/](docs/)
- 🎥 **Tutoriales**: [YouTube Channel](https://youtube.com/tu-canal)
- 📖 **Blog**: [Medium](https://medium.com/@tu-usuario)

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- **Hugging Face**: Por los modelos pre-entrenados
- **Streamlit**: Por el framework web
- **Plotly**: Por las visualizaciones interactivas
- **Comunidad Python**: Por las librerías de código abierto

---

<div align="center">

**¡Gracias por usar Smart Text Classifier!** 🎉

[⭐ Star](https://github.com/tu-usuario/smart-text-classifier) | [🐛 Report Bug](https://github.com/tu-usuario/smart-text-classifier/issues) | [✨ Request Feature](https://github.com/tu-usuario/smart-text-classifier/issues)

</div>
