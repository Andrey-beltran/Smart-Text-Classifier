# 🚀 Deployment Guide - Smart Text Classifier

## 📋 Tabla de Contenidos
- [Deployment Local](#deployment-local)
- [Deployment en Cloud](#deployment-en-cloud)
- [Docker](#docker)
- [Heroku](#heroku)
- [Streamlit Cloud](#streamlit-cloud)
- [AWS](#aws)
- [Google Cloud](#google-cloud)
- [Azure](#azure)
- [Configuración de Producción](#configuración-de-producción)
- [Monitoreo y Logging](#monitoreo-y-logging)
- [Troubleshooting](#troubleshooting)

## 🏠 Deployment Local

### **Requisitos del Sistema**

```bash
# Sistema operativo
- Windows 10/11, macOS 10.14+, Ubuntu 18.04+
- Python 3.11 o superior
- 4GB RAM mínimo (8GB recomendado)
- 2GB espacio en disco

# Dependencias del sistema
- Git
- pip (gestor de paquetes Python)
- Visual Studio Build Tools (Windows)
```

### **Instalación Paso a Paso**

#### **1. Clonar el Repositorio**
```bash
git clone https://github.com/tu-usuario/smart-text-classifier.git
cd smart-text-classifier
```

#### **2. Crear Entorno Virtual**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

#### **3. Instalar Dependencias**
```bash
pip install -r requirements.txt
```

#### **4. Verificar Instalación**
```bash
python -c "import streamlit; print('Streamlit instalado correctamente')"
```

#### **5. Ejecutar la Aplicación**
```bash
# Modo simple (recomendado)
py -3.11 -m streamlit run app_simple.py

# Modo avanzado (requiere PyTorch)
py -3.11 -m streamlit run app.py
```

#### **6. Acceder a la Aplicación**
- **URL Local**: http://localhost:8501
- **URL de Red**: http://192.168.x.x:8501

### **Configuración Avanzada**

#### **Archivo de Configuración Streamlit**
```toml
# .streamlit/config.toml
[server]
port = 8501
headless = true
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

#### **Variables de Entorno**
```bash
# .env
MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
CONFIDENCE_THRESHOLD=0.7
MAX_LENGTH=128
BATCH_SIZE=32
LOG_LEVEL=INFO
```

## ☁️ Deployment en Cloud

### **Streamlit Cloud (Recomendado)**

#### **1. Preparar el Repositorio**
```bash
# Asegurar que el código esté en GitHub
git add .
git commit -m "Preparar para deployment"
git push origin main
```

#### **2. Conectar con Streamlit Cloud**
1. Ir a [share.streamlit.io](https://share.streamlit.io)
2. Conectar cuenta de GitHub
3. Seleccionar repositorio
4. Configurar archivo principal: `app_simple.py`

#### **3. Configuración de Deployment**
```toml
# .streamlit/config.toml
[server]
port = 8501
headless = true
enableCORS = false

[browser]
gatherUsageStats = false
```

#### **4. Variables de Entorno en Streamlit Cloud**
```bash
# En la interfaz de Streamlit Cloud
MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
CONFIDENCE_THRESHOLD=0.7
LOG_LEVEL=INFO
```

### **Heroku**

#### **1. Preparar Archivos**
```bash
# Procfile
echo "web: streamlit run app_simple.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# runtime.txt
echo "python-3.11.9" > runtime.txt
```

#### **2. Configurar Heroku**
```bash
# Instalar Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Crear aplicación
heroku create smart-text-classifier

# Configurar variables de entorno
heroku config:set MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
heroku config:set CONFIDENCE_THRESHOLD=0.7
heroku config:set LOG_LEVEL=INFO
```

#### **3. Deploy**
```bash
# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main

# Ver logs
heroku logs --tail
```

### **Docker**

#### **1. Crear Dockerfile**
```dockerfile
# Dockerfile
FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY . .

# Exponer puerto
EXPOSE 8501

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "app_simple.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### **2. Crear .dockerignore**
```dockerignore
# .dockerignore
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis

.DS_Store
.vscode
.idea
```

#### **3. Construir y Ejecutar**
```bash
# Construir imagen
docker build -t smart-text-classifier .

# Ejecutar contenedor
docker run -p 8501:8501 smart-text-classifier

# Ejecutar en background
docker run -d -p 8501:8501 --name smart-text-classifier smart-text-classifier
```

#### **4. Docker Compose**
```yaml
# docker-compose.yml
version: '3.8'

services:
  smart-text-classifier:
    build: .
    ports:
      - "8501:8501"
    environment:
      - MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
      - CONFIDENCE_THRESHOLD=0.7
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

```bash
# Ejecutar con Docker Compose
docker-compose up -d
```

## 🌐 AWS

### **AWS EC2**

#### **1. Crear Instancia EC2**
```bash
# Instancia recomendada
- Tipo: t3.medium (2 vCPU, 4GB RAM)
- AMI: Ubuntu 20.04 LTS
- Almacenamiento: 20GB SSD
```

#### **2. Configurar Servidor**
```bash
# Conectar por SSH
ssh -i "tu-key.pem" ubuntu@tu-ip

# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar Python 3.11
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv -y

# Instalar Git
sudo apt install git -y
```

#### **3. Deploy de la Aplicación**
```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/smart-text-classifier.git
cd smart-text-classifier

# Crear entorno virtual
python3.11 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
streamlit run app_simple.py --server.port=8501 --server.address=0.0.0.0
```

#### **4. Configurar Nginx (Opcional)**
```bash
# Instalar Nginx
sudo apt install nginx -y

# Configurar proxy reverso
sudo nano /etc/nginx/sites-available/smart-text-classifier

# Contenido del archivo
server {
    listen 80;
    server_name tu-dominio.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}

# Habilitar sitio
sudo ln -s /etc/nginx/sites-available/smart-text-classifier /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### **AWS Elastic Beanstalk**

#### **1. Preparar Aplicación**
```bash
# Crear archivo de configuración
mkdir .ebextensions
nano .ebextensions/python.config

# Contenido del archivo
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: application.py
  aws:elasticbeanstalk:application:environment:
    PYTHONPATH: "/var/app/current:$PYTHONPATH"
```

#### **2. Crear application.py**
```python
# application.py
import os
import subprocess
import sys

def application(environ, start_response):
    # Ejecutar Streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", "app_simple.py", "--server.port=8501", "--server.address=0.0.0.0"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Simular respuesta WSGI
    status = '200 OK'
    headers = [('Content-type', 'text/html')]
    start_response(status, headers)
    
    return [b'<html><body><h1>Smart Text Classifier</h1><p>Application is running</p></body></html>']
```

#### **3. Deploy con EB CLI**
```bash
# Instalar EB CLI
pip install awsebcli

# Inicializar aplicación
eb init

# Crear entorno
eb create production

# Deploy
eb deploy
```

## 🔵 Google Cloud

### **Google Cloud Run**

#### **1. Preparar Dockerfile**
```dockerfile
# Dockerfile para Cloud Run
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["streamlit", "run", "app_simple.py", "--server.port=8080", "--server.address=0.0.0.0"]
```

#### **2. Deploy con gcloud CLI**
```bash
# Instalar gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Configurar proyecto
gcloud config set project tu-proyecto-id

# Construir y subir imagen
gcloud builds submit --tag gcr.io/tu-proyecto-id/smart-text-classifier

# Deploy a Cloud Run
gcloud run deploy smart-text-classifier \
  --image gcr.io/tu-proyecto-id/smart-text-classifier \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080
```

### **Google App Engine**

#### **1. Crear app.yaml**
```yaml
# app.yaml
runtime: python311

env_variables:
  MODEL_NAME: "distilbert-base-uncased-finetuned-sst-2-english"
  CONFIDENCE_THRESHOLD: "0.7"
  LOG_LEVEL: "INFO"

automatic_scaling:
  min_instances: 1
  max_instances: 10
  target_cpu_utilization: 0.6

resources:
  cpu: 1
  memory_gb: 2
  disk_size_gb: 10
```

#### **2. Crear main.py**
```python
# main.py
import os
import subprocess
import sys

def main():
    # Ejecutar Streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", "app_simple.py", "--server.port=8080", "--server.address=0.0.0.0"]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
```

#### **3. Deploy**
```bash
# Deploy a App Engine
gcloud app deploy

# Ver aplicación
gcloud app browse
```

## 🔷 Azure

### **Azure App Service**

#### **1. Preparar Aplicación**
```bash
# Crear archivo de configuración
nano web.config

# Contenido del archivo
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <system.webServer>
    <handlers>
      <add name="PythonHandler" path="*" verb="*" modules="httpPlatformHandler" resourceType="Unspecified"/>
    </handlers>
    <httpPlatform processPath="D:\home\Python311\python.exe" arguments="D:\home\site\wwwroot\app_simple.py" stdoutLogEnabled="true" stdoutLogFile="D:\home\LogFiles\python.log" startupTimeLimit="60" startupRetryCount="3">
    </httpPlatform>
  </system.webServer>
</configuration>
```

#### **2. Deploy con Azure CLI**
```bash
# Instalar Azure CLI
# https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

# Login
az login

# Crear grupo de recursos
az group create --name smart-text-classifier-rg --location eastus

# Crear plan de App Service
az appservice plan create --name smart-text-classifier-plan --resource-group smart-text-classifier-rg --sku B1 --is-linux

# Crear aplicación web
az webapp create --resource-group smart-text-classifier-rg --plan smart-text-classifier-plan --name smart-text-classifier-app --runtime "PYTHON|3.11"

# Deploy código
az webapp deployment source config --resource-group smart-text-classifier-rg --name smart-text-classifier-app --repo-url https://github.com/tu-usuario/smart-text-classifier.git --branch main --manual-integration
```

## ⚙️ Configuración de Producción

### **Variables de Entorno**

```bash
# .env.production
MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
CONFIDENCE_THRESHOLD=0.7
MAX_LENGTH=128
BATCH_SIZE=32
LOG_LEVEL=INFO
ENVIRONMENT=production
DEBUG=false
```

### **Configuración de Logging**

```python
# logging_config.py
import logging
import os
from datetime import datetime

def setup_logging():
    """Configura logging para producción"""
    
    # Crear directorio de logs
    os.makedirs('logs', exist_ok=True)
    
    # Configurar logging
    logging.basicConfig(
        level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)
```

### **Configuración de Seguridad**

```python
# security.py
import os
import secrets

class SecurityConfig:
    """Configuración de seguridad"""
    
    SECRET_KEY = os.getenv('SECRET_KEY', secrets.token_hex(32))
    ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    @staticmethod
    def validate_request(request):
        """Valida solicitudes entrantes"""
        # Implementar validación de seguridad
        pass
```

## 📊 Monitoreo y Logging

### **Health Check**

```python
# health_check.py
import streamlit as st
import time
import psutil
import os

def health_check():
    """Verifica el estado del sistema"""
    
    health_status = {
        'status': 'healthy',
        'timestamp': time.time(),
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'python_version': os.sys.version,
        'streamlit_version': st.__version__
    }
    
    return health_status

def display_health_status():
    """Muestra el estado de salud del sistema"""
    
    health = health_check()
    
    st.subheader("🏥 Estado del Sistema")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("CPU", f"{health['cpu_percent']:.1f}%")
    
    with col2:
        st.metric("Memoria", f"{health['memory_percent']:.1f}%")
    
    with col3:
        st.metric("Disco", f"{health['disk_percent']:.1f}%")
    
    # Mostrar detalles
    with st.expander("Detalles del Sistema"):
        st.json(health)
```

### **Métricas de Rendimiento**

```python
# metrics.py
import time
from collections import defaultdict
from typing import Dict, List

class PerformanceMetrics:
    """Métricas de rendimiento del sistema"""
    
    def __init__(self):
        self.request_count = 0
        self.total_processing_time = 0
        self.error_count = 0
        self.sentiment_counts = defaultdict(int)
        
    def record_request(self, processing_time: float, sentiment: str, error: bool = False):
        """Registra una solicitud"""
        self.request_count += 1
        self.total_processing_time += processing_time
        
        if error:
            self.error_count += 1
        else:
            self.sentiment_counts[sentiment] += 1
    
    def get_metrics(self) -> Dict:
        """Obtiene métricas actuales"""
        avg_processing_time = self.total_processing_time / max(self.request_count, 1)
        error_rate = self.error_count / max(self.request_count, 1)
        
        return {
            'total_requests': self.request_count,
            'average_processing_time': avg_processing_time,
            'error_rate': error_rate,
            'sentiment_distribution': dict(self.sentiment_counts)
        }
```

## 🐛 Troubleshooting

### **Problemas Comunes**

#### **1. Error de PyTorch**
```bash
# Problema: OSError: [WinError 1114] Error en una rutina de inicialización de biblioteca de vínculos dinámicos (DLL)

# Solución 1: Usar modelo simple
py -3.11 -m streamlit run app_simple.py

# Solución 2: Reinstalar PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Solución 3: Instalar Visual Studio Build Tools
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

#### **2. Error de Puerto**
```bash
# Problema: Puerto 8501 ya está en uso

# Solución: Cambiar puerto
streamlit run app_simple.py --server.port 8502
```

#### **3. Error de Memoria**
```bash
# Problema: Out of memory

# Solución: Reducir batch size
export BATCH_SIZE=16
streamlit run app_simple.py
```

#### **4. Error de Dependencias**
```bash
# Problema: ModuleNotFoundError

# Solución: Reinstalar dependencias
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

### **Logs de Debugging**

```python
# debug.py
import logging
import traceback

def debug_classification(text: str):
    """Debug de clasificación"""
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    try:
        logger.debug(f"Clasificando texto: {text[:50]}...")
        
        # Tu código de clasificación aquí
        result = classify_text(text)
        
        logger.debug(f"Resultado: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error en clasificación: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
```

### **Monitoreo de Errores**

```python
# error_monitoring.py
import logging
from datetime import datetime

class ErrorMonitor:
    """Monitor de errores"""
    
    def __init__(self):
        self.error_log = []
        
    def log_error(self, error: Exception, context: str = ""):
        """Registra un error"""
        error_entry = {
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context
        }
        
        self.error_log.append(error_entry)
        
        # Log también en archivo
        logger = logging.getLogger(__name__)
        logger.error(f"Error: {error_entry}")
        
    def get_error_summary(self) -> dict:
        """Obtiene resumen de errores"""
        if not self.error_log:
            return {'total_errors': 0}
            
        error_types = {}
        for error in self.error_log:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
        return {
            'total_errors': len(self.error_log),
            'error_types': error_types,
            'recent_errors': self.error_log[-5:]  # Últimos 5 errores
        }
```

---

**Esta guía de deployment proporciona todas las opciones necesarias para desplegar tu Smart Text Classifier en diferentes entornos, desde local hasta cloud.**
