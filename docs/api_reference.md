# 📚 API Reference - Smart Text Classifier

## 📋 Tabla de Contenidos
- [Clases Principales](#clases-principales)
- [Funciones de Utilidad](#funciones-de-utilidad)
- [Configuración](#configuración)
- [Ejemplos de Uso](#ejemplos-de-uso)
- [Códigos de Error](#códigos-de-error)
- [Mejores Prácticas](#mejores-prácticas)

## 🏗️ Clases Principales

### **SentimentClassifier**

Clasificador principal de sentimientos usando modelos de IA pre-entrenados.

```python
class SentimentClassifier:
    """Clasificador principal de sentimientos"""
    
    def __init__(self):
        """
        Inicializa el clasificador con modelo de IA.
        
        Raises:
            OSError: Si no se puede cargar el modelo PyTorch
            ImportError: Si faltan dependencias de transformers
        """
        
    def classify_text(self, text: str) -> Tuple[str, float]:
        """
        Clasifica un texto individual.
        
        Args:
            text (str): Texto a clasificar
            
        Returns:
            Tuple[str, float]: (sentimiento, confianza)
                - sentimiento: 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
                - confianza: Valor entre 0.0 y 1.0
                
        Raises:
            ValueError: Si el texto está vacío o es inválido
            RuntimeError: Si hay error en el modelo
            
        Example:
            >>> classifier = SentimentClassifier()
            >>> sentiment, confidence = classifier.classify_text("I love this!")
            >>> print(f"Sentimiento: {sentiment}, Confianza: {confidence:.3f}")
            Sentimiento: POSITIVE, Confianza: 0.95
        """
        
    def classify_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Clasifica múltiples textos de forma eficiente.
        
        Args:
            texts (List[str]): Lista de textos a clasificar
            
        Returns:
            List[Tuple[str, float]]: Lista de tuplas (sentimiento, confianza)
            
        Raises:
            ValueError: Si la lista está vacía
            RuntimeError: Si hay error en el procesamiento por lotes
            
        Example:
            >>> texts = ["I love this!", "This is terrible!", "It's okay."]
            >>> results = classifier.classify_batch(texts)
            >>> for i, (sent, conf) in enumerate(results):
            ...     print(f"Texto {i+1}: {sent} (confianza: {conf:.3f})")
            Texto 1: POSITIVE (confianza: 0.95)
            Texto 2: NEGATIVE (confianza: 0.90)
            Texto 3: NEUTRAL (confianza: 0.70)
        """
        
    def classify_with_confidence_threshold(self, text: str, threshold: float = 0.7) -> Tuple[str, float, str]:
        """
        Clasifica con umbral de confianza personalizado.
        
        Args:
            text (str): Texto a clasificar
            threshold (float): Umbral de confianza (0.0-1.0)
            
        Returns:
            Tuple[str, float, str]: (sentimiento, confianza, nivel_confianza)
                - nivel_confianza: 'Alta confianza', 'Confianza media', 'Baja confianza'
                
        Example:
            >>> sentiment, confidence, level = classifier.classify_with_confidence_threshold("This is okay", 0.8)
            >>> print(f"Sentimiento: {sentiment}, Nivel: {level}")
            Sentimiento: NEUTRAL, Nivel: Baja confianza
        """
        
    def get_model_info(self) -> dict:
        """
        Obtiene información del modelo cargado.
        
        Returns:
            dict: Información del modelo
                - model_name: Nombre del modelo
                - model_type: Tipo de modelo
                - device: Dispositivo de procesamiento
                - max_length: Longitud máxima de entrada
        """
```

### **SimpleSentimentClassifier**

Clasificador simple basado en reglas predefinidas.

```python
class SimpleSentimentClassifier:
    """Clasificador simple basado en reglas"""
    
    def __init__(self):
        """
        Inicializa el clasificador con palabras clave predefinidas.
        
        Attributes:
            positive_keywords (List[str]): Palabras positivas
            negative_keywords (List[str]): Palabras negativas
            neutral_keywords (List[str]): Palabras neutrales
        """
        
    def classify_text(self, text: str) -> Tuple[str, float]:
        """
        Clasifica usando reglas predefinidas.
        
        Args:
            text (str): Texto a clasificar
            
        Returns:
            Tuple[str, float]: (sentimiento, confianza)
            
        Logic:
            1. Busca palabras positivas -> POSITIVE (0.95)
            2. Busca palabras negativas -> NEGATIVE (0.90)
            3. Por defecto -> NEUTRAL (0.70)
            
        Example:
            >>> simple_classifier = SimpleSentimentClassifier()
            >>> sentiment, confidence = simple_classifier.classify_text("I love this!")
            >>> print(f"Sentimiento: {sentiment}, Confianza: {confidence}")
            Sentimiento: POSITIVE, Confianza: 0.95
        """
        
    def classify_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Clasifica múltiples textos con reglas.
        
        Args:
            texts (List[str]): Lista de textos
            
        Returns:
            List[Tuple[str, float]]: Lista de resultados
        """
        
    def classify_with_confidence_threshold(self, text: str, threshold: float = 0.7) -> Tuple[str, float, str]:
        """
        Clasifica con umbral de confianza personalizado.
        
        Args:
            text (str): Texto a clasificar
            threshold (float): Umbral de confianza
            
        Returns:
            Tuple[str, float, str]: (sentimiento, confianza, nivel_confianza)
        """
```

### **MultilingualSentimentClassifier**

Clasificador multilingüe con detección automática de idiomas.

```python
class MultilingualSentimentClassifier:
    """Clasificador multilingüe"""
    
    def __init__(self):
        """
        Inicializa el clasificador multilingüe.
        
        Attributes:
            classifiers (dict): Diccionario de clasificadores por idioma
            performance_monitor (PerformanceMonitor): Monitor de rendimiento
        """
        
    def detect_language(self, text: str) -> str:
        """
        Detecta el idioma del texto.
        
        Args:
            text (str): Texto a analizar
            
        Returns:
            str: Código de idioma (en, es, fr, de, it, pt)
            
        Raises:
            Exception: Si no se puede detectar el idioma
            
        Example:
            >>> classifier = MultilingualSentimentClassifier()
            >>> lang = classifier.detect_language("Me encanta este producto!")
            >>> print(f"Idioma detectado: {lang}")
            Idioma detectado: es
        """
        
    def classify_text(self, text: str, lang: str = None) -> Tuple[str, float]:
        """
        Clasifica texto en idioma específico.
        
        Args:
            text (str): Texto a clasificar
            lang (str, optional): Código de idioma. Si es None, se detecta automáticamente
            
        Returns:
            Tuple[str, float]: (sentimiento, confianza)
            
        Example:
            >>> sentiment, confidence = classifier.classify_text("Me encanta!", "es")
            >>> print(f"Sentimiento: {sentiment}, Confianza: {confidence:.3f}")
            Sentimiento: POSITIVE, Confianza: 0.92
        """
        
    def classify_batch(self, texts: List[str], langs: List[str] = None) -> List[Tuple[str, float]]:
        """
        Clasifica múltiples textos multilingües.
        
        Args:
            texts (List[str]): Lista de textos
            langs (List[str], optional): Lista de idiomas correspondientes
            
        Returns:
            List[Tuple[str, float]]: Lista de resultados
        """
```

## 🛠️ Funciones de Utilidad

### **PerformanceMonitor**

Monitor de rendimiento del sistema.

```python
class PerformanceMonitor:
    """Monitor de rendimiento del sistema"""
    
    def __init__(self):
        """Inicializa el monitor de rendimiento"""
        self.start_time = None
        self.end_time = None
        
    def start_timer(self):
        """Inicia el cronómetro"""
        
    def stop_timer(self):
        """Detiene el cronómetro"""
        
    def get_elapsed_time(self) -> float:
        """
        Obtiene el tiempo transcurrido.
        
        Returns:
            float: Tiempo en segundos
        """
        
    def get_processing_rate(self, num_items: int) -> float:
        """
        Calcula la tasa de procesamiento.
        
        Args:
            num_items (int): Número de elementos procesados
            
        Returns:
            float: Elementos por segundo
        """
```

### **Funciones de Conveniencia**

```python
def get_classifier() -> SentimentClassifier:
    """
    Obtiene una instancia singleton del clasificador.
    
    Returns:
        SentimentClassifier: Instancia única del clasificador
    """

def get_simple_classifier() -> SimpleSentimentClassifier:
    """
    Obtiene una instancia singleton del clasificador simple.
    
    Returns:
        SimpleSentimentClassifier: Instancia única del clasificador simple
    """

def get_multilingual_classifier() -> MultilingualSentimentClassifier:
    """
    Obtiene una instancia singleton del clasificador multilingüe.
    
    Returns:
        MultilingualSentimentClassifier: Instancia única del clasificador multilingüe
    """

# Funciones de compatibilidad
def classify_text(text: str) -> Tuple[str, float]:
    """
    Función de compatibilidad para clasificación simple.
    
    Args:
        text (str): Texto a clasificar
        
    Returns:
        Tuple[str, float]: (sentimiento, confianza)
    """

def classify_batch(texts: List[str]) -> List[Tuple[str, float]]:
    """
    Función de compatibilidad para clasificación por lotes.
    
    Args:
        texts (List[str]): Lista de textos
        
    Returns:
        List[Tuple[str, float]]: Lista de resultados
    """

def classify_text_multilingual(text: str, lang: str = None) -> Tuple[str, float]:
    """
    Función de compatibilidad para clasificación multilingüe.
    
    Args:
        text (str): Texto a clasificar
        lang (str, optional): Código de idioma
        
    Returns:
        Tuple[str, float]: (sentimiento, confianza)
    """

def classify_batch_multilingual(texts: List[str], langs: List[str] = None) -> List[Tuple[str, float]]:
    """
    Función de compatibilidad para clasificación multilingüe por lotes.
    
    Args:
        texts (List[str]): Lista de textos
        langs (List[str], optional): Lista de idiomas
        
    Returns:
        List[Tuple[str, float]]: Lista de resultados
    """

def detect_language(text: str) -> str:
    """
    Función de compatibilidad para detección de idiomas.
    
    Args:
        text (str): Texto a analizar
        
    Returns:
        str: Código de idioma detectado
    """
```

## ⚙️ Configuración

### **Config Class**

```python
class Config:
    """Configuración centralizada del sistema"""
    
    # Modelos de IA
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
    MULTILINGUAL_MODELS = {
        "en": "distilbert-base-uncased-finetuned-sst-2-english",
        "es": "dccuchile/bert-base-spanish-wwm-cased-finetuned-sentiment-spanish",
        "fr": "nlptown/bert-base-multilingual-uncased-sentiment",
        "de": "nlptown/bert-base-multilingual-uncased-sentiment",
        "it": "nlptown/bert-base-multilingual-uncased-sentiment",
        "pt": "nlptown/bert-base-multilingual-uncased-sentiment"
    }
    
    # Parámetros de procesamiento
    CONFIDENCE_THRESHOLD = 0.7
    MAX_LENGTH = 128
    BATCH_SIZE = 32
    
    # Configuración de logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configuración de Streamlit
    STREAMLIT_THEME = "light"
    STREAMLIT_LAYOUT = "wide"
```

## 📊 Ejemplos de Uso

### **Ejemplo 1: Análisis Básico**

```python
from model.predict_simple import classify_text, classify_batch

# Análisis individual
texto = "I love this product! It's amazing!"
sentimiento, confianza = classify_text(texto)
print(f"Sentimiento: {sentimiento}, Confianza: {confianza:.3f}")

# Análisis por lotes
textos = [
    "This is wonderful!",
    "I hate this!",
    "It's okay, nothing special."
]
resultados = classify_batch(textos)
for i, (sent, conf) in enumerate(resultados):
    print(f"Texto {i+1}: {sent} (confianza: {conf:.3f})")
```

### **Ejemplo 2: Análisis con Umbral de Confianza**

```python
from model.predict_simple import classify_with_confidence_threshold

# Análisis con umbral personalizado
texto = "This is okay, nothing special."
sentimiento, confianza, nivel = classify_with_confidence_threshold(texto, 0.8)

print(f"Sentimiento: {sentimiento}")
print(f"Confianza: {confianza:.3f}")
print(f"Nivel de confianza: {nivel}")
```

### **Ejemplo 3: Análisis Multilingüe**

```python
from multilingual import classify_text_multilingual, detect_language

# Detectar idioma
texto_es = "Me encanta este producto!"
idioma = detect_language(texto_es)
print(f"Idioma detectado: {idioma}")

# Clasificar en idioma específico
sentimiento, confianza = classify_text_multilingual(texto_es, "es")
print(f"Sentimiento: {sentimiento} (confianza: {confianza:.3f})")
```

### **Ejemplo 4: Uso Avanzado con Clases**

```python
from model.predict import SentimentClassifier
from utils import PerformanceMonitor

# Crear instancias
classifier = SentimentClassifier()
monitor = PerformanceMonitor()

# Análisis con monitoreo
monitor.start_timer()
sentimiento, confianza = classifier.classify_text("I love this!")
monitor.stop_timer()

# Mostrar métricas
tiempo = monitor.get_elapsed_time()
velocidad = monitor.get_processing_rate(1)

print(f"Sentimiento: {sentimiento}")
print(f"Confianza: {confianza:.3f}")
print(f"Tiempo de procesamiento: {tiempo:.3f} segundos")
print(f"Velocidad: {velocidad:.2f} textos/segundo")
```

### **Ejemplo 5: Integración con Streamlit**

```python
import streamlit as st
from model.predict_simple import classify_text, classify_batch

# Interfaz simple
st.title("Análisis de Sentimientos")

# Análisis individual
texto = st.text_area("Ingresa tu texto:")
if st.button("Analizar"):
    if texto.strip():
        sentimiento, confianza = classify_text(texto)
        st.success(f"Sentimiento: {sentimiento} (confianza: {confianza:.3f})")
    else:
        st.warning("Por favor ingresa un texto")

# Análisis por lotes
textos_lote = st.text_area("Ingresa múltiples textos (uno por línea):")
if st.button("Analizar Lote"):
    if textos_lote.strip():
        textos = [t.strip() for t in textos_lote.split('\n') if t.strip()]
        resultados = classify_batch(textos)
        
        for i, (sent, conf) in enumerate(resultados):
            st.write(f"Texto {i+1}: {sent} (confianza: {conf:.3f})")
```

## ❌ Códigos de Error

### **Errores Comunes**

```python
# ValueError: Texto vacío o inválido
try:
    classify_text("")
except ValueError as e:
    print(f"Error: {e}")

# RuntimeError: Error en el modelo
try:
    classifier = SentimentClassifier()
except RuntimeError as e:
    print(f"Error del modelo: {e}")

# ImportError: Dependencias faltantes
try:
    from transformers import pipeline
except ImportError as e:
    print(f"Dependencia faltante: {e}")
```

### **Manejo de Errores Recomendado**

```python
def safe_classify_text(text: str) -> Tuple[str, float]:
    """
    Clasificación segura con manejo de errores.
    
    Args:
        text (str): Texto a clasificar
        
    Returns:
        Tuple[str, float]: (sentimiento, confianza) o ('ERROR', 0.0)
    """
    try:
        if not text or not text.strip():
            return 'NEUTRAL', 0.5
            
        return classify_text(text)
        
    except ValueError as e:
        print(f"Error de validación: {e}")
        return 'NEUTRAL', 0.5
        
    except RuntimeError as e:
        print(f"Error del modelo: {e}")
        return 'ERROR', 0.0
        
    except Exception as e:
        print(f"Error inesperado: {e}")
        return 'ERROR', 0.0
```

## 🎯 Mejores Prácticas

### **1. Validación de Entrada**

```python
def validate_text_input(text: str) -> bool:
    """Valida la entrada de texto"""
    if not text or not text.strip():
        return False
    
    if len(text) > 10000:  # Límite de caracteres
        return False
        
    return True
```

### **2. Manejo de Memoria**

```python
def process_large_batch(texts: List[str], batch_size: int = 100) -> List[Tuple[str, float]]:
    """Procesa lotes grandes de forma eficiente"""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = classify_batch(batch)
        results.extend(batch_results)
        
        # Limpiar memoria si es necesario
        if i % 1000 == 0:
            import gc
            gc.collect()
    
    return results
```

### **3. Caching de Resultados**

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_classify_text(text: str) -> Tuple[str, float]:
    """Clasificación con caché para textos repetidos"""
    return classify_text(text)
```

### **4. Logging y Monitoreo**

```python
import logging

logger = logging.getLogger(__name__)

def monitored_classify_text(text: str) -> Tuple[str, float]:
    """Clasificación con logging"""
    logger.info(f"Clasificando texto: {text[:50]}...")
    
    start_time = time.time()
    result = classify_text(text)
    end_time = time.time()
    
    logger.info(f"Resultado: {result[0]}, Tiempo: {end_time - start_time:.3f}s")
    return result
```

### **5. Configuración Flexible**

```python
def create_classifier(config: dict) -> BaseClassifier:
    """Crea clasificador basado en configuración"""
    classifier_type = config.get('type', 'simple')
    
    if classifier_type == 'simple':
        return SimpleSentimentClassifier()
    elif classifier_type == 'advanced':
        return SentimentClassifier()
    elif classifier_type == 'multilingual':
        return MultilingualSentimentClassifier()
    else:
        raise ValueError(f"Tipo de clasificador no soportado: {classifier_type}")
```

---

**Esta API Reference proporciona una guía completa para usar todas las funcionalidades del Smart Text Classifier de manera efectiva y segura.**
