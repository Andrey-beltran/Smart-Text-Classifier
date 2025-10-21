from transformers import pipeline
import torch
import logging
from config import Config
from utils import PerformanceMonitor

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentClassifier:
    """Clase para clasificación de sentimientos optimizada"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.MODEL_NAME
        self.classifier = None
        self.performance_monitor = PerformanceMonitor()
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo de clasificación"""
        try:
            logger.info(f"Cargando modelo: {self.model_name}")
            self.classifier = pipeline(
                "sentiment-analysis", 
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Modelo cargado exitosamente")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {str(e)}")
            raise
    
    def classify_text(self, text: str) -> tuple:
        """Clasifica el sentimiento de un texto individual"""
        if not text or not text.strip():
            return "NEUTRAL", 0.5
        
        try:
            self.performance_monitor.start_timer()
            result = self.classifier(text)
            self.performance_monitor.stop_timer()
            
            label = result[0]['label']
            score = result[0]['score']
            
            logger.debug(f"Texto clasificado: {label} (confianza: {score:.3f})")
            return label, round(score, 3)
            
        except Exception as e:
            logger.error(f"Error al clasificar texto: {str(e)}")
            return "NEUTRAL", 0.5
    
    def classify_batch(self, texts: list) -> list:
        """Clasifica el sentimiento de múltiples textos de manera optimizada"""
        if not texts:
            return []
        
        # Filtrar textos vacíos
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        
        if not valid_texts:
            return [("NEUTRAL", 0.5)] * len(texts)
        
        try:
            self.performance_monitor.start_timer()
            
            # Procesar en lotes para mejor rendimiento
            batch_size = 32
            results = []
            
            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i:i + batch_size]
                batch_results = self.classifier(batch)
                
                for result in batch_results:
                    label = result['label']
                    score = result['score']
                    results.append((label, round(score, 3)))
            
            self.performance_monitor.stop_timer()
            
            # Agregar resultados para textos vacíos
            final_results = []
            valid_idx = 0
            
            for text in texts:
                if text and text.strip():
                    final_results.append(results[valid_idx])
                    valid_idx += 1
                else:
                    final_results.append(("NEUTRAL", 0.5))
            
            processing_rate = self.performance_monitor.get_processing_rate(len(valid_texts))
            logger.info(f"Procesados {len(valid_texts)} textos en {self.performance_monitor.get_elapsed_time():.2f}s "
                       f"({processing_rate:.2f} textos/seg)")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error en clasificación por lotes: {str(e)}")
            return [("NEUTRAL", 0.5)] * len(texts)
    
    def classify_with_confidence_threshold(self, text: str, threshold: float = None) -> tuple:
        """Clasifica texto con un umbral de confianza mínimo"""
        threshold = threshold or Config.CONFIDENCE_THRESHOLD
        label, score = self.classify_text(text)
        
        if score >= threshold:
            confidence_level = "Alta confianza"
        elif score >= 0.5:
            confidence_level = "Confianza media"
        else:
            confidence_level = "Baja confianza"
        
        return label, score, confidence_level
    
    def get_model_info(self) -> dict:
        """Retorna información sobre el modelo"""
        return {
            'model_name': self.model_name,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'max_length': Config.MAX_LENGTH,
            'confidence_threshold': Config.CONFIDENCE_THRESHOLD
        }

# Instancia global del clasificador
_sentiment_classifier = None

def get_classifier() -> SentimentClassifier:
    """Obtiene la instancia global del clasificador (singleton)"""
    global _sentiment_classifier
    if _sentiment_classifier is None:
        _sentiment_classifier = SentimentClassifier()
    return _sentiment_classifier

# Funciones de compatibilidad con el código existente
def classify_text(text: str) -> tuple:
    """Función de compatibilidad para clasificar texto individual"""
    classifier = get_classifier()
    return classifier.classify_text(text)

def classify_batch(texts: list) -> list:
    """Función de compatibilidad para clasificar múltiples textos"""
    classifier = get_classifier()
    return classifier.classify_batch(texts)

def classify_with_confidence_threshold(text: str, threshold: float = 0.7) -> tuple:
    """Función de compatibilidad para clasificar con umbral de confianza"""
    classifier = get_classifier()
    return classifier.classify_with_confidence_threshold(text, threshold)
