# multilingual.py
import logging
from typing import Dict, List, Tuple, Optional
from transformers import pipeline
import torch
from config import Config

logger = logging.getLogger(__name__)

class MultilingualSentimentClassifier:
    """Clasificador de sentimientos multilingüe"""
    
    def __init__(self):
        self.models = {}
        self.language_detector = None
        self._load_models()
    
    def _load_models(self):
        """Carga modelos para diferentes idiomas"""
        try:
            # Modelo para inglés (por defecto)
            self.models['en'] = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Modelo multilingüe para español, francés, alemán, etc.
            self.models['multilingual'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Modelo específico para español
            try:
                self.models['es'] = pipeline(
                    "sentiment-analysis",
                    model="pysentimiento/robertuito-sentiment-analysis",
                    device=0 if torch.cuda.is_available() else -1
                )
            except Exception as e:
                logger.warning(f"No se pudo cargar el modelo español: {e}")
                self.models['es'] = self.models['multilingual']
            
            logger.info("Modelos multilingües cargados exitosamente")
            
        except Exception as e:
            logger.error(f"Error al cargar modelos multilingües: {e}")
            raise
    
    def detect_language(self, text: str) -> str:
        """Detecta el idioma del texto"""
        # Detección simple basada en palabras clave
        text_lower = text.lower()
        
        # Palabras clave en español
        spanish_words = ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una', 'está', 'muy', 'más', 'pero', 'como', 'todo', 'esta', 'sobre', 'entre', 'cuando', 'hasta', 'desde', 'hacia', 'durante', 'mediante', 'excepto', 'salvo', 'según', 'contra', 'bajo', 'sobre', 'tras', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante', 'en', 'entre', 'hacia', 'hasta', 'mediante', 'para', 'por', 'según', 'sin', 'so', 'sobre', 'tras', 'versus', 'vía']
        
        # Palabras clave en francés
        french_words = ['le', 'la', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus', 'par', 'grand', 'en', 'une', 'être', 'et', 'à', 'il', 'avoir', 'ne', 'je', 'son', 'que', 'je', 'ne', 'pas', 'tout', 'plus', 'par', 'grand', 'en', 'une', 'être', 'et', 'à', 'il', 'avoir', 'ne', 'je', 'son', 'que']
        
        # Palabras clave en alemán
        german_words = ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für', 'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als', 'auch', 'es', 'an', 'werden', 'aus', 'er', 'hat', 'daß', 'sie', 'nach', 'wird', 'bei', 'einer', 'um', 'am', 'sind', 'noch', 'wie', 'einem', 'über', 'einen', 'so', 'zum', 'war', 'haben', 'nur', 'oder', 'aber', 'vor', 'zur', 'bis', 'mehr', 'durch', 'man', 'sein', 'wurde', 'sei', 'in', 'den', 'nicht', 'werden', 'sie', 'nach', 'wird', 'bei', 'einer', 'um', 'am', 'sind', 'noch', 'wie', 'einem', 'über', 'einen', 'so', 'zum', 'war', 'haben', 'nur', 'oder', 'aber', 'vor', 'zur', 'bis', 'mehr', 'durch', 'man', 'sein', 'wurde', 'sei']
        
        # Contar palabras por idioma
        spanish_count = sum(1 for word in spanish_words if word in text_lower)
        french_count = sum(1 for word in french_words if word in text_lower)
        german_count = sum(1 for word in german_words if word in text_lower)
        
        # Determinar idioma basado en el conteo
        if spanish_count > french_count and spanish_count > german_count:
            return 'es'
        elif french_count > german_count:
            return 'fr'
        elif german_count > 0:
            return 'de'
        else:
            return 'en'  # Por defecto inglés
    
    def classify_text(self, text: str, language: Optional[str] = None) -> Tuple[str, float]:
        """Clasifica el sentimiento de un texto en cualquier idioma"""
        if not text or not text.strip():
            return "NEUTRAL", 0.5
        
        # Detectar idioma si no se especifica
        if language is None:
            language = self.detect_language(text)
        
        try:
            # Seleccionar modelo apropiado
            if language == 'es' and 'es' in self.models:
                model = self.models['es']
            elif language in ['fr', 'de', 'it', 'pt'] and 'multilingual' in self.models:
                model = self.models['multilingual']
            else:
                model = self.models['en']  # Por defecto inglés
            
            # Clasificar
            result = model(text)
            
            # Normalizar etiquetas
            label = self._normalize_label(result[0]['label'], language)
            score = result[0]['score']
            
            logger.debug(f"Texto clasificado en {language}: {label} (confianza: {score:.3f})")
            return label, round(score, 3)
            
        except Exception as e:
            logger.error(f"Error al clasificar texto en {language}: {e}")
            return "NEUTRAL", 0.5
    
    def _normalize_label(self, label: str, language: str) -> str:
        """Normaliza las etiquetas de sentimiento"""
        label_lower = label.lower()
        
        # Mapeo de etiquetas a formato estándar
        if 'positive' in label_lower or 'positif' in label_lower or 'positivo' in label_lower:
            return 'POSITIVE'
        elif 'negative' in label_lower or 'négatif' in label_lower or 'negativo' in label_lower:
            return 'NEGATIVE'
        elif 'neutral' in label_lower or 'neutre' in label_lower or 'neutro' in label_lower:
            return 'NEUTRAL'
        else:
            return 'NEUTRAL'  # Por defecto
    
    def classify_batch(self, texts: List[str], languages: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Clasifica múltiples textos en diferentes idiomas"""
        if not texts:
            return []
        
        results = []
        
        for i, text in enumerate(texts):
            language = languages[i] if languages and i < len(languages) else None
            result = self.classify_text(text, language)
            results.append(result)
        
        return results
    
    def get_supported_languages(self) -> List[str]:
        """Retorna la lista de idiomas soportados"""
        return ['en', 'es', 'fr', 'de', 'it', 'pt']
    
    def get_language_name(self, code: str) -> str:
        """Retorna el nombre del idioma en español"""
        language_names = {
            'en': 'Inglés',
            'es': 'Español',
            'fr': 'Francés',
            'de': 'Alemán',
            'it': 'Italiano',
            'pt': 'Portugués'
        }
        return language_names.get(code, 'Desconocido')

# Instancia global del clasificador multilingüe
_multilingual_classifier = None

def get_multilingual_classifier() -> MultilingualSentimentClassifier:
    """Obtiene la instancia global del clasificador multilingüe"""
    global _multilingual_classifier
    if _multilingual_classifier is None:
        _multilingual_classifier = MultilingualSentimentClassifier()
    return _multilingual_classifier

# Funciones de conveniencia
def classify_text_multilingual(text: str, language: str = None) -> Tuple[str, float]:
    """Clasifica texto en cualquier idioma soportado"""
    classifier = get_multilingual_classifier()
    return classifier.classify_text(text, language)

def classify_batch_multilingual(texts: List[str], languages: List[str] = None) -> List[Tuple[str, float]]:
    """Clasifica múltiples textos en diferentes idiomas"""
    classifier = get_multilingual_classifier()
    return classifier.classify_batch(texts, languages)

def detect_language(text: str) -> str:
    """Detecta el idioma de un texto"""
    classifier = get_multilingual_classifier()
    return classifier.detect_language(text)

