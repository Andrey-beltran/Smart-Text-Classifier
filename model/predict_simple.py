# model/predict_simple.py
"""
Versión simplificada del clasificador que no depende de PyTorch
Usa análisis de texto basado en reglas para demostrar la funcionalidad
"""

import re
import logging
from typing import List, Tuple
from config import Config

logger = logging.getLogger(__name__)

class SimpleSentimentClassifier:
    """Clasificador de sentimientos basado en reglas"""
    
    def __init__(self):
        # Palabras positivas
        self.positive_words = {
            'love', 'amazing', 'wonderful', 'great', 'excellent', 'fantastic', 
            'awesome', 'brilliant', 'outstanding', 'perfect', 'beautiful', 
            'delicious', 'incredible', 'marvelous', 'superb', 'terrific',
            'good', 'nice', 'best', 'favorite', 'enjoy', 'happy', 'pleased',
            'satisfied', 'impressed', 'thrilled', 'excited', 'delighted'
        }
        
        # Palabras negativas
        self.negative_words = {
            'hate', 'terrible', 'awful', 'bad', 'horrible', 'disgusting',
            'worst', 'disappointed', 'angry', 'frustrated', 'annoyed',
            'upset', 'sad', 'depressed', 'miserable', 'disgusted',
            'disgusting', 'repulsive', 'revolting', 'atrocious', 'appalling',
            'dreadful', 'pathetic', 'useless', 'worthless', 'disgusting'
        }
        
        # Palabras de intensificación
        self.intensifiers = {
            'very', 'extremely', 'incredibly', 'absolutely', 'completely',
            'totally', 'really', 'so', 'quite', 'rather', 'pretty', 'fairly'
        }
        
        # Palabras de negación
        self.negation_words = {
            'not', 'no', 'never', 'none', 'nothing', 'nobody', 'nowhere',
            'neither', 'nor', "don't", "doesn't", "didn't", "won't", "can't"
        }
    
    def preprocess_text(self, text: str) -> str:
        """Preprocesa el texto para análisis"""
        # Convertir a minúsculas
        text = text.lower()
        
        # Remover caracteres especiales excepto espacios y puntuación básica
        text = re.sub(r'[^\w\s!?.,]', '', text)
        
        # Normalizar espacios
        text = ' '.join(text.split())
        
        return text
    
    def calculate_sentiment_score(self, text: str) -> Tuple[str, float]:
        """Calcula el puntaje de sentimiento basado en palabras clave"""
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        positive_score = 0
        negative_score = 0
        
        # Analizar cada palabra
        for i, word in enumerate(words):
            # Verificar si hay negación antes de la palabra
            has_negation = False
            if i > 0 and words[i-1] in self.negation_words:
                has_negation = True
            
            # Verificar intensificadores
            intensity = 1.0
            if i > 0 and words[i-1] in self.intensifiers:
                intensity = 1.5
            
            # Contar palabras positivas y negativas
            if word in self.positive_words:
                if has_negation:
                    negative_score += intensity
                else:
                    positive_score += intensity
            elif word in self.negative_words:
                if has_negation:
                    positive_score += intensity
                else:
                    negative_score += intensity
        
        # Calcular puntaje final
        total_score = positive_score + negative_score
        
        if total_score == 0:
            return 'NEUTRAL', 0.5
        
        if positive_score > negative_score:
            confidence = min(0.95, positive_score / total_score)
            return 'POSITIVE', confidence
        elif negative_score > positive_score:
            confidence = min(0.95, negative_score / total_score)
            return 'NEGATIVE', confidence
        else:
            return 'NEUTRAL', 0.5
    
    def classify_text(self, text: str) -> Tuple[str, float]:
        """Clasifica el sentimiento de un texto"""
        if not text or not text.strip():
            return 'NEUTRAL', 0.5
        
        try:
            sentiment, confidence = self.calculate_sentiment_score(text)
            logger.debug(f"Texto clasificado: {sentiment} (confianza: {confidence:.3f})")
            return sentiment, round(confidence, 3)
        except Exception as e:
            logger.error(f"Error al clasificar texto: {str(e)}")
            return 'NEUTRAL', 0.5
    
    def classify_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Clasifica múltiples textos"""
        if not texts:
            return []
        
        results = []
        for text in texts:
            if text and text.strip():
                result = self.classify_text(text)
                results.append(result)
            else:
                results.append(('NEUTRAL', 0.5))
        
        return results

# Instancia global del clasificador simple
_simple_classifier = None

def get_simple_classifier() -> SimpleSentimentClassifier:
    """Obtiene la instancia global del clasificador simple"""
    global _simple_classifier
    if _simple_classifier is None:
        _simple_classifier = SimpleSentimentClassifier()
    return _simple_classifier

# Funciones de compatibilidad
def classify_text(text: str) -> Tuple[str, float]:
    """Función de compatibilidad para clasificar texto individual"""
    classifier = get_simple_classifier()
    return classifier.classify_text(text)

def classify_batch(texts: List[str]) -> List[Tuple[str, float]]:
    """Función de compatibilidad para clasificar múltiples textos"""
    classifier = get_simple_classifier()
    return classifier.classify_batch(texts)

def classify_with_confidence_threshold(text: str, threshold: float = 0.7) -> Tuple[str, float, str]:
    """Clasifica texto con un umbral de confianza mínimo"""
    classifier = get_simple_classifier()
    label, score = classifier.classify_text(text)
    
    if score >= threshold:
        confidence_level = "Alta confianza"
    elif score >= 0.5:
        confidence_level = "Confianza media"
    else:
        confidence_level = "Baja confianza"
    
    return label, score, confidence_level
