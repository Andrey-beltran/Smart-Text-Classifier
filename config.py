# config.py
import os
from typing import Dict, Any

class Config:
    """Configuración centralizada para la aplicación"""
    
    # Configuración del modelo
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
    MAX_LENGTH = 512
    CONFIDENCE_THRESHOLD = 0.7
    
    # Configuración de la aplicación
    APP_TITLE = "Smart Text Classifier"
    APP_ICON = "🤖"
    PAGE_LAYOUT = "wide"
    
    # Configuración de archivos
    DATA_DIR = "data"
    MODEL_DIR = "model"
    SAMPLES_CSV = os.path.join(DATA_DIR, "samples.csv")
    MODEL_CHECKPOINT_DIR = "model_checkpoint"
    
    # Configuración de entrenamiento
    TRAINING_EPOCHS = 3
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 100
    WEIGHT_DECAY = 0.01
    
    # Configuración de logging
    LOG_LEVEL = "INFO"
    LOG_DIR = "logs"
    
    # Configuración de UI
    SENTIMENT_COLORS = {
        'POSITIVE': '#28a745',
        'NEGATIVE': '#dc3545', 
        'NEUTRAL': '#ffc107'
    }
    
    SENTIMENT_LABELS = {
        'POSITIVE': 'Positivo',
        'NEGATIVE': 'Negativo',
        'NEUTRAL': 'Neutral'
    }
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Retorna la configuración del modelo"""
        return {
            'model_name': cls.MODEL_NAME,
            'max_length': cls.MAX_LENGTH,
            'confidence_threshold': cls.CONFIDENCE_THRESHOLD
        }
    
    @classmethod
    def get_training_config(cls) -> Dict[str, Any]:
        """Retorna la configuración de entrenamiento"""
        return {
            'epochs': cls.TRAINING_EPOCHS,
            'batch_size': cls.BATCH_SIZE,
            'learning_rate': cls.LEARNING_RATE,
            'warmup_steps': cls.WARMUP_STEPS,
            'weight_decay': cls.WEIGHT_DECAY
        }

