# utils.py
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
from config import Config

# Configurar logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Clase para procesar datos de texto"""
    
    @staticmethod
    def load_csv_data(file_path: str) -> pd.DataFrame:
        """Carga datos desde un archivo CSV"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Datos cargados exitosamente desde {file_path}: {len(df)} registros")
            return df
        except FileNotFoundError:
            logger.warning(f"Archivo {file_path} no encontrado")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error al cargar {file_path}: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def validate_text_data(df: pd.DataFrame, text_column: str = 'text') -> bool:
        """Valida que el DataFrame tenga la columna de texto requerida"""
        if text_column not in df.columns:
            logger.error(f"Columna '{text_column}' no encontrada en los datos")
            return False
        
        if df[text_column].isna().all():
            logger.error("Todos los valores de texto están vacíos")
            return False
        
        return True
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Limpia el texto eliminando caracteres innecesarios"""
        if not isinstance(text, str):
            return ""
        
        # Eliminar espacios extra y caracteres de nueva línea
        text = text.strip()
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def preprocess_texts(texts: List[str]) -> List[str]:
        """Preprocesa una lista de textos"""
        return [DataProcessor.clean_text(text) for text in texts if text]

class MetricsCalculator:
    """Clase para calcular métricas de análisis de sentimientos"""
    
    @staticmethod
    def calculate_sentiment_distribution(results: List[Tuple[str, float]]) -> Dict[str, int]:
        """Calcula la distribución de sentimientos"""
        sentiments = [result[0] for result in results]
        distribution = {}
        
        for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            distribution[sentiment] = sentiments.count(sentiment)
        
        return distribution
    
    @staticmethod
    def calculate_confidence_stats(results: List[Tuple[str, float]]) -> Dict[str, float]:
        """Calcula estadísticas de confianza"""
        scores = [result[1] for result in results]
        
        if not scores:
            return {}
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'median': np.median(scores)
        }
    
    @staticmethod
    def calculate_accuracy_metrics(results: List[Tuple[str, float]], threshold: float = 0.7) -> Dict[str, float]:
        """Calcula métricas de precisión basadas en umbral de confianza"""
        high_confidence_count = sum(1 for _, score in results if score >= threshold)
        total_count = len(results)
        
        if total_count == 0:
            return {}
        
        return {
            'high_confidence_rate': high_confidence_count / total_count,
            'low_confidence_rate': (total_count - high_confidence_count) / total_count
        }

class FileManager:
    """Clase para manejar archivos y directorios"""
    
    @staticmethod
    def ensure_directory_exists(directory: str) -> bool:
        """Asegura que un directorio existe"""
        try:
            import os
            os.makedirs(directory, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error al crear directorio {directory}: {str(e)}")
            return False
    
    @staticmethod
    def save_results_to_csv(results: List[Tuple[str, float]], texts: List[str], 
                           output_path: str) -> bool:
        """Guarda resultados en un archivo CSV"""
        try:
            df = pd.DataFrame({
                'text': texts,
                'sentiment': [result[0] for result in results],
                'confidence': [result[1] for result in results]
            })
            df.to_csv(output_path, index=False)
            logger.info(f"Resultados guardados en {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error al guardar resultados: {str(e)}")
            return False

class PerformanceMonitor:
    """Clase para monitorear el rendimiento"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start_timer(self):
        """Inicia el temporizador"""
        import time
        self.start_time = time.time()
    
    def stop_timer(self):
        """Detiene el temporizador"""
        import time
        self.end_time = time.time()
    
    def get_elapsed_time(self) -> float:
        """Retorna el tiempo transcurrido en segundos"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def get_processing_rate(self, item_count: int) -> float:
        """Calcula la tasa de procesamiento (ítems por segundo)"""
        elapsed_time = self.get_elapsed_time()
        if elapsed_time > 0:
            return item_count / elapsed_time
        return 0.0

