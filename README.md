# Smart-Text-Classifier
Descripción Sistema completo de análisis de sentimientos que combina: - **IA Avanzada**: Modelos de Hugging Face (BERT, RoBERTa) - **Análisis Simple**: Sistema basado en reglas - **Interfaz Web**: Dashboard interactivo con Streamlit - **Soporte Multilingüe**: 6 idiomas soportados

  Características
- ✅ Análisis individual y por lotes
- ✅ Carga de archivos CSV
- ✅ Visualizaciones interactivas
- ✅ Métricas detalladas
- ✅ Descarga de resultados
- ✅ Soporte multilingüe

  Tecnologías
- **Backend**: Python 3.11, PyTorch, Transformers
- **Frontend**: Streamlit, Plotly
- **ML**: Hugging Face, scikit-learn
- **Data**: Pandas, NumPy

  Documentación de Arquitectura:
Diagrama del Sistema:

graph TD
    A[Usuario] --> B[Streamlit App]
    B --> C[Clasificador]
    C --> D[Modelo Simple]
    C --> E[Modelo IA]
    D --> F[Análisis por Reglas]
    E --> G[Hugging Face]
    F --> H[Resultados]
    G --> H
    H --> I[Visualizaciones]
    I --> J[Plotly Charts]
