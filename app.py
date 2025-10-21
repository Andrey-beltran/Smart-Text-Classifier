import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from model.predict import classify_text, classify_batch
from multilingual import classify_text_multilingual, classify_batch_multilingual, detect_language, get_multilingual_classifier
import time

# Configuración de la página
st.set_page_config(
    page_title="Smart Text Classifier", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🧠 Smart Text Classifier</h1>', unsafe_allow_html=True)
st.markdown("### Analiza el sentimiento de cualquier texto usando IA con Hugging Face 🤗")

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración")

# Opciones de análisis
analysis_mode = st.sidebar.selectbox(
    "Modo de análisis:",
    ["Texto individual", "Análisis por lotes", "Carga de archivo CSV"]
)

# Configuración de idioma
language_mode = st.sidebar.selectbox(
    "Modo de idioma:",
    ["Automático", "Inglés", "Español", "Francés", "Alemán", "Italiano", "Portugués"]
)

# Mapeo de idiomas
language_mapping = {
    "Automático": None,
    "Inglés": "en",
    "Español": "es", 
    "Francés": "fr",
    "Alemán": "de",
    "Italiano": "it",
    "Portugués": "pt"
}

selected_language = language_mapping[language_mode]

# Función para mostrar métricas
def display_metrics(results):
    if not results:
        return
    
    col1, col2, col3 = st.columns(3)
    
    # Contar sentimientos
    sentiments = [r[0] for r in results]
    positive_count = sentiments.count('POSITIVE')
    negative_count = sentiments.count('NEGATIVE')
    neutral_count = sentiments.count('NEUTRAL')
    total = len(sentiments)
    
    with col1:
        st.metric("Positivos", positive_count, f"{positive_count/total*100:.1f}%")
    with col2:
        st.metric("Negativos", negative_count, f"{negative_count/total*100:.1f}%")
    with col3:
        st.metric("Neutrales", neutral_count, f"{neutral_count/total*100:.1f}%")

# Función para crear gráfico de distribución
def create_sentiment_chart(results):
    if not results:
        return None
    
    sentiments = [r[0] for r in results]
    sentiment_counts = pd.Series(sentiments).value_counts()
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Distribución de Sentimientos",
        color_discrete_map={
            'POSITIVE': '#28a745',
            'NEGATIVE': '#dc3545',
            'NEUTRAL': '#ffc107'
        }
    )
    return fig

# Función para crear gráfico de confianza
def create_confidence_chart(results):
    if not results:
        return None
    
    sentiments = [r[0] for r in results]
    scores = [r[1] for r in results]
    
    fig = go.Figure()
    
    colors = {'POSITIVE': '#28a745', 'NEGATIVE': '#dc3545', 'NEUTRAL': '#ffc107'}
    
    for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        sentiment_scores = [score for sent, score in zip(sentiments, scores) if sent == sentiment]
        if sentiment_scores:
            fig.add_trace(go.Box(
                y=sentiment_scores,
                name=sentiment,
                marker_color=colors[sentiment]
            ))
    
    fig.update_layout(
        title="Distribución de Confianza por Sentimiento",
        yaxis_title="Nivel de Confianza",
        xaxis_title="Sentimiento"
    )
    
    return fig

# Contenido principal según el modo seleccionado
if analysis_mode == "Texto individual":
    st.header("📝 Análisis de Texto Individual")
    
    # Área de texto
    user_input = st.text_area(
        "✍️ Escribe o pega tu texto aquí:",
        height=200,
        placeholder="Ejemplo: 'I love this product! It's amazing!'"
    )
    
    # Botón de análisis
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("🔍 Analizar Sentimiento", type="primary", use_container_width=True)
    
    if analyze_btn:
        if user_input.strip():
            with st.spinner("Analizando sentimiento..."):
                if selected_language is None:
                    # Detectar idioma automáticamente
                    detected_lang = detect_language(user_input)
                    st.info(f"🌍 Idioma detectado: {detected_lang.upper()}")
                    label, score = classify_text_multilingual(user_input, detected_lang)
                else:
                    label, score = classify_text_multilingual(user_input, selected_language)
            
            # Mostrar resultado
            st.success("✅ Análisis completado!")
            
            # Métricas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sentimiento", label)
            with col2:
                st.metric("Confianza", f"{score:.3f}")
            with col3:
                st.metric("Idioma", language_mode)
            
            # Visualización del resultado
            sentiment_class = f"sentiment-{label.lower()}"
            st.markdown(f'<p class="{sentiment_class}">Resultado: {label} (confianza: {score:.3f})</p>', 
                       unsafe_allow_html=True)
            
            # Barra de progreso para la confianza
            progress_value = score
            st.progress(progress_value)
            st.caption(f"Nivel de confianza: {score:.1%}")
            
        else:
            st.warning("⚠️ Por favor ingresa un texto para analizar.")

elif analysis_mode == "Análisis por lotes":
    st.header("📊 Análisis por Lotes")
    
    # Textos de ejemplo multilingües
    sample_texts = [
        "I love this product! It's amazing!",
        "Este producto es terrible, lo odio.",
        "Le temps est agréable aujourd'hui.",
        "Ich fühle mich traurig wegen der Nachrichten.",
        "Che giornata meravigliosa!"
    ]
    
    # Área para múltiples textos
    texts_input = st.text_area(
        "📝 Ingresa múltiples textos (uno por línea):",
        height=200,
        value="\n".join(sample_texts),
        help="Escribe cada texto en una línea separada"
    )
    
    if st.button("🔍 Analizar Todos", type="primary"):
        if texts_input.strip():
            texts = [text.strip() for text in texts_input.split('\n') if text.strip()]
            
            with st.spinner(f"Analizando {len(texts)} textos..."):
                if selected_language is None:
                    # Detectar idiomas automáticamente para cada texto
                    detected_languages = [detect_language(text) for text in texts]
                    st.info(f"🌍 Idiomas detectados: {', '.join(set(detected_languages))}")
                    results = classify_batch_multilingual(texts, detected_languages)
                else:
                    results = classify_batch_multilingual(texts, [selected_language] * len(texts))
            
            st.success(f"✅ Análisis completado para {len(texts)} textos!")
            
            # Mostrar métricas
            display_metrics(results)
            
            # Crear gráficos
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = create_sentiment_chart(results)
                if fig_pie:
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                fig_box = create_confidence_chart(results)
                if fig_box:
                    st.plotly_chart(fig_box, use_container_width=True)
            
            # Tabla de resultados
            st.subheader("📋 Resultados Detallados")
            df_results = pd.DataFrame(results, columns=['Texto', 'Sentimiento', 'Confianza'])
            df_results['Texto'] = texts
            st.dataframe(df_results, use_container_width=True)
            
            # Botón de descarga
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Descargar resultados como CSV",
                data=csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ Por favor ingresa al menos un texto para analizar.")

elif analysis_mode == "Carga de archivo CSV":
    st.header("📁 Análisis desde Archivo CSV")
    
    uploaded_file = st.file_uploader(
        "Selecciona un archivo CSV:",
        type=['csv'],
        help="El archivo debe tener una columna 'text' con los textos a analizar"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Archivo cargado exitosamente! {len(df)} registros encontrados.")
            
            # Mostrar preview del archivo
            st.subheader("👀 Vista previa del archivo:")
            st.dataframe(df.head(), use_container_width=True)
            
            # Verificar si tiene columna 'text'
            if 'text' in df.columns:
                if st.button("🔍 Analizar Archivo Completo", type="primary"):
                    texts = df['text'].dropna().tolist()
                    
                    with st.spinner(f"Analizando {len(texts)} textos del archivo..."):
                        if selected_language is None:
                            # Detectar idiomas automáticamente
                            detected_languages = [detect_language(text) for text in texts]
                            st.info(f"🌍 Idiomas detectados: {', '.join(set(detected_languages))}")
                            results = classify_batch_multilingual(texts, detected_languages)
                        else:
                            results = classify_batch_multilingual(texts, [selected_language] * len(texts))
                    
                    st.success(f"✅ Análisis completado para {len(texts)} textos!")
                    
                    # Mostrar métricas
                    display_metrics(results)
                    
                    # Crear gráficos
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_pie = create_sentiment_chart(results)
                        if fig_pie:
                            st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        fig_box = create_confidence_chart(results)
                        if fig_box:
                            st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Agregar resultados al DataFrame original
                    df_results = df.copy()
                    df_results['sentiment'] = [r[0] for r in results]
                    df_results['confidence'] = [r[1] for r in results]
                    
                    st.subheader("📋 Resultados con Sentimientos:")
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Botón de descarga
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar resultados como CSV",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv"
                    )
            else:
                st.error("❌ El archivo CSV debe tener una columna llamada 'text'")
                st.info("💡 Las columnas disponibles son: " + ", ".join(df.columns.tolist()))
                
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🤖 Smart Text Classifier - Powered by Hugging Face Transformers</p>
    <p>Desarrollado con ❤️ usando Streamlit</p>
</div>
""", unsafe_allow_html=True)
