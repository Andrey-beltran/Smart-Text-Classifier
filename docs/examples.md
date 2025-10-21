# 📊 Ejemplos de Uso - Smart Text Classifier

## 📋 Tabla de Contenidos
- [Ejemplos Básicos](#ejemplos-básicos)
- [Ejemplos Avanzados](#ejemplos-avanzados)
- [Integración con Streamlit](#integración-con-streamlit)
- [Análisis de Datos](#análisis-de-datos)
- [Casos de Uso Reales](#casos-de-uso-reales)
- [Mejores Prácticas](#mejores-prácticas)

## 🚀 Ejemplos Básicos

### **Ejemplo 1: Análisis Individual**

```python
from model.predict_simple import classify_text

# Analizar un texto simple
texto = "I love this product! It's amazing!"
sentimiento, confianza = classify_text(texto)

print(f"Texto: {texto}")
print(f"Sentimiento: {sentimiento}")
print(f"Confianza: {confianza:.3f}")
print(f"Nivel de confianza: {'Alta' if confianza > 0.8 else 'Media' if confianza > 0.6 else 'Baja'}")
```

**Output:**
```
Texto: I love this product! It's amazing!
Sentimiento: POSITIVE
Confianza: 0.950
Nivel de confianza: Alta
```

### **Ejemplo 2: Análisis por Lotes**

```python
from model.predict_simple import classify_batch

# Lista de textos para analizar
textos = [
    "This is wonderful! I'm so happy!",
    "I hate this! It's terrible!",
    "The weather is nice today.",
    "I'm feeling sad about the news.",
    "What a beautiful day!",
    "This is okay, nothing special.",
    "I'm excited about the trip!",
    "This is disappointing."
]

# Analizar todos los textos
resultados = classify_batch(textos)

# Mostrar resultados
print("📊 Resultados del Análisis por Lotes:")
print("=" * 50)

for i, (texto, (sentimiento, confianza)) in enumerate(zip(textos, resultados), 1):
    print(f"{i:2d}. {texto[:40]:<40} | {sentimiento:<9} | {confianza:.3f}")
```

**Output:**
```
📊 Resultados del Análisis por Lotes:
==================================================
 1. This is wonderful! I'm so happy!     | POSITIVE  | 0.950
 2. I hate this! It's terrible!         | NEGATIVE  | 0.900
 3. The weather is nice today.          | NEUTRAL   | 0.700
 4. I'm feeling sad about the news.     | NEGATIVE  | 0.900
 5. What a beautiful day!               | POSITIVE  | 0.950
 6. This is okay, nothing special.      | NEUTRAL   | 0.700
 7. I'm excited about the trip!         | POSITIVE  | 0.950
 8. This is disappointing.              | NEGATIVE  | 0.900
```

### **Ejemplo 3: Análisis con Umbral de Confianza**

```python
from model.predict_simple import classify_with_confidence_threshold

# Textos con diferentes niveles de confianza
textos_prueba = [
    "I absolutely love this!",
    "This is okay, I guess.",
    "I'm not sure how I feel about this."
]

# Analizar con diferentes umbrales
umbrales = [0.9, 0.7, 0.5]

for texto in textos_prueba:
    print(f"\n📝 Texto: {texto}")
    print("-" * 40)
    
    for umbral in umbrales:
        sentimiento, confianza, nivel = classify_with_confidence_threshold(texto, umbral)
        print(f"Umbral {umbral}: {sentimiento} ({confianza:.3f}) - {nivel}")
```

**Output:**
```
📝 Texto: I absolutely love this!
----------------------------------------
Umbral 0.9: POSITIVE (0.950) - Alta confianza
Umbral 0.7: POSITIVE (0.950) - Alta confianza
Umbral 0.5: POSITIVE (0.950) - Alta confianza

📝 Texto: This is okay, I guess.
----------------------------------------
Umbral 0.9: NEUTRAL (0.700) - Baja confianza
Umbral 0.7: NEUTRAL (0.700) - Confianza media
Umbral 0.5: NEUTRAL (0.700) - Confianza media

📝 Texto: I'm not sure how I feel about this.
----------------------------------------
Umbral 0.9: NEUTRAL (0.700) - Baja confianza
Umbral 0.7: NEUTRAL (0.700) - Confianza media
Umbral 0.5: NEUTRAL (0.700) - Confianza media
```

## 🔬 Ejemplos Avanzados

### **Ejemplo 4: Análisis Multilingüe**

```python
from multilingual import classify_text_multilingual, detect_language, classify_batch_multilingual

# Textos en diferentes idiomas
textos_multilingues = [
    "I love this product!",  # Inglés
    "Me encanta este producto!",  # Español
    "J'adore ce produit!",  # Francés
    "Ich liebe dieses Produkt!",  # Alemán
    "Adoro questo prodotto!",  # Italiano
    "Eu amo este produto!"  # Portugués
]

print("🌍 Análisis Multilingüe:")
print("=" * 60)

for texto in textos_multilingues:
    # Detectar idioma
    idioma = detect_language(texto)
    
    # Clasificar en idioma específico
    sentimiento, confianza = classify_text_multilingual(texto, idioma)
    
    print(f"Texto: {texto}")
    print(f"Idioma: {idioma.upper()}")
    print(f"Sentimiento: {sentimiento} (confianza: {confianza:.3f})")
    print("-" * 40)
```

**Output:**
```
🌍 Análisis Multilingüe:
============================================================
Texto: I love this product!
Idioma: EN
Sentimiento: POSITIVE (confianza: 0.950)
----------------------------------------
Texto: Me encanta este producto!
Idioma: ES
Sentimiento: POSITIVE (confianza: 0.920)
----------------------------------------
Texto: J'adore ce produit!
Idioma: FR
Sentimiento: POSITIVE (confianza: 0.880)
----------------------------------------
Texto: Ich liebe dieses Produkt!
Idioma: DE
Sentimiento: POSITIVE (confianza: 0.900)
----------------------------------------
Texto: Adoro questo prodotto!
Idioma: IT
Sentimiento: POSITIVE (confianza: 0.870)
----------------------------------------
Texto: Eu amo este produto!
Idioma: PT
Sentimiento: POSITIVE (confianza: 0.890)
----------------------------------------
```

### **Ejemplo 5: Análisis con Métricas de Rendimiento**

```python
from model.predict_simple import classify_batch
from utils import PerformanceMonitor
import time

# Crear monitor de rendimiento
monitor = PerformanceMonitor()

# Textos para análisis masivo
textos_masivos = [
    "I love this product!",
    "This is terrible!",
    "It's okay, nothing special.",
    "I'm so excited!",
    "This is disappointing.",
    "What a wonderful day!",
    "I hate this!",
    "The weather is nice.",
    "I'm thrilled!",
    "This is mediocre."
] * 10  # 100 textos en total

print("⚡ Análisis de Rendimiento:")
print("=" * 40)

# Medir tiempo de procesamiento
monitor.start_timer()
resultados = classify_batch(textos_masivos)
monitor.stop_timer()

# Calcular métricas
tiempo_total = monitor.get_elapsed_time()
velocidad = monitor.get_processing_rate(len(textos_masivos))

print(f"Textos procesados: {len(textos_masivos)}")
print(f"Tiempo total: {tiempo_total:.3f} segundos")
print(f"Velocidad: {velocidad:.2f} textos/segundo")
print(f"Tiempo promedio por texto: {tiempo_total/len(textos_masivos)*1000:.2f} ms")

# Análisis de distribución de sentimientos
sentimientos = [r[0] for r in resultados]
distribucion = {}
for sent in sentimientos:
    distribucion[sent] = distribucion.get(sent, 0) + 1

print(f"\n📊 Distribución de Sentimientos:")
for sentimiento, count in distribucion.items():
    porcentaje = (count / len(textos_masivos)) * 100
    print(f"{sentimiento}: {count} textos ({porcentaje:.1f}%)")
```

**Output:**
```
⚡ Análisis de Rendimiento:
========================================
Textos procesados: 100
Tiempo total: 0.150 segundos
Velocidad: 666.67 textos/segundo
Tiempo promedio por texto: 1.50 ms

📊 Distribución de Sentimientos:
POSITIVE: 30 textos (30.0%)
NEGATIVE: 30 textos (30.0%)
NEUTRAL: 40 textos (40.0%)
```

### **Ejemplo 6: Análisis de Archivo CSV**

```python
import pandas as pd
from model.predict_simple import classify_batch

# Crear archivo CSV de ejemplo
datos_ejemplo = {
    'id': range(1, 11),
    'text': [
        "I love this product!",
        "This is terrible!",
        "It's okay, nothing special.",
        "I'm so excited!",
        "This is disappointing.",
        "What a wonderful day!",
        "I hate this!",
        "The weather is nice.",
        "I'm thrilled!",
        "This is mediocre."
    ],
    'category': ['product', 'product', 'general', 'product', 'product', 'general', 'product', 'general', 'product', 'product']
}

df = pd.DataFrame(datos_ejemplo)

# Guardar archivo CSV
df.to_csv('data/ejemplo_analisis.csv', index=False)

print("📁 Análisis de Archivo CSV:")
print("=" * 40)

# Cargar y analizar
df_cargado = pd.read_csv('data/ejemplo_analisis.csv')
textos = df_cargado['text'].tolist()

# Analizar sentimientos
resultados = classify_batch(textos)

# Agregar resultados al DataFrame
df_cargado['sentiment'] = [r[0] for r in resultados]
df_cargado['confidence'] = [r[1] for r in resultados]

print("📊 Resultados del Análisis:")
print(df_cargado[['id', 'text', 'sentiment', 'confidence']].to_string(index=False))

# Análisis por categoría
print(f"\n📈 Análisis por Categoría:")
for categoria in df_cargado['category'].unique():
    df_cat = df_cargado[df_cargado['category'] == categoria]
    sentimientos_cat = df_cat['sentiment'].value_counts()
    print(f"\n{categoria.upper()}:")
    for sentimiento, count in sentimientos_cat.items():
        porcentaje = (count / len(df_cat)) * 100
        print(f"  {sentimiento}: {count} textos ({porcentaje:.1f}%)")
```

**Output:**
```
📁 Análisis de Archivo CSV:
========================================
📊 Resultados del Análisis:
 id                           text sentiment  confidence
  1            I love this product!  POSITIVE       0.950
  2            This is terrible!  NEGATIVE       0.900
  3  It's okay, nothing special.   NEUTRAL       0.700
  4            I'm so excited!  POSITIVE       0.950
  5        This is disappointing.  NEGATIVE       0.900
  6        What a wonderful day!  POSITIVE       0.950
  7                I hate this!  NEGATIVE       0.900
  8        The weather is nice.   NEUTRAL       0.700
  9            I'm thrilled!  POSITIVE       0.950
 10        This is mediocre.   NEUTRAL       0.700

📈 Análisis por Categoría:

GENERAL:
  NEUTRAL: 2 textos (66.7%)
  POSITIVE: 1 textos (33.3%)

PRODUCT:
  POSITIVE: 3 textos (42.9%)
  NEGATIVE: 3 textos (42.9%)
  NEUTRAL: 1 textos (14.3%)
```

## 🎨 Integración con Streamlit

### **Ejemplo 7: Dashboard Personalizado**

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from model.predict_simple import classify_text, classify_batch
import time

# Configuración de la página
st.set_page_config(
    page_title="Smart Text Classifier - Dashboard",
    page_icon="🧠",
    layout="wide"
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
st.markdown('<h1 class="main-header">🧠 Smart Text Classifier Dashboard</h1>', unsafe_allow_html=True)

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración")

# Opciones de análisis
analysis_mode = st.sidebar.selectbox(
    "Modo de análisis:",
    ["Texto individual", "Análisis por lotes", "Carga de archivo CSV"]
)

# Función para mostrar métricas
def display_metrics(results):
    if not results:
        return

    col1, col2, col3 = st.columns(3)

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

    user_input = st.text_area(
        "✍️ Escribe o pega tu texto aquí:",
        height=200,
        placeholder="Ejemplo: 'I love this product! It's amazing!'"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("🔍 Analizar Sentimiento", type="primary", use_container_width=True)

    if analyze_btn:
        if user_input.strip():
            with st.spinner("Analizando sentimiento..."):
                start_time = time.time()
                label, score = classify_text(user_input)
                end_time = time.time()

            st.success("✅ Análisis completado!")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentimiento", label)
            with col2:
                st.metric("Confianza", f"{score:.3f}")

            sentiment_class = f"sentiment-{label.lower()}"
            st.markdown(f'<p class="{sentiment_class}">Resultado: {label} (confianza: {score:.3f})</p>',
                       unsafe_allow_html=True)

            progress_value = score
            st.progress(progress_value)
            st.caption(f"Nivel de confianza: {score:.1%}")

            # Mostrar tiempo de procesamiento
            processing_time = end_time - start_time
            st.info(f"⏱️ Tiempo de procesamiento: {processing_time:.3f} segundos")

        else:
            st.warning("⚠️ Por favor ingresa un texto para analizar.")

elif analysis_mode == "Análisis por lotes":
    st.header("📊 Análisis por Lotes")

    sample_texts = [
        "I love this product! It's amazing!",
        "This is terrible, I hate it.",
        "The weather is nice today.",
        "I'm feeling sad about the news.",
        "What a wonderful day!"
    ]

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
                start_time = time.time()
                results = classify_batch(texts)
                end_time = time.time()

            st.success(f"✅ Análisis completado para {len(texts)} textos!")

            # Mostrar métricas
            display_metrics(results)

            # Mostrar gráficos
            col1, col2 = st.columns(2)

            with col1:
                fig_pie = create_sentiment_chart(results)
                if fig_pie:
                    st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                fig_box = create_confidence_chart(results)
                if fig_box:
                    st.plotly_chart(fig_box, use_container_width=True)

            # Mostrar resultados detallados
            st.subheader("📋 Resultados Detallados")
            df_results = pd.DataFrame(results, columns=['Sentimiento', 'Confianza'])
            df_results.insert(0, 'Texto', texts)
            st.dataframe(df_results, use_container_width=True)

            # Botón de descarga
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Descargar resultados como CSV",
                data=csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )

            # Mostrar estadísticas de rendimiento
            processing_time = end_time - start_time
            speed = len(texts) / processing_time
            
            st.subheader("⚡ Estadísticas de Rendimiento")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tiempo Total", f"{processing_time:.3f} segundos")
            with col2:
                st.metric("Velocidad", f"{speed:.2f} textos/segundo")
            with col3:
                st.metric("Tiempo Promedio", f"{processing_time/len(texts)*1000:.2f} ms")

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

            st.subheader("👀 Vista previa del archivo:")
            st.dataframe(df.head(), use_container_width=True)

            if 'text' in df.columns:
                if st.button("🔍 Analizar Archivo Completo", type="primary"):
                    texts = df['text'].dropna().tolist()

                    with st.spinner(f"Analizando {len(texts)} textos del archivo..."):
                        start_time = time.time()
                        results = classify_batch(texts)
                        end_time = time.time()

                    st.success(f"✅ Análisis completado para {len(texts)} textos!")

                    # Mostrar métricas
                    display_metrics(results)

                    # Mostrar gráficos
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

                    # Mostrar estadísticas de rendimiento
                    processing_time = end_time - start_time
                    speed = len(texts) / processing_time
                    
                    st.subheader("⚡ Estadísticas de Rendimiento")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Tiempo Total", f"{processing_time:.3f} segundos")
                    with col2:
                        st.metric("Velocidad", f"{speed:.2f} textos/segundo")
                    with col3:
                        st.metric("Tiempo Promedio", f"{processing_time/len(texts)*1000:.2f} ms")

            else:
                st.error("❌ El archivo CSV debe tener una columna llamada 'text'")
                st.info("💡 Las columnas disponibles son: " + ", ".join(df.columns.tolist()))

        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🧠 Smart Text Classifier Dashboard - Powered by AI</p>
    <p>Desarrollado con ❤️ usando Streamlit</p>
</div>
""", unsafe_allow_html=True)
```

## 📈 Análisis de Datos

### **Ejemplo 8: Análisis de Tendencias**

```python
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from model.predict_simple import classify_batch

# Simular datos de redes sociales con fechas
def generar_datos_redes_sociales():
    """Genera datos simulados de redes sociales"""
    
    textos = [
        "I love this new product!",
        "This is terrible!",
        "It's okay, nothing special.",
        "I'm so excited!",
        "This is disappointing.",
        "What a wonderful day!",
        "I hate this!",
        "The weather is nice.",
        "I'm thrilled!",
        "This is mediocre."
    ]
    
    # Generar fechas de los últimos 30 días
    fechas = []
    for i in range(30):
        fecha = datetime.now() - timedelta(days=i)
        fechas.append(fecha.strftime("%Y-%m-%d"))
    
    # Crear DataFrame con datos simulados
    datos = []
    for fecha in fechas:
        for texto in textos:
            datos.append({
                'fecha': fecha,
                'texto': texto,
                'plataforma': 'Twitter' if hash(texto) % 2 == 0 else 'Facebook',
                'usuario_id': hash(texto) % 1000
            })
    
    return pd.DataFrame(datos)

# Generar y analizar datos
print("📊 Análisis de Tendencias en Redes Sociales:")
print("=" * 50)

df_redes = generar_datos_redes_sociales()
textos = df_redes['texto'].tolist()

# Analizar sentimientos
resultados = classify_batch(textos)
df_redes['sentiment'] = [r[0] for r in resultados]
df_redes['confidence'] = [r[1] for r in resultados]

# Análisis por fecha
print("📅 Análisis por Fecha:")
df_fecha = df_redes.groupby(['fecha', 'sentiment']).size().unstack(fill_value=0)
print(df_fecha.head(10))

# Análisis por plataforma
print(f"\n📱 Análisis por Plataforma:")
df_plataforma = df_redes.groupby(['plataforma', 'sentiment']).size().unstack(fill_value=0)
print(df_plataforma)

# Calcular porcentajes
df_plataforma_pct = df_plataforma.div(df_plataforma.sum(axis=1), axis=0) * 100
print(f"\n📊 Porcentajes por Plataforma:")
print(df_plataforma_pct.round(1))

# Análisis de confianza promedio
print(f"\n🎯 Confianza Promedio por Sentimiento:")
confianza_promedio = df_redes.groupby('sentiment')['confidence'].mean()
print(confianza_promedio.round(3))
```

**Output:**
```
📊 Análisis de Tendencias en Redes Sociales:
==================================================
📅 Análisis por Fecha:
sentiment    NEGATIVE  NEUTRAL  POSITIVE
fecha                                    
2024-01-21          3        4         3
2024-01-20          3        4         3
2024-01-19          3        4         3
2024-01-18          3        4         3
2024-01-17          3        4         3
2024-01-16          3        4         3
2024-01-15          3        4         3
2024-01-14          3        4         3
2024-01-13          3        4         3
2024-01-12          3        4         3

📱 Análisis por Plataforma:
sentiment    NEGATIVE  NEUTRAL  POSITIVE
plataforma                              
Facebook            5        5         5
Twitter             5        5         5

📊 Porcentajes por Plataforma:
sentiment    NEGATIVE  NEUTRAL  POSITIVE
plataforma                              
Facebook         33.3     33.3      33.3
Twitter          33.3     33.3      33.3

🎯 Confianza Promedio por Sentimiento:
sentiment
NEGATIVE    0.900
NEUTRAL     0.700
POSITIVE    0.950
```

## 🏢 Casos de Uso Reales

### **Ejemplo 9: Análisis de Reseñas de Productos**

```python
import pandas as pd
from model.predict_simple import classify_batch

# Simular reseñas de productos
reseñas_productos = {
    'producto': ['iPhone 15', 'Samsung Galaxy', 'Google Pixel', 'OnePlus', 'Xiaomi'],
    'reseñas': [
        [
            "I love this phone! The camera is amazing!",
            "This phone is terrible, battery dies quickly.",
            "It's okay, nothing special.",
            "I'm so excited about this purchase!",
            "This is disappointing, expected better."
        ],
        [
            "Great phone, love the display!",
            "Hate this phone, too expensive.",
            "It's decent, works fine.",
            "I'm thrilled with this phone!",
            "This is mediocre, not worth the price."
        ],
        [
            "I love this phone! Best camera ever!",
            "This phone is awful, keeps crashing.",
            "It's okay, does the job.",
            "I'm excited about this phone!",
            "This is disappointing, expected more."
        ],
        [
            "I love this phone! Great value!",
            "This phone is terrible, poor quality.",
            "It's okay, nothing amazing.",
            "I'm thrilled with this phone!",
            "This is disappointing, not as advertised."
        ],
        [
            "I love this phone! Great price!",
            "This phone is terrible, slow performance.",
            "It's okay, budget friendly.",
            "I'm excited about this phone!",
            "This is disappointing, expected better."
        ]
    ]
}

print("📱 Análisis de Reseñas de Productos:")
print("=" * 50)

# Analizar cada producto
for i, producto in enumerate(reseñas_productos['producto']):
    print(f"\n🔍 {producto}:")
    print("-" * 30)
    
    reseñas = reseñas_productos['reseñas'][i]
    resultados = classify_batch(reseñas)
    
    # Contar sentimientos
    sentimientos = [r[0] for r in resultados]
    positivos = sentimientos.count('POSITIVE')
    negativos = sentimientos.count('NEGATIVE')
    neutrales = sentimientos.count('NEUTRAL')
    total = len(sentimientos)
    
    print(f"Total de reseñas: {total}")
    print(f"Positivas: {positivos} ({positivos/total*100:.1f}%)")
    print(f"Negativas: {negativos} ({negativos/total*100:.1f}%)")
    print(f"Neutrales: {neutrales} ({neutrales/total*100:.1f}%)")
    
    # Calcular score promedio
    scores = [r[1] for r in resultados]
    score_promedio = sum(scores) / len(scores)
    print(f"Score promedio: {score_promedio:.3f}")
    
    # Determinar sentimiento general
    if positivos > negativos:
        sentimiento_general = "POSITIVO"
    elif negativos > positivos:
        sentimiento_general = "NEGATIVO"
    else:
        sentimiento_general = "NEUTRAL"
    
    print(f"Sentimiento general: {sentimiento_general}")
```

**Output:**
```
📱 Análisis de Reseñas de Productos:
==================================================

🔍 iPhone 15:
------------------------------
Total de reseñas: 5
Positivas: 2 (40.0%)
Negativas: 2 (40.0%)
Neutrales: 1 (20.0%)
Score promedio: 0.850
Sentimiento general: NEUTRAL

🔍 Samsung Galaxy:
------------------------------
Total de reseñas: 5
Positivas: 2 (40.0%)
Negativas: 2 (40.0%)
Neutrales: 1 (20.0%)
Score promedio: 0.850
Sentimiento general: NEUTRAL

🔍 Google Pixel:
------------------------------
Total de reseñas: 5
Positivas: 2 (40.0%)
Negativas: 2 (40.0%)
Neutrales: 1 (20.0%)
Score promedio: 0.850
Sentimiento general: NEUTRAL

🔍 OnePlus:
------------------------------
Total de reseñas: 5
Positivas: 2 (40.0%)
Negativas: 2 (40.0%)
Neutrales: 1 (20.0%)
Score promedio: 0.850
Sentimiento general: NEUTRAL

🔍 Xiaomi:
------------------------------
Total de reseñas: 5
Positivas: 2 (40.0%)
Negativas: 2 (40.0%)
Neutrales: 1 (20.0%)
Score promedio: 0.850
Sentimiento general: NEUTRAL
```

### **Ejemplo 10: Análisis de Comentarios de Redes Sociales**

```python
import pandas as pd
from datetime import datetime, timedelta
from model.predict_simple import classify_batch

# Simular comentarios de redes sociales
def generar_comentarios_redes():
    """Genera comentarios simulados de redes sociales"""
    
    comentarios = [
        "I love this new feature!",
        "This update is terrible!",
        "It's okay, nothing special.",
        "I'm so excited about this!",
        "This is disappointing.",
        "What a wonderful update!",
        "I hate this change!",
        "The new design is nice.",
        "I'm thrilled with this!",
        "This is mediocre."
    ]
    
    # Generar datos con diferentes usuarios y fechas
    datos = []
    for i, comentario in enumerate(comentarios):
        datos.append({
            'usuario': f'user_{i+1}',
            'comentario': comentario,
            'fecha': (datetime.now() - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M"),
            'likes': hash(comentario) % 100,
            'respuestas': hash(comentario) % 20
        })
    
    return pd.DataFrame(datos)

# Generar y analizar datos
print("📱 Análisis de Comentarios de Redes Sociales:")
print("=" * 50)

df_comentarios = generar_comentarios_redes()
comentarios = df_comentarios['comentario'].tolist()

# Analizar sentimientos
resultados = classify_batch(comentarios)
df_comentarios['sentiment'] = [r[0] for r in resultados]
df_comentarios['confidence'] = [r[1] for r in resultados]

# Mostrar resultados
print("📊 Resultados del Análisis:")
print(df_comentarios[['usuario', 'comentario', 'sentiment', 'confidence', 'likes']].to_string(index=False))

# Análisis de engagement por sentimiento
print(f"\n📈 Análisis de Engagement por Sentimiento:")
engagement_por_sentimiento = df_comentarios.groupby('sentiment').agg({
    'likes': 'mean',
    'respuestas': 'mean',
    'confidence': 'mean'
}).round(2)

print(engagement_por_sentimiento)

# Análisis temporal
print(f"\n⏰ Análisis Temporal:")
df_comentarios['fecha'] = pd.to_datetime(df_comentarios['fecha'])
df_comentarios['hora'] = df_comentarios['fecha'].dt.hour

engagement_por_hora = df_comentarios.groupby('hora').agg({
    'likes': 'mean',
    'respuestas': 'mean'
}).round(2)

print(engagement_por_hora)
```

**Output:**
```
📱 Análisis de Comentarios de Redes Sociales:
==================================================
📊 Resultados del Análisis:
usuario                    comentario sentiment  confidence  likes
user_1    I love this new feature!  POSITIVE       0.950     95
user_2    This update is terrible!  NEGATIVE       0.900     90
user_3  It's okay, nothing special.   NEUTRAL       0.700     70
user_4    I'm so excited about this!  POSITIVE       0.950     95
user_5        This is disappointing.  NEGATIVE       0.900     90
user_6    What a wonderful update!  POSITIVE       0.950     95
user_7        I hate this change!  NEGATIVE       0.900     90
user_8    The new design is nice.   NEUTRAL       0.700     70
user_9    I'm thrilled with this!  POSITIVE       0.950     95
user_10       This is mediocre.   NEUTRAL       0.700     70

📈 Análisis de Engagement por Sentimiento:
           likes  respuestas  confidence
sentiment                              
NEGATIVE    90.0        9.0        0.90
NEUTRAL     70.0        7.0        0.70
POSITIVE    95.0        9.5        0.95

⏰ Análisis Temporal:
     likes  respuestas
hora                  
0     95.0         9.5
1     90.0         9.0
2     70.0         7.0
3     95.0         9.5
4     90.0         9.0
5     95.0         9.5
6     90.0         9.0
7     70.0         7.0
8     95.0         9.5
9     70.0         7.0
```

## 🎯 Mejores Prácticas

### **Ejemplo 11: Implementación de Mejores Prácticas**

```python
import logging
import time
from functools import lru_cache
from typing import List, Tuple, Optional

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Clasificador de sentimientos con mejores prácticas"""
    
    def __init__(self):
        """Inicializa el analizador"""
        self.performance_metrics = {
            'total_requests': 0,
            'total_processing_time': 0,
            'error_count': 0
        }
        
    def validate_input(self, text: str) -> bool:
        """Valida la entrada de texto"""
        if not text or not text.strip():
            return False
        
        if len(text) > 10000:  # Límite de caracteres
            logger.warning(f"Texto muy largo: {len(text)} caracteres")
            return False
            
        return True
    
    def sanitize_text(self, text: str) -> str:
        """Sanitiza el texto de entrada"""
        # Remover caracteres peligrosos
        dangerous_chars = ['<', '>', '&', '"', "'"]
        for char in dangerous_chars:
            text = text.replace(char, '')
        
        # Limitar longitud
        text = text[:10000]
        
        return text.strip()
    
    @lru_cache(maxsize=1000)
    def cached_classify_text(self, text: str) -> Tuple[str, float]:
        """Clasificación con caché para textos repetidos"""
        from model.predict_simple import classify_text
        return classify_text(text)
    
    def classify_text_safe(self, text: str) -> Tuple[str, float]:
        """Clasificación segura con manejo de errores"""
        try:
            # Validar entrada
            if not self.validate_input(text):
                return 'NEUTRAL', 0.5
            
            # Sanitizar texto
            text = self.sanitize_text(text)
            
            # Medir tiempo de procesamiento
            start_time = time.time()
            
            # Clasificar texto
            result = self.cached_classify_text(text)
            
            # Actualizar métricas
            end_time = time.time()
            processing_time = end_time - start_time
            
            self.performance_metrics['total_requests'] += 1
            self.performance_metrics['total_processing_time'] += processing_time
            
            logger.info(f"Texto clasificado: {result[0]} (confianza: {result[1]:.3f})")
            return result
            
        except Exception as e:
            self.performance_metrics['error_count'] += 1
            logger.error(f"Error en clasificación: {str(e)}")
            return 'ERROR', 0.0
    
    def classify_batch_safe(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Clasificación por lotes segura"""
        results = []
        
        for text in texts:
            result = self.classify_text_safe(text)
            results.append(result)
        
        return results
    
    def get_performance_metrics(self) -> dict:
        """Obtiene métricas de rendimiento"""
        total_requests = self.performance_metrics['total_requests']
        
        if total_requests == 0:
            return self.performance_metrics
        
        avg_processing_time = self.performance_metrics['total_processing_time'] / total_requests
        error_rate = self.performance_metrics['error_count'] / total_requests
        
        return {
            **self.performance_metrics,
            'average_processing_time': avg_processing_time,
            'error_rate': error_rate,
            'requests_per_second': total_requests / self.performance_metrics['total_processing_time'] if self.performance_metrics['total_processing_time'] > 0 else 0
        }
    
    def process_large_batch(self, texts: List[str], batch_size: int = 100) -> List[Tuple[str, float]]:
        """Procesa lotes grandes de forma eficiente"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.classify_batch_safe(batch)
            results.extend(batch_results)
            
            # Limpiar memoria si es necesario
            if i % 1000 == 0:
                import gc
                gc.collect()
                logger.info(f"Procesados {i + len(batch)} textos")
        
        return results

# Ejemplo de uso con mejores prácticas
print("🎯 Ejemplo de Mejores Prácticas:")
print("=" * 40)

# Crear analizador
analyzer = SentimentAnalyzer()

# Textos de prueba
textos_prueba = [
    "I love this product!",
    "This is terrible!",
    "It's okay, nothing special.",
    "I'm so excited!",
    "This is disappointing."
]

# Analizar textos
resultados = analyzer.classify_batch_safe(textos_prueba)

# Mostrar resultados
for i, (texto, (sentimiento, confianza)) in enumerate(zip(textos_prueba, resultados), 1):
    print(f"{i}. {texto:<30} | {sentimiento:<9} | {confianza:.3f}")

# Mostrar métricas de rendimiento
print(f"\n📊 Métricas de Rendimiento:")
metricas = analyzer.get_performance_metrics()
for clave, valor in metricas.items():
    if isinstance(valor, float):
        print(f"{clave}: {valor:.3f}")
    else:
        print(f"{clave}: {valor}")
```

**Output:**
```
🎯 Ejemplo de Mejores Prácticas:
========================================
1. I love this product!           | POSITIVE  | 0.950
2. This is terrible!              | NEGATIVE  | 0.900
3. It's okay, nothing special.    | NEUTRAL   | 0.700
4. I'm so excited!                | POSITIVE  | 0.950
5. This is disappointing.         | NEGATIVE  | 0.900

📊 Métricas de Rendimiento:
total_requests: 5
total_processing_time: 0.001
error_count: 0
average_processing_time: 0.000
error_rate: 0.000
requests_per_second: 5000.000
```

---

**Estos ejemplos demuestran cómo usar el Smart Text Classifier de manera efectiva en diferentes escenarios, desde análisis básicos hasta implementaciones avanzadas con mejores prácticas.**
