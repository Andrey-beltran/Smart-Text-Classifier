import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch
import os

def load_data(csv_path="data/samples.csv"):
    """Carga los datos desde el archivo CSV"""
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"Archivo {csv_path} no encontrado. Creando datos de ejemplo...")
        return create_sample_data()

def create_sample_data():
    """Crea datos de ejemplo si no existe el archivo CSV"""
    sample_data = {
        'text': [
            "I love this product! It's amazing!",
            "This is terrible, I hate it.",
            "The weather is nice today.",
            "I'm feeling sad about the news.",
            "What a wonderful day!",
            "This movie is boring.",
            "I'm so excited about the trip!",
            "This food tastes awful.",
            "The book is interesting.",
            "I'm disappointed with the service."
        ],
        'label': [
            'POSITIVE', 'NEGATIVE', 'NEUTRAL', 'NEGATIVE', 
            'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 
            'POSITIVE', 'NEGATIVE'
        ]
    }
    return pd.DataFrame(sample_data)

def preprocess_data(df):
    """Preprocesa los datos para el entrenamiento"""
    # Mapear etiquetas a números
    label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
    df['label_id'] = df['label'].map(label_mapping)
    
    # Dividir en entrenamiento y validación
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), 
        df['label_id'].tolist(), 
        test_size=0.2, 
        random_state=42
    )
    
    return train_texts, val_texts, train_labels, val_labels

def tokenize_data(texts, tokenizer, max_length=128):
    """Tokeniza los textos"""
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

def train_model(model_name="distilbert-base-uncased", output_dir="./model_checkpoint"):
    """Entrena el modelo de clasificación de sentimientos"""
    
    # Cargar datos
    df = load_data()
    train_texts, val_texts, train_labels, val_labels = preprocess_data(df)
    
    # Cargar tokenizer y modelo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3,
        id2label={0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"},
        label2id={"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    )
    
    # Tokenizar datos
    train_encodings = tokenize_data(train_texts, tokenizer)
    val_encodings = tokenize_data(val_texts, tokenizer)
    
    # Crear datasets
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })
    
    val_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': val_labels
    })
    
    # Configurar argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # Crear trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Entrenar modelo
    print("Iniciando entrenamiento...")
    trainer.train()
    
    # Guardar modelo
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"Modelo entrenado y guardado en {output_dir}")
    return trainer, tokenizer

if __name__ == "__main__":
    trainer, tokenizer = train_model()

