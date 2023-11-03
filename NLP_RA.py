# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 09:45:27 2023

@author: Fantasma
"""

from pdfminer.high_level import extract_text
import re
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, GRU, MultiHeadAttention, Dense
from tensorflow.keras.models import Model


archivo_csv = ''
# Ruta del archivo PDF que deseas convertir a texto
pdf_file_path = 'E:/DESCARGAS 10-23/0xWord/0xWord/LI0X0057-Indice.pdf'

def extract_text_from_pdf(pdf_path):
    text = ''
    try:
        text = extract_text(pdf_path)
    except Exception as e:
        print(f"Error al procesar el PDF: {str(e)}")

    return text

# Llama a la función para extraer el texto del PDF
texto_extraido = extract_text_from_pdf(pdf_file_path)


# Si deseas guardar el texto en un archivo de texto
with open('texto_extraido.txt', 'w', encoding='utf-8') as file:
    file.write(texto_extraido)



def clean(texto):
    # Eliminar caracteres no alfabéticos y números
    #texto = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]', '', texto)
    #eliminamos links 
    texto = texto.lower()
    texto = texto.replace('ñ', 'n')
    texto = texto.replace('á', 'a')
    texto = texto.replace('é', 'e')
    texto = texto.replace('í', 'i')
    texto = texto.replace('ó', 'o')
    texto = texto.replace('ú', 'u')
    texto = re.sub(r"https?://[A-Za-z0-9./]+", '', texto)
    #nos quedamos solo con caracteres 
    texto = re.sub(r"[^A-Za-z!?']",' ', texto)
    texto = re.sub(r"[...]", ' ', texto)
    #quitamos espacios vacios
    texto = re.sub(r" +", ' ', texto)
    #retornamos variable

  

    # Eliminar espacios en blanco adicionales
    texto = ' '.join(texto.split())

    return texto

texto_limpio = clean(texto_extraido)

#prueba devectorizacion
text_vec = tf.keras.layers.TextVectorization(split="character",
                                             standardize="lower")
text_vec.adapt([texto_limpio])
endeado = text_vec([texto_limpio])[0]


endeado -= 2 #el drop de tokens 0 paddeado y 1 no conocido porque no se usa
n_tokens = text_vec.vocabulary_size() - 2 #numero de distincion 39
data = len(endeado) #total de numeros para caracteres = 1115394
    

# Crear un conjunto de caracteres únicos en el texto
chars = sorted(list(set(texto_limpio)))
char_to_index = {char: index for index, char in enumerate(chars)}
index_to_char = {index: char for index, char in enumerate(chars)}


#prueba de secuencia  

def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length))
    if shuffle:
        ds = ds.shuffle(buffer_size=100_000, seed=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)

# Llama a la función para crear tu conjunto de datos
sequence = endeado  # Reemplaza con tu secuencia de texto
length = 50  # Longitud deseada para las secuencias
shuffle = True  # Opcional, si deseas mezclar las secuencias
seed = 42  # Opcional, semilla para la mezcla aleatoria
batch_size = 64  # Tamaño del lote

# Crea el conjunto de datos
dataset = to_dataset(sequence, length, shuffle, seed, batch_size)

length = 50

tf.random.set_seed(42)
entrenamiento = to_dataset(endeado[:1_000_000], length =length, shuffle=True,
                           seed=42)
validacion = to_dataset(endeado[1_000_000: 1_060_000], length=length)
testeo = to_dataset(endeado[1_060_000:], length=length)




# Imprimir el texto limpio
print("Texto limpio:")
print(texto_limpio)
print(dataset)



