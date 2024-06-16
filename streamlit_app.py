import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import cv2
import requests
import tempfile
import os

# URLs públicas de los modelos en Google Cloud Storage
MODELO_SEGMENTACION_URL = "https://storage.googleapis.com/modelos-interfaz/Hope.h5"
MODELO_CLASIFICACION_URL = "https://storage.googleapis.com/modelos-interfaz/Clas.h5"

# Función para cargar la imagen y preprocesarla
def load_and_preprocess_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path).convert('L')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Función para obtener la región de los ventrículos
def obtener_region_ventriculos(segmented_image):
    segmented_array = np.array(segmented_image)
    contours, _ = cv2.findContours(segmented_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(segmented_array)
        cv2.drawContours(mask, [largest_contour], 0, (255), -1)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = segmented_array[y:y+h, x:x+w]
        resized_image = cv2.resize(cropped_image, (224, 224))
        return Image.fromarray(resized_image)
    else:
        return None

# Interfaz de Streamlit
st.set_page_config(page_title="Diagnóstico Cardíaco", page_icon=":heart:")
st.title("Diagnóstico Cardíaco Asistido por IA")
st.subheader("Segmentación y Clasificación Ventricular")

# Estilos personalizados (CSS)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #333333;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    .stTitle {
        text-align: center;
        color: #2196F3;
    }
    .stSubheader {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cargar modelos desde las URLs
with st.spinner("Cargando modelos..."):
    # ... (Descargar y cargar modelos de segmentación y clasificación) ...

# Cargar la imagen
uploaded_file = st.file_uploader("Sube una imagen de resonancia magnética cardíaca", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Mostrar imagen original
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen Original', use_column_width=True)

    with st.spinner("Procesando..."):
        # Segmentación
        preprocessed_image = load_and_preprocess_image(uploaded_file)
        segmented_output = modelo_segmentacion.predict(preprocessed_image)
        segmented_image = (segmented_output[0, :, :, 0] > 0.5).astype(np.uint8) * 255
        segmented_image = Image.fromarray(segmented_image).convert('L')

        # Crear imagen combinada (máscara + original)
        image_array = np.array(image)
        segmented_array = np.array(segmented_image)

        # Aplicar la máscara a la imagen original
        combined_image = np.where(segmented_array[:, :, None] > 0, image_array, 0)
        combined_image = Image.fromarray(combined_image)

        # Mostrar imágenes
        st.image(segmented_image, caption='Imagen Segmentada', use_column_width=True)
        st.image(combined_image, caption='Imagen Combinada', use_column_width=True)

        # Preprocesamiento para la clasificación
        imagen_recortada = obtener_region_ventriculos(segmented_image)
        # ... (Resto del código de clasificación y visualización de resultados) ...
