import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import cv2
from google.cloud import storage

# Configuración de Google Cloud Storage
bucket_name = "modelos_interfaz"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

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
    # ... (tu código existente para obtener_region_ventriculos)

# Cargar modelos desde Google Cloud Storage
def load_model_from_gcs(model_name):
    blob = bucket.blob(model_name)
    blob.download_to_filename(model_name)
    return models.load_model(model_name)

# Interfaz de Streamlit
st.set_page_config(page_title="Diagnóstico Cardíaco", page_icon=":heart:")
st.title("Diagnóstico Cardíaco Asistido por IA")
st.subheader("Segmentación y Clasificación Ventricular")

# Estilos personalizados (CSS)
# ... (tu código CSS)

# Cargar modelos desde GCS
with st.spinner("Cargando modelos..."):
    modelo_segmentacion = load_model_from_gcs("Hope.h5")
    modelo_clasificacion = load_model_from_gcs("Clas.h5")

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
        st.image(segmented_image, caption='Imagen Segmentada', use_column_width=True)

        # Preprocesamiento para la clasificación
        imagen_recortada = obtener_region_ventriculos(segmented_image)

        # Preprocesamiento adicional para el modelo de clasificación
        img_array = np.array(imagen_recortada)
        # ... (resto de tu código de preprocesamiento)

        # Clasificación
        clasificacion_output = modelo_clasificacion.predict(img_array)

    # Mostrar resultado de la clasificación
    clase_predicha = np.argmax(clasificacion_output)
    resultado = "Diástole" if clase_predicha == 0 else "Sístole"

    if resultado == "Diástole":
        st.markdown(f'<h1 style="color:blue; text-align:center;">Clasificación: {resultado}</h1>', unsafe_allow_html=True)
    else:
        st.markdown(f'<h1 style="color:red; text-align:center;">Clasificación: {resultado}</h1>', unsafe_allow_html=True)
