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

# Función para cargar modelos con caché
@st.cache(allow_output_mutation=True)
def load_model(url):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
        response = requests.get(url)
        response.raise_for_status()
        temp_file.write(response.content)
        temp_file.flush()
        model = models.load_model(temp_file.name)
    os.remove(temp_file.name)
    return model

# Cargar modelos desde las URLs
with st.spinner("Cargando modelos..."):
    modelo_segmentacion = load_model(MODELO_SEGMENTACION_URL)
    modelo_clasificacion = load_model(MODELO_CLASIFICACION_URL)

# Función para cargar y preprocesar la imagen
def load_and_preprocess_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path).convert('L')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Función para obtener la región de los ventrículos y la máscara
def obtener_region_ventriculos(segmented_image, original_size):
    segmented_array = np.array(segmented_image)
    contours, _ = cv2.findContours(segmented_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:  # Verifica si se encontraron contornos
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(segmented_array)
        cv2.drawContours(mask, [largest_contour], 0, (255), -1)
        
        # Obtener el bounding box del contorno más grande
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = segmented_array[y:y+h, x:x+w]
        resized_image = cv2.resize(cropped_image, (224, 224))

        # Escalar contorno a la imagen original
        scale_x = original_size[0] / segmented_image.size[0]
        scale_y = original_size[1] / segmented_image.size[1]
        scaled_contour = np.array([[[int(point[0][0] * scale_x), int(point[0][1] * scale_y)]] for point in largest_contour])

        return Image.fromarray(resized_image), mask, scaled_contour
    else:
        return None, None, None  # Devuelve None si no se encontraron contornos

# Función para dibujar contorno en la imagen original
def dibujar_contorno(imagen_original, contorno):
    original_array = np.array(imagen_original)
    
    # Convertir a RGB si es necesario
    if len(original_array.shape) == 2:
        original_array = cv2.cvtColor(original_array, cv2.COLOR_GRAY2RGB)
    elif original_array.shape[2] == 4:  # Si tiene un canal alfa, convertir a RGB
        original_array = cv2.cvtColor(original_array, cv2.COLOR_RGBA2RGB)
    
    # Cambiar el color de la línea aquí (formato BGR)
    cv2.drawContours(original_array, [contorno], -1, (0, 0, 255), 2)  # Color rojo (BGR: (0, 0, 255))
    
    return Image.fromarray(original_array)

# Interfaz de Streamlit
st.set_page_config(page_title="Diagnóstico Cardíaco", page_icon=":heart:")
st.title("Diagnóstico Cardíaco Asistido por IA")
st.subheader("Segmentación y Clasificación Ventricular")

# Estilos personalizados (CSS)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
        color: #333333;
    }
    .stButton button {
        background-color: #0066cc;
        color: white;
    }
    .stTitle, .stSubheader {
        text-align: center;
        color: #333333;
    }
    .resultado {
        color: #0066cc;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cargar la imagen
uploaded_file = st.file_uploader("Sube una imagen de resonancia magnética cardíaca", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Mostrar imagen original
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagen Original', use_column_width=True)

        with st.spinner("Procesando..."):
            # Segmentación
            preprocessed_image = load_and_preprocess_image(uploaded_file)
            segmented_output = modelo_segmentacion.predict(preprocessed_image)
            segmented_image = (segmented_output[0, :, :, 0] > 0.5).astype(np.uint8) * 255
            segmented_image = Image.fromarray(segmented_image).convert('L')

            # Preprocesamiento para la clasificación
            imagen_recortada, mask, contorno = obtener_region_ventriculos(segmented_image, image.size)

            if imagen_recortada is not None:  # Verifica si se encontró la región
                # Preprocesamiento adicional para el modelo de clasificación
                img_array = np.array(imagen_recortada)
                if img_array.ndim == 2:
                    img_array = np.stack((img_array,) * 3, axis=-1)
                img_array = cv2.resize(img_array, (224, 224))
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Clasificación
                clasificacion_output = modelo_clasificacion.predict(img_array)

                # Mostrar resultado de la clasificación
                clase_predicha = np.argmax(clasificacion_output)
                resultado = "Diástole" if clase_predicha == 0 else "Sístole"

                # Dibujar contorno en la imagen original
                imagen_con_contorno = dibujar_contorno(image, contorno)
                st.image(imagen_con_contorno, caption='Imagen con Contorno', use_column_width=True)

                # Mostrar resultado en blanco
                st.markdown(f'<div class="resultado">{resultado}</div>', unsafe_allow_html=True)
            else:
                st.error("No se pudo encontrar la región de los ventrículos en la imagen.")
    except Exception as
