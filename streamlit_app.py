import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
import cv2
import altair as alt
import pandas as pd


# Función para cargar la imagen y preprocesarla (puedes personalizar esto)
def load_and_preprocess_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path).convert('L') 
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=-1)  
    img_array = np.expand_dims(img_array, axis=0)   
    return img_array

# Función para obtener la región de los ventrículos (adaptada a tu modelo de clasificación)
def obtener_region_ventriculos(segmented_image):
    segmented_array = np.array(segmented_image)
    
    # Buscamos los contornos en la imagen segmentada
    contours, _ = cv2.findContours(segmented_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Asumimos que el contorno más grande corresponde a la región de interés (ventrículos)
    largest_contour = max(contours, key=cv2.contourArea)

    # Creamos una máscara con el mismo tamaño de la imagen y la rellenamos de ceros
    mask = np.zeros_like(segmented_array)

    # Dibujamos el contorno más grande en la máscara
    cv2.drawContours(mask, [largest_contour], 0, (255), -1)
    
    # Extraemos la región de interés usando la máscara
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = segmented_array[y:y+h, x:x+w]

    # Redimensionamos la imagen recortada al tamaño de entrada del modelo de clasificación (224x224 en este caso)
    resized_image = cv2.resize(cropped_image, (224, 224))

    # Convertimos la imagen redimensionada en una imagen PIL para poder mostrarla en Streamlit
    resized_image = Image.fromarray(resized_image)

    return resized_image

# Interfaz de Streamlit
st.set_page_config(page_title="Diagnóstico Cardíaco", page_icon=":heart:")  # Título y ícono de página

st.title("Diagnóstico Cardíaco Asistido por IA")  # Título más descriptivo
st.subheader("Segmentación y Clasificación Ventricular")

# Estilos personalizados (CSS)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #333333;  /* Fondo gris claro */
    }
    .stButton button {
        background-color: #4CAF50; /* Verde */
        color: white;
    }
    .stTitle {
        text-align: center;
        color: #2196F3;  /* Azul */
    }
    .stSubheader {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cargar modelos
modelo_segmentacion = models.load_model("Hope.h5")
modelo_clasificacion = models.load_model("Clas.h5")

# Cargar la imagen
uploaded_file = st.file_uploader("Sube una imagen de resonancia magnética cardíaca", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Mostrar imagen original
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen Original', use_column_width=True)

    with st.spinner("Procesando..."):  # Indicador de progreso
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
        print("Forma de la imagen recortada:", img_array.shape)  # Imprimir forma

        # Conversión a RGB (si es necesario)
        if img_array.ndim == 2:  # Si la imagen es en escala de grises
            img_array = np.stack((img_array,)*3, axis=-1)  

        # Redimensionar
        img_array = cv2.resize(img_array, (224, 224))

        # Normalización y adición de dimensiones
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)  
        print("Forma de la imagen para clasificación:", img_array.shape)  # Imprimir forma

        # Clasificación
        clasificacion_output = modelo_clasificacion.predict(img_array)

    # Mostrar resultado de la clasificación
    clase_predicha = np.argmax(clasificacion_output)  # Asumiendo que la salida es un array de probabilidades
    if clase_predicha == 0:
        resultado = "Diástole"
    else:
        resultado = "Sístole"
    st.write(f"Clasificación: {resultado}")


    resultado = "Diástole" if clase_predicha == 0 else "Sístole"

    if resultado == "Diástole":
        st.markdown(f'<h1 style="color:blue; text-align:center;">Clasificación: {resultado}</h1>', unsafe_allow_html=True)
    else:
        st.markdown(f'<h1 style="color:red; text-align:center;">Clasificación: {resultado}</h1>', unsafe_allow_html=True)
