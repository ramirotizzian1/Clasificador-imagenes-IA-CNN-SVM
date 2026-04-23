import streamlit as st
import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

# --- Configuración Visual ---
st.set_page_config(page_title="IA Identificador de Roedores", page_icon="🐀")
st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    .prediction-card {
        background-color: white; padding: 2rem; border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Carga de Recursos (Cache) ---
@st.cache_resource
def load_vgg16_extractor():
    """Instancia la CNN para extracción de características."""
    base = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    return Model(inputs=base.input, outputs=base.output)

@st.cache_resource
def load_svm_components():
    """Carga el SVM y el Scaler desde el archivo ligero."""
    model_path = "modelo_final_svm.pkl"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

# --- Lógica de Negocio ---
def predict(image, cnn_model, components):
    # Preprocesar
    img = cv2.resize(image, (128, 128))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Extraer características (Vector de 8192 elementos para VGG16)
    features = cnn_model.predict(img, verbose=0).flatten()
    
    # Escalar y Clasificar con SVM
    features_scaled = components['scaler'].transform([features])
    prediction = components['svm'].predict(features_scaled)[0]
    
    # Obtener probabilidad si el modelo lo permite
    probs = components['svm'].predict_proba(features_scaled)[0]
    confidence = np.max(probs) * 100
    
    return components['classes'][prediction], confidence

# --- Interfaz de Usuario ---
st.title("🔬 Clasificador Taxonómico de Roedores")
st.write("Sube una fotografía de la mandíbula para identificar la especie mediante IA híbrida.")

cnn = load_vgg16_extractor()
components = load_svm_components()

if components is None:
    st.warning("⚠️ El modelo no ha sido entrenado. Ejecuta `python train.py` primero.")
else:
    file = st.file_uploader("Arrastra tu imagen aquí...", type=["jpg", "png", "jpeg"])
    
    if file:
        col1, col2 = st.columns(2)
        image = Image.open(file)
        
        with col1:
            st.image(image, caption="Imagen Original", use_container_width=True)
            
        with col2:
            if st.button("🚀 Identificar Especie"):
                with st.spinner("Analizando micro-características..."):
                    img_np = np.array(image.convert('RGB'))
                    label, conf = predict(img_np, cnn, components)
                    
                    st.markdown(f"""
                        <div class="prediction-card">
                            <h3 style="color: #4CAF50;">Resultado</h3>
                            <h1 style="margin:0;">{label}</h1>
                            <p style="color: #666;">Confianza: {conf:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if label == "mandibulasAa":
                        st.info("**Akodon azarae:** Especie común en pastizales.")
                    else:
                        st.info("**Calomys musculinus:** Especie clave en estudios de virología.")

st.sidebar.info("""
**Arquitectura Híbrida**
- Extractor: VGG16 (CNN)
- Clasificador: SVM (RBF Kernel)
- Entrenamiento: GridSearchCV
""")
