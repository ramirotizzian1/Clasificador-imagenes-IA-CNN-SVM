# 🐀 Clasificador de Especies de Roedores (CNN + SVM)

Este proyecto utiliza Visión Artificial y Machine Learning para la clasificación taxonómica de roedores (*Akodon azarae* y *Calomys musculinus*) mediante el análisis de imágenes de sus mandíbulas.

## 🚀 Arquitectura Híbrida
El sistema utiliza un enfoque híbrido para maximizar la precisión con conjuntos de datos específicos:
1.  **Extractor de Características (CNN):** Se utiliza el modelo **VGG16** (preentrenado en ImageNet) para extraer patrones morfológicos complejos de las imágenes.
2.  **Clasificador (SVM):** Las características extraídas se procesan mediante una **Máquina de Soporte Vectorial (SVM)** optimizada con `GridSearchCV`, logrando una frontera de decisión robusta.

## 📈 Resultados
- **Precisión en Entrenamiento:** 100.00%
- **Precisión en Validación:** 98.85%
- El modelo demuestra una excelente capacidad de generalización al trabajar con Transfer Learning sobre imágenes biológicas.

## 🛠️ Tecnologías Utilizadas
- **Lenguaje:** Python
- **Deep Learning:** TensorFlow / Keras (VGG16)
- **Machine Learning:** Scikit-Learn (SVM, StandardScaler)
- **Visión por Computadora:** OpenCV
- **Interfaz Web:** Streamlit
- **Serialización:** Joblib

## 📁 Estructura del Proyecto
```text
├── Training_set/       # Dataset organizado por clases (Aa / Mc)
├── Imagenes_prueba/    # Imágenes para testeo rápido
├── app.py              # Aplicación Web (Frontend)
├── requirements.txt    # Dependencias necesarias
└── README.md           # Documentación
```

## ⚙️ Cómo Ejecutarlo

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Correr la App Web:**
   ```bash
   streamlit run app.py
   ```

---
**Autor:** Ramiro Tizzian  
*Trabajo Final - Técnicas de Inteligencia Artificial*
