import cv2
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# --- Configuración ---
IMAGE_SIZE = (128, 128)
DATA_DIR = 'Training_set'
MODEL_OUT = "modelo_final_svm.pkl"

def load_vgg16_extractor():
    """Carga VGG16 para extracción de características sin las capas densas."""
    print("🚀 Cargando VGG16 (ImageNet)...")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    return Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(data_dir, cnn_model):
    """Procesa imágenes y extrae vectores de características."""
    features, labels = [], []
    class_names = sorted(os.listdir(data_dir))
    
    print(f"📦 Clases detectadas: {class_names}")
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path): continue
        
        print(f"🖼️ Procesando {class_name}...")
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is None: continue
            
            image = cv2.resize(image, IMAGE_SIZE)
            image = img_to_array(image) / 255.0
            image = np.expand_dims(image, axis=0)
            
            # Predicción de la CNN (Extracción)
            feat = cnn_model.predict(image, verbose=0).flatten()
            features.append(feat)
            labels.append(label)
            
    return np.array(features), np.array(labels), class_names

def train_system():
    # 1. Extracción
    cnn = load_vgg16_extractor()
    X, y, classes = extract_features(DATA_DIR, cnn)
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 3. Escalado (Crucial para SVM)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 4. Entrenamiento SVM con GridSearch
    print("🧠 Entrenando SVM con GridSearchCV...")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf']
    }
    grid = GridSearchCV(svm.SVC(probability=True), param_grid, cv=5)
    grid.fit(X_train, y_train)
    
    best_svm = grid.best_estimator_
    
    # 5. Evaluación
    y_pred = best_svm.predict(X_test)
    print("\n✅ ENTRENAMIENTO COMPLETADO")
    print(f"Precisión Final: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # 6. Guardar solo lo necesario
    # NO guardamos la CNN (pesa 500MB), solo SVM y Scaler (<1MB)
    joblib.dump({
        'svm': best_svm,
        'scaler': scaler,
        'classes': classes
    }, MODEL_OUT)
    
    print(f"\n💾 Modelo ligero guardado en: {MODEL_OUT}")

if __name__ == "__main__":
    train_system()
