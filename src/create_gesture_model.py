import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def create_simple_gesture_model():
    """
    Crea un modelo simple de clasificación de gestos
    """
    # Crear directorio models si no existe
    if not os.path.exists('models'):
        os.makedirs('models')

    # Generar datos de ejemplo (esto es solo un placeholder)
    n_samples = 1000
    n_features = 42  # 21 puntos x 2 coordenadas
    
    # Generar datos aleatorios para 26 letras (A-Z)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 26, n_samples)
    
    # Entrenar un clasificador simple
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    # Guardar el modelo
    model_path = os.path.join('models', 'gesture_clf.pkl')
    joblib.dump(clf, model_path)
    print(f"Modelo guardado en {model_path}")

if __name__ == "__main__":
    create_simple_gesture_model() 