import numpy as np
import os

def create_model_files():
    """
    Crea archivos de modelo placeholder
    """
    # Crear directorio models si no existe
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Crear palm_detection_without_custom_op.tflite (placeholder)
    with open('models/palm_detection_without_custom_op.tflite', 'wb') as f:
        f.write(b'placeholder')
    
    # Crear hand_landmark.tflite (placeholder)
    with open('models/hand_landmark.tflite', 'wb') as f:
        f.write(b'placeholder')
    
    # Crear anchors.csv
    anchors = np.random.rand(2944, 4)  # Tamaño típico de anclas
    np.savetxt('models/anchors.csv', anchors, delimiter=',')
    
    print("Archivos de modelo creados en el directorio 'models/'")

if __name__ == "__main__":
    create_model_files() 