import cv2
import numpy as np

# Definir las conexiones entre puntos para dibujar la mano
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),  # Índice
    (0, 9), (9, 10), (10, 11), (11, 12),  # Medio
    (0, 13), (13, 14), (14, 15), (15, 16),  # Anular
    (0, 17), (17, 18), (18, 19), (19, 20)  # Meñique
]

# Definir las clases de gestos (letras del alfabeto)
classes = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z'
}

def draw_points(points, frame):
    """
    Dibuja los puntos y conexiones de la mano en el frame
    """
    # Dibujar puntos
    for point in points:
        cv2.circle(frame, point, 4, (0, 255, 0), -1)
    
    # Dibujar conexiones
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], (255, 0, 0), 2)

def predict_sign(joints, classifier, int_to_char):
    """
    Predice el gesto (letra) basado en los joints de la mano
    """
    try:
        # Normalizar joints
        joints = np.array(joints).reshape(1, -1)
        joints = (joints - np.mean(joints)) / np.std(joints)
        
        # Predecir
        prediction = classifier.predict(joints)[0]
        return int_to_char.get(prediction, '?')
    except Exception as e:
        print(f"Error en predicción: {e}")
        return '?'

def draw_sign(word, frame, position, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    Dibuja la palabra formada por los gestos en el frame
    """
    text = ''.join(word)
    cv2.putText(frame, text, position, font, 1, (255, 255, 255), 2) 