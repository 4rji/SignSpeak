import cv2
import os
import numpy as np
import importlib.util
import sys
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))


# Configurar el path para los imports
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Debug: Imprimir información de paths
print("Directorio de la aplicación:", APP_DIR)
print("Python path:", sys.path)

try:
    from src.hand_tracker_nms import HandTrackerNMS
    import src.extra
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Contenido del directorio src:", os.listdir(os.path.join(APP_DIR, 'src')))
    raise

# Verificar e importar joblib dinámicamente
spec = importlib.util.find_spec("joblib")
if spec is None:
    print("Error: No se encontró el módulo 'joblib'. Intenta instalarlo con 'pip install joblib'.")
    exit(1)
else:
    joblib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(joblib)

# Definir constantes
WINDOW = "Hand Tracking"
BASE_PATH = "models"
PALM_MODEL_PATH = os.path.join(BASE_PATH, "palm_detection_without_custom_op.tflite")
LANDMARK_MODEL_PATH = os.path.join(BASE_PATH, "hand_landmark.tflite")
ANCHORS_PATH = os.path.join(BASE_PATH, "anchors.csv")
GESTURE_MODEL_PATH = os.path.join(BASE_PATH, "gesture_clf.pkl")

# Cargar modelos
connections = src.extra.connections
int_to_char = src.extra.classes

detector = HandTrackerNMS(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)

try:
    gesture_clf = joblib.load(GESTURE_MODEL_PATH)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo {GESTURE_MODEL_PATH}")
    exit(1)

# Verificar que existen todos los archivos necesarios
required_files = [
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    GESTURE_MODEL_PATH
]

for file_path in required_files:
    if not os.path.exists(file_path):
        print(f"Error: No se encontró el archivo {file_path}")
        print("Ejecute los siguientes comandos para crear los archivos necesarios:")
        print("python src/create_models.py")
        print("python src/create_gesture_model.py")
        exit(1)

cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture(0)

word = []
letter = ""
staticGesture = 0

while True:
    hasFrame, frame = capture.read()
    if not hasFrame:
        print("Error: No se pudo capturar el frame de la cámara.")
        break
    
    try:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        points, bboxes, joints = detector(image)
        
        if points is not None:
            src.extra.draw_points(points, frame)
            pred_sign = src.extra.predict_sign(joints, gesture_clf, int_to_char)
            
            if letter == pred_sign:
                staticGesture += 1
            else:
                letter = pred_sign
                staticGesture = 0
            
            if staticGesture > 6:
                word.append(letter)
                staticGesture = 0
        else:
            if word and word[-1] != " ":
                staticGesture += 1
                if staticGesture > 6:
                    word.append(" ")
                    staticGesture = 0
        
        src.extra.draw_sign(word, frame, (50, 460))
        
    except Exception as e:
        print(f"Error en procesamiento del frame: {e}")
        
    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(1)
    
    if key == 27:  # ESC para salir
        break
    if key == 8:  # BACKSPACE para borrar
        if word:
            word.pop()

capture.release()
cv2.destroyAllWindows()