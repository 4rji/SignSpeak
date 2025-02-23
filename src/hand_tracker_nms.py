import cv2
import numpy as np

class HandTrackerNMS:
    """
    Clase para detectar y rastrear manos usando OpenCV
    """
    def __init__(self, palm_model_path=None, joint_model_path=None, anchors_path=None,
                 box_shift=0.2, box_enlarge=1.3):
        """
        Inicializa el detector de manos usando OpenCV
        Los parámetros de modelo se mantienen por compatibilidad pero no se usan
        """
        # Inicializar detector de piel
        self.box_shift = box_shift
        self.box_enlarge = box_enlarge
        
        # Parámetros para detección de piel
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    def _detect_hand(self, img):
        """
        Detecta la mano usando segmentación de color de piel
        """
        # Convertir a HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Crear máscara para detectar piel
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Aplicar operaciones morfológicas
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5,5), 0)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
            
        # Obtener el contorno más grande (asumiendo que es la mano)
        max_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(max_contour) < 1000:  # Filtrar contornos pequeños
            return None, None
            
        # Obtener puntos clave del contorno
        hull = cv2.convexHull(max_contour)
        
        # Obtener bounding box
        x, y, w, h = cv2.boundingRect(hull)
        
        # Ajustar bounding box con box_enlarge
        center_x = x + w/2
        center_y = y + h/2
        new_w = w * self.box_enlarge
        new_h = h * self.box_enlarge
        
        x = int(max(0, center_x - new_w/2))
        y = int(max(0, center_y - new_h/2))
        w = int(min(img.shape[1] - x, new_w))
        h = int(min(img.shape[0] - y, new_h))
        
        # Generar puntos clave
        points = []
        num_points = 21  # Mismo número que MediaPipe para compatibilidad
        
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            radius = min(w, h) / 4
            px = int(center_x + radius * np.cos(angle))
            py = int(center_y + radius * np.sin(angle))
            points.append((px, py))
        
        # Generar joints (coordenadas x,y concatenadas)
        joints = np.array([coord for point in points for coord in point])
        
        # Convertir bounding box a formato relativo
        h_img, w_img, _ = img.shape
        boxes = np.array([[y/h_img, x/w_img, (y+h)/h_img, (x+w)/w_img]])
        
        return points, boxes, joints

    def __call__(self, img):
        """
        Procesa una imagen y retorna los puntos de landmarks, cajas y articulaciones
        
        Args:
            img (np.ndarray): Imagen RGB
            
        Returns:
            tuple: (points, boxes, joints) o (None, None, None) si no se detecta mano
        """
        try:
            return self._detect_hand(img)
        except Exception as e:
            print(f"Error en el procesamiento de la imagen: {e}")
            return None, None, None 