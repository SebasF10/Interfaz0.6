import os
import cv2
import imutils
import mediapipe as mp
from dataclasses import dataclass
from typing import Tuple, Optional
import logging
from pathlib import Path

# Configuración de logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    
@dataclass
class FaceDetectionConfig:
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_size: Tuple[int, int] = (30, 30)
    padding_factor: float = 0.1
    max_faces: int = 200
    output_size: Tuple[int, int] = (720, 720)

class FaceDetector:
    def __init__(self, camera_config: CameraConfig, detection_config: FaceDetectionConfig):
        self.camera_config = camera_config
        self.detection_config = detection_config
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def initialize_camera(self) -> Optional[cv2.VideoCapture]:
        """Inicializa y configura la cámara."""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            logging.error("No se pudo abrir la cámara")
            return None
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_config.height)
        cap.set(cv2.CAP_PROP_FPS, self.camera_config.fps)
        
        return cap
        
    def process_frame(self, frame: cv2.Mat) -> Tuple[cv2.Mat, cv2.Mat]:
        """Preprocesa el frame para la detección facial."""
        frame = imutils.resize(frame, width=320)
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        return frame, gray
        
    def detect_faces(self, gray: cv2.Mat) -> list:
        """Detecta rostros en la imagen en escala de grises."""
        return self.face_classifier.detectMultiScale(
            gray,
            scaleFactor=self.detection_config.scale_factor,
            minNeighbors=self.detection_config.min_neighbors,
            minSize=self.detection_config.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
    def extract_face_roi(self, frame: cv2.Mat, face: tuple) -> Tuple[cv2.Mat, tuple]:
        """Extrae la región de interés (ROI) del rostro con padding."""
        x, y, w, h = face
        padding = int(self.detection_config.padding_factor * w)
        
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
        
    def save_face(self, face_roi: cv2.Mat, output_path: Path, count: int) -> bool:
        """Guarda el rostro detectado en el disco."""
        if face_roi.size == 0:
            return False
            
        face_roi = cv2.resize(face_roi, 
                            self.detection_config.output_size, 
                            interpolation=cv2.INTER_LANCZOS4)
        
        filename = output_path / f'rostro_{count}.jpg'
        return cv2.imwrite(str(filename), face_roi)

def ejecutar(nombre_carpeta: str):
    """Función principal para ejecutar la detección facial."""
    # Configuración de rutas
    ruta_data = Path("C:/Users/USER/Desktop/Interfaz-0.5-main/Eye_System/formularios/Data")
    persona_data = ruta_data / nombre_carpeta
    persona_data.mkdir(parents=True, exist_ok=True)
    logging.info(f"Directorio de salida: {persona_data}")

    # Inicialización de configuraciones
    camera_config = CameraConfig()
    detection_config = FaceDetectionConfig()
    detector = FaceDetector(camera_config, detection_config)
    
    # Inicialización de la cámara
    cap = detector.initialize_camera()
    if cap is None:
        return
    
    count = 0
    fps_update_counter = 0
    current_fps = 0
    
    # Inicialización de FaceMesh
    with detector.mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        refine_landmarks=True) as face_mesh:
        
        while count < detection_config.max_faces:
            ret, frame = cap.read()
            if not ret:
                logging.error("Error al leer frame")
                break
                
            frame, gray = detector.process_frame(frame)
            faces = detector.detect_faces(gray)
            
            # Actualización de FPS
            fps_update_counter = (fps_update_counter + 1) % 30
            if fps_update_counter == 0:
                current_fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Visualización de FPS
            cv2.putText(frame, f"FPS:{current_fps}", (5, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            if len(faces) > 0:
                for face in faces:
                    face_roi, (x1, y1, x2, y2) = detector.extract_face_roi(frame.copy(), face)
                    
                    if detector.save_face(face_roi, persona_data, count):
                        count += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Procesamiento de malla facial
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        detector.mp_drawing.draw_landmarks(
                            frame,
                            face_landmarks,
                            detector.mp_face_mesh.FACEMESH_TESSELATION,
                            detector.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                            detector.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1))
            
            cv2.imshow("Captura de Rostros con Malla Facial", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    
    cap.release()
    cv2.destroyAllWindows()
    logging.info(f"Proceso completado. Se capturaron {count} imágenes.")