"""
Módulo para la detección de puntos clave del cuerpo (landmarks) utilizando MediaPipe Tasks API.
"""

import cv2
import os
os.environ['GLOG_minloglevel'] = '2'
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Any, Dict, Optional, Tuple, List
import time

class PoseDetector:
    """
    Clase para inicializar y ejecutar el modelo de estimación de postura de MediaPipe.
    Mantiene la instancia del modelo para evitar fugas de memoria y utiliza la nueva API (Tasks).
    """
    
    def __init__(self, static_image_mode: bool = False, model_complexity: int = 1,
                 smooth_landmarks: bool = True, min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7) -> None:
        """
        Inicializa el detector de postura, instanciando el modelo de MediaPipe Tasks API.
        
        Args:
            static_image_mode: Si es True, trata cada imagen de entrada como independiente.
            model_complexity: Complejidad del modelo de postura (0, 1, 2).
            smooth_landmarks: Parámetro legacy, manejado internamente por el modelo en Tasks API.
            min_detection_confidence: Confianza mínima para la detección.
            min_tracking_confidence: Confianza mínima para el seguimiento.
        """
        self.drawing_utils = vision.drawing_utils
        self.pose_connections = vision.PoseLandmarksConnections.POSE_LANDMARKS
        
        # Determinar el nombre del modelo según la complejidad
        if model_complexity == 0:
            model_name = "pose_landmarker_lite.task"
        elif model_complexity == 2:
            model_name = "pose_landmarker_heavy.task"
        else:
            model_name = "pose_landmarker_full.task"
            
        # Descargar el modelo si no existe localmente (La nueva API requiere el archivo .task explícitamente)
        if not os.path.exists(model_name):
            print(f"Descargando el modelo de MediaPipe ({model_name})...")
            url = f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/{model_name.replace('.task','')}/float16/1/{model_name}"
            urllib.request.urlretrieve(url, model_name)
            print("Descarga completada.")
            
        # Configurar las opciones de MediaPipe Tasks
        base_options = python.BaseOptions(model_asset_path=model_name)
        running_mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.static_image_mode = static_image_mode
        self.start_time = time.time()
        
    def process_frame(self, frame: np.ndarray) -> Any:
        """
        Procesa un frame para detectar la postura.
        
        Args:
            frame: El frame de imagen en formato BGR.
            
        Returns:
            Los resultados de la detección (PoseLandmarkerResult).
        """
        # Convertir la imagen de BGR (OpenCV) a RGB y luego al formato Image de MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        if self.static_image_mode:
            results = self.detector.detect(mp_image)
        else:
            # En modo video se necesita un timestamp en milisegundos que vaya incrementando
            timestamp_ms = int((time.time() - self.start_time) * 1000)
            results = self.detector.detect_for_video(mp_image, timestamp_ms)
            
        return results

    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Dibuja los landmarks de la postura detectada en el frame original.
        
        Args:
            frame: El frame original en formato BGR.
            results: Los resultados de la detección obtenidos de process_frame.
            
        Returns:
            El frame original modificado con los landmarks dibujados.
        """
        if results and results.pose_landmarks:
            # results.pose_landmarks es una lista de listas de landmarks (una por cada persona detectada)
            for pose_landmarks in results.pose_landmarks:
                self.drawing_utils.draw_landmarks(
                    frame,
                    pose_landmarks,
                    self.pose_connections,
                    self.drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
        return frame

    def get_landmarks(self, results: Any) -> Optional[List[Any]]:
        """
        Extrae la lista de puntos clave (landmarks) de los resultados.
        
        Args:
            results: Resultados devueltos por el procesamiento del modelo.
            
        Returns:
            Lista de landmarks detectados para la primera persona, o None si no hay detección.
        """
        if results and results.pose_landmarks and len(results.pose_landmarks) > 0:
            return results.pose_landmarks[0]
        return None
        
    def release_resources(self) -> None:
        """
        Libera los recursos asociados al modelo de MediaPipe.
        """
        self.detector.close()
