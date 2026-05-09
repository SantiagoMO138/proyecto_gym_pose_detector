"""
Módulo para manejar la captura y procesamiento inicial de video usando OpenCV.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Any

class VideoHandler:
    """
    Clase responsable de gestionar la entrada de video, ya sea desde una cámara web
    o un archivo de video.
    """
    
    def __init__(self, source: int | str = 0) -> None:
        """
        Inicializa el manejador de video.
        
        Args:
            source: Índice de la cámara web (int) o ruta al archivo de video (str).
        """
        self.source = source
        self.capture: Optional[cv2.VideoCapture] = None
        
    def start_capture(self) -> bool:
        """
        Inicia la captura de video usando la fuente especificada.
        
        Returns:
            True si la captura se inició correctamente, False en caso contrario.
        """
        self.capture = cv2.VideoCapture(self.source)
        return self.capture.isOpened()
        
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lee el siguiente frame del video.
        
        Returns:
            Una tupla que contiene un booleano indicando el éxito de la lectura y el frame (o None).
        """
        if self.capture is not None and self.capture.isOpened():
            return self.capture.read()
        return False, None
        
    def release(self) -> None:
        """
        Detiene la captura de video y libera los recursos.
        """
        if self.capture is not None:
            self.capture.release()
