"""
Detector de personas usando YOLOv8n (Ultralytics).
Solo detecta la clase 'person' (id 0) y devuelve el bbox más grande.
"""

from typing import Tuple, Optional
import numpy as np


class PersonDetector:
    """
    Wrapper ligero sobre YOLOv8n para detectar personas en un frame.
    """

    def __init__(self, model: str = "yolov8n.pt", confidence: float = 0.5) -> None:
        """
        Inicializa el detector de personas.

        Args:
            model: Nombre o ruta al modelo YOLO. "yolov8n.pt" se descarga
                   automáticamente si no existe (~6 MB).
            confidence: Umbral mínimo de confianza para considerar una detección.
        """
        from ultralytics import YOLO

        self.model = YOLO(model, verbose=False)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """
        Detecta personas en el frame y retorna el bbox más grande.

        Args:
            frame: Imagen BGR de OpenCV.

        Returns:
            (persona_detectada, bbox_mas_grande)
            bbox = (x1, y1, x2, y2) en píxeles, o None si no hay persona.
        """
        results = self.model(frame, verbose=False)

        best_bbox = None
        best_area = 0.0

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                if cls_id != 0 or conf < self.confidence:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_bbox = (x1, y1, x2, y2)

        return best_bbox is not None, best_bbox

    def draw_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Dibuja el bounding box de la persona en el frame.

        Args:
            frame: Frame BGR.
            bbox: (x1, y1, x2, y2).

        Returns:
            Frame con el bbox dibujado.
        """
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame
