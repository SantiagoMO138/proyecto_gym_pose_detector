"""
Detector de emociones usando DeepFace.
Analiza emociones faciales en un frame (o región de interés).
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2


class EmotionDetector:
    """
    Wrapper sobre DeepFace para análisis de emociones faciales.
    Corre en un hilo separado para no bloquear el loop principal.
    """

    def __init__(self) -> None:
        """
        Inicializa el detector de emociones.
        No carga modelo aquí; DeepFace lo hace lazy en el primer analyze().
        """
        self._ready = False

    def analyze(
        self,
        frame: np.ndarray,
        person_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Analiza el frame y retorna la emoción dominante + todas las emociones.

        Args:
            frame: Imagen BGR de OpenCV.
            person_bbox: Bounding box de la persona (x1, y1, x2, y2).
                         Si se provee, recorta la región para acelerar la búsqueda del rostro.

        Returns:
            dict con claves:
                - dominant_emotion: str o None
                - confidence: float (probabilidad de la emoción dominante)
                - emotions: dict {emotion: probability, ...} las 7 emociones
                - face_detected: bool
        """
        try:
            from deepface import DeepFace

            # Recortar región de la persona para acelerar
            if person_bbox is not None:
                x1, y1, x2, y2 = person_bbox
                # Asegurar que los límites estén dentro del frame
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    frame = frame[y1:y2, x1:x2]

            result = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv",
                silent=True,
            )

            # DeepFace retorna lista de dicts (una por cada rostro detectado)
            if isinstance(result, list) and len(result) > 0:
                face_data = result[0]
            elif isinstance(result, dict):
                face_data = result
            else:
                return self._empty_result()

            emotions = face_data.get("emotion", {})
            dominant = face_data.get("dominant_emotion")
            face_detected = face_data.get("face_confidence", 0) > 0.5

            if not emotions or dominant is None:
                return self._empty_result()

            return {
                "dominant_emotion": dominant,
                "confidence": emotions.get(dominant, 0.0),
                "emotions": emotions,
                "face_detected": face_detected,
            }

        except Exception:
            return self._empty_result()

    def _empty_result(self) -> Dict[str, Any]:
        """Retorna un resultado vacío."""
        return {
            "dominant_emotion": None,
            "confidence": 0.0,
            "emotions": {},
            "face_detected": False,
        }
