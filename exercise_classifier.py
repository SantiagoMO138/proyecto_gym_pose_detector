"""
Clasificador de ejercicios usando ViT (Vision Transformer) de HuggingFace.
Modelo: averrous/workout_model_vit (22 clases, descargado localmente)

Solo 2 clases activan el seguimiento de ejercicio:
    - "barbell biceps curl" / "hammer curl"  ->  bicep_curl
    - "lateral raises"                        ->  lateral_raise

Las restantes 20 clases del modelo actúan como detector implícito de fondo:
si el modelo predice cualquier otra clase (squat, plank, etc.), la predicción
se ignora y no se produce cambio de ejercicio.
"""

import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch


class ExerciseClassifier:
    """
    Wrapper para clasificar ejercicios a partir de frames de video usando ViT.
    Soporta inferencia en GPU (CUDA) si está disponible.
    """

    # Mapeo de etiquetas del modelo a IDs internos de ejercicio.
    # Cualquier clase del modelo que NO esté en este diccionario se considera
    # "fondo" (no ejercicio activo) y no activa el cambio de tracker.
    CLASS_MAP = {
        "barbell biceps curl": "bicep_curl",
        "hammer curl": "bicep_curl",
        "lateral raises": "lateral_raise",
    }

    def __init__(
        self,
        model_path: str = "models/workout_model_vit",
        confidence_threshold: float = 0.5,
    ) -> None:
        """
        Inicializa el clasificador cargando el modelo y procesador desde la ruta local.

        Args:
            model_path: Ruta local al modelo descargado (config + safetensors + preprocessor).
            confidence_threshold: Confianza mínima para considerar válida una predicción.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold

        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageClassification.from_pretrained(model_path).to(
            self.device
        )
        self.model.eval()

        # Configuración para half-precision en GPU (ahorro de memoria + velocidad)
        if self.device.type == "cuda":
            self.model = self.model.half()

    def classify(self, frame: np.ndarray) -> dict:
        """
        Clasifica un frame de OpenCV (BGR) y retorna la predicción.

        Args:
            frame: Frame de imagen en formato BGR (OpenCV).

        Returns:
            dict con claves:
                - exercise: str o None (ejercicio interno mapeado, None si no pasa umbral o es fondo).
                - confidence: float (probabilidad de la clase top-1).
                - label: str (etiqueta textual original del modelo).
        """
        # BGR → RGB → PIL.Image
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        inputs = self.processor(images=pil_image, return_tensors="pt")

        # Mover tensores al dispositivo y convertir a half-precision en GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.device.type == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].half()

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        confidence, pred_idx = probs.max(dim=-1)
        confidence = confidence.item()
        pred_idx = pred_idx.item()

        label = self.model.config.id2label[pred_idx]

        exercise = None
        if confidence >= self.confidence_threshold:
            exercise = self.CLASS_MAP.get(label.lower().strip())

        return {
            "exercise": exercise,
            "confidence": confidence,
            "label": label,
        }
