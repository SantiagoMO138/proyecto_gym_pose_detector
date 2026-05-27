"""
Ventana principal de GymPose AI.
Layout: sidebar izquierdo | video central | sidebar derecho.
Pipeline de procesamiento ejecutado vía QTimer (~30fps).
"""

import cv2
import threading
from collections import Counter
from typing import Optional

from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout
from PySide6.QtCore import Qt, QTimer

from video_handler import VideoHandler
from pose_detector import PoseDetector
from biomechanics import BicepCurlTracker, LateralRaiseTracker
from exercise_classifier import ExerciseClassifier
from person_detector import PersonDetector
from emotion_detector import EmotionDetector
from gui.video_widget import VideoWidget
from gui.left_sidebar import LeftSidebar
from gui.right_sidebar import RightSidebar


def _run_emotion(frame, bbox, state, lock):
    """Función de hilo para ejecutar DeepFace sin bloquear el loop principal."""
    detector = EmotionDetector()
    result = detector.analyze(frame, bbox)
    with lock:
        state["result"] = result
        state["ready"] = True


class MainWindow(QMainWindow):
    """Ventana principal con layout de 3 paneles y procesamiento en tiempo real."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GymPose AI")
        self.setFixedSize(1280, 720)

        # ── Widgets de UI ──
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        self.left_sidebar = LeftSidebar()
        self.video_widget = VideoWidget()
        self.right_sidebar = RightSidebar()

        layout.addWidget(self.left_sidebar)
        layout.addWidget(self.video_widget, 1)  # stretch
        layout.addWidget(self.right_sidebar)

        # ── Componentes de procesamiento ──
        self.video_handler = VideoHandler(0)
        self.pose_detector = PoseDetector()
        self.classifier = ExerciseClassifier()
        self.person_detector = PersonDetector()

        # Trackers
        self.trackers = [BicepCurlTracker(), LateralRaiseTracker()]
        self.current_tracker_index = 0
        self.current_tracker = self.trackers[self.current_tracker_index]

        # ── Estado ──
        self.frame_count = 0
        self.prediction_window: list[Optional[str]] = []
        self.last_detected_label = "-"
        self.last_confidence = 0.0
        self.last_detected_exercise = None
        self.last_emotion_data = None
        self.emotion_lock = threading.Lock()
        self.emotion_state = {"result": None, "ready": False}
        self.person_lost = False

        # ── Constantes ──
        self.CLASSIFY_INTERVAL = 30
        self.PREDICTION_WINDOW_SIZE = 5
        self.CONFIDENCE_THRESHOLD = 0.5
        self.EMOTION_INTERVAL = 60
        self.EXERCISE_MAP = {"bicep_curl": 0, "lateral_raise": 1}

        # ── Iniciar cámara ──
        if not self.video_handler.start_capture():
            print("Error: No se pudo abrir la cámara.")
            return

        # ── Conectar señales ──
        self.right_sidebar.target_spin.valueChanged.connect(
            self.left_sidebar.set_target_reps
        )

        # ── Timer de procesamiento (~30fps) ──
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(33)

        print("GymPose AI iniciado. Presiona 'q' o cierra la ventana para salir.")

    def process_frame(self):
        """Loop principal de procesamiento: lee frame, corre pipeline, actualiza UI."""
        success, frame = self.video_handler.read_frame()
        if not success or frame is None:
            return

        self.frame_count += 1
        h, w = frame.shape[:2]

        # ── GATE: YOLOv8n ──
        has_person, person_bbox = self.person_detector.detect(frame)

        # Leer resultado de emoción si el hilo terminó
        with self.emotion_lock:
            if self.emotion_state["ready"]:
                self.last_emotion_data = self.emotion_state["result"]
                self.emotion_state["ready"] = False

        if not has_person:
            # Limpiar reps la primera vez
            if not self.person_lost:
                self.current_tracker.reps = 0
                self.person_lost = True

            # Mostrar frame sin procesar + actualizar sidebars
            self.video_widget.set_frame(frame)
            self.left_sidebar.update_no_person()
            self.right_sidebar.update_sidebar(
                person_detected=False,
                emotion_data=None,
                classifier_label="-",
                classifier_conf=0.0,
                angle=None,
            )
            return

        self.person_lost = False

        # ── A PARTIR DE AQUÍ: FLUJO NORMAL (sin cambios en biomecánica) ──

        # MediaPipe pose
        results = self.pose_detector.process_frame(frame)
        frame_drawn = self.pose_detector.draw_landmarks(frame.copy(), results)

        # Biomecánica
        landmarks = self.pose_detector.get_landmarks(results)
        debug_angle = None
        if landmarks:
            self.current_tracker.update(landmarks, w, h, side='left')
            debug_angle = getattr(self.current_tracker, 'current_angle', None)

        # ── ViT classifier cada N frames ──
        is_classifying = False
        if self.frame_count % self.CLASSIFY_INTERVAL == 0:
            is_classifying = True
            classification = self.classifier.classify(frame)
            self.last_detected_label = classification["label"]
            self.last_confidence = classification["confidence"]
            self.last_detected_exercise = classification["exercise"]

            detected_exercise = classification["exercise"]
            if detected_exercise is not None:
                self.prediction_window.append(detected_exercise)
                if len(self.prediction_window) > self.PREDICTION_WINDOW_SIZE:
                    self.prediction_window.pop(0)

                # Votación por mayoría simple
                if len(self.prediction_window) >= 1:
                    vote_counts = Counter(self.prediction_window)
                    winner, winner_count = vote_counts.most_common(1)[0]

                    if winner_count > len(self.prediction_window) / 2:
                        winner_idx = self.EXERCISE_MAP.get(winner)
                        if winner_idx is not None and winner_idx != self.current_tracker_index:
                            print(
                                f"[Auto] Cambio: {self.current_tracker.name} → "
                                f"{self.trackers[winner_idx].name} (conf: {self.last_confidence:.2%})"
                            )
                            self.current_tracker_index = winner_idx
                            self.current_tracker = self.trackers[self.current_tracker_index]
                    else:
                        if len(vote_counts) > 1:
                            print(
                                f"[Auto] Sin mayoría: {dict(vote_counts)} "
                                f"(actual: {self.current_tracker.name})"
                            )

        # ── DeepFace cada 60 frames en hilo ──
        if self.frame_count % self.EMOTION_INTERVAL == 0:
            with self.emotion_lock:
                if not self.emotion_state["ready"]:
                    t = threading.Thread(
                        target=_run_emotion,
                        args=(frame.copy(), person_bbox, self.emotion_state, self.emotion_lock),
                        daemon=True,
                    )
                    t.start()

        # ── Actualizar UI ──
        self.video_widget.set_frame(frame_drawn)
        self.left_sidebar.update_tracker(self.current_tracker)
        self.right_sidebar.update_sidebar(
            person_detected=True,
            emotion_data=self.last_emotion_data,
            classifier_label=self.last_detected_label,
            classifier_conf=self.last_confidence,
            angle=debug_angle,
        )

    def closeEvent(self, event):
        """Libera recursos al cerrar la ventana."""
        print("Cerrando la aplicación...")
        self.timer.stop()
        self.video_handler.release()
        self.pose_detector.release_resources()
        cv2.destroyAllWindows()
        event.accept()
