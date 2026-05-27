"""
Punto de entrada principal para la aplicación GymPose AI.
Orquesta la captura de video, la detección de postura y el análisis biomecánico.
Incluye:
  - Clasificación automática de ejercicios con ViT (Vision Transformer)
  - Detección de personas con YOLOv8n (gate: sin persona → skip todo)
  - Detección de emociones con DeepFace (cada 60 frames, hilo separado)
  - HUD moderno y limpio (hud_renderer.py)
"""

import cv2
import threading
from collections import Counter
from typing import Optional
from video_handler import VideoHandler
from pose_detector import PoseDetector
from biomechanics import BicepCurlTracker, LateralRaiseTracker
from exercise_classifier import ExerciseClassifier
from person_detector import PersonDetector
from emotion_detector import EmotionDetector
from hud_renderer import HUDRenderer


def _run_emotion(frame, bbox, state, lock):
    """Función de hilo para ejecutar DeepFace sin bloquear el loop principal."""
    detector = EmotionDetector()
    result = detector.analyze(frame, bbox)
    with lock:
        state["result"] = result
        state["ready"] = True


def main() -> None:
    """
    Función principal que inicializa los componentes y ejecuta el bucle de la aplicación.
    """
    video_handler = VideoHandler(0)  # Utiliza la cámara predeterminada
    pose_detector = PoseDetector()

    # Clasificador automático de ejercicios (ViT)
    print("Cargando clasificador de ejercicios...")
    classifier = ExerciseClassifier()
    print(f"Clasificador listo. Dispositivo: {classifier.device}")

    # Detector de personas (YOLOv8n)
    print("Cargando detector de personas (YOLOv8n)...")
    person_detector = PersonDetector()
    print("YOLOv8n listo.")

    # Renderizador de HUD moderno
    hud_renderer = HUDRenderer(target_reps=12)

    # Selector de Ejercicios (solo 2: bicep curl y lateral raise)
    trackers = [BicepCurlTracker(), LateralRaiseTracker()]
    current_tracker_index = 0
    current_tracker = trackers[current_tracker_index]

    # Mapeo de ejercicios clasificados a índices de tracker
    EXERCISE_MAP = {
        "bicep_curl": 0,
        "lateral_raise": 1,
    }

    # Configuración de Ventana a Pantalla Completa
    window_name = "GymPose AI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Iniciar la captura de video
    if not video_handler.start_capture():
        print("Error: No se pudo abrir la fuente de video (cámara).")
        return

    print("Iniciando GymPose AI. Presiona 'q' para salir.")

    # --- Lógica de clasificación automática ---
    frame_count = 0
    CLASSIFY_INTERVAL = 30      # Cada 30 frames (~1s a 30fps)
    PREDICTION_WINDOW_SIZE = 5
    CONFIDENCE_THRESHOLD = 0.5
    EMOTION_INTERVAL = 60       # Cada 60 frames (~2s)

    prediction_window: list[Optional[str]] = []
    last_detected_label = "-"
    last_confidence = 0.0
    last_detected_exercise = None

    # --- Estado de emoción (hilo) ---
    last_emotion_data = None
    emotion_lock = threading.Lock()
    emotion_state = {"result": None, "ready": False}

    # --- Estado de persona ---
    person_lost = False

    # Bucle principal de procesamiento de video
    while True:
        success, frame = video_handler.read_frame()
        if not success or frame is None:
            print("Error o fin de captura de video.")
            break

        frame_count += 1
        h, w, _ = frame.shape

        # ── GATE: YOLOv8n person detection ──
        has_person, person_bbox = person_detector.detect(frame)

        # Leer resultado de emoción si el hilo terminó
        with emotion_lock:
            if emotion_state["ready"]:
                last_emotion_data = emotion_state["result"]
                emotion_state["ready"] = False

        if not has_person:
            # Limpiar contador la primera vez que se pierde la persona
            if not person_lost:
                current_tracker.reps = 0
                person_lost = True

            # Overlay de "persona no detectada"
            overlay = frame.copy()
            bar_h = 70
            cv2.rectangle(overlay, (0, 0), (w, bar_h), (50, 50, 50), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(
                frame,
                "Persona no detectada",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Mostrar frame y manejar teclado
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Cerrando la aplicación...")
                break
            continue

        person_lost = False

        # ── A PARTIR DE AQUÍ: FLUJO NORMAL (sin cambios en biomecánica) ──

        # Detección de postura
        results = pose_detector.process_frame(frame)

        # Dibujo de landmarks en el frame
        frame_drawn = pose_detector.draw_landmarks(frame.copy(), results)

        # Procesamiento Biomecánico
        landmarks = pose_detector.get_landmarks(results)

        if landmarks:
            current_tracker.update(landmarks, w, h, side='left')

            # Debug Mode Angle
            debug_angle = getattr(current_tracker, 'current_angle', 0.0)
            if debug_angle is not None:
                cv2.putText(
                    frame_drawn,
                    f"Angulo: {debug_angle:.1f}°",
                    (15, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        # --- Clasificación automática cada N frames ---
        is_classifying = False
        if frame_count % CLASSIFY_INTERVAL == 0:
            is_classifying = True
            classification = classifier.classify(frame)
            last_detected_label = classification["label"]
            last_confidence = classification["confidence"]
            last_detected_exercise = classification["exercise"]

            detected_exercise = classification["exercise"]
            if detected_exercise is not None:
                prediction_window.append(detected_exercise)
                if len(prediction_window) > PREDICTION_WINDOW_SIZE:
                    prediction_window.pop(0)

                # Votación por mayoría simple sobre el tamaño real de la ventana
                if len(prediction_window) >= 1:
                    vote_counts = Counter(prediction_window)
                    winner, winner_count = vote_counts.most_common(1)[0]

                    if winner_count > len(prediction_window) / 2:
                        winner_idx = EXERCISE_MAP.get(winner)
                        if winner_idx is not None and winner_idx != current_tracker_index:
                            print(
                                f"[Auto] Cambio de ejercicio: {current_tracker.name} → {trackers[winner_idx].name} "
                                f"(confianza: {last_confidence:.2%})"
                            )
                            current_tracker_index = winner_idx
                            current_tracker = trackers[current_tracker_index]
                    else:
                        # Debug: sin mayoría aún
                        if len(vote_counts) > 1:
                            print(
                                f"[Auto] Sin mayoría aún: {dict(vote_counts)} "
                                f"(actual: {current_tracker.name})"
                            )

        # --- DeepFace emoción cada 60 frames en hilo separado ---
        if frame_count % EMOTION_INTERVAL == 0:
            with emotion_lock:
                if not emotion_state["ready"]:
                    t = threading.Thread(
                        target=_run_emotion,
                        args=(frame.copy(), person_bbox, emotion_state, emotion_lock),
                        daemon=True,
                    )
                    t.start()

        # ── HUD moderno ──
        hud_renderer.render(
            frame_drawn,
            current_tracker,
            last_detected_label,
            last_confidence,
            last_emotion_data,
            person_detected=True,
            is_classifying=is_classifying,
        )

        # Mostrar el resultado
        cv2.imshow(window_name, frame_drawn)

        # Manejo de Teclado (solo 'q' para salir)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Cerrando la aplicación...")
            break

    # Liberar recursos
    video_handler.release()
    pose_detector.release_resources()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
