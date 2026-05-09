"""
Punto de entrada principal para la aplicación GymPose AI.
Orquesta la captura de video, la detección de postura y el análisis biomecánico.
"""

import cv2
import math
import textwrap
from typing import Optional
from video_handler import VideoHandler
from pose_detector import PoseDetector
from biomechanics import BiomechanicsAnalyzer, BicepCurlTracker, LateralRaiseTracker, SideBicepCurlTracker

def draw_multiline_text(frame, text, x, y, font_scale, color, thickness, max_width=40):
    """
    Dibuja texto en múltiples líneas si excede el max_width de caracteres.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    # textwrap corta el texto respetando las palabras completas
    wrapped_text = textwrap.wrap(text, width=max_width)

    y_offset = y
    for line in wrapped_text:
        cv2.putText(frame, line, (x, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
        # Incrementa el offset Y para la siguiente línea (ajusta el 35 según el tamaño de fuente)
        y_offset += int(35 * font_scale) 

def main() -> None:
    """
    Función principal que inicializa los componentes y ejecuta el bucle de la aplicación.
    """
    video_handler = VideoHandler(0)  # Utiliza la cámara predeterminada
    pose_detector = PoseDetector()
    biomechanics_analyzer = BiomechanicsAnalyzer()
    
    # Selector de Ejercicios
    trackers = [BicepCurlTracker(), LateralRaiseTracker(), SideBicepCurlTracker()]
    current_tracker_index = 0
    current_tracker = trackers[current_tracker_index]
    
    # Configuración de Ventana a Pantalla Completa
    window_name = "GymPose AI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Cooldown para gestos
    gesture_cooldown = 0
    
    # Iniciar la captura de video
    if not video_handler.start_capture():
        print("Error: No se pudo abrir la fuente de video (cámara).")
        return
        
    print("Iniciando GymPose AI. Presiona 'q' para salir.")
    
    # Bucle principal de procesamiento de video
    while True:
        success, frame = video_handler.read_frame()
        if not success or frame is None:
            print("Error o fin de captura de video.")
            break
            
        # Detección de postura
        results = pose_detector.process_frame(frame)
        
        # Dibujo de landmarks en el frame
        frame_drawn = pose_detector.draw_landmarks(frame.copy(), results)
        
        # Obtener dimensiones del frame
        h, w, _ = frame_drawn.shape
        
        # Procesamiento Biomecánico
        landmarks = pose_detector.get_landmarks(results)
        
        if gesture_cooldown > 0:
            gesture_cooldown -= 1
            
        if landmarks:
            current_tracker.update(landmarks, w, h, side='left')
            
            # Lógica de Cambio por Gestos (Gesto "X" con las muñecas)
            try:
                wrist_left = landmarks[15]
                wrist_right = landmarks[16]
                shoulder_left = landmarks[11]
                shoulder_right = landmarks[12]
                
                # Distancia euclidiana 3D en coordenadas normalizadas
                dx = wrist_left.x - wrist_right.x
                dy = wrist_left.y - wrist_right.y
                dz = wrist_left.z - wrist_right.z
                dist_3d = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                cv2.putText(frame_drawn, f"Debug - Dist 3D: {dist_3d:.3f}", (w - 250, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                
                if gesture_cooldown == 0 and dist_3d < 0.25 and (wrist_left.y < shoulder_left.y) and (wrist_right.y < shoulder_right.y):
                    print("Gesto detectado: Cambio de ejercicio")
                    current_tracker_index = (current_tracker_index + 1) % len(trackers)
                    current_tracker = trackers[current_tracker_index]
                    gesture_cooldown = 30
            except Exception:
                pass
                
            # Debug Mode Angle
            debug_angle = getattr(current_tracker, 'current_angle', 0.0)
            if debug_angle is not None:
                cv2.putText(frame_drawn, f"Angulo Codo: {debug_angle:.1f}*", (15, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
        # HUD (Heads-Up Display)
        # Dibujar rectángulo semitransparente como fondo del HUD
        overlay = frame_drawn.copy()
        cv2.rectangle(overlay, (0, 0), (w, 140), (200, 200, 200), -1)
        # Mezclar overlay con frame original
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame_drawn, 1 - alpha, 0, frame_drawn)
        
        # Nombre del Ejercicio
        cv2.putText(frame_drawn, f"Ejercicio: {current_tracker.name} (1: Curl, 2: Elev., 3: Lateral)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Reps
        cv2.putText(frame_drawn, 'REPS', (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_drawn, str(current_tracker.reps), (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
        
        # Stage
        cv2.putText(frame_drawn, 'STAGE', (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        stage_text = current_tracker.stage if current_tracker.stage else "-"
        cv2.putText(frame_drawn, stage_text, (150, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
        
        # Feedback
        cv2.putText(frame_drawn, 'FEEDBACK', (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Color del feedback: Verde si es buena postura, rojo si hay error
        fb_color = (0, 150, 0) if current_tracker.feedback == "Buena postura" else (0, 0, 255)
        draw_multiline_text(frame_drawn, current_tracker.feedback, 350, 100, 0.8, fb_color, 2, max_width=30)
        
        # Mostrar el resultado
        cv2.imshow(window_name, frame_drawn)
        
        # Manejo de Teclado
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            current_tracker_index = 0
            current_tracker = trackers[current_tracker_index]
        elif key == ord('2'):
            current_tracker_index = 1
            current_tracker = trackers[current_tracker_index]
        elif key == ord('3'):
            current_tracker_index = 2
            current_tracker = trackers[current_tracker_index]
        elif key == ord('q'):
            print("Cerrando la aplicación...")
            break
            
    # Liberar recursos
    video_handler.release()
    pose_detector.release_resources()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
