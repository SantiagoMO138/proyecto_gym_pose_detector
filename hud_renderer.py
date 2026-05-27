"""
Renderizador de HUD moderno y limpio para GymPose AI.
Diseño: tema oscuro con acentos cyan, paneles sólidos, tipografía jerárquica.
"""

import cv2
import numpy as np


class HUDRenderer:
    """
    Renderiza un HUD visualmente atractivo pero limpio sobre el frame de video.
    Usa paneles sólidos oscuros con acentos de color.
    """

    # Paleta BGR
    BG_DARK = (25, 35, 45)          # Fondo panel
    BG_DARKER = (15, 20, 30)        # Fondo más oscuro (barras)
    ACCENT = (255, 229, 0)          # Cyan
    SUCCESS = (76, 175, 80)         # Verde
    WARNING = (7, 193, 255)         # Ámbar
    ERROR = (82, 82, 255)           # Rojo
    WHITE = (255, 255, 255)
    GRAY = (176, 190, 197)
    DARK_GRAY = (80, 80, 80)

    def __init__(self, target_reps: int = 12) -> None:
        self.target_reps = target_reps

    def render(
        self,
        frame: np.ndarray,
        tracker,
        classifier_label: str,
        classifier_conf: float,
        emotion_data,
        person_detected: bool,
        is_classifying: bool,
    ) -> np.ndarray:
        """
        Renderiza todo el HUD sobre el frame.

        Args:
            frame: Frame BGR de OpenCV (modificado in-place).
            tracker: Instancia del tracker actual (con .name, .reps, .stage, .feedback).
            classifier_label: Etiqueta textual del clasificador ViT.
            classifier_conf: Confianza del clasificador.
            emotion_data: Dict de emoción o None.
            person_detected: True si YOLO detectó persona.
            is_classifying: True si el clasificador está corriendo este frame.

        Returns:
            Frame con HUD renderizado.
        """
        h, w = frame.shape[:2]

        # ── Top bar (info general) ──
        self._render_top_bar(frame, w, tracker.name, person_detected, emotion_data,
                             classifier_label, classifier_conf)

        # ── Bottom bar (reps + stage + feedback) ──
        self._render_bottom_bar(frame, w, h, tracker.reps, tracker.stage, tracker.feedback)

        # ── Rep progress bar (línea fina inferior) ──
        self._render_rep_progress(frame, w, h, tracker.reps)

        # ── Indicador de clasificación activa (punto cyan) ──
        if is_classifying:
            cv2.circle(frame, (w - 25, 25), 5, self.ACCENT, -1)
            cv2.circle(frame, (w - 25, 25), 5, self.WHITE, 1)

        return frame

    # ─────────────────────────────────────────────
    # Paneles
    # ─────────────────────────────────────────────

    def _render_top_bar(self, frame, w, exercise_name, person_detected, emotion_data,
                        classifier_label, classifier_conf):
        """Barra superior sólida con info del ejercicio, persona, emoción y clasificador."""
        bar_h = 70
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), self.BG_DARKER, -1)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)

        y_line1 = 28
        y_line2 = 52

        # ── Línea 1: Ejercicio (izq) ──
        cv2.putText(
            frame,
            exercise_name.upper(),
            (20, y_line1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            self.ACCENT,
            2,
            cv2.LINE_AA,
        )

        # ── Línea 1: Persona (centro-izq) ──
        person_x = w // 2 - 180
        person_color = self.SUCCESS if person_detected else self.ERROR
        cv2.circle(frame, (person_x, y_line1 - 5), 5, person_color, -1)
        cv2.putText(
            frame,
            "OK" if person_detected else "NO",
            (person_x + 14, y_line1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.GRAY,
            1,
            cv2.LINE_AA,
        )

        # ── Línea 1: Emoción dominante (centro-der) ──
        if emotion_data and emotion_data.get("dominant_emotion"):
            dom = emotion_data["dominant_emotion"]
            conf = emotion_data.get("confidence", 0)
            emotion_text = f"{dom} {conf:.0f}%"
            emotion_color = self._emotion_color(dom)
            cv2.putText(
                frame,
                emotion_text,
                (w // 2 + 20, y_line1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                emotion_color,
                1,
                cv2.LINE_AA,
            )

        # ── Línea 2: Clasificador (derecha) ──
        if classifier_label != "-":
            clf_text = f"{classifier_label}"
            clf_color = self.WARNING if classifier_conf >= 0.5 else self.GRAY
            # Dibujar fondo para legibilidad
            (tw, th), _ = cv2.getTextSize(clf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (w - tw - 30, y_line2 - th - 2), (w - 20, y_line2 + 4),
                          self.BG_DARKER, -1)
            cv2.putText(
                frame,
                clf_text,
                (w - tw - 25, y_line2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                clf_color,
                1,
                cv2.LINE_AA,
            )

    def _render_bottom_bar(self, frame, w, h, reps, stage, feedback):
        """Barra inferior con REPS, STAGE y FEEDBACK."""
        bar_y = h - 130
        bar_h = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y), (w, h), self.BG_DARKER, -1)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)

        # Dividir en 3 secciones
        sec_w = w // 3

        # ── Sección 1: REPS ──
        self._draw_section_label(frame, "REPS", 20, bar_y + 25)
        cv2.putText(
            frame,
            str(reps),
            (20, bar_y + 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.2,
            self.WHITE,
            3,
            cv2.LINE_AA,
        )

        # ── Sección 2: STAGE ──
        self._draw_section_label(frame, "STAGE", sec_w + 20, bar_y + 25)
        stage_text = stage.upper() if stage else "-"
        stage_color = self.ACCENT if stage == "arriba" else self.GRAY
        cv2.putText(
            frame,
            stage_text,
            (sec_w + 20, bar_y + 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            stage_color,
            2,
            cv2.LINE_AA,
        )

        # ── Sección 3: FEEDBACK ──
        self._draw_section_label(frame, "FEEDBACK", 2 * sec_w + 20, bar_y + 25)
        is_good = feedback == "Buena postura"
        fb_color = self.SUCCESS if is_good else self.ERROR
        fb_text = feedback if feedback else "-"

        # Si el texto es largo, dividirlo en líneas cortas
        max_chars = 22
        if len(fb_text) > max_chars:
            words = fb_text.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= max_chars:
                    current_line += (" " if current_line else "") + word
                else:
                    lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
        else:
            lines = [fb_text]

        y_offset = bar_y + 55
        for line in lines:
            cv2.putText(
                frame,
                line,
                (2 * sec_w + 20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                fb_color,
                1,
                cv2.LINE_AA,
            )
            y_offset += 22

    def _render_rep_progress(self, frame, w, h, reps):
        """Barra de progreso fina en el borde inferior."""
        bar_y = h - 6
        bar_h = 6
        progress = min(reps / self.target_reps, 1.0)
        fill_w = int(w * progress)

        # Fondo
        cv2.rectangle(frame, (0, bar_y), (w, bar_y + bar_h), self.DARK_GRAY, -1)
        # Relleno
        if fill_w > 0:
            cv2.rectangle(frame, (0, bar_y), (fill_w, bar_y + bar_h), self.ACCENT, -1)
        # Borde
        cv2.rectangle(frame, (0, bar_y), (w, bar_y + bar_h), self.BG_DARKER, 1)

    # ─────────────────────────────────────────────
    # Utilidades
    # ─────────────────────────────────────────────

    def _draw_section_label(self, frame, text, x, y):
        """Dibuja un label secundario en color gris."""
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.GRAY,
            1,
            cv2.LINE_AA,
        )

    def _emotion_color(self, emotion: str):
        """Retorna color BGR según la emoción."""
        mapping = {
            "happy": self.SUCCESS,
            "surprise": self.ACCENT,
            "neutral": self.GRAY,
            "sad": (255, 150, 0),      # Naranja-azul
            "angry": self.ERROR,
            "fear": (255, 0, 255),     # Magenta
            "disgust": (0, 128, 128),  # Verde oscuro
        }
        return mapping.get(emotion.lower(), self.GRAY)
