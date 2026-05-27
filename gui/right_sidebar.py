"""
Panel lateral derecho: estado persona, emoción, clasificador y configuración.
"""

from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QSpinBox, QWidget


class RightSidebar(QFrame):
    """Sidebar derecho con datos de detección y configuración."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(220)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── Persona ──
        self.person_status = QLabel("● Sin persona")
        self.person_status.setStyleSheet("color: #f85149; font-size: 13px; font-weight: bold;")
        layout.addWidget(self.person_status)

        layout.addWidget(self._separator())

        # ── Emoción ──
        self.emotion_title = QLabel("EMOCIÓN")
        self.emotion_title.setStyleSheet("color: #8b949e; font-size: 10px; letter-spacing: 1px;")
        layout.addWidget(self.emotion_title)

        self.emotion_value = QLabel("-")
        self.emotion_value.setStyleSheet("color: #8b949e; font-size: 16px; font-weight: bold;")
        layout.addWidget(self.emotion_value)

        layout.addWidget(self._separator())

        # ── Clasificador ──
        self.classifier_title = QLabel("CLASIFICADOR")
        self.classifier_title.setStyleSheet("color: #8b949e; font-size: 10px; letter-spacing: 1px;")
        layout.addWidget(self.classifier_title)

        self.classifier_value = QLabel("-")
        self.classifier_value.setStyleSheet("color: #8b949e; font-size: 12px;")
        self.classifier_value.setWordWrap(True)
        layout.addWidget(self.classifier_value)

        layout.addWidget(self._separator())

        # ── Target reps ──
        self.target_title = QLabel("TARGET REPS")
        self.target_title.setStyleSheet("color: #8b949e; font-size: 10px; letter-spacing: 1px;")
        layout.addWidget(self.target_title)

        self.target_spin = QSpinBox()
        self.target_spin.setRange(1, 50)
        self.target_spin.setValue(12)
        layout.addWidget(self.target_spin)

        layout.addWidget(self._separator())

        # ── Ángulo (debug) ──
        self.angle_title = QLabel("ÁNGULO")
        self.angle_title.setStyleSheet("color: #8b949e; font-size: 10px; letter-spacing: 1px;")
        layout.addWidget(self.angle_title)

        self.angle_value = QLabel("-")
        self.angle_value.setStyleSheet("color: #8b949e; font-size: 14px; font-family: monospace;")
        layout.addWidget(self.angle_value)

        layout.addStretch()

    def update_sidebar(self, person_detected: bool, emotion_data, classifier_label: str,
                       classifier_conf: float, angle: float = None):
        """Actualiza todos los widgets del sidebar."""
        # Persona
        if person_detected:
            self.person_status.setText("● Persona OK")
            self.person_status.setStyleSheet("color: #3fb950; font-size: 13px; font-weight: bold;")
        else:
            self.person_status.setText("● Sin persona")
            self.person_status.setStyleSheet("color: #f85149; font-size: 13px; font-weight: bold;")

        # Emoción
        if emotion_data and emotion_data.get("dominant_emotion"):
            dom = emotion_data["dominant_emotion"]
            conf = emotion_data.get("confidence", 0)
            color = self._emotion_color(dom)
            self.emotion_value.setStyleSheet(
                f"color: {color}; font-size: 16px; font-weight: bold;"
            )
            self.emotion_value.setText(f"{dom} {conf:.0f}%")
        else:
            self.emotion_value.setStyleSheet("color: #8b949e; font-size: 16px;")
            self.emotion_value.setText("-")

        # Clasificador
        if classifier_label and classifier_label != "-":
            conf_text = f"{classifier_conf:.0%}" if classifier_conf > 0 else ""
            self.classifier_value.setText(f"{classifier_label}\n{conf_text}")
            self.classifier_value.setStyleSheet("color: #d29922; font-size: 12px;")
        else:
            self.classifier_value.setText("-")
            self.classifier_value.setStyleSheet("color: #8b949e; font-size: 12px;")

        # Ángulo
        if angle is not None:
            self.angle_value.setText(f"{angle:.1f}°")
            self.angle_value.setStyleSheet("color: #e6edf3; font-size: 14px; font-family: monospace;")
        else:
            self.angle_value.setText("-")
            self.angle_value.setStyleSheet("color: #8b949e; font-size: 14px; font-family: monospace;")

    def get_target_reps(self) -> int:
        """Retorna el valor actual del target reps."""
        return self.target_spin.value()

    def _emotion_color(self, emotion: str) -> str:
        """Retorna código hex según la emoción."""
        mapping = {
            "happy": "#3fb950",
            "surprise": "#58a6ff",
            "neutral": "#8b949e",
            "sad": "#d29922",
            "angry": "#f85149",
            "fear": "#bc8cff",
            "disgust": "#7ee787",
        }
        return mapping.get(emotion.lower(), "#8b949e")

    def _separator(self) -> QWidget:
        """Crea una línea separadora fina."""
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #30363d;")
        return line
