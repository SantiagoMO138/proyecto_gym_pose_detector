"""
Panel lateral izquierdo: información del ejercicio, reps, stage y feedback.
"""

from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QProgressBar, QWidget
from PySide6.QtCore import Qt


class LeftSidebar(QFrame):
    """Sidebar izquierdo con datos del ejercicio activo."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(220)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── Ejercicio ──
        self.exercise_label = QLabel("CURL DE BÍCEPS")
        self.exercise_label.setStyleSheet("color: #58a6ff; font-size: 15px; font-weight: bold;")
        self.exercise_label.setWordWrap(True)
        layout.addWidget(self.exercise_label)

        layout.addWidget(self._separator())

        # ── REPS ──
        self.reps_title = QLabel("REPS")
        self.reps_title.setStyleSheet("color: #8b949e; font-size: 10px; letter-spacing: 1px;")
        layout.addWidget(self.reps_title)

        self.reps_value = QLabel("0")
        self.reps_value.setAlignment(Qt.AlignCenter)
        self.reps_value.setStyleSheet("color: #e6edf3; font-size: 52px; font-weight: bold;")
        layout.addWidget(self.reps_value)

        self.reps_progress = QProgressBar()
        self.reps_progress.setMaximum(12)
        self.reps_progress.setValue(0)
        self.reps_progress.setTextVisible(False)
        self.reps_progress.setFixedHeight(6)
        layout.addWidget(self.reps_progress)

        layout.addWidget(self._separator())

        # ── STAGE ──
        self.stage_title = QLabel("STAGE")
        self.stage_title.setStyleSheet("color: #8b949e; font-size: 10px; letter-spacing: 1px;")
        layout.addWidget(self.stage_title)

        self.stage_value = QLabel("-")
        self.stage_value.setStyleSheet("color: #e6edf3; font-size: 20px; font-weight: bold;")
        layout.addWidget(self.stage_value)

        layout.addWidget(self._separator())

        # ── FEEDBACK ──
        self.feedback_title = QLabel("FEEDBACK")
        self.feedback_title.setStyleSheet("color: #8b949e; font-size: 10px; letter-spacing: 1px;")
        layout.addWidget(self.feedback_title)

        self.feedback_value = QLabel("Esperando...")
        self.feedback_value.setStyleSheet("color: #8b949e; font-size: 13px;")
        self.feedback_value.setWordWrap(True)
        layout.addWidget(self.feedback_value)

        layout.addStretch()

    def update_tracker(self, tracker):
        """Actualiza todos los widgets con los datos del tracker actual."""
        self.exercise_label.setText(tracker.name.upper())
        self.reps_value.setText(str(tracker.reps))
        self.reps_progress.setValue(tracker.reps)
        self.stage_value.setText(tracker.stage.upper() if tracker.stage else "-")

        is_good = tracker.feedback == "Buena postura"
        color = "#3fb950" if is_good else "#f85149"
        self.feedback_value.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: bold;")
        self.feedback_value.setText(tracker.feedback)

        # Color del stage según valor
        if tracker.stage == "arriba":
            self.stage_value.setStyleSheet("color: #58a6ff; font-size: 20px; font-weight: bold;")
        else:
            self.stage_value.setStyleSheet("color: #e6edf3; font-size: 20px; font-weight: bold;")

    def set_target_reps(self, value: int):
        """Cambia el máximo de la barra de progreso."""
        self.reps_progress.setMaximum(value)

    def update_no_person(self):
        """Muestra estado de espera cuando no hay persona."""
        self.feedback_value.setStyleSheet("color: #8b949e; font-size: 13px;")
        self.feedback_value.setText("Esperando persona...")
        self.stage_value.setText("-")
        self.stage_value.setStyleSheet("color: #8b949e; font-size: 20px;")
        self.reps_progress.setValue(0)

    def _separator(self) -> QWidget:
        """Crea una línea separadora fina."""
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #30363d;")
        return line
