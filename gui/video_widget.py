"""
Widget de video: QLabel que convierte frames OpenCV BGR a QPixmap.
"""

import cv2
import numpy as np
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap


class VideoWidget(QLabel):
    """Muestra frames de OpenCV centrados y escalados."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #000000;")

    def set_frame(self, frame: np.ndarray):
        """Convierte un frame BGR de OpenCV a QPixmap y lo muestra."""
        if frame is None or frame.size == 0:
            return

        # BGR → RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Escalar al tamaño del widget manteniendo aspecto
        scaled = pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled)
