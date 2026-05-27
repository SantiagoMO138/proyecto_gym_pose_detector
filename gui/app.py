"""
Entry point de la aplicación GymPose AI con PySide6.
"""

import sys
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow


def run():
    """Inicializa la aplicación Qt y muestra la ventana principal."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Paleta oscura global
    app.setStyleSheet("""
        QMainWindow { background-color: #0d1117; }
        QWidget { background-color: #0d1117; }
        QFrame#sidebar {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
        }
        QLabel {
            color: #e6edf3;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QProgressBar {
            background-color: #21262d;
            border: none;
            border-radius: 3px;
            height: 6px;
        }
        QProgressBar::chunk {
            background-color: #58a6ff;
            border-radius: 3px;
        }
        QSpinBox {
            background-color: #21262d;
            color: #e6edf3;
            border: 1px solid #30363d;
            border-radius: 4px;
            padding: 5px;
        }
        QSpinBox::up-button, QSpinBox::down-button {
            background-color: #30363d;
            border-radius: 2px;
        }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
