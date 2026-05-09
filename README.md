1. README.md (Para el repositorio y usuarios finales)
Markdown
# 🏋️‍♂️ GymPose AI: Detección y Corrección de Ejercicios

## 📌 Descripción
GymPose AI es una aplicación de visión por computadora diseñada para analizar y corregir la postura durante la ejecución de ejercicios de gimnasio. Utiliza **MediaPipe Pose** para la extracción de landmarks corporales en tiempo real y **OpenCV** para el procesamiento del flujo de video. El sistema calcula ángulos biomecánicos críticos para determinar si un ejercicio (como una sentadilla o un curl de bíceps) se está ejecutando con la técnica correcta.

## 🚀 Instalación y Configuración del Entorno

Es estrictamente necesario utilizar un entorno virtual para aislar las dependencias del proyecto.

### 1. Crear y Activar el Entorno Virtual
Abre la terminal en la raíz del proyecto:

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
En macOS/Linux:

Bash
python3 -m venv venv
source venv/bin/activate
2. Instalar Dependencias
Asegúrate de tener pip actualizado antes de instalar las librerías:

Bash
python -m pip install --upgrade pip
pip install -r requirements.txt
(El archivo requirements.txt debe contener: mediapipe, opencv-python, numpy).

💻 Uso
Para iniciar el detector con tu cámara web:

Bash
python main.py