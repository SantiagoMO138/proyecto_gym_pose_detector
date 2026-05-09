# Presentación del Proyecto: GymPose AI

## 1. Introducción y Objetivos

**GymPose AI** es una aplicación de visión artificial y seguimiento de fitness desarrollada en Python. Su objetivo principal es actuar como un entrenador personal virtual, capaz de analizar la biomecánica del usuario en tiempo real durante sus rutinas de ejercicio. 

**Objetivos Clave:**
*   **Monitoreo en Tiempo Real:** Analizar el movimiento del usuario a través de una cámara web.
*   **Corrección Postural:** Prevenir lesiones y maximizar la efectividad del ejercicio mediante alertas visuales al detectar compensaciones o malos movimientos (ej. balanceo lumbar, encogimiento de hombros).
*   **Experiencia Fluida:** Ofrecer un conteo automático de repeticiones y control de la aplicación sin necesidad de tocar el teclado, utilizando gestos.

---

## 2. Arquitectura y Tecnologías

El proyecto está estructurado modularmente para separar la captura de video, la detección del cuerpo y la lógica biomecánica.

*   **MediaPipe Tasks API (`pose_detector.py`):** Es el núcleo de la detección de postura. Se migró recientemente a la nueva API de tareas de MediaPipe para asegurar la compatibilidad con Python 3.12 y mejorar la estabilidad. Utiliza modelos pre-entrenados (como `pose_landmarker_full.task`) para obtener coordenadas 3D de los puntos articulares (landmarks).
*   **OpenCV (`video_handler.py` y `main.py`):** Se encarga de la captura de video desde la cámara web, el procesamiento de las imágenes fotograma a fotograma y la renderización de la interfaz gráfica de usuario (HUD) en modo de pantalla completa.
*   **NumPy (`biomechanics.py`):** Utilizado exhaustivamente para el cálculo eficiente de álgebra vectorial en 3D (productos punto, normas) necesarios para obtener los ángulos articulares exactos.

---

## 3. Características Principales y Funcionalidad

El sistema actual cuenta con un robusto motor biomecánico que soporta múltiples ejercicios y lógicas complejas:

*   **Ejercicios Soportados:**
    *   **Curl de Bíceps (Bilateral):** Exige sincronía entre ambos brazos. Detecta asimetrías, balanceo del tronco y separación de los codos.
    *   **Elevaciones Laterales:** Mide el ángulo de abducción del hombro. Evita la elevación excesiva, hiperextensión del codo y compensación con inclinación del tronco.
    *   **Curl Lateral a 1 Brazo:** Diseñado para la vista lateral. Detecta trampas de impulso (muñeca subiendo prematuramente) y balanceo.
*   **Suavizado de Señal (EMA):** Se aplica un Suavizado Exponencial (Exponential Moving Average) a los ángulos en tiempo real para evitar conteos falsos por "ruido" en la detección de la cámara.
*   **HUD Interactivo y Feedback Visual:** La pantalla muestra el ejercicio actual, el contador de repeticiones, la fase del movimiento (arriba/abajo) y texto dinámico multilínea que cambia a color rojo si la postura es incorrecta y verde si es "Buena postura".
*   **Control por Gestos:** El usuario puede cambiar de ejercicio cruzando las muñecas frente a los hombros (Gesto de "X"), lo que elimina la necesidad de interactuar con el mouse o teclado durante el entrenamiento.

---

## 4. Estado Actual y Próximos Pasos

**Estado Actual:**
El proyecto ha superado con éxito la fase de refactorización hacia la API moderna de MediaPipe. El motor biomecánico (`biomechanics.py`) es altamente sofisticado, manejando visibilidad de landmarks, ejes en 3D y reglas estrictas para validar cada repetición. La interfaz de pantalla completa funciona de manera estable.

**Próximos Pasos Recomendados (Roadmap):**
1.  **Ampliación del Catálogo de Ejercicios:** Incorporar movimientos de tren inferior como Sentadillas (Squats) y Peso Muerto (Deadlifts), requiriendo nuevas lógicas de seguimiento de rodillas y cadera.
2.  **Calibración Personalizada:** Añadir una fase inicial donde la app "mida" las proporciones del usuario para ajustar los umbrales de distancia y ángulo de manera dinámica.
3.  **Persistencia de Datos y Estadísticas:** Guardar el progreso del usuario (repeticiones, series, calidad promedio de la postura) en una base de datos local o en la nube (ej. SQLite o exportación CSV).
4.  **Evolución de la Interfaz:** Migrar la interfaz puramente de OpenCV hacia una arquitectura web o de escritorio más rica (ej. PyQt, Streamlit o un frontend web conectado vía WebSockets).
