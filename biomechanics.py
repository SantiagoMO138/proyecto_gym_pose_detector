"""
Módulo para el cálculo y análisis de métricas biomecánicas basadas en la postura detectada.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np


def calculate_angle(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    c: Tuple[float, float, float],
) -> float:
    """
    Calcula el ángulo formado por tres puntos (A, B, C) donde B es el vértice.
    Utiliza álgebra vectorial (producto punto) en 3D.

    Args:
        a: Coordenadas (x, y, z) del primer punto.
        b: Coordenadas (x, y, z) del vértice.
        c: Coordenadas (x, y, z) del tercer punto.

    Returns:
        El ángulo absoluto en grados [0, 180].
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    v1 = a - b
    v2 = c - b

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    cosine_angle = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


def is_visible(*args: Any, threshold: float = 0.5) -> bool:
    """
    Verifica que los landmarks superen el umbral de visibilidad.
    Imprime qué punto falla para facilitar el debug.
    Acepta tanto landmarks posicionales como un diccionario en args[0].
    """
    if len(args) == 1 and isinstance(args[0], dict):
        landmarks_dict = args[0]
    else:
        landmarks_dict = {f"Landmark_{i}": lm for i, lm in enumerate(args)}
        
    for name, lm in landmarks_dict.items():
        if getattr(lm, 'visibility', 0) < threshold:
            print(f"Baja visibilidad en {name}: {getattr(lm, 'visibility', 0):.2f}")
            return False
    return True


def _build_vertical_ref(hip: Tuple, height: int) -> Tuple:
    """
    Construye un punto de referencia vertical normalizado por la altura del frame.
    El punto se ubica un 20% de la altura del frame por encima de la cadera,
    lo que garantiza invarianza respecto a la resolución.

    Args:
        hip: Coordenadas (x, y, z) de la cadera.
        height: Alto del frame en píxeles.

    Returns:
        Coordenadas (x, y, z) del punto de referencia vertical.
    """
    offset = height * 0.20
    return (hip[0], hip[1] - offset, hip[2])


def _ema(current: float, previous: Optional[float], alpha: float) -> float:
    """
    Aplica suavizado exponencial (EMA) a una señal escalar.

    Args:
        current: Valor actual de la señal.
        previous: Valor suavizado en el frame anterior. Si es None, retorna current.
        alpha: Factor de suavizado en (0, 1]. Valores altos = más reactivo.

    Returns:
        Valor suavizado.
    """
    if previous is None:
        return current
    return alpha * current + (1.0 - alpha) * previous


class BicepCurlTracker:
    """
    Seguimiento y evaluación biomecánica del ejercicio 'Curl de Bíceps' bilateral.

    La rep se contabiliza solo cuando AMBOS brazos alcanzan simultáneamente la
    flexión completa (elbow_angle < FLEXION_THRESHOLD), lo que impide contar
    repeticiones asimétricas. Cada brazo tiene su propio estado de stage y EMA
    independiente; la FSM del contador es conjunta.

    Ángulos monitoreados por brazo:
        - elbow_angle  : Hombro–Codo–Muñeca. Controla el contador.
        - sway_angle   : Cadera–Hombro–Codo. Detecta separación del codo del torso.
        - trunk_angle  : Hombro–Cadera–VerticalRef. Detecta balanceo lumbar.

    Restricciones de postura (en orden de prioridad):
        1. Balanceo lumbar: trunk_angle ∈ (20°, 80°).
        2. Separación de codo: sway_angle > 30°.
        3. Asimetría de contracción: diferencia de ángulo de codo entre brazos > 25°
           durante la fase activa → un brazo lidera demasiado al otro.
        4. Encogimiento de hombro: diferencia de altura entre hombros > umbral.
    """

    EXTENSION_THRESHOLD = 140.0
    FLEXION_THRESHOLD   = 75.0

    TRUNK_SWAY_MIN            = 20.0
    TRUNK_SWAY_MAX            = 30
    ELBOW_SWAY_MAX            = 30.0
    ASYMMETRY_THRESHOLD       = 25.0  # Diferencia máxima de ángulo entre brazos (grados)
    SHOULDER_SHRUG_THRESHOLD  = 0.04  # Diferencia y normalizada entre hombros

    # Visibilidad mínima absoluta para considerar un brazo como activo
    MIN_ARM_VISIBILITY = 0.6

    def __init__(self) -> None:
        self.reps = 0
        self.stage: Optional[str] = None
        self.feedback = "Buena postura"
        self.name = "Curl de Biceps"
        self.current_angle = 0.0
        self.alpha = 0.4

        # Estado EMA independiente por lado
        self._prev_elbow_l: Optional[float] = None
        self._prev_elbow_r: Optional[float] = None
        self._prev_sway_l:  Optional[float] = None
        self._prev_sway_r:  Optional[float] = None
        self._prev_trunk:   Optional[float] = None

        # Stage independiente por brazo
        self._stage_l: Optional[str] = None
        self._stage_r: Optional[str] = None

    def _arm_coords(
        self,
        lm_shoulder: Any, lm_elbow: Any, lm_wrist: Any, lm_hip: Any,
        width: int, height: int, z_weight: float,
    ) -> Tuple[Tuple, Tuple, Tuple, Tuple]:
        shoulder = (lm_shoulder.x * width, lm_shoulder.y * height, lm_shoulder.z * width * z_weight)
        elbow    = (lm_elbow.x * width,    lm_elbow.y * height,    lm_elbow.z * width * z_weight)
        wrist    = (lm_wrist.x * width,    lm_wrist.y * height,    lm_wrist.z * width * z_weight)
        hip      = (lm_hip.x * width,      lm_hip.y * height,      lm_hip.z * width * z_weight)
        return shoulder, elbow, wrist, hip

    def update(self, landmarks: List[Any], width: int, height: int, side: str = 'left') -> None:
        """
        Actualiza el estado del tracker procesando ambos brazos simultáneamente.
        La rep se cuenta solo si ambos brazos completan la flexión.

        Args:
            landmarks: Lista de NormalizedLandmark de MediaPipe.
            width: Ancho del frame en píxeles.
            height: Alto del frame en píxeles.
            side: Parámetro legacy, ignorado.
        """
        if not landmarks:
            return

        try:
            vis_l = getattr(landmarks[11], 'visibility', 0)
            vis_r = getattr(landmarks[12], 'visibility', 0)

            has_left  = vis_l >= self.MIN_ARM_VISIBILITY
            has_right = vis_r >= self.MIN_ARM_VISIBILITY

            # Si ningún brazo es visible, no procesar
            if not has_left and not has_right:
                return

            is_frontal = abs(landmarks[11].x - landmarks[12].x) > 0.15
            z_weight   = 0.2 if is_frontal else 1.0

            # Usar la cadera más visible como referencia del tronco
            lm_hip = landmarks[23] if vis_l >= vis_r else landmarks[24]
            hip_coords_l = (landmarks[23].x * width, landmarks[23].y * height, landmarks[23].z * width * z_weight)
            hip_coords_r = (landmarks[24].x * width, landmarks[24].y * height, landmarks[24].z * width * z_weight)
            v_ref_l = _build_vertical_ref(hip_coords_l, height)

            smoothed_elbow_l: Optional[float] = None
            smoothed_elbow_r: Optional[float] = None
            smoothed_sway_l:  Optional[float] = None
            smoothed_sway_r:  Optional[float] = None

            # --- Brazo izquierdo ---
            if has_left and is_visible(landmarks[13], landmarks[15], threshold=0.4):
                s, e, w, h = self._arm_coords(
                    landmarks[11], landmarks[13], landmarks[15], landmarks[23],
                    width, height, z_weight,
                )
                ea = calculate_angle(s, e, w)
                sa = calculate_angle(h, s, e)
                self._prev_elbow_l = _ema(ea, self._prev_elbow_l, self.alpha)
                self._prev_sway_l  = _ema(sa, self._prev_sway_l,  self.alpha)
                smoothed_elbow_l   = self._prev_elbow_l
                smoothed_sway_l    = self._prev_sway_l

                if smoothed_elbow_l > self.EXTENSION_THRESHOLD:
                    self._stage_l = "abajo"

            # --- Brazo derecho ---
            if has_right and is_visible(landmarks[14], landmarks[16], threshold=0.4):
                s, e, w, h = self._arm_coords(
                    landmarks[12], landmarks[14], landmarks[16], landmarks[24],
                    width, height, z_weight,
                )
                ea = calculate_angle(s, e, w)
                sa = calculate_angle(h, s, e)
                self._prev_elbow_r = _ema(ea, self._prev_elbow_r, self.alpha)
                self._prev_sway_r  = _ema(sa, self._prev_sway_r,  self.alpha)
                smoothed_elbow_r   = self._prev_elbow_r
                smoothed_sway_r    = self._prev_sway_r

                if smoothed_elbow_r > self.EXTENSION_THRESHOLD:
                    self._stage_r = "abajo"

            # Tronco (usa brazo más visible)
            if has_left:
                s_trunk = (landmarks[11].x * width, landmarks[11].y * height, landmarks[11].z * width * z_weight)
                trunk_angle = calculate_angle(s_trunk, hip_coords_l, v_ref_l)
            else:
                s_trunk = (landmarks[12].x * width, landmarks[12].y * height, landmarks[12].z * width * z_weight)
                trunk_angle = calculate_angle(s_trunk, hip_coords_r, _build_vertical_ref(hip_coords_r, height))
            self._prev_trunk = _ema(trunk_angle, self._prev_trunk, self.alpha)
            smoothed_trunk = self._prev_trunk

            # Ángulo representativo para debug (promedio de los brazos procesados este frame)
            angles_available = [a for a in [smoothed_elbow_l, smoothed_elbow_r] if a is not None]
            if angles_available:
                self.current_angle = float(np.mean(angles_available))

            # --- Contador bilateral ---
            # Modo bilateral: ambos brazos visibles y con EMA inicializado.
            # La FSM conjunta requiere que AMBOS hayan llegado a extensión (stage_x == "abajo")
            # antes de aceptar una flexión. El flag "or None" se elimina para evitar contar
            # cuando un brazo no fue procesado en el frame actual.
            if has_left and has_right:
                l_extended = self._stage_l == "abajo"
                r_extended = self._stage_r == "abajo"
                l_flexed   = smoothed_elbow_l is not None and smoothed_elbow_l < self.FLEXION_THRESHOLD
                r_flexed   = smoothed_elbow_r is not None and smoothed_elbow_r < self.FLEXION_THRESHOLD

                # Transición a "abajo" solo cuando ambos brazos están extendidos
                if l_extended and r_extended:
                    self.stage = "abajo"

                # Contar rep solo cuando stage es "abajo" Y ambos brazos flexionaron
                if self.stage == "abajo" and l_flexed and r_flexed:
                    self.stage = "arriba"
                    self.reps += 1
                    # Resetear stages por brazo para forzar nueva extensión completa
                    self._stage_l = None
                    self._stage_r = None
            else:
                # Modo unilateral: opera con el brazo visible
                active_elbow = smoothed_elbow_l if has_left else smoothed_elbow_r
                if active_elbow is not None:
                    if active_elbow > self.EXTENSION_THRESHOLD:
                        self.stage = "abajo"
                    if active_elbow < self.FLEXION_THRESHOLD and self.stage == "abajo":
                        self.stage = "arriba"
                        self.reps += 1

            # --- Corrección postural ---
            # Restricción 1: balanceo lumbar
            if self.TRUNK_SWAY_MIN < smoothed_trunk < self.TRUNK_SWAY_MAX:
                self.feedback = "Manten palda recta"

            # Restricción 2: separación de codo (brazo más infractor)
            elif any(s is not None and s > self.ELBOW_SWAY_MAX
                     for s in [smoothed_sway_l, smoothed_sway_r]):
                self.feedback = "Codos pegados"

            # Restricción 3: asimetría de contracción entre brazos
            elif (smoothed_elbow_l is not None and smoothed_elbow_r is not None and
                  abs(smoothed_elbow_l - smoothed_elbow_r) > self.ASYMMETRY_THRESHOLD):
                self.feedback = "Nivela ambos brazos"

            # Restricción 4: encogimiento de hombro
            elif (self.stage == "arriba" and
                  (landmarks[12].y - landmarks[11].y) > self.SHOULDER_SHRUG_THRESHOLD):
                self.feedback = "No encogues el hombro"

            else:
                self.feedback = "Buena postura"

        except Exception:
            pass


class LateralRaiseTracker:
    """
    Seguimiento y evaluación biomecánica del ejercicio 'Elevaciones Laterales de Hombro'.

    Ángulos monitoreados:
        - shoulder_angle : Cadera–Hombro–Codo. Ángulo de abducción del húmero.
        - elbow_angle    : Hombro–Codo–Muñeca. Detecta hiperextensión del codo.
        - trunk_angle    : Hombro–Cadera–VerticalRef. Detecta inclinación lateral.

    Restricciones de postura (en orden de prioridad):
        1. Plano de movimiento: el codo no debe adelantarse al hombro (flexión anterior
           disfrazada de abducción). Se detecta por la diferencia de profundidad z entre
           codo y hombro en coordenadas normalizadas.
        2. Elevación excesiva: shoulder_angle > 100° → activa trapezoide y supraespinoso
           de forma no deseada para el objetivo del ejercicio.
        3. Inclinación lateral del tronco: trunk_angle > 15° → compensación con el torso.
        4. Muñeca por debajo del codo durante elevación: indica que el antebrazo cuelga,
           reduciendo la activación del deltoides medial.
        5. Hiperextensión del codo: elbow_angle > 175°.
        6. Simetría de hombros: diferencia de altura entre hombros activo y opuesto
           excede umbral → compensación por encogimiento.
    """

    # Umbrales del contador.
    # Con la referencia vertical en el hombro, el brazo colgando produce ~180° y
    # el brazo elevado al horizontal produce ~90°. Los umbrales se invierten respecto
    # a la formulación anterior que usaba calculate_angle(hip, shoulder, elbow).
    STAGE_DOWN_THRESHOLD = 150.0   # Brazo en reposo: ángulo > 150° → stage "abajo"
    STAGE_UP_THRESHOLD   = 100.0   # Brazo elevado:   ángulo < 100° → rep contada

    # Umbrales de corrección (escala de ángulo: reposo=180°, horizontal=90°, excesivo<80°)
    SHOULDER_MAX          = 80.0   # Elevación excesiva: ángulo < 80° (más allá del horizontal)
    ELBOW_HYPEREXTENSION  = 175.0
    TRUNK_LATERAL_MAX     = 40.0
    FORWARD_PLANE_THRESHOLD  = 0.1
    WRIST_BELOW_ELBOW_MARGIN = 10.0
    SHOULDER_SHRUG_THRESHOLD = 0.04

    def __init__(self) -> None:
        self.reps = 0
        self.stage: Optional[str] = None
        self.feedback = "Buena postura"
        self.name = "Elevaciones Laterales"
        self.current_angle = 0.0
        self.alpha = 0.4

        self._prev_shoulder: Optional[float] = None
        self._prev_elbow: Optional[float] = None
        self._prev_trunk: Optional[float] = None

    def update(self, landmarks: List[Any], width: int, height: int, side: str = 'left') -> None:
        """
        Actualiza el estado del tracker con los landmarks del frame actual.

        Args:
            landmarks: Lista de NormalizedLandmark de MediaPipe.
            width: Ancho del frame en píxeles.
            height: Alto del frame en píxeles.
            side: Parámetro legacy, ignorado (se usa visibilidad).
        """
        if not landmarks:
            return

        try:
            if getattr(landmarks[11], 'visibility', 0) >= getattr(landmarks[12], 'visibility', 0):
                idx_shoulder, idx_elbow, idx_wrist, idx_hip = 11, 13, 15, 23
                idx_shoulder_opposite = 12
            else:
                idx_shoulder, idx_elbow, idx_wrist, idx_hip = 12, 14, 16, 24
                idx_shoulder_opposite = 11

            lm_shoulder = landmarks[idx_shoulder]
            lm_elbow    = landmarks[idx_elbow]
            lm_wrist    = landmarks[idx_wrist]
            lm_hip      = landmarks[idx_hip]
            lm_shoulder_opp = landmarks[idx_shoulder_opposite]

            if not is_visible(lm_shoulder, lm_elbow, lm_wrist, lm_hip):
                return

            shoulder = (lm_shoulder.x * width, lm_shoulder.y * height, lm_shoulder.z * width)
            elbow    = (lm_elbow.x * width,    lm_elbow.y * height,    lm_elbow.z * width)
            wrist    = (lm_wrist.x * width,    lm_wrist.y * height,    lm_wrist.z * width)
            hip      = (lm_hip.x * width,      lm_hip.y * height,      lm_hip.z * width)
            v_ref    = _build_vertical_ref(hip, height)

            # Ángulo de abducción: vértice en hombro, entre la vertical del tronco
            # (hombro→punto encima del hombro) y el vector hombro→codo.
            # Esto mide cuánto se aleja el brazo del eje del tronco, que es lo que
            # define la elevación lateral independientemente de la posición de la cadera.
            shoulder_up_ref = (shoulder[0], shoulder[1] - height * 0.20, shoulder[2])
            shoulder_angle = calculate_angle(shoulder_up_ref, shoulder, elbow)

            elbow_angle    = calculate_angle(shoulder, elbow, wrist)
            trunk_angle    = calculate_angle(shoulder, hip, v_ref)

            self._prev_shoulder = _ema(shoulder_angle, self._prev_shoulder, self.alpha)
            self._prev_elbow    = _ema(elbow_angle,    self._prev_elbow,    self.alpha)
            self._prev_trunk    = _ema(trunk_angle,    self._prev_trunk,    self.alpha)

            smoothed_shoulder = self._prev_shoulder
            smoothed_elbow    = self._prev_elbow
            smoothed_trunk    = self._prev_trunk
            self.current_angle = smoothed_shoulder

            # --- Contador ---
            # Brazo en reposo (colgando) → ángulo alto (~180°) → stage "abajo"
            # Brazo elevado al horizontal → ángulo bajo (~90°) → rep contada
            if smoothed_shoulder > self.STAGE_DOWN_THRESHOLD:
                self.stage = "abajo"
            if smoothed_shoulder < self.STAGE_UP_THRESHOLD and self.stage == "abajo":
                self.reps += 1
                self.stage = "arriba"

            # --- Corrección postural ---

            # Restricción 1: plano de movimiento incorrecto (flexión anterior).
            # En la Tasks API, z es profundidad relativa a la cadera: valores más negativos
            # indican que el punto está más adelante respecto al cuerpo.
            # Si el codo está significativamente más adelante que el hombro, el brazo
            # sube hacia el frente en lugar de hacia el lateral.
            elbow_forward = lm_shoulder.z - lm_elbow.z
            if smoothed_shoulder < self.STAGE_UP_THRESHOLD and elbow_forward > self.FORWARD_PLANE_THRESHOLD:
                self.feedback = "Sube lateralmente"

            elif smoothed_shoulder < self.SHOULDER_MAX:
                self.feedback = "No pases del hombro"

            elif smoothed_trunk > self.TRUNK_LATERAL_MAX:
                self.feedback = "Nivela tu torso"

            elif (smoothed_shoulder < self.STAGE_UP_THRESHOLD and
                  lm_wrist.y * height > lm_elbow.y * height + self.WRIST_BELOW_ELBOW_MARGIN):
                self.feedback = "Nivela la muñeca"

            elif smoothed_elbow > self.ELBOW_HYPEREXTENSION:
                self.feedback = "No extiendas el codo"

            elif (smoothed_shoulder < self.STAGE_UP_THRESHOLD and
                  (lm_shoulder_opp.y - lm_shoulder.y) > self.SHOULDER_SHRUG_THRESHOLD):
                self.feedback = "No encogues el hombro"

            else:
                self.feedback = "Buena postura"

        except Exception:
            pass


class BiomechanicsAnalyzer:
    """
    Clase responsable de calcular métricas biomecánicas genéricas.
    """

    def __init__(self) -> None:
        pass

    def analyze_posture(self, landmarks: List[Any]) -> Dict[str, Any]:
        """
        Analiza un conjunto completo de landmarks para extraer métricas relevantes.
        """
        return {}