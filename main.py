import os
import re
import math
import sys
import time
import random
import multiprocessing as mproc

import cv2
import numpy as np
import pygame
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =====================================================================
#  CONFIGURACIÓN GENERAL
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WIDTH, HEIGHT = 800, 600
FPS = 30
# Posición de las ventanas en pantalla (x, y en píxeles)
GAME_WINDOW_POS = (50, 80)
CAMERA_WINDOW_POS = (GAME_WINDOW_POS[0] + WIDTH + 20, GAME_WINDOW_POS[1])
DRONE_SPEED = 9
PIPE_SPEED = 6
# Dificultad progresiva
DIFFICULTY_START_SCORE = 5   # a partir de este puntaje empieza a subir
DIFFICULTY_STEP = 1          # cada cuántos puntos sube un nivel de dificultad

SPACING_DECREASE = 15        # px que se acorta la distancia entre tubos por nivel
MIN_PIPE_SPACING = 220       # límite mínimo de distancia entre tubos

GAP_DECREASE = 4             # px que se cierra el hueco por nivel
MIN_PIPE_GAP = 150           # límite mínimo del hueco

SPEED_INCREASE = 0.3         # aumento de velocidad por nivel
MAX_PIPE_SPEED = 10          # límite máximo de velocidad

CAMERA_INDEX = 0
MODEL_PATH = os.path.join(BASE_DIR, "models", "pose_landmarker_lite.task")

# =====================================================================
#  CONFIGURACIÓN DEL MENÚ PRINCIPAL
# =====================================================================
# Si pones un archivo 'title_logo.png' con transparencia en la carpeta del juego,
# se usará como logo central. Si no existe, se dibuja el emblema Sci-Fi automático.
TITLE_LOGO_PATH = os.path.join(BASE_DIR, "title_logo.png")
GAME_TITLE = "FLAPPY DRONE"
GAME_SUBTITLE = "SPACE ODYSSEY"

# =====================================================================
#  CONFIGURACIÓN DE LA MÚSICA
# =====================================================================
# Pon tu archivo junto a main.py (o cambia el nombre aquí). Acepta .mp3, .ogg y .wav.
MUSIC_PATH = os.path.join(BASE_DIR, "music.mp3")
MUSIC_VOLUME = 0.5                      # volumen inicial (0.0 a 1.0)
MUSIC_VOLUME_STEP = 0.05                # cuánto cambia con cada pulsación de + / -
# Teclas en el juego:  +  sube el volumen  |  -  lo baja  |  M  silencia / reactiva

# =====================================================================
#  CONFIGURACIÓN DEL PORTAL
# =====================================================================
PORTAL_FIRST_LEVEL = 5               # primer nivel en el que aparece un portal (normal: 10)
PORTAL_EVERY = 5                       # luego aparece cada N niveles (10, 15, 20, 25...)
PORTAL_WIDTH, PORTAL_HEIGHT = 100, 180

# =====================================================================
#  CONFIGURACIÓN DE LA PANTALLA FINAL (nave llegando al planeta)
# =====================================================================
ENDING_FRAMES = 210                    # duración de la animación (210 = ~7 s a 30 FPS)
ENDING_PLANET_RADIUS = 200             # tamaño final del planeta

# =====================================================================
#  CONFIGURACIÓN DE SPACE INVADERS
# =====================================================================
# --- Enemigos ---
# Se cargan 4 estados desde la carpeta `invaders/` (se generan con extract_invaders_assets.py):
#   enemy_idle.png (vuelo), enemy_attack.png (al disparar),
#   enemy_damage.png (impacto) y enemy_destroy.png (explosión).
# Si falta alguno, se dibuja un enemigo verde simple.
INVADER_SPRITES_PATH = os.path.join(BASE_DIR, "invaders")
INVADER_SIZE = (52, 30)                # hitbox (ancho, alto); la nave se escala a este ancho
INVADERS_ROWS, INVADERS_COLS = 4, 8
INVADERS_SPACING = (20, 15)            # separación horizontal y vertical entre enemigos
# OJO: INVADERS_COLS * (ancho + separación) debe ser menor a ~760 px
INVADER_BOB_AMPLITUDE = 2              # cuánto flotan arriba/abajo (px). 0 = quietos
INVADER_BOB_PERIOD_MS = 1000           # duración de un ciclo de flotado
INVADER_ATTACK_MS = 250                # cuánto se ve el sprite de ataque al disparar
INVADER_DAMAGE_MS = 90                 # destello de daño justo antes de explotar
INVADER_DESTROY_MS = 450               # duración de la explosión
INVADER_DESTROY_SCALE = (0.9, 1.8)     # tamaño de la explosión: inicio -> final (x tamaño de la nave)

# --- Proyectiles ---
# Se cargan desde la carpeta `bullets/`: player_fly_0..2.png / player_hit_0..2.png
# y enemy_fly_0..2.png / enemy_hit_0..2.png (vuelo apunta hacia ARRIBA; el del enemigo se voltea solo).
BULLETS_SPRITES_PATH = os.path.join(BASE_DIR, "bullets")
BULLET_ANIM_FPS = 12                   # velocidad de la animación en vuelo
BULLET_HIT_FPS = 14                    # velocidad de la animación de impacto
BULLET_SPRITE_LENGTH = 44              # largo visual del disparo (px), incluida la cola
BULLET_HIT_SIZE = 40                   # tamaño visual del impacto (px)

PLAYER_BULLET_SIZE = (8, 16)           # hitbox del proyectil del jugador (la cabeza del disparo)
PLAYER_BULLET_COLOR = (0, 255, 255)    # color si no hay sprite

ENEMY_BULLET_SIZE = (8, 16)            # hitbox del proyectil enemigo
ENEMY_BULLET_COLOR = (255, 80, 80)     # color si no hay sprite

INVADERS_ENEMY_SPEED = 2               # velocidad horizontal inicial de los enemigos
INVADERS_DROP = 20                     # cuánto bajan al tocar el borde
INVADERS_PLAYER_SPEED = 10
INVADERS_FIRE_DELAY = 400              # ms entre disparos automáticos del jugador
INVADERS_ENEMY_FIRE_DELAY = 900        # ms entre disparos enemigos
INVADERS_BULLET_SPEED = 14
INVADERS_ENEMY_BULLET_SPEED = 6
INVADERS_LIVES = 3

# Gesto: brazo extendido a un lado, a la altura del hombro
INVADERS_ARM_TOLERANCE = 0.10          # qué tan cerca de la altura del hombro (0-1)
INVADERS_ARM_EXTENSION = 1.0           # qué tan separado del cuerpo (en anchos de hombro)
INVADERS_INVERT_CONTROLS = False       # pon True si se mueve al lado contrario

# =====================================================================
#  CONFIGURACIÓN DE LA NAVE (tamaño y modelo 3D opcional)
# =====================================================================
DRONE_SIZE = (70, 58)      # tamaño (ancho, alto) de la nave; también es su hitbox

# Modelo 3D: .obj (sin instalar nada) o .glb/.gltf/.stl/.ply (requiere: pip install trimesh).
# Si el archivo no existe, se usa el sprite 2D (ship.jfif) como antes.
# Prueba y orienta tu modelo con:  python main.py --preview
SHIP_3D_PATH = os.path.join(BASE_DIR, "ship.glb")
SHIP_3D_COLOR = (0, 200, 255)          # color base del modelo (no usa texturas)
SHIP_3D_BASE_ROT = (0, -90, 0)         # rotación (x, y, z) en grados para orientar el modelo
SHIP_3D_TILT_MAX = 25                  # inclinación máxima (grados) al subir/bajar
SHIP_3D_TILT_STEP = 3                  # paso de la caché de inclinación (grados)
SHIP_3D_SUPERSAMPLE = 3                # suavizado de bordes (más = más lento al arrancar)
SHIP_3D_LIGHT = (0.4, 0.6, 0.7)        # dirección de la luz (x, y, z)
SHIP_3D_USE_MATERIALS = True           # True: usa los colores/texturas del modelo; False: solo SHIP_3D_COLOR

# Mundo 1 (Flappy): la nave se ve de PERFIL con la nariz a la derecha (SHIP_3D_BASE_ROT).
# Mundo 2 (Space Invaders): la nave se ve de PLANTA (desde arriba) con la nariz hacia arriba.
# SHIP_3D_TOPVIEW_ROT es la rotación extra que pasa de perfil a planta; (90, 0, 90) funciona
# cuando el perfil está bien orientado. Ajústala con:  python main.py --preview  (tecla T)
INVADERS_SHIP_SIZE = (70, 58)          # tamaño (ancho, alto) de la nave en Space Invaders
SHIP_3D_TOPVIEW_ROT = (90, 0, 90)

# =====================================================================
#  CONFIGURACIÓN DE LOS FONDOS (uno por mundo)
# =====================================================================
# Cada PATH puede ser una CARPETA (varias imágenes; se cambia con la tecla B)
# o un ARCHIVO de imagen. Si no existe, se usa el COLOR de respaldo.
# SCROLL_SPEED: 0 = fondo estático (la imagen se ajusta a la ventana)
#               > 0 = fondo que se desplaza hacia la izquierda

# --- Mundo 1: Flappy ---
WORLD1_BACKGROUND_PATH = os.path.join(BASE_DIR, "backgrounds")
WORLD1_BACKGROUND_COLOR = (20, 24, 40)
WORLD1_BACKGROUND_SCROLL_SPEED = 2

# --- Mundo 2: Space Invaders ---
WORLD2_BACKGROUND_PATH = os.path.join(BASE_DIR, "backgrounds_world2")
WORLD2_BACKGROUND_COLOR = (5, 5, 25)
WORLD2_BACKGROUND_SCROLL_SPEED = 1

# =====================================================================
#  CONFIGURACIÓN DE LOS TUBOS (los "rectángulos" con los que se choca)
# =====================================================================
PIPE_WIDTH = 70            # ancho del tubo
PIPE_GAP = 220             # espacio libre por donde pasa el dron
PIPE_SPACING = 400         # distancia horizontal entre tubos
PIPE_MIN_SEGMENT = 100     # altura mínima de cada tubo
PIPE_HITBOX_PADDING = 0    # >0 hace la colisión más permisiva (en píxeles)

# Tentáculos: PNG con transparencia (tentacle_0.png, tentacle_1.png, ...) en esta carpeta.
# Se generan con:  python extract_tentacles.py
PIPE_SPRITES_PATH = os.path.join(BASE_DIR, "pipes")
TENTACLE_PREFIX = "tentacle_"

# Animación de vaivén: la base queda fija y una onda recorre el tentáculo hasta la punta.
PIPE_ANIM_FPS = 16               # velocidad de la animación (frames por segundo)
TENTACLE_ANIM_FRAMES = 10        # frames por ciclo (más = más suave, pero usa más memoria)
TENTACLE_SWAY_AMPLITUDE = 20     # cuánto se mueve la punta de lado a lado (px). 0 = sin movimiento
TENTACLE_SWAY_WAVES = 2          # cuántas "ondas" caben a lo largo del tentáculo
TENTACLE_SWAY_STRIP = 3          # alto (px) de cada franja al deformar; 3 mantiene el look pixel-art

# Colores de respaldo si no hay sprites
COLOR_PIPE = (0, 255, 100)
COLOR_PIPE_BORDER = (0, 150, 50)

COLOR_DRONE = (0, 200, 255)
COLOR_TEXT = (255, 255, 255)

# =====================================================================
#  LANDMARKS Y GESTOS
# =====================================================================
L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
]

GESTURE_TO_CODE = {"HOVER": 0, "UP": 1, "DOWN": 2}
CODE_TO_GESTURE = {v: k for k, v in GESTURE_TO_CODE.items()}

# Gestos del modo Space Invaders
INV_TO_CODE = {"NONE": 0, "LEFT": 1, "RIGHT": 2}
CODE_TO_INV = {v: k for k, v in INV_TO_CODE.items()}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")


# =====================================================================
#  UTILIDADES PARA CARGAR IMÁGENES / ANIMACIONES
# =====================================================================
def _natural_key(text):
    """Orden natural: pipe_2.png antes que pipe_10.png."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def list_images(path):
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path)
                 if f.lower().endswith(IMAGE_EXTS)]
        return sorted(files, key=_natural_key)
    if os.path.isfile(path):
        return [path]
    return []


def load_animation(path, sheet_frames=1):
    """Devuelve una lista de superficies (frames) desde carpeta, archivo o spritesheet."""
    files = list_images(path)
    if not files:
        return []

    if len(files) == 1 and sheet_frames > 1:
        sheet = pygame.image.load(files[0]).convert_alpha()
        fw = sheet.get_width() // sheet_frames
        return [sheet.subsurface((i * fw, 0, fw, sheet.get_height())).copy()
                for i in range(sheet_frames)]

    frames = []
    for f in files:
        try:
            img = pygame.image.load(f).convert_alpha()
            img.set_colorkey((65, 65, 65))
            frames.append(img)
        except pygame.error as e:
            print(f"ADVERTENCIA: no se pudo cargar '{f}': {e}")
    return frames


# =====================================================================
#  PROCESO DE LA CÁMARA (ventana independiente de OpenCV)
# =====================================================================
def draw_pose(frame, landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 255, 0), 3)
    for p in points:
        cv2.circle(frame, p, 5, (0, 0, 255), -1)


def get_gesture(lm):
    needed = [L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP]
    if any(lm[i].visibility < 0.5 for i in needed):
        return "HOVER"

    shoulder_y = (lm[L_SHOULDER].y + lm[R_SHOULDER].y) / 2
    hip_y = (lm[L_HIP].y + lm[R_HIP].y) / 2

    if lm[L_WRIST].y < shoulder_y and lm[R_WRIST].y < shoulder_y:
        return "UP"
    if lm[L_WRIST].y > hip_y and lm[R_WRIST].y > hip_y:
        return "DOWN"
    return "HOVER"


def get_invaders_gesture(lm):
    """Un brazo extendido a un lado, a la altura del hombro:
    - brazo en el lado derecho de la imagen (vista espejo) -> RIGHT
    - brazo en el lado izquierdo de la imagen              -> LEFT
    Dos brazos extendidos (o ninguno) -> NONE."""
    needed = [L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST]
    if any(lm[i].visibility < 0.5 for i in needed):
        return "NONE"

    center_x = (lm[L_SHOULDER].x + lm[R_SHOULDER].x) / 2
    shoulder_y = (lm[L_SHOULDER].y + lm[R_SHOULDER].y) / 2
    shoulder_w = abs(lm[L_SHOULDER].x - lm[R_SHOULDER].x)

    active = []
    for wrist in (L_WRIST, R_WRIST):
        at_shoulder_height = abs(lm[wrist].y - shoulder_y) < INVADERS_ARM_TOLERANCE
        extended = abs(lm[wrist].x - center_x) > shoulder_w * INVADERS_ARM_EXTENSION
        if at_shoulder_height and extended:
            active.append("RIGHT" if lm[wrist].x > center_x else "LEFT")

    if len(active) == 1:
        return active[0]
    return "NONE"


def camera_process(gesture_value, invaders_value, stop_event):
    """Corre en un proceso aparte: lee la cámara, detecta la pose y
    publica los gestos en memoria compartida para el juego."""
    if not os.path.exists(MODEL_PATH):
        print(f"[Cámara] Error: no se encontró el modelo en {MODEL_PATH}")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    GAME_WINDOW_POS = (50, 80)
    CAMERA_WINDOW_POS = (GAME_WINDOW_POS[0] + WIDTH + 20, GAME_WINDOW_POS[1])
    if not cap.isOpened():
        print("[Cámara] Error: no se pudo abrir la cámara. Usa las flechas del teclado.")
        return

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    window_name = "Camara - Body Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_name, *CAMERA_WINDOW_POS)
    start_time = time.time()
    last_ts = -1

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # MediaPipe exige timestamps estrictamente crecientes
            ts = int((time.time() - start_time) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts

            result = landmarker.detect_for_video(mp_image, ts)

            if result.pose_landmarks:
                pose = result.pose_landmarks[0]
                gesture = get_gesture(pose)
                inv_gesture = get_invaders_gesture(pose)
                draw_pose(frame, pose)
            else:
                gesture = "HOVER"
                inv_gesture = "NONE"

            gesture_value.value = GESTURE_TO_CODE[gesture]
            invaders_value.value = INV_TO_CODE[inv_gesture]

            cv2.putText(frame, f"Gesto: {gesture}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Invaders: {inv_gesture}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            cv2.putText(frame, "Q: cerrar camara", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    gesture_value.value = GESTURE_TO_CODE["HOVER"]
    invaders_value.value = INV_TO_CODE["NONE"]
    cap.release()
    cv2.destroyAllWindows()


# =====================================================================
#  MÚSICA
# =====================================================================
class MusicPlayer:
    """Música de fondo en bucle con control de volumen (+ / -) y silencio (M)."""

    def __init__(self):
        self.enabled = False
        self.volume = max(0.0, min(1.0, MUSIC_VOLUME))
        self.muted = False
        self.hud_until = 0

    def start(self):
        """Inicia la música. Llamar después de pygame.init()."""
        if not MUSIC_PATH or not os.path.isfile(MUSIC_PATH):
            print(f"AVISO: no se encontró la música '{MUSIC_PATH}'. El juego seguirá sin música.")
            return
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(MUSIC_PATH)
            pygame.mixer.music.set_volume(self.volume)
            pygame.mixer.music.play(-1)                      # -1 = en bucle
            self.enabled = True
        except pygame.error as e:
            print(f"ADVERTENCIA: no se pudo reproducir la música: {e}")

    def _apply(self):
        pygame.mixer.music.set_volume(0.0 if self.muted else self.volume)
        self.hud_until = pygame.time.get_ticks() + 1500      # muestra el indicador 1.5 s

    def change_volume(self, delta):
        self.volume = max(0.0, min(1.0, round(self.volume + delta, 2)))
        if delta > 0:
            self.muted = False                               # subir el volumen quita el silencio
        self._apply()

    def toggle_mute(self):
        self.muted = not self.muted
        self._apply()

    def handle_event(self, event):
        if not self.enabled or event.type != pygame.KEYDOWN:
            return
        if event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
            self.change_volume(MUSIC_VOLUME_STEP)
        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.change_volume(-MUSIC_VOLUME_STEP)
        elif event.key == pygame.K_m:
            self.toggle_mute()

    def draw(self, surface, font):
        """Indicador de volumen arriba a la derecha (aparece unos segundos al cambiarlo)."""
        if not self.enabled or pygame.time.get_ticks() > self.hud_until:
            return
        label = "Música: silenciada" if self.muted else f"Volumen: {round(self.volume * 100)}%"
        text = font.render(label, True, COLOR_TEXT)
        x = WIDTH - text.get_width() - 20
        surface.blit(text, (x, 15))
        bar = pygame.Rect(x, 15 + text.get_height() + 4, text.get_width(), 8)
        pygame.draw.rect(surface, (60, 60, 80), bar)
        fill = 0 if self.muted else int(bar.width * self.volume)
        pygame.draw.rect(surface, (255, 200, 0), (bar.x, bar.y, fill, bar.height))
        pygame.draw.rect(surface, COLOR_TEXT, bar, 1)


music = MusicPlayer()


# =====================================================================
#  CLASES DEL JUEGO
# =====================================================================
class Background:
    def __init__(self, path, color, scroll_speed):
        self.color = color
        self.scroll_speed = scroll_speed
        self.files = list_images(path)
        self.index = 0
        self.offset = 0
        self.image = None
        self._load()

    def _load(self):
        self.image = None
        if not self.files:
            return
        try:
            img = pygame.image.load(self.files[self.index]).convert()
            if self.scroll_speed == 0:
                self.image = pygame.transform.smoothscale(img, (WIDTH, HEIGHT))
            else:
                scale = HEIGHT / img.get_height()
                size = (max(1, int(img.get_width() * scale)), HEIGHT)
                self.image = pygame.transform.smoothscale(img, size)
        except pygame.error as e:
            print(f"ADVERTENCIA: no se pudo cargar el fondo: {e}")

    def next(self):
        if len(self.files) > 1:
            self.index = (self.index + 1) % len(self.files)
            self.offset = 0
            self._load()

    def update(self):
        self.offset += self.scroll_speed

    def draw(self, surface):
        if self.image is None:
            surface.fill(self.color)
            return
        w = self.image.get_width()
        x = -(self.offset % w) if self.scroll_speed else 0
        while x < WIDTH:
            surface.blit(self.image, (x, 0))
            x += w


def get_difficulty(score):
    level = max(0, score - DIFFICULTY_START_SCORE) // DIFFICULTY_STEP
    spacing = max(MIN_PIPE_SPACING, PIPE_SPACING - level * SPACING_DECREASE)
    gap = max(MIN_PIPE_GAP, PIPE_GAP - level * GAP_DECREASE)
    speed = min(MAX_PIPE_SPEED, PIPE_SPEED + level * SPEED_INCREASE)
    return level, spacing, gap, speed


def _rot_matrix(rx, ry, rz):
    """Matriz de rotación a partir de ángulos en grados (x, y, z)."""
    ax, ay, az = np.radians([rx, ry, rz])
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def _load_texture(value, base_dir):
    """Carga la textura de un 'map_Kd' como arreglo (alto, ancho, 3). None si falla."""
    value = value.strip()
    names = [value, value.split()[-1]] if value else []
    for name in names:
        name = name.replace("\\", "/")
        for candidate in (os.path.join(base_dir, name), os.path.join(base_dir, os.path.basename(name))):
            if os.path.isfile(candidate):
                try:
                    surf = pygame.image.load(candidate)
                    return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2)).astype(float)
                except pygame.error as e:
                    print(f"ADVERTENCIA: no se pudo leer la textura '{candidate}': {e}")
                    return None
    print(f"ADVERTENCIA: no se encontró la textura '{value}' (ponla junto al .obj)")
    return None


def _parse_mtl(mtl_path):
    """Lee un .mtl. Devuelve {nombre: {'kd': (r,g,b) 0-1 o None, 'texture': arreglo o None}}."""
    materials, current = {}, None
    base_dir = os.path.dirname(mtl_path)
    with open(mtl_path, encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            key, val = parts[0].lower(), (parts[1] if len(parts) > 1 else "")
            if key == "newmtl":
                current = {"kd": None, "texture": None}
                materials[val.strip()] = current
            elif current is None:
                continue
            elif key == "kd":
                nums = [float(x) for x in val.split()[:3]]
                if len(nums) == 3:
                    current["kd"] = tuple(nums)
            elif key == "map_kd":
                current["texture"] = _load_texture(val, base_dir)
    return materials


def _kd_to_rgb(kd):
    """Color difuso (0-1) -> RGB 0-255."""
    return np.clip(np.asarray(kd, dtype=float) * 255.0, 0, 255)


def _sample_texture(tex, uv):
    """Color promedio de cada cara: 4 muestras de la textura dentro del triángulo.
    uv: (k, 3, 2) -> devuelve (k, 3) en RGB 0-255."""
    weights = np.array([[1 / 3, 1 / 3, 1 / 3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
    pts = np.einsum("pv,kvc->kpc", weights, uv)               # (k, 4, 2)
    h, w = tex.shape[:2]
    # Las UV fuera de 0-1 se repiten (mosaico), pero 1.0 exacto es el borde y no debe dar la vuelta a 0
    pts = np.where((pts >= 0) & (pts <= 1), pts, pts % 1.0)
    x = (pts[..., 0] * (w - 1)).astype(int)
    y = ((1 - pts[..., 1]) * (h - 1)).astype(int)
    return tex[y, x].mean(axis=1)


def _load_obj(path):
    base_dir = os.path.dirname(path)
    verts, vcols, uvs = [], [], []
    tris, tri_uv, tri_mat = [], [], []
    materials, current_mat = {}, ""

    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
                if len(p) >= 7:                                # color por vértice (opcional)
                    vcols.append([float(p[4]), float(p[5]), float(p[6])])
            elif line.startswith("vt "):
                p = line.split()
                uvs.append([float(p[1]), float(p[2])])
            elif line.startswith("mtllib"):
                mtl_path = os.path.join(base_dir, line[6:].strip().replace("\\", "/"))
                if os.path.isfile(mtl_path):
                    materials.update(_parse_mtl(mtl_path))
                else:
                    print(f"ADVERTENCIA: no se encontró '{os.path.basename(mtl_path)}' "
                          "(ponlo junto al .obj para conservar los materiales)")
            elif line.startswith("usemtl"):
                current_mat = line[6:].strip()
            elif line.startswith("f "):
                idx, uv_idx = [], []
                for tok in line.split()[1:]:
                    parts = tok.split("/")
                    i = int(parts[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                    if len(parts) > 1 and parts[1]:
                        t = int(parts[1])
                        uv_idx.append(t - 1 if t > 0 else len(uvs) + t)
                    else:
                        uv_idx.append(-1)
                for k in range(1, len(idx) - 1):               # triangula polígonos
                    tris.append((idx[0], idx[k], idx[k + 1]))
                    tri_uv.append((uv_idx[0], uv_idx[k], uv_idx[k + 1]))
                    tri_mat.append(current_mat)

    verts = np.array(verts, dtype=float)
    faces = np.array(tris, dtype=int)
    if len(faces) == 0:
        return verts, faces, None

    colors = np.full((len(faces), 3), np.nan)                  # NaN = usar SHIP_3D_COLOR
    tri_mat = np.array(tri_mat, dtype=object)
    tri_uv = np.array(tri_uv, dtype=int)

    # 1) Color difuso (Kd) de cada material
    for name, mat in materials.items():
        sel = np.where(tri_mat == name)[0]
        if len(sel) and mat["kd"] is not None:
            colors[sel] = _kd_to_rgb(mat["kd"])

    # 2) Colores por vértice, si el archivo los trae
    if len(vcols) == len(verts) and len(vcols):
        colors[:] = np.array(vcols)[faces].mean(axis=1) * 255

    # 3) Texturas (map_Kd): color promedio por cara según sus coordenadas UV
    if uvs:
        uv_arr = np.array(uvs, dtype=float)
        for name, mat in materials.items():
            if mat["texture"] is None:
                continue
            sel = np.where(tri_mat == name)[0]
            sel = sel[(tri_uv[sel] >= 0).all(axis=1) & (tri_uv[sel] < len(uv_arr)).all(axis=1)]
            if len(sel):
                colors[sel] = _sample_texture(mat["texture"], uv_arr[tri_uv[sel]])

    return verts, faces, (None if np.isnan(colors).all() else colors)


def _trimesh_face_colors(g):
    """Color por cara (F, 3) de una pieza de trimesh: textura, color base del material o colores
    propios. None si la pieza no trae color."""
    vis = g.visual
    faces = np.asarray(g.faces, dtype=int)

    if getattr(vis, "kind", None) == "texture":
        mat = getattr(vis, "material", None)
        uv = getattr(vis, "uv", None)

        # Textura de color base: color promedio de cada cara según sus UV
        tex = getattr(mat, "baseColorTexture", None)
        if tex is None:
            tex = getattr(mat, "image", None)
        if tex is not None and uv is not None and len(uv) == len(g.vertices):
            arr = np.asarray(tex.convert("RGB"), dtype=float)        # (alto, ancho, 3)
            return _sample_texture(arr, np.asarray(uv, dtype=float)[faces])

        # Material sin textura: color base (glTF) o difuso
        factor = getattr(mat, "baseColorFactor", None)
        if factor is None:
            factor = getattr(mat, "diffuse", None)
        if factor is not None:
            rgb = np.asarray(factor, dtype=float).ravel()[:3]
            if rgb.max() > 1:
                rgb = rgb / 255.0
            return np.tile(_kd_to_rgb(rgb), (len(faces), 1))
        return None

    if getattr(vis, "defined", False):                              # colores por cara / vértice
        return np.asarray(vis.face_colors)[:, :3].astype(float)
    return None


def _load_with_trimesh(path):
    """Formatos .glb/.gltf/.stl/.ply (requiere trimesh). Conserva colores y texturas base."""
    try:
        import trimesh
    except ImportError:
        raise RuntimeError("para este formato instala trimesh:  pip install trimesh")

    loaded = trimesh.load(path)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.dump() if hasattr(g, "faces")]   # con sus transformaciones
    else:
        geoms = [loaded]

    all_v, all_f, all_c = [], [], []
    offset = 0
    for g in geoms:
        v = np.asarray(g.vertices, dtype=float)
        f = np.asarray(g.faces, dtype=int)
        c = np.full((len(f), 3), np.nan)
        try:
            fc = _trimesh_face_colors(g)
            if fc is not None:
                c = fc
        except Exception as e:
            print(f"ADVERTENCIA: no se pudieron leer los colores de una pieza: {e}")
        all_v.append(v)
        all_f.append(f + offset)
        all_c.append(c)
        offset += len(v)

    colors = np.vstack(all_c)
    return np.vstack(all_v), np.vstack(all_f), (None if np.isnan(colors).all() else colors)


def load_mesh(path):
    """Carga un modelo 3D. Devuelve (vértices Nx3, triángulos Mx3, colores por cara Mx3 o None)."""
    if path.lower().endswith(".obj"):
        return _load_obj(path)
    return _load_with_trimesh(path)


class Ship3D:
    """Renderizador 3D por software (proyección ortográfica, sombreado plano).
    Genera superficies de pygame y las guarda en caché por ángulo de inclinación."""

    def __init__(self, path, size, color, base_rot, extra_rot=(0, 0, 0), tilt_range=None, mesh=None):
        self.size = size
        self.color = np.array(color, dtype=float)
        self.extra_rot = tuple(extra_rot)
        self.tilt_range = SHIP_3D_TILT_MAX if tilt_range is None else tilt_range
        self.mesh = mesh if mesh is not None else load_mesh(path)   # se puede reutilizar
        verts, self.faces, face_colors = self.mesh
        if len(verts) == 0 or len(self.faces) == 0:
            raise RuntimeError("el modelo no tiene geometría")
        self.verts = verts - (verts.max(axis=0) + verts.min(axis=0)) / 2
        if len(self.faces) > 5000:
            print(f"AVISO: el modelo tiene {len(self.faces)} caras; con más de ~5000 "
                  "el arranque puede tardar. Simplifícalo en Blender (Decimate).")

        # Colores por cara: los del material, o SHIP_3D_COLOR donde no haya
        self.face_colors = None
        if SHIP_3D_USE_MATERIALS and face_colors is not None:
            fc = face_colors.copy()
            fc[np.isnan(fc).any(axis=1)] = self.color
            self.face_colors = fc
        origen = "materiales del modelo" if self.face_colors is not None else "color único (SHIP_3D_COLOR)"
        if mesh is None:
            print(f"[Nave 3D] {len(self.faces)} caras | colores: {origen}")

        self.cache = {}
        self.set_base_rot(*base_rot)

    def set_base_rot(self, rx, ry, rz, extra_rot=None):
        """Orientación del modelo: rotación base (perfil) y, opcionalmente, una rotación extra."""
        if extra_rot is not None:
            self.extra_rot = tuple(extra_rot)
        self.base_rot = (rx, ry, rz)
        self.base = _rot_matrix(*self.extra_rot) @ _rot_matrix(rx, ry, rz)
        self.cache.clear()

        # Escala fija para que el modelo quepa también inclinado al máximo
        half_x = half_y = 1e-9
        for tilt in (-self.tilt_range, 0, self.tilt_range):
            v = self.verts @ (_rot_matrix(0, 0, tilt) @ self.base).T
            half_x = max(half_x, (v[:, 0].max() - v[:, 0].min()) / 2)
            half_y = max(half_y, (v[:, 1].max() - v[:, 1].min()) / 2)
        w, h = self.size[0] * SHIP_3D_SUPERSAMPLE, self.size[1] * SHIP_3D_SUPERSAMPLE
        self.scale = min((w / 2) / half_x, (h / 2) / half_y) * 0.95

    def surface(self, tilt=0.0):
        step = max(1, SHIP_3D_TILT_STEP)
        key = int(round(tilt / step)) * step
        if key not in self.cache:
            self.cache[key] = self._render(key)
        return self.cache[key]

    def _render(self, tilt):
        v = self.verts @ (_rot_matrix(0, 0, tilt) @ self.base).T
        v = v - (v.max(axis=0) + v.min(axis=0)) / 2
        tri = v[self.faces]                                  # (F, 3, 3)

        # Normales, orientadas hacia la cámara (iluminación de dos caras)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(n, axis=1, keepdims=True)
        length[length == 0] = 1
        n = n / length
        n = n * np.where(n[:, 2:3] < 0, -1, 1)

        light = np.array(SHIP_3D_LIGHT, dtype=float)
        light /= np.linalg.norm(light)
        intensity = 0.35 + 0.65 * np.clip(n @ light, 0, 1)
        base = self.face_colors if self.face_colors is not None else self.color
        colors = np.clip(base * intensity[:, None], 0, 255).astype(int).tolist()

        # Algoritmo del pintor: de lo más lejano a lo más cercano
        order = np.argsort(tri[:, :, 2].mean(axis=1))
        w, h = self.size[0] * SHIP_3D_SUPERSAMPLE, self.size[1] * SHIP_3D_SUPERSAMPLE
        sx = (tri[:, :, 0] * self.scale + w / 2).tolist()
        sy = (-tri[:, :, 1] * self.scale + h / 2).tolist()

        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        for i in order:
            pts = [(sx[i][0], sy[i][0]), (sx[i][1], sy[i][1]), (sx[i][2], sy[i][2])]
            c = tuple(colors[i])
            pygame.draw.polygon(surf, c, pts)
            pygame.draw.polygon(surf, c, pts, 1)              # cierra grietas entre caras
        return pygame.transform.smoothscale(surf, self.size)


_SHIP_MODEL = None
_SHIP_MODEL_TRIED = False


def get_ship_model():
    """Devuelve el modelo 3D de la nave (se carga una sola vez) o None si no hay."""
    global _SHIP_MODEL, _SHIP_MODEL_TRIED
    if _SHIP_MODEL_TRIED:
        return _SHIP_MODEL
    _SHIP_MODEL_TRIED = True
    if not SHIP_3D_PATH or not os.path.isfile(SHIP_3D_PATH):
        return None
    try:
        _SHIP_MODEL = Ship3D(SHIP_3D_PATH, DRONE_SIZE, SHIP_3D_COLOR, SHIP_3D_BASE_ROT)
    except Exception as e:
        print(f"ADVERTENCIA: no se pudo cargar el modelo 3D '{SHIP_3D_PATH}': {e}")
    return _SHIP_MODEL


_SHIP_TOP = None
_SHIP_TOP_TRIED = False


def get_ship_topview():
    """Modelo 3D de la nave visto de planta (nariz hacia arriba), o None si no hay modelo."""
    global _SHIP_TOP, _SHIP_TOP_TRIED
    if _SHIP_TOP_TRIED:
        return _SHIP_TOP
    _SHIP_TOP_TRIED = True
    side = get_ship_model()
    if side is None:
        return None
    try:
        _SHIP_TOP = Ship3D(SHIP_3D_PATH, INVADERS_SHIP_SIZE, SHIP_3D_COLOR, SHIP_3D_BASE_ROT,
                           extra_rot=SHIP_3D_TOPVIEW_ROT, tilt_range=0, mesh=side.mesh)
    except Exception as e:
        print(f"ADVERTENCIA: no se pudo crear la vista de planta de la nave: {e}")
    return _SHIP_TOP


def get_invaders_ship_image(drone):
    """Imagen de la nave para Space Invaders: de planta si hay modelo 3D; si no, el sprite 2D."""
    top = get_ship_topview()
    return top.surface(0) if top else drone.image


class Drone:
    def __init__(self):
        self.width, self.height = DRONE_SIZE
        self.x = WIDTH // 4
        self.y = HEIGHT // 2
        self.tilt = 0.0
        self._last_y = self.y

        self.model = get_ship_model()
        if self.model:
            # Modelo 3D: esta imagen (sin inclinar) también se usa en Invaders y el final
            self.image = self.model.surface(0)
        else:
            sprite_path = os.path.join(BASE_DIR, "ship.jfif")
            try:
                self.image = pygame.image.load(sprite_path).convert_alpha()
                self.image = pygame.transform.scale(self.image, (self.width, self.height))
            except (FileNotFoundError, pygame.error):
                print(f"ADVERTENCIA: no se encontró '{sprite_path}'. Usando cuadrado por defecto.")
                self.image = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                pygame.draw.rect(self.image, COLOR_DRONE, (0, 0, self.width, self.height), border_radius=8)
                pygame.draw.rect(self.image, (255, 255, 255), (0, 0, self.width, self.height), 2, border_radius=8)

        self.rect = self.image.get_rect()
        self.rect.x = self.x
        self.rect.y = self.y

    def move(self, direction):
        if direction == "UP":
            self.y -= DRONE_SPEED
        elif direction == "DOWN":
            self.y += DRONE_SPEED
        self.y = max(0, min(HEIGHT - self.height, self.y))
        self.rect.y = self.y

    def draw(self, surface):
        if self.model:
            # La nave se inclina según su movimiento vertical (arriba = nariz arriba)
            dy = self.y - self._last_y
            self._last_y = self.y
            target = -dy * SHIP_3D_TILT_MAX / max(1, DRONE_SPEED)
            target = max(-SHIP_3D_TILT_MAX, min(SHIP_3D_TILT_MAX, target))
            self.tilt += (target - self.tilt) * 0.3
            surface.blit(self.model.surface(self.tilt), self.rect)
        else:
            surface.blit(self.image, self.rect)


class TentacleAnim:
    """Frames animados de un tentáculo, ya listos para dibujar y para colisionar.
    `bottom_*`: base abajo y punta hacia arriba. `top_*`: lo mismo pero invertido."""

    def __init__(self, sprite):
        w, h = sprite.get_size()
        n = max(1, TENTACLE_ANIM_FRAMES) if TENTACLE_SWAY_AMPLITUDE > 0 else 1
        pad = int(TENTACLE_SWAY_AMPLITUDE) + 1
        self.bottom, self.top = [], []
        for f in range(n):
            frame = pygame.Surface((w + 2 * pad, h), pygame.SRCALPHA)
            for y0 in range(0, h, TENTACLE_SWAY_STRIP):
                sh = min(TENTACLE_SWAY_STRIP, h - y0)
                u = 1 - (y0 + sh / 2) / h                   # 0 en la base, 1 en la punta
                phase = 2 * math.pi * (f / n - TENTACLE_SWAY_WAVES * u)
                dx = round(TENTACLE_SWAY_AMPLITUDE * (u ** 1.5) * math.sin(phase))
                frame.blit(sprite, (pad + dx, y0), (0, y0, w, sh))
            self.bottom.append(frame)
            self.top.append(pygame.transform.flip(frame, False, True))
        self.bottom_masks = [pygame.mask.from_surface(f) for f in self.bottom]
        self.top_masks = [pygame.mask.from_surface(f) for f in self.top]
        self.n = n


def load_tentacles(path):
    """Carga los sprites tentacle_*.png (con transparencia) y pre-genera su animación."""
    anims = []
    for f in list_images(path):
        if not os.path.basename(f).lower().startswith(TENTACLE_PREFIX):
            continue
        try:
            anims.append(TentacleAnim(pygame.image.load(f).convert_alpha()))
        except pygame.error as e:
            print(f"ADVERTENCIA: no se pudo cargar '{f}': {e}")
    return anims


class Pipe:
    """Par de tentáculos (arriba y abajo). La punta de cada uno marca el borde del hueco
    y la base sale por el borde de la pantalla. La colisión usa la forma real del frame actual."""

    def __init__(self, frames, gap=PIPE_GAP):
        self.x = WIDTH
        self.passed = False
        self.bottom_height = random.randint(PIPE_MIN_SEGMENT, HEIGHT - gap - PIPE_MIN_SEGMENT)
        self.top_height = HEIGHT - gap - self.bottom_height

        # Rectángulos de respaldo (solo se usan si no hay sprites)
        self.rect_top = pygame.Rect(self.x, 0, PIPE_WIDTH, self.top_height)
        self.rect_bottom = pygame.Rect(self.x, HEIGHT - self.bottom_height, PIPE_WIDTH, self.bottom_height)

        self.bottom_anim = self.top_anim = None
        self.anim_t = 0.0
        if frames:
            # Dos tentáculos distintos cuando se puede
            self.bottom_anim, self.top_anim = (random.sample(frames, 2) if len(frames) >= 2
                                               else (frames[0], frames[0]))
            # Desfase aleatorio para que no se muevan todos al mismo tiempo
            self.bottom_offset = random.randrange(self.bottom_anim.n)
            self.top_offset = random.randrange(self.top_anim.n)
        self._update_positions()

    def _bottom_index(self):
        return (int(self.anim_t) + self.bottom_offset) % self.bottom_anim.n

    def _top_index(self):
        return (int(self.anim_t) + self.top_offset) % self.top_anim.n

    def _update_positions(self):
        self.rect_top.x = int(self.x)
        self.rect_bottom.x = int(self.x)
        if self.bottom_anim:
            b = self.bottom_anim.bottom[0]
            t = self.top_anim.top[0]
            cx = self.x + PIPE_WIDTH / 2           # los tentáculos se centran en la columna
            self.bottom_pos = (int(cx - b.get_width() / 2), HEIGHT - self.bottom_height)
            self.top_pos = (int(cx - t.get_width() / 2), self.top_height - t.get_height())

    def move(self, speed=PIPE_SPEED):
        self.x -= speed
        self.anim_t += PIPE_ANIM_FPS / FPS         # la animación avanza con el juego
        self._update_positions()

    def collides(self, rect):
        pad = PIPE_HITBOX_PADDING * 2
        if not self.bottom_anim:
            return (rect.colliderect(self.rect_top.inflate(-pad, -pad)) or
                    rect.colliderect(self.rect_bottom.inflate(-pad, -pad)))
        r = rect.inflate(-pad, -pad)
        probe = pygame.Mask(r.size, fill=True)
        for mask, pos in ((self.top_anim.top_masks[self._top_index()], self.top_pos),
                          (self.bottom_anim.bottom_masks[self._bottom_index()], self.bottom_pos)):
            if mask.overlap(probe, (r.x - pos[0], r.y - pos[1])):
                return True
        return False

    def draw(self, surface, frame_index=0):
        if self.bottom_anim:
            surface.blit(self.top_anim.top[self._top_index()], self.top_pos)
            surface.blit(self.bottom_anim.bottom[self._bottom_index()], self.bottom_pos)
        else:
            pygame.draw.rect(surface, COLOR_PIPE, self.rect_top)
            pygame.draw.rect(surface, COLOR_PIPE, self.rect_bottom)
            pygame.draw.rect(surface, COLOR_PIPE_BORDER, self.rect_top, 3)
            pygame.draw.rect(surface, COLOR_PIPE_BORDER, self.rect_bottom, 3)


class Portal:
    """Portal que aparece cada PORTAL_EVERY niveles desde PORTAL_FIRST_LEVEL.
    Al tocarlo se entra a Space Invaders."""

    def __init__(self):
        self.x = WIDTH
        self.passed = True
        cy = random.randint(PORTAL_HEIGHT // 2 + 40, HEIGHT - PORTAL_HEIGHT // 2 - 40)
        self.rect = pygame.Rect(self.x, cy - PORTAL_HEIGHT // 2, PORTAL_WIDTH, PORTAL_HEIGHT)

    def move(self, speed=PIPE_SPEED):
        self.x -= speed
        self.rect.x = int(self.x)

    def touches(self, rect):
        return self.rect.inflate(-30, -30).colliderect(rect)

    def draw(self, surface, frame_index=0):
        t = pygame.time.get_ticks() / 1000
        center = self.rect.center

        # Centro oscuro
        pygame.draw.ellipse(surface, (15, 0, 40), self.rect)

        # Anillos giratorios
        colors = [(160, 90, 255), (0, 220, 255)]
        for i in range(6):
            s = 1 - i * 0.14
            ring = pygame.Rect(0, 0, int(self.rect.width * s), int(self.rect.height * s))
            ring.center = center
            start = t * (2 + i * 0.5) + i
            pygame.draw.arc(surface, colors[i % 2], ring, start, start + 3.8, 4)

        pygame.draw.ellipse(surface, (200, 160, 255), self.rect, 3)


# =====================================================================
#  TRANSICIÓN Y MINIJUEGO SPACE INVADERS
# =====================================================================
def load_scaled_frames(path, size, sheet_frames=1):
    """Carga un sprite/animación y lo escala a `size`. Devuelve [] si no existe."""
    return [pygame.transform.scale(f, size) for f in load_animation(path, sheet_frames)]


def pick_frame(frames, now, fps):
    return frames[int(now / 1000 * fps) % len(frames)]


def load_invader_sprites():
    """Carga los 4 estados del enemigo y los escala a INVADER_SIZE.
    Devuelve None si falta algún archivo (entonces se dibuja un enemigo simple)."""
    imgs = {}
    try:
        for state in ("idle", "attack", "damage", "destroy"):
            f = os.path.join(INVADER_SPRITES_PATH, f"enemy_{state}.png")
            if not os.path.isfile(f):
                return None
            imgs[state] = pygame.image.load(f).convert_alpha()
    except pygame.error as e:
        print(f"ADVERTENCIA: no se pudieron cargar los sprites del enemigo: {e}")
        return None
    body = imgs["idle"].get_bounding_rect()
    k = INVADER_SIZE[0] / body.w
    for state, img in imgs.items():
        size = (max(1, round(img.get_width() * k)), max(1, round(img.get_height() * k)))
        imgs[state] = pygame.transform.smoothscale(img, size)
    body = imgs["idle"].get_bounding_rect()          # la nave en reposo es la referencia de alineación
    return {"frames": imgs, "offset": body.topleft, "body_center": body.center}


def load_bullet_sprites():
    """Disparos del jugador y del enemigo: {'player'|'enemy': {'fly': [...], 'hit': [...]}} o None."""
    sets = {}
    try:
        for owner in ("player", "enemy"):
            entry = {}
            for kind, length in (("fly", BULLET_SPRITE_LENGTH), ("hit", BULLET_HIT_SIZE)):
                frames = []
                for i in range(12):
                    f = os.path.join(BULLETS_SPRITES_PATH, f"{owner}_{kind}_{i}.png")
                    if not os.path.isfile(f):
                        break
                    img = pygame.image.load(f).convert_alpha()
                    k = length / max(img.get_size())
                    img = pygame.transform.smoothscale(
                        img, (max(1, round(img.get_width() * k)), max(1, round(img.get_height() * k))))
                    if owner == "enemy" and kind == "fly":
                        img = pygame.transform.flip(img, False, True)   # cae hacia abajo
                    frames.append(img)
                entry[kind] = frames
            sets[owner] = entry if entry["fly"] and entry["hit"] else None
    except pygame.error as e:
        print(f"ADVERTENCIA: no se pudieron cargar los sprites de disparos: {e}")
        return None
    return sets


def draw_effects(screen, effects, now, inv_sprites, bullet_sprites):
    """Dibuja impactos de disparo y explosiones de enemigos. Devuelve los efectos que siguen activos."""
    alive = []
    for fx in effects:
        t = now - fx["start"]
        if fx["type"] == "hit":
            frames = bullet_sprites[fx["owner"]]["hit"]
            i = int(t / 1000 * BULLET_HIT_FPS)
            if i >= len(frames):
                continue
            img = frames[i]
            screen.blit(img, img.get_rect(center=fx["pos"]))
        else:  # "die": destello de daño y luego explosión que crece y se desvanece
            r = fx["rect"]
            frames = inv_sprites["frames"]
            if t < INVADER_DAMAGE_MS:
                ox, oy = inv_sprites["offset"]
                screen.blit(frames["damage"], (r.x - ox, r.y - oy))
            elif t < INVADER_DAMAGE_MS + INVADER_DESTROY_MS:
                p = (t - INVADER_DAMAGE_MS) / INVADER_DESTROY_MS
                k = INVADER_DESTROY_SCALE[0] + (INVADER_DESTROY_SCALE[1] - INVADER_DESTROY_SCALE[0]) * p
                src = frames["destroy"]
                img = pygame.transform.smoothscale(
                    src, (max(1, int(src.get_width() * k)), max(1, int(src.get_height() * k))))
                img.set_alpha(int(255 * (1 - p) ** 0.7))
                cx, cy = inv_sprites["body_center"]
                screen.blit(img, (r.centerx - int(cx * k), r.centery - int(cy * k)))
            else:
                continue
        alive.append(fx)
    return alive


def portal_transition(screen, clock, center):
    """Círculo que se expande desde el portal y cubre la pantalla."""
    snapshot = screen.copy()
    max_r = int((WIDTH ** 2 + HEIGHT ** 2) ** 0.5)
    for r in range(0, max_r, 40):
        screen.blit(snapshot, (0, 0))
        pygame.draw.circle(screen, (120, 60, 255), center, r)
        pygame.display.flip()
        pygame.event.pump()
        clock.tick(FPS)


def run_invaders(screen, clock, background, invaders_value, shutdown, player_image, fonts):
    """Space Invaders controlado con el brazo. Devuelve 'win' o 'lose'."""
    font_large, font_small = fonts
    inv_w, inv_h = INVADER_SIZE

    # Sprites (opcionales): enemigo y proyectiles. Si no existen, se dibujan formas simples
    inv_sprites = load_invader_sprites()
    bullet_sprites = load_bullet_sprites()
    if bullet_sprites and not (bullet_sprites["player"] and bullet_sprites["enemy"]):
        bullet_sprites = None
    enemy_fallback = None
    if not inv_sprites:
        enemy_fallback = pygame.Surface(INVADER_SIZE, pygame.SRCALPHA)
        pygame.draw.ellipse(enemy_fallback, (120, 255, 120), (0, 0, inv_w, inv_h))
        pygame.draw.circle(enemy_fallback, (0, 0, 0), (int(inv_w * 0.3), int(inv_h * 0.45)), 4)
        pygame.draw.circle(enemy_fallback, (0, 0, 0), (int(inv_w * 0.7), int(inv_h * 0.45)), 4)

    pb_w, pb_h = PLAYER_BULLET_SIZE
    eb_w, eb_h = ENEMY_BULLET_SIZE
    effects = []                 # impactos y explosiones en curso
    enemy_attack_until = {}      # id(enemigo) -> momento hasta el que muestra el sprite de ataque

    # Enemigos
    enemies = []
    gap_x, gap_y = INVADERS_SPACING
    grid_w = INVADERS_COLS * inv_w + (INVADERS_COLS - 1) * gap_x
    start_x = (WIDTH - grid_w) // 2
    for row in range(INVADERS_ROWS):
        for col in range(INVADERS_COLS):
            enemies.append(pygame.Rect(start_x + col * (inv_w + gap_x), 60 + row * (inv_h + gap_y), inv_w, inv_h))
    total_enemies = len(enemies)
    enemy_phase = {id(e): random.uniform(0, 2 * math.pi) for e in enemies}
    direction = 1
    move_acc = 0.0

    # Jugador
    player = player_image.get_rect(midbottom=(WIDTH // 2, HEIGHT - 20))
    player_bullets = []
    enemy_bullets = []
    lives = INVADERS_LIVES
    score = 0
    last_shot = 0
    last_enemy_shot = pygame.time.get_ticks()
    invulnerable_until = 0

    state = "playing"
    end_time = 0

    while True:
        for event in pygame.event.get():
            music.handle_event(event)
            if event.type == pygame.QUIT:
                shutdown()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                shutdown()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_b:
                background.next()

        now = pygame.time.get_ticks()

        if state == "playing":
            # --- Control: brazo (cámara) o flechas (respaldo) ---
            gesture = CODE_TO_INV.get(invaders_value.value, "NONE")
            if INVADERS_INVERT_CONTROLS:
                gesture = {"LEFT": "RIGHT", "RIGHT": "LEFT"}.get(gesture, gesture)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                gesture = "LEFT"
            elif keys[pygame.K_RIGHT]:
                gesture = "RIGHT"

            if gesture == "LEFT":
                player.x -= INVADERS_PLAYER_SPEED
            elif gesture == "RIGHT":
                player.x += INVADERS_PLAYER_SPEED
            player.x = max(0, min(WIDTH - player.width, player.x))

            # --- Disparo automático del jugador ---
            if now - last_shot > INVADERS_FIRE_DELAY:
                player_bullets.append(pygame.Rect(player.centerx - pb_w // 2, player.top - pb_h, pb_w, pb_h))
                last_shot = now

            # --- Movimiento de los enemigos ---
            remaining = len(enemies)
            speed = INVADERS_ENEMY_SPEED + (1 - remaining / total_enemies) * 3
            move_acc += speed * direction
            step = int(move_acc)
            move_acc -= step
            if step:
                left = min(e.left for e in enemies) + step
                right = max(e.right for e in enemies) + step
                if left < 10 or right > WIDTH - 10:
                    direction *= -1
                    move_acc = 0
                    for e in enemies:
                        e.y += INVADERS_DROP
                else:
                    for e in enemies:
                        e.x += step

            # --- Disparo enemigo (desde el enemigo más bajo de una columna) ---
            if now - last_enemy_shot > INVADERS_ENEMY_FIRE_DELAY:
                bottom = {}
                for e in enemies:
                    if e.x not in bottom or e.y > bottom[e.x].y:
                        bottom[e.x] = e
                shooter = random.choice(list(bottom.values()))
                enemy_bullets.append(pygame.Rect(shooter.centerx - eb_w // 2, shooter.bottom, eb_w, eb_h))
                enemy_attack_until[id(shooter)] = now + INVADER_ATTACK_MS
                last_enemy_shot = now

            # --- Mover balas ---
            for b in player_bullets:
                b.y -= INVADERS_BULLET_SPEED
            for b in enemy_bullets:
                b.y += INVADERS_ENEMY_BULLET_SPEED
            player_bullets = [b for b in player_bullets if b.bottom > 0]
            enemy_bullets = [b for b in enemy_bullets if b.top < HEIGHT]

            # --- Balas del jugador vs enemigos ---
            for b in player_bullets[:]:
                hit = next((e for e in enemies if b.colliderect(e)), None)
                if hit:
                    enemies.remove(hit)
                    player_bullets.remove(b)
                    score += 10
                    if inv_sprites:
                        effects.append({"type": "die", "rect": hit.copy(), "start": now})
                    if bullet_sprites:
                        effects.append({"type": "hit", "owner": "player", "pos": b.midtop, "start": now})

            # --- Balas enemigas vs jugador ---
            for b in enemy_bullets[:]:
                if b.colliderect(player) and now > invulnerable_until:
                    enemy_bullets.remove(b)
                    lives -= 1
                    if bullet_sprites:
                        effects.append({"type": "hit", "owner": "enemy", "pos": b.midbottom, "start": now})
                    invulnerable_until = now + 1000

            # --- Condiciones de fin ---
            if not enemies:
                state = "win"
                end_time = now
            elif lives <= 0 or max(e.bottom for e in enemies) >= player.top:
                state = "lose"
                end_time = now

        elif now - end_time > 2500:
            return state

        # --- Dibujo ---
        background.update()
        background.draw(screen)

        for e in enemies:
            if inv_sprites:
                sprite_state = "attack" if now < enemy_attack_until.get(id(e), 0) else "idle"
                bob = round(INVADER_BOB_AMPLITUDE * math.sin(
                    now / INVADER_BOB_PERIOD_MS * 2 * math.pi + enemy_phase[id(e)]))
                ox, oy = inv_sprites["offset"]
                screen.blit(inv_sprites["frames"][sprite_state], (e.x - ox, e.y - oy + bob))
            else:
                screen.blit(enemy_fallback, e)
        for b in player_bullets:
            if bullet_sprites:
                spr = pick_frame(bullet_sprites["player"]["fly"], now, BULLET_ANIM_FPS)
                screen.blit(spr, (b.centerx - spr.get_width() // 2, b.top))   # la cabeza va arriba
            else:
                pygame.draw.rect(screen, PLAYER_BULLET_COLOR, b)
        for b in enemy_bullets:
            if bullet_sprites:
                spr = pick_frame(bullet_sprites["enemy"]["fly"], now, BULLET_ANIM_FPS)
                screen.blit(spr, (b.centerx - spr.get_width() // 2, b.bottom - spr.get_height()))
            else:
                pygame.draw.rect(screen, ENEMY_BULLET_COLOR, b)
        effects = draw_effects(screen, effects, now, inv_sprites, bullet_sprites)

        if now > invulnerable_until or (now // 100) % 2 == 0:
            screen.blit(player_image, player)

        screen.blit(font_small.render(f"Puntos: {score}", True, COLOR_TEXT), (20, 15))
        screen.blit(font_small.render(f"Vidas: {lives}", True, (255, 120, 120)), (20, 45))
        screen.blit(font_small.render(
            "Brazo derecho/izquierdo extendido a la altura del hombro", True, (255, 255, 0)),
            (20, HEIGHT - 35))

        if state == "win":
            text = font_large.render("¡VICTORIA!", True, (120, 255, 120))
            screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - 30))
        elif state == "lose":
            text = font_large.render("¡PERDISTE!", True, (255, 50, 50))
            screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - 30))

        music.draw(screen, font_small)
        pygame.display.flip()
        clock.tick(FPS)


def make_planet_surface(radius):
    """Dibuja un planeta (océanos, bandas, continentes y sombra) de radio `radius`."""
    rng = random.Random(7)
    size = radius * 2
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    surf.fill((45, 105, 200, 255))

    # Bandas de nubes
    for i in range(6):
        y = int(size * (0.08 + i * 0.16))
        color = (70, 140, 225) if i % 2 else (35, 85, 170)
        pygame.draw.rect(surf, color, (0, y, size, int(size * 0.06)))

    # Continentes
    for _ in range(10):
        cx = rng.randint(int(size * 0.1), int(size * 0.9))
        cy = rng.randint(int(size * 0.1), int(size * 0.9))
        r = rng.randint(int(radius * 0.08), int(radius * 0.22))
        pygame.draw.circle(surf, (70, 175, 110), (cx, cy), r)

    # Sombra en forma de media luna (lado derecho)
    shade = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(shade, (0, 0, 20, 150), (radius, radius), radius)
    pygame.draw.circle(shade, (0, 0, 0, 0), (int(radius * 0.7), int(radius * 0.85)), radius)
    surf.blit(shade, (0, 0))

    # Máscara circular para recortar los bordes
    mask = pygame.Surface((size, size), pygame.SRCALPHA)
    mask.fill((0, 0, 0, 0))
    pygame.draw.circle(mask, (255, 255, 255, 255), (radius, radius), radius)
    surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    return surf


def run_ending(screen, clock, shutdown, ship_image, fonts):
    """Pantalla final: la nave llega a un planeta. Termina al pulsar R (reiniciar)."""
    font_large, font_small = fonts
    rng = random.Random()
    stars = [[rng.randint(0, WIDTH), rng.randint(0, HEIGHT), rng.uniform(0.5, 3), rng.randint(1, 2)]
             for _ in range(90)]

    planet_full = make_planet_surface(ENDING_PLANET_RADIUS)
    planet_center = (int(WIDTH * 0.68), HEIGHT // 2)
    ship_w, ship_h = ship_image.get_size()

    frame = 0
    while True:
        for event in pygame.event.get():
            music.handle_event(event)
            if event.type == pygame.QUIT:
                shutdown()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    shutdown()
                if event.key in (pygame.K_r, pygame.K_SPACE, pygame.K_RETURN) and frame >= ENDING_FRAMES:
                    return

        t = min(1.0, frame / ENDING_FRAMES)
        ease = t * t * (3 - 2 * t)          # entrada y salida suaves

        # Fondo y estrellas (se frenan al llegar)
        screen.fill((5, 5, 20))
        for s in stars:
            s[0] -= s[2] * (1 - 0.8 * ease)
            if s[0] < 0:
                s[0] = WIDTH
                s[1] = rng.randint(0, HEIGHT)
            v = min(255, int(120 + 45 * s[2]))
            pygame.draw.circle(screen, (v, v, v), (int(s[0]), int(s[1])), s[3])

        # Planeta que crece al acercarse
        radius = max(10, int(50 + (ENDING_PLANET_RADIUS - 50) * ease))
        glow = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for i in range(5):
            pygame.draw.circle(glow, (110, 170, 255, 12 + i * 12), planet_center, radius + 50 - i * 10)
        screen.blit(glow, (0, 0))
        planet = pygame.transform.smoothscale(planet_full, (radius * 2, radius * 2))
        screen.blit(planet, planet.get_rect(center=planet_center))

        # Nave que se acerca y se hace más pequeña (perspectiva)
        scale = 1.0 - 0.65 * ease
        sw, sh = max(4, int(ship_w * scale)), max(4, int(ship_h * scale))
        ship = pygame.transform.smoothscale(ship_image, (sw, sh))
        end_x = planet_center[0] - radius * 0.3
        end_y = planet_center[1] - radius * 0.1
        ship_x = -60 + (end_x + 60) * ease
        ship_y = HEIGHT * 0.8 + (end_y - HEIGHT * 0.8) * ease
        screen.blit(ship, ship.get_rect(center=(int(ship_x), int(ship_y))))

        # Mensaje final
        if frame >= ENDING_FRAMES:
            title = font_large.render("¡LLEGASTE A TU DESTINO!", True, (255, 255, 255))
            sub = font_small.render("Presiona 'R' para volver al menú o 'Q' para salir", True, (255, 255, 0))
            screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 40))
            screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, HEIGHT - 60))

        frame += 1
        music.draw(screen, font_small)
        pygame.display.flip()
        clock.tick(FPS)


# =====================================================================
#  MENÚ PRINCIPAL (ESTILO SONIC CD / SCI-FI)
# =====================================================================
def draw_scifi_emblem(surface, center, ship_image, t, title_logo=None):
    """Dibuja el emblema central estilo Sonic CD (o la imagen title_logo.png si existe)."""
    cx, cy = center
    bob = int(math.sin(t * 3.0) * 6)
    cy += bob

    if title_logo is not None:
        rect = title_logo.get_rect(center=(cx, cy))
        surface.blit(title_logo, rect)
        return

    # --- 1. Alas mecánicas / Sci-Fi (inspiradas en el emblema de Sonic) ---
    wing_bars = [
        (190, -35, 22, (0, 220, 255)),
        (165, -12, 20, (0, 170, 230)),
        (140,  10, 18, (0, 130, 200)),
        (110,  30, 16, (0, 90, 160)),
    ]
    for length, y_off, height, col in wing_bars:
        # Ala izquierda
        pts_l = [
            (cx - 55, cy + y_off),
            (cx - 55 - length, cy + y_off - 14),
            (cx - 40 - length, cy + y_off + height - 14),
            (cx - 45, cy + y_off + height),
        ]
        # Ala derecha
        pts_r = [
            (cx + 55, cy + y_off),
            (cx + 55 + length, cy + y_off - 14),
            (cx + 40 + length, cy + y_off + height - 14),
            (cx + 45, cy + y_off + height),
        ]
        pygame.draw.polygon(surface, col, pts_l)
        pygame.draw.polygon(surface, (200, 250, 255), pts_l, 2)
        pygame.draw.polygon(surface, col, pts_r)
        pygame.draw.polygon(surface, (200, 250, 255), pts_r, 2)

    # --- 2. Anillo central de energía ---
    ring_radius = 78
    glow = pygame.Surface((ring_radius * 2 + 40, ring_radius * 2 + 40), pygame.SRCALPHA)
    pygame.draw.circle(glow, (0, 200, 255, 45), (ring_radius + 20, ring_radius + 20), ring_radius + 14)
    surface.blit(glow, glow.get_rect(center=(cx, cy)))

    pygame.draw.circle(surface, (10, 16, 36), (cx, cy), ring_radius)
    pygame.draw.circle(surface, (0, 230, 255), (cx, cy), ring_radius, 5)
    pygame.draw.circle(surface, (255, 200, 60), (cx, cy), ring_radius - 9, 2)

    # Arco giratorio interno
    arc_rect = pygame.Rect(0, 0, (ring_radius - 16) * 2, (ring_radius - 16) * 2)
    arc_rect.center = (cx, cy)
    pygame.draw.arc(surface, (0, 255, 200), arc_rect, t * 2.5, t * 2.5 + 2.2, 4)
    pygame.draw.arc(surface, (0, 255, 200), arc_rect, t * 2.5 + math.pi, t * 2.5 + math.pi + 2.2, 4)

    # --- 3. Nave protagonista en el centro del anillo ---
    ship_scaled = pygame.transform.smoothscale(ship_image, (96, 74))
    surface.blit(ship_scaled, ship_scaled.get_rect(center=(cx, cy - 6)))

    # --- 4. Cinta / Banner inferior con el título Sci-Fi ---
    banner_w, banner_h = 420, 64
    banner_rect = pygame.Rect(0, 0, banner_w, banner_h)
    banner_rect.center = (cx, cy + 72)

    # Puntas laterales del listón
    left_ribbon = [
        (banner_rect.left - 28, banner_rect.centery - 8),
        (banner_rect.left + 8, banner_rect.top + 8),
        (banner_rect.left + 8, banner_rect.bottom + 8),
        (banner_rect.left - 28, banner_rect.bottom + 18),
        (banner_rect.left - 14, banner_rect.centery + 4),
    ]
    right_ribbon = [
        (banner_rect.right + 28, banner_rect.centery - 8),
        (banner_rect.right - 8, banner_rect.top + 8),
        (banner_rect.right - 8, banner_rect.bottom + 8),
        (banner_rect.right + 28, banner_rect.bottom + 18),
        (banner_rect.right + 14, banner_rect.centery + 4),
    ]
    pygame.draw.polygon(surface, (140, 20, 60), left_ribbon)
    pygame.draw.polygon(surface, (140, 20, 60), right_ribbon)

    # Cuerpo del banner
    pygame.draw.rect(surface, (200, 30, 80), banner_rect, border_radius=10)
    pygame.draw.rect(surface, (0, 240, 255), banner_rect, 3, border_radius=10)

    # Texto con efecto metálico/neón Sci-Fi
    font_title = pygame.font.SysFont("Impact", 46, italic=True)
    font_sub = pygame.font.SysFont("Consolas", 16, bold=True)

    title_shadow = font_title.render(GAME_TITLE, True, (10, 10, 30))
    title_cyan = font_title.render(GAME_TITLE, True, (0, 240, 255))
    title_white = font_title.render(GAME_TITLE, True, (240, 250, 255))

    tx = cx - title_white.get_width() // 2
    ty = banner_rect.top + 2
    surface.blit(title_shadow, (tx + 3, ty + 3))
    surface.blit(title_cyan, (tx - 2, ty - 1))
    surface.blit(title_white, (tx, ty))

    sub_surf = font_sub.render(GAME_SUBTITLE, True, (255, 230, 120))
    surface.blit(sub_surf, (cx - sub_surf.get_width() // 2, banner_rect.bottom - 19))


def run_menu(screen, clock, background, shutdown, ship_image, is_dead=False, last_score=0):
    """Muestra el menú principal o de reintento con un único botón minimalista."""
    font_btn = pygame.font.SysFont("Consolas", 26, bold=True)
    font_info = pygame.font.SysFont("Arial", 22, bold=True)

    title_logo = None
    if os.path.isfile(TITLE_LOGO_PATH):
        try:
            img = pygame.image.load(TITLE_LOGO_PATH).convert_alpha()
            scale = min(460 / max(1, img.get_width()), 260 / max(1, img.get_height()), 1.0)
            if scale < 1.0:
                img = pygame.transform.smoothscale(
                    img, (int(img.get_width() * scale), int(img.get_height() * scale))
                )
            title_logo = img
        except pygame.error as e:
            print(f"ADVERTENCIA: no se pudo cargar '{TITLE_LOGO_PATH}': {e}")

    btn_rect = pygame.Rect(0, 0, 260, 54)
    btn_rect.center = (WIDTH // 2, HEIGHT - 95)
    btn_label = "REINTENTAR" if is_dead else "JUGAR"

    while True:
        mouse_pos = pygame.mouse.get_pos()
        hover = btn_rect.collidepoint(mouse_pos)

        for event in pygame.event.get():
            music.handle_event(event)
            if event.type == pygame.QUIT:
                shutdown()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    shutdown()
                if event.key == pygame.K_b:
                    background.next()
                # Un solo botón de acción en teclado (Espacio, Enter o R)
                if event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_r):
                    return
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if hover:
                    return

        t = pygame.time.get_ticks() / 1000.0

        # 1. Fondo del juego en movimiento
        background.update()
        background.draw(screen)

        # Velo oscuro sutil para que el logo y el botón resalten sobre el fondo
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((5, 8, 22, 95))
        screen.blit(overlay, (0, 0))

        # 2. Emblema / Logo central estilo Sonic CD
        draw_scifi_emblem(screen, (WIDTH // 2, HEIGHT // 2 - 55), ship_image, t, title_logo)

        # Si viene de morir, muestra el puntaje obtenido de forma limpia
        if is_dead:
            msg = font_info.render(f"¡NAVE DESTRUIDA!  •  PUNTOS: {last_score}", True, (255, 110, 110))
            screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, btn_rect.top - 42))

        # 3. Botón único minimalista
        pulse = int((math.sin(t * 5.0) + 1) * 35)
        border_col = (255, 255, 255) if hover else (0, min(255, 200 + pulse), 255)
        fill_col = (0, 130, 190, 220) if hover else (12, 24, 52, 200)

        btn_surf = pygame.Surface(btn_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(btn_surf, fill_col, btn_surf.get_rect(), border_radius=12)
        screen.blit(btn_surf, btn_rect.topleft)
        pygame.draw.rect(screen, border_col, btn_rect, 2, border_radius=12)

        txt = font_btn.render(btn_label, True, (255, 255, 255))
        screen.blit(txt, txt.get_rect(center=btn_rect.center))

        music.draw(screen, font_info)
        pygame.display.flip()
        clock.tick(FPS)


# =====================================================================
#  JUEGO (proceso principal)
# =====================================================================
def run_game():
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{GAME_WINDOW_POS[0]},{GAME_WINDOW_POS[1]}"
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Body Tracking Flappy Drone")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("Arial", 48, bold=True)
    font_small = pygame.font.SysFont("Arial", 24)
    music.start()

    # --- Lanzar la cámara en un segundo proceso ---
    gesture_value = mproc.Value("i", 0)
    invaders_value = mproc.Value("i", 0)
    stop_event = mproc.Event()
    cam = mproc.Process(target=camera_process,
                        args=(gesture_value, invaders_value, stop_event), daemon=True)
    cam.start()

    def shutdown():
        stop_event.set()
        cam.join(timeout=2)
        if cam.is_alive():
            cam.terminate()
        pygame.quit()
        sys.exit()

    background = Background(WORLD1_BACKGROUND_PATH, WORLD1_BACKGROUND_COLOR,
                            WORLD1_BACKGROUND_SCROLL_SPEED)
    invaders_background = Background(WORLD2_BACKGROUND_PATH, WORLD2_BACKGROUND_COLOR,
                                     WORLD2_BACKGROUND_SCROLL_SPEED)
    pipe_frames = load_tentacles(PIPE_SPRITES_PATH)
    if not pipe_frames:
        print("AVISO: no hay sprites tentacle_*.png en 'pipes/', se usan rectángulos de color.")

    drone = Drone()

    # --- Mostrar el menú principal antes de empezar ---
    run_menu(screen, clock, background, shutdown, drone.image, is_dead=False)

    pipes = [Pipe(pipe_frames)]
    score = 0
    level = 0
    next_portal_level = PORTAL_FIRST_LEVEL

    while True:
        for event in pygame.event.get():
            music.handle_event(event)
            if event.type == pygame.QUIT:
                shutdown()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    shutdown()
                if event.key == pygame.K_b:
                    background.next()

        # Gesto de la cámara; las flechas sirven de respaldo/pruebas
        gesture = CODE_TO_GESTURE.get(gesture_value.value, "HOVER")
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            gesture = "UP"
        elif keys[pygame.K_DOWN]:
            gesture = "DOWN"

        level, spacing, gap, speed = get_difficulty(score)

        background.update()
        drone.move(gesture)

        enter_portal = False
        died = False
        for pipe in pipes:
            pipe.move(speed)

            # El portal no mata: al tocarlo se entra al minijuego
            if isinstance(pipe, Portal):
                if pipe.touches(drone.rect):
                    enter_portal = True
                continue

            if pipe.collides(drone.rect):
                died = True

            if not pipe.passed and pipe.x < drone.rect.x:
                score += 1
                pipe.passed = True

        # --- Si la nave muere, abre el menú de un solo botón ---
        if died:
            run_menu(screen, clock, background, shutdown, drone.image, is_dead=True, last_score=score)
            drone = Drone()
            pipes = [Pipe(pipe_frames)]
            score = 0
            level = 0
            next_portal_level = PORTAL_FIRST_LEVEL
            continue

        if enter_portal:
            portal_transition(screen, clock, drone.rect.center)
            result = run_invaders(screen, clock, invaders_background, invaders_value,
                                  shutdown, get_invaders_ship_image(drone), (font_large, font_small))
            if result == "win":
                # Ganó Space Invaders: pantalla final y luego vuelve al menú principal
                run_ending(screen, clock, shutdown, drone.image, (font_large, font_small))
                run_menu(screen, clock, background, shutdown, drone.image, is_dead=False)
                drone = Drone()
                pipes = [Pipe(pipe_frames)]
                score = 0
                level = 0
                next_portal_level = PORTAL_FIRST_LEVEL
            else:
                # Perdió: vuelve a Flappy con su puntuación; el siguiente portal
                # aparece en el próximo nivel múltiplo de PORTAL_EVERY
                drone = Drone()
                pipes = [Pipe(pipe_frames, gap)]
            continue

        if pipes[0].x < -150:
            pipes.pop(0)

        if pipes[-1].x < WIDTH - spacing:
            if level >= next_portal_level:
                pipes.append(Portal())
                next_portal_level += PORTAL_EVERY
            else:
                pipes.append(Pipe(pipe_frames, gap))

        # --- Dibujo ---
        background.draw(screen)
        level_text = font_small.render(f"Dificultad: {level}", True, (255, 150, 0))
        screen.blit(level_text, (20, 110))

        for pipe in pipes:
            pipe.draw(screen)

        drone.draw(screen)

        score_text = font_large.render(f"Puntos: {score}", True, COLOR_TEXT)
        gesture_text = font_small.render(f"Gesto: {gesture}", True, (255, 255, 0))
        screen.blit(score_text, (20, 20))
        screen.blit(gesture_text, (20, 80))

        music.draw(screen, font_small)
        pygame.display.flip()
        clock.tick(FPS)


def run_ship_preview():
    """Vista previa del modelo 3D para orientarlo.
    T: alternar PERFIL (mundo 1) / PLANTA (mundo 2) | Flechas: rotar X/Y | A y D: rotar Z | Q: salir."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Vista previa de la nave 3D")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)

    if not os.path.isfile(SHIP_3D_PATH):
        print(f"No se encontró el modelo: {SHIP_3D_PATH}")
        return
    model = Ship3D(SHIP_3D_PATH, (480, 360), SHIP_3D_COLOR, SHIP_3D_BASE_ROT, tilt_range=0)
    side_rot = list(SHIP_3D_BASE_ROT)
    top_rot = list(SHIP_3D_TOPVIEW_ROT)
    top = False

    def refresh():
        model.set_base_rot(*side_rot, extra_rot=top_rot if top else (0, 0, 0))

    running = True
    while running:
        rot = top_rot if top else side_rot            # lo que se está editando ahora
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_t:
                    top = not top
                elif event.key == pygame.K_UP:
                    rot[0] += 15
                elif event.key == pygame.K_DOWN:
                    rot[0] -= 15
                elif event.key == pygame.K_RIGHT:
                    rot[1] += 15
                elif event.key == pygame.K_LEFT:
                    rot[1] -= 15
                elif event.key == pygame.K_a:
                    rot[2] += 15
                elif event.key == pygame.K_d:
                    rot[2] -= 15
                refresh()

        screen.fill((20, 24, 40))
        img = model.surface(0)
        screen.blit(img, img.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 30)))
        if top:
            lines = [f"PLANTA (mundo 2)   SHIP_3D_TOPVIEW_ROT = {tuple(top_rot)}",
                     "Vista desde arriba: la nariz debe apuntar hacia ARRIBA",
                     "T: perfil/planta   Flechas: X/Y   A/D: Z   Q: salir"]
        else:
            lines = [f"PERFIL (mundo 1)   SHIP_3D_BASE_ROT = {tuple(side_rot)}",
                     "Vista de lado: la nariz debe apuntar a la DERECHA",
                     "T: perfil/planta   Flechas: X/Y   A/D: Z   Q: salir"]
        for i, t in enumerate(lines):
            color = (255, 255, 0) if i == 0 else (255, 255, 255)
            screen.blit(font.render(t, True, color), (20, HEIGHT - 100 + i * 28))
        pygame.display.flip()
        clock.tick(30)

    print(f"SHIP_3D_BASE_ROT = {tuple(side_rot)}")
    print(f"SHIP_3D_TOPVIEW_ROT = {tuple(top_rot)}")
    pygame.quit()


if __name__ == "__main__":
    mproc.freeze_support()  # necesario en Windows / ejecutables empaquetados
    if "--preview" in sys.argv:
        run_ship_preview()
    else:
        run_game()