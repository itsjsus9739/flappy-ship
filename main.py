import os
import re
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
#  CONFIGURACIÓN DEL PORTAL
# =====================================================================
PORTAL_FIRST_LEVEL = 0                 # primer nivel en el que aparece un portal (normal: 10)
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
# SPRITE_PATH puede ser: un archivo (png/jpg), una CARPETA con frames
# (invader_0.png, invader_1.png, ...) o un spritesheet horizontal
# (en ese caso indica cuántos frames tiene en INVADER_SHEET_FRAMES).
# Si no existe, se dibuja un enemigo verde simple.
INVADER_SPRITE_PATH = os.path.join(BASE_DIR, "invader.png")
INVADER_SHEET_FRAMES = 1
INVADER_ANIM_FPS = 6                   # velocidad de la animación del enemigo
INVADER_SIZE = (40, 30)                # tamaño (ancho, alto) de cada enemigo
INVADERS_ROWS, INVADERS_COLS = 4, 8
INVADERS_SPACING = (20, 15)            # separación horizontal y vertical entre enemigos
# OJO: INVADERS_COLS * (ancho + separación) debe ser menor a ~760 px

# --- Proyectiles (mismas opciones de sprite que los enemigos) ---
BULLET_ANIM_FPS = 10                   # velocidad de animación de los proyectiles

PLAYER_BULLET_SPRITE_PATH = os.path.join(BASE_DIR, "bullet_player.png")
PLAYER_BULLET_SHEET_FRAMES = 1
PLAYER_BULLET_SIZE = (6, 18)           # tamaño (ancho, alto) del proyectil del jugador
PLAYER_BULLET_COLOR = (0, 255, 255)    # color si no hay sprite

ENEMY_BULLET_SPRITE_PATH = os.path.join(BASE_DIR, "bullet_enemy.png")
ENEMY_BULLET_SHEET_FRAMES = 1
ENEMY_BULLET_SIZE = (6, 18)            # tamaño (ancho, alto) del proyectil enemigo
ENEMY_BULLET_COLOR = (255, 80, 80)     # color si no hay sprite
ENEMY_BULLET_FLIP_Y = False            # True si tu sprite apunta hacia arriba y debe caer boca abajo

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
DRONE_SIZE = (50, 38)      # tamaño (ancho, alto) de la nave; también es su hitbox

# Modelo 3D: .obj (sin instalar nada) o .glb/.gltf/.stl/.ply (requiere: pip install trimesh).
# Si el archivo no existe, se usa el sprite 2D (ship.jfif) como antes.
# Prueba y orienta tu modelo con:  python main.py --preview
SHIP_3D_PATH = os.path.join(BASE_DIR, "ship.obj")
SHIP_3D_COLOR = (0, 200, 255)          # color base del modelo (no usa texturas)
SHIP_3D_BASE_ROT = (0, 0, 0)           # rotación (x, y, z) en grados para orientar el modelo
SHIP_3D_TILT_MAX = 25                  # inclinación máxima (grados) al subir/bajar
SHIP_3D_TILT_STEP = 3                  # paso de la caché de inclinación (grados)
SHIP_3D_SUPERSAMPLE = 3                # suavizado de bordes (más = más lento al arrancar)
SHIP_3D_LIGHT = (0.4, 0.6, 0.7)        # dirección de la luz (x, y, z)

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

# Sprites: CARPETA con frames (pipe_0.png, pipe_1.png, ...) o ARCHIVO.
# Si es un solo archivo con varios frames en horizontal (spritesheet),
# indica cuántos frames tiene en PIPE_SHEET_FRAMES.
PIPE_SPRITES_PATH = os.path.join(BASE_DIR, "pipes")
PIPE_SHEET_FRAMES = 7
PIPE_ANIM_FPS = 8          # velocidad de la animación

# "tile"    = repite el sprite hacia abajo/arriba sin deformarlo
# "stretch" = estira el sprite para llenar todo el tubo
PIPE_FILL_MODE = "tile"
# Voltea verticalmente el tubo de arriba para que quede "boca abajo"
PIPE_FLIP_TOP = True

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
            frames.append(pygame.image.load(f).convert_alpha())
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
     # Posición de las ventanas en pantalla (x, y en píxeles)
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


def load_mesh(path):
    """Carga un modelo 3D. Devuelve (vértices Nx3, triángulos Mx3)."""
    if path.lower().endswith(".obj"):
        verts, faces = [], []
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    p = line.split()
                    verts.append([float(p[1]), float(p[2]), float(p[3])])
                elif line.startswith("f "):
                    idx = []
                    for tok in line.split()[1:]:
                        i = int(tok.split("/")[0])
                        idx.append(i - 1 if i > 0 else len(verts) + i)
                    for k in range(1, len(idx) - 1):      # triangula polígonos
                        faces.append((idx[0], idx[k], idx[k + 1]))
        return np.array(verts, dtype=float), np.array(faces, dtype=int)

    try:
        import trimesh
    except ImportError:
        raise RuntimeError("para este formato instala trimesh:  pip install trimesh")
    mesh = trimesh.load(path, force="mesh")
    return np.array(mesh.vertices, dtype=float), np.array(mesh.faces, dtype=int)


class Ship3D:
    """Renderizador 3D por software (proyección ortográfica, sombreado plano).
    Genera superficies de pygame y las guarda en caché por ángulo de inclinación."""

    def __init__(self, path, size, color, base_rot):
        self.size = size
        self.color = np.array(color, dtype=float)
        verts, self.faces = load_mesh(path)
        if len(verts) == 0 or len(self.faces) == 0:
            raise RuntimeError("el modelo no tiene geometría")
        self.verts = verts - (verts.max(axis=0) + verts.min(axis=0)) / 2
        if len(self.faces) > 5000:
            print(f"AVISO: el modelo tiene {len(self.faces)} caras; con más de ~5000 "
                  "el arranque puede tardar. Simplifícalo en Blender (Decimate).")
        self.cache = {}
        self.set_base_rot(*base_rot)

    def set_base_rot(self, rx, ry, rz):
        self.base_rot = (rx, ry, rz)
        self.base = _rot_matrix(rx, ry, rz)
        self.cache.clear()

        # Escala fija para que el modelo quepa también inclinado al máximo
        half_x = half_y = 1e-9
        for tilt in (-SHIP_3D_TILT_MAX, 0, SHIP_3D_TILT_MAX):
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
        tri = v[self.faces]                                   # (F, 3, 3)

        # Normales, orientadas hacia la cámara (iluminación de dos caras)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(n, axis=1, keepdims=True)
        length[length == 0] = 1
        n = n / length
        n = n * np.where(n[:, 2:3] < 0, -1, 1)

        light = np.array(SHIP_3D_LIGHT, dtype=float)
        light /= np.linalg.norm(light)
        intensity = 0.35 + 0.65 * np.clip(n @ light, 0, 1)
        colors = np.clip(self.color * intensity[:, None], 0, 255).astype(int).tolist()

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


def build_pipe_surface(frame, height):
    """Crea la superficie de un tubo de PIPE_WIDTH x height a partir de un frame."""
    if PIPE_FILL_MODE == "stretch":
        return pygame.transform.scale(frame, (PIPE_WIDTH, height))

    # modo "tile": se repite el sprite sin deformarlo
    surf = pygame.Surface((PIPE_WIDTH, height), pygame.SRCALPHA)
    scale = PIPE_WIDTH / frame.get_width()
    tile_h = max(1, int(frame.get_height() * scale))
    tile = pygame.transform.scale(frame, (PIPE_WIDTH, tile_h))
    y = 0
    while y < height:
        surf.blit(tile, (0, y))
        y += tile_h
    return surf


class Pipe:
    def __init__(self, frames, gap=PIPE_GAP):
        self.x = WIDTH
        self.passed = False
        self.bottom_height = random.randint(PIPE_MIN_SEGMENT, HEIGHT - gap - PIPE_MIN_SEGMENT)
        self.top_height = HEIGHT - gap - self.bottom_height

        self.rect_top = pygame.Rect(self.x, 0, PIPE_WIDTH, self.top_height)
        self.rect_bottom = pygame.Rect(self.x, HEIGHT - self.bottom_height, PIPE_WIDTH, self.bottom_height)

        # Se pre-generan las superficies de cada frame de animación
        self.top_surfaces = []
        self.bottom_surfaces = []
        for f in frames:
            self.bottom_surfaces.append(build_pipe_surface(f, self.bottom_height))
            top = build_pipe_surface(f, self.top_height)
            if PIPE_FLIP_TOP:
                top = pygame.transform.flip(top, False, True)
            self.top_surfaces.append(top)

    def move(self, speed=PIPE_SPEED):
        self.x -= speed
        self.rect_top.x = int(self.x)
        self.rect_bottom.x = int(self.x)

    def collides(self, rect):
        pad = PIPE_HITBOX_PADDING * 2
        return (rect.colliderect(self.rect_top.inflate(-pad, -pad)) or
                rect.colliderect(self.rect_bottom.inflate(-pad, -pad)))

    def draw(self, surface, frame_index):
        if self.bottom_surfaces:
            i = frame_index % len(self.bottom_surfaces)
            surface.blit(self.top_surfaces[i], self.rect_top.topleft)
            surface.blit(self.bottom_surfaces[i], self.rect_bottom.topleft)
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

    def draw(self, surface, frame_index):
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
    enemy_frames = load_scaled_frames(INVADER_SPRITE_PATH, INVADER_SIZE, INVADER_SHEET_FRAMES)
    if not enemy_frames:
        enemy_img = pygame.Surface(INVADER_SIZE, pygame.SRCALPHA)
        pygame.draw.ellipse(enemy_img, (120, 255, 120), (0, 0, inv_w, inv_h))
        pygame.draw.circle(enemy_img, (0, 0, 0), (int(inv_w * 0.3), int(inv_h * 0.45)), 4)
        pygame.draw.circle(enemy_img, (0, 0, 0), (int(inv_w * 0.7), int(inv_h * 0.45)), 4)
        enemy_frames = [enemy_img]

    pb_w, pb_h = PLAYER_BULLET_SIZE
    eb_w, eb_h = ENEMY_BULLET_SIZE
    player_bullet_frames = load_scaled_frames(
        PLAYER_BULLET_SPRITE_PATH, PLAYER_BULLET_SIZE, PLAYER_BULLET_SHEET_FRAMES)
    enemy_bullet_frames = load_scaled_frames(
        ENEMY_BULLET_SPRITE_PATH, ENEMY_BULLET_SIZE, ENEMY_BULLET_SHEET_FRAMES)
    if ENEMY_BULLET_FLIP_Y:
        enemy_bullet_frames = [pygame.transform.flip(f, False, True) for f in enemy_bullet_frames]

    # Enemigos
    enemies = []
    gap_x, gap_y = INVADERS_SPACING
    grid_w = INVADERS_COLS * inv_w + (INVADERS_COLS - 1) * gap_x
    start_x = (WIDTH - grid_w) // 2
    for row in range(INVADERS_ROWS):
        for col in range(INVADERS_COLS):
            enemies.append(pygame.Rect(start_x + col * (inv_w + gap_x), 60 + row * (inv_h + gap_y), inv_w, inv_h))
    total_enemies = len(enemies)
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

            # --- Balas enemigas vs jugador ---
            for b in enemy_bullets[:]:
                if b.colliderect(player) and now > invulnerable_until:
                    enemy_bullets.remove(b)
                    lives -= 1
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
            screen.blit(pick_frame(enemy_frames, now, INVADER_ANIM_FPS), e)
        for b in player_bullets:
            if player_bullet_frames:
                screen.blit(pick_frame(player_bullet_frames, now, BULLET_ANIM_FPS), b)
            else:
                pygame.draw.rect(screen, PLAYER_BULLET_COLOR, b)
        for b in enemy_bullets:
            if enemy_bullet_frames:
                screen.blit(pick_frame(enemy_bullet_frames, now, BULLET_ANIM_FPS), b)
            else:
                pygame.draw.rect(screen, ENEMY_BULLET_COLOR, b)

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
            if event.type == pygame.QUIT:
                shutdown()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    shutdown()
                if event.key == pygame.K_r and frame >= ENDING_FRAMES:
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
            sub = font_small.render("Presiona 'R' para jugar de nuevo o 'Q' para salir", True, (255, 255, 0))
            screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 40))
            screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, HEIGHT - 60))

        frame += 1
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
    pipe_frames = load_animation(PIPE_SPRITES_PATH, PIPE_SHEET_FRAMES)
    if not pipe_frames:
        print("AVISO: no hay sprites de tubos, se usan rectángulos de color.")

    drone = Drone()
    pipes = [Pipe(pipe_frames)]
    score = 0
    level = 0
    next_portal_level = PORTAL_FIRST_LEVEL
    game_over = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                shutdown()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    shutdown()
                if event.key == pygame.K_b:
                    background.next()
                if event.key == pygame.K_r and game_over:
                    drone = Drone()
                    pipes = [Pipe(pipe_frames)]
                    score = 0
                    level = 0
                    next_portal_level = PORTAL_FIRST_LEVEL
                    game_over = False

        # Gesto de la cámara; las flechas sirven de respaldo/pruebas
        gesture = CODE_TO_GESTURE.get(gesture_value.value, "HOVER")
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            gesture = "UP"
        elif keys[pygame.K_DOWN]:
            gesture = "DOWN"

        if not game_over:
            level, spacing, gap, speed = get_difficulty(score)

            background.update()
            drone.move(gesture)

            enter_portal = False
            for pipe in pipes:
                pipe.move(speed)

                # El portal no mata: al tocarlo se entra al minijuego
                if isinstance(pipe, Portal):
                    if pipe.touches(drone.rect):
                        enter_portal = True
                    continue

                if pipe.collides(drone.rect):
                    game_over = True

                if not pipe.passed and pipe.x < drone.rect.x:
                    score += 1
                    pipe.passed = True

            if enter_portal:
                portal_transition(screen, clock, drone.rect.center)
                result = run_invaders(screen, clock, invaders_background, invaders_value,
                                      shutdown, drone.image, (font_large, font_small))
                if result == "win":
                    # Ganó Space Invaders: pantalla final y el juego se reinicia con R
                    run_ending(screen, clock, shutdown, drone.image, (font_large, font_small))
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

        anim_index = int(pygame.time.get_ticks() / 1000 * PIPE_ANIM_FPS)
        for pipe in pipes:
            pipe.draw(screen, anim_index)

        drone.draw(screen)

        score_text = font_large.render(f"Puntos: {score}", True, COLOR_TEXT)
        gesture_text = font_small.render(f"Gesto: {gesture}", True, (255, 255, 0))
        help_text = font_small.render("B: cambiar fondo", True, COLOR_TEXT)
        screen.blit(score_text, (20, 20))
        screen.blit(gesture_text, (20, 80))
        screen.blit(help_text, (20, HEIGHT - 35))

        if game_over:
            go_text = font_large.render("¡CHOCASTE!", True, (255, 50, 50))
            restart_text = font_small.render("Presiona 'R' para reiniciar o 'Q' para salir", True, COLOR_TEXT)
            screen.blit(go_text, (WIDTH // 2 - go_text.get_width() // 2, HEIGHT // 2 - 50))
            screen.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT // 2 + 20))

        pygame.display.flip()
        clock.tick(FPS)


def run_ship_preview():
    """Vista previa del modelo 3D para encontrar la orientación correcta.
    Flechas: rotar X/Y | A y D: rotar Z | Q o Esc: salir."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Vista previa de la nave 3D")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)

    if not os.path.isfile(SHIP_3D_PATH):
        print(f"No se encontró el modelo: {SHIP_3D_PATH}")
        return
    model = Ship3D(SHIP_3D_PATH, (480, 360), SHIP_3D_COLOR, SHIP_3D_BASE_ROT)
    rx, ry, rz = SHIP_3D_BASE_ROT

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_UP:
                    rx += 15
                elif event.key == pygame.K_DOWN:
                    rx -= 15
                elif event.key == pygame.K_RIGHT:
                    ry += 15
                elif event.key == pygame.K_LEFT:
                    ry -= 15
                elif event.key == pygame.K_a:
                    rz += 15
                elif event.key == pygame.K_d:
                    rz -= 15
                model.set_base_rot(rx, ry, rz)

        screen.fill((20, 24, 40))
        img = model.surface(0)
        screen.blit(img, img.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 30)))
        lines = [f"SHIP_3D_BASE_ROT = ({rx}, {ry}, {rz})",
                 "La nariz debe apuntar a la DERECHA de la pantalla",
                 "Flechas: rotar X/Y   A/D: rotar Z   Q: salir"]
        for i, t in enumerate(lines):
            color = (255, 255, 0) if i == 0 else (255, 255, 255)
            screen.blit(font.render(t, True, color), (20, HEIGHT - 100 + i * 28))
        pygame.display.flip()
        clock.tick(30)

    print(f"SHIP_3D_BASE_ROT = ({rx}, {ry}, {rz})")
    pygame.quit()


if __name__ == "__main__":
    mproc.freeze_support()  # necesario en Windows / ejecutables empaquetados
    if "--preview" in sys.argv:
        run_ship_preview()
    else:
        run_game()