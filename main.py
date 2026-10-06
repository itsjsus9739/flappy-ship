import os
import re
import sys
import time
import random
import multiprocessing as mproc

import cv2
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

CAMERA_INDEX = 0
MODEL_PATH = os.path.join(BASE_DIR, "models", "pose_landmarker_lite.task")

# =====================================================================
#  CONFIGURACIÓN DEL FONDO
# =====================================================================
# Puede ser una CARPETA (varias imágenes, se cambia con la tecla B)
# o un ARCHIVO de imagen. Si no existe, se usa BACKGROUND_COLOR.
BACKGROUND_PATH = os.path.join(BASE_DIR, "backgrounds")
BACKGROUND_COLOR = (20, 24, 40)
# 0 = fondo estático (la imagen se ajusta a la ventana)
# > 0 = fondo que se desplaza hacia la izquierda (parallax simple)
BACKGROUND_SCROLL_SPEED = 2

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


def camera_process(gesture_value, stop_event):
    """Corre en un proceso aparte: lee la cámara, detecta la pose y
    publica el gesto en memoria compartida para el juego."""
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
                draw_pose(frame, pose)
            else:
                gesture = "HOVER"

            gesture_value.value = GESTURE_TO_CODE[gesture]

            cv2.putText(frame, f"Gesto: {gesture}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, "Q: cerrar camara", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    gesture_value.value = GESTURE_TO_CODE["HOVER"]
    cap.release()
    cv2.destroyAllWindows()


# =====================================================================
#  CLASES DEL JUEGO
# =====================================================================
class Background:
    def __init__(self):
        self.files = list_images(BACKGROUND_PATH)
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
            if BACKGROUND_SCROLL_SPEED == 0:
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
        self.offset += BACKGROUND_SCROLL_SPEED

    def draw(self, surface):
        if self.image is None:
            surface.fill(BACKGROUND_COLOR)
            return
        w = self.image.get_width()
        x = -(self.offset % w) if BACKGROUND_SCROLL_SPEED else 0
        while x < WIDTH:
            surface.blit(self.image, (x, 0))
            x += w


class Drone:
    def __init__(self):
        self.width = 50
        self.height = 38
        self.x = WIDTH // 4
        self.y = HEIGHT // 2

        sprite_path = os.path.join(BASE_DIR, "flappy.png")
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
    def __init__(self, frames):
        self.x = WIDTH
        self.passed = False
        self.bottom_height = random.randint(PIPE_MIN_SEGMENT, HEIGHT - PIPE_GAP - PIPE_MIN_SEGMENT)
        self.top_height = HEIGHT - PIPE_GAP - self.bottom_height

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

    def move(self):
        self.x -= PIPE_SPEED
        self.rect_top.x = self.x
        self.rect_bottom.x = self.x

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
    stop_event = mproc.Event()
    cam = mproc.Process(target=camera_process, args=(gesture_value, stop_event), daemon=True)
    cam.start()

    def shutdown():
        stop_event.set()
        cam.join(timeout=2)
        if cam.is_alive():
            cam.terminate()
        pygame.quit()
        sys.exit()

    background = Background()
    pipe_frames = load_animation(PIPE_SPRITES_PATH, PIPE_SHEET_FRAMES)
    if not pipe_frames:
        print("AVISO: no hay sprites de tubos, se usan rectángulos de color.")

    drone = Drone()
    pipes = [Pipe(pipe_frames)]
    score = 0
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
                    game_over = False

        # Gesto de la cámara; las flechas sirven de respaldo/pruebas
        gesture = CODE_TO_GESTURE.get(gesture_value.value, "HOVER")
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            gesture = "UP"
        elif keys[pygame.K_DOWN]:
            gesture = "DOWN"

        if not game_over:
            background.update()
            drone.move(gesture)

            for pipe in pipes:
                pipe.move()

                if pipe.collides(drone.rect):
                    game_over = True

                if not pipe.passed and pipe.x < drone.rect.x:
                    score += 1
                    pipe.passed = True

            if pipes[0].x < -PIPE_WIDTH:
                pipes.pop(0)

            if pipes[-1].x < WIDTH - PIPE_SPACING:
                pipes.append(Pipe(pipe_frames))

        # --- Dibujo ---
        background.draw(screen)

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


if __name__ == "__main__":
    mproc.freeze_support()  # necesario en Windows / ejecutables empaquetados
    run_game()