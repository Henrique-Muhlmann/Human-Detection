# -*- coding: utf-8 -*-
"""
===============================================================================
 SISTEMA DE DETECCAO DE PARTES HUMANAS EM CELULA ROBOTIZADA  (Versao 1)
===============================================================================

Arquitetura: DEFESA EM PROFUNDIDADE (3 camadas redundantes)
-----------------------------------------------------------
  Camada 1  - MediaPipe Hands (Tasks API) : maos e dedos (21 landmarks).
  Camada 2  - MediaPipe Pose  (Tasks API) : braco / antebraco (ombro->cotovelo
                                            ->punho), com a linha do membro.
  Camada 3  - Pele + Movimento (OpenCV)   : rede de seguranca que captura o que
                                            as redes neurais perdem (dedo parcial,
                                            mao com motion blur, parte na borda).

  Regra de decisao: SE QUALQUER camada detectar -> ALERTA.
  Isso minimiza o FALSO-NEGATIVO (nao detectar humano presente), que e a
  falha perigosa num sistema de seguranca.

IMPORTANTE - API do MediaPipe
-----------------------------
O mediapipe >= 0.10.3x REMOVEU a API legada `mp.solutions.hands/pose`.
Este script usa a nova **Tasks API** (`mediapipe.tasks`), que exige dois
arquivos de modelo no mesmo diretorio:
    - hand_landmarker.task
    - pose_landmarker_lite.task
(baixados do repositorio oficial de modelos do Google MediaPipe).

!!! AVISO DE SEGURANCA FUNCIONAL !!!
------------------------------------
Este software NAO e um dispositivo de seguranca certificado. NAO substitui
cortinas de luz, scanners a laser ou reles de seguranca (ISO 13849 / IEC 61496).
Use-o como camada AUXILIAR de alerta, prototipo de P&D e coletor de dataset.
A parada fisica do robo deve sempre depender de hardware certificado.

Dependencias
------------
    pip install opencv-python mediapipe numpy

Execucao
--------
    python human_safety_detector.py
    (tecla 'q' = sair | 'd' = liga/desliga visualizacao da mascara de pele)

Autor: Henrique Amaral Muhlmann
===============================================================================
"""

import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# =============================================================================
# CONFIGURACAO (ajuste fino para o seu ambiente / iluminacao)
# =============================================================================
class Config:
    # --- Camera ---
    CAMERA_INDEX = 0  # 0 = webcam padrao; troque para a sua camera USB
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FLIP_HORIZONTAL = True  # espelha como um "espelho" (mais natural no teste)

    # --- Caminhos dos modelos (Tasks API) ---
    HAND_MODEL = "hand_landmarker.task"
    POSE_MODEL = "pose_landmarker_lite.task"

    # --- MediaPipe Hands ---
    HANDS_MAX = 4
    HANDS_DET_CONF = 0.4  # um pouco mais sensivel para seguranca

    # --- MediaPipe Pose ---
    POSE_DET_CONF = 0.4

    # --- Camada de pele + movimento (rede de seguranca) ---
    USE_SKIN_MOTION_LAYER = True
    SKIN_MIN_AREA = 2500  # area minima (px) de blob de pele p/ alertar
    MOTION_HISTORY = 200
    MOTION_VAR_THRESHOLD = 25
    SKIN_REQUIRES_MOTION = True  # so alerta pele se houver movimento junto

    # --- Estabilizacao temporal do alerta (histerese contra flicker) ---
    ALERT_ON_FRAMES = 2
    ALERT_OFF_FRAMES = 8


# =============================================================================
# CAMADA 1 + 2 : DETECTOR MEDIAPIPE (Tasks API) — maos/dedos + braco/antebraco
# =============================================================================
class MediaPipeDetector:
    """Encapsula HandLandmarker e PoseLandmarker (Tasks API, modo VIDEO)."""

    # Conexoes da mao (pares de indices de landmarks) para desenhar o esqueleto.
    HAND_CONNECTIONS = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),  # polegar
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),  # indicador
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),  # medio
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),  # anelar
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),  # mindinho
        (0, 17),  # base da palma
    ]

    # Indices dos landmarks de Pose que representam os membros superiores.
    # (esquema BlazePose: 11/12 ombros, 13/14 cotovelos, 15/16 punhos)
    POSE_LEFT_ARM = [11, 13, 15]
    POSE_RIGHT_ARM = [12, 14, 16]

    def __init__(self, cfg: Config):
        self.cfg = cfg

        for path in (cfg.HAND_MODEL, cfg.POSE_MODEL):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Modelo nao encontrado: {path}\n"
                    "Baixe os modelos da Tasks API do MediaPipe e coloque-os "
                    "no mesmo diretorio deste script."
                )

        # --- HandLandmarker ---
        hand_opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=cfg.HAND_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.HANDS_MAX,
            min_hand_detection_confidence=cfg.HANDS_DET_CONF,
            min_tracking_confidence=0.4,
        )
        self.hand = vision.HandLandmarker.create_from_options(hand_opts)

        # --- PoseLandmarker ---
        pose_opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=cfg.POSE_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=cfg.POSE_DET_CONF,
            min_tracking_confidence=0.4,
        )
        self.pose = vision.PoseLandmarker.create_from_options(pose_opts)

    def process(self, frame_bgr, timestamp_ms):
        """
        Processa um frame (BGR). MediaPipe Tasks espera RGB encapsulado em mp.Image.
        Retorna: (detectou: bool, lista_de_rotulos: list[str], frame_anotado)
        """
        labels = []
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # --- Camada 1: maos e dedos ---
        hand_res = self.hand.detect_for_video(mp_image, timestamp_ms)
        if hand_res.hand_landmarks:
            labels.append("MAO/DEDO")
            for lms in hand_res.hand_landmarks:
                pts = [(int(p.x * w), int(p.y * h)) for p in lms]
                for a, b in self.HAND_CONNECTIONS:
                    cv2.line(frame_bgr, pts[a], pts[b], (0, 255, 0), 2)
                for px, py in pts:
                    cv2.circle(frame_bgr, (px, py), 4, (0, 0, 255), -1)
                # Destaca as pontas dos dedos (4,8,12,16,20).
                for tip in (4, 8, 12, 16, 20):
                    cv2.circle(frame_bgr, pts[tip], 7, (255, 0, 255), 2)

        # --- Camada 2: braco / antebraco via Pose ---
        pose_res = self.pose.detect_for_video(mp_image, timestamp_ms)
        if pose_res.pose_landmarks:
            arm_visible = False
            for lms in pose_res.pose_landmarks:
                for chain in (self.POSE_LEFT_ARM, self.POSE_RIGHT_ARM):
                    chain_pts = []
                    for idx in chain:
                        lm = lms[idx]
                        # 'visibility'/'presence' indicam se o ponto esta no quadro.
                        vis = getattr(lm, "visibility", 1.0)
                        if vis > 0.5 and 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0:
                            chain_pts.append((int(lm.x * w), int(lm.y * h)))
                    if len(chain_pts) >= 2:
                        arm_visible = True
                        # Desenha o segmento do membro (ombro->cotovelo->punho).
                        for j in range(len(chain_pts) - 1):
                            cv2.line(
                                frame_bgr,
                                chain_pts[j],
                                chain_pts[j + 1],
                                (0, 255, 255),
                                4,
                            )
                        for px, py in chain_pts:
                            cv2.circle(frame_bgr, (px, py), 8, (0, 200, 255), -1)
            if arm_visible:
                labels.append("BRACO/ANTEBRACO")

        return (len(labels) > 0), labels, frame_bgr

    def close(self):
        self.hand.close()
        self.pose.close()


# =============================================================================
# CAMADA 3 : PELE + MOVIMENTO (rede de seguranca para deteccoes parciais)
# =============================================================================
class SkinMotionDetector:
    """
    Detecta regioes de cor de pele que tambem apresentam movimento.
    Captura casos que as redes neurais perdem: dedo parcial na borda,
    mao borrada por velocidade, parte do membro sem topologia reconhecivel.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=cfg.MOTION_HISTORY,
            varThreshold=cfg.MOTION_VAR_THRESHOLD,
            detectShadows=False,
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    @staticmethod
    def _skin_mask(frame_bgr):
        """Mascara de pele combinando YCrCb e HSV para robustez a iluminacao."""
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(
            ycrcb, np.array([0, 133, 77], np.uint8), np.array([255, 173, 127], np.uint8)
        )
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(
            hsv, np.array([0, 30, 60], np.uint8), np.array([25, 150, 255], np.uint8)
        )
        return cv2.bitwise_and(mask_ycrcb, mask_hsv)

    def process(self, frame_bgr, debug=False):
        """Retorna: (detectou: bool, frame_anotado, mascara_debug ou None)"""
        skin = self._skin_mask(frame_bgr)
        motion = self.bg.apply(frame_bgr)
        combined = (
            cv2.bitwise_and(skin, motion) if self.cfg.SKIN_REQUIRES_MOTION else skin
        )
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self.kernel)
        combined = cv2.dilate(combined, self.kernel, iterations=2)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        detected = False
        for c in contours:
            if cv2.contourArea(c) >= self.cfg.SKIN_MIN_AREA:
                detected = True
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 140, 255), 2)
                cv2.putText(
                    frame_bgr,
                    "PELE+MOV",
                    (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 140, 255),
                    2,
                )
        return detected, frame_bgr, (combined if debug else None)


# =============================================================================
# ESTABILIZADOR TEMPORAL DO ALERTA (histerese contra flicker)
# =============================================================================
class AlertStabilizer:
    def __init__(self, on_frames, off_frames):
        self.on_frames, self.off_frames = on_frames, off_frames
        self.alert = False
        self._pos = self._neg = 0

    def update(self, raw_detection: bool) -> bool:
        if raw_detection:
            self._pos += 1
            self._neg = 0
            if self._pos >= self.on_frames:
                self.alert = True
        else:
            self._neg += 1
            self._pos = 0
            if self._neg >= self.off_frames:
                self.alert = False
        return self.alert


# =============================================================================
# OVERLAY VISUAL (HUD do operador)
# =============================================================================
def draw_hud(frame, alert, labels, fps):
    h, w = frame.shape[:2]
    if alert:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 16)
        banner = "!! ALERTA: PARTE HUMANA DETECTADA !!"
        (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        cx = (w - tw) // 2
        cv2.rectangle(frame, (cx - 20, 20), (cx + tw + 20, 80), (0, 0, 255), -1)
        cv2.putText(
            frame, banner, (cx, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3
        )
        det = " + ".join(sorted(set(labels))) if labels else "deteccao"
        cv2.putText(
            frame,
            f"Origem: {det}",
            (cx, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    else:
        cv2.putText(
            frame, "AREA LIVRE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2
        )
    cv2.putText(
        frame,
        f"FPS: {fps:4.1f}",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "q=sair  d=debug",
        (w - 240, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2,
    )
    return frame


# =============================================================================
# LOOP PRINCIPAL (JANELA AO VIVO)
# =============================================================================
def main():
    cfg = Config()

    cap = cv2.VideoCapture(cfg.CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(
            f"Nao foi possivel abrir a camera (indice {cfg.CAMERA_INDEX})."
        )

    mp_det = MediaPipeDetector(cfg)
    skin_det = SkinMotionDetector(cfg) if cfg.USE_SKIN_MOTION_LAYER else None
    stabilizer = AlertStabilizer(cfg.ALERT_ON_FRAMES, cfg.ALERT_OFF_FRAMES)

    debug = False
    prev_t = time.time()
    fps = 0.0
    t0 = time.time()

    print("[INFO] Janela ao vivo iniciada. 'q' = sair, 'd' = debug da mascara.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Falha ao capturar frame.")
                break
            if cfg.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            # Timestamp monotonico crescente em ms (exigido pelo modo VIDEO).
            ts_ms = int((time.time() - t0) * 1000)

            mp_hit, labels, frame = mp_det.process(frame, ts_ms)

            skin_hit = False
            dbg_mask = None
            if skin_det is not None:
                skin_hit, frame, dbg_mask = skin_det.process(frame, debug=debug)
                if skin_hit:
                    labels.append("PELE+MOV")

            alert = stabilizer.update(mp_hit or skin_hit)

            now = time.time()
            inst = 1.0 / max(now - prev_t, 1e-6)
            fps = 0.9 * fps + 0.1 * inst if fps > 0 else inst
            prev_t = now

            frame = draw_hud(frame, alert, labels, fps)

            cv2.imshow("Protecao de Celula Robotizada - Deteccao Humana", frame)
            if debug and dbg_mask is not None:
                cv2.imshow("DEBUG: mascara pele+movimento", dbg_mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d"):
                debug = not debug
                if not debug:
                    cv2.destroyWindow("DEBUG: mascara pele+movimento")
    finally:
        mp_det.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Sistema finalizado.")


if __name__ == "__main__":
    main()
