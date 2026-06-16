# -*- coding: utf-8 -*-
"""
Teste HEADLESS (sem janela GUI) do sistema de deteccao.
Captura N frames da camera real, roda as 3 camadas de deteccao e salva
imagens anotadas em disco, alem de um relatorio de deteccao por frame.

Uso: python test_headless.py
"""
import os
import time
import cv2

# Reaproveita as classes ja escritas no sistema principal.
from human_safety_detector import (
    Config, MediaPipeDetector, SkinMotionDetector, AlertStabilizer, draw_hud
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")
os.makedirs(OUT_DIR, exist_ok=True)

N_FRAMES = 40          # quantos frames capturar
SAVE_EVERY = 8         # salvar imagem a cada X frames


def main():
    cfg = Config()
    cap = cv2.VideoCapture(cfg.CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Camera nao abriu.")

    mp_det = MediaPipeDetector(cfg)
    skin_det = SkinMotionDetector(cfg)
    stabilizer = AlertStabilizer(cfg.ALERT_ON_FRAMES, cfg.ALERT_OFF_FRAMES)

    print(f"[INFO] Capturando {N_FRAMES} frames... mostre uma mao/dedo/braco para a camera!")
    prev_t = time.time()
    fps = 0.0
    saved = 0
    detections_count = 0

    for i in range(N_FRAMES):
        ok, frame = cap.read()
        if not ok:
            print(f"[WARN] frame {i} falhou")
            continue

        mp_hit, labels, frame = mp_det.process(frame)
        skin_hit, frame, _ = skin_det.process(frame, debug=False)
        if skin_hit:
            labels.append("PELE+MOV")

        raw = mp_hit or skin_hit
        alert = stabilizer.update(raw)
        if alert:
            detections_count += 1

        now = time.time()
        inst = 1.0 / max(now - prev_t, 1e-6)
        fps = 0.9 * fps + 0.1 * inst if fps > 0 else inst
        prev_t = now

        frame = draw_hud(frame, alert, labels, fps)

        status = "ALERTA" if alert else "livre"
        print(f"  frame {i:02d} | {status:6s} | fps={fps:4.1f} | origem={sorted(set(labels))}")

        if i % SAVE_EVERY == 0:
            path = os.path.join(OUT_DIR, f"frame_{i:02d}_{status}.jpg")
            cv2.imwrite(path, frame)
            saved += 1

    mp_det.close()
    cap.release()

    print(f"\n[RESULTADO] {saved} imagens salvas em: {OUT_DIR}")
    print(f"[RESULTADO] Frames com ALERTA: {detections_count}/{N_FRAMES}")
    print(f"[RESULTADO] FPS medio aproximado: {fps:.1f}")


if __name__ == "__main__":
    main()
