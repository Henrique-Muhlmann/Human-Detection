<div align="center">

# Human Detection — Proteção de Célula Robotizada

### Sistema de Visão Computacional para detecção de partes humanas em tempo real

Detecta **braço · antebraço · mão · dedos** entrando no campo de visão de uma câmera *top-down*
sobre uma célula robótica, disparando alerta visual imediato.

<br>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks_API-00A6FB?style=for-the-badge&logo=google&logoColor=white)
![Status](https://img.shields.io/badge/Status-Protótipo_v1-success?style=for-the-badge)
![Tempo Real](https://img.shields.io/badge/Tempo_Real-Webcam_%2F_USB-orange?style=for-the-badge)

</div>

---

## ⚠️ Aviso de Segurança Funcional — LEIA PRIMEIRO

> [!WARNING]
> **Este software NÃO é um dispositivo de segurança certificado.**
> Ele **não substitui** cortinas de luz, scanners a laser ou relés de segurança
> exigidos pelas normas **ISO 13849-1** e **IEC 61496**.
>
> Use-o como **camada auxiliar de alerta**, **protótipo de P&D** e **coletor de dataset**.
> A parada física do robô deve **sempre** depender de hardware de segurança certificado,
> capaz de levar o robô a parada segura (categoria 0/1) por meio de um CLP/relé de segurança.

---

## 🎯 Visão Geral

| Item | Descrição |
|------|-----------|
| **Cenário** | Câmera RGB instalada acima da mesa, apontada verticalmente para baixo (*top-down*). |
| **Objetivo** | Detectar **imediatamente** qualquer parte do corpo humano no campo de visão. |
| **Partes alvo** | Braço, antebraço, mão e **dedos** (mesmo que apenas um dedo esteja visível). |
| **Prioridades** | Baixa latência · Alta sensibilidade · Mínimo falso-negativo. |
| **Plataforma** | Python · Webcam ou câmera USB · Tempo real. |

---

## 🧠 Arquitetura — Defesa em Profundidade

O sistema combina **3 camadas redundantes**. Se **qualquer uma** detectar presença humana, o **alerta dispara**.
Essa redundância minimiza o **falso-negativo** (não detectar um humano presente) — a falha perigosa num sistema de segurança.

```
                   ┌──────────────────────────────────────────────┐
   Câmera ─────▶   │   FRAME (BGR)                                 │
   top-down        └──────────────────────────────────────────────┘
                            │              │               │
                            ▼              ▼               ▼
                  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
                  │  CAMADA 1    │ │  CAMADA 2    │ │   CAMADA 3       │
                  │ MediaPipe    │ │ MediaPipe    │ │ Pele + Movimento │
                  │ Hands        │ │ Pose         │ │ (OpenCV)         │
                  │ mão / dedos  │ │ braço/       │ │ rede de          │
                  │              │ │ antebraço    │ │ segurança        │
                  └──────┬───────┘ └──────┬───────┘ └────────┬─────────┘
                         │                │                  │
                         └────────────────┼──────────────────┘
                                          ▼
                                ┌───────────────────┐
                                │  OR  +  Histerese  │  ──▶  🔴 ALERTA
                                └───────────────────┘
```

| Camada | Tecnologia | O que captura | Pontos fortes / limitações |
|:------:|------------|---------------|----------------------------|
| **1** | MediaPipe **Hands** (Tasks API) | Mão e dedos — 21 landmarks | Alta precisão; falha com luva / motion blur. |
| **2** | MediaPipe **Pose** (Tasks API) | Braço/antebraço (ombro→cotovelo→punho) | Bom para membros; espera ver parte do corpo. |
| **3** | **OpenCV** (pele YCrCb+HSV ∩ movimento MOG2) | Qualquer pele que se move | Cobre o que as redes neurais perdem (dedo parcial, borda, blur). |

---

## 📦 Estrutura do Projeto

```
Human Detection/
├── human_safety_detector.py     # ★ Sistema principal (janela ao vivo)
├── test_headless.py             # Teste sem GUI (validação/dataset)
├── hand_landmarker.task         # Modelo MediaPipe (mão) — Tasks API
├── pose_landmarker_lite.task    # Modelo MediaPipe (pose) — Tasks API
├── test_output/                 # Saídas de validação headless
└── README.md
```

---

## 🚀 Instalação

> [!IMPORTANT]
> O **MediaPipe ≥ 0.10.3x removeu a API legada** `mp.solutions`. Este projeto usa a
> **Tasks API** (`mediapipe.tasks`), que exige os arquivos `hand_landmarker.task` e
> `pose_landmarker_lite.task` no diretório do projeto (já incluídos).

```bash
# 1. Dependências
pip install opencv-python mediapipe numpy

# 2. (Se precisar rebaixar os modelos)
#   hand:  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
#   pose:  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

**Ambiente validado:** Python 3.12 · OpenCV 4.13 · NumPy 2.x · MediaPipe 0.10.35 · Windows 11.

---

## ▶️ Como Usar

```bash
python human_safety_detector.py
```

| Tecla | Ação |
|:-----:|------|
| `q` | Sair |
| `d` | Liga/desliga a visualização da máscara de pele+movimento (debug) |

### Legenda visual na tela

| Elemento | Significado |
|----------|-------------|
| 🟢 Esqueleto verde + 🟣 pontas rosa | Mão / dedos (Camada 1) |
| 🟡 Linha amarela ombro→cotovelo→punho | Braço / antebraço (Camada 2) |
| 🟠 Caixa laranja `PELE+MOV` | Pele em movimento (Camada 3) |
| 🔴 Borda vermelha + banner `ALERTA` | Presença humana confirmada |
| `AREA LIVRE` (verde) | Nenhuma detecção |

---

## ⚙️ Configuração

Todos os parâmetros ficam na classe `Config` em `human_safety_detector.py`:

```python
CAMERA_INDEX        = 0      # índice da webcam/câmera USB
FRAME_WIDTH/HEIGHT  = 1280x720
FLIP_HORIZONTAL     = True   # espelha a imagem (modo "espelho")
HANDS_DET_CONF      = 0.4    # confiança mín. da mão (↓ = mais sensível)
POSE_DET_CONF       = 0.4    # confiança mín. da pose
SKIN_MIN_AREA       = 2500   # área mín. (px) de blob de pele p/ alertar
SKIN_REQUIRES_MOTION= True   # exige movimento junto à pele (↓ falso-positivo)
ALERT_ON_FRAMES     = 2      # frames p/ ligar o alerta (histerese)
ALERT_OFF_FRAMES    = 8      # frames p/ desligar o alerta
```

> 💡 **Dica de calibração:** sob iluminação industrial estável, a camada de pele fica
> muito precisa. Se houver luvas ou peças cor de pele, ajuste `SKIN_MIN_AREA` e priorize
> as camadas neurais — ou migre para o YOLO customizado (abaixo).

---

## 🛣️ Roadmap — Versão de Produção (YOLO11 customizado)

A v1 atual já resolve o protótipo **e coleta o dataset**. Caminho recomendado para produção:

1. **Capturar** milhares de frames *top-down* reais (mãos, dedos parciais na borda, braços, com/sem luva, várias iluminações, com motion blur).
2. **Rotular** 4 classes — `mao`, `dedo`, `braco`, `antebraco` (Roboflow / CVAT).
3. **Treinar YOLO11** (não YOLOv12):
   ```python
   from ultralytics import YOLO
   model = YOLO("yolo11n.pt")
   model.train(data="celula.yaml", epochs=100, imgsz=960, batch=16)
   ```
4. **Exportar** para TensorRT/ONNX (latência mínima na borda) e manter a camada
   pele+movimento como *fallback* de segurança.

> **Por que YOLO11 e não YOLOv12?** A própria Ultralytics recomenda YOLO11/YOLO26 para
> produção — o YOLOv12 (atenção pesada) tem pior throughput em CPU e instabilidade de
> treino, indesejáveis num sistema de segurança.

---

## 📊 Comparativo de Arquiteturas

| Solução | Dedo/mão parcial | Braço | Top-down | Latência | Veredito |
|---------|:---------------:|:-----:|:--------:|:--------:|----------|
| MediaPipe Hands | mão sim / dedo isolado não | ❌ | 🟡 | ⚡⚡⚡ | Preciso, frágil sozinho |
| MediaPipe Pose | ❌ | ✅ | 🟡 | ⚡⚡ | Complementar |
| **YOLO11 (treinado)** | ✅ | ✅ | ✅ | ⚡⚡ | **Melhor p/ produção** |
| YOLOv12 | ✅ | ✅ | ✅ | ⚡ | Desaconselhado agora |
| Detectron2 | ✅ | ✅ | ✅ | 🐢 | Overkill p/ borda |

---

## 📚 Referências

- [MediaPipe Hands — documentação](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html)
- [Ultralytics YOLO11 / YOLO12 — docs](https://docs.ultralytics.com/models/yolo12)
- ISO 13849-1 · IEC 61496-1/-2 — segurança de máquinas (ESPE)

---

<div align="center">

**Desenvolvido com foco em segurança industrial e visão computacional.**

*Protótipo v1*

</div>
