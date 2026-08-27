# Aether-1.5B — FRD-MoS Swarm-RAG Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![Torch cu121](https://img.shields.io/badge/torch-cu121%20%7C%20bf16-ee4c2c)](https://pytorch.org)
[![Colab T4](https://img.shields.io/badge/Colab-T4%2016GB%20%7C%205.2GB%20VRAM-yellow)](train_colab.ipynb)
[![PPHQ 4-bit](https://img.shields.io/badge/PPHQ-4bit%20%7C%20750MB-green)](quant/pphq.py)
[![License MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

> **Волновое ядро + непрерывное поле MoS + Swarm-RAG Truth-Seeker** — R&D-прототип LLM нового поколения, адаптированный под 5-часовой прогон на бесплатной **NVIDIA T4 16GB** (Google Colab) и инференс на **2GB VRAM**.

```
[ 1M Context / Web Data ] → [ Fractal Wave Embedding ] → [ FRD-Core + MoS Continuous Field ]
        → [ Dynamic Phase Swarm: Planner | Refiner | Fact-Checker ] → [ Holographic Collapse ] → Truth
```

## ✨ Особенности ТЗ

| Модуль | Что делает | Где |
|---|---|---|
| **FRD** `O(T log T)` | FFT-интерференция + PhaseNorm вместо Attention, 1M → 2048 компрессия на CPU | `model/frd_core.py:1` |
| **MoS Low-Rank r=16** | `ΔW = U·Vᵀ` без материализации `2048×2048` → 128× экономия VRAM | `model/mos_field.py:1` |
| **GaLore 8-bit r=128** | `Adam 12GB → 1.5GB` через `PᵀG` + 8-bit моменты, SVD/200 шагов | `optim/galore_adamw8bit.py:1` |
| **PSTC** | Consistency Loss `φₜ→φₜ₊₁` с EMA-учителем вместо BPTT | `train/train_pstc.py:1` |
| **PPHQ 3+1-bit** | 8 фаз + 1-bit амплитуда + STE → 750MB (1.5B) / 118MB (237M) | `quant/pphq.py:1` |
| **Swarm-RAG** | `E=var(φ)>0.35` триггерит `QueryRefiner → Search → NLI-filter` | `swarm/truth_seeker.py:1` |

## 📦 Структура

```
Aether-1.5B/
├── model/
│   ├── frd_core.py        # FRD осцилляторы, PhaseNorm, FRDCompressor 1M→2048
│   └── mos_field.py       # Low-Rank MoS + FRDMoSBlock
├── optim/
│   └── galore_adamw8bit.py
├── quant/
│   └── pphq.py
├── swarm/
│   └── truth_seeker.py
├── train/
│   └── train_pstc.py      # 4-этапный PSTC тренер
├── train_colab.ipynb      # 5-часовой прогон T4 (рекомендовано)
├── Dockerfile.colab       # Эмуляция Colab T4 локально
├── docker-compose.colab.yml
└── requirements.txt
```

## 🚀 Быстрый старт (Colab T4) — скопировали → запустили

### Вариант A: 1-кликом в Colab (рекомендовано)

1. Открой [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MrModelOS/Aether-1.5B/blob/main/train_colab.ipynb)
   или вручную: `colab.research.google.com → Файл → Открыть блокнот → GitHub → MrModelOS/Aether-1.5B → train_colab.ipynb`
2. `Среда выполнения → Сменить среду выполнения → T4 GPU → Сохранить`
3. `Среда выполнения → Выполнить все` — прогон идет по таймлайну:

```
[00:00-00:40] Этап 1 Anchor (lr 1e-3, λ 0.1)
[00:40-02:40] Этап 2 PSTC+MoS (lr 5e-4, λ 0.5)
[02:40-04:15] Этап 3 Swarm-Harmonics (E_thresh 0.35)
[04:15-05:00] Этап 4 PPHQ экспорт → aether_export/aether_1.5b_bf16.pt (750MB pphq)
```

4. Забери модель: левый сайдбар `Файлы → aether_export/ → Скачать` или
   ```python
   from google.colab import drive; drive.mount('/content/drive')
   !cp -r aether_export /content/drive/MyDrive/Aether-1.5B
   ```

### Вариант B: клонирование + ручной запуск в Colab

```python
# Ячейка 1 — клонируем
!git clone https://github.com/MrModelOS/Aether-1.5B.git && cd Aether-1.5B && ls -R

# Ячейка 2 — зависимости (в Colab torch уже есть, это для чистого рантайма)
!pip -q install -r requirements.txt

# Ячейка 3 — sanity (CPU, 10 сек)
!python -m train.train_pstc
# Ожидаем: PSTC step: {'loss': ~10.9, ...}

# Ячейка 4 — запускаем ноутбук или напрямую:
#   Открой train_colab.ipynb и жми Выполнить все
#   или
!python -c "import train_colab"  # headless прогон
```

### Вариант C: локальный Docker (эмуляция T4)

```bash
git clone https://github.com/MrModelOS/Aether-1.5B.git && cd Aether-1.5B
docker compose -f docker-compose.colab.yml up --build
# -> http://localhost:8888/lab  (Jupyter)
# внутри контейнера: python -m train.train_pstc
```

## 🔧 Локальная проверка (без GPU)

```bash
pip install -r requirements.txt
python -m train.train_pstc          # PSTC smoke — ~2 сек на CPU
python -c "from model.mos_field import FRDMoSBlock; import torch; m=FRDMoSBlock(256); print(m(torch.randn(2,32,256), torch.randn(2,32,256)).shape)"
```

## 📊 VRAM бюджет

| Режим | Веса | Adam/GaLore | Активации | Пик | Диск |
|---|---|---|---|---|---|
| **237M** `dim=2048×24` (Этап 1) | 0.44GB bf16 | 0.3GB | 0.38GB | **<2GB** | 118MB pphq |
| **1.5B SwiGLU** (Этап 2) | 3.0GB bf16 | 1.5GB | 0.7GB | **5.2GB** | 715MB pphq |
| **Инференс 2GB** | — | — | — | **1.1–1.2GB** | 750MB |

## 🗺 Роадмап

- [x] Этап 1 — 237M волновой скелет (валидация всей математики, 5ч T4)
- [ ] Этап 2 — 1.5B `FRD-SwiGLU dim→8192` (физические 1.5B, тот же код)
- [ ] Датасет «Бульон Мышления» 2B токенов (20% Anchor / 40% Reasoning / 40% RAG)

## 📄 Лицензия

MIT — см. [LICENSE](LICENSE)
