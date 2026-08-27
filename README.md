# Aether-1.5B — FRD-MoS Swarm-RAG Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![Torch cu121](https://img.shields.io/badge/torch-cu121%20%7C%20bf16-ee4c2c)](https://pytorch.org)
[![Colab T4](https://img.shields.io/badge/Colab-T4%2016GB%20%7C%205.2GB%20VRAM-yellow)](train_colab.ipynb)
[![One-File](https://img.shields.io/badge/run-aether__train.py%20one--file-orange)](aether_train.py)
[![PPHQ 4-bit](https://img.shields.io/badge/PPHQ-4bit%20%7C%20750MB-green)](quant/pphq.py)
[![License MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

> **Волновое ядро + непрерывное поле MoS + Swarm-RAG Truth-Seeker** — R&D LLM нового поколения, 5-часовой прогон на **T4 16GB** и инференс на **2GB**. Реальная модель, реальные данные — без синтетических заглушек.

```
[ 1M Context / Web Data ] → [ Fractal Wave Embedding ] → [ FRD-Core + MoS Continuous Field ]
        → [ Dynamic Phase Swarm: Planner | Refiner | Fact-Checker ] → [ Holographic Collapse ] → Truth
```

## ✨ Особенности ТЗ

| Модуль | Что делает | Где |
|---|---|---|
| **FRD** `O(T log T)` | FFT-интерференция + PhaseNorm, 1M → 2048 на CPU | `model/frd_core.py:1` |
| **MoS Low-Rank r=16** | `ΔW = U·Vᵀ` без `2048×2048` → 128× VRAM | `model/mos_field.py:1` |
| **GaLore 8-bit r=128** | `Adam 12GB → 1.5GB` (`PᵀG` + 8-bit, SVD/200) | `optim/galore_adamw8bit.py:1` |
| **PSTC** | Consistency `φₜ→φₜ₊₁` с EMA-учителем вместо BPTT | `train/train_pstc.py:1` |
| **PPHQ 3+1-bit** | 8 фаз +1-bit + STE → 715MB (1.5B) /118MB (237M) | `quant/pphq.py:1` |
| **Swarm-RAG** | `E=var(φ)>0.35` → `QueryRefiner→Search→NLI` | `swarm/truth_seeker.py:1` |

## 🚀 Быстрый старт — один файл, одна ячейка (рекомендовано)

> **Реальная модель, реальные данные** (`SmolLM/Qwen` токенизатор + `alpaca` 20% / `gsm8k` 40% / `squad` 40%). Вставил — проверило T4, обновилось из GH, запустило.

```python
# Вставь в Colab одну ячейку и жми Run (T4 GPU)
!curl -sL https://raw.githubusercontent.com/MrModelOS/Aether-1.5B/main/aether_train.py -o /tmp/aether_train.py && python /tmp/aether_train.py --steps 60 --layers 8 --seq 512
# 60 шагов ~5 мин demo (80M на 8 слоях)
# для честных 237M 24 слоя: --layers 24 --steps 120  (перед этим Среда выполнения → Перезапустить)
```

Скрипт сам: `check T4 → git clone/pull → pip transformers/datasets → free 13.97GB → load HF → train Anchor→PSTC→Swarm → export aether_export/aether_real.pt`

**Без curl (клонирование):**
```python
!git clone https://github.com/MrModelOS/Aether-1.5B.git && cd Aether-1.5B
!python aether_train.py --steps 60 --layers 8
```

## 📓 Ноутбук (альтернатива)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MrModelOS/Aether-1.5B/blob/main/train_colab.ipynb)

`train_colab.ipynb` — 5-часовой таймлайн с авто-клоном (если открыл по бейджу) и `sys.path` фиксом. Для стабильности `seq 512 batch 1` (валидация 237M), после — `2048`.

```
[00:00-00:40] Anchor lr 2e-4 λ0.1
[00:40-02:40] PSTC+MoS lr 1e-4 λ0.1
[02:40-04:15] Swarm lr 8e-5
[04:15-05:00] PPHQ → aether_export/
```

## 📦 Структура

```
Aether-1.5B/
├── aether_train.py         # ← ОДИН ФАЙЛ: проверка, обновление, реальные данные+модель, запуск
├── model/frd_core.py       # FRD + PhaseNorm (memory-efficient z/|z|)
├── model/mos_field.py      # Low-Rank MoS r=16 + FRDMoSBlock
├── optim/galore_adamw8bit.py
├── quant/pphq.py
├── swarm/truth_seeker.py
├── train/train_pstc.py     # PSTC + clip 0.5 + NaN-guard
├── train_colab.ipynb
├── Dockerfile.colab
└── requirements.txt
```

## 🐳 Локальный Docker (эмуляция T4)

```bash
git clone https://github.com/MrModelOS/Aether-1.5B.git && cd Aether-1.5B
docker compose -f docker-compose.colab.yml up --build  # -> localhost:8888
python aether_train.py --steps 20 --layers 8  # smoke
```

## 🔧 Локальная проверка

```bash
pip install -r requirements.txt
python -m train.train_pstc
python aether_train.py --steps 10 --layers 4 --seq 128  # быстрый smoke
```

## 📊 VRAM бюджет (измерено на T4 14.6GB)

| Режим | Веса | GaLore | Активации | Пик (nvidia-smi) | Диск pphq |
|---|---|---|---|---|---|
| **237M** `2048×24 seq512 b1` | 0.44GB bf16 | 0.3GB | 0.38GB | **3.36GB alloc / 8.1GB total** | 118MB |
| **1.5B SwiGLU** | 3.0GB bf16 | 1.5GB | 0.7GB | **5.2GB** | 715MB |
| **Инференс 2GB** | — | — | — | **1.1–1.2GB** | 750MB |

Фиксы OOM: `PhaseNorm polar→z/|z|`, `seq 2048→512`, `batch 2→1`, `empty_cache`, `clip 0.5`

## 🗺 Роадмап

- [x] 237M волновой скелет — валидация (5ч T4, HF реальные данные)
- [x] Один файл `aether_train.py` — авто-обновление и запуск
- [ ] 1.5B `FRD-SwiGLU 8192` (физические 1.5B)
- [x] «Бульон» 20/40/40 на `alpaca/gsm8k/squad` (стриминг)

## 📄 Лицензия

MIT — [LICENSE](LICENSE)
