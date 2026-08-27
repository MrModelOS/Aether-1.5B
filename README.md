# Aether-1.5B — FRD-MoS (Swarm-RAG Engine)

Реализация ТЗ: волновое ядро + непрерывное поле MoS + PPHQ квантование, адаптированная под T4 16GB.

## Структура
- `model/frd_core.py` — FFT-осцилляторы + PhaseNorm + оффлайн-компрессор 1M->2048
- `model/mos_field.py` — Low-Rank MoS r=16 (128x экономия VRAM)
- `optim/galore_adamw8bit.py` — GaLore + 8-bit AdamW (~1.5GB вместо 12GB)
- `quant/pphq.py` — 3-bit phase +1-bit amp + STE
- `train/train_pstc.py` — PSTC Consistency Loss (EMA teacher)
- `swarm/truth_seeker.py` — E_thresh = var(phi) триггер поиска

## Запуск на T4
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m train.train_pstc  # sanity на CPU
```
VRAM бюджет: модель 1.5B bf16 ~3GB + GaLore ~1.5GB + активации ~0.7GB = 5.2GB (влезает в T4)
Инференс PPHQ: 750MB на диске, ~1.2GB VRAM
