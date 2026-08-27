"""
PPHQ: Phase-Polar Holographic Quantization 3-bit Phase + 1-bit Amp (quant/pphq.py:1)

Зачем: Сжать 1.5B до 750 МБ (4 бита/параметр) с сохранением фазы.
- 3 бита фаза = 8 углов по 45° (0°,45°,...,315°)
- 1 бит амплитуда = активен/неактивен (порог)
- STE (Straight-Through Estimator) позволяет обучаться сквозь round() -> градиент проходит как identity

Формат хранения: uint8 где [0:2]=phase_idx, [3]=amp_bit. Упаковка 2 веса в 1 байт.
"""
import math
import torch
import torch.nn as nn

class PPHQSTE(torch.autograd.Function):
    """STE для фазового квантования (quant/pphq.py:14)"""
    @staticmethod
    def forward(ctx, weight: torch.Tensor):
        # weight: real valued, интерпретируем как угол через atan? Упрощаем: weight -> фаза
        # Нормируем вес в [-pi, pi]
        # Для прототипа: считаем что weight уже в радианах (phase_angles)
        # Квантуем: 8 уровней
        phase_idx = torch.round(weight / (math.pi / 4))  # шаг 45°
        phase_idx = torch.remainder(phase_idx, 8)
        q_phase = phase_idx * (math.pi / 4)  # обратно в радианы
        # STE: forward квантует, backward пропускает градиент
        return q_phase

    @staticmethod
    def backward(ctx, grad_output):
        # STE: dL/dw = dL/dq
        return grad_output

class PPHQLinear(nn.Module):
    """Линейный слой с PPHQ-эмуляцией (quant/pphq.py:35)"""
    def __init__(self, in_features: int, out_features: int, amp_thresh: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.amp_thresh = amp_thresh
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.phase = nn.Parameter(torch.rand(out_features, in_features) * 2 * math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Квантуем фазу через STE
        q_phase = PPHQSTE.apply(self.phase)  # [out,in]
        # 1-bit амплитуда: gate = (abs(weight) > thresh)
        amp_gate = (self.weight.abs() > self.amp_thresh).float()  # STE упрощен
        # Эффективный вес = amp_gate * cos(q_phase)  (проекция фазы на real ось)
        w_eff = amp_gate * torch.cos(q_phase) * self.weight.abs().clamp(min=1e-6)
        # STE для amp_gate: в backward считаем как identity (не блокируем)
        return x @ w_eff.T

    def export_pphq(self) -> dict:
        """Экспорт в 4-бит формат (quant/pphq.py:57)"""
        with torch.no_grad():
            phase_idx = torch.round(self.phase / (math.pi / 4)).remainder(8).to(torch.uint8)  # 0..7 (3 бита)
            amp_bit = (self.weight.abs() > self.amp_thresh).to(torch.uint8)  # 0/1 (1 бит)
            packed = (phase_idx & 0x07) | ((amp_bit & 0x01) << 3)  # [0:2] phase, [3] amp
            # Упаковка 2x4-bit в 1 byte
            flat = packed.view(-1)
            if flat.numel() % 2 == 1:
                flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)])
            high = flat[0::2] & 0x0F
            low = flat[1::2] & 0x0F
            bytes_packed = (high << 4) | low
            return {
                "bytes": bytes_packed.cpu().numpy().tobytes(),
                "shape": tuple(self.weight.shape),
                "numel": self.weight.numel(),
                "size_mb": bytes_packed.numel() / 1024 / 1024
            }

def estimate_pphq_size(num_params: int) -> float:
    """Оценка размера в МБ (quant/pphq.py:78)"""
    return num_params * 0.5 / 1024 / 1024  # 4 бита = 0.5 байта
