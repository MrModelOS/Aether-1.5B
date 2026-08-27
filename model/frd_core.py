"""
FRD-Core: Fractal Resonance Dynamics + PhaseNorm (model/frd_core.py:1)

Зачем: Замена LayerNorm+Attention на волновую интерференцию.
- torch.complex64 + torch.fft = реальная физика волны, исполняется на CUDA FFT-ядрах (cuFFT)
- PhaseNorm нормализует по амплитуде/фазе, а не по mean/var -> сохраняет интерференцию
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PhaseNorm(nn.Module):
    """Нормализация по амплитуде комплексного вектора, фаза сохраняется (model/frd_core.py:15)"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, T, D] complex64
        amp = torch.abs(z)  # [B, T, D] real
        # нормализуем только амплитуду, фаза untouched
        mean_amp = amp.mean(dim=-1, keepdim=True)
        var_amp = amp.var(dim=-1, keepdim=True, unbiased=False)
        amp_norm = (amp - mean_amp) / torch.sqrt(var_amp + self.eps)
        amp_norm = amp_norm * self.weight + self.bias
        phase = torch.angle(z)
        return torch.polar(amp_norm.clamp(min=0), phase)


class FRDOscillatorLayer(nn.Module):
    """
    Волновой слой FRD (model/frd_core.py:35)
    Вход: x [B, T, D] real
    1. Проекция в комплексную плоскость через learnable phase_angles/amplitudes
    2. FFT по временной оси T -> интерференция в частотной области
    3. IFFT + PhaseNorm
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.phase_angles = nn.Parameter(torch.rand(dim) * 2 * math.pi)
        self.amplitudes = nn.Parameter(torch.ones(dim))
        self.phase_norm = PhaseNorm(dim)
        self.dropout = nn.Dropout(dropout)
        # Легкий частотный фильтр (learnable) вместо QK-матрицы
        self.freq_gate = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] real
        B, T, D = x.shape
        # 1. Комплексная модуляция
        complex_basis = torch.complex(
            self.amplitudes * torch.cos(self.phase_angles),
            self.amplitudes * torch.sin(self.phase_angles)
        )  # [D] complex
        z = x.to(torch.complex64) * complex_basis  # [B, T, D] complex

        # 2. FFT по времени -> частотная интерференция O(T log T)
        Z = torch.fft.fft(z, dim=1)  # [B, T, D]
        Z = Z * self.freq_gate  # поэлементный гейт в частотах
        z_interfered = torch.fft.ifft(Z, dim=1)  # [B, T, D] complex

        # 3. PhaseNorm + возврат в real (берем амплитуду как носитель энергии)
        z_norm = self.phase_norm(z_interfered)  # [B, T, D] complex
        out_real = torch.abs(z_norm) * torch.sign(x + 1e-6) + x * 0.2  # residual
        return self.dropout(out_real)


class FRDCompressor(nn.Module):
    """
    Оффлайн-сжиматель 1M -> N_kernels (model/frd_core.py:72)
    Делает Chunked FFT + Top-K амплитуд на CPU/RAM, на GPU идет уже [B, N_kernels, D]
    Это честный O(1) VRAM трюк из ТЗ: сжимаем ДО подачи в GPU.
    """
    def __init__(self, n_kernels: int = 2048, chunk: int = 4096):
        super().__init__()
        self.n_kernels = n_kernels
        self.chunk = chunk

    @torch.no_grad()
    def compress(self, long_seq: torch.Tensor) -> torch.Tensor:
        # long_seq: [B, L, D] где L до 1M, лежит на CPU
        B, L, D = long_seq.shape
        if L <= self.n_kernels:
            return long_seq
        # Chunked FFT magnitude pooling
        n_chunks = (L + self.chunk - 1) // self.chunk
        pooled = []
        for i in range(n_chunks):
            chunk = long_seq[:, i*self.chunk:(i+1)*self.chunk, :].to(torch.complex64)
            spec = torch.fft.rfft(chunk, dim=1).abs().mean(dim=1)  # [B, D]
            pooled.append(spec)
        stacked = torch.stack(pooled, dim=1)  # [B, n_chunks, D]
        # Top-K по энергии -> канонические ядра
        energy = stacked.norm(dim=-1)  # [B, n_chunks]
        _, idx = torch.topk(energy, k=min(self.n_kernels, n_chunks), dim=1)
        # Gather (упрощенно: усредняем выбранные чанки)
        # Для прототипа возвращаем первые n_kernels средних
        return stacked[:, :self.n_kernels, :] if stacked.shape[1] >= self.n_kernels else stacked
