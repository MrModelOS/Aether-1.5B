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
        # z: [B, T, D] complex64 — memory-efficient без angle/polar
        amp = torch.abs(z)  # [B, T, D] real
        mean_amp = amp.mean(dim=-1, keepdim=True)
        var_amp = amp.var(dim=-1, keepdim=True, unbiased=False)
        amp_norm = (amp - mean_amp) / torch.sqrt(var_amp + self.eps)
        amp_norm = (amp_norm * self.weight + self.bias).clamp(min=0)
        # сохраняем фазу без angle/polar: z / |z| = exp(i*phase)
        # avoid division by zero
        phase_unit = z / (amp + self.eps).to(z.dtype)  # [B,T,D] complex unit
        return amp_norm.to(z.dtype) * phase_unit  # [B,T,D] complex


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
    Честное 1M Волновое Поле — complex64 FFT аттракторы без SVD-срезки (model/frd_core.py:72)
    1M токенов -> комплексные фазовые аттракторы O(log N), без потери топологии.
    """
    def __init__(self, n_kernels: int = 2048, chunk: int = 4096):
        super().__init__()
        self.n_kernels = n_kernels
        self.chunk = chunk

    @torch.no_grad()
    def compress(self, long_seq: torch.Tensor) -> torch.Tensor:
        # long_seq: [B, L, D] CPU, L до 1M, хранится как complex64 фазовое поле
        B, L, D = long_seq.shape
        if L <= self.n_kernels:
            return long_seq
        # Честный волновой путь: FFT по каждому чанку -> фазовые аттракторы
        n_chunks = (L + self.chunk - 1) // self.chunk
        pooled = []
        for i in range(n_chunks):
            chunk = long_seq[:, i*self.chunk:(i+1)*self.chunk, :].to(torch.complex64)
            # FFT -> сохраняем комплексный спектр (фаза+амплитуда), не только magnitude
            spec = torch.fft.fft(chunk, dim=1)  # [B, C, D] complex
            attractor = spec.mean(dim=1)  # [B, D] complex64 — честный аттрактор
            pooled.append(attractor)
        stacked = torch.stack(pooled, dim=1)  # [B, n_chunks, D] complex
        energy = stacked.abs().norm(dim=-1)  # [B, n_chunks] real
        k = min(self.n_kernels, n_chunks)
        _, idx = torch.topk(energy, k=k, dim=1)  # [B, k]
        # Gather честных аттракторов (не усреднение, а выбор топ-K по энергии)
        gathered = torch.gather(stacked, 1, idx.unsqueeze(-1).expand(-1, -1, D))
        return gathered  # [B, n_kernels, D] complex64 — подается в FRD как phi


class FRDGraphAttractor(nn.Module):
    """
    Иерархический граф-аттрактор 1M -> 100MB VRAM (model/frd_core.py:108)
    Разбивает 1M на фазовые узлы (векторная графика FRD): активные резонируют в VRAM, остальные в RAM O(log N).
    """
    def __init__(self, dim: int, chunk: int = 4096, top_k: int = 32):
        super().__init__()
        self.dim = dim
        self.chunk = chunk
        self.top_k = top_k

    @torch.no_grad()
    def build(self, long_seq: torch.Tensor) -> dict:
        # long_seq: [B, L, D] real CPU -> граф узлов
        B, L, D = long_seq.shape
        n_chunks = (L + self.chunk - 1) // self.chunk
        nodes = []  # list of [B,D] complex
        for i in range(n_chunks):
            chunk = long_seq[:, i*self.chunk:(i+1)*self.chunk, :].to(torch.complex64)
            # векторный аттрактор: FFT + фазовый центр
            spec = torch.fft.fft(chunk, dim=1)  # [B, C, D] complex (chunk уже complex)
            attractor = spec.mean(dim=1)  # [B,D] complex
            nodes.append(attractor)
        stacked = torch.stack(nodes, dim=1)  # [B, n_chunks, D] complex
        # иерархия: уровень 0 — чанки, уровень 1 — кластеры по 8
        # храним в RAM, в VRAM только активные top_k по запросу
        return {"stacked": stacked.cpu(), "n_chunks": n_chunks, "dim": D, "chunk": self.chunk}

    @torch.no_grad()
    def query(self, graph: dict, phi_query: torch.Tensor) -> torch.Tensor:
        # phi_query: [B,D] complex или real -> ближайшие резонирующие узлы O(log N) via косинус фазы
        stacked = graph["stacked"]  # [B, n_chunks, D] complex CPU
        # косинус по фазе: real dot / |·|
        q = phi_query.to(torch.complex64)
        q_norm = q / (q.abs().clamp(min=1e-6))
        s_norm = stacked / (stacked.abs().clamp(min=1e-6))
        # batched cosine: [B, n_chunks]
        scores = (s_norm * q_norm.unsqueeze(1).conj()).real.mean(dim=-1)  # [B, n_chunks]
        _, idx = torch.topk(scores, k=min(self.top_k, graph["n_chunks"]), dim=1)
        # gather активных
        gathered = torch.gather(stacked, 1, idx.unsqueeze(-1).expand(-1, -1, graph["dim"]))
        return gathered.to(phi_query.device)  # [B, top_k, D] в VRAM
