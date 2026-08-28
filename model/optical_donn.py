"""
DONN: Diffractive Optical Neural Network — физический слой трубы
Экран1 (U0) -> 3см дифракция (Angular Spectrum) -> Экран2 phase mask e^{iφ} -> линза FFT -> |U|² камера OV5647
160k токенов -> зоны 2592x1944, Tang Nano 20K считывает argmax
"""
import math, torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

class OpticalLayer(nn.Module):
    """
    Физический Forward Pass трубы (model/optical_donn.py:1)
    size: (Ny, Nx) = (2160, 3840) 4K, wavelength 532nm, z 3cm, dx 8um
    """
    def __init__(self, size=(2160, 3840), wavelength=532e-9, z=0.03, dx=8e-6):
        super().__init__()
        self.size = size
        Ny, Nx = size
        # обучаемые веса — фаза Экран2 [0,2pi], 8.3M пикселей
        self.phase_weights = nn.Parameter(torch.rand(Ny, Nx) * 2 * math.pi)
        # ядро дифракции (угловой спектр) на 3 см
        kx = 2 * math.pi * fft.fftfreq(Nx, d=dx)
        ky = 2 * math.pi * fft.fftfreq(Ny, d=dx)
        KY, KX = torch.meshgrid(ky, kx, indexing='ij')
        k0 = 2 * math.pi / wavelength
        # kz может быть комплексным (эванесцентные волны)
        kz = torch.sqrt(torch.complex(k0**2 - KX**2 - KY**2, torch.zeros_like(KX)))
        propagator = torch.exp(1j * kz * z)  # [Ny,Nx] complex
        self.register_buffer('propagator', propagator)
        # шум железки (Hardware-in-the-Loop)
        self.register_buffer('cam_noise_std', torch.tensor(0.02))
        self.register_buffer('defocus_std', torch.tensor(0.005))

    def forward(self, input_pattern, add_noise=True, quant_8bit=False):
        """
        input_pattern: [B, Ny, Nx] float [0,1] — 2D трафарет с Экран1 (сжатый 1M контекст)
        return: intensity [B, Ny, Nx] — карта яркости на камере
        """
        # подготовка фазы для квантования 256 уровней (8-bit PNG)
        phase = self.phase_weights
        if quant_8bit:
            # STE квантование 8-bit: 256 уровней
            phase_q = torch.round(phase / (2*math.pi) * 255) / 255 * 2 * math.pi
            phase = phase + (phase_q - phase).detach()  # STE
        # 1. Поле после Экран1
        field = torch.complex(input_pattern, torch.zeros_like(input_pattern))  # [B,Ny,Nx]
        # 2. Пролет 3 см (дифракция)
        field_fft = fft.fft2(field, dim=(-2,-1))
        # propagator [Ny,Nx] -> [1,Ny,Nx]
        field_prop = fft.ifft2(field_fft * self.propagator.unsqueeze(0), dim=(-2,-1))
        if add_noise and self.training:
            field_prop = field_prop * torch.exp(1j * torch.randn_like(field_prop.real) * self.defocus_std)
        # 3. Экран2 phase mask
        phase_mask = torch.exp(1j * phase).unsqueeze(0)  # [1,Ny,Nx]
        field_after = field_prop * phase_mask
        # 4. Линза -> FFT + интенсивность
        focused = fft.fft2(field_after, dim=(-2,-1))
        intensity = torch.abs(focused) ** 2  # [B,Ny,Nx]
        if add_noise and self.training:
            intensity = intensity + torch.randn_like(intensity) * self.cam_noise_std * intensity.mean()
            intensity = intensity.clamp(min=0)
        return intensity

    def export_png(self, path="phase_mask.png"):
        """Сохранить фазу как 8-bit PNG для Экран2"""
        import numpy as np
        from PIL import Image
        phase = self.phase_weights.detach().cpu()
        img = (phase / (2*math.pi) * 255).round().clamp(0,255).to(torch.uint8).numpy()
        Image.fromarray(img).save(path)
        return path

class OpticalTokenizerHead(nn.Module):
    """Зонирование камеры 2592x1944 под 160k токенов"""
    def __init__(self, cam_size=(1944, 2592), vocab=160000, screen_size=(2160,3840)):
        super().__init__()
        self.cam_h, self.cam_w = cam_size
        self.vocab = vocab
        self.screen_h, self.screen_w = screen_size
        # сетка зон: примерно 400x400 зон -> 160k
        self.grid_h = int(math.sqrt(vocab * self.cam_h / self.cam_w))
        self.grid_w = (vocab + self.grid_h -1)// self.grid_h
        # зоны интерполируются на screen size
    def intensity_to_logits(self, intensity):
        # intensity [B, Ny, Nx] -> downsample к cam + pool по зонам -> logits [B, vocab]
        # упрощение: adaptive pool
        # intensity 2160x3840 -> cam 1944x2592 via interpolate
        x = F.interpolate(intensity.unsqueeze(1), size=(self.cam_h, self.cam_w), mode='bilinear', align_corners=False) # [B,1,H,W]
        # pool в grid
        logits = F.adaptive_avg_pool2d(x, (self.grid_h, self.grid_w)) # [B,1,gh,gw]
        logits = logits.flatten(1)[:, :self.vocab] # [B, vocab]
        return logits

# --- Training helper ---
def optical_forward_loss(layer: OpticalLayer, head: OpticalTokenizerHead, input_pattern, target_ids):
    intensity = layer(input_pattern, add_noise=True, quant_8bit=False)
    logits = head.intensity_to_logits(intensity)
    loss = F.cross_entropy(logits, target_ids)
    return loss, logits, intensity
