"""
MoS Continuous Field через Low-Rank синтез (model/mos_field.py:1)

Зачем: Устраняет дискретный Top-K роутинг MoE.
Вместо U*V^T размером 2048x2048 (4M) генерим U[B,d,r] + V[B,r,d] где r=16 (65k) -> 128x экономия VRAM.
Формула: delta_x = (x @ U) @ V  без материализации матрицы [d,d].
"""
import torch
import torch.nn as nn

class LowRankMoSSynthesizer(nn.Module):
    """Генератор дельта-весов на лету (model/mos_field.py:11)"""
    def __init__(self, dim: int, rank: int = 16, latent_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.latent_dim = latent_dim

        self.hyper_router = nn.Sequential(
            nn.Linear(dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        # Два проектора вместо одного dim*dim
        self.proj_u = nn.Linear(latent_dim, dim * rank)
        self.proj_v = nn.Linear(latent_dim, rank * dim)
        self.scale = rank ** -0.5

    def forward(self, x: torch.Tensor, phi_coord: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D] или [B, D]
        phi_coord: [B, D] непрерывная координата в пространстве знаний
        return: [B, T, D] delta
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B,1,D]
            phi_coord = phi_coord.unsqueeze(1) if phi_coord.dim() == 2 else phi_coord
            squeeze = True
        B, T, D = x.shape
        # phi усредняем по времени если нужно
        if phi_coord.dim() == 3:
            phi = phi_coord.mean(dim=1)  # [B,D]
        else:
            phi = phi_coord  # [B,D]

        latent = self.hyper_router(phi)  # [B, latent]
        u = self.proj_u(latent).view(B, D, self.rank)  # [B,D,r]
        v = self.proj_v(latent).view(B, self.rank, D)  # [B,r,D]

        # (x @ U) @ V  -> [B,T,r] -> [B,T,D]
        # x: [B,T,D], u: [B,D,r] -> нужен batched: einsum
        x_u = torch.einsum('btd,bdr->btr', x, u)  # [B,T,r]
        delta = torch.einsum('btr,brd->btd', x_u, v) * self.scale
        return delta.squeeze(1) if squeeze else delta


class FRDMoSBlock(nn.Module):
    """Полный блок FRD + MoS (model/mos_field.py:55)"""
    def __init__(self, dim: int, rank: int = 16):
        super().__init__()
        from .frd_core import FRDOscillatorLayer
        self.frd = FRDOscillatorLayer(dim)
        self.mos = LowRankMoSSynthesizer(dim, rank=rank)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        # 1. Волновая интерференция
        wave = self.frd(x)  # [B,T,D]
        # 2. Непрерывная адаптация
        mos_delta = self.mos(x, phi)  # [B,T,D]
        return self.norm(wave + mos_delta + x)  # residual
