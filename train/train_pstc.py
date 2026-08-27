"""
PSTC: Phase-Space Trajectory Collapse через Consistency Loss (train/train_pstc.py:1)

Зачем: Честная замена "обучению без backprop".
- Учитель (EMA) дает траекторию phi_{t+1} = Teacher(phi_t)
- Студент учится схлопывать phi_t -> phi_{t+1} за один шаг (consistency), отсекая BPTT
- + обычный LM loss на "Бульоне Мышления"

Экономия VRAM: не храним граф на всю траекторию, только один шаг + EMA (без градиента)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class PSTCTrainer:
    """Тренер PSTC (train/train_pstc.py:15)"""
    def __init__(self, model: nn.Module, lr=5e-4, rank=128, ema_decay=0.999):
        self.model = model
        self.teacher = deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.ema_decay = ema_decay

        from optim.galore_adamw8bit import GaLoreAdamW8bit
        # GaLore rank 128 -> ~1.5ГБ вместо 12ГБ
        self.optimizer = GaLoreAdamW8bit(model.parameters(), lr=lr, rank=rank, update_proj_gap=200)

    @torch.no_grad()
    def _ema_update(self):
        for p, pt in zip(self.model.parameters(), self.teacher.parameters()):
            pt.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    def step(self, input_ids: torch.Tensor, phi_seq: torch.Tensor, lambda_cons=0.5):
        """
        input_ids: [B, T] токены
        phi_seq: [B, T, D] координаты MoS (из FRD-компрессора или эмбеддера)
        """
        self.model.train()
        B, T, D = phi_seq.shape

        # 1. LM loss (стандартный next-token)
        # Для прототипа: модель возвращает logits [B,T,vocab] (заглушка)
        # Реально: нужна голова lm_head
        logits = self.model.forward_lm(input_ids, phi_seq) if hasattr(self.model, 'forward_lm') else None
        if logits is not None:
            lm_loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                      input_ids[:, 1:].reshape(-1), ignore_index=-100)
        else:
            # Заглушка если forward_lm не реализован: MSE на phi как self-supervised
            lm_loss = F.mse_loss(phi_seq[:, 1:], phi_seq[:, :-1].detach()) * 0.1

        # 2. Consistency Loss: phi_{t+1}_student ~ phi_{t+1}_teacher
        # FRDMoSBlock ожидает [B,T,D], поэтому не flatten а слайс по T
        with torch.no_grad():
            phi_teacher = self.teacher(phi_seq[:, :-1], phi_seq[:, :-1])
            if isinstance(phi_teacher, tuple):
                phi_teacher = phi_teacher[0]

        phi_student = self.model(phi_seq[:, :-1], phi_seq[:, :-1])
        if isinstance(phi_student, tuple):
            phi_student = phi_student[0]

        cons_loss = F.mse_loss(phi_student, phi_teacher.detach())

        loss = lm_loss + lambda_cons * cons_loss

        self.optimizer.zero_grad()
        loss.backward()
        # FIX2: жесткий клип 0.5 + NaN чек для 24x2048 стабильности
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        # NaN/Inf защита
        has_nan = False
        for p in self.model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                has_nan = True
                break
        if has_nan:
            self.optimizer.zero_grad()
            return {"loss": float("nan"), "lm": float("nan"), "cons": cons_loss.item()}
        self.optimizer.step()
        self._ema_update()

        return {"loss": loss.item(), "lm": lm_loss.item(), "cons": cons_loss.item()}

# Пример 5-часового таймлайна на T4 (train/train_pstc.py:75)
# [00:00-00:40] Anchor Kernel (lr 1e-3, lambda_cons 0.1)
# [00:40-02:40] PSTC+MoS (lr 5e-4, lambda_cons 0.5)
# [02:40-04:15] Swarm-Harmonics (включаем swarm/truth_seeker)
# [04:15-05:00] Export PPHQ (quant/pphq.py)

if __name__ == "__main__":
    # Sanity на CPU
    from model.mos_field import FRDMoSBlock
    dim = 256
    model = FRDMoSBlock(dim)
    # Добавляем заглушку forward_lm для теста
    def forward_lm(self, ids, phi):
        # phi [B,T,D] -> logits [B,T, vocab=32000]
        B,T,D = phi.shape
        return torch.randn(B,T,32000)
    import types
    model.forward_lm = types.MethodType(forward_lm, model)

    trainer = PSTCTrainer(model, lr=5e-4)
    ids = torch.randint(0, 32000, (2, 128))
    phi = torch.randn(2, 128, dim)
    stats = trainer.step(ids, phi)
    print(f"PSTC step: {stats}")
