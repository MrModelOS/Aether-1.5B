"""
GaLore-AdamW 8-bit (optim/galore_adamw8bit.py:1)

Зачем: AdamW m,v = 12 ГБ для 1.5B. GaLore проецирует градиент в low-rank подпространство.
- G [d,d] -> P^T G Q где P,Q из SVD(G), храним m,v в [r,d] -> ~1.5 ГБ
- 8-bit квантование моментов (dynamic per-tensor) -> еще ~2x сжатие
- SVD каждые 200 шагов (дорого, делаем редко)

Основано на: Zhao et al. GaLore (2024). Упрощенная но рабочая реализация для T4.
"""
import math
import torch
from torch.optim import Optimizer

def quantize_8bit(tensor: torch.Tensor):
    """Симметричное 8-bit квантование (optim/galore_adamw8bit.py:15)"""
    amax = tensor.abs().max().clamp(min=1e-8)
    scale = 127.0 / amax
    q = (tensor * scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale

def dequantize_8bit(q: torch.Tensor, scale: float):
    return q.float() / scale

class GaLoreAdamW8bit(Optimizer):
    """GaLore + 8-bit AdamW (optim/galore_adamw8bit.py:24)"""
    def __init__(self, params, lr=5e-4, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, rank=128, update_proj_gap=200, scale=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        rank=rank, update_proj_gap=update_proj_gap, scale=scale)
        super().__init__(params, defaults)
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.step_count = 0

    @torch.no_grad()
    def _get_projectors(self, grad: torch.Tensor, rank: int):
        # grad: [out, in] 2D. Для 1D/иных — без проекции
        if grad.dim() != 2 or min(grad.shape) < rank:
            return None, None
        try:
            # экономный SVD на float32 (на T4 приемлемо каждые 200 шагов)
            U, _, Vt = torch.linalg.svd(grad.float(), full_matrices=False)
            P = U[:, :rank]   # [out, r]
            Q = Vt[:rank, :].T  # [in, r] ??? для двухсторонней проекции
            # Упрощаем до односторонней проекции (как в оригинале для больших матриц)
            return P.to(grad.device).to(grad.dtype), None
        except Exception:
            return None, None

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        self.step_count += 1

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("GaLore не поддерживает sparse grad")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Проекторы
                    if grad.dim() == 2 and min(grad.shape) >= group['rank']:
                        P, _ = self._get_projectors(grad, group['rank'])
                        state['P'] = P
                    else:
                        state['P'] = None
                    # Инициализируем моменты в проекционном пространстве
                    if state['P'] is not None:
                        # G_proj = P^T G  -> [r, in]
                        proj_shape = (group['rank'], grad.shape[1])
                    else:
                        proj_shape = grad.shape
                    state['m_q'], state['m_scale'] = quantize_8bit(torch.zeros(proj_shape, device=grad.device))
                    state['v_q'], state['v_scale'] = quantize_8bit(torch.zeros(proj_shape, device=grad.device))

                # Обновляем проекторы каждые gap шагов
                if state['P'] is not None and self.step_count % group['update_proj_gap'] == 0:
                    P, _ = self._get_projectors(grad, group['rank'])
                    if P is not None:
                        state['P'] = P

                # Проекция градиента
                if state['P'] is not None:
                    # grad_proj = P^T @ grad  [r, in]
                    grad_proj = state['P'].T @ grad  # [r, in]
                else:
                    grad_proj = grad

                # Де-квантуем моменты
                m = dequantize_8bit(state['m_q'], state['m_scale']).to(grad.device)
                v = dequantize_8bit(state['v_q'], state['v_scale']).to(grad.device)

                state['step'] += 1
                beta1, beta2 = group['betas']
                m.mul_(beta1).add_(grad_proj, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad_proj, grad_proj, value=1 - beta2)

                # Квантуем обратно
                state['m_q'], state['m_scale'] = quantize_8bit(m)
                state['v_q'], state['v_scale'] = quantize_8bit(v)

                # Bias correction
                bias1 = 1 - beta1 ** state['step']
                bias2 = 1 - beta2 ** state['step']
                m_hat = m / bias1
                v_hat = v / bias2

                # Шаг в проекционном пространстве
                update_proj = m_hat / (v_hat.sqrt() + group['eps'])
                if group['weight_decay'] != 0:
                    # decoupled decay в проекции? применяем к p напрямую ниже
                    pass

                # Проецируем обратно: delta = P @ update_proj
                if state['P'] is not None:
                    update = state['P'] @ update_proj  # [out, in]
                else:
                    update = update_proj

                # Weight decay (decoupled)
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                p.data.add_(update, alpha=-group['lr'] * group['scale'])

        return loss
