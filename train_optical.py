#!/usr/bin/env python3
"""
Обучение DONN под трубу: 1M -> 2D трафарет -> OpticalLayer -> камера 160k
Запуск: python train_optical.py --steps 500 --batch 4 --seq 512
Colab: curl -sL https://raw.githubusercontent.com/MrModelOS/Aether-1.5B/main/train_optical.py -o /tmp/opt.py && python /tmp/opt.py --steps 500
"""
import torch, pathlib, argparse, math
from model.optical_donn import OpticalLayer, OpticalTokenizerHead

def make_2d_pattern(ids, size=(2160,3840)):
    # ids [B, T] -> 2D трафарет [B, Ny, Nx] (упрощение: tile + embed)
    B, T = ids.shape
    Ny, Nx = size
    # нормируем ids в [0,1] и тайлим
    pat = ids.float() / 160000.0  # vocab 160k
    # растягиваем T -> Ny*Nx via repeat
    pat = pat.unsqueeze(-1).expand(B, T, Nx*T//T) # placeholder
    # проще: adaptive 1D -> 2D via interpolate
    pat = pat.unsqueeze(1) # [B,1,T]
    pat = torch.nn.functional.interpolate(pat, size=Nx, mode='linear', align_corners=False) # [B,1,Nx]
    pat = pat.expand(-1, Ny, -1) # [B,Ny,Nx]
    return pat.clamp(0,1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", nargs=2, type=int, default=[2160,3840])
    ap.add_argument("--export", type=str, default="aether_export/phase_mask.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device} size {args.size} steps {args.steps}")

    layer = OpticalLayer(size=tuple(args.size)).to(device)
    head = OpticalTokenizerHead(cam_size=(1944,2592), vocab=160000, screen_size=tuple(args.size)).to(device)
    opt = torch.optim.AdamW(layer.parameters(), lr=args.lr, weight_decay=0.01)

    # синтетический датасет (замени на реальный токенизатор 160k)
    for step in range(1, args.steps+1):
        ids = torch.randint(0, 160000, (args.batch, args.seq), device=device)
        # мишень — следующий токен? упрощаем: target = ids[:, -1] (последний)
        target = ids[:, 0]  # для demo: предсказать первый токен из паттерна
        pattern = make_2d_pattern(ids, tuple(args.size)).to(device)
        loss, logits, intensity = torch.nn.functional.cross_entropy(head.intensity_to_logits(layer(pattern)), target), None, None
        # прямой вызов helper
        from model.optical_donn import optical_forward_loss
        loss, logits, intensity = optical_forward_loss(layer, head, pattern, target)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0); opt.step()
        if step % 20 == 0 or step==1:
            pred = logits.argmax(-1)
            acc = (pred==target).float().mean().item()
            print(f"step {step:04d} loss {loss.item():.3f} acc {acc:.3f} mean_int {intensity.mean().item():.3e}")

    pathlib.Path(args.export).parent.mkdir(parents=True, exist_ok=True)
    layer.export_png(args.export)
    print(f"[export] {args.export} 8-bit PNG для Экран2")
    torch.save(layer.state_dict(), "aether_export/optical.pt")
    print("[export] aether_export/optical.pt")

if __name__ == "__main__":
    main()
