#!/usr/bin/env python3
"""
Aether-1.5B — единый скрипт обучения для Google Colab T4
Вставил в Colab → всё проверил, обновился, запустил

Запуск в Colab (одна ячейка):
!curl -sL https://raw.githubusercontent.com/MrModelOS/Aether-1.5B/main/aether_train.py -o /tmp/aether_train.py && python /tmp/aether_train.py --steps 120

Или локально:
python aether_train.py --steps 120 --batch 1 --seq 512
"""
import os, sys, pathlib, subprocess, gc, json, math, argparse, shutil

# --- 0. Авто-обновление из GH если запущен в /content ---
def self_update():
    try:
        # если мы в колабе и репо уже склонировано — pull
        for cand in [pathlib.Path("/content/Aether-1.5B"), pathlib.Path("Aether-1.5B")]:
            if (cand / "aether_train.py").exists():
                print(f"[update] git pull {cand}")
                subprocess.run(["git", "-C", str(cand), "pull", "--ff-only"], check=False)
                # копируем свежую версию в текущий файл если нужно
                src = cand / "aether_train.py"
                dst = pathlib.Path(__file__).resolve()
                if src.resolve() != dst.resolve() and src.exists():
                    shutil.copy(src, dst)
                    print(f"[update] copied {src} -> {dst}, restarting")
                    os.execv(sys.executable, [sys.executable, str(dst)] + sys.argv[1:])
        # если модель рядом не найдена и мы в /content без репо — клонируем
        if not pathlib.Path("model/frd_core.py").exists() and not pathlib.Path("/content/Aether-1.5B/model/frd_core.py").exists():
            if pathlib.Path("/content").exists():
                print("[update] cloning MrModelOS/Aether-1.5B -> /content/Aether-1.5B")
                subprocess.run(["git", "clone", "https://github.com/MrModelOS/Aether-1.5B.git", "/content/Aether-1.5B"], check=False)
                # добавляем в путь
                if "/content/Aether-1.5B" not in sys.path:
                    sys.path.insert(0, "/content/Aether-1.5B")
                os.chdir("/content/Aether-1.5B")
    except Exception as e:
        print(f"[update] warn: {e}")

self_update()

# добавляем корень в sys.path
for cand in [pathlib.Path.cwd(), pathlib.Path("/content/Aether-1.5B"), pathlib.Path(__file__).parent]:
    if (cand / "model" / "frd_core.py").exists() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Проверка железа ---
def check_hw():
    print(f"Python {sys.version.split()[0]} | Torch {torch.__version__} | CUDA {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
        print(f"VRAM free: {torch.cuda.mem_get_info()[0]/1024**3:.2f} GB")
    else:
        print("WARN: CUDA не найден — будет медленно на CPU")

# --- 2. Реальные данные ---
def get_tokenizer():
    try:
        from transformers import AutoTokenizer
        # маленький но реальный токенизатор, влезает в T4 (Qwen 1.5B токенизатор = 151k vocab тяжелый, берем SmolLM 360M ~49k)
        for name in ["HuggingFaceTB/SmolLM-360M", "Qwen/Qwen2.5-0.5B", "gpt2"]:
            try:
                tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
                if tok.pad_token is None: tok.pad_token = tok.eos_token
                print(f"[data] tokenizer: {name} vocab={len(tok)}")
                return tok, len(tok)
            except Exception as e:
                print(f"[data] tokenizer {name} fail: {e}")
    except ImportError:
        pass
    print("[data] transformers не установлен — fallback vocab 32000")
    return None, 32000

def get_real_datasets(tokenizer, seq_len=512, streaming=True):
    """Загружает реальные HF датасеты, фолбэк на синтетику если оффлайн"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[data] datasets не установлен — синтетика")
        return None

    # 3 реальных источника под 20/40/40
    # Anchor 20%: alpaca / ultrachat (вежливость)
    # Reasoning 40%: gsm8k + math
    # RAG 40%: squad / hotpot
    def tokenize_fn(ex):
        # ex может быть с полями instruction/input/output или question/answer
        txt = ""
        if "instruction" in ex: txt = (ex.get("instruction","") + " " + ex.get("input","") + " " + ex.get("output","")).strip()
        elif "question" in ex and "answer" in ex: txt = ex["question"] + " " + str(ex["answer"])
        elif "text" in ex: txt = ex["text"]
        else: txt = str(ex)
        if tokenizer:
            ids = tokenizer(txt, truncation=True, max_length=seq_len, padding="max_length")["input_ids"]
        else:
            import hashlib as h
            ids = [(h.md5(txt.encode()).digest()[i%16] % 32000) for i in range(seq_len)]
        return {"ids": ids}

    datasets = {}
    try:
        print("[data] loading tatsu-lab/alpaca (anchor 20%)...")
        ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=streaming)
        # берем 2000 примеров
        datasets["anchor"] = ds.take(2000) if streaming else ds.select(range(min(2000, len(ds))))
    except Exception as e:
        print(f"[data] alpaca fail: {e}")
    try:
        print("[data] loading gsm8k (reasoning 40%)...")
        ds = load_dataset("gsm8k", "main", split="train", streaming=streaming)
        datasets["reasoning"] = ds.take(2000) if streaming else ds.select(range(min(2000, len(ds))))
    except Exception as e:
        print(f"[data] gsm8k fail: {e}")
    try:
        print("[data] loading squad (rag 40%)...")
        ds = load_dataset("squad", split="train", streaming=streaming)
        datasets["rag"] = ds.take(2000) if streaming else ds.select(range(min(2000, len(ds))))
    except Exception as e:
        print(f"[data] squad fail: {e}")

    if not datasets:
        print("[data] все HF загрузки провалились — синтетика")
        return None

    # оборачиваем в torch Iterable
    class RealBroth(torch.utils.data.IterableDataset):
        def __init__(self, datasets, tok, seq_len):
            self.datasets = datasets; self.tok=tok; self.seq_len=seq_len
        def __iter__(self):
            import itertools, random
            # round-robin 20/40/40
            it_anchor = iter(self.datasets.get("anchor", []))
            it_reason = iter(self.datasets.get("reasoning", []))
            it_rag = iter(self.datasets.get("rag", []))
            while True:
                # 20% anchor
                for _ in range(2):
                    try: ex = next(it_reason)
                    except: it_reason = iter(self.datasets["reasoning"]); ex = next(it_reason)
                    yield self._to_tensor(ex)
                try: ex = next(it_anchor)
                except: it_anchor = iter(self.datasets["anchor"]); ex = next(it_anchor)
                yield self._to_tensor(ex)
                for _ in range(2):
                    try: ex = next(it_rag)
                    except: it_rag = iter(self.datasets["rag"]); ex = next(it_rag)
                    yield self._to_tensor(ex)
        def _to_tensor(self, ex):
            txt = ""
            if isinstance(ex, dict):
                if "instruction" in ex: txt = ex["instruction"] + " " + ex.get("input","") + " " + ex.get("output","")
                elif "question" in ex: txt = ex["question"] + " " + str(ex.get("answer",""))
                elif "text" in ex: txt = ex["text"]
                else: txt = json.dumps(ex, ensure_ascii=False)[:2000]
            if self.tok:
                ids = self.tok(txt, truncation=True, max_length=self.seq_len, padding="max_length")["input_ids"]
                ids = torch.tensor(ids, dtype=torch.long)
                phi = torch.randn(self.seq_len, 2048) * 0.8  # phi пока синтетика, но ids реальные
            else:
                import hashlib
                ids = torch.tensor([(hashlib.md5(txt.encode()).digest()[i%16] % 32000) for i in range(self.seq_len)], dtype=torch.long)
                phi = torch.randn(self.seq_len, 2048) * 0.8
            return ids, phi

    return RealBroth(datasets, tokenizer, seq_len)

# --- 3. Реальная модель ---
def build_model(vocab_size, dim=2048, layers=24, rank_mos=16):
    from model.mos_field import FRDMoSBlock
    class AetherMoS(nn.Module):
        def __init__(self, vocab, dim, layers, rank):
            super().__init__()
            self.embed = nn.Embedding(vocab, dim)
            self.blocks = nn.ModuleList([FRDMoSBlock(dim, rank=rank) for _ in range(layers)])
            self.lm_head = nn.Linear(dim, vocab, bias=False)
            # вес тайинг как в реальных LLM
            self.lm_head.weight = self.embed.weight
        def forward(self, x, phi):
            for blk in self.blocks: x = blk(x, phi)
            return x
        def forward_lm(self, input_ids, phi_seq):
            x = self.embed(input_ids)
            h = self.forward(x, phi_seq)
            return self.lm_head(h)
    m = AetherMoS(vocab_size, dim, layers, rank_mos)
    print(f"[model] AetherMoS vocab={vocab_size} dim={dim} layers={layers} rank={rank_mos} -> {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")
    return m

# --- 4. Тренировка ---
def train(args):
    check_hw()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # чистим VRAM
    gc.collect()
    if device.type=="cuda": torch.cuda.empty_cache()
    print(f"[train] device={device} steps={args.steps} batch={args.batch} seq={args.seq}")

    tokenizer, vocab = get_tokenizer()
    # если vocab большой (>100k) — усекаем dim до 2048 все равно ок, но можно уменьшить
    # для T4 237M dim 2048 vocab 32000 — норм

    # данные
    real_ds = get_real_datasets(tokenizer, seq_len=args.seq, streaming=True)
    if real_ds is None:
        print("[train] fallback синтетика (как раньше) — но это не реальные данные")
        import torch.utils.data as data
        class Synth(data.Dataset):
            def __len__(self): return 5000
            def __getitem__(self, idx):
                import random
                r = random.random()
                scale = 0.5 if r<0.2 else (1.0 if r<0.6 else 1.5)
                return torch.randint(0, vocab, (args.seq,)), torch.randn(args.seq, 2048)*scale
        loader = data.DataLoader(Synth(), batch_size=args.batch, shuffle=True)
        def loader_iter():
            while True:
                for b in loader: yield b
        it = loader_iter()
    else:
        loader = torch.utils.data.DataLoader(real_ds, batch_size=args.batch, num_workers=0)
        it = iter(loader)
        # для Iterable — бесконечный

    model = build_model(vocab, dim=args.dim, layers=args.layers, rank_mos=16).to(device)
    # опционально bf16
    use_bf16 = device.type=="cuda" and torch.cuda.is_bf16_supported()
    print(f"[train] bf16={use_bf16}")

    from train.train_pstc import PSTCTrainer
    trainer = PSTCTrainer(model, lr=args.lr, rank=128, ema_decay=0.999)
    # Стабильные lr
    STAGES = [
        ("Anchor", args.steps//3, 2e-4, 0.1),
        ("PSTC", args.steps//3, 1e-4, 0.1),
        ("Swarm", args.steps - 2*(args.steps//3), 8e-5, 0.1),
    ]
    global_step=0
    import time
    start=time.time()
    for name, steps, lr, lmb in STAGES:
        print(f"\n=== {name} lr={lr} lambda={lmb} steps={steps} ===")
        for g in trainer.optimizer.param_groups: g['lr']=lr
        for s in range(steps):
            try:
                ids, phi = next(it)
            except StopIteration:
                it = iter(loader); ids, phi = next(it)
            ids, phi = ids.to(device), phi.to(device)
            # phi должен быть [B,T,D] — если из real_ds phi [seq,D] то unsqueeze
            if phi.dim()==2: phi = phi.unsqueeze(0).expand(ids.shape[0], -1, -1)
            if phi.shape[-1] != args.dim:
                # проекция если dim не 2048
                phi = phi[..., :args.dim] if phi.shape[-1] > args.dim else F.pad(phi, (0, args.dim - phi.shape[-1]))
            stats = trainer.step(ids, phi, lambda_cons=lmb)
            global_step+=1
            if global_step % 5 == 0 or s==0:
                vram = torch.cuda.memory_allocated()/1024**3 if device.type=="cuda" else 0
                print(f"step {global_step:03d} loss {stats['loss']:.3f} lm {stats['lm']:.3f} cons {stats['cons']:.4f} VRAM {vram:.2f}GB")
            if global_step % 20 == 0 and device.type=="cuda":
                torch.cuda.empty_cache()
            if stats['loss'] != stats['loss']:  # NaN
                print("NaN loss — стоп")
                return
    print(f"\n[done] {global_step} steps in {(time.time()-start)/60:.1f} min")
    # экспорт
    out = pathlib.Path("aether_export"); out.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out / "aether_real.pt")
    print(f"[export] saved {out/'aether_real.pt'}")
    # PPHQ оценка
    try:
        from quant.pphq import estimate_pphq_size
        print(f"[export] PPHQ 237M -> {estimate_pphq_size(sum(p.numel() for p in model.parameters())):.0f} MB")
    except: pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60, help="всего шагов (60=~5мин demo, 500=~1ч)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=8, help="8=~80M быстро, 24=237M честно (нужен рестарт)")
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()
    # авто-установка зависимостей если нет
    try:
        import transformers, datasets
    except ImportError:
        print("[setup] pip install transformers datasets")
        subprocess.run([sys.executable, "-m", "pip", "-q", "install", "transformers", "datasets"], check=False)
    train(args)
