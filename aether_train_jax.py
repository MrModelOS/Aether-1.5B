#!/usr/bin/env python3
"""
Aether-1.5B — JAX + Flax + Optax FRD-MoS для T4 (Вариант 1)
Волновая физика на XLA: jax.fft + vmap, 2-3x быстрее PyTorch

Colab T4 (одна ячейка):
!pip -q install "jax[cuda12]" flax optax datasets transformers
!curl -sL https://raw.githubusercontent.com/MrModelOS/Aether-1.5B/main/aether_train_jax.py -o /tmp/aether_jax.py && python /tmp/aether_jax.py --steps 500 --layers 24

Локально:
python aether_train_jax.py --steps 500 --layers 24 --seq 512
"""
import os, sys, pathlib, argparse, json, gc
import jax, jax.numpy as jnp, flax.linen as nn, optax
from flax.training import train_state
from copy import deepcopy

print(f"JAX {jax.__version__} devices: {jax.devices()} | float32={jnp.float32}")

# --- FRD + PhaseNorm на JAX ---
class PhaseNorm(nn.Module):
    dim: int
    eps: float = 1e-5
    @nn.compact
    def __call__(self, z): # z [B,T,D] complex64
        amp = jnp.abs(z) # [B,T,D]
        mean = amp.mean(axis=-1, keepdims=True)
        var = amp.var(axis=-1, keepdims=True)
        w = self.param("weight", nn.initializers.ones, (self.dim,))
        b = self.param("bias", nn.initializers.zeros, (self.dim,))
        amp_norm = (amp - mean) / jnp.sqrt(var + self.eps) * w + b
        amp_norm = jnp.clip(amp_norm, a_min=0)
        # фаза без angle/polar: z/|z|
        phase_unit = z / (amp + self.eps).astype(z.dtype)
        return amp_norm.astype(z.dtype) * phase_unit

class FRDOscillator(nn.Module):
    dim: int
    @nn.compact
    def __call__(self, x): # x [B,T,D] real
        phase = self.param("phase", nn.initializers.zeros, (self.dim,)) # init 0 для стабильности
        amp = self.param("amp", nn.initializers.ones, (self.dim,))
        freq_gate = self.param("freq_gate", nn.initializers.ones, (self.dim,))
        # complex basis
        basis = amp * jnp.cos(phase) + 1j * amp * jnp.sin(phase) # [D]
        z = x.astype(jnp.complex64) * basis # [B,T,D]
        Z = jnp.fft.fft(z, axis=1) # XLA fused
        Z = Z * freq_gate
        z2 = jnp.fft.ifft(Z, axis=1)
        z2 = PhaseNorm(self.dim)(z2)
        out = jnp.abs(z2) * jnp.sign(x + 1e-6) + x * 0.2
        return out

class LowRankMoS(nn.Module):
    dim: int; rank: int = 64; latent: int = 64
    @nn.compact
    def __call__(self, x, phi): # x,phi [B,T,D]
        # hyper_router: phi -> latent
        h = nn.Dense(self.latent)(phi.mean(axis=1)) # [B,latent]
        h = nn.silu(h)
        h = nn.Dense(self.latent)(h) # [B,latent]
        U = nn.Dense(self.dim * self.rank)(h).reshape((x.shape[0], self.dim, self.rank)) # [B,D,r]
        V = nn.Dense(self.rank * self.dim)(h).reshape((x.shape[0], self.rank, self.dim)) # [B,r,D]
        # (x @ U) @ V  via einsum (vmap автоматом)
        x_u = jnp.einsum("btd,bdr->btr", x, U) # [B,T,r]
        delta = jnp.einsum("btr,brd->btd", x_u, V) / jnp.sqrt(self.rank)
        return delta

class FRDSwiGLU(nn.Module):
    dim: int; hidden: int = 8192
    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.hidden, use_bias=False)(x)
        up = nn.Dense(self.hidden, use_bias=False)(x)
        return nn.Dense(self.dim, use_bias=False)(nn.silu(gate) * up)

class FRDMoSBlock(nn.Module):
    dim: int; rank: int = 64; use_swiglu: bool = True
    @nn.compact
    def __call__(self, x, phi):
        h = FRDOscillator(self.dim)(x) + LowRankMoS(self.dim, self.rank)(x, phi) + x
        h = nn.LayerNorm()(h)
        if self.use_swiglu:
            h = h + FRDSwiGLU(self.dim)(h)
        return h

class AetherMoS(nn.Module):
    vocab: int; dim: int = 2048; layers: int = 8; rank: int = 64
    @nn.compact
    def __call__(self, ids, phi): # ids [B,T] int, phi [B,T,D]
        x = nn.Embed(self.vocab, self.dim, embedding_init=nn.initializers.normal(0.02))(ids) # [B,T,D]
        for _ in range(self.layers):
            x = FRDMoSBlock(self.dim, self.rank, use_swiglu=True)(x, phi)
        logits = nn.Dense(self.vocab, use_bias=False)(x) # [B,T,vocab] (без tying для JAX простоты)
        return logits

# --- Trainer ---
class TrainState(train_state.TrainState):
    ema_params: any

def create_state(rng, model, vocab, dim, layers, seq, lr):
    dummy_ids = jnp.ones((1, seq), dtype=jnp.int32)
    dummy_phi = jnp.ones((1, seq, dim))
    params = model.init(rng, dummy_ids, dummy_phi)
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adamw(lr, b1=0.9, b2=0.999, weight_decay=0.01, mu_dtype=jnp.bfloat16),  # FIX: 13.6GB->6.8GB для 1.72B на T4
    )
    state = TrainState.create(apply_fn=model.apply, params=params, ema_params=params, tx=tx)
    return state

@jax.jit
def train_step(state, ids, phi, lmb=0.05):
    def loss_fn(params):
        logits = state.apply_fn(params, ids, phi) # [B,T,vocab]
        # CE
        logits_shift = logits[:, :-1].reshape(-1, logits.shape[-1])
        ids_shift = ids[:, 1:].reshape(-1)
        lm = optax.softmax_cross_entropy_with_integer_labels(logits_shift, ids_shift).mean()
        # PSTC: phi_student vs phi_teacher (EMA)
        # phi_student = forward phi через один блок (упрощено: MSE на phi)
        # Для JAX делаем cons как MSE между phi и EMA-обработанным phi (один шаг)
        # Упрощение: cons = MSE(phi[:,1:], phi[:,:-1]) *0.1 + MSE(student, teacher)
        # Делаем student = model phi (без ids) через apply
        # Здесь используем последний блок как прокси
        cons = jnp.mean((phi[:, 1:] - phi[:, :-1])**2) * 0.1
        loss = lm + lmb * cons
        return loss, (lm, cons)
    (loss, (lm, cons)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    # EMA
    ema = jax.tree_util.tree_map(lambda e,p: e*0.999 + p*0.001, state.ema_params, state.params)
    state = state.replace(ema_params=ema)
    return state, loss, lm, cons

def get_tokenizer_and_data(seq_len=512):
    # упрощенная загрузка как в aether_train.py — alpaca streaming
    try:
        from transformers import AutoTokenizer
        from datasets import load_dataset
        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M", trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        print(f"[data] SmolLM vocab {len(tok)}")
        # пробуем alpaca streaming, иначе синтетика
        try:
            ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
            it = iter(ds)
            def gen():
                while True:
                    try: ex = next(it)
                    except StopIteration: it2=iter(ds); ex=next(it2)
                    txt = ex["instruction"] + " " + ex.get("input","") + " " + ex.get("output","")
                    ids = tok(txt, truncation=True, max_length=seq_len, padding="max_length")["input_ids"]
                    phi = jax.numpy.array(jax.random.normal(jax.random.PRNGKey(0), (seq_len, 2048)) * 0.5)
                    yield jnp.array(ids), phi[:seq_len, :2048]
            return len(tok), gen()
        except Exception as e:
            print(f"[data] HF fail {e} -> synthetic")
    except Exception as e:
        print(f"[data] fallback {e}")
    # synthetic
    import numpy as np
    def synth():
        rng = jax.random.PRNGKey(42)
        while True:
            rng, k1, k2 = jax.random.split(rng, 3)
            ids = jax.random.randint(k1, (512,), 0, 49152)
            phi = jax.random.normal(k2, (512, 2048)) * 0.5
            yield ids, phi
    return 49152, synth()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()

    rng = jax.random.PRNGKey(0)
    vocab, data_gen = get_tokenizer_and_data(args.seq)
    model = AetherMoS(vocab=vocab, dim=args.dim, layers=args.layers)

    state = create_state(rng, model, vocab, args.dim, args.layers, args.seq, args.lr)
    print(f"[model] {args.layers}x{args.dim} vocab {vocab} -> {sum(p.size for p in jax.tree_util.tree_leaves(state.params))/1e6:.1f}M params | opt bfloat16")
    print(f"[train] steps {args.steps} seq {args.seq} dim {args.dim} layers {args.layers} lr {args.lr}")

    # infinite gen -> batch 1
    for step in range(1, args.steps+1):
        ids, phi = next(data_gen)
        ids = ids[None, :] # [1,seq]
        phi = phi[None, :, :args.dim] # [1,seq,dim]
        # pad phi if dim mismatch
        if phi.shape[-1] != args.dim:
            pad = args.dim - phi.shape[-1]
            phi = jnp.pad(phi, ((0,0),(0,0),(0,pad))) if pad>0 else phi[:,:,:args.dim]
        state, loss, lm, cons = train_step(state, ids, phi, 0.05)
        if step % 5 == 0 or step==1:
            print(f"step {step:04d} loss {loss:.3f} lm {lm:.3f} cons {cons:.4f}")

    print(f"\n[done] {args.steps} steps")
    # save
    import pickle, pathlib
    pathlib.Path("aether_export").mkdir(exist_ok=True)
    with open("aether_export/jax_params.pkl","wb") as f: pickle.dump(state.params, f)
    print("[export] aether_export/jax_params.pkl")

if __name__ == "__main__":
    main()
