"""
Autoresearch-MLX pretraining script. Single-file, Apple Silicon native.
Usage: uv run train.py
"""

import os
import gc
import sys
import time
import math
import subprocess
from dataclasses import dataclass, asdict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers
from mlx.utils import tree_map

from prepare import MAX_SEQ_LEN, TIME_BUDGET, EVAL_TOKENS, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

EPS = 1.1920929e-07  # float32 machine epsilon

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    """RMS norm without learnable weight (matching original's raw F.rms_norm)."""
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + EPS)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    # Concat-style RoPE with the standard negative-sin-on-second-row convention.
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return mx.concatenate([y1, y2], axis=3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def __call__(self, x, ve, cos_sin, window_size, mask):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * mx.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + mx.expand_dims(gate, axis=-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # Transpose to [B, H, T, D] for SDPA
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # MLX SDPA handles GQA natively — no need to expand k,v heads
        scale = 1.0 / math.sqrt(self.head_dim)
        # Use precomputed causal mask (sliced to current sequence length)
        attn_mask = mask[:T, :T]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=attn_mask)

        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.square(nn.relu(x))
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, ve, cos_sin, window_size, mask):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Plain mx.array attributes — MLX's nn.Module introspects these as trainable
        # parameters automatically. No wrapper needed unlike PyTorch.
        self.resid_lambdas = mx.ones((config.n_layer,))
        self.x0_lambdas = mx.zeros((config.n_layer,))

        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = {
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        }

        # Rotary embeddings — precomputed, then frozen
        rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(rotary_seq_len, head_dim)
        self.cos = cos
        self.sin = sin
        # Precompute causal masks for all unique window sizes (avoids recomputing per layer/step)
        T = config.sequence_len
        full_causal = mx.tril(mx.ones((T, T), dtype=mx.bool_))
        unique_windows = set(self.window_sizes)
        self._masks = {}
        for ws in unique_windows:
            if ws > 0 and ws < T:
                self._masks[ws] = mx.triu(full_causal, k=1 - ws)
            else:
                self._masks[ws] = full_causal

        # Freeze so nn.value_and_grad skips these (Critical note #11)
        self.freeze(keys=["cos", "sin", "_masks"])

    def init_weights(self):
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        # Embedding and unembedding
        self.wte.weight = mx.random.normal(self.wte.weight.shape).astype(mx.bfloat16)
        self.lm_head.weight = mx.random.normal(self.lm_head.weight.shape) * 0.001

        # Transformer blocks
        for block in self.blocks:
            block.attn.c_q.weight = mx.random.uniform(-s, s, block.attn.c_q.weight.shape)
            block.attn.c_k.weight = mx.random.uniform(-s, s, block.attn.c_k.weight.shape)
            block.attn.c_v.weight = mx.random.uniform(-s, s, block.attn.c_v.weight.shape)
            block.attn.c_proj.weight = mx.zeros_like(block.attn.c_proj.weight)
            block.mlp.c_fc.weight = mx.random.uniform(-s, s, block.mlp.c_fc.weight.shape)
            block.mlp.c_proj.weight = mx.zeros_like(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas = mx.ones((self.config.n_layer,))
        self.x0_lambdas = mx.full((self.config.n_layer,), 0.1)

        # Value embeddings
        for ve in self.value_embeds.values():
            ve.weight = mx.random.uniform(-s, s, ve.weight.shape).astype(mx.bfloat16)

        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.blocks:
            if block.attn.ve_gate is not None:
                block.attn.ve_gate.weight = mx.zeros_like(block.attn.ve_gate.weight)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        channel_range = mx.arange(0, head_dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)
        cos = mx.cos(freqs).astype(mx.bfloat16)
        sin = mx.sin(freqs).astype(mx.bfloat16)
        # Shape: [1, T, 1, D//2] for broadcasting with [B, T, H, D//2]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert len(pattern) > 0, "window_pattern must not be empty"
        assert all(c in "SL" for c in pattern), f"window_pattern must contain only S/L, got '{pattern}'"
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": long_window, "S": short_window}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = long_window
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        params = self.parameters()
        nparams = sum(x.size for x in self._iter_params(params))

        value_embeds_numel = sum(ve.weight.size for ve in self.value_embeds.values())
        wte_numel = self.wte.weight.size
        resid_numel = self.resid_lambdas.size
        x0_numel = self.x0_lambdas.size
        nparams_exclude = wte_numel + value_embeds_numel + resid_numel + x0_numel

        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            effective_seq = t if window_size < 0 else min(window_size, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def _iter_params(self, params):
        """Recursively iterate over all leaf mx.array parameters."""
        if isinstance(params, mx.array):
            yield params
        elif isinstance(params, dict):
            for v in params.values():
                yield from self._iter_params(v)
        elif isinstance(params, (list, tuple)):
            for v in params:
                yield from self._iter_params(v)

    def count_params(self):
        """Count total trainable parameters."""
        return sum(x.size for x in self._iter_params(self.trainable_parameters()))

    def count_all_params(self):
        """Count all parameters (including frozen)."""
        return sum(x.size for x in self._iter_params(self.parameters()))

    def num_scaling_params(self):
        wte = self.wte.weight.size
        value_embeds = sum(ve.weight.size for ve in self.value_embeds.values())
        lm_head = self.lm_head.weight.size
        transformer_matrices = sum(
            x.size for block in self.blocks
            for x in self._iter_params(block.parameters())
        )
        scalars = self.resid_lambdas.size + self.x0_lambdas.size
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer_groups(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                               weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        """Classify parameters into optimizer groups by path.

        Returns a dict mapping each parameter path to its optimizer config:
        {param_path: {kind: "muon"|"adamw", lr: ..., ...}}
        """
        model_dim = self.config.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")

        groups = {}

        # Classify each parameter by its path
        for path, param in self._all_params_with_paths():
            if "wte" in path:
                groups[path] = dict(kind='adamw', lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0)
            elif "lm_head" in path:
                groups[path] = dict(kind='adamw', lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0)
            elif "value_embeds" in path:
                groups[path] = dict(kind='adamw', lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0)
            elif path == "resid_lambdas":
                groups[path] = dict(kind='adamw', lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0)
            elif path == "x0_lambdas":
                groups[path] = dict(kind='adamw', lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0)
            elif "blocks" in path and "weight" in path:
                groups[path] = dict(kind='muon', lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay)
            # Skip frozen params (cos, sin)

        return groups

    def _all_params_with_paths(self, prefix=""):
        """Yield (path, param) for all trainable parameters."""
        for path_part, value in self.trainable_parameters().items():
            full_path = f"{prefix}{path_part}" if prefix else path_part
            if isinstance(value, mx.array):
                yield full_path, value
            elif isinstance(value, dict):
                for sub_path, sub_val in self._flatten_params(value, full_path + "."):
                    yield sub_path, sub_val
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    for sub_path, sub_val in self._flatten_params(item, f"{full_path}.{i}."):
                        yield sub_path, sub_val

    def _flatten_params(self, obj, prefix):
        if isinstance(obj, mx.array):
            yield prefix.rstrip("."), obj
        elif isinstance(obj, dict):
            for k, v in obj.items():
                yield from self._flatten_params(v, f"{prefix}{k}.")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                yield from self._flatten_params(v, f"{prefix}{i}.")

    def __call__(self, idx: "mx.array", targets: "mx.array | None" = None, reduction: str = 'mean') -> "mx.array":
        B, T = idx.shape
        assert T <= self.cos.shape[1]
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            ws = self.window_sizes[i]
            x = block(x, ve, cos_sin, ws, self._masks[ws])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        # Explicit float32 for numerical stability (Critical note #4)
        logits = logits.astype(mx.float32)
        logits = softcap * mx.tanh(logits / softcap)

        if targets is not None:
            # Cross-entropy in float32 (Critical note #4)
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_targets = targets.reshape(-1)
            ce = nn.losses.cross_entropy(flat_logits, flat_targets, reduction='none')
            if reduction == 'mean':
                loss = mx.mean(ce)
            else:
                loss = ce
            return loss
        return logits


# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def adamw_step(p, grad, exp_avg, exp_avg_sq, step_count, lr, beta1, beta2, eps, wd):
    """Single AdamW parameter update."""
    p = p * (1 - lr * wd)
    # Lerp: a + t * (b - a)
    exp_avg = exp_avg + (1 - beta1) * (grad - exp_avg)
    exp_avg_sq = exp_avg_sq + (1 - beta2) * (grad * grad - exp_avg_sq)
    bias1 = 1 - beta1 ** step_count
    bias2 = 1 - beta2 ** step_count
    denom = mx.sqrt(exp_avg_sq / bias2) + eps
    step_size = lr / bias1
    p = p - step_size * exp_avg / denom
    return p, exp_avg, exp_avg_sq


def muon_step(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
              momentum, lr, wd, beta2, ns_steps, red_dim):
    """Batched Muon step for a group of same-shaped matrix parameters."""
    # Nesterov momentum: buf = lerp(buf, grad, 1-momentum), g = lerp(grad, buf, momentum)
    momentum_buffer = momentum_buffer + (1 - momentum) * (stacked_grads - momentum_buffer)
    g = stacked_grads + momentum * (momentum_buffer - stacked_grads)

    # Polar Express orthogonalization (5-step Neumann series)
    X = g.astype(mx.bfloat16)
    X = X / (mx.linalg.norm(X, axis=(-2, -1), keepdims=True) * 1.02 + 1e-6)
    if g.shape[-2] > g.shape[-1]:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = mx.matmul(mx.swapaxes(X, -2, -1), X)
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ mx.swapaxes(X, -2, -1)
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # NorMuon variance reduction
    v_mean = mx.mean(g.astype(mx.float32) ** 2, axis=red_dim, keepdims=True)
    red_dim_size = g.shape[red_dim]
    v_norm_sq = mx.sum(v_mean, axis=(-2, -1), keepdims=True) * red_dim_size
    v_norm = mx.sqrt(v_norm_sq)

    second_momentum_buffer = second_momentum_buffer + (1 - beta2) * (v_mean.astype(second_momentum_buffer.dtype) - second_momentum_buffer)

    step_size = mx.rsqrt(mx.clip(second_momentum_buffer, a_min=1e-10, a_max=None))
    scaled_sq_sum = (v_mean * red_dim_size) * (step_size.astype(mx.float32) ** 2)
    v_norm_new = mx.sqrt(mx.sum(scaled_sq_sum, axis=(-2, -1), keepdims=True))
    final_scale = step_size * (v_norm / mx.clip(v_norm_new, a_min=1e-10, a_max=None))
    g = g * final_scale.astype(g.dtype)

    # Cautious weight decay: only decay params where gradient agrees with param sign.
    # This prevents decay from fighting the gradient direction. Follows the Muon
    # reference implementation where the mask gates decay rather than the update.
    mask = (g * stacked_params) >= 0
    stacked_params = stacked_params - lr * g - lr * wd * stacked_params * mask

    return stacked_params, momentum_buffer, second_momentum_buffer


# Compile individual step functions (Critical note #3: NOT the full step method)
adamw_step_compiled = mx.compile(adamw_step)
muon_step_compiled = mx.compile(muon_step)


class MuonAdamW:
    """Combined optimizer: Muon for 2D matrix params, AdamW for others.

    Interface: optimizer.update(model, grads) — modifies model in place.
    Following MLX optimizer conventions.
    """

    def __init__(self, model: GPT, param_groups: dict[str, dict]) -> None:
        """
        param_groups: dict mapping param_path -> {kind, lr, ...}
        """
        self.param_groups = param_groups
        self.state = {}
        self.step_count = 0

        # Pre-classify params for efficient iteration
        self._adamw_paths = []
        self._muon_groups = {}  # shape -> {paths, state_key}

        for path, config in param_groups.items():
            config['initial_lr'] = config['lr']
            if config['kind'] == 'adamw':
                self._adamw_paths.append(path)
            elif config['kind'] == 'muon':
                # Group muon params by shape for batched polar decomposition
                param = self._get_param(model, path)
                shape_key = param.shape
                if shape_key not in self._muon_groups:
                    self._muon_groups[shape_key] = {
                        'paths': [],
                        'state_key': f"muon:{'x'.join(str(dim) for dim in shape_key)}",
                    }
                self._muon_groups[shape_key]['paths'].append(path)

    def _get_param(self, model, path):
        """Navigate model tree to get parameter at path."""
        obj = model
        try:
            for part in path.split("."):
                if isinstance(obj, dict):
                    obj = obj[part]
                elif isinstance(obj, (list, tuple)):
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
        except (KeyError, IndexError, AttributeError, ValueError) as e:
            raise RuntimeError(
                f"Parameter path '{path}' not found in model: {e}\n"
                "Check that optimizer groups match the model structure."
            ) from e
        return obj

    def _set_param(self, model, path, value):
        """Navigate model tree to set parameter at path."""
        parts = path.split(".")
        obj = model
        try:
            for part in parts[:-1]:
                if isinstance(obj, dict):
                    obj = obj[part]
                elif isinstance(obj, (list, tuple)):
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            last = parts[-1]
            if isinstance(obj, dict):
                obj[last] = value
            elif isinstance(obj, (list, tuple)):
                obj[int(last)] = value
            else:
                setattr(obj, last, value)
        except (KeyError, IndexError, AttributeError, ValueError) as e:
            raise RuntimeError(
                f"Parameter path '{path}' not found in model: {e}\n"
                "Check that optimizer groups match the model structure."
            ) from e

    def update(self, model: GPT, grads: dict) -> None:
        """Update model parameters given gradients. Modifies model in place."""
        self.step_count += 1

        # Flatten gradients for path-based lookup
        flat_grads = {}
        self._flatten_grads(grads, "", flat_grads)

        # AdamW updates
        for path in self._adamw_paths:
            if path not in flat_grads:
                continue
            grad = flat_grads[path]
            param = self._get_param(model, path)
            cfg = self.param_groups[path]

            if path not in self.state:
                self.state[path] = {
                    'exp_avg': mx.zeros_like(param),
                    'exp_avg_sq': mx.zeros_like(param),
                }
            st = self.state[path]
            new_p, new_avg, new_avg_sq = adamw_step_compiled(
                param, grad, st['exp_avg'], st['exp_avg_sq'],
                self.step_count, cfg['lr'], cfg['betas'][0], cfg['betas'][1],
                cfg['eps'], cfg['weight_decay']
            )
            self._set_param(model, path, new_p)
            st['exp_avg'] = new_avg
            st['exp_avg_sq'] = new_avg_sq

        # Muon updates (batched per shape group)
        for shape, group in self._muon_groups.items():
            paths = group['paths']
            valid_paths = [p for p in paths if p in flat_grads]
            if not valid_paths:
                continue
            if len(valid_paths) != len(paths):
                missing = [p for p in paths if p not in flat_grads]
                raise RuntimeError(
                    f"Missing gradients for Muon parameter group {shape}: {missing}"
                )

            grads_list = [flat_grads[p] for p in paths]
            params_list = [self._get_param(model, p) for p in paths]
            stacked_grads = mx.stack(grads_list)
            stacked_params = mx.stack(params_list)

            group_key = group['state_key']
            cfg = self.param_groups[paths[0]]

            if group_key not in self.state:
                state_shape = (len(paths), shape[-2], 1) if shape[-2] >= shape[-1] else (len(paths), 1, shape[-1])
                self.state[group_key] = {
                    'momentum_buffer': mx.zeros((len(paths), *shape), dtype=stacked_params.dtype),
                    'second_momentum_buffer': mx.zeros(state_shape, dtype=stacked_params.dtype),
                }
            st = self.state[group_key]

            red_dim = -1 if shape[-2] >= shape[-1] else -2
            lr_scaled = cfg['lr'] * max(1.0, shape[-2] / shape[-1])**0.5

            new_params, new_mom, new_sec_mom = muon_step_compiled(
                stacked_grads, stacked_params,
                st['momentum_buffer'], st['second_momentum_buffer'],
                cfg['momentum'], lr_scaled, cfg['weight_decay'],
                cfg['beta2'], cfg['ns_steps'], red_dim
            )
            st['momentum_buffer'] = new_mom
            st['second_momentum_buffer'] = new_sec_mom

            # Unbind and update individual params
            for i, p in enumerate(paths):
                self._set_param(model, p, new_params[i])

    def _flatten_grads(self, grads, prefix, result):
        """Flatten nested gradient dict to path -> array."""
        if isinstance(grads, mx.array):
            result[prefix.rstrip(".")] = grads
        elif isinstance(grads, dict):
            for k, v in grads.items():
                self._flatten_grads(v, f"{prefix}{k}.", result)
        elif isinstance(grads, (list, tuple)):
            for i, v in enumerate(grads):
                self._flatten_grads(v, f"{prefix}{i}.", result)


# ---------------------------------------------------------------------------
# Memory Tier System
# ---------------------------------------------------------------------------

def detect_memory_tier() -> tuple[str, float, int]:
    """Auto-detect system RAM and return (tier_name, total_ram_gb, device_batch_size)."""
    try:
        total_ram_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
    except (AttributeError, ValueError):
        print(f"Warning: could not detect system RAM, defaulting to Compact tier (8GB)")
        total_ram_gb = 8.0  # fallback to Compact

    if total_ram_gb < 12:
        tier, batch = "compact", 2
        print(f"Memory: {total_ram_gb:.1f}GB < 12GB threshold → {tier} tier")
    elif total_ram_gb < 28:
        tier, batch = "standard", 8
        print(f"Memory: {total_ram_gb:.1f}GB < 28GB threshold → {tier} tier")
    elif total_ram_gb < 80:
        tier, batch = "pro", 16
        print(f"Memory: {total_ram_gb:.1f}GB < 80GB threshold → {tier} tier")
    else:
        tier, batch = "ultra", 32
        print(f"Memory: {total_ram_gb:.1f}GB >= 80GB threshold → {tier} tier")
    return tier, total_ram_gb, batch


def detect_peak_flops() -> float:
    """Estimate Apple Silicon bf16 peak TFLOPS from chip name.

    Approximate values — actual throughput varies by workload and thermal state.
    Falls back to 2 TFLOPS (M1 base) if detection fails.
    """
    # Mapping: substring in brand string -> approximate bf16 peak FLOPS
    chip_flops = {
        "M4 Ultra": 20e12,
        "M4 Max": 10e12,
        "M4 Pro": 5e12,
        "M4": 2.5e12,
        "M3 Ultra": 16e12,
        "M3 Max": 8e12,
        "M3 Pro": 4e12,
        "M3": 2e12,
        "M2 Ultra": 16e12,
        "M2 Max": 8e12,
        "M2 Pro": 4e12,
        "M2": 2e12,
        "M1 Ultra": 16e12,
        "M1 Max": 8e12,
        "M1 Pro": 4e12,
        "M1": 2e12,
    }
    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, timeout=5
        ).strip()
        # Check longer names first (e.g. "M4 Max" before "M4")
        for chip, flops in chip_flops.items():
            if chip in brand:
                print(f"Detected {chip} — using {flops/1e12:.1f} TFLOPS for MFU estimate")
                return flops
    except Exception:
        pass
    # Fallback: conservative M1-base estimate
    print("Warning: could not detect chip type, using 2.0 TFLOPS for MFU estimate")
    return 2e12


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "L"    # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**16 # ~65K tokens per optimizer step
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0

# Model size
DEPTH = 4

if __name__ == "__main__":

    # ---------------------------------------------------------------------------
    # Setup: tokenizer, model, optimizer, dataloader
    # ---------------------------------------------------------------------------

    t_start = time.perf_counter()
    mx.random.seed(42)

    # Validate data exists before proceeding
    from prepare import TOKENIZER_DIR, DATA_DIR, LEGACY_TOKENIZER_DIRS, LEGACY_DATA_DIRS, _resolve_existing_dir
    tok_dir = _resolve_existing_dir(TOKENIZER_DIR, LEGACY_TOKENIZER_DIRS, ["tokenizer.pkl"])
    if not os.path.exists(os.path.join(tok_dir, "tokenizer.pkl")):
        print("Error: Tokenizer not found. Run 'uv run prepare.py' first.")
        sys.exit(1)
    # Check preferred and legacy data dirs for parquet files
    def _has_parquets(d):
        return os.path.isdir(d) and any(f.endswith(".parquet") for f in os.listdir(d))
    if not any(_has_parquets(d) for d in [DATA_DIR] + LEGACY_DATA_DIRS):
        print("Error: Data not found. Run 'uv run prepare.py' first.")
        sys.exit(1)

    # Memory tier auto-detection
    MEMORY_TIER, TOTAL_RAM_GB, AUTO_BATCH_SIZE = detect_memory_tier()
    DEVICE_BATCH_SIZE = AUTO_BATCH_SIZE  # override by editing directly

    # Display memory tier
    usable_gb = max(0, TOTAL_RAM_GB - 3)  # ~3GB for macOS
    print(f"Memory tier: {MEMORY_TIER.title()} ({TOTAL_RAM_GB:.0f}GB detected, ~{usable_gb:.0f}GB usable for training)")
    print(f"DEVICE_BATCH_SIZE auto-set to {DEVICE_BATCH_SIZE} (override by editing directly)")
    print()

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")

    def build_model_config(depth):
        base_dim = depth * ASPECT_RATIO
        model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
        num_heads = model_dim // HEAD_DIM
        return GPTConfig(
            sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
            n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
            window_pattern=WINDOW_PATTERN,
        )

    config = build_model_config(DEPTH)
    print(f"Model config: {asdict(config)}")

    # Create model — no meta device dance needed (MLX is lazy)
    model = GPT(config)
    model.init_weights()

    param_counts = model.num_scaling_params()
    print("Parameter counts:")
    for key, value in param_counts.items():
        print(f"  {key:24s}: {value:,}")
    num_params = param_counts['total']
    num_flops_per_token = model.estimate_flops()
    print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    if DEVICE_BATCH_SIZE <= 0:
        print(f"Error: DEVICE_BATCH_SIZE must be positive, got {DEVICE_BATCH_SIZE}")
        sys.exit(1)
    tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
    if TOTAL_BATCH_SIZE % tokens_per_fwdbwd != 0:
        print(f"Error: TOTAL_BATCH_SIZE ({TOTAL_BATCH_SIZE}) must be divisible by "
              f"DEVICE_BATCH_SIZE * MAX_SEQ_LEN ({DEVICE_BATCH_SIZE} * {MAX_SEQ_LEN} = {tokens_per_fwdbwd})")
        sys.exit(1)
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

    if "--dry-run" in sys.argv:
        print()
        print("--- dry run summary ---")
        print(f"Memory tier:              {MEMORY_TIER}")
        print(f"Device batch size:        {DEVICE_BATCH_SIZE}")
        print(f"Gradient accumulation:    {grad_accum_steps}")
        print(f"Total batch size:         {TOTAL_BATCH_SIZE:,} tokens")
        print(f"Total parameters:         {num_params:,}")
        print(f"Trainable parameters:     {model.count_params():,}")
        print(f"FLOPs per token:          {num_flops_per_token:e}")
        sys.exit(0)

    # Build optimizer
    param_group_config = model.setup_optimizer_groups(
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        scalar_lr=SCALAR_LR,
        adam_betas=ADAM_BETAS,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer = MuonAdamW(model, param_group_config)

    # Note: mx.compile on model object breaks optimizer's path-based parameter
    # navigation. Optimizer step functions (adamw_step, muon_step) are already
    # compiled individually — that's where the perf benefit comes from.

    train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
    x, y, epoch = next(train_loader)  # prefetch first batch

    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    APPLE_SILICON_BF16_PEAK_FLOPS = detect_peak_flops()

    # Schedules (all based on progress = training_time / TIME_BUDGET)

    def get_lr_multiplier(progress):
        if progress < WARMUP_RATIO:
            return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        elif progress < 1.0 - WARMDOWN_RATIO:
            return 1.0
        else:
            cooldown = (1.0 - progress) / WARMDOWN_RATIO
            return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

    def get_muon_momentum(step):
        frac = min(step / 300, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    def get_weight_decay(progress):
        return WEIGHT_DECAY * (1 - progress)

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------

    # Loss function pre-scales by grad_accum_steps (Critical note #2)
    def loss_fn(model, x, y):
        return model(x, y) / grad_accum_steps

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    t_start_training = time.perf_counter()
    smooth_train_loss = 0
    total_training_time = 0
    step = 0

    # Reset peak memory for accurate measurement (Critical note #9)
    mx.reset_peak_memory()

    while True:
        t0 = time.perf_counter()

        # Gradient accumulation with mx.eval per micro-step (Critical note #1)
        # Without mx.eval(), lazy evaluation builds the full computation graph across all
        # micro-steps. On Compact tier (16 steps), this OOMs immediately.
        accumulated_grads = tree_map(mx.zeros_like, model.trainable_parameters())
        total_loss = mx.array(0.0)
        last_micro_loss = None

        for micro_step in range(grad_accum_steps):
            loss, grads = loss_and_grad_fn(model, x, y)
            accumulated_grads = tree_map(lambda a, g: a + g, accumulated_grads, grads)
            total_loss = total_loss + loss
            last_micro_loss = loss * grad_accum_steps
            # CRITICAL: mx.eval() inside the loop bounds peak memory to ONE micro-step
            mx.eval(total_loss, accumulated_grads, last_micro_loss)
            x, y, epoch = next(train_loader)  # prefetch next batch

        # Progress and schedules
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(progress)

        for path, cfg in optimizer.param_groups.items():
            cfg['lr'] = cfg['initial_lr'] * lrm
            if cfg['kind'] == 'muon':
                cfg['momentum'] = muon_momentum
                cfg['weight_decay'] = muon_weight_decay

        optimizer.update(model, accumulated_grads)
        mx.eval(model.parameters())

        train_loss_f = last_micro_loss.item()

        # Fast fail: abort if loss is exploding
        if math.isnan(train_loss_f) or train_loss_f > 100:
            print("FAIL")
            sys.exit(1)

        t1 = time.perf_counter()
        dt = t1 - t0

        if step > 10:
            total_training_time += dt

        # Logging
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
        pct_done = 100 * progress
        tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
        mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / APPLE_SILICON_BF16_PEAK_FLOPS
        remaining = max(0, TIME_BUDGET - total_training_time)

        print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

        # GC management (Python's GC causes ~500ms stalls)
        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        # Time's up — but only stop after warmup steps so we don't count compilation
        if step > 10 and total_training_time >= TIME_BUDGET:
            break

    print()  # newline after \r training log

    total_tokens = step * TOTAL_BATCH_SIZE

    # Final eval
    eval_batches = EVAL_TOKENS // (DEVICE_BATCH_SIZE * MAX_SEQ_LEN)
    print(f"Evaluating val_bpb ({eval_batches} batches at batch_size={DEVICE_BATCH_SIZE})...", flush=True)
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

    # Final summary
    t_end = time.perf_counter()
    startup_time = t_start_training - t_start
    steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / APPLE_SILICON_BF16_PEAK_FLOPS if total_training_time > 0 else 0
    peak_memory_mb = mx.get_peak_memory() / (1024**2)

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_memory_mb:   {peak_memory_mb:.1f}")
    print(f"mfu_percent:      {steady_state_mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {DEPTH}")
    print(f"memory_tier:      {MEMORY_TIER}")
