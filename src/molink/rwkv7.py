from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _time_shift(x: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(x[:, :1, :])
    return torch.cat([zero, x[:, :-1, :]], dim=1)


def _wkv7_reference(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    head_size: int,
) -> torch.Tensor:
    bsz, seq_len, dim = r.shape
    n_head = dim // head_size

    r = r.view(bsz, seq_len, n_head, head_size).float()
    k = k.view(bsz, seq_len, n_head, head_size).float()
    v = v.view(bsz, seq_len, n_head, head_size).float()
    a = a.view(bsz, seq_len, n_head, head_size).float()
    b = b.view(bsz, seq_len, n_head, head_size).float()
    w = torch.exp(-torch.exp(w.view(bsz, seq_len, n_head, head_size).float()))

    out = torch.zeros((bsz, seq_len, n_head, head_size), device=r.device, dtype=torch.float)
    state = torch.zeros((bsz, n_head, head_size, head_size), device=r.device, dtype=torch.float)

    for t in range(seq_len):
        kk = k[:, t].view(bsz, n_head, 1, head_size)
        rr = r[:, t].view(bsz, n_head, head_size, 1)
        vv = v[:, t].view(bsz, n_head, head_size, 1)
        aa = a[:, t].view(bsz, n_head, head_size, 1)
        bb = b[:, t].view(bsz, n_head, 1, head_size)
        state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
        out[:, t] = (state @ rr).view(bsz, n_head, head_size)

    return out.view(bsz, seq_len, dim).to(dtype=r.dtype)


def wkv7_op(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    head_size: int,
    use_cuda_wkv: bool = False,
) -> torch.Tensor:
    if use_cuda_wkv and hasattr(torch.ops, "wkv7") and hasattr(torch.ops.wkv7, "forward"):
        bsz, seq_len, dim = r.shape
        n_head = dim // head_size
        y = torch.empty_like(r)
        torch.ops.wkv7.forward(
            bsz,
            seq_len,
            dim,
            n_head,
            r.contiguous(),
            w.contiguous(),
            k.contiguous(),
            v.contiguous(),
            a.contiguous(),
            b.contiguous(),
            y,
        )
        return y
    return _wkv7_reference(r, w, k, v, a, b, head_size)


@dataclass
class RWKV7Config:
    vocab_size: int
    n_layer: int
    n_embd: int
    head_size: int
    max_seq_len: int = 1024
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    use_cuda_wkv: bool = False


class RWKVTimeMixX070(nn.Module):
    def __init__(self, config: RWKV7Config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.head_size = config.head_size
        self.n_head = config.n_embd // self.head_size
        if config.n_embd % self.head_size != 0:
            raise ValueError("n_embd must be divisible by head_size")

        h = self.n_head
        c = config.n_embd

        with torch.no_grad():
            if config.n_layer > 1:
                ratio_0_to_1 = layer_id / (config.n_layer - 1)
            else:
                ratio_0_to_1 = 0.0
            ratio_1_to_almost0 = 1.0 - (layer_id / max(config.n_layer, 1))
            ddd = torch.ones(1, 1, c)
            for i in range(c):
                ddd[0, 0, i] = i / c
            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                shape = x.shape
                if len(shape) == 2:
                    gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                    nn.init.orthogonal_(x, gain=gain * scale)
                elif len(shape) == 3:
                    gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                    for idx in range(shape[0]):
                        nn.init.orthogonal_(x[idx], gain=gain * scale)
                else:
                    raise ValueError("Unsupported tensor shape for orthogonal init")
                return x

            www = torch.zeros(c)
            zigzag = torch.zeros(c)
            linear = torch.zeros(c)
            for n in range(c):
                linear[n] = n / (c - 1) - 0.5
                zigzag[n] = ((n % self.head_size) - ((self.head_size - 1) / 2)) / (
                    (self.head_size - 1) / 2
                )
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (c - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

            d_decay_lora = max(32, int(round((2.5 * (c**0.5)) / 32) * 32))
            self.w1 = nn.Parameter(torch.zeros(c, d_decay_lora))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(d_decay_lora, c), 0.1))
            self.w0 = nn.Parameter(www.reshape(1, 1, c) + 0.5 + zigzag * 2.5)

            d_aaa_lora = max(32, int(round((2.5 * (c**0.5)) / 32) * 32))
            self.a1 = nn.Parameter(torch.zeros(c, d_aaa_lora))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(d_aaa_lora, c), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, c) - 0.19 + zigzag * 0.3 + linear * 0.4)

            d_mv_lora = max(32, int(round((1.7 * (c**0.5)) / 32) * 32))
            self.v1 = nn.Parameter(torch.zeros(c, d_mv_lora))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(d_mv_lora, c), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, c) + 0.73 - linear * 0.4)

            d_gate_lora = max(32, int(round((5 * (c**0.5)) / 32) * 32))
            self.g1 = nn.Parameter(torch.zeros(c, d_gate_lora))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(d_gate_lora, c), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1, 1, c) + 0.71 - linear * 0.1)
            self.k_a = nn.Parameter(torch.zeros(1, 1, c) + 1.02)
            self.r_k = nn.Parameter(torch.zeros(h, self.head_size) - 0.04)

        self.receptance = nn.Linear(c, c, bias=False)
        self.key = nn.Linear(c, c, bias=False)
        self.value = nn.Linear(c, c, bias=False)
        self.output = nn.Linear(c, c, bias=False)
        self.ln_x = nn.GroupNorm(h, c, eps=64e-5)

        self.receptance.weight.data.uniform_(-0.5 / (c**0.5), 0.5 / (c**0.5))
        self.key.weight.data.uniform_(-0.05 / (c**0.5), 0.05 / (c**0.5))
        self.value.weight.data.uniform_(-0.5 / (c**0.5), 0.5 / (c**0.5))
        self.output.weight.data.zero_()

    def forward(self, x: torch.Tensor, v_first: torch.Tensor):
        bsz, seq_len, dim = x.size()
        h = self.n_head

        xx = _time_shift(x) - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(bsz, seq_len, h, -1), dim=-1, p=2.0).view(bsz, seq_len, dim)
        k = k * (1 + (a - 1) * self.k_a)

        x = wkv7_op(r, w, k, v, -kk, kk * a, self.head_size, self.config.use_cuda_wkv)
        x = self.ln_x(x.view(bsz * seq_len, dim)).view(bsz, seq_len, dim)

        x = x + (
            (r.view(bsz, seq_len, h, -1) * k.view(bsz, seq_len, h, -1) * self.r_k)
            .sum(dim=-1, keepdim=True)
            * v.view(bsz, seq_len, h, -1)
        ).view(bsz, seq_len, dim)
        x = self.output(x * g)
        return x, v_first


class RWKVChannelMixX070(nn.Module):
    def __init__(self, config: RWKV7Config, layer_id: int):
        super().__init__()
        self.config = config
        c = config.n_embd
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / max(config.n_layer, 1))
            ddd = torch.ones(1, 1, c)
            for i in range(c):
                ddd[0, 0, i] = i / c
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(c, c * 4, bias=False)
        self.value = nn.Linear(c * 4, c, bias=False)

        self.key.weight.data.uniform_(-0.5 / (c**0.5), 0.5 / (c**0.5))
        self.value.weight.data.zero_()

    def forward(self, x: torch.Tensor):
        xx = _time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)


class RWKVBlock(nn.Module):
    def __init__(self, config: RWKV7Config, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.att = RWKVTimeMixX070(config, layer_id)
        self.ffn = RWKVChannelMixX070(config, layer_id)

    def forward(self, x: torch.Tensor, v_first: torch.Tensor):
        if self.layer_id == 0:
            x = self.ln0(x)
        x_att, v_first = self.att(self.ln1(x), v_first)
        x = x + x_att
        x = x + self.ffn(self.ln2(x))
        return x, v_first


class RWKV7Model(nn.Module):
    def __init__(self, config: RWKV7Config):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([RWKVBlock(config, i) for i in range(config.n_layer)])
        self.ln_out = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.emb(idx)
        if self.dropout is not None:
            x = self.dropout(x)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)
        x = self.ln_out(x)
        logits = self.head(x)
        return logits


class RWKV7ForCausalLM(nn.Module):
    def __init__(self, config: RWKV7Config):
        super().__init__()
        self.config = config
        self.model = RWKV7Model(config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            next_logits = logits[:, -1, :]
            next_token = _sample_logits(next_logits, temperature=temperature, top_p=top_p, top_k=top_k)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if eos_token_id is not None:
                if torch.all(next_token.squeeze(-1) == eos_token_id):
                    break
        return input_ids


def _sample_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    # Apply temperature scaling before softmax for numerical stability
    if temperature <= 0:
        # For temperature <= 0, use greedy sampling
        return logits.argmax(dim=-1, keepdim=True)

    if temperature != 1.0:
        logits = logits / temperature

    probs = F.softmax(logits.float(), dim=-1)
    next_tokens = []

    for row in probs:
        filtered = row.clone()

        # Apply top-k filtering
        if top_k > 0:
            # Ensure top_k doesn't exceed vocabulary size
            k = min(top_k, filtered.size(-1))
            topk_vals, topk_idx = torch.topk(filtered, k)
            mask = torch.zeros_like(filtered)
            mask[topk_idx] = topk_vals
            filtered = mask

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(filtered, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            # Remove tokens with cumulative probability above the threshold
            cutoff = cumulative > top_p
            if torch.any(cutoff):
                # Keep at least one token
                cutoff_idx = torch.nonzero(cutoff, as_tuple=False)
                if len(cutoff_idx) > 0:
                    cutoff_idx = cutoff_idx[0].item()
                    # Keep tokens up to and including the cutoff
                    sorted_probs[cutoff_idx + 1:] = 0.0
            filtered = torch.zeros_like(filtered)
            filtered[sorted_idx] = sorted_probs

        # Normalize probabilities
        prob_sum = filtered.sum()
        if prob_sum > 0:
            filtered = filtered / prob_sum
        else:
            # Fallback: uniform distribution if all probabilities are zero
            filtered = torch.ones_like(filtered) / filtered.size(-1)

        # Sample from the filtered distribution
        next_tokens.append(torch.multinomial(filtered, num_samples=1))

    return torch.stack(next_tokens, dim=0)
