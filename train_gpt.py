import os
import sys
import argparse
import copy
import glob
import math
import threading
import time
import uuid
from dataclasses import dataclass
from collections import defaultdict
from itertools import accumulate
from pathlib import Path

# --- KAGGLE T4 COMPATIBILITY SETTINGS ---
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["DISABLE_FP8"] = "True"  # T4 does not support FP8
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # T4 crashes with compile
# ----------------------------------------

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

# Assume dion is installed via: pip install -e .[train]
try:
    from dion import Dion
except ImportError:
    print(
        "Error: Dion not installed. Please run: pip install git+https://github.com/microsoft/dion.git"
    )
    sys.exit(1)


# -----------------------------------------------------------------------------
# Distributed Data Parallel Helper for Standard Optimizers
# Since we removed NorMuon (which did its own sync), we need to manually
# sync gradients for the Dion optimizer across GPUs.
def sync_gradients(optimizer):
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                # Average gradients across all GPUs
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


# -----------------------------------------------------------------------------
# DistAdam (Kept original as it works fine on T4)


class DistAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        label_order = ["lm_head", "scalars", "value_embed", "embed"]
        params_by_label = defaultdict(list)
        for p in params:
            params_by_label[getattr(p, "label", None)].append(p)
        param_groups = []
        for label in label_order:
            if label in params_by_label:
                param_groups.append(dict(params=params_by_label[label]))
        if None in params_by_label:
            param_groups.append(dict(params=params_by_label[None]))
        super().__init__(param_groups, defaults)
        for p in params:
            chunk_size = p.size(0) // self.world_size
            exp_avg = torch.zeros_like(
                p[:chunk_size], dtype=torch.float16, device=p[0].device
            )
            exp_avg_sq = torch.zeros_like(exp_avg)
            self.state[p] = dict(step=0, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)

        self.should_sync = False
        self._reduce_scatter_hooks = []
        self._reduce_scatter_futures = {}
        self.register_backward_hooks()

    def register_backward_hooks(self):
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for param in params:
                hook = param.register_post_accumulate_grad_hook(self._sync_gradient)
                self._reduce_scatter_hooks.append(hook)

    @torch.no_grad()
    def _sync_gradient(self, param):
        if not self.should_sync:
            return
        grad = param.grad
        rank_size = grad.shape[0] // self.world_size
        grad_slice = torch.empty_like(grad[:rank_size])
        self._reduce_scatter_futures[param] = (
            dist.reduce_scatter_tensor(
                grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
            ).get_future(),
            grad_slice,
        )

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        all_gather_futures: list[torch.Future] = []

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for param in group["params"]:
                if param not in self._reduce_scatter_futures:
                    continue

                fut, g_slice = self._reduce_scatter_futures[param]
                fut.wait()

                rank_size = param.shape[0] // self.world_size
                p_slice = param[rank * rank_size : (rank + 1) * rank_size]
                lr = group["lr"] * getattr(param, "lr_mul", 1.0)
                state = self.state[param]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(param, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                bias1 = 1 - beta1**t
                bias2 = 1 - beta2**t
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (bias2**0.5 / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)

                all_gather_futures.append(
                    dist.all_gather_into_tensor(
                        param, p_slice, async_op=True
                    ).get_future()
                )

        self._reduce_scatter_futures.clear()
        torch.futures.collect_all(all_gather_futures).wait()


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions


def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_fp8=False,  # Forced False via Env var logic below
        x_s=1.0,
        w_s=1.0,
        grad_s=1.0,
    ):
        super().__init__(in_features, out_features, bias=False)
        # Force disable FP8 regardless of arg because T4 crashes otherwise
        self.use_fp8 = False
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()

    def forward(self, x: Tensor):
        # Standard Linear for T4
        return F.linear(x, self.weight.type_as(x))


class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.reset()

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(
            0, 1, steps=self.head_dim // 4, dtype=torch.float32, device=device
        )
        angular_freq = torch.cat(
            [angular_freq, angular_freq.new_zeros(self.head_dim // 4)]
        )
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)
        theta = torch.outer(t, angular_freq)
        self.cos = nn.Buffer(theta.cos().to(torch.float16), persistent=False)
        self.sin = nn.Buffer(theta.sin().to(torch.float16), persistent=False)
        self.angular_freq = angular_freq
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int = 1, beta: int = 32):
        rotations = args.block_size * old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (
            1 - scaling_factor
        )
        t = torch.arange(
            self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device
        )
        theta = torch.outer(t, self.angular_freq)
        self.cos.copy_(theta.cos())
        self.sin.copy_(theta.sin())
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1


def rotary(x_BTHD: Tensor, cos: Tensor, sin: Tensor):
    assert cos.size(0) >= x_BTHD.size(-3)
    cos, sin = (
        cos[None, : x_BTHD.size(-3), None, :],
        sin[None, : x_BTHD.size(-3), None, :],
    )
    x1, x2 = x_BTHD.chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), 3)


@dataclass
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    seqlens: torch.Tensor
    bm_size: int
    cos: torch.Tensor
    sin: torch.Tensor
    attn_scale: float
    key_shift: bool


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim
        assert self.hdim == self.dim
        std = 0.5 * (self.dim**-0.5)
        bound = (3**0.5) * std
        self.qkvo_w = nn.Parameter(torch.empty(self.dim * 4, self.hdim))
        self.qkvo_w.label = "attn"
        with torch.no_grad():
            self.qkvo_w[: self.dim * 3].uniform_(-bound, bound)
            self.qkvo_w[self.dim * 3 :].zero_()

        self.attn_gate = CastedLinear(12, num_heads)
        self.attn_gate.weight.label = "attn_gate"

    def forward(self, x: Tensor, attn_args: AttnArgs):
        B, T = x.size(0), x.size(1)
        assert B == 1
        cos, sin = attn_args.cos, attn_args.sin
        ve, sa_lambdas, key_shift = (
            attn_args.ve,
            attn_args.sa_lambdas,
            attn_args.key_shift,
        )
        seqlens, attn_scale, bm_size = (
            attn_args.seqlens,
            attn_args.attn_scale,
            attn_args.bm_size,
        )

        q, k, v = (
            F.linear(x, sa_lambdas[0] * self.qkvo_w[: self.dim * 3].type_as(x))
            .view(B, T, 3 * self.num_heads, self.head_dim)
            .chunk(3, dim=-2)
        )
        q, k = norm(q), norm(k)
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        if key_shift:
            k[:, 1:, :, self.head_dim // 4 : self.head_dim // 2] = k[
                :, :-1, :, self.head_dim // 4 : self.head_dim // 2
            ]
            k[:, 1:, :, self.head_dim // 4 + self.head_dim // 2 :] = k[
                :, :-1, :, self.head_dim // 4 + self.head_dim // 2 :
            ]
        if ve is not None:
            v = v + ve.view_as(v)

        # Efficient mask construction
        q_idx = torch.arange(T, device=x.device).view(-1, 1)
        k_idx = torch.arange(T, device=x.device).view(1, -1)
        mask = q_idx >= k_idx
        if bm_size is not None:
            mask = mask & (q_idx - k_idx < bm_size)
        doc_ids = torch.searchsorted(
            seqlens, torch.arange(T, device=x.device), right=True
        )
        mask = mask & (doc_ids.view(-1, 1) == doc_ids.view(1, -1))

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=attn_scale
        )
        y = y.transpose(1, 2)
        y = y.view(B, T, self.num_heads, self.head_dim)
        y = y * torch.sigmoid(
            self.attn_gate(x[..., : self.attn_gate.weight.size(-1)])
        ).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, sa_lambdas[1] * self.qkvo_w[self.dim * 3 :].type_as(y))
        return y


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = nn.Parameter(torch.empty(hdim, dim))
        self.c_proj = nn.Parameter(torch.empty(hdim, dim))
        self.c_fc.label = "mlp"
        self.c_proj.label = "mlp"
        self.c_proj.lr_mul = 2.0
        std = 0.5 * (dim**-0.5)
        bound = (3**0.5) * std
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_()

    def forward(self, x: Tensor):
        x = F.linear(x, self.c_fc.type_as(x))
        x = F.relu(x).square()
        x = F.linear(x, self.c_proj.T.type_as(x))
        return x


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        self.attn = (
            CausalSelfAttention(dim, head_dim, num_heads) if layer_idx != 6 else None
        )
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, attn_args: AttnArgs):
        if self.attn is not None:
            x = x + self.attn(norm(x), attn_args)
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        return x


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        model_dim: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.embed.weight.label = "embed"
        self.smear_gate = CastedLinear(12, 1)
        self.smear_gate.weight.label = "smear_gate"
        self.value_embeds = nn.ModuleList(
            [nn.Embedding(vocab_size, model_dim) for _ in range(3)]
        )
        for embed in self.value_embeds:
            nn.init.zeros_(embed.weight)
        for ve in self.value_embeds:
            ve.weight.label = "value_embed"
        self.blocks = nn.ModuleList(
            [Block(model_dim, head_dim, num_heads, i) for i in range(num_layers)]
        )
        self.yarn = Yarn(head_dim, max_seq_len)

        self.lm_head = CastedLinear(
            model_dim,
            vocab_size,
            use_fp8=False,  # Disabled for T4
            x_s=(model_dim**0.5) / 448,
            w_s=2**-9,
            grad_s=1 / 448,
        )
        self.lm_head.weight.label = "lm_head"
        pad = (-num_layers * 4 - 3) % dist.get_world_size()
        self.scalars = nn.Parameter(
            torch.cat(
                [
                    1.1 * torch.ones(num_layers),
                    0 * torch.ones(num_layers),
                    *[torch.tensor([0.5, 1.0]) for _ in range(num_layers)],
                    torch.zeros(1),
                    0.5 * torch.ones(1),
                    -1.5 * torch.ones(1),
                    torch.ones(pad),
                ]
            )
        )
        self.scalars.label = "scalars"
        for param in self.embed.parameters():
            param.lr_mul = 75.0
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.0
        self.lm_head.weight.lr_mul = 1.0
        self.scalars.lr_mul = 5.0

    def forward(
        self,
        input_seq: Tensor,
        target_seq: Tensor,
        seqlens: Tensor,
        ws_short: int,
        ws_long: int,
    ):
        assert input_seq.ndim == 1
        skip_connections = []
        skip_in = [3]
        skip_out = [6]
        x_backout = None
        backout_layer = 7

        resid_lambdas = self.scalars[: 1 * self.num_layers]
        x0_lambdas = self.scalars[1 * self.num_layers : 2 * self.num_layers]
        sa_lambdas = self.scalars[2 * self.num_layers : 4 * self.num_layers].view(-1, 2)
        smear_lambda = self.scalars[4 * self.num_layers]
        backout_lambda = self.scalars[4 * self.num_layers + 1]
        skip_lambda = self.scalars[4 * self.num_layers + 2]

        short_bm = ws_short * args.block_size
        long_bm = ws_long * args.block_size
        bm_sizes = [
            short_bm,
            short_bm,
            short_bm,
            long_bm,
            short_bm,
            short_bm,
            None,
            short_bm,
            short_bm,
            short_bm,
            long_bm,
        ]
        key_shift = [b == long_bm for b in bm_sizes]

        x = self.embed(input_seq)
        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve = [ve[1], ve[2]] + [None] * (self.num_layers - 5) + [ve[0], ve[1], ve[2]]

        smear_gate_out = smear_lambda * torch.sigmoid(
            self.smear_gate(x[1:, : self.smear_gate.weight.size(-1)])
        )
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])

        for i in range(self.num_layers):
            attn_args = AttnArgs(
                ve=ve[i],
                sa_lambdas=sa_lambdas[i],
                seqlens=seqlens,
                bm_size=bm_sizes[i],
                cos=self.yarn.cos,
                sin=self.yarn.sin,
                attn_scale=self.yarn.attn_scale,
                key_shift=key_shift[i],
            )
            if i in skip_out:
                gate = torch.sigmoid(skip_lambda)
                x = x + gate * skip_connections.pop()
            if i == 0:
                x = (resid_lambdas[0] + x0_lambdas[0]) * x
            else:
                x = resid_lambdas[i] * x + x0_lambdas[i] * x0
            x = self.blocks[i](x, attn_args)
            if i in skip_in:
                skip_connections.append(x)
            if i == backout_layer:
                x_backout = x

        x -= backout_lambda * x_backout
        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.sigmoid(logits / 7.5)
        logits_for_loss = logits.float() if not self.training else logits
        loss = F.cross_entropy(
            logits_for_loss.view(-1, logits_for_loss.size(-1)),
            target_seq,
            reduction="sum" if self.training else "mean",
        )
        return loss


# -----------------------------------------------------------------------------
# Data Loader (Unchanged)
def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens
    return tokens


BOS_ID = 50256


class BOSFinder:
    def __init__(self, tokens: Tensor, world_size: int = 1, quickload: bool = False):
        self.tokens = tokens
        self.size = tokens.numel()
        self.quickload = quickload
        if quickload:
            self.bos_idx = (
                (tokens[:4_000_000] == BOS_ID)
                .nonzero(as_tuple=True)[0]
                .to(torch.int64)
                .cpu()
                .numpy()
            )
            self.thread = None
            self.ready = threading.Event()
            self.start()
        else:
            self.bos_idx = (
                (tokens == BOS_ID)
                .nonzero(as_tuple=True)[0]
                .to(torch.int64)
                .cpu()
                .numpy()
            )
        self.i = 0
        self.world_size = world_size
        self.batch_iter = 0

    def _load(self):
        self.bos_idx_async = (
            (self.tokens == BOS_ID)
            .nonzero(as_tuple=True)[0]
            .to(torch.int64)
            .cpu()
            .numpy()
        )
        self.ready.set()

    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()

    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        self.bos_idx = self.bos_idx_async

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        if self.quickload and self.batch_iter == 5:
            self.get()
        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]
        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    raise StopIteration(f"Insufficient BOS ahead; hit tail of shard.")
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                end = min(
                    self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                    cur + max_seq_len,
                    cur + num_tokens_local - cur_len + 1,
                )
                ends[r].append(end)
                cur_len += end - cur
                idx += 1
            assert cur_len == num_tokens_local + 1
        self.i = idx
        self.batch_iter += 1
        return starts, ends


class DataPreloader:
    def __init__(self, file_iter, world_size: int = 1):
        self.file_iter = file_iter
        self.world_size = world_size
        self.thread = None
        self.data = None
        self.ready = threading.Event()

    def _load(self):
        tokens = _load_data_shard(next(self.file_iter))
        self.data = (tokens, BOSFinder(tokens, self.world_size))
        self.ready.set()

    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()

    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        return self.data


def distributed_data_generator(
    filename_pattern: str,
    num_tokens: int,
    max_seq_len: int,
    grad_accum_steps: int = 1,
    align_to_bos: bool = True,
):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert num_tokens % (world_size * grad_accum_steps) == 0
    num_tokens = num_tokens // grad_accum_steps
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")
    file_iter = iter(files)
    tokens = _load_data_shard(next(file_iter))
    if align_to_bos:
        finder = BOSFinder(tokens, world_size=world_size, quickload=True)
        preloader = DataPreloader(file_iter, world_size)
        preloader.start()
    else:
        pos = 0
    while True:
        num_tokens_local = num_tokens // world_size
        max_num_docs = next_multiple_of_n(num_tokens_local // 300, n=128)
        if align_to_bos:
            try:
                seq_starts, seq_ends = finder.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = (
                    torch.tensor(seq_starts[rank]),
                    torch.tensor(seq_ends[rank]),
                )
            except StopIteration:
                tokens, finder = preloader.get()
                preloader.start()
                continue
            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1
            cum_lengths = (end_idxs - start_idxs).cumsum(0)
        else:
            if pos + num_tokens + 1 >= len(tokens):
                tokens, pos = _load_data_shard(next(file_iter)), 0
            pos_local = pos + rank * num_tokens_local
            buf = tokens[pos_local : pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local)
            _targets = buf[1:].view(num_tokens_local)
            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            pos += num_tokens
        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1 : len(cum_lengths) + 1] = cum_lengths
        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        _cum_lengths = _cum_lengths.to(dtype=torch.int32)
        new_params = yield (
            _inputs.to(device="cuda", non_blocking=True),
            _targets.to(device="cuda", non_blocking=True),
            _cum_lengths.to(device="cuda", non_blocking=True),
        )
        if new_params is not None:
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * new_grad_accum_steps) == 0
            num_tokens = new_num_tokens // new_grad_accum_steps
            max_seq_len = new_max_seq_len


# -----------------------------------------------------------------------------
# Main Execution


@dataclass
class Hyperparameters:
    train_files: str = "data/fineweb10B/fineweb_train_*.bin"
    val_files: str = "data/fineweb10B/fineweb_val_*.bin"
    val_tokens: int = 10485760
    total_accum_steps: int = 8
    sample_size = 32
    train_bs_schedule: tuple = (
        8 * sample_size * 8,
        16 * sample_size * 8,
        24 * sample_size * 8,
    )
    train_bs_extension: int = 24 * sample_size * 8
    train_max_seq_len: int = 128 * 16
    val_batch_size: int = 24 * sample_size * 8
    num_scheduled_iterations: int = 2070
    num_extension_iterations: int = 40
    num_iterations: int = num_scheduled_iterations + num_extension_iterations
    cooldown_frac: float = 0.55
    run_id: str = f"{uuid.uuid4()}"
    val_loss_every: int = 250
    save_checkpoint: bool = False
    block_size: int = 128
    ws_schedule: tuple = (3, 7, 11)
    ws_final: int = 13
    ws_validate_post_yarn_ext: int = 20
    disable_compile: bool = True  # Force disabled
    compile_threads: int = 4
    compile_mode: str = "default"  # Unused but kept

    def __post_init__(self):
        self.num_iterations = (
            self.num_scheduled_iterations + self.num_extension_iterations
        )
        if self.total_accum_steps != 8:
            scale = self.total_accum_steps / 8
            self.train_bs_schedule = tuple(
                int(x * scale) for x in self.train_bs_schedule
            )
            self.train_bs_extension = int(self.train_bs_extension * scale)
            self.val_batch_size = int(self.val_batch_size * scale)


def get_args(defaults):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--train_files", type=str, default=defaults.train_files)
    parser.add_argument("--val_files", type=str, default=defaults.val_files)
    parser.add_argument("--val_tokens", type=int, default=defaults.val_tokens)
    parser.add_argument(
        "--total_accum_steps", type=int, default=defaults.total_accum_steps
    )
    parser.add_argument(
        "--train_bs_schedule",
        type=int,
        nargs="+",
        default=list(defaults.train_bs_schedule),
    )
    parser.add_argument(
        "--train_bs_extension", type=int, default=defaults.train_bs_extension
    )
    parser.add_argument(
        "--train_max_seq_len", type=int, default=defaults.train_max_seq_len
    )
    parser.add_argument("--val_batch_size", type=int, default=defaults.val_batch_size)
    parser.add_argument(
        "--num_scheduled_iterations",
        type=int,
        default=defaults.num_scheduled_iterations,
    )
    parser.add_argument(
        "--num_extension_iterations",
        type=int,
        default=defaults.num_extension_iterations,
    )
    parser.add_argument("--cooldown_frac", type=float, default=defaults.cooldown_frac)
    parser.add_argument("--run_id", type=str, default=defaults.run_id)
    parser.add_argument("--val_loss_every", type=int, default=defaults.val_loss_every)
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--block_size", type=int, default=defaults.block_size)
    parser.add_argument(
        "--ws_schedule", type=int, nargs="+", default=list(defaults.ws_schedule)
    )
    parser.add_argument("--ws_final", type=int, default=defaults.ws_final)
    parser.add_argument(
        "--ws_validate_post_yarn_ext",
        type=int,
        default=defaults.ws_validate_post_yarn_ext,
    )
    parser.add_argument("--disable_compile", action="store_true")
    parser.add_argument("--compile_threads", type=int, default=defaults.compile_threads)
    parser.add_argument("--compile_mode", type=str, default=defaults.compile_mode)
    return parser.parse_args()


hp_defaults = Hyperparameters()
parsed_args = get_args(hp_defaults)
params = vars(parsed_args)
params["train_bs_schedule"] = tuple(params["train_bs_schedule"])
params["ws_schedule"] = tuple(params["ws_schedule"])
args = Hyperparameters(**params)

# Force Disable Compile again to be safe
args.disable_compile = True
os.environ["TORCH_COMPILE_DISABLE"] = "1"

data_path = os.environ.get("DATA_PATH", ".")
args.train_files = os.path.join(data_path, args.train_files)
args.val_files = os.path.join(data_path, args.val_files)

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
grad_accum_steps = args.total_accum_steps // world_size
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
from datetime import timedelta

dist.init_process_group(backend="nccl", device_id=device, timeout=timedelta(minutes=30))
dist.barrier()
master_process = rank == 0

logfile = None
if master_process:
    run_id = args.run_id
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"


def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)


print0(f"RANK: {rank}, WORLD_SIZE: {world_size}, DEVICE: {device}")
print0(f"grad_accum_steps: {grad_accum_steps}")
print0("Kaggle T4 Mode Enabled: Disable FP8=True, Disable Compile=True, Optimizer=Dion")
print0("=" * 100)

model: nn.Module = GPT(
    vocab_size=50257,
    num_layers=11,
    num_heads=6,
    head_dim=128,
    model_dim=768,
    max_seq_len=args.val_batch_size // (grad_accum_steps * world_size),
).cuda()
for m in model.modules():
    if isinstance(m, (nn.Embedding, nn.Linear)):
        m.half()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# OPTIMIZER SETUP
hidden_matrix_params = [
    p
    for n, p in model.blocks.named_parameters()
    if p.ndim >= 2 and "embed" not in n and "gate" not in n
]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]
gate_params = [p for n, p in model.named_parameters() if "gate" in n]

optimizer1 = DistAdam(
    embed_params + scalar_params + head_params,
    lr=0.008,
    betas=(0.65, 0.95),
    eps=1e-8,
    weight_decay=0.0,
)
# Replaced NorMuon with Dion
optimizer2 = Dion(
    hidden_matrix_params + gate_params,
    lr=0.023,
    momentum=0.95,
    weight_decay=0.02,  # Adjusted for Dion
)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]


def get_lr(step: int):
    if step > args.num_scheduled_iterations:
        return 0.1
    lr_max = 1.0
    x = step / args.num_scheduled_iterations
    if x > 2 / 3:
        lr_max = 1.93
    elif x > 1 / 3:
        lr_max = 1.51
    if x >= 1 - args.cooldown_frac:
        w = (1 - x) / args.cooldown_frac
        return lr_max * w + (1 - w) * 0.1
    return lr_max


def get_ws(step: int):
    if step >= args.num_scheduled_iterations:
        return args.ws_final // 2, args.ws_final
    x = step / args.num_scheduled_iterations
    ws_idx = int(len(args.ws_schedule) * x)
    return args.ws_schedule[ws_idx] // 2, args.ws_schedule[ws_idx]


def get_bs(step: int):
    if step >= args.num_scheduled_iterations:
        return args.train_bs_extension
    x = step / args.num_scheduled_iterations
    bs_idx = int(len(args.train_bs_schedule) * x)
    return args.train_bs_schedule[bs_idx]


def get_muon_momentum(
    step: int,
    muon_warmup_steps=300,
    muon_cooldown_steps=50,
    momentum_min=0.85,
    momentum_max=0.95,
):
    momentum_cd_start = args.num_iterations - muon_cooldown_steps
    if step < muon_warmup_steps:
        frac = step / muon_warmup_steps
        momentum = momentum_min + frac * (momentum_max - momentum_min)
    elif step > momentum_cd_start:
        frac = (step - momentum_cd_start) / muon_cooldown_steps
        momentum = momentum_max - frac * (momentum_max - momentum_min)
    else:
        momentum = momentum_max
    return momentum


def step_optimizers(step: int, optimizers, model):
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)

    # Dion momentum
    momentum = get_muon_momentum(step)
    for group in optimizers[1].param_groups:
        group["momentum"] = momentum

    if step % 2 == 0:
        # SYNC GRADIENTS FOR DION (Optimizer 2)
        # DistAdam (Optimizer 1) handles its own sync via hooks
        sync_gradients(optimizers[1])
        optimizers[1].step()
        optimizers[1].zero_grad(set_to_none=True)
    else:
        # Sync Dion
        sync_gradients(optimizers[1])
        # Sync DistAdam triggers via should_sync
        for optimizer in optimizers:
            optimizer.step()
        model.zero_grad(set_to_none=True)
        optimizers[0].should_sync = False


# SKIP WARMUP FOR DION (Simpler logic for T4 stability)
# Directly initializing Training
step_batch_size = args.train_bs_schedule[0]
print0(
    f"Initializing training loader (batch_size={step_batch_size}, seq_len={args.train_max_seq_len}, grad_accum={grad_accum_steps})..."
)
train_loader = distributed_data_generator(
    args.train_files,
    step_batch_size,
    args.train_max_seq_len,
    grad_accum_steps=grad_accum_steps,
)

import gc

gc.collect()

training_time_ms = 0
torch.cuda.synchronize()
t0 = time.perf_counter()
train_steps = args.num_iterations
print0(f"Starting training for {train_steps} iterations...")
ws_short, ws_long = get_ws(0)

for step in range(train_steps + 1):
    last_step = step == train_steps
    ws_short, new_ws_long = get_ws(step)
    if new_ws_long != ws_long:
        model.yarn.apply(ws_long, new_ws_long)
        ws_long = new_ws_long

    # VALIDATION
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        if last_step:
            ws_long = args.ws_validate_post_yarn_ext
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size
        val_loader = distributed_data_generator(
            args.val_files,
            args.val_batch_size,
            -1,
            grad_accum_steps=grad_accum_steps,
            align_to_bos=False,
        )
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets, cum_seqlens = next(val_loader)
                val_loss += model(inputs, targets, cum_seqlens, ws_short, ws_long)
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(
            f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms",
            console=True,
        )
        model.train()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        break

    # TRAINING
    new_step_batch_size = get_bs(step)
    send_args = (
        (new_step_batch_size, args.train_max_seq_len, grad_accum_steps)
        if new_step_batch_size != step_batch_size
        else None
    )
    step_batch_size = new_step_batch_size

    for idx in range(grad_accum_steps):
        if idx == grad_accum_steps - 1 and step % 2 == 1:
            optimizers[0].should_sync = True
        inputs, targets, cum_seqlens = train_loader.send(send_args)
        (
            model(inputs, targets, cum_seqlens, ws_short, ws_long) / grad_accum_steps
        ).backward()

    step_optimizers(step, optimizers, model)

    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    if (step + 1) % 10 == 0:
        print0(
            f"step:{step + 1}/{train_steps} time:{approx_training_time_ms:.0f}ms",
            console=True,
        )

dist.destroy_process_group()
