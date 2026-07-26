import argparse
from pathlib import Path
import time
from typing import List, Sequence, Tuple
import types

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models import sampler, xyzflow
from models.vae import AutoencoderKL


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

PATCH_ORDER: Sequence[str] = ("A", "B", "C", "D")
PATCH_TIME_SCHEDULE = {
    "A": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
    "B": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
    "C": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
    "D": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
}
PATCH_STEP_COUNTS = {patch: len(times) - 1 for patch, times in PATCH_TIME_SCHEDULE.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an image with a custom ARD+XYZFlow patch schedule")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the XYZFlow checkpoint (.pth)")
    parser.add_argument("--vae-path", type=str, required=True, help="Path to the VAE checkpoint (.ckpt)")
    parser.add_argument("--output", type=str, default="generated/xyzflow_sample.png", help="Output image path")
    parser.add_argument("--label", type=int, default=0, help="Class label (0-999) to condition on")
    parser.add_argument("--cfg", type=float, default=2.3, help="Classifier-free guidance scale")
    parser.add_argument(
        "--retain-encoder-layers",
        type=int,
        default=-1,
        help="Number of encoder blocks that retain full KV history (default: all)",
    )
    parser.add_argument(
        "--retain-decoder-layers",
        type=int,
        default=-1,
        help="Number of decoder blocks that retain full KV history (default: all)",
    )
    parser.add_argument("--img-size", type=int, default=256, help="Target image resolution")
    parser.add_argument("--vae-embed-dim", type=int, default=16, help="Embedding dimension for the VAE")
    parser.add_argument("--vae-stride", type=int, default=16, help="Stride for the VAE tokenizer")
    parser.add_argument("--patch-size", type=int, default=1, help="Number of tokens grouped per patch")
    parser.add_argument("--class-num", type=int, default=1000, help="Total number of classes")
    parser.add_argument("--label-drop-prob", type=float, default=0.1, help="Label drop prob used during training")
    parser.add_argument("--attn-dropout", type=float, default=0.0, help="Attention dropout")
    parser.add_argument("--proj-dropout", type=float, default=0.0, help="Projection dropout")
    parser.add_argument(
        "--arch",
        type=str,
        default="huge",
        choices=("base", "large", "huge"),
        help="XYZFlow backbone size matching the checkpoint",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="euler",
        choices=("euler", "euler_maruyama"),
        help="Use deterministic Euler or stochastic Euler-Maruyama sampler",
    )
    parser.add_argument(
        "--uniform-steps",
        type=int,
        default=None,
        help="Override schedule so every patch uses this many Euler steps",
    )
    parser.add_argument(
        "--per-patch-steps",
        type=str,
        default=None,
        help="Comma separated step counts for patches A,B,C,D (e.g. '8,4,2,1')",
    )
    parser.add_argument(
        "--per-patch-durations",
        type=str,
        default=None,
        help="Custom dt per patch, format 'A:0.2|0.2|...;B:...' summing to 1.0 per patch",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
    parser.add_argument("--print-schedule", action="store_true", help="Print the ARD+XYZFlow KV schedule before sampling")
    parser.add_argument("--print-cache", action="store_true", help="Print final KV cache lengths after sampling")
    parser.add_argument("--no-compile", action="store_false", dest="compile", help="Disable torch.compile for the model")
    parser.set_defaults(compile=True)
    return parser.parse_args()


def parse_patch_steps(arg: str | None) -> dict[str, int] | None:
    if not arg:
        return None
    values = [v.strip() for v in arg.split(",") if v.strip()]
    if len(values) not in (0, len(PATCH_ORDER)):
        raise ValueError("per-patch-steps must provide counts for A,B,C,D")
    steps = {}
    for patch, value in zip(PATCH_ORDER, values):
        steps[patch] = int(value)
    return steps


def parse_patch_durations(arg: str | None) -> dict[str, List[float]] | None:
    if not arg:
        return None
    mapping: dict[str, List[float]] = {}
    chunks = [chunk.strip() for chunk in arg.split(";") if chunk.strip()]
    for chunk in chunks:
        if ":" not in chunk:
            raise ValueError("per-patch-durations entries must look like 'A:0.2|0.3|...'")
        name, values = chunk.split(":", 1)
        name = name.strip().upper()
        if name not in PATCH_ORDER:
            raise ValueError(f"Unknown patch '{name}' in per-patch-durations")
        durations = [float(v) for v in values.split("|") if v]
        if any(dt <= 0 for dt in durations):
            raise ValueError("Durations must be positive")
        total = sum(durations)
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Durations for patch {name} must sum to 1.0, got {total}")
        mapping[name] = durations
    return mapping


def durations_to_schedule(durations: Sequence[float]) -> List[float]:
    times = [1.0]
    current = 1.0
    for dt in durations:
        current -= dt
        times.append(current)
    return times


def prepare_device(requested_device: str) -> torch.device:
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested_device.startswith("cuda"):
        raise RuntimeError("CUDA requested but not available")
    return torch.device("cpu")


def extend_attention_history(model, retain_encoder_layers: int, retain_decoder_layers: int):
    block_tokens = model.seq_len // model.clusters

    def should_retain(is_encoder: bool, idx: int) -> bool:
        target = retain_encoder_layers if is_encoder else retain_decoder_layers
        if target < 0:
            return True
        return idx < target

    def make_forward(attn_module):
        def forward(self, x: torch.Tensor, mask, update_cache: bool = True, scale_index=None) -> torch.Tensor:  # type: ignore[override]
            B, N, C = x.shape
            if self.training:
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                q = self.rope(q)
                k = self.rope(k)
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask,
                    dropout_p=self.attn_drop if self.training else 0.,
                )
                out = out.transpose(1, 2).reshape(B, N, C)
            else:
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                q = self.rope(q, scale_index)
                k = self.rope(k, scale_index)
                if self.k is not None:
                    prev_k = self.k[:k.shape[0]]
                    prev_v = self.v[:k.shape[0]]
                    k = torch.cat([prev_k, k], dim=2)
                    v = torch.cat([prev_v, v], dim=2)
                retain_all = getattr(self, "_retain_full_history", False)
                if update_cache and mask is not None and not retain_all:
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                else:
                    out = F.scaled_dot_product_attention(q, k, v)
                out = out.permute(0, 2, 1, 3).reshape(B, N, C)

                if retain_all:
                    self.k = k
                    self.v = v
                elif update_cache:
                    self.k = k[:, :, :-block_tokens]
                    self.v = v[:, :, :-block_tokens]
            out = self.proj(out)
            out = self.proj_drop(out)
            return out
        return types.MethodType(forward, attn_module)

    for idx, blk in enumerate(model.encoder_blocks):
        attn = blk.attn
        retain_flag = should_retain(True, idx)
        attn._retain_full_history = retain_flag
        attn.forward = make_forward(attn)
    for idx, blk in enumerate(model.decoder_blocks):
        attn = blk.attn
        retain_flag = should_retain(False, idx)
        attn._retain_full_history = retain_flag
        attn.forward = make_forward(attn)


def load_components(args, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vae = AutoencoderKL(
        embed_dim=args.vae_embed_dim,
        ch_mult=(1, 1, 2, 2, 4),
        ckpt_path=args.vae_path,
    ).to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    model_fns = {
        "base": xyzflow.xyzflow_base,
        "large": xyzflow.xyzflow_large,
        "huge": xyzflow.xyzflow_huge,
    }
    model = model_fns[args.arch](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        label_drop_prob=args.label_drop_prob,
        class_num=args.class_num,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
    ).to(device)
    model.eval()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint)
    del checkpoint

    extend_attention_history(
        model,
        retain_encoder_layers=args.retain_encoder_layers,
        retain_decoder_layers=args.retain_decoder_layers,
    )
    if args.compile:
        if hasattr(model, "compile"):
            model.compile(mode="max-autotune-no-cudagraphs")
    return model, vae


def decode_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor + 1) / 2
    array = tensor.mul(255).byte().cpu().numpy()
    array = array.transpose(1, 2, 0)
    return Image.fromarray(array)


def clear_kv_cache(model) -> None:
    for blk in model.encoder_blocks:
        blk.attn.k = None
        blk.attn.v = None
    for blk in model.decoder_blocks:
        blk.attn.k = None
        blk.attn.v = None


def describe_schedule_rows(
    override_steps: int | None = None,
    override_map: dict[str, int] | None = None,
    duration_map: dict[str, List[float]] | None = None,
) -> List[Tuple[int, str, str, str]]:
    rows: List[Tuple[int, str, str, str]] = []
    kv_cache: List[str] = []
    order = 1
    for patch in PATCH_ORDER:
        if duration_map and patch in duration_map:
            times = durations_to_schedule(duration_map[patch])
        elif override_map and patch in override_map:
            steps = override_map[patch]
            times = [1.0 - i / steps for i in range(steps + 1)]
        elif override_steps is not None:
            steps = override_steps
            times = [1.0 - i / steps for i in range(steps + 1)]
        else:
            times = PATCH_TIME_SCHEDULE[patch]
        for step_idx in range(1, len(times)):
            target = f"{patch}{step_idx}"
            interval = f"{times[step_idx - 1]:.1f}→{times[step_idx]:.1f}"
            context = ", ".join(kv_cache) if kv_cache else "–"
            rows.append((order, target, interval, context))
            kv_cache.append(target)
            order += 1
    return rows


def print_schedule(rows: Sequence[Tuple[int, str, str, str]]) -> None:
    headers = ("顺序", "当前去噪目标", "时间区间", "已缓存 token")
    table = [headers] + [tuple(map(str, row)) for row in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]

    def fmt(row: Sequence[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))

    separator = "-+-".join("-" * w for w in widths)
    print(fmt(headers))
    print(separator)
    for row in table[1:]:
        print(fmt(row))


def sample_tokens_with_patch_schedule(
    model: torch.nn.Module,
    cfg_scale: float,
    label_tensor: torch.Tensor,
    uniform_steps: int | None = None,
    per_patch_steps: dict[str, int] | None = None,
    per_patch_durations: dict[str, List[float]] | None = None,
    sampler_name: str = "euler",
) -> torch.Tensor:
    if model.clusters != len(PATCH_ORDER):
        raise ValueError(f"Model has {model.clusters} clusters but schedule defines {len(PATCH_ORDER)} patches")

    tokens_per_patch = model.seq_len // model.clusters
    latent_dim = getattr(model.pred, "out_features", 16)
    device = label_tensor.device

    cumulative_tokens = [(idx + 1) * tokens_per_patch for idx in range(model.clusters)]
    clear_kv_cache(model)

    sampler_map = {
        "euler": sampler.euler_sampler,
        "euler_maruyama": sampler.euler_maruyama_sampler,
    }
    sampler_fn = sampler_map[sampler_name]

    prev_cond = None
    generated: List[torch.Tensor] = []

    for patch_idx, patch_name in enumerate(PATCH_ORDER):
        if per_patch_durations and patch_name in per_patch_durations:
            durations = per_patch_durations[patch_name]
            steps = len(durations)
            time_schedule = durations_to_schedule(durations)
        elif per_patch_steps and patch_name in per_patch_steps:
            steps = per_patch_steps[patch_name]
            time_schedule = np.linspace(1.0, 0.0, steps + 1, dtype=np.float64).tolist()
        elif uniform_steps is not None:
            steps = uniform_steps
            time_schedule = np.linspace(1.0, 0.0, steps + 1, dtype=np.float64).tolist()
        else:
            steps = PATCH_STEP_COUNTS[patch_name]
            time_schedule = PATCH_TIME_SCHEDULE[patch_name]
        if steps <= 0:
            continue

        if model.clusters == 1:
            scaled_cfg = cfg_scale
        else:
            scaled_cfg = (cfg_scale - 1.0) * patch_idx / (model.clusters - 1.0) + 1.0

        latents = torch.randn(
            label_tensor.shape[0],
            tokens_per_patch,
            latent_dim,
            device=device,
            dtype=torch.float32,
        )

        mask_slice = model.mask[:, :, :cumulative_tokens[patch_idx], :cumulative_tokens[patch_idx]]

        sampler_kwargs = dict(
            model=model.sample_inference,
            latents=latents,
            y=label_tensor,
            scale_index=patch_idx,
            num_steps=steps,
            condition=prev_cond,
            cfg_scale=scaled_cfg,
            mask=mask_slice,
        )
        if sampler_name == "euler":
            sampler_kwargs["time_schedule"] = time_schedule
        z_sample = sampler_fn(**sampler_kwargs).float()

        prev_cond = z_sample
        generated.append(z_sample)

    tokens = model.unclusterify(torch.cat(generated, dim=1))
    return tokens


def summarize_cache(model):
    def collect(prefix: str, blocks) -> List[str]:
        lines: List[str] = []
        for idx, blk in enumerate(blocks):
            attn = blk.attn
            if attn.k is None:
                length = 0
            else:
                length = int(attn.k.shape[2])
            lines.append(f"{prefix}{idx}: {length}")
        return lines

    enc = collect("enc", model.encoder_blocks)
    dec = collect("dec", model.decoder_blocks)
    return enc + dec


def main():
    args = parse_args()
    device = prepare_device(args.device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, vae = load_components(args, device)
    label = torch.tensor([args.label], device=device, dtype=torch.long)

    per_patch_override = parse_patch_steps(args.per_patch_steps)
    per_patch_duration_override = parse_patch_durations(args.per_patch_durations)

    if args.print_schedule:
        print("KV cache / ARD schedule (custom patch timings):")
        schedule = describe_schedule_rows(args.uniform_steps, per_patch_override, per_patch_duration_override)
        print_schedule(schedule)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    start = time.time()
    sample_start = None
    sample_end = None
    with torch.inference_mode():
        autocast_enable = device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=autocast_enable):
            sample_start = time.time()
            tokens = sample_tokens_with_patch_schedule(
                model,
                args.cfg,
                label,
                args.uniform_steps,
                per_patch_override,
                per_patch_duration_override,
                args.sampler,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            sample_end = time.time()
            latents = (tokens / 0.2325).float()
            images = vae.decode(latents)

    if args.print_cache:
        cache_lines = summarize_cache(model)
        print("Final KV cache lengths:")
        for line in cache_lines:
            print("  " + line)

    image = decode_to_image(images[0])
    image.save(output_path)

    elapsed = time.time() - start
    if sample_start is not None and sample_end is not None:
        sample_elapsed = sample_end - sample_start
        print(f"Sample time (model-only): {sample_elapsed:.2f}s")
    print(f"Image saved to {output_path} (elapsed {elapsed:.2f}s)")


if __name__ == "__main__":
    main()
