"""Distillation training for reduced-step XYZFlow models using teacher trajectories."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import List, Sequence, Tuple, Optional
import types

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

from datasets.trajectory_sequence import TrajectorySequenceDataset
from models import xyzflow
from models.vae import AutoencoderKL

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


PATCH_NAMES = ("A", "B", "C", "D")


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    if decay <= 0.0:
        ema_model.load_state_dict(model.state_dict())
        return
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, param in model_params.items():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    for name, buf in model.named_buffers():
        if name in ema_buffers:
            ema_buffers[name].copy_(buf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-architecture student distillation with reduced steps")
    parser.add_argument("--trajectory-dir", type=str, required=True, help="Root directory with packed trajectories")
    parser.add_argument("--checkpoint", type=str, required=True, help="Teacher XYZFlow checkpoint path used for initialization")
    parser.add_argument("--model-size", type=str, default="huge", choices=["base", "large", "huge"], help="XYZFlow model variant")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save checkpoints/logs")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Preferred device when not using DDP")
    parser.add_argument("--cfg", type=float, default=2.3, help="Classifier-free guidance scale")
    parser.add_argument("--image-cfg", type=float, default=None, help="CFG scale used for image logging (defaults to --cfg)")
    parser.add_argument("--retain-encoder-layers", type=int, default=6, help="Encoder layers retaining full KV history")
    parser.add_argument("--retain-decoder-layers", type=int, default=0, help="Decoder layers retaining full KV history")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per process")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit on number of trajectory samples")
    parser.add_argument("--student-step-values", type=str, default=None, help="Comma-separated step ids to supervise (e.g. '0,10,20,30,40,50')")
    parser.add_argument("--student-num-steps", type=int, default=None, help="Evenly subsample this many steps from available teacher steps")
    parser.add_argument("--log-dir", type=str, default=None, help="TensorBoard log directory (defaults to <output>/events)")
    parser.add_argument("--image-log-interval", type=int, default=200, help="Training steps between image logs (0 disables logging)")
    parser.add_argument("--image-log-count", type=int, default=4, help="Number of labels to visualize per log event")
    parser.add_argument("--image-step-values", type=str, default="5,10,50", help="Comma-separated sampling steps per patch to visualize")
    parser.add_argument("--image-log-labels", type=str, default="0", help="Comma-separated class labels for visualization; leave empty to use dataset order")
    parser.add_argument(
        "--image-log-seed",
        type=str,
        default="1234",
        help="Comma-separated list of random seeds for image logging (e.g. '42,43,44,45')",
    )
    parser.add_argument("--vae-path", type=str, default=None, help="Path to KL VAE checkpoint for decoding latents")
    parser.add_argument("--grad-accumulation-steps", type=int, default=1, help="Number of batches to accumulate gradients before optimizer.step()")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, help="Gradient norm clipping value (<=0 disables clipping)")
    parser.add_argument("--log-interval", type=int, default=100, help="Steps between speed/loss logs")
    parser.add_argument("--ema-decay", type=float, default=0.9999, help="EMA decay factor (set <=0 to disable EMA)")
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=0,
        help="Save a checkpoint every N global steps (0 disables step-based saving)",
    )
    parser.add_argument(
        "--viz-trajectory-dir",
        type=str,
        default=None,
        help="Trajectory directory to replay for TensorBoard visualization (uses stored xt instead of random noise)",
    )
    parser.add_argument(
        "--viz-trajectory-sample",
        type=int,
        default=0,
        help="Sample index to replay from the visualization trajectory directory",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="none",
        choices=("none", "fp16", "bf16"),
        help="Enable mixed precision training with the given dtype. 'none' keeps full precision.",
    )
    return parser.parse_args()


def setup_distributed() -> Tuple[bool, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def clear_kv_cache(model) -> None:
    for blk in model.encoder_blocks:
        blk.attn.k = None
        blk.attn.v = None
    for blk in model.decoder_blocks:
        blk.attn.k = None
        blk.attn.v = None


def detach_kv_cache(model) -> None:
    with torch.no_grad():
        for blk in model.encoder_blocks:
            if blk.attn.k is not None:
                blk.attn.k = blk.attn.k.detach()
            if blk.attn.v is not None:
                blk.attn.v = blk.attn.v.detach()
        for blk in model.decoder_blocks:
            if blk.attn.k is not None:
                blk.attn.k = blk.attn.k.detach()
            if blk.attn.v is not None:
                blk.attn.v = blk.attn.v.detach()


def extend_attention_history(model, retain_encoder: int, retain_decoder: int) -> None:
    block_tokens = model.seq_len // model.clusters

    def make_forward(attn_module):
        def forward(self, x, mask, update_cache=True, scale_index=None):
            B, N, C = x.shape
            if self.training:
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                q = self.rope(q)
                k = self.rope(k)
                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask,
                    dropout_p=self.attn_drop if self.training else 0.0,
                )
                out = out.transpose(1, 2).reshape(B, N, C)
            else:
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                q = self.rope(q, scale_index)
                k = self.rope(k, scale_index)
                retain_all = getattr(self, "_retain_full_history", False)

                if self.k is not None:
                    prev_k = self.k[:k.shape[0]]
                    prev_v = self.v[:k.shape[0]]
                    k = torch.cat([prev_k, k], dim=2)
                    v = torch.cat([prev_v, v], dim=2)

                if update_cache and mask is not None and not retain_all:
                    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                else:
                    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
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
        blk.attn._retain_full_history = idx < retain_encoder
        blk.attn.forward = make_forward(blk.attn)
    for idx, blk in enumerate(model.decoder_blocks):
        blk.attn._retain_full_history = idx < retain_decoder
        blk.attn.forward = make_forward(blk.attn)


def select_student_steps(
    available_steps: Sequence[int],
    *,
    explicit_values: Sequence[int] | None,
    target_count: int | None,
) -> Tuple[List[int], List[int]]:
    if explicit_values:
        missing = [step for step in explicit_values if step not in available_steps]
        if missing:
            raise ValueError(f"Requested steps {missing} not present in trajectory data {list(available_steps)}")
        indices = [available_steps.index(step) for step in explicit_values]
        return indices, list(explicit_values)

    total = len(available_steps)
    if target_count is None or target_count >= total:
        return list(range(total)), list(available_steps)
    if target_count <= 0:
        raise ValueError("student-num-steps must be positive")

    if target_count == 1:
        return [total - 1], [available_steps[-1]]

    positions = [i * (total - 1) / (target_count - 1) for i in range(target_count)]
    indices = sorted({int(round(p)) for p in positions})
    if len(indices) > target_count:
        indices = indices[:target_count]
    while len(indices) < target_count:
        for idx in range(total):
            if idx not in indices:
                indices.append(idx)
                if len(indices) == target_count:
                    break
    indices = sorted(indices)
    step_values = [available_steps[idx] for idx in indices]
    return indices, step_values


def forward_step_batch(
    model,
    base_model,
    xt: torch.Tensor,
    label: torch.Tensor,
    patch_idx: int,
    timestep: torch.Tensor,
    cfg_scale: float,
    *,
    update_cache: bool,
    mask: torch.Tensor | None,
    condition: torch.Tensor | None,
) -> torch.Tensor:
    device = label.device
    xt = xt.to(device=device)
    batch_tokens = xt
    step_factor = 1
    if condition is not None and update_cache:
        condition = condition.to(device=device, dtype=xt.dtype)
        batch_tokens = torch.cat([condition, xt], dim=1)
        step_factor = 2

    labels = label
    time_vector = timestep.to(device=device, dtype=xt.dtype)

    if cfg_scale > 1.0:
        batch_tokens = torch.cat([batch_tokens, batch_tokens], dim=0)
        labels = torch.cat([label, torch.full_like(label, 1000)], dim=0)
        time_vector = torch.cat([time_vector, time_vector], dim=0)

    if update_cache and step_factor == 2:
        prefix = torch.zeros_like(time_vector)
        time_input = torch.cat([prefix, time_vector], dim=0)
    else:
        time_input = time_vector

    preds = model(
        batch_tokens,
        labels,
        time_input,
        update_cache,
        patch_idx,
        mask if update_cache else None,
    )

    if cfg_scale > 1.0:
        cond, uncond = preds.chunk(2)
        preds = uncond + cfg_scale * (cond - uncond)

    preds = preds[:, -base_model.seq_len // base_model.clusters :]
    return preds


def parse_step_values(step_str: str) -> List[int]:
    values: List[int] = []
    for token in step_str.split(","):
        if token.strip():
            values.append(int(token.strip()))
    return values


def parse_seed_values(seed_str: str | Sequence[int]) -> List[int]:
    if isinstance(seed_str, str):
        tokens = [token.strip() for token in seed_str.split(",") if token.strip()]
        if not tokens:
            return []
        return [int(token) for token in tokens]
    return [int(value) for value in seed_str]


def prepare_log_labels(
    dataset: TrajectorySequenceDataset,
    desired: Sequence[int],
    count: int,
) -> List[int]:
    if desired:
        return list(desired)[: max(1, count)]
    labels: List[int] = []
    for key in dataset.keys:  # type: ignore[attr-defined]
        if key.label not in labels:
            labels.append(key.label)
        if len(labels) >= count:
            break
    return labels[: max(1, count)] or [dataset.keys[0].label]  # type: ignore[attr-defined]


def log_images(
    writer: SummaryWriter,
    model: torch.nn.Module,
    vae: AutoencoderKL | None,
    *,
    labels: Sequence[int],
    step_values: Sequence[int],
    cfg_scale: float,
    device: torch.device,
    seeds: Sequence[int],
    global_step: int,
    trajectory_dir: Optional[str] = None,
    trajectory_sample: Optional[int] = None,
) -> None:
    if vae is None or not labels or not step_values:
        return
    if not seeds:
        seeds = [0]

    base_model = model.module if isinstance(model, DDP) else model
    was_training = base_model.training
    base_model.eval()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    reference_dir = Path(trajectory_dir) if trajectory_dir else None
    reference_missing: set[Tuple[int, int]] = set()

    def load_reference_latents(label_val: int, step_val: int) -> torch.Tensor | None:
        if reference_dir is None or trajectory_sample is None:
            return None
        tokens: List[torch.Tensor] = []
        for patch_name in PATCH_NAMES:
            path = (
                reference_dir
                / f"patch_{patch_name}"
                / f"step_{step_val:03d}"
                / f"label{label_val:03d}"
                / f"sample{trajectory_sample:04d}.pt"
            )
            if not path.exists():
                reference_missing.add((label_val, step_val))
                return None
            record = torch.load(path, map_location="cpu")
            tokens.append(record["xt"])
        tokens_cat = torch.cat(tokens, dim=0).unsqueeze(0)
        tokens_cat = tokens_cat.to(device=device, dtype=base_model.class_emb.weight.dtype)
        latents_tokens = base_model.unclusterify(tokens_cat)
        latents = (latents_tokens / 0.2325).to(device=device, dtype=torch.float32)
        return latents

    try:
        for sample_idx, label_val in enumerate(labels):
            label_tensor = torch.tensor([label_val], device=device, dtype=torch.long)
            for step_val in step_values:
                use_reference = reference_dir is not None
                reference_latents = None
                if use_reference:
                    reference_latents = load_reference_latents(label_val, step_val)
                    if reference_latents is None:
                        use_reference = False
                for seed_idx, seed_val in enumerate(seeds):
                    torch.manual_seed(seed_val)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed_val)

                    if use_reference:
                        with torch.inference_mode():
                            images = vae.decode(reference_latents).clamp(-1, 1)
                    else:
                        clear_kv_cache(base_model)
                        with torch.inference_mode():
                            tokens = base_model.sample_tokens(num_steps=step_val, cfg=cfg_scale, label=label_tensor)
                            latents = (tokens / 0.2325).to(device=device, dtype=torch.float32)
                            images = vae.decode(latents).clamp(-1, 1)
                    images = (images + 1) / 2
                    tag = f"student_samples/seed_{seed_val}/step_{step_val}/label{label_val}_sample{sample_idx}"
                    writer.add_image(tag, images[0].detach().cpu(), global_step)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if was_training:
            base_model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if reference_missing:
        missing_items = ", ".join(f"(label={lbl}, step={step})" for lbl, step in sorted(reference_missing))
        print(f"[ImageLog] Missing reference trajectory files for: {missing_items}; falling back to random sampling.")


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg_scale: float,
    epoch: int,
    *,
    selected_indices: Sequence[int],
    writer: SummaryWriter | None,
    rank: int,
    steps_per_epoch: int,
    total_epochs: int,
    image_interval: int,
    image_labels: Sequence[int],
    image_steps: Sequence[int],
    vae: AutoencoderKL | None,
    image_seeds: Sequence[int],
    viz_trajectory_dir: Optional[str] = None,
    viz_trajectory_sample: Optional[int] = None,
    grad_accum_steps: int,
    clip_grad_norm: float,
    log_interval: int,
    ema_model: torch.nn.Module | None,
    ema_decay: float,
    global_start_time: float,
) -> float:
    base_model = model.module if isinstance(model, DDP) else model
    block_tokens = base_model.seq_len // base_model.clusters
    mask_cache = [
        base_model.mask[:, :, : block_tokens * (i + 1), : block_tokens * (i + 1)].to(device)
        for i in range(base_model.clusters)
    ]

    grad_accum_steps = max(1, grad_accum_steps)
    total_loss = 0.0
    total_batches = 0
    total_training_steps = steps_per_epoch * total_epochs
    interval_loss = 0.0
    interval_batches = 0
    interval_start = time.time()

    num_patches = base_model.clusters
    num_steps = len(selected_indices)

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        xt = batch["xt"].to(device)
        velocity = batch["velocity"].to(device)
        timestep = batch["timestep"].to(device)
        label = batch["label"].to(device)

        clear_kv_cache(base_model)
        prev_patch_latent = None

        batch_loss = 0.0

        for patch_idx in range(num_patches):
            mask = mask_cache[patch_idx]
            if patch_idx > 0:
                prev_patch_latent = xt[:, patch_idx - 1, selected_indices[-1]]

            for local_step, step_idx in enumerate(selected_indices):
                update_cache = local_step == 0
                cond = prev_patch_latent if (update_cache and prev_patch_latent is not None) else None

                preds = forward_step_batch(
                    model,
                    base_model,
                    xt[:, patch_idx, step_idx],
                    label,
                    patch_idx,
                    timestep[:, patch_idx, step_idx],
                    cfg_scale,
                    update_cache=update_cache,
                    mask=mask if update_cache else None,
                    condition=cond,
                )

                target = velocity[:, patch_idx, step_idx]
                loss = F.mse_loss(preds, target)
                retain = not (patch_idx == num_patches - 1 and local_step == num_steps - 1)
                (loss / (num_patches * num_steps * grad_accum_steps)).backward(retain_graph=retain)
                batch_loss += loss.item()

                detach_kv_cache(base_model)

        total_loss += batch_loss
        total_batches += 1
        interval_loss += batch_loss
        interval_batches += 1

        global_step = epoch * steps_per_epoch + batch_idx
        if rank == 0:
            elapsed_global = max(time.time() - global_start_time, 1e-6)
            steps_done = global_step + 1
            steps_per_sec_global = steps_done / elapsed_global
            remaining_steps = max(total_training_steps - steps_done, 0)
            eta_seconds = remaining_steps / max(steps_per_sec_global, 1e-6)
            elapsed_hms = time.strftime("%H:%M:%S", time.gmtime(elapsed_global))
            eta_hms = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
            print(
                f"[Progress] step {steps_done}/{total_training_steps} "
                f"(epoch {epoch + 1}/{total_epochs}) batch_loss={batch_loss:.6f} "
                f"elapsed={elapsed_hms} eta={eta_hms} speed={steps_per_sec_global:.2f} it/s"
            )

        if writer is not None and rank == 0:
            writer.add_scalar("train/loss_batch", batch_loss, global_step)
            if image_interval > 0 and global_step > 0 and (global_step % image_interval == 0):
                log_images(
                    writer,
                    ema_model if ema_model is not None else model,
                    vae,
                    labels=image_labels,
                    step_values=image_steps,
                    cfg_scale=cfg_scale,
                    device=device,
                    seeds=image_seeds,
                    global_step=global_step,
                    trajectory_dir=viz_trajectory_dir,
                    trajectory_sample=viz_trajectory_sample,
                )

        should_step = ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx == steps_per_epoch - 1)
        if should_step:
            if clip_grad_norm > 0:
                clip_grad_norm_(base_model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema_model is not None:
                update_ema(ema_model, base_model, ema_decay)

        if (
            rank == 0
            and log_interval > 0
            and (global_step + 1) % log_interval == 0
        ):
            elapsed = max(time.time() - interval_start, 1e-6)
            steps_per_sec = interval_batches / elapsed
            avg_interval_loss = interval_loss / max(1, interval_batches)
            print(
                f"[Stats] step={global_step + 1:06d} loss={avg_interval_loss:.6f} "
                f"({steps_per_sec:.2f} it/s)"
            )
            if writer is not None:
                writer.add_scalar("train/loss_interval", avg_interval_loss, global_step)
                writer.add_scalar("train/steps_per_sec", steps_per_sec, global_step)
            interval_loss = 0.0
            interval_batches = 0
            interval_start = time.time()

    avg_loss = total_loss / max(1, total_batches)
    return avg_loss


def main() -> None:
    args = parse_args()
    is_distributed, rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}") if is_distributed else torch.device(args.device)

    dataset = TrajectorySequenceDataset(args.trajectory_dir, max_samples=args.max_samples)
    sampler = None
    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=True,
    )

    explicit_values = None
    if args.student_step_values:
        explicit_values = [int(token.strip()) for token in args.student_step_values.split(",") if token.strip()]

    selected_indices, selected_steps = select_student_steps(
        dataset.step_ids,
        explicit_values=explicit_values,
        target_count=args.student_num_steps,
    )

    if rank == 0:
        print(f"Using student supervision steps: {selected_steps}")

    image_step_values = parse_step_values(args.image_step_values) if args.image_step_values else []
    if rank == 0 and not image_step_values and args.image_log_interval > 0:
        print("[WARN] image-step-values is empty; disabling image logging")
    image_labels_arg = parse_step_values(args.image_log_labels) if args.image_log_labels else []
    log_labels = prepare_log_labels(dataset, image_labels_arg, args.image_log_count)
    image_seeds = parse_seed_values(args.image_log_seed)
    if not image_seeds:
        image_seeds = [0]

    constructor = {
        "base": xyzflow.xyzflow_base,
        "large": xyzflow.xyzflow_large,
        "huge": xyzflow.xyzflow_huge,
    }[args.model_size]

    model = constructor(
        img_size=256,
        vae_stride=16,
        patch_size=1,
        vae_embed_dim=16,
        label_drop_prob=0.1,
        class_num=1000,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.requires_grad_(True)
    model.eval()
    extend_attention_history(model, args.retain_encoder_layers, args.retain_decoder_layers)

    ema_model: torch.nn.Module | None = None
    if args.ema_decay > 0:
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False

    def ddp_forward(self, x, label, time_input, update_cache, patch_idx, mask=None):
        return self.sample_inference(x, label, time_input, update_cache, patch_idx, mask)

    model.forward = ddp_forward.__get__(model, type(model))
    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    base_model = model.module if isinstance(model, DDP) else model
    if ema_model is not None:
        update_ema(ema_model, base_model, 0.0)
    log_model_for_images = ema_model if ema_model is not None else base_model

    vae: AutoencoderKL | None = None
    if (
        args.image_log_interval > 0
        and args.vae_path
        and rank == 0
    ):
        vae = AutoencoderKL(
            embed_dim=16,
            ch_mult=(1, 1, 2, 2, 4),
            ckpt_path=args.vae_path,
        ).to(device)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
    elif rank == 0 and args.image_log_interval > 0:
        print("[WARN] image logging requested but --vae-path not provided; disabling image logs")
        image_step_values = []
        log_labels = []
        image_seeds = []

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        print("=" * 72)
        print("Teacher-architecture student distillation")
        print(f"World size    : {world_size}")
        print(f"Output dir    : {output_dir}")
        print(f"Log dir       : {Path(args.log_dir) if args.log_dir else Path('distill_tensorboard_output') / output_dir.name}")
        print(f"LR            : {args.lr}")
        print(f"CFG scale     : {args.cfg}")
        print(f"Step schedule : {selected_steps}")
        print(f"Grad accum    : {args.grad_accumulation_steps}")
        print(f"Clip grad norm: {args.clip_grad_norm}")
        print(f"Log interval  : {args.log_interval}")
        print(f"EMA decay     : {args.ema_decay}")
        if args.image_log_interval > 0 and image_step_values:
            print(f"Image steps   : {image_step_values}")
            print(f"Image labels  : {log_labels}")
        else:
            print("Image logging : disabled")
        print(f"Dataset size  : {len(dataset)}")
        print(f"Batches/epoch : {len(loader)}")
        print("=" * 72)

    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_root = Path("distill_tensorboard_output")
        if rank == 0:
            log_root.mkdir(parents=True, exist_ok=True)
        log_dir = log_root / output_dir.name
    writer = SummaryWriter(log_dir=str(log_dir)) if rank == 0 else None

    if writer is not None and vae is not None and args.image_log_interval > 0 and image_step_values:
        log_images(
            writer,
            log_model_for_images,
            vae,
            labels=log_labels,
            step_values=image_step_values,
            cfg_scale=args.cfg,
            device=device,
            seeds=image_seeds,
            global_step=0,
            trajectory_dir=args.viz_trajectory_dir,
            trajectory_sample=args.viz_trajectory_sample,
        )
        if rank == 0:
            print(f"[INFO] Initial student samples logged to {log_dir} (global_step=0)")

    steps_per_epoch = len(loader)
    total_epochs = args.epochs
    vae_for_rank = vae if rank == 0 else None
    global_start_time = time.time()

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        avg_loss = train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            args.cfg,
            epoch,
            selected_indices=selected_indices,
            writer=writer,
            rank=rank,
            steps_per_epoch=steps_per_epoch,
            total_epochs=total_epochs,
            image_interval=args.image_log_interval,
            image_labels=log_labels,
            image_steps=image_step_values,
            vae=vae_for_rank,
            image_seeds=image_seeds,
            viz_trajectory_dir=args.viz_trajectory_dir,
            viz_trajectory_sample=args.viz_trajectory_sample,
            grad_accum_steps=args.grad_accumulation_steps,
            clip_grad_norm=args.clip_grad_norm,
            log_interval=args.log_interval,
            ema_model=ema_model,
            ema_decay=args.ema_decay,
            global_start_time=global_start_time,
        )
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.6f}")
            checkpoint = {
                "model": (model.module if isinstance(model, DDP) else model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": ema_model.state_dict() if ema_model is not None else None,
                "args": vars(args),
            }
            torch.save(
                checkpoint,
                output_dir / f"student_teacher_epoch{epoch+1}.pth",
            )

    if writer is not None:
        writer.close()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
