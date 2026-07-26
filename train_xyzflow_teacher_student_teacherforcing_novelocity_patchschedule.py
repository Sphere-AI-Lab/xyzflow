"""
Teacher-forcing variant of the teacher-student distillation loop for XYZFlow.

This implementation flattens every supervised patch/step pair into a single
autoregressive sequence so that each iteration runs one forward/backward pass,
mirroring standard LLM-style training while respecting the ARD KV schedule.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from contextlib import nullcontext
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.trajectory_sequence import TrajectorySequenceDataset
from models import xyzflow
from models.vae import AutoencoderKL
from generate_single_image_patchschedule import (
    PATCH_ORDER as DEFAULT_PATCH_ORDER,
    PATCH_TIME_SCHEDULE as DEFAULT_PATCH_TIME_SCHEDULE,
    sample_tokens_with_patch_schedule,
)

from train_xyzflow_teacher_student import (
    parse_args as base_parse_args,
    setup_distributed,
    extend_attention_history,
    select_student_steps,
    parse_step_values,
    parse_seed_values,
    prepare_log_labels,
    update_ema,
)


PATCH_ORDER: Sequence[str] = DEFAULT_PATCH_ORDER
PATCH_TIME_SCHEDULE = DEFAULT_PATCH_TIME_SCHEDULE
PATCH_STEP_COUNTS = {name: len(times) - 1 for name, times in PATCH_TIME_SCHEDULE.items()}
MAX_PATCH_STEPS = max(PATCH_STEP_COUNTS.values())


def make_patch_index_map(selected_indices: Sequence[int]) -> Dict[str, List[int]]:
    if len(selected_indices) < MAX_PATCH_STEPS:
        raise ValueError(
            f"Need at least {MAX_PATCH_STEPS} supervised steps, got {len(selected_indices)}"
        )
    per_patch: Dict[str, List[int]] = {}
    for patch in PATCH_ORDER:
        count = PATCH_STEP_COUNTS[patch]
        per_patch[patch] = [selected_indices[i] for i in range(count)]
    return per_patch


def build_owner_and_final_flags() -> Tuple[List[int], List[bool]]:
    owners: List[int] = []
    finals: List[bool] = []
    for patch_idx, patch in enumerate(PATCH_ORDER):
        count = PATCH_STEP_COUNTS[patch]
        for step_idx in range(count):
            owners.append(patch_idx)
            finals.append(step_idx == count - 1)
    return owners, finals


def assemble_schedule_sequence(
    xt: torch.Tensor,
    timestep: torch.Tensor,
    *,
    per_patch_indices: Dict[str, Sequence[int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    segments_xt: List[torch.Tensor] = []
    segments_timestep: List[torch.Tensor] = []
    segments_xt_target: List[torch.Tensor] = []
    segments_dt: List[torch.Tensor] = []

    for patch_idx, patch_name in enumerate(PATCH_ORDER):
        indices = per_patch_indices[patch_name]
        patch_xt = xt[:, patch_idx, indices]
        patch_timestep = timestep[:, patch_idx, indices]

        patch_xt_target = patch_xt.clone()
        if patch_xt.shape[1] > 1:
            patch_xt_target[:, :-1] = patch_xt[:, 1:]

        patch_dt = torch.zeros_like(patch_timestep)
        if patch_timestep.shape[1] > 1:
            patch_dt[:, :-1] = patch_timestep[:, 1:] - patch_timestep[:, :-1]

        for step_idx in range(len(indices)):
            segments_xt.append(patch_xt[:, step_idx])
            segments_timestep.append(patch_timestep[:, step_idx])
            segments_xt_target.append(patch_xt_target[:, step_idx])
            segments_dt.append(patch_dt[:, step_idx])

    sequence_xt = torch.stack(segments_xt, dim=1)
    sequence_timestep = torch.stack(segments_timestep, dim=1)
    sequence_xt_target = torch.stack(segments_xt_target, dim=1)
    sequence_dt = torch.stack(segments_dt, dim=1)
    return sequence_xt, sequence_timestep, sequence_xt_target, sequence_dt


def build_causal_mask(
    total_segments: int,
    segment_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    seq_len = total_segments * segment_len
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    zeros = torch.zeros(1, device=device, dtype=dtype)
    neg_inf = torch.full((1,), float("-inf"), device=device, dtype=dtype)
    mask = torch.where(causal > 0, zeros, neg_inf)
    return mask.unsqueeze(0).unsqueeze(0)


def build_sequence_tail_mask(
    owner_indices: Sequence[int],
    final_segment_flags: Sequence[bool],
    segment_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    total_segments = len(owner_indices)
    allowed = torch.zeros(total_segments, total_segments, device=device, dtype=torch.bool)

    for query in range(total_segments):
        for key in range(query + 1):
            if owner_indices[key] == owner_indices[query]:
                allowed[query, key] = True
            elif final_segment_flags[key]:
                allowed[query, key] = True

    allowed = allowed.repeat_interleave(segment_len, dim=0).repeat_interleave(segment_len, dim=1)
    zeros = torch.zeros(1, device=device, dtype=dtype)
    neg_inf = torch.full((1,), float("-inf"), device=device, dtype=dtype)
    mask = torch.where(allowed, zeros, neg_inf)
    return mask.unsqueeze(0).unsqueeze(0)


def patch_rope_for_index_map(rope_module):
    if getattr(rope_module, "_index_map_ready", False):
        return

    original_forward = rope_module.forward

    def forward_with_index(self, x, scale_index=None):
        index_map = getattr(self, "_index_map", None)
        if index_map is not None:
            if index_map.shape[0] != x.shape[2]:
                idx = index_map[: x.shape[2]]
            else:
                idx = index_map
            freqs_cos = self.freqs_cos[idx].unsqueeze(0).unsqueeze(0)
            freqs_sin = self.freqs_sin[idx].unsqueeze(0).unsqueeze(0)
            return x * freqs_cos + rotate_half(x) * freqs_sin
        return original_forward(x, scale_index)

    from models.rope import rotate_half  # local import to avoid circular deps

    rope_module.forward = forward_with_index.__get__(rope_module, type(rope_module))
    rope_module._index_map_ready = True


def set_rope_index_map(blocks, index_map: torch.Tensor) -> None:
    for blk in blocks:
        rope = blk.attn.rope
        patch_rope_for_index_map(rope)
        rope._index_map = index_map


def clear_rope_index_map(blocks) -> None:
    for blk in blocks:
        rope = blk.attn.rope
        if hasattr(rope, "_index_map"):
            rope._index_map = None


def log_step_alignment_metrics(
    writer: SummaryWriter,
    pred_xt: torch.Tensor,
    target_xt: torch.Tensor,
    valid_mask: torch.Tensor,
    global_step: int,
) -> None:
    if not torch.any(valid_mask).item():
        return
    with torch.no_grad():
        pred = pred_xt[valid_mask]
        target = target_xt[valid_mask]
        diff = pred - target
        mae = diff.abs().mean().item()
        mse = diff.pow(2).mean().item()
        rmse = mse ** 0.5
        pred_norm = pred.pow(2).mean().sqrt().item()
        target_norm = target.pow(2).mean().sqrt().item()

    writer.add_scalar("teacher_alignment/step_mae", mae, global_step)
    writer.add_scalar("teacher_alignment/step_rmse", rmse, global_step)
    writer.add_scalar("teacher_alignment/step_pred_norm", pred_norm, global_step)
    writer.add_scalar("teacher_alignment/step_target_norm", target_norm, global_step)


def log_images(
    writer: SummaryWriter,
    model: torch.nn.Module,
    vae: AutoencoderKL | None,
    *,
    labels: Sequence[int],
    step_values: Sequence[int],
    cfg_scale: float,
    device: torch.device,
    seed: int | None = None,
    global_step: int,
    seeds: Sequence[int] | None = None,
    trajectory_dir: str | None = None,
    trajectory_sample: int | None = None,
) -> None:
    if vae is None or not labels:
        return

    seed_pool: List[int] = []
    if seeds:
        seed_pool.extend(int(s) for s in seeds)
    elif seed is not None:
        seed_pool.append(int(seed))
    else:
        seed_pool.append(0)

    base_model = model.module if isinstance(model, DDP) else model
    was_training = base_model.training
    base_model.eval()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        for label_val in labels:
            label_tensor = torch.tensor([label_val], device=device, dtype=torch.long)
            for seed_idx, local_seed in enumerate(seed_pool):
                torch.manual_seed(local_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(local_seed)
                with torch.inference_mode():
                    tokens = sample_tokens_with_patch_schedule(
                        base_model,
                        cfg_scale,
                        label_tensor,
                    )
                    latents = (tokens / 0.2325).to(device=device, dtype=torch.float32)
                    images = vae.decode(latents).clamp(-1, 1)
                images = (images + 1) / 2
                tag = f"student_samples/label{label_val}_seed{local_seed}"
                writer.add_image(tag, images[0].detach().cpu(), global_step)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if was_training:
            base_model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_checkpoint(
    output_dir: Path,
    base_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_model: torch.nn.Module | None,
    args_dict: Dict[str, Any] | None,
    *,
    epoch: int,
    global_step: int,
) -> Path:
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model": base_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema_model.state_dict() if ema_model is not None else None,
        "args": args_dict,
    }
    filename = output_dir / f"student_teacher_step{global_step:07d}.pth"
    torch.save(checkpoint, filename)
    return filename


def forward_teacher_sequence(
    base_model: torch.nn.Module,
    sequence_xt: torch.Tensor,
    timesteps: torch.Tensor,
    label: torch.Tensor,
    *,
    owner_indices: Sequence[int],
    final_segment_flags: Sequence[bool],
    cfg_scale: float,
    retain_encoder_layers: int,
    retain_decoder_layers: int,
) -> torch.Tensor:
    device = sequence_xt.device
    dtype = sequence_xt.dtype
    batch_size, total_segments, segment_len, _ = sequence_xt.shape

    segments: List[torch.Tensor] = []
    time_vectors: List[torch.Tensor] = []
    for seg_idx in range(total_segments):
        segments.append(sequence_xt[:, seg_idx].to(device=device, dtype=dtype))
        time_vectors.append(timesteps[:, seg_idx].to(device=device, dtype=dtype))

    sequence_tokens = torch.cat(segments, dim=1)
    base_batch = sequence_tokens.shape[0]

    time_values = torch.cat([vec.reshape(-1) for vec in time_vectors], dim=0)

    batch_tokens = sequence_tokens
    labels = label
    cfg_active = cfg_scale > 1.0
    if cfg_active:
        batch_tokens = torch.cat([batch_tokens, batch_tokens], dim=0)
        labels = torch.cat([label, torch.full_like(label, 1000)], dim=0)
        time_values = torch.cat([time_values, time_values], dim=0)

    token_embeddings = base_model.z_proj(batch_tokens)

    scaled_timesteps = (time_values * 1000).long()
    time_condition = base_model.time_embed(scaled_timesteps).squeeze()
    class_condition = base_model.class_emb(labels.long()).squeeze()

    if class_condition.ndim == 1:
        class_condition = class_condition.unsqueeze(0)
    if time_condition.ndim == 1:
        time_condition = time_condition.unsqueeze(0)

    step_factor = total_segments
    condition = class_condition.repeat(step_factor, 1) + time_condition

    causal_mask = build_causal_mask(
        total_segments=total_segments,
        segment_len=segment_len,
        device=token_embeddings.device,
        dtype=token_embeddings.dtype,
    )
    tail_mask = build_sequence_tail_mask(
        owner_indices,
        final_segment_flags,
        segment_len,
        device=token_embeddings.device,
        dtype=token_embeddings.dtype,
    )

    encoder_full_limit = len(base_model.encoder_blocks) if retain_encoder_layers < 0 else min(retain_encoder_layers, len(base_model.encoder_blocks))
    decoder_full_limit = len(base_model.decoder_blocks) if retain_decoder_layers < 0 else min(retain_decoder_layers, len(base_model.decoder_blocks))

    def make_positional_embedding(raw_embed: torch.Tensor) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        for owner in owner_indices:
            start = owner * segment_len
            end = start + segment_len
            chunks.append(raw_embed[:, start:end])
        return torch.cat(chunks, dim=1).to(device=token_embeddings.device, dtype=token_embeddings.dtype)

    encoder_pos_embed = make_positional_embedding(base_model.encoder_pos_embed_learned)
    decoder_pos_embed = make_positional_embedding(base_model.decoder_pos_embed_learned)

    index_map = torch.tensor(
        [owner * segment_len + offset for owner in owner_indices for offset in range(segment_len)],
        device=token_embeddings.device,
        dtype=torch.long,
    )

    def encoder_forward(x_tokens: torch.Tensor) -> torch.Tensor:
        set_rope_index_map(base_model.encoder_blocks, index_map)
        x = x_tokens + encoder_pos_embed
        x = base_model.z_proj_ln(x)
        for idx, blk in enumerate(base_model.encoder_blocks):
            mask = causal_mask if idx < encoder_full_limit else tail_mask
            x = blk(x, condition, mask, update_cache=False, scale_index=0)
        x = base_model.encoder_norm(x)
        clear_rope_index_map(base_model.encoder_blocks)
        return x

    def decoder_forward(x_tokens: torch.Tensor) -> torch.Tensor:
        set_rope_index_map(base_model.decoder_blocks, index_map)
        x = base_model.decoder_embed(x_tokens)
        x = x + decoder_pos_embed
        for idx, blk in enumerate(base_model.decoder_blocks):
            mask = causal_mask if idx < decoder_full_limit else tail_mask
            x = blk(x, condition, mask, update_cache=False, scale_index=0)
        x = base_model.decoder_norm(x)
        x = base_model.pred(x)
        clear_rope_index_map(base_model.decoder_blocks)
        return x

    encoded = encoder_forward(token_embeddings)
    decoded = decoder_forward(encoded)

    if cfg_active:
        cond, uncond = decoded.chunk(2, dim=0)
        decoded = uncond + cfg_scale * (cond - uncond)

    decoded = decoded[:base_batch]
    preds = decoded.reshape(batch_size, total_segments, segment_len, -1)
    return preds


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg_scale: float,
    image_cfg_scale: float,
    epoch: int,
    *,
    owner_indices: Sequence[int],
    final_segment_flags: Sequence[bool],
    per_patch_indices: Dict[str, Sequence[int]],
    writer: SummaryWriter | None,
    rank: int,
    steps_per_epoch: int,
    total_epochs: int,
    image_interval: int,
    image_labels: Sequence[int],
    image_steps: Sequence[int],
    vae: AutoencoderKL | None,
    image_seeds: Sequence[int],
    viz_trajectory_dir: str | None,
    viz_trajectory_sample: int | None,
    grad_accum_steps: int,
    clip_grad_norm: float,
    log_interval: int,
    ema_model: torch.nn.Module | None,
    ema_decay: float,
    global_start_time: float,
    use_autocast: bool,
    autocast_dtype: torch.dtype | None,
    scaler: GradScaler | None,
    retain_encoder_layers: int,
    retain_decoder_layers: int,
    save_every_steps: int,
    output_dir: Path,
    args_dict: Dict[str, Any],
) -> float:
    base_model = model.module if isinstance(model, DDP) else model
    base_model.train()

    grad_accum_steps = max(1, grad_accum_steps)
    total_loss = 0.0
    total_batches = 0
    total_training_steps = steps_per_epoch * total_epochs
    interval_loss = 0.0
    interval_batches = 0
    interval_start = time.time()

    optimizer.zero_grad(set_to_none=True)
    autocast_enabled = use_autocast and device.type == "cuda"
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    def autocast_context():
        if not autocast_enabled:
            return nullcontext()
        kwargs = {"device_type": autocast_device}
        if autocast_dtype is not None:
            kwargs["dtype"] = autocast_dtype
        return torch.autocast(**kwargs)

    use_scaler = scaler is not None and scaler.is_enabled()

    for batch_idx, batch in enumerate(loader):
        xt = batch["xt"].to(device, non_blocking=True)
        timestep = batch["timestep"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        (
            sequence_xt,
            sequence_timestep,
            sequence_xt_target,
            sequence_dt,
        ) = assemble_schedule_sequence(
            xt,
            timestep,
            per_patch_indices=per_patch_indices,
        )

        valid_mask = sequence_dt != 0
        pred_xt = None

        with autocast_context():
            preds = forward_teacher_sequence(
                base_model,
                sequence_xt,
                sequence_timestep,
                label,
                owner_indices=owner_indices,
                final_segment_flags=final_segment_flags,
                cfg_scale=cfg_scale,
                retain_encoder_layers=retain_encoder_layers,
                retain_decoder_layers=retain_decoder_layers,
            )
            dt = sequence_dt.unsqueeze(-1).unsqueeze(-1).to(dtype=preds.dtype)
            current_xt = sequence_xt.to(dtype=preds.dtype)
            target_xt = sequence_xt_target.to(dtype=preds.dtype)
            pred_xt = current_xt + dt * preds

            if valid_mask.any():
                loss = F.mse_loss(pred_xt[valid_mask], target_xt[valid_mask])
            else:
                loss = torch.zeros((), device=preds.device, dtype=preds.dtype)

        if pred_xt is None:
            pred_xt = sequence_xt.to(dtype=preds.dtype)

        loss_to_backprop = loss / grad_accum_steps
        if use_scaler:
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()
        batch_loss = loss.item()

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
                log_step_alignment_metrics(
                    writer,
                    pred_xt.detach().float() if pred_xt is not None else sequence_xt.detach().float(),
                    sequence_xt_target.detach().float(),
                    valid_mask.detach(),
                    global_step,
                )
                log_images(
                    writer,
                    ema_model if ema_model is not None else model,
                    vae,
                    labels=image_labels,
                    step_values=image_steps,
                    cfg_scale=image_cfg_scale,
                    device=device,
                    seeds=image_seeds,
                    global_step=global_step,
                    trajectory_dir=viz_trajectory_dir,
                    trajectory_sample=viz_trajectory_sample,
                )

        if rank == 0 and save_every_steps > 0:
            step_id = global_step + 1
            if step_id % save_every_steps == 0:
                ckpt_path = save_checkpoint(
                    output_dir,
                    base_model,
                    optimizer,
                    ema_model,
                    args_dict,
                    epoch=epoch,
                    global_step=step_id,
                )
                print(f"[Checkpoint] Saved {ckpt_path.name} (step {step_id})")

        should_step = ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx == steps_per_epoch - 1)
        if should_step:
            if use_scaler:
                if clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(base_model.parameters(), max_norm=clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
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
    args = base_parse_args()
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
        if len(explicit_values) < MAX_PATCH_STEPS:
            raise ValueError(
                f"Need at least {MAX_PATCH_STEPS} student-step-values for this schedule"
            )
    else:
        explicit_values = dataset.step_ids[:MAX_PATCH_STEPS]

    selected_indices, selected_steps = select_student_steps(
        dataset.step_ids,
        explicit_values=explicit_values,
        target_count=MAX_PATCH_STEPS,
    )

    if rank == 0:
        print(f"Using student supervision steps: {selected_steps}")

    per_patch_indices = make_patch_index_map(selected_indices)
    owner_indices, final_segment_flags = build_owner_and_final_flags()

    image_step_values = parse_step_values(args.image_step_values) if args.image_step_values else []
    if rank == 0 and not image_step_values and args.image_log_interval > 0:
        print("[WARN] image-step-values is empty; disabling image logging")
    image_labels_arg = parse_step_values(args.image_log_labels) if args.image_log_labels else []
    log_labels = prepare_log_labels(dataset, image_labels_arg, args.image_log_count)
    image_cfg_scale = args.image_cfg if args.image_cfg is not None else args.cfg
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
    del checkpoint
    model.requires_grad_(True)
    model.train()
    extend_attention_history(model, args.retain_encoder_layers, args.retain_decoder_layers)

    ema_model: torch.nn.Module | None = None
    if args.ema_decay > 0:
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False

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
        print("=" * 72)
        print("Teacher-forcing XYZFlow student distillation")
        print(f"World size    : {world_size}")
        print(f"Output dir    : {output_dir}")
        print(f"Log dir       : {Path(args.log_dir) if args.log_dir else Path('distill_tensorboard_output') / output_dir.name}")
        print(f"Trajectory dir: {args.trajectory_dir}")
        print(f"Checkpoint    : {args.checkpoint}")
        print(f"Mixed precision: {args.mixed_precision}")
        print("=" * 72)

    writer = None
    if rank == 0:
        log_dir = Path(args.log_dir) if args.log_dir else Path("distill_tensorboard_output") / output_dir.name
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))

    global_start_time = time.time()
    mixed_precision = args.mixed_precision.lower()
    if mixed_precision != "none" and device.type != "cuda":
        if rank == 0:
            print(f"[WARN] disabling mixed precision '{mixed_precision}' because CUDA is not available on this rank")
        mixed_precision = "none"
    use_autocast = mixed_precision in {"fp16", "bf16"}
    autocast_dtype = None
    if use_autocast:
        autocast_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    scaler = GradScaler() if mixed_precision == "fp16" and use_autocast else None

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        avg_loss = train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            args.cfg,
            image_cfg_scale,
            epoch,
            owner_indices=owner_indices,
            final_segment_flags=final_segment_flags,
            per_patch_indices=per_patch_indices,
            writer=writer,
            rank=rank,
            steps_per_epoch=len(loader),
            total_epochs=args.epochs,
            image_interval=args.image_log_interval,
            image_labels=log_labels,
            image_steps=image_step_values,
            vae=vae,
            image_seeds=image_seeds,
            viz_trajectory_dir=args.viz_trajectory_dir,
            viz_trajectory_sample=args.viz_trajectory_sample,
            grad_accum_steps=args.grad_accumulation_steps,
            clip_grad_norm=args.clip_grad_norm,
            log_interval=args.log_interval,
            ema_model=ema_model,
            ema_decay=args.ema_decay,
            global_start_time=global_start_time,
            use_autocast=use_autocast,
            autocast_dtype=autocast_dtype,
            scaler=scaler,
            retain_encoder_layers=args.retain_encoder_layers,
            retain_decoder_layers=args.retain_decoder_layers,
            save_every_steps=args.save_every_steps,
            output_dir=output_dir,
            args_dict=vars(args),
        )
        if rank == 0:
            print(f"[Epoch {epoch + 1}] avg_loss={avg_loss:.6f}")

    if rank == 0 and args.save_every_steps > 0 and len(loader) > 0:
        final_step = args.epochs * len(loader)
        ckpt_path = save_checkpoint(
            output_dir,
            base_model,
            optimizer,
            ema_model,
            vars(args),
            epoch=args.epochs - 1,
            global_step=final_step,
        )
        print(f"[Checkpoint] Saved final {ckpt_path.name} (step {final_step})")

    if rank == 0 and writer is not None:
        writer.close()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
