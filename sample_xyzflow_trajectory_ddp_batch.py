
"""
Multi-GPU DDP trajectory sampling for XYZFlow models with BATCH support (Standard Teacher Inference).

This version supports batch-wise parallel sampling within each GPU for faster generation.
Uses the standard XYZFlow inference pipeline without KV cache modifications.

Usage:
    torchrun --nproc_per_node=8 sample_xyzflow_trajectory_ddp_batch.py \
        --checkpoint /path/to/XYZFlow-H.pth \
        --model-size huge \
        --output-dir trajectories \
        --num-samples 1 \
        --batch-size 32 \
        --steps-per-patch 50 \
        --save-steps 0 10 20 30 40 50 \
        --cfg 2.3

Output structure:
    trajectories/
    ├── label000/
    │   ├── sample0000.pt
    │   └── sample0001.pt
    ├── label001/
    └── ...
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from models import sampler as sampler_module
from models import xyzflow


PATCH_NAMES = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP XYZFlow latent trajectory sampler with batch support")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to XYZFlow checkpoint")
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["base", "large", "huge"],
        default="huge",
        help="XYZFlow model variant",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for trajectories")
    parser.add_argument("--num-samples", type=int, default=100, help="Samples per label")
    parser.add_argument("--num-classes", type=int, default=1000, help="Total number of classes when sampling a contiguous range")
    parser.add_argument(
        "--label-start",
        type=int,
        default=0,
        help="First class label to sample (inclusive)",
    )
    parser.add_argument(
        "--label-end",
        type=int,
        default=None,
        help="Last class label to sample (inclusive). If omitted, uses num-classes starting from label-start.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Number of labels to process in parallel per GPU")
    parser.add_argument("--steps-per-patch", type=int, default=50, help="Denoising steps per patch")
    parser.add_argument(
        "--save-steps",
        type=int,
        nargs="*",
        default=[0, 10, 20, 30, 40, 50],
        help="Denoising steps to record",
    )
    parser.add_argument("--cfg", type=float, default=2.3, help="Classifier-free guidance scale")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Computation precision for sampler (controls AMP)",
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Enable TF32 matmuls on Ampere+ GPUs for faster sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (use a negative value to randomize each run)",
    )
    parser.add_argument(
        "--metadata-flush-interval",
        type=int,
        default=10,
        help="Write metadata to disk every N processed labels per rank (0 disables periodic flush)",
    )
    parser.add_argument(
        "--write-workers",
        type=int,
        default=4,
        help="Number of concurrent writer threads per rank for PT files",
    )
    parser.add_argument(
        "--max-pending-writes",
        type=int,
        default=256,
        help="Max async write futures to keep in flight before blocking",
    )

    # DDP parameters
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP (set by torchrun)")

    return parser.parse_args()


def setup_ddp():
    """Initialize DDP environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def load_model(checkpoint_path: str, model_size: str, device: torch.device):
    """Load standard XYZFlow model (no KV cache modifications)."""
    constructor = {
        "base": xyzflow.xyzflow_base,
        "large": xyzflow.xyzflow_large,
        "huge": xyzflow.xyzflow_huge,
    }[model_size]

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

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()

    del checkpoint
    return model


def clear_kv_cache(model):
    """Clear KV cache in all attention layers."""
    for blk in model.encoder_blocks:
        blk.attn.k = None
        blk.attn.v = None
    for blk in model.decoder_blocks:
        blk.attn.k = None
        blk.attn.v = None


def euler_sampler_with_records_batch(
    model,
    latents: torch.Tensor,
    labels: torch.Tensor,
    patch_index: int,
    *,
    condition: torch.Tensor | None,
    num_steps: int,
    cfg_scale: float,
    mask: torch.Tensor | None,
    save_steps: Sequence[int],
) -> Tuple[torch.Tensor, List[List[Dict[str, torch.Tensor]]]]:
    """
    Batch Euler sampler with trajectory recording.

    Args:
        latents: [batch, 64, 16] initial noise
        labels: [batch] class labels
        patch_index: current patch index (0-3)
        condition: [batch, 64, 16] previous patch output (if any)
        num_steps: number of denoising steps
        cfg_scale: CFG scale
        mask: attention mask
        save_steps: which steps to record

    Returns:
        final_latents: [batch, 64, 16] final denoised latents
        records_batch: List of length batch, each containing list of dicts
    """
    device = latents.device
    dtype = latents.dtype
    batch = labels.shape[0]

    save_steps_set = set(save_steps)
    t_schedule = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64, device=device)

    x_t = latents.to(torch.float64)
    records_batch: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(batch)]

    def append_record(step_idx: int, t_cur: torch.Tensor, v_cur: torch.Tensor, tokens: torch.Tensor):
        """Record trajectory at specified steps for all samples in batch."""
        cond_v = v_cur[:batch]  # conditional branch (before CFG)
        t_cond = t_cur[:batch]
        for b in range(batch):
            records_batch[b].append({
                "step": torch.tensor(step_idx, dtype=torch.int32),
                "timestep": t_cond[b:b+1].detach().cpu().to(torch.float64),
                "xt": tokens[b:b+1].detach().cpu().to(dtype),
                "velocity": cond_v[b:b+1].detach().cpu().to(dtype),
            })

    # Main denoising loop
    for step_idx, (t_cur, t_next) in enumerate(zip(t_schedule[:-2], t_schedule[1:-1])):
        dt = t_next - t_cur

        # Prepare model input
        if condition is not None and step_idx == 0:
            model_input = torch.cat([condition, x_t.to(dtype)], dim=1)  # [batch, 128, 16]
        else:
            model_input = x_t.to(dtype)  # [batch, 64, 16]

        # Classifier-free guidance
        if cfg_scale > 1.0:
            model_input = torch.cat([model_input, model_input], dim=0)  # [batch*2, 64/128, 16]
            y_cur = torch.cat([labels, torch.ones_like(labels) * 1000], dim=0)
        else:
            y_cur = labels

        time_input = torch.ones(model_input.size(0), dtype=torch.float64, device=device) * t_cur

        # Model forward
        update_cache = (step_idx == 0)
        if update_cache:
            # First step: include time embedding prefix and use mask
            prefix = torch.zeros(
                model_input.size(0) * (1 if patch_index else 0),
                dtype=model_input.dtype,
                device=device
            )
            time_arg = torch.cat([prefix, time_input.to(dtype=model_input.dtype)], dim=0)
            v_cur = model.sample_inference(
                model_input,
                y_cur,
                time_arg,
                True,  # update_cache=True
                patch_index,
                mask,
            ).to(torch.float64)
        else:
            # Subsequent steps: no mask, update_cache=False
            v_cur = model.sample_inference(
                model_input,
                y_cur,
                time_input.to(dtype=model_input.dtype),
                False,  # update_cache=False
                patch_index,
            ).to(torch.float64)

        # Extract current tokens for recording
        current_tokens = model_input[:batch, -model.seq_len // model.clusters :]

        # Record if needed
        if step_idx in save_steps_set:
            append_record(step_idx, time_input, v_cur, current_tokens)

        # Compute drift
        input_tail = model_input[:, -model.seq_len // model.clusters :]
        score = sampler_module.get_score_from_velocity(v_cur, input_tail, time_input, path_type="linear")
        diffusion = sampler_module.compute_diffusion(t_cur)
        d_raw = v_cur - 0.5 * diffusion * score
        # 
        # Apply CFG
        if cfg_scale > 1.0:
            d_cond, d_uncond = d_raw.chunk(2)
            d_cur = d_uncond + cfg_scale * (d_cond - d_uncond)
        else:
            d_cur = d_raw

        # Euler-Maruyama step
        eps_i = torch.randn_like(x_t)
        deps = eps_i * torch.sqrt(torch.abs(dt))
        x_next = x_t + d_cur[:batch] * dt + torch.sqrt(diffusion) * deps
        x_t = x_next

    # Final step (t = 0)
    if num_steps in save_steps_set:
        t_cur = t_schedule[-2]
        t_next = t_schedule[-1]
        dt = t_next - t_cur

        model_input = x_t.to(dtype)
        if cfg_scale > 1.0:
            model_input = torch.cat([model_input, model_input], dim=0)
            y_cur = torch.cat([labels, torch.ones_like(labels) * 1000], dim=0)
        else:
            y_cur = labels

        time_input = torch.ones(model_input.size(0), dtype=model_input.dtype, device=device) * t_cur
        v_final = model.sample_inference(
            model_input,
            y_cur,
            time_input,
            False,
            patch_index,
        ).to(torch.float64)

        # Compute final drift for deterministic step
        input_tail = model_input[:, -model.seq_len // model.clusters :]
        score = sampler_module.get_score_from_velocity(v_final, input_tail, time_input, path_type="linear")
        diffusion = sampler_module.compute_diffusion(t_cur)
        d_raw = v_final - 0.5 * diffusion * score

        if cfg_scale > 1.0:
            d_cond, d_uncond = d_raw.chunk(2)
            d_cur = d_uncond + cfg_scale * (d_cond - d_uncond)
        else:
            d_cur = d_raw

        x_t = x_t + d_cur[:batch] * dt

        # Recompute velocity at the fully denoised state (t = 0) so that stored pairs match inference.
        zero_input = x_t.to(dtype)
        if cfg_scale > 1.0:
            zero_input = torch.cat([zero_input, zero_input], dim=0)
            y_zero = torch.cat([labels, torch.ones_like(labels) * 1000], dim=0)
        else:
            y_zero = labels

        time_zero = torch.zeros(zero_input.size(0), dtype=zero_input.dtype, device=device)
        v_zero = model.sample_inference(
            zero_input,
            y_zero,
            time_zero,
            False,
            patch_index,
        ).to(torch.float64)

        if cfg_scale > 1.0:
            v_store = v_zero[:batch]
        else:
            v_store = v_zero

        for b in range(batch):
            records_batch[b].append({
                "step": torch.tensor(num_steps, dtype=torch.int32),
                "timestep": torch.tensor([0.0], dtype=torch.float64),
                "xt": x_t[b:b+1].detach().cpu().to(dtype),
                "velocity": v_store[b:b+1].detach().cpu().to(dtype),
            })

    return x_t.to(dtype), records_batch


def main():
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("batch-size must be >= 1")

    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    precision_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    amp_dtype = precision_map[args.precision]
    amp_enabled = args.precision != "fp32"

    # Set seed (negative seed -> randomize per run)
    if args.seed >= 0:
        seed = args.seed + rank
    else:
        seed = int.from_bytes(os.urandom(8), "little") + rank
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))

    if rank == 0:
        print(f"Starting trajectory sampling with {world_size} GPUs")
        print(f"Batch size per GPU: {args.batch_size}")
        if args.label_end is not None:
            label_info = f"labels {args.label_start}..{args.label_end}"
        else:
            label_info = f"{args.num_classes} labels starting from {args.label_start}"
        print(f"Output directory: {args.output_dir}")
        print(f"Sampling label range: {label_info}")
        print(f"Steps per patch: {args.steps_per_patch}")
        print(f"Save steps: {args.save_steps}")
        print(f"Using STANDARD XYZFlow inference (no KV cache modifications)")

    # Ensure save_steps includes 0
    save_steps = sorted(set(args.save_steps))
    if save_steps[0] != 0:
        save_steps.insert(0, 0)

    # Load model (standard XYZFlow, no modifications)
    model = load_model(args.checkpoint, args.model_size, device)

    # Create output directory structure (per label directories)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    if world_size > 1:
        dist.barrier()

    # Divide labels among ranks
    if args.label_end is not None:
        if args.label_end < args.label_start:
            raise ValueError("label-end must be >= label-start")
        all_labels = list(range(args.label_start, args.label_end + 1))
    else:
        if args.num_classes < 1:
            raise ValueError("num-classes must be >= 1 when label-end is not set")
        all_labels = list(range(args.label_start, args.label_start + args.num_classes))
    labels_per_rank = len(all_labels) // world_size
    start_idx = rank * labels_per_rank
    end_idx = start_idx + labels_per_rank if rank < world_size - 1 else len(all_labels)
    my_labels = all_labels[start_idx:end_idx]

    if rank == 0:
        print(f"Each rank processes ~{labels_per_rank} labels")
    if my_labels:
        print(f"Rank {rank}: processing labels {my_labels[0]} to {my_labels[-1]}")
    else:
        print(f"Rank {rank}: no labels assigned")

    # Generate trajectories in batches
    total_samples = len(my_labels) * args.num_samples
    processed = 0

    save_steps = [int(s) for s in save_steps]
    write_workers = max(1, args.write_workers)
    executor = ThreadPoolExecutor(max_workers=write_workers)
    pending_writes = []
    max_pending_writes = max(1, args.max_pending_writes)

    total_batches = (len(my_labels) + args.batch_size - 1) // args.batch_size
    total_iters = total_batches * args.num_samples

    with torch.inference_mode():
        progress = tqdm(total=total_iters, disable=(rank != 0), desc="sampling")
        # Interleave samples across labels so that every label gets populated early
        for sample_idx in range(args.num_samples):
            for batch_start in range(0, len(my_labels), args.batch_size):
                batch_end = min(batch_start + args.batch_size, len(my_labels))
                batch_labels = my_labels[batch_start:batch_end]
                batch_size = len(batch_labels)

                # Clear KV cache for new batch
                clear_kv_cache(model)

                # Prepare batch tensors
                label_tensor = torch.tensor(batch_labels, device=device, dtype=torch.long)  # [batch]
                prev_patch_latent = None

                for patch_index, patch_name in enumerate(PATCH_NAMES):
                    latents = torch.randn(batch_size, model.seq_len // model.clusters, 16, device=device)

                    mask_len = (model.seq_len // model.clusters) * (patch_index + 1)
                    mask = model.mask[:, :, :mask_len, :mask_len]

                    final_latent, records_batch = euler_sampler_with_records_batch(
                        model,
                        latents,
                        label_tensor,
                        patch_index,
                        condition=prev_patch_latent,
                        num_steps=args.steps_per_patch,
                        cfg_scale=args.cfg,
                        mask=mask,
                        save_steps=save_steps,
                    )

                    for b, label_val in enumerate(batch_labels):
                        for rec in records_batch[b]:
                            step_id = int(rec["step"].item())

                            label_dir = (
                                output_dir
                                / f"patch_{patch_name}"
                                / f"step_{step_id:03d}"
                                / f"label{int(label_val):03d}"
                            )
                            label_dir.mkdir(parents=True, exist_ok=True)

                            payload = {
                                "xt": rec["xt"].squeeze(0).to(torch.float32).cpu(),
                                "velocity": rec["velocity"].squeeze(0).to(torch.float32).cpu(),
                                "timestep": float(rec["timestep"].item()),
                                "label": int(label_val),
                                "patch": patch_index,
                                "patch_name": patch_name,
                                "step": step_id,
                                "sample": int(sample_idx),
                            }
                            path = label_dir / f"sample{sample_idx:04d}.pt"
                            pending_writes.append(executor.submit(torch.save, payload, path))
                            if len(pending_writes) >= max_pending_writes:
                                pending_writes.pop(0).result()

                    prev_patch_latent = final_latent

                processed += batch_size
                if rank == 0 and processed % 100 == 0:
                    print(f"Progress: {processed}/{total_samples} samples ({100*processed/total_samples:.1f}%)")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if rank == 0:
                    progress.update(1)

        if rank == 0:
            progress.close()

    for future in pending_writes:
        future.result()
    executor.shutdown(wait=True)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        total_pt = sum(1 for _ in output_dir.rglob("*.pt"))
        print(f"\nCompleted! Saved {total_pt} PT trajectory files to {output_dir}")
        print("Structure: patch_X/step_YYY/labelZZZ/sample*.pt")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
