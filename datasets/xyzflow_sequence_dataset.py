"""Fixed-length sequence dataset for XYZFlow trajectory distillation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

PATCH_NAMES: Tuple[str, ...] = ("patch_A", "patch_B", "patch_C", "patch_D")


class XYZFlowSequenceDataset(Dataset):
    """Loads teacher trajectories as packed sequences suitable for Transformer training."""

    def __init__(
        self,
        root: str | Path,
        *,
        patch_order: Sequence[str] = PATCH_NAMES,
        dtype: np.dtype = np.float32,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Trajectory directory not found: {self.root}")

        self.patch_order = tuple(patch_order)
        self.dtype = dtype

        self.patch_dirs = [self.root / p for p in self.patch_order]
        for patch_dir in self.patch_dirs:
            if not patch_dir.exists():
                raise FileNotFoundError(f"Missing patch directory: {patch_dir}")

        self.steps = self._collect_steps(self.patch_dirs[0])
        if not self.steps:
            raise RuntimeError(f"No step_* directories found in {self.patch_dirs[0]}")

        base_files = sorted((self.patch_dirs[0] / f"step_{self.steps[0]:03d}").glob("label*_sample*.npz"))
        if not base_files:
            raise RuntimeError("No trajectory files found in base step directory")

        keys: List[Tuple[int, int]] = []
        for f in base_files:
            stem = f.stem  # label{label}_sample{xxxx}
            label = int(stem.split("_")[0][5:])
            sample = int(stem.split("_")[1][6:])
            keys.append((label, sample))

        if max_samples is not None:
            keys = keys[:max_samples]

        self.keys = self._filter_complete(keys)
        if not self.keys:
            raise RuntimeError("No complete trajectories across all patches/steps")
        self.key_to_index = {key: idx for idx, key in enumerate(self.keys)}

        probe = np.load(
            self.patch_dirs[0]
            / f"step_{self.steps[0]:03d}"
            / f"label{self.keys[0][0]}_sample{self.keys[0][1]:04d}.npz"
        )
        self.tokens = probe["xt"].shape[1]
        self.dim = probe["xt"].shape[2]

        self.num_patches = len(self.patch_order)
        self.num_steps = len(self.steps)
        self.patch_grid_size = int(np.sqrt(self.tokens))
        self.patch_rows = int(np.sqrt(self.num_patches))
        self._sequence_length = self.num_patches * self.num_steps * self.tokens

        self.step_offsets = self._build_step_offsets()
        self.clean_indices = self._compute_clean_indices()
        self.full_history_mask = self._build_full_history_mask()

    @staticmethod
    def _collect_steps(patch_dir: Path) -> List[int]:
        step_ids = []
        for step_dir in sorted(patch_dir.iterdir()):
            if step_dir.is_dir() and step_dir.name.startswith("step_"):
                step_ids.append(int(step_dir.name.split("_")[1]))
        return step_ids

    def _filter_complete(self, keys: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        complete = []
        for label, sample in keys:
            ok = True
            for patch_dir in self.patch_dirs:
                for step in self.steps:
                    path = patch_dir / f"step_{step:03d}" / f"label{label}_sample{sample:04d}.npz"
                    if not path.exists():
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                complete.append((label, sample))
        return complete

    def _build_step_offsets(self) -> Dict[int, List[Tuple[int, int]]]:
        offsets: Dict[int, List[Tuple[int, int]]] = {step: [] for step in self.steps}
        cursor = 0
        for _patch in self.patch_order:
            for step in self.steps:
                offsets[step].append((cursor, cursor + self.tokens))
                cursor += self.tokens
        return offsets

    def _compute_clean_indices(self) -> torch.Tensor:
        indices: List[int] = []
        clean_step = self.steps[-1]
        for start, end in self.step_offsets[clean_step]:
            indices.extend(range(start, end))
        return torch.tensor(indices, dtype=torch.long)

    def _build_full_history_mask(self) -> torch.Tensor:
        mask = torch.triu(torch.full((self.sequence_length, self.sequence_length), float("-inf")), diagonal=1)
        mask = torch.where(mask == 0, torch.zeros_like(mask), mask)
        return mask

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def attention_mask(self) -> torch.Tensor:
        return self.full_history_mask

    def build_attention_masks(self, *, num_layers: int, full_history_layers: int) -> List[torch.Tensor]:
        masks: List[torch.Tensor] = []
        global_mask = self.full_history_mask.clone()

        clean_mask = torch.full((self.sequence_length, self.sequence_length), float("-inf"))
        clean_mask[:, self.clean_indices] = 0.0
        clean_mask[torch.arange(self.sequence_length), torch.arange(self.sequence_length)] = 0.0

        for layer in range(num_layers):
            if layer < full_history_layers:
                masks.append(global_mask.clone())
            else:
                masks.append(clean_mask.clone())
        return masks

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        label, sample = self.keys[idx]
        tokens = np.zeros((self.sequence_length, self.dim), dtype=self.dtype)
        velocity = np.zeros_like(tokens)
        timestep = np.zeros((self.sequence_length,), dtype=np.float32)

        cursor = 0
        for patch_dir in self.patch_dirs:
            for step in self.steps:
                path = patch_dir / f"step_{step:03d}" / f"label{label}_sample{sample:04d}.npz"
                data = np.load(path)
                xt = data["xt"][0]
                vel = data["velocity"][0]
                t = float(data["timestep"][0])

                tokens[cursor : cursor + self.tokens] = xt
                velocity[cursor : cursor + self.tokens] = vel
                timestep[cursor : cursor + self.tokens] = t
                cursor += self.tokens

        return {
            "tokens": torch.from_numpy(tokens),
            "velocity": torch.from_numpy(velocity),
            "timestep": torch.from_numpy(timestep),
            "label": torch.tensor(label, dtype=torch.long),
            "sample": torch.tensor(sample, dtype=torch.long),
        }

    def get_patch_step_tokens(self, seq_tokens: torch.Tensor, step: int) -> torch.Tensor:
        slices = self.step_offsets[step]
        patches = [seq_tokens[start:end] for start, end in slices]
        return torch.stack(patches, dim=0)

    def get_sample(self, label: int, sample: int) -> Dict[str, torch.Tensor]:
        key = (label, sample)
        idx = self.key_to_index.get(key)
        if idx is None:
            raise KeyError(f"No trajectory for label={label}, sample={sample}")
        return self[idx]


__all__ = ["XYZFlowSequenceDataset"]
