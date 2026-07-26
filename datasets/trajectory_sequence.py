"""Dataset utilities for loading packed XYZFlow trajectories as fixed-length sequences."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


PATCH_NAMES: Tuple[str, ...] = ("A", "B", "C", "D")
FILENAME_RE = re.compile(r"label(\d+)_sample(\d+)\.npz")
LABEL_DIR_RE = re.compile(r"label(\d+)")
SAMPLE_PT_RE = re.compile(r"sample(\d+)\.pt")


@dataclass(frozen=True)
class TrajectoryKey:
    label: int
    sample: int


class TrajectorySequenceDataset(Dataset):
    """Loads precomputed teacher trajectories from either NPZ or PT format."""

    def __init__(
        self,
        root: str | Path,
        *,
        patch_names: Sequence[str] = PATCH_NAMES,
        dtype: np.dtype = np.float32,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Trajectory directory not found: {self.root}")

        self.patch_names = tuple(patch_names)
        self.dtype = dtype
        self._pt_format = self._has_pt_format()

        if self._pt_format:
            self._init_from_pt(max_samples=max_samples)
        else:
            self._init_from_npz(max_samples=max_samples)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int):
        key = self.keys[index]

        if self._pt_format:
            xt_np = np.zeros(
                (self.num_patches, self.num_steps, self.tokens, self.embed_dim),
                dtype=self.dtype,
            )
            # velocity_np = np.zeros_like(xt_np)
            timestep_np = np.zeros((self.num_patches, self.num_steps), dtype=np.float32)

            for patch_idx, patch_name in enumerate(self.patch_names):
                patch_dir = self.root / f"patch_{patch_name}"
                for step_idx, step in enumerate(self.step_ids):
                    file_path = (
                        patch_dir
                        / f"step_{step:03d}"
                        / f"label{key.label:03d}"
                        / f"sample{key.sample:04d}.pt"
                    )
                    if not file_path.exists():
                        raise FileNotFoundError(f"Missing trajectory file: {file_path}")
                    data = torch.load(file_path, map_location="cpu")
                    xt_np[patch_idx, step_idx] = data["xt"].numpy()
                    # velocity_np[patch_idx, step_idx] = data["velocity"].numpy()
                    timestep_np[patch_idx, step_idx] = float(data["timestep"])

            xt = torch.from_numpy(xt_np)
            # velocity = torch.from_numpy(velocity_np)
            timestep = torch.from_numpy(timestep_np)
        else:
            xt_np = np.zeros(
                (self.num_patches, self.num_steps, self.tokens, self.embed_dim),
                dtype=self.dtype,
            )
            # velocity_np = np.zeros_like(xt_np)
            timestep_np = np.zeros((self.num_patches, self.num_steps), dtype=np.float32)

            for patch_idx, patch_name in enumerate(self.patch_names):
                patch_dir = self.patch_dirs[patch_name]
                for step_idx, step in enumerate(self.step_ids):
                    file_path = (
                        patch_dir
                        / f"step_{step:03d}"
                        / f"label{key.label}_sample{key.sample:04d}.npz"
                    )
                    if not file_path.exists():
                        raise FileNotFoundError(f"Missing trajectory file: {file_path}")
                    data_np = np.load(file_path)
                    xt_np[patch_idx, step_idx] = data_np["xt"][0]
                    # velocity_np[patch_idx, step_idx] = data_np["velocity"][0]
                    timestep_np[patch_idx, step_idx] = float(data_np["timestep"][0])

            xt = torch.from_numpy(xt_np)
            # velocity = torch.from_numpy(velocity_np)
            timestep = torch.from_numpy(timestep_np)

        sample = {
            "xt": xt,
            # "velocity": velocity,
            "timestep": timestep,
            "label": torch.tensor(key.label, dtype=torch.long),
            "sample": torch.tensor(key.sample, dtype=torch.long),
        }
        return sample

    def _has_pt_format(self) -> bool:
        for patch_name in self.patch_names:
            patch_dir = self.root / f"patch_{patch_name}"
            if not patch_dir.exists():
                return False
        first_patch = self.root / f"patch_{self.patch_names[0]}"
        if not first_patch.exists():
            return False
        for step_dir in first_patch.glob("step_*"):
            if not step_dir.is_dir():
                continue
            for label_dir in step_dir.glob("label*"):
                if label_dir.is_dir() and any(label_dir.glob("sample*.pt")):
                    return True
        return False

    def _init_from_npz(self, *, max_samples: int | None) -> None:
        self.patch_dirs = {
            name: self.root / f"patch_{name}"
            for name in self.patch_names
        }
        missing = [name for name, path in self.patch_dirs.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing patch directories for: {', '.join(missing)} in {self.root}"
            )

        first_patch_dir = self.patch_dirs[self.patch_names[0]]
        self.step_ids = sorted(
            int(p.name.split("_")[1])
            for p in first_patch_dir.iterdir()
            if p.is_dir() and p.name.startswith("step_")
        )
        if not self.step_ids:
            raise RuntimeError(f"No step directories found in {first_patch_dir}")

        base_step_dir = first_patch_dir / f"step_{self.step_ids[0]:03d}"
        if not base_step_dir.exists():
            raise RuntimeError(f"Base step directory missing: {base_step_dir}")

        sample_files = sorted(base_step_dir.glob("label*_sample*.npz"))
        if not sample_files:
            raise RuntimeError(f"No trajectory files found in {base_step_dir}")

        keys: List[TrajectoryKey] = []
        for path in sample_files:
            match = FILENAME_RE.match(path.name)
            if match is None:
                continue
            label = int(match.group(1))
            sample = int(match.group(2))
            keys.append(TrajectoryKey(label=label, sample=sample))

        probe = np.load(sample_files[0])
        xt = probe["xt"]
        if xt.ndim != 3:
            raise ValueError(
                f"Expected xt with 3 dims [1, tokens, embed_dim], got shape {xt.shape}"
            )
        _, self.tokens, self.embed_dim = xt.shape

        self.num_patches = len(self.patch_names)
        self.num_steps = len(self.step_ids)

        valid_keys: List[TrajectoryKey] = []
        for key in keys:
            if all(
                (self.patch_dirs[patch_name] / f"step_{step:03d}" / f"label{key.label}_sample{key.sample:04d}.npz").exists()
                for patch_name in self.patch_names
                for step in self.step_ids
            ):
                valid_keys.append(key)

        if max_samples is not None:
            valid_keys = valid_keys[:max_samples]

        if not valid_keys:
            raise RuntimeError("No complete trajectory samples found across all patches/steps")

        self.keys = valid_keys
        self._finalize_common()

    def _init_from_pt(self, *, max_samples: int | None) -> None:
        self.patch_dirs = {
            name: self.root / f"patch_{name}"
            for name in self.patch_names
        }

        first_patch_dir = self.patch_dirs[self.patch_names[0]]
        self.step_ids = sorted(
            int(p.name.split("_")[1])
            for p in first_patch_dir.iterdir()
            if p.is_dir() and p.name.startswith("step_")
        )
        if not self.step_ids:
            raise RuntimeError(f"No step directories found in {first_patch_dir}")

        base_step_dir = first_patch_dir / f"step_{self.step_ids[0]:03d}"
        label_dirs = sorted(
            d for d in base_step_dir.iterdir() if d.is_dir() and LABEL_DIR_RE.match(d.name)
        )
        candidate_keys: List[TrajectoryKey] = []
        for label_dir in label_dirs:
            match_label = LABEL_DIR_RE.match(label_dir.name)
            if not match_label:
                continue
            label = int(match_label.group(1))
            for sample_path in sorted(label_dir.glob("sample*.pt")):
                match_sample = SAMPLE_PT_RE.match(sample_path.name)
                if not match_sample:
                    continue
                sample = int(match_sample.group(1))
                candidate_keys.append(TrajectoryKey(label=label, sample=sample))

        candidate_keys.sort(key=lambda k: (k.label, k.sample))
        if max_samples is not None:
            candidate_keys = candidate_keys[:max_samples]
        if not candidate_keys:
            raise RuntimeError("No PT trajectory files found")

        probe_path = (
            first_patch_dir
            / f"step_{self.step_ids[0]:03d}"
            / f"label{candidate_keys[0].label:03d}"
            / f"sample{candidate_keys[0].sample:04d}.pt"
        )
        probe = torch.load(probe_path, map_location="cpu")
        xt = probe["xt"].to(torch.float32)
        self.tokens = xt.shape[0]
        self.embed_dim = xt.shape[1]
        self.num_patches = len(self.patch_names)
        self.num_steps = len(self.step_ids)

        valid_keys: List[TrajectoryKey] = []
        for key in candidate_keys:
            complete = True
            for patch_name in self.patch_names:
                patch_dir = self.patch_dirs[patch_name]
                for step in self.step_ids:
                    file_path = (
                        patch_dir
                        / f"step_{step:03d}"
                        / f"label{key.label:03d}"
                        / f"sample{key.sample:04d}.pt"
                    )
                    if not file_path.exists():
                        complete = False
                        break
                if not complete:
                    break
            if complete:
                valid_keys.append(key)

        if not valid_keys:
            raise RuntimeError("No complete PT trajectories found across patches/steps")

        self.keys = valid_keys
        self._finalize_common()

    def _finalize_common(self) -> None:
        self.sequence_length = self.num_patches * self.num_steps * self.tokens
        self.patch_grid_size = int(np.sqrt(self.tokens))
        self.step_offsets: Dict[int, List[Tuple[int, int]]] = {step: [] for step in self.step_ids}
        cursor = 0
        for _ in range(self.num_patches):
            for step in self.step_ids:
                self.step_offsets[step].append((cursor, cursor + self.tokens))
                cursor += self.tokens
        clean_indices: List[int] = []
        if self.step_ids:
            last_step = self.step_ids[-1]
            for start, end in self.step_offsets[last_step]:
                clean_indices.extend(range(start, end))
        self.clean_indices = torch.tensor(clean_indices, dtype=torch.long)
        mask = torch.triu(torch.full((self.sequence_length, self.sequence_length), float("-inf")), diagonal=1)
        mask = torch.where(mask == 0, torch.zeros_like(mask), mask)
        self.full_history_mask = mask

    def get_sample(self, label: int, sample: int) -> Dict[str, torch.Tensor]:
        key = TrajectoryKey(label=label, sample=sample)
        if key not in self.keys:
            raise KeyError(f"No trajectory for label={label}, sample={sample}")
        idx = self.keys.index(key)
        return self[idx]



def build_dataloader(
    root: str | Path,
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    max_samples: int | None = None,
) -> DataLoader:
    dataset = TrajectorySequenceDataset(root, max_samples=max_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
