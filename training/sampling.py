"""Epoch-level two-source sampler migrated from the original notebook."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from torch.utils.data import Dataset as TorchDataset
    from torch.utils.data import Sampler
except ImportError:  # Keep pure sampler tests available before GPU deps are installed.
    class TorchDataset:  # type: ignore[no-redef]
        pass

    class Sampler:  # type: ignore[no-redef]
        pass


class TwoSourceTrainerDataset(TorchDataset):
    """Expose separate task pools through the global index space used by Trainer."""

    def __init__(self, train_pools: Dict[str, Any], source_names: Sequence[str]):
        if not isinstance(train_pools, dict) or not train_pools:
            raise ValueError("train_pools must be a non-empty dict")

        self.train_pools = dict(train_pools)
        self.source_names = [str(value) for value in source_names]
        missing = [name for name in self.source_names if name not in self.train_pools]
        if missing:
            raise ValueError(f"source_names contains missing pools: {missing}")

        self.lengths: Dict[str, int] = {}
        self.offsets: Dict[str, int] = {}
        cursor = 0
        for name in self.source_names:
            n = len(self.train_pools[name])
            if n <= 0:
                raise ValueError(f"train_pools[{name!r}] is empty")
            self.offsets[name] = cursor
            self.lengths[name] = int(n)
            cursor += int(n)
        self.total_len = int(cursor)

    def __len__(self) -> int:
        return self.total_len

    def _locate(self, global_index: int) -> Tuple[str, int]:
        idx = int(global_index)
        if idx < 0 or idx >= self.total_len:
            raise IndexError(f"global index out of range: {idx}")
        for name in self.source_names:
            start = self.offsets[name]
            end = start + self.lengths[name]
            if start <= idx < end:
                return name, idx - start
        raise IndexError(f"cannot locate global index: {idx}")

    def __getitem__(self, global_index: int) -> Dict[str, Any]:
        source_name, local_index = self._locate(global_index)
        row = dict(self.train_pools[source_name][int(local_index)])
        row["source"] = source_name
        return row


class BalancedEpochRatioSampler(Sampler):
    """Original 70/30 -> 60/40 -> 50/50 usage-balanced sampler."""

    def __init__(
        self,
        *,
        source_lengths: Dict[str, int],
        source_offsets: Dict[str, int],
        source_names: Sequence[str],
        epoch_ratio_schedule: Optional[Sequence[Dict[str, float]]] = None,
        samples_per_epoch: Optional[int] = None,
        seed: int = 42,
        shuffle_epoch_indices: bool = True,
        process_rank: int = 0,
        num_processes: int = 1,
        verbose: bool = True,
    ):
        self.source_names = [str(value) for value in source_names]
        self.source_lengths = {str(key): int(value) for key, value in source_lengths.items()}
        self.source_offsets = {str(key): int(value) for key, value in source_offsets.items()}

        for name in self.source_names:
            if name not in self.source_lengths:
                raise ValueError(f"missing source length for {name!r}")
            if name not in self.source_offsets:
                raise ValueError(f"missing source offset for {name!r}")
            if self.source_lengths[name] <= 0:
                raise ValueError(f"source {name!r} is empty")

        self.epoch_ratio_schedule = self._normalize_ratio_schedule(epoch_ratio_schedule)
        if samples_per_epoch is None:
            samples_per_epoch = sum(self.source_lengths[name] for name in self.source_names)
        self.samples_per_epoch = int(samples_per_epoch)
        if self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be > 0")

        self.seed = int(seed)
        self.shuffle_epoch_indices = bool(shuffle_epoch_indices)
        self.process_rank = int(process_rank or 0)
        self.num_processes = max(1, int(num_processes or 1))
        if not (0 <= self.process_rank < self.num_processes):
            raise ValueError(
                f"invalid process rank/world size: rank={self.process_rank}, "
                f"world={self.num_processes}"
            )
        self.verbose = bool(verbose)

        self.usage_counts: Dict[str, np.ndarray] = {
            name: np.zeros(self.source_lengths[name], dtype=np.int64)
            for name in self.source_names
        }
        self.epoch = 0
        self.last_epoch_info: Dict[str, Any] = {}
        self.current_epoch_global_indices: Optional[List[int]] = None
        self.current_epoch_local_indices: Optional[List[int]] = None
        self.current_epoch_number: Optional[int] = None
        self._resume_replay_global_indices: Optional[List[int]] = None
        self._resume_replay_epoch: Optional[int] = None
        self._resume_replay_once = False

    def _normalize_ratio_schedule(
        self, schedule: Optional[Sequence[Dict[str, float]]]
    ) -> List[Dict[str, float]]:
        if schedule is None:
            return [{name: 1.0 for name in self.source_names}]
        if (
            not isinstance(schedule, Sequence)
            or isinstance(schedule, (str, bytes))
            or len(schedule) == 0
        ):
            raise ValueError("epoch_ratio_schedule must be a non-empty sequence of dicts")

        normalized: List[Dict[str, float]] = []
        expected = set(self.source_names)
        for index, point in enumerate(schedule):
            if not isinstance(point, dict):
                raise ValueError(f"epoch_ratio_schedule[{index}] must be a dict")
            keys = {str(key) for key in point}
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            if missing or extra:
                raise ValueError(
                    f"epoch_ratio_schedule[{index}] keys mismatch: "
                    f"missing={missing}, extra={extra}"
                )

            output: Dict[str, float] = {}
            total = 0.0
            for name in self.source_names:
                value = float(point[name])
                if value < 0:
                    raise ValueError(f"ratio for {name!r} must be >= 0")
                output[name] = value
                total += value
            if total <= 0:
                raise ValueError("ratio sum must be > 0")
            normalized.append(output)
        return normalized

    def _ratio_for_epoch(self, epoch: int) -> Dict[str, float]:
        index = min(int(epoch), len(self.epoch_ratio_schedule) - 1)
        ratio = dict(self.epoch_ratio_schedule[index])
        total = sum(float(ratio[name]) for name in self.source_names)
        return {name: float(ratio[name]) / total for name in self.source_names}

    def _counts_for_epoch(self, ratio: Dict[str, float]) -> Dict[str, int]:
        raw = np.array(
            [float(ratio[name]) * float(self.samples_per_epoch) for name in self.source_names],
            dtype=np.float64,
        )
        floors = np.floor(raw).astype(np.int64)
        remainder = int(self.samples_per_epoch - int(floors.sum()))
        if remainder > 0:
            fractional_order = np.argsort(-(raw - floors))
            for position in fractional_order[:remainder]:
                floors[int(position)] += 1
        return {name: int(floors[index]) for index, name in enumerate(self.source_names)}

    def _balanced_take_local_indices(
        self, source_name: str, n: int, rng: np.random.Generator
    ) -> List[int]:
        n = int(n)
        if n <= 0:
            return []
        counts = self.usage_counts[source_name]
        if counts.size == 0:
            raise ValueError(f"cannot sample from empty source: {source_name}")

        selected: List[int] = []
        remaining = n
        while remaining > 0:
            min_count = int(counts.min())
            candidates = np.flatnonzero(counts == min_count)
            if candidates.size == 0:
                raise RuntimeError(
                    f"no candidates for source={source_name}, min_count={min_count}"
                )
            candidates = candidates.copy()
            rng.shuffle(candidates)
            take = min(remaining, int(candidates.size))
            picked = candidates[:take].astype(np.int64)
            selected.extend(int(value) for value in picked.tolist())
            counts[picked] += 1
            remaining -= take
        return selected

    def _build_epoch_global_indices(self, epoch: int) -> List[int]:
        ratio = self._ratio_for_epoch(epoch)
        per_source_n = self._counts_for_epoch(ratio)
        rng = np.random.default_rng(self.seed + int(epoch) * 1_000_003)
        global_indices: List[int] = []
        per_source_local: Dict[str, int] = {}

        for source_position, name in enumerate(self.source_names):
            n = int(per_source_n[name])
            source_rng = np.random.default_rng(
                self.seed + int(epoch) * 1_000_003 + source_position * 9_176
            )
            local_indices = self._balanced_take_local_indices(name, n, source_rng)
            offset = self.source_offsets[name]
            global_indices.extend(offset + int(value) for value in local_indices)
            per_source_local[name] = len(local_indices)

        if self.shuffle_epoch_indices:
            rng.shuffle(global_indices)

        usage_stats: Dict[str, Any] = {}
        for name in self.source_names:
            counts = self.usage_counts[name]
            unique_counts, unique_frequency = np.unique(counts, return_counts=True)
            usage_stats[name] = {
                "min": int(counts.min()),
                "max": int(counts.max()),
                "histogram": {
                    int(key): int(value)
                    for key, value in zip(unique_counts, unique_frequency)
                },
            }

        local_len = len(range(self.process_rank, len(global_indices), self.num_processes))
        self.last_epoch_info = {
            "epoch": int(epoch),
            "ratio": ratio,
            "requested_counts": per_source_n,
            "actual_counts": per_source_local,
            "global_samples_per_epoch": int(len(global_indices)),
            "process_rank": int(self.process_rank),
            "num_processes": int(self.num_processes),
            "local_samples_this_process": int(local_len),
            "usage_stats": usage_stats,
        }

        if self.verbose and self.process_rank == 0:
            print("\n[EPOCH MIXING SAMPLER]", flush=True)
            print(f"epoch={epoch}", flush=True)
            print(f"ratio={ratio}", flush=True)
            print(f"counts={per_source_n}", flush=True)
            print(f"usage_stats={usage_stats}", flush=True)

        self.current_epoch_number = int(epoch)
        self.current_epoch_global_indices = [int(value) for value in global_indices]
        self.current_epoch_local_indices = [
            int(value) for value in global_indices[self.process_rank :: self.num_processes]
        ]
        return global_indices

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "source_names": list(self.source_names),
            "source_lengths": dict(self.source_lengths),
            "source_offsets": dict(self.source_offsets),
            "epoch": int(self.epoch),
            "samples_per_epoch": int(self.samples_per_epoch),
            "seed": int(self.seed),
            "shuffle_epoch_indices": bool(self.shuffle_epoch_indices),
            "process_rank": int(self.process_rank),
            "num_processes": int(self.num_processes),
            "epoch_ratio_schedule": [dict(value) for value in self.epoch_ratio_schedule],
            "usage_counts": {
                name: counts.astype(np.int64).tolist()
                for name, counts in self.usage_counts.items()
            },
            "last_epoch_info": self.last_epoch_info,
            "current_epoch_number": self.current_epoch_number,
            "current_epoch_global_indices": self.current_epoch_global_indices,
            "current_epoch_local_indices": self.current_epoch_local_indices,
        }

    def load_state_dict(
        self, state: Dict[str, Any], *, replay_current_epoch: bool = False
    ) -> None:
        if not isinstance(state, dict):
            raise TypeError(f"sampler state must be a dict, got {type(state)}")
        saved_names = [str(value) for value in state.get("source_names", [])]
        if saved_names != self.source_names:
            raise ValueError(
                f"cannot restore sampler: source_names mismatch: "
                f"checkpoint={saved_names}, current={self.source_names}"
            )
        saved_lengths = {
            str(key): int(value) for key, value in state.get("source_lengths", {}).items()
        }
        if saved_lengths != self.source_lengths:
            raise ValueError(
                f"cannot restore sampler: source_lengths mismatch: "
                f"checkpoint={saved_lengths}, current={self.source_lengths}"
            )

        restored_counts: Dict[str, np.ndarray] = {}
        usage_counts = state.get("usage_counts", {})
        for name in self.source_names:
            if name not in usage_counts:
                raise ValueError(f"missing usage_counts[{name!r}]")
            values = np.asarray(usage_counts[name], dtype=np.int64)
            if values.shape[0] != self.source_lengths[name]:
                raise ValueError(
                    f"usage_counts[{name!r}] has {values.shape[0]} values; "
                    f"expected {self.source_lengths[name]}"
                )
            restored_counts[name] = values
        self.usage_counts = restored_counts

        self.epoch = int(state.get("epoch", 0) or 0)
        self.last_epoch_info = dict(state.get("last_epoch_info", {}) or {})
        self.current_epoch_number = state.get("current_epoch_number")
        if self.current_epoch_number is not None:
            self.current_epoch_number = int(self.current_epoch_number)
        current_global = state.get("current_epoch_global_indices")
        self.current_epoch_global_indices = (
            None if current_global is None else [int(value) for value in current_global]
        )
        current_local = state.get("current_epoch_local_indices")
        self.current_epoch_local_indices = (
            None if current_local is None else [int(value) for value in current_local]
        )

        self._resume_replay_global_indices = None
        self._resume_replay_epoch = None
        self._resume_replay_once = False
        if replay_current_epoch and self.current_epoch_global_indices is not None:
            self._resume_replay_global_indices = list(self.current_epoch_global_indices)
            self._resume_replay_epoch = self.current_epoch_number
            self._resume_replay_once = True

    def __iter__(self):
        if self._resume_replay_once and self._resume_replay_global_indices is not None:
            global_indices = [int(value) for value in self._resume_replay_global_indices]
            local_indices = global_indices[self.process_rank :: self.num_processes]
            self.current_epoch_global_indices = global_indices
            self.current_epoch_local_indices = [int(value) for value in local_indices]
            self.current_epoch_number = self._resume_replay_epoch
            self._resume_replay_once = False
            self._resume_replay_global_indices = None
            self._resume_replay_epoch = None
            if self.verbose and self.process_rank == 0:
                print("\n[EPOCH MIXING SAMPLER - REPLAY RESUMED EPOCH]", flush=True)
                print(f"epoch={self.current_epoch_number}", flush=True)
                print(f"local_samples_this_process={len(local_indices)}", flush=True)
            return iter(local_indices)

        epoch = int(self.epoch)
        self.epoch += 1
        global_indices = self._build_epoch_global_indices(epoch)
        local_indices = global_indices[self.process_rank :: self.num_processes]
        return iter(local_indices)

    def __len__(self) -> int:
        return len(range(self.process_rank, self.samples_per_epoch, self.num_processes))


# Names used in the notebook are retained as aliases for direct regression
# comparisons while the public names above are used by the refactored code.
_TwoSourceTrainerDataset = TwoSourceTrainerDataset
_BalancedEpochRatioSampler = BalancedEpochRatioSampler
