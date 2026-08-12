"""Dynamic-padding collator migrated from the original training notebook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


IGNORE_INDEX = -100


@dataclass
class SFTAnswerOnlyCollator:
    tokenizer: Any
    label_pad_token_id: int = IGNORE_INDEX
    loss_weight_pad_value: float = 0.0

    def __call__(self, features: List[Dict[str, Any]]):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "SFTAnswerOnlyCollator requires PyTorch."
            ) from exc

        pad_keys = {"input_ids", "attention_mask", "token_type_ids"}
        labels = [feature["labels"] for feature in features]
        loss_weights = (
            [feature["loss_weights"] for feature in features]
            if "loss_weights" in features[0]
            else None
        )
        tokenizer_features = [
            {key: value for key, value in feature.items() if key in pad_keys}
            for feature in features
        ]
        batch = self.tokenizer.pad(
            tokenizer_features,
            padding=True,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]

        padded_labels = []
        for labels_one in labels:
            if isinstance(labels_one, torch.Tensor):
                labels_one = labels_one.tolist()
            if len(labels_one) > max_len:
                raise ValueError(f"Label length {len(labels_one)} > padded max_len {max_len}.")
            padded_labels.append(
                labels_one + [self.label_pad_token_id] * (max_len - len(labels_one))
            )
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        if loss_weights is not None:
            padded_loss_weights = []
            for weights_one in loss_weights:
                if isinstance(weights_one, torch.Tensor):
                    weights_one = weights_one.tolist()
                if len(weights_one) > max_len:
                    raise ValueError(
                        f"loss_weights length {len(weights_one)} > padded max_len {max_len}."
                    )
                padded_loss_weights.append(
                    weights_one
                    + [self.loss_weight_pad_value] * (max_len - len(weights_one))
                )
            batch["loss_weights"] = torch.tensor(
                padded_loss_weights,
                dtype=torch.float,
            )

        if not hasattr(self, "_debug_printed"):
            self._debug_printed = True
            print("\n[COLLATOR DEBUG]")
            print("feature keys[0]:", list(features[0].keys()))
            print("batch keys:", list(batch.keys()))
            print("input_ids shape:", tuple(batch["input_ids"].shape))
            print("labels shape:", tuple(batch["labels"].shape))
            if "loss_weights" in batch:
                print("loss_weights shape:", tuple(batch["loss_weights"].shape))
                print("loss_weights[0][:20]:", batch["loss_weights"][0][:20].tolist())
        return batch
