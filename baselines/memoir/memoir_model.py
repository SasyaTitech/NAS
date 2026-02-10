import copy
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

from .memoir_utils import brackets_to_periods, parent_module


class TopHasher:
    def __init__(self, input_dim: int, top_k: int, *, device: torch.device):
        self.top_k = int(top_k)
        self.input_dim = int(input_dim)
        self.permutation = torch.randperm(self.input_dim, device=device)

    def get_active_indices(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            if x.shape[0] != self.input_dim:
                raise ValueError(
                    f"TopHasher expected dim={self.input_dim}, got {x.shape[0]}"
                )
            _, indices = torch.topk(x.abs(), self.top_k, dim=0)
            return self.permutation[indices]
        if x.dim() == 2:
            if x.shape[1] != self.input_dim:
                raise ValueError(
                    f"TopHasher expected dim={self.input_dim}, got {x.shape[1]}"
                )
            _, indices = torch.topk(x.abs(), self.top_k, dim=1)
            return self.permutation[indices]
        raise ValueError(f"TopHasher expected 1D/2D tensor, got shape={tuple(x.shape)}")


class _EarlyStopMeter:
    def __init__(self):
        self.pre = 0.0
        self.val = 1e9

    def stop(self) -> bool:
        return abs(self.val - self.pre) <= 1e-4 and self.val <= 0.02

    def update(self, val: float) -> None:
        self.pre = self.val
        self.val = float(val)


class MEMOIRAdapter(torch.nn.Module):
    def __init__(self, config, layer: torch.nn.Module):
        super().__init__()
        self.config = config
        self.layer = layer
        self.weight = self.layer.weight
        self.device = self.weight.device

        self.prompt_feature_agg = config.prompt_feature_agg
        if self.prompt_feature_agg not in ("last", "mean", "mean_decentered"):
            raise ValueError(f"Unknown prompt_feature_agg: {self.prompt_feature_agg}")

        self.new_weight = torch.nn.Parameter(
            self.weight.detach().clone(), requires_grad=False
        )
        self.new_weight.data.zero_()

        self.original_layer = copy.deepcopy(self.layer)
        for p in self.original_layer.parameters():
            p.requires_grad = False
        self.training = False

        self.hasher = TopHasher(
            input_dim=self.new_weight.shape[1],
            top_k=self.config.top_k,
            device=self.device,
        )

        self.masks_for_edited_samples: Optional[Tensor] = None

        bg = torch.load(self.config.dir_background_features, map_location="cpu")[:100, :]
        bg_mean = torch.mean(bg, dim=0)
        if bg_mean.shape[0] != self.new_weight.shape[1]:
            raise ValueError(
                "Background feature dimension mismatch: "
                f"expected {self.new_weight.shape[1]}, got {bg_mean.shape[0]}"
            )
        self.loaded_irrelevant_sample_mean_features = bg_mean.to(self.device)

    def set_parameter_tunable(self) -> None:
        self.new_weight.requires_grad = True

    def _aggregate_prompt_features(self, x: Tensor, boundaries: Tensor) -> Tensor:
        """
        x: [B, S, D], boundaries: [B] (0-based index of last prompt token)
        returns: [B, D]
        """
        bsz, seq_len, _ = x.shape
        boundaries = boundaries.to(device=x.device).clamp(min=0, max=seq_len - 1)

        if self.prompt_feature_agg == "last":
            batch_idx = torch.arange(bsz, device=x.device)
            return x[batch_idx, boundaries, :]

        pos = torch.arange(seq_len, device=x.device).view(1, seq_len)
        mask = pos <= boundaries.view(bsz, 1)  # [B, S]
        denom = mask.sum(dim=1).clamp(min=1).unsqueeze(1)  # [B, 1]
        masked_sum = (x * mask.unsqueeze(2)).sum(dim=1)  # [B, D]
        mean = masked_sum / denom

        if self.prompt_feature_agg == "mean_decentered":
            return mean - self.loaded_irrelevant_sample_mean_features.unsqueeze(0)
        return mean

    def _ensure_masks_tensor(self) -> Tensor:
        if self.masks_for_edited_samples is None:
            return torch.empty((0, self.new_weight.shape[1]), dtype=torch.bool, device=self.device)
        return self.masks_for_edited_samples

    def new_weight_forward(self, x: Tensor) -> Tensor:
        bsz, _, dim = x.shape

        if self.training:
            prompt_boundary = self.last_prompt_token_loc[0]
            boundaries = torch.full((bsz,), int(prompt_boundary), device=x.device, dtype=torch.long)
            prompt_agg = self._aggregate_prompt_features(x, boundaries)
            active_indices = self.hasher.get_active_indices(prompt_agg[0])  # [top_k]
            active_mask = torch.zeros(dim, dtype=torch.bool, device=x.device)
            active_mask[active_indices] = True

            if self.masks_for_edited_samples is None:
                self.masks_for_edited_samples = active_mask.view(1, -1)
            else:
                if not torch.any(torch.all(self.masks_for_edited_samples == active_mask, dim=1)):
                    self.masks_for_edited_samples = torch.vstack(
                        (self.masks_for_edited_samples, active_mask)
                    )

            x_hashed = x * active_mask.view(1, 1, -1)
            return F.linear(x_hashed, self.new_weight)

        if not hasattr(self, "last_prompt_token_loc_inference"):
            raise RuntimeError(
                "MEMOIR inference requires last_prompt_token_loc_inference to be set."
            )
        boundaries = getattr(self, "last_prompt_token_loc_inference")
        if not isinstance(boundaries, Tensor):
            boundaries = torch.tensor(boundaries, device=x.device)
        boundaries = boundaries.to(device=x.device, dtype=torch.long)
        if boundaries.numel() != bsz:
            raise ValueError(
                f"Expected {bsz} prompt boundaries, got {boundaries.numel()}"
            )

        prompt_agg = self._aggregate_prompt_features(x, boundaries)  # [B, D]
        active_indices = self.hasher.get_active_indices(prompt_agg)  # [B, top_k]
        active_mask = torch.zeros((bsz, dim), dtype=torch.bool, device=x.device)
        active_mask.scatter_(1, active_indices, True)

        saved_masks = self._ensure_masks_tensor()  # [M, D]
        if saved_masks.numel() == 0:
            return torch.zeros((bsz, x.shape[1], self.new_weight.shape[0]), device=x.device, dtype=x.dtype)

        overlap_counts = active_mask.float() @ saved_masks.T.float()  # [B, M]
        overlap_ratios = overlap_counts / float(self.config.top_k)
        best_overlap, best_idx = torch.max(overlap_ratios, dim=1)  # [B]

        relevant = best_overlap >= float(self.config.irr_threshold)
        best_masks = saved_masks[best_idx]  # [B, D]
        final_mask = torch.where(relevant.view(-1, 1), best_masks, torch.zeros_like(best_masks))

        x_hashed = x * final_mask.to(dtype=x.dtype).view(bsz, 1, dim)
        return F.linear(x_hashed, self.new_weight)

    def forward(self, *args) -> Tensor:
        return self.original_layer(*args) + self.new_weight_forward(*args)


class MEMOIRWrapper(torch.nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

        inner = config.inner_params[0]
        suffixes = [".weight", ".bias"]
        layer_path = inner.rsplit(".", 1)[0] if any(inner.endswith(s) for s in suffixes) else inner
        self.layer = layer_path

        for _, p in self.model.named_parameters():
            p.requires_grad = False

        edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        layer_name = self.layer.rsplit(".", 1)[-1]
        existing = getattr(edit_module, layer_name)
        if type(existing) is not MEMOIRAdapter:
            adapter = MEMOIRAdapter(config, existing)
            setattr(edit_module, layer_name, adapter)

    def get_adapter_layer(self) -> MEMOIRAdapter:
        edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        layer_name = self.layer.rsplit(".", 1)[-1]
        adapter = getattr(edit_module, layer_name)
        if type(adapter) is not MEMOIRAdapter:
            raise RuntimeError("MEMOIR adapter layer not inserted correctly")
        return adapter

    def edit(self, tokens: dict, *, last_prompt_token_loc: Tensor) -> None:
        adapter = self.get_adapter_layer()
        adapter.training = True
        adapter.set_parameter_tunable()

        adapter.last_prompt_token_loc = last_prompt_token_loc

        loss_meter = _EarlyStopMeter()
        optimizer = torch.optim.SGD([adapter.new_weight], self.config.edit_lr, weight_decay=1e-5)

        for _ in range(int(self.config.n_iter)):
            logits = self.model(**tokens).logits

            shift_logits = logits[:-1, :-1, :].contiguous()
            shift_labels = tokens["labels"][:-1, 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_logits.size(0), -1)

            label_mask = torch.zeros_like(loss, dtype=torch.bool)
            for i, col_index in enumerate(last_prompt_token_loc[:-1]):
                start = max(0, int(col_index.item()) - 1)
                label_mask[i, start:] = True
            ft_loss = ((loss * label_mask).sum(1) / label_mask.sum(1)).mean()

            if loss_meter.stop():
                break

            optimizer.zero_grad()
            ft_loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.new_weight, max_norm=1.0)
            optimizer.step()
            loss_meter.update(ft_loss.item())

        adapter.training = False
        if hasattr(adapter, "last_prompt_token_loc"):
            delattr(adapter, "last_prompt_token_loc")
