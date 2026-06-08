import math
from typing import Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.util.calc_tensor_size import calc_tensors_size


def _cast_to_device(t: torch.Tensor, to_device: torch.device) -> torch.Tensor:
    if t.device.type != to_device.type:
        return t.to(to_device)
    return t


class OFTLayer(BaseLayerPatch):
    """OneTrainer OFTv2/DOFT layer patch."""

    def __init__(
        self,
        oft_r_weight: torch.Tensor,
        is_scaled: bool,
        dora_scale: torch.Tensor | None,
        initial_norm: torch.Tensor | None,
        dora_multiplier: torch.Tensor | None = None,
        dora_log_multiplier: torch.Tensor | None = None,
    ):
        self.oft_r_weight = oft_r_weight
        self.is_scaled = is_scaled
        self.dora_scale = dora_scale
        self.initial_norm = initial_norm
        self.dora_multiplier = dora_multiplier
        self.dora_log_multiplier = dora_log_multiplier

    @classmethod
    def from_state_dict_values(cls, values: Dict[str, torch.Tensor]):
        return cls(
            oft_r_weight=values["oft_R.weight"],
            is_scaled="oft_R.scaled_oft" in values,
            dora_scale=values.get("dora_scale", None),
            initial_norm=values.get("initial_norm", None),
            dora_multiplier=values.get("dora_multiplier", None),
            dora_log_multiplier=values.get("dora_log_multiplier", None),
        )

    @staticmethod
    def _block_size_from_n_elements(n_elements: int) -> int:
        block_size = int(round((1 + math.sqrt(1 + 8 * n_elements)) / 2))
        if block_size * (block_size - 1) // 2 != n_elements:
            raise ValueError(f"Invalid OFT rotation weight shape: n_elements={n_elements}.")
        return block_size

    @staticmethod
    def _skew_symmetric(vec: torch.Tensor, block_size: int) -> torch.Tensor:
        batch_size = vec.shape[0]
        matrix = torch.zeros(batch_size, block_size, block_size, device=vec.device, dtype=vec.dtype)
        rows, cols = torch.triu_indices(block_size, block_size, 1, device=vec.device)

        # index_put works around a PyTorch advanced-indexing issue seen in OneTrainer's implementation.
        batch_idx = torch.arange(batch_size, device=vec.device)[:, None]
        matrix = matrix.index_put((batch_idx, rows, cols), vec)

        return matrix - matrix.transpose(-2, -1)

    @classmethod
    def _cayley_batch(cls, q: torch.Tensor, block_size: int) -> torch.Tensor:
        q_skew = cls._skew_symmetric(q, block_size)
        result = torch.eye(block_size, device=q.device, dtype=q.dtype).repeat(q.shape[0], 1, 1)

        # Match OneTrainer's default Cayley-Neumann approximation with 5 terms.
        result.add_(q_skew, alpha=2.0)
        q_squared = torch.bmm(q_skew, q_skew)
        result.add_(q_squared, alpha=2.0)
        q_power = q_squared
        for _ in range(3, 4):
            q_power = torch.bmm(q_power, q_skew)
            result.add_(q_power, alpha=2.0)
        q_power = torch.bmm(q_power, q_skew)
        result.add_(q_power)

        return result

    def _rotation_matrix(self, block_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        weight = self.oft_r_weight.to(device=device, dtype=dtype)
        if self.is_scaled:
            weight = weight / (2 * math.sqrt(block_size - 1))
        return self._cayley_batch(weight, block_size)

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        orig_weight = _cast_to_device(orig_weight, self.oft_r_weight.device)
        work_weight = orig_weight.to(dtype=self.oft_r_weight.dtype)
        r_loaded, n_elements = self.oft_r_weight.shape
        block_size = self._block_size_from_n_elements(n_elements)

        out_features = work_weight.shape[0]
        in_features = math.prod(work_weight.shape[1:])
        if in_features % block_size != 0:
            raise ValueError(
                f"OFT block size {block_size} is not compatible with input feature count {in_features}."
            )

        rank = in_features // block_size
        block_share = r_loaded == 1
        if not block_share and r_loaded != rank:
            raise ValueError(f"OFT block count mismatch: loaded={r_loaded}, expected={rank}.")

        rotation = self._rotation_matrix(block_size, work_weight.dtype, work_weight.device)
        if block_share:
            rotation = rotation.repeat(rank, 1, 1)

        if work_weight.dim() == 4:
            rotation_for_weight = rotation
        else:
            rotation_for_weight = rotation.transpose(-1, -2)

        flat_weight = work_weight.reshape(out_features, rank, block_size)
        rotated_weight = torch.einsum("ork,rkc->orc", flat_weight, rotation_for_weight).reshape(work_weight.shape)

        if self.dora_log_multiplier is not None:
            dora_multiplier = torch.exp(
                self.dora_log_multiplier.to(device=work_weight.device, dtype=work_weight.dtype)
            ).reshape(out_features, *([1] * (work_weight.dim() - 1)))
            rotated_weight = rotated_weight * dora_multiplier
        elif self.dora_multiplier is not None:
            dora_multiplier = self.dora_multiplier.to(device=work_weight.device, dtype=work_weight.dtype).reshape(
                out_features, *([1] * (work_weight.dim() - 1))
            )
            rotated_weight = rotated_weight * dora_multiplier
        elif self.dora_scale is not None:
            dora_scale = self.dora_scale.to(device=work_weight.device, dtype=work_weight.dtype).reshape(
                out_features, *([1] * (work_weight.dim() - 1))
            )
            if self.initial_norm is not None:
                norm = self.initial_norm.to(device=work_weight.device, dtype=work_weight.dtype).reshape(
                    out_features, *([1] * (work_weight.dim() - 1))
                )
            elif work_weight.dim() == 4:
                norm = work_weight.reshape(out_features, -1).norm(dim=1).reshape(out_features, 1, 1, 1)
            else:
                norm = work_weight.norm(dim=1, keepdim=True)

            rotated_weight = rotated_weight * (dora_scale / (norm + torch.finfo(work_weight.dtype).eps))

        return rotated_weight - work_weight

    def get_parameters(self, orig_parameters: dict[str, torch.Tensor], weight: float) -> dict[str, torch.Tensor]:
        if any(p.device.type == "meta" for p in orig_parameters.values()):
            raise RuntimeError(
                "The base model quantization format (likely bitsandbytes) is not compatible with OFT patches."
            )

        return {"weight": self.get_weight(orig_parameters["weight"]) * weight}

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        self.oft_r_weight = self.oft_r_weight.to(device=device, dtype=dtype)
        if self.dora_scale is not None:
            self.dora_scale = self.dora_scale.to(device=device, dtype=dtype)
        if self.initial_norm is not None:
            self.initial_norm = self.initial_norm.to(device=device, dtype=dtype)
        if self.dora_multiplier is not None:
            self.dora_multiplier = self.dora_multiplier.to(device=device, dtype=dtype)
        if self.dora_log_multiplier is not None:
            self.dora_log_multiplier = self.dora_log_multiplier.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        return calc_tensors_size(
            [self.oft_r_weight, self.dora_scale, self.initial_norm, self.dora_multiplier, self.dora_log_multiplier]
        )
