import math

import pytest
import torch

from invokeai.backend.patches.layers.oft_layer import OFTLayer
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict


def _skew_symmetric(vec: torch.Tensor, block_size: int) -> torch.Tensor:
    batch_size = vec.shape[0]
    matrix = torch.zeros(batch_size, block_size, block_size, device=vec.device, dtype=vec.dtype)
    rows, cols = torch.triu_indices(block_size, block_size, 1, device=vec.device)
    matrix[:, rows, cols] = vec
    return matrix - matrix.transpose(-2, -1)


def _cayley_batch(weight: torch.Tensor, block_size: int) -> torch.Tensor:
    q_skew = _skew_symmetric(weight, block_size)
    result = torch.eye(block_size, device=weight.device, dtype=weight.dtype).repeat(weight.shape[0], 1, 1)
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


def _reference_oft_delta(orig_weight: torch.Tensor, oft_r_weight: torch.Tensor, is_scaled: bool) -> torch.Tensor:
    r_loaded, n_elements = oft_r_weight.shape
    block_size = int(round((1 + math.sqrt(1 + 8 * n_elements)) / 2))
    rank = math.prod(orig_weight.shape[1:]) // block_size
    effective_weight = oft_r_weight
    if is_scaled:
        effective_weight = effective_weight / (2 * math.sqrt(block_size - 1))
    rotation = _cayley_batch(effective_weight, block_size)
    if r_loaded == 1:
        rotation = rotation.repeat(rank, 1, 1)
    if orig_weight.dim() != 4:
        rotation = rotation.transpose(-1, -2)

    rotated_weight = torch.einsum(
        "ork,rkc->orc", orig_weight.reshape(orig_weight.shape[0], rank, block_size), rotation
    ).reshape(orig_weight.shape)
    return rotated_weight - orig_weight


def test_oft_layer_rotates_linear_weight():
    orig_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
    oft_r_weight = torch.tensor([[0.01], [-0.02]], dtype=torch.float32)
    layer = OFTLayer(oft_r_weight=oft_r_weight, is_scaled=False, dora_scale=None, initial_norm=None)

    assert torch.allclose(layer.get_weight(orig_weight), _reference_oft_delta(orig_weight, oft_r_weight, False))


def test_scaled_oft_layer_rotates_linear_weight():
    orig_weight = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 10
    oft_r_weight = torch.tensor([[0.01, -0.02, 0.03, -0.04, 0.05, -0.06]], dtype=torch.float32)
    layer = OFTLayer(oft_r_weight=oft_r_weight, is_scaled=True, dora_scale=None, initial_norm=None)

    assert torch.allclose(layer.get_weight(orig_weight), _reference_oft_delta(orig_weight, oft_r_weight, True))


def test_oft_layer_rotates_conv2d_weight():
    orig_weight = torch.arange(36, dtype=torch.float32).reshape(2, 2, 3, 3) / 10
    oft_r_weight = torch.tensor(
        [
            [0.01, -0.02, 0.03],
            [-0.04, 0.05, -0.06],
            [0.07, -0.08, 0.09],
            [-0.10, 0.11, -0.12],
            [0.13, -0.14, 0.15],
            [-0.16, 0.17, -0.18],
        ],
        dtype=torch.float32,
    )
    layer = OFTLayer(oft_r_weight=oft_r_weight, is_scaled=False, dora_scale=None, initial_norm=None)

    assert torch.allclose(layer.get_weight(orig_weight), _reference_oft_delta(orig_weight, oft_r_weight, False))


def test_doft_layer_applies_output_magnitude_scale():
    orig_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
    oft_r_weight = torch.tensor([[0.01], [-0.02]], dtype=torch.float32)
    dora_scale = torch.tensor([[1.0], [1.2], [1.4]], dtype=torch.float32)
    initial_norm = torch.tensor([[0.9], [1.1], [1.3]], dtype=torch.float32)
    layer = OFTLayer(
        oft_r_weight=oft_r_weight,
        is_scaled=False,
        dora_scale=dora_scale,
        initial_norm=initial_norm,
    )

    rotated_weight = orig_weight + _reference_oft_delta(orig_weight, oft_r_weight, False)
    expected = rotated_weight * (dora_scale / (initial_norm + torch.finfo(torch.float32).eps)) - orig_weight

    assert torch.allclose(layer.get_weight(orig_weight), expected)


def test_doft_layer_applies_output_multiplier():
    orig_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
    oft_r_weight = torch.tensor([[0.01], [-0.02]], dtype=torch.float32)
    dora_multiplier = torch.tensor([1.0, 1.2, 1.4], dtype=torch.float32)
    layer = OFTLayer(
        oft_r_weight=oft_r_weight,
        is_scaled=False,
        dora_scale=None,
        initial_norm=None,
        dora_multiplier=dora_multiplier,
    )

    rotated_weight = orig_weight + _reference_oft_delta(orig_weight, oft_r_weight, False)
    expected = rotated_weight * dora_multiplier.reshape(3, 1) - orig_weight

    assert torch.allclose(layer.get_weight(orig_weight), expected)


def test_doft_layer_applies_exp_output_log_multiplier():
    orig_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
    oft_r_weight = torch.tensor([[0.01], [-0.02]], dtype=torch.float32)
    dora_log_multiplier = torch.log(torch.tensor([1.0, 1.2, 1.4], dtype=torch.float32))
    layer = OFTLayer(
        oft_r_weight=oft_r_weight,
        is_scaled=False,
        dora_scale=None,
        initial_norm=None,
        dora_log_multiplier=dora_log_multiplier,
    )

    rotated_weight = orig_weight + _reference_oft_delta(orig_weight, oft_r_weight, False)
    expected = rotated_weight * torch.exp(dora_log_multiplier).reshape(3, 1) - orig_weight

    assert torch.allclose(layer.get_weight(orig_weight), expected)


def test_any_lora_layer_detects_doft_before_dora():
    layer = any_lora_layer_from_state_dict(
        {
            "oft_R.weight": torch.zeros(2, 1),
            "dora_log_multiplier": torch.zeros(3),
        }
    )

    assert isinstance(layer, OFTLayer)


def test_oft_layer_rejects_incompatible_block_count():
    layer = OFTLayer(oft_r_weight=torch.zeros(3, 1), is_scaled=False, dora_scale=None, initial_norm=None)

    with pytest.raises(ValueError, match="OFT block count mismatch"):
        layer.get_weight(torch.zeros(3, 4))
