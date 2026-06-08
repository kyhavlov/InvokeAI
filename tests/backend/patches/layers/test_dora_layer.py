import torch

from invokeai.backend.patches.layers.dora_layer import DoRALayer


def test_dora_layer_supports_output_channel_scale_shape():
    orig_weight = torch.tensor(
        [
            [3.0, 4.0],
            [5.0, 12.0],
            [8.0, 15.0],
        ]
    )
    dora_scale = orig_weight.norm(dim=1, keepdim=True)
    layer = DoRALayer(
        up=torch.zeros(3, 1),
        down=torch.zeros(1, 2),
        dora_scale=dora_scale,
        alpha=1.0,
        bias=None,
    )

    assert torch.allclose(layer.get_weight(orig_weight), torch.zeros_like(orig_weight))


def test_dora_layer_preserves_input_channel_scale_shape():
    orig_weight = torch.tensor(
        [
            [3.0, 4.0],
            [5.0, 12.0],
            [8.0, 15.0],
        ]
    )
    dora_scale = orig_weight.transpose(0, 1).norm(dim=1, keepdim=True).transpose(0, 1)
    layer = DoRALayer(
        up=torch.zeros(3, 1),
        down=torch.zeros(1, 2),
        dora_scale=dora_scale,
        alpha=1.0,
        bias=None,
    )

    assert torch.allclose(layer.get_weight(orig_weight), torch.zeros_like(orig_weight))
