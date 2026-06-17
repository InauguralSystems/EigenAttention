import pytest
import torch

from eigen_attention.attention import EigenAttention


def test_forward_shape_preserved():
    layer = EigenAttention(dim=32, num_heads=4)
    x = torch.randn(2, 10, 32)
    out, attn = layer(x)
    assert out.shape == (2, 10, 32)
    assert attn.shape == (2, 4, 10, 10)


def test_attention_rows_sum_to_one():
    torch.manual_seed(0)
    layer = EigenAttention(dim=16, num_heads=2)
    x = torch.randn(1, 8, 16)
    _, attn = layer(x)
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_self_diagonal_is_not_maximum():
    """The point of the primitive: self should not dominate attention rows."""
    torch.manual_seed(1)
    layer = EigenAttention(dim=32, num_heads=4)
    x = torch.randn(2, 12, 32)
    _, attn = layer(x)
    # for each (batch, head, query) row, the diagonal index should NOT be argmax.
    B, H, L, _ = attn.shape
    diag_is_max = 0
    for b in range(B):
        for h in range(H):
            for i in range(L):
                if attn[b, h, i].argmax().item() == i:
                    diag_is_max += 1
    # standard softmax(QK^T) on random Q, K from x typically picks self
    # ~30-60% of the time. We expect << that with EigenAttention.
    fraction = diag_is_max / (B * H * L)
    assert fraction < 0.15, f"diagonal-is-max fraction {fraction:.3f} too high"


def test_gradients_flow():
    layer = EigenAttention(dim=16, num_heads=4)
    x = torch.randn(2, 6, 16, requires_grad=True)
    out, _ = layer(x)
    out.sum().backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    for p in layer.parameters():
        assert p.grad is not None and not torch.isnan(p.grad).any()


def test_causal_mask_zeros_future():
    torch.manual_seed(2)
    layer = EigenAttention(dim=16, num_heads=2, causal=True)
    x = torch.randn(1, 6, 16)
    _, attn = layer(x)
    L = 6
    upper = torch.triu(torch.ones(L, L), diagonal=1).bool()
    assert torch.allclose(attn[0, 0][upper], torch.zeros(upper.sum()), atol=1e-6)


def test_external_mask_is_additive():
    torch.manual_seed(3)
    layer = EigenAttention(dim=16, num_heads=2)
    x = torch.randn(1, 5, 16)
    # mask out the last key for every query.
    mask = torch.zeros(1, 1, 5, 5)
    mask[..., -1] = float("-inf")
    _, attn = layer(x, attn_mask=mask)
    assert torch.allclose(attn[..., -1], torch.zeros_like(attn[..., -1]), atol=1e-6)


def test_dim_not_divisible_by_heads_raises():
    with pytest.raises(ValueError, match="must be divisible"):
        EigenAttention(dim=10, num_heads=4)


def test_wrong_input_dim_raises():
    layer = EigenAttention(dim=16, num_heads=2)
    with pytest.raises(ValueError, match="expected dim=16"):
        layer(torch.randn(1, 5, 8))


def test_wrong_input_rank_raises():
    layer = EigenAttention(dim=16, num_heads=2)
    with pytest.raises(ValueError, match="expected \\(B, L, D\\)"):
        layer(torch.randn(5, 16))
