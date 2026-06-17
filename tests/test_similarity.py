import math

import pytest
import torch

from eigen_attention.similarity import lorentz_displacement, lorentz_inner


def test_self_displacement_is_zero():
    q = torch.randn(4, 8)
    d = lorentz_displacement(q.unsqueeze(0), q.unsqueeze(0)).squeeze(0)
    diag = torch.diagonal(d)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-5)


def test_parallel_displacement_is_zero():
    u = torch.tensor([[1.0, 2.0, 3.0]])
    v = 2.0 * u
    d = lorentz_displacement(u.unsqueeze(0), v.unsqueeze(0)).squeeze()
    assert torch.allclose(d, torch.tensor(0.0), atol=1e-5)


def test_orthogonal_displacement_equals_norm_product():
    u = torch.tensor([[3.0, 0.0]])
    v = torch.tensor([[0.0, 4.0]])
    d = lorentz_displacement(u.unsqueeze(0), v.unsqueeze(0)).squeeze()
    assert torch.allclose(d, torch.tensor(12.0), atol=1e-5)


def test_antiparallel_displacement_is_twice_norm_product():
    u = torch.tensor([[1.0, 0.0, 0.0]])
    v = torch.tensor([[-1.0, 0.0, 0.0]])
    d = lorentz_displacement(u.unsqueeze(0), v.unsqueeze(0)).squeeze()
    assert torch.allclose(d, torch.tensor(2.0), atol=1e-5)


def test_displacement_is_nonnegative():
    torch.manual_seed(0)
    q = torch.randn(2, 16, 32)
    k = torch.randn(2, 16, 32)
    d = lorentz_displacement(q, k)
    assert (d >= -1e-5).all()


def test_lorentz_inner_is_nonpositive():
    torch.manual_seed(1)
    q = torch.randn(3, 8, 4)
    k = torch.randn(3, 8, 4)
    inner = lorentz_inner(q, k)
    assert (inner <= 1e-5).all()


def test_displacement_shape_2d():
    q = torch.randn(5, 7)
    k = torch.randn(11, 7)
    d = lorentz_displacement(q, k)
    assert d.shape == (5, 11)


def test_displacement_shape_3d():
    q = torch.randn(2, 5, 7)
    k = torch.randn(2, 11, 7)
    d = lorentz_displacement(q, k)
    assert d.shape == (2, 5, 11)


def test_displacement_shape_4d_multihead():
    q = torch.randn(2, 4, 5, 7)
    k = torch.randn(2, 4, 11, 7)
    d = lorentz_displacement(q, k)
    assert d.shape == (2, 4, 5, 11)


def test_displacement_is_symmetric_when_q_equals_k():
    torch.manual_seed(2)
    x = torch.randn(1, 8, 16)
    d = lorentz_displacement(x, x)
    assert torch.allclose(d, d.transpose(-1, -2), atol=1e-5)


def test_displacement_is_differentiable():
    q = torch.randn(2, 4, 8, requires_grad=True)
    k = torch.randn(2, 4, 8, requires_grad=True)
    d = lorentz_displacement(q, k)
    d.sum().backward()
    assert q.grad is not None and not torch.isnan(q.grad).any()
    assert k.grad is not None and not torch.isnan(k.grad).any()


def test_feature_dim_mismatch_raises():
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 9)
    with pytest.raises(ValueError, match="feature dim mismatch"):
        lorentz_displacement(q, k)


def test_inner_equals_negative_displacement():
    torch.manual_seed(3)
    q = torch.randn(2, 5, 8)
    k = torch.randn(2, 7, 8)
    assert torch.allclose(lorentz_inner(q, k), -lorentz_displacement(q, k), atol=1e-5)
