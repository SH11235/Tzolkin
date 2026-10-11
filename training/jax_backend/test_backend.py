"""Check masking and differentiation independently of the Rust reference suite."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import pytest

from backend import PARAMETERS, bc_loss, bc_step, log_probabilities, logits, pack, pullback


def test_padding_is_excluded_and_singleton_has_zero_gradient():
    parameters = np.linspace(-0.01, 0.01, PARAMETERS, dtype=np.float32)
    x = np.zeros((1, 4, 512), np.float32)
    mask = np.array([[True, False, False, False]])
    probabilities = np.exp(np.asarray(log_probabilities(parameters, x, mask)))
    np.testing.assert_array_equal(probabilities, [[1, 0, 0, 0]])
    updated, loss = bc_step(parameters, x, mask, np.array([0], np.int32), np.float32(0.001))
    assert float(loss) == 0
    np.testing.assert_array_equal(updated, parameters)


def test_prefix_reuse_matches_full_affine_and_pullback_finite_difference():
    rng = np.random.default_rng(19)
    parameters = rng.normal(0, 0.02, PARAMETERS).astype(np.float32)
    x = rng.normal(0, 0.1, (2, 3, 512)).astype(np.float32)
    x[:, :, :384] = x[:, :1, :384]
    weights = parameters[:16384].reshape(32, 512)
    reference = np.tanh(x @ weights.T + parameters[16384:16416]) @ parameters[16416:-1] + parameters[-1]
    np.testing.assert_allclose(logits(parameters, x), reference, atol=1e-6, rtol=1e-5)
    dlogits = rng.normal(size=(2, 3)).astype(np.float32)
    gradient = np.asarray(pullback(parameters, x, dlogits))
    for i in (0, 383, 384, 16384, 16416, PARAMETERS - 1):
        plus, minus = parameters.copy(), parameters.copy()
        plus[i] += 0.001
        minus[i] -= 0.001
        numeric = (np.sum(np.asarray(logits(plus, x)) * dlogits)
                   - np.sum(np.asarray(logits(minus, x)) * dlogits)) / 0.002
        np.testing.assert_allclose(gradient[i], numeric, atol=2e-5, rtol=3e-3)


def test_padding_cannot_change_loss_or_gradient():
    rng = np.random.default_rng(29)
    p = rng.normal(0, 0.02, PARAMETERS).astype(np.float32)
    x = rng.normal(0, 0.1, (1, 4, 512)).astype(np.float32)
    x[:, :, :384] = x[:, :1, :384]
    mask = np.array([[True, True, False, False]])
    chosen = np.array([1], np.int32)
    a = jax.value_and_grad(bc_loss)(p, x, mask, chosen)
    x[:, 2:, 384:] = 100
    b = jax.value_and_grad(bc_loss)(p, x, mask, chosen)
    for left, right in zip(a, b):
        np.testing.assert_allclose(left, right, atol=1e-7, rtol=1e-6)


def test_pack_rejects_unbounded_padding():
    with pytest.raises(ValueError, match="capacity"):
        pack([{"features": [None] * 4096, "chosen": 0}] * 17)
