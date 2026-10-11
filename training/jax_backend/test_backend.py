"""Check masking and differentiation independently of the Rust reference suite."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import jax
import numpy as np
import pytest

from backend import PARAMETERS, bc_loss, bc_step, load_batch, log_probabilities, logits, pack, pullback


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


def singleton_batch():
    return {
        "schema": "tzolkin-jax-numeric-batch-v1", "featureSchema": 2,
        "featureCount": 512, "hiddenCount": 32, "parameterCount": PARAMETERS,
        "sourceKind": "coreNumericFixture", "datasetFingerprint": None,
        "parameters": [0.0] * PARAMETERS,
        "samples": [{"id": "singleton", "split": "train", "phase": "Playing",
                     "chosen": 0, "features": [[0.0] * 512], "logits": [0.0],
                     "dlogits": [0.0], "gradient": [0.0] * PARAMETERS}],
    }


def test_cpu_only_cli_accepts_stationary_singleton(tmp_path):
    source, output = tmp_path / "batch.json", tmp_path / "verify.json"
    raw = json.dumps(singleton_batch()).encode()
    source.write_bytes(raw)
    env = os.environ.copy()
    env.pop("JAX_PLATFORMS", None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    command = [sys.executable, str(Path(__file__).with_name("run.py")), "verify",
               "--device", "cpu", "--input", str(source), "--output", str(output)]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["verification"]["bcLossBefore"] == 0
    assert report["verification"]["bcLossAfter"] == 0
    assert report["sourceSha256"] == hashlib.sha256(raw).hexdigest()


def test_input_hash_binds_loaded_snapshot(tmp_path):
    path = tmp_path / "batch.json"
    raw = json.dumps(singleton_batch()).encode()
    path.write_bytes(raw)
    data, digest = load_batch(path)
    path.write_bytes(b"changed after loading")
    assert data["samples"][0]["id"] == "singleton"
    assert digest == hashlib.sha256(raw).hexdigest()
