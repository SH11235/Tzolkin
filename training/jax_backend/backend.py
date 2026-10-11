"""Experimental numeric backend for Rust's public 512 -> 32 -> 1 policy.

Inputs have already been encoded by Rust. This module owns no game rules,
sampling contract, dataset admission, or deployable policy provenance.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

FEATURES = 512
HIDDEN = 32
CONTEXT = 384
PARAMETERS = FEATURES * HIDDEN + HIDDEN + HIDDEN + 1
VERSION = "jax-public-mlp-f32-v1"


def load_batch(path: Path) -> dict:
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("Batch exceeds 128 MiB")
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = ("tzolkin-jax-numeric-batch-v1", 2, FEATURES, HIDDEN, PARAMETERS)
    actual = tuple(data.get(key) for key in (
        "schema", "featureSchema", "featureCount", "hiddenCount", "parameterCount"
    ))
    if actual != expected:
        raise ValueError("Unsupported feature/model contract")
    if data.get("sourceKind") not in {"coreNumericFixture", "verifiedNativePolicyExport"}:
        raise ValueError("Unsupported numeric source kind")
    fingerprint = data.get("datasetFingerprint")
    if data["sourceKind"] == "verifiedNativePolicyExport" and (
        not isinstance(fingerprint, str) or len(fingerprint) != 64
        or any(c not in "0123456789abcdef" for c in fingerprint)
    ):
        raise ValueError("Invalid source dataset fingerprint")
    params = np.asarray(data["parameters"], dtype=np.float32)
    if params.shape != (PARAMETERS,) or not np.isfinite(params).all():
        raise ValueError("Invalid parameters")
    samples = data["samples"]
    if not 1 <= len(samples) <= 128:
        raise ValueError("Invalid sample count")
    rows = 0
    ids = set()
    for sample in samples:
        x = np.asarray(sample["features"], dtype=np.float32)
        count = len(x)
        if x.shape != (count, FEATURES) or not 1 <= count <= 4096:
            raise ValueError("Invalid candidate shape")
        if not np.isfinite(x).all() or np.max(np.abs(x)) > 1024:
            raise ValueError("Invalid features")
        if not np.array_equal(x[:, :CONTEXT], np.broadcast_to(x[0, :CONTEXT], (count, CONTEXT))):
            raise ValueError("Candidates do not share a context")
        if (type(sample["chosen"]) is not int or not 0 <= sample["chosen"] < count
                or sample["split"] not in {"train", "validation"}):
            raise ValueError("Invalid target/split")
        if sample["id"] in ids:
            raise ValueError("Duplicate sample")
        ids.add(sample["id"])
        for key, shape in (("logits", (count,)), ("dlogits", (count,)), ("gradient", (PARAMETERS,))):
            value = np.asarray(sample[key], dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid reference {key}")
        rows += count
    if rows > 32768:
        raise ValueError("Batch exceeds row capacity")
    return data


def pack(samples: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad only to this batch's power-of-two candidate count, never to 4096 blindly."""
    if not samples:
        raise ValueError("Empty partition")
    width = 1 << (max(len(s["features"]) for s in samples) - 1).bit_length()
    if len(samples) * width > 65536:
        raise ValueError("Padded batch exceeds capacity; use candidate-count buckets")
    x = np.zeros((len(samples), width, FEATURES), dtype=np.float32)
    mask = np.zeros((len(samples), width), dtype=np.bool_)
    targets = np.asarray([s["chosen"] for s in samples], dtype=np.int32)
    for i, sample in enumerate(samples):
        count = len(sample["features"])
        x[i, :count] = sample["features"]
        mask[i, :count] = True
    return x, mask, targets


@jax.jit
def logits(parameters, features):
    """Flat Rust layout is hidden-major W1, b1, Wp, bp. Use highest matmul precision."""
    boundary = FEATURES * HIDDEN
    weights = parameters[:boundary].reshape(HIDDEN, FEATURES)
    bias = parameters[boundary:boundary + HIDDEN]
    output = parameters[boundary + HIDDEN:-1]
    context = jnp.matmul(features[:, 0, :CONTEXT], weights[:, :CONTEXT].T,
                         precision=jax.lax.Precision.HIGHEST)
    tail = jnp.matmul(features[:, :, CONTEXT:], weights[:, CONTEXT:].T,
                      precision=jax.lax.Precision.HIGHEST)
    hidden = jnp.tanh(context[:, None, :] + tail + bias)
    return jnp.matmul(hidden, output, precision=jax.lax.Precision.HIGHEST) + parameters[-1]


@jax.jit
def log_probabilities(parameters, features, mask):
    return jax.nn.log_softmax(jnp.where(mask, logits(parameters, features), -jnp.inf), axis=-1)


@jax.jit
def bc_loss(parameters, features, mask, chosen):
    logp = log_probabilities(parameters, features, mask)
    return -jnp.mean(jnp.take_along_axis(logp, chosen[:, None], axis=1))


bc_value_and_grad = jax.jit(jax.value_and_grad(bc_loss))


@jax.jit
def pullback(parameters, features, dlogits):
    """Caller supplies sign/normalization; no implied RL objective or sampling semantics."""
    _, backward = jax.vjp(lambda p: logits(p, features), parameters)
    return backward(dlogits)[0]


@jax.jit
def bc_step(parameters, features, mask, chosen, learning_rate):
    loss, gradient = bc_value_and_grad(parameters, features, mask, chosen)
    return parameters - learning_rate * gradient, loss


def on_device(device, *arrays):
    return tuple(jax.device_put(a, device) for a in arrays)
