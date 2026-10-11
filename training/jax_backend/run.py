"""Verify Rust parity, benchmark devices, or run bounded experimental BC training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

# Set before importing JAX; leave room for the desktop and other sessions.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

from backend import (VERSION, bc_loss, bc_step, load_batch, logits,
                     log_probabilities, on_device, pack, pullback)


def device_for(name):
    devices = jax.devices(name)
    if not devices:
        raise RuntimeError(f"No {name} device available")
    return devices[0]


def verify(data, device):
    samples = data["samples"]
    features, mask, chosen = pack(samples)
    parameters = np.asarray(data["parameters"], np.float32)
    dlogits = np.zeros(mask.shape, np.float32)
    reference_gradient = np.zeros(parameters.shape, np.float64)
    for i, s in enumerate(samples):
        dlogits[i, :len(s["dlogits"])] = s["dlogits"]
        reference_gradient += np.asarray(s["gradient"])
    p, x, m, c, d = on_device(device, parameters, features, mask, chosen, dlogits)
    predicted = np.asarray(logits(p, x))
    error = 0.0
    for i, s in enumerate(samples):
        actual = predicted[i, :len(s["logits"])]
        np.testing.assert_allclose(actual, s["logits"], atol=2e-5, rtol=2e-4)
        error = max(error, float(np.max(np.abs(actual - s["logits"]))))
    gradient = np.asarray(pullback(p, x, d))
    np.testing.assert_allclose(gradient, reference_gradient, atol=2e-5, rtol=3e-4)
    probabilities = np.exp(np.asarray(log_probabilities(p, x, m)))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=2e-6)
    if np.any(probabilities[~mask] != 0):
        raise RuntimeError("Padding received probability")
    initial_loss = float(bc_loss(p, x, m, c))
    updated, _ = bc_step(p, x, m, c, np.float32(0.001))
    new_loss = float(bc_loss(updated, x, m, c))
    loss_tolerance = 5e-7 * max(1.0, abs(initial_loss))
    if (not np.isfinite(new_loss) or not np.isfinite(np.asarray(updated)).all()
            or new_loss > initial_loss + loss_tolerance):
        raise RuntimeError("BC step produced nonfinite values or increased loss")
    return {"device": str(device), "samples": len(samples), "logitMaxAbsError": error,
            "gradientMaxAbsError": float(np.max(np.abs(gradient - reference_gradient))),
            "bcLossBefore": initial_loss, "bcLossAfter": new_loss,
            "parity": "tolerance-checked, not bit-identical"}


def benchmark(data, device, repeats, batch_copies):
    samples = data["samples"] * batch_copies
    features, mask, chosen = pack(samples)
    started = time.perf_counter()
    p, x, m, c = on_device(device, np.asarray(data["parameters"], np.float32), features, mask, chosen)
    jax.block_until_ready((p, x, m, c))
    transfer_seconds = time.perf_counter() - started
    started = time.perf_counter()
    jax.block_until_ready(bc_step(p, x, m, c, np.float32(0.001)))
    compile_and_first = time.perf_counter() - started
    results = {"device": str(device), "batchShape": list(features.shape),
               "uniqueSamples": len(data["samples"]), "batchCopies": batch_copies,
               "hostToDeviceSeconds": transfer_seconds, "compileAndFirstStepSeconds": compile_and_first}
    for name, operation in (
        ("forward", lambda: logits(p, x)),
        ("bcStep", lambda: bc_step(p, x, m, c, np.float32(0.001))),
    ):
        jax.block_until_ready(operation())
        elapsed = []
        for _ in range(repeats):
            started = time.perf_counter()
            jax.block_until_ready(operation())
            elapsed.append(time.perf_counter() - started)
        median = float(np.median(elapsed))
        results[name] = {"medianSeconds": median, "p95Seconds": float(np.percentile(elapsed, 95)),
                         "samplesPerSecond": len(samples) / median}
    return results


def train(data, device, steps, learning_rate, source_sha256):
    train_samples = [s for s in data["samples"] if s["split"] == "train"]
    validation_samples = [s for s in data["samples"] if s["split"] == "validation"]
    if not validation_samples:
        raise ValueError("Validation partition required")
    x, m, c = on_device(device, *pack(train_samples))
    vx, vm, vc = on_device(device, *pack(validation_samples))
    p = jax.device_put(np.asarray(data["parameters"], np.float32), device)
    initial = {"train": float(bc_loss(p, x, m, c)), "validation": float(bc_loss(p, vx, vm, vc))}
    began = time.perf_counter()
    for _ in range(steps):
        p, _ = bc_step(p, x, m, c, np.float32(learning_rate))
    parameters = np.asarray(p)
    if not np.isfinite(parameters).all():
        raise RuntimeError("Nonfinite trained parameters")
    final = {"train": float(bc_loss(p, x, m, c)), "validation": float(bc_loss(p, vx, vm, vc))}
    return {"schema": "tzolkin-jax-experimental-weights-v1", "backendVersion": VERSION,
            "jaxVersion": jax.__version__, "device": str(device), "sourceSha256": source_sha256,
            "sourceKind": data["sourceKind"], "datasetFingerprint": data["datasetFingerprint"],
            "featureSchema": data["featureSchema"], "steps": steps, "learningRate": learning_rate,
            "featureCount": data["featureCount"], "hiddenCount": data["hiddenCount"],
            "parameterCount": data["parameterCount"],
            "parametersSha256": hashlib.sha256(parameters.astype("<f4").tobytes()).hexdigest(),
            "optimizer": "full-batch-sgd", "trainSamples": len(train_samples),
            "validationSamples": len(validation_samples), "initialLoss": initial, "finalLoss": final,
            "compileAndTrainSeconds": time.perf_counter() - began,
            "parameters": parameters.tolist(), "deployable": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "benchmark", "train"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--batch-copies", type=int, default=1,
                        help="Benchmark only: repeat fixed examples to measure batch-size sensitivity")
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000 or not 1 <= args.repeats <= 1000:
        parser.error("steps/repeats must be in 1..1000")
    if not 1 <= args.batch_copies <= 16 or (args.command != "benchmark" and args.batch_copies != 1):
        parser.error("batch-copies must be in 1..16 and only used with benchmark")
    if not np.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 0.01:
        parser.error("learning-rate must be in (0, 0.01]")
    if args.output.exists():
        parser.error("Output already exists")
    data, digest = load_batch(args.input)
    device = device_for(args.device)
    # Benchmark cold compilation before verification warms the same JIT functions.
    # Every command must pass reference verification before publishing a result.
    performance = benchmark(data, device, args.repeats, args.batch_copies) if args.command == "benchmark" else None
    parity = verify(data, device)
    if args.command == "verify":
        result = {"backendVersion": VERSION, "jaxVersion": jax.__version__,
                  "sourceSha256": digest, "verification": parity}
    elif args.command == "benchmark":
        result = {"backendVersion": VERSION, "jaxVersion": jax.__version__,
                  "sourceSha256": digest, "verification": parity,
                  "benchmark": performance,
                  "scope": "encoded fixed batch; excludes simulation, audit, encoding and storage"}
    else:
        result = train(data, device, args.steps, args.learning_rate, digest)
        result["verification"] = parity
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, allow_nan=False, indent=2)
        file.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "parameters"}, indent=2))


if __name__ == "__main__":
    main()
