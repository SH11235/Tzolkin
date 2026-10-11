# Experimental JAX backend

Rust retains game rules, legal actions, feature encoding, source verification,
and existing BC/RL owners. This opt-in backend runs the public policy's
512-input, 32-hidden-unit MLP on a CUDA GPU or CPU. It supplies masked batched
inference, automatic parameter pullback from caller-supplied logit gradients,
and a bounded BC training experiment. It does not change the app's CPU or
connect to a running RL session.

## WSL2 setup

Use Ubuntu WSL2 with the Windows NVIDIA driver. Check `nvidia-smi` inside WSL;
do not install a Linux display driver. Install `uv` in Linux if unavailable.
From this checkout's directory **inside WSL**, run:

```bash
bash training/jax_backend/setup-wsl.sh
export TZOLKIN_JAX_PYTHON="$HOME/.venvs/tzolkin-jax/bin/python"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
export CARGO_TARGET_DIR="$HOME/.cache/tzolkin-jax-target"
```

The dedicated Python 3.12 environment and pinned CUDA 12 dependencies are on
the Linux filesystem. `TZOLKIN_JAX_ENV` overrides the environment directory.
The application does not require Python/JAX. The lock also pins pytest for the
backend tests. `uv pip sync` installs exactly this environment's packages.

## Rust/GPU parity and a training smoke run

```bash
mkdir -p implementation.local/jax
cargo run --release -p tzolkin-ai --example jax-export -- \
  fixture implementation.local/jax/fixture.json

"$TZOLKIN_JAX_PYTHON" -m pytest training/jax_backend -q
"$TZOLKIN_JAX_PYTHON" training/jax_backend/run.py verify \
  --input implementation.local/jax/fixture.json \
  --output implementation.local/jax/verify-gpu.json
"$TZOLKIN_JAX_PYTHON" training/jax_backend/run.py benchmark \
  --input implementation.local/jax/fixture.json --device gpu \
  --output implementation.local/jax/benchmark-gpu.json
"$TZOLKIN_JAX_PYTHON" training/jax_backend/run.py benchmark \
  --input implementation.local/jax/fixture.json --device cpu \
  --output implementation.local/jax/benchmark-cpu.json
"$TZOLKIN_JAX_PYTHON" training/jax_backend/run.py train \
  --input implementation.local/jax/fixture.json --steps 50 \
  --output implementation.local/jax/smoke-weights.json
```

All outputs are new files; reruns must use different output names. Fixture
labels are heuristic choices from short 3/4-player core trajectories with
Setup and Playing observations. They test the numeric pipeline, not playing
strength. The fixture's train/validation labels are smoke partitions, not an
independent validation experiment.

## Export a real native policy dataset

```bash
cargo run --release -p tzolkin-ai --example jax-export -- \
  dataset PATH_TO_NATIVE_POLICY_DATASET implementation.local/jax/native-batch.json
"$TZOLKIN_JAX_PYTHON" training/jax_backend/run.py train \
  --input implementation.local/jax/native-batch.json --steps 50 \
  --output implementation.local/jax/native-experimental-weights.json
```

The Rust exporter uses `load_policy_dataset` and consumes each Train/Validation
iterator through EOF, preserving source re-encoding and integrity checks even
for unselected rows. It selects up to 64 examples and 16,384 candidate rows per
partition, preserving the dataset's split. It does not select Test examples
for training or tuning (the full dataset loader still verifies their integrity).
The resulting JSON binds the dataset fingerprint and sample IDs. The Python
runner records the input file SHA-256. These bindings are not producer
authentication or a Python-side reconstruction of Rust's admission checks.

Selection also respects the runner's 65,536 padded-row limit across both
partitions, skipping candidates that would exceed it. Exported JSON is limited
to 128 MiB before publication. The runner hashes the same bounded byte snapshot
it parses, so a later path change cannot change the recorded source hash.

This first export initializes weights with the existing Rust model's seed 7;
it does not import a trained RL owner. Training is full-batch SGD on the bounded
selected examples, not the existing Scalar Adam trainer or the current Mixed
RL objective. Experimental weights use a separate schema, cannot be loaded
as an existing BC/RL checkpoint, and are not deployment artifacts. Do not
overwrite checkpoints belonging to another experiment.

## Numerical and performance limits

Rust's flat weight order is preserved: hidden-major W1, b1, Wp, bp. The first
384 feature columns share context within one candidate set and are computed
once. Candidate rows are padded to this batch's power-of-two maximum; a mask
removes padding from softmax and loss. Capacity guards reject oversized batches
rather than allocating 4,096 candidates for every observation. Larger datasets
need candidate-count buckets/streaming before sustained training.

Every CLI command checks logits and caller-supplied pullbacks against Rust's
Scalar reference, and checks that a small BC update does not increase loss
beyond f32 tolerance. Singleton/stationary batches are valid. GPU/f32
reductions differ from Rust's ordered f32 forward and f64 gradient accumulation;
tolerance checks replace bit equality **for this experimental backend only**.
Existing audit contracts and sampler/RNG semantics are unchanged. Passing the
fixture does not prove equality on all learned weights/states.

Benchmarks explicitly synchronize asynchronous device work and report host
transfer, first-step compilation, steady forward and BC-step medians/p95.
They measure encoded batches, not new game actions per second, and exclude
simulation, feature encoding, source audit and disk storage. CPU/GPU results
must use the same input SHA. This prototype has no integrated Rust-vs-JAX full
rollout benchmark or claim of overall RL speedup.

For batch-size sensitivity, add `--batch-copies 8` to both benchmark commands
and use new output names. This repeats the same examples in memory, not new
independent training data. Reports state the unique sample count, copy count,
and padded batch shape; the same capacity guard still applies. Small batches
can be faster on CPU because GPU launch/synchronization overhead dominates.
For a CPU-only machine, install CPU JAX and run with `JAX_PLATFORMS=cpu`.
The CLI itself does not force CUDA initialization for `--device cpu`.

Before connecting a long-run collector, qualify the current actor/critic and
actual objective against held-out recorded batches, measure full-pipeline
throughput, implement batching without changing policy sampling, and keep
new backend/optimizer lineage separate. A bridge for validated Mixed-RL
targets and critic updates is follow-up work.

References: [JAX installation](https://docs.jax.dev/en/latest/installation.html),
[JAX constraints](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html),
[benchmarking](https://docs.jax.dev/en/latest/benchmarking.html).
