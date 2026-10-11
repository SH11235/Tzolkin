#!/usr/bin/env bash
set -euo pipefail

# Run in WSL2/Linux. The environment lives on the Linux filesystem.
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_env="${TZOLKIN_JAX_ENV:-$HOME/.venvs/tzolkin-jax}"
command -v uv >/dev/null || { echo 'Install uv in Linux before running this script.' >&2; exit 1; }
if [[ ! -x "$task_env/bin/python" ]]; then
  uv venv --python 3.12 "$task_env"
fi
uv pip sync --python "$task_env/bin/python" "$task_dir/requirements-wsl.lock"
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu "$task_env/bin/python" - <<'PY'
import jax
import jax.numpy as jnp
gpu = jax.devices("gpu")[0]
result = jax.device_put(jnp.ones((32, 32)), gpu)
result = (result @ result).block_until_ready()
assert float(result[0, 0]) == 32
print(f"JAX {jax.__version__}: {gpu}; GPU matmul OK")
PY
