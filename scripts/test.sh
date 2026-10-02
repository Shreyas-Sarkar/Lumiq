#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${LUMIQ_TEST_PYTHON:-$repo_root/backend/.venv/bin/python}"
if [[ ! -x "$python_bin" && -x "$repo_root/backend/venv/bin/python" ]]; then
  python_bin="$repo_root/backend/venv/bin/python"
fi
cd "$repo_root/backend"
"$python_bin" -m unittest discover -s tests -v
