#!/usr/bin/env bash
# Offline baseline: no service imports, API calls, or dependency installs.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python3 - <<'PY'
import ast
from pathlib import Path

root = Path('backend')
sources = [root / 'main.py', root / 'config.py']
for directory in ('db', 'models', 'routers', 'services'):
    sources.extend(sorted((root / directory).rglob('*.py')))
for path in sources:
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
print(f'PASS: Python syntax ({len(sources)} backend files)')
PY

if [[ ! -x frontend/node_modules/.bin/tsc ]]; then
  echo 'Missing frontend dependencies. Run npm ci from frontend/.' >&2
  exit 1
fi
cd frontend
./node_modules/.bin/tsc --noEmit --incremental false
echo 'PASS: frontend TypeScript'
echo 'Baseline passed; runtime behavior and live integrations need separate checks.'
