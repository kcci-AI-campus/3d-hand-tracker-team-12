#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$project_dir/dist/local_camera" ]]; then
  echo 'First run: bash scripts/build_pi.sh' >&2
  exit 1
fi
exec "$project_dir/dist/local_camera" "$@"
