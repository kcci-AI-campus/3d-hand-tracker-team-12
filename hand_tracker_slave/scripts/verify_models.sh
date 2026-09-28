#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir/models"
if ! sha256sum --check SHA256SUMS; then
  echo 'Official converted models are missing/corrupt. Restore models/ from the ZIP or reconvert on a PC; see conversion/README.md.' >&2
  exit 1
fi
