#!/usr/bin/env bash
# Build hand_tracker_master on the master Pi (Raspberry Pi OS 64-bit). hand_tracker_slave must sit
# next to this folder: the master uses its landmark models and code for its own camera.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
slave_dir="$(cd -- "$project_dir/../hand_tracker_slave" 2>/dev/null && pwd || true)"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != aarch64 ]]; then
  echo 'Build on Raspberry Pi OS 64-bit (Linux aarch64).' >&2
  exit 1
fi
if [[ -z "$slave_dir" || ! -f "$slave_dir/hand_detector.hpp" ]]; then
  echo 'hand_tracker_slave must be next to hand_tracker_master (copy both folders to the master Pi).' >&2
  exit 1
fi
jobs="${JOBS:-1}"
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo 'JOBS must be a positive integer.' >&2
  exit 2
fi
for tool in cmake ctest ninja c++ curl sha256sum; do
  command -v "$tool" >/dev/null || { echo "Missing $tool; see README.md" >&2; exit 1; }
done
bash "$slave_dir/scripts/verify_models.sh"
build_dir="$project_dir/build"
{
  cmake -S "$project_dir" -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release "$@"
  cmake --build "$build_dir" --parallel "$jobs"
  ctest --test-dir "$build_dir" --output-on-failure
  rm -rf "$project_dir/dist"
  mkdir -p "$project_dir/dist"
  cp "$build_dir/bin/hand_tracker_master" "$project_dir/dist/hand_tracker_master"
  cp -r "$build_dir/bin/models" "$project_dir/dist/models"
  printf '\nRun: bash "%s/run_master.sh"\n' "$project_dir"
} 2>&1 | tee "$project_dir/build_pi.log"
