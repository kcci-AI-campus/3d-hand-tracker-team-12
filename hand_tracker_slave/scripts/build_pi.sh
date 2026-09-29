#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != aarch64 ]]; then
  echo 'Build on Raspberry Pi OS 64-bit (Linux aarch64).' >&2
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
bash "$project_dir/scripts/verify_models.sh"
build_dir="$project_dir/build"
{
  cmake -S "$project_dir" -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release "$@"
  cmake --build "$build_dir" --target hand_tracker_slave uv_sender_test cli_test --parallel "$jobs"
  ctest --test-dir "$build_dir" --output-on-failure
  mkdir -p "$project_dir/dist/models"
  cp "$build_dir/bin/hand_tracker_slave" "$project_dir/dist/hand_tracker_slave"
  cp "$project_dir/models/"*.param "$project_dir/models/"*.bin "$project_dir/dist/models/"
  cp "$project_dir/models/SHA256SUMS" "$project_dir/models/model_manifest.json" "$project_dir/dist/models/"
  # Remove only the retired lite model names from earlier builds.
  for dir in "$build_dir/bin/models" "$project_dir/dist/models"; do
    rm -f -- "$dir/palm-lite-op.param" "$dir/palm-lite-op.bin" "$dir/hand_lite-op.param" "$dir/hand_lite-op.bin"
  done
  printf '\nRun: bash "%s/run_slave.sh"\n' "$project_dir"
} 2>&1 | tee "$project_dir/build_pi.log"
