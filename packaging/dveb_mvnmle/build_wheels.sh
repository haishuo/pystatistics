#!/usr/bin/env bash
set -euo pipefail

root=${1:-/project}
output=${2:-/output}
builder=(cp311-cp311 cp312-cp312)

mkdir -p "$output/raw" "$output/repaired" "$output/final" "$output/logs"

"/opt/python/cp311-cp311/bin/python" \
  "$root/packaging/dveb_mvnmle/build_portable_artifact.py" \
  | tee "$output/logs/artifact-build.json"

for tag in "${builder[@]}"; do
  python="/opt/python/$tag/bin/python"
  "$python" -m pip install --disable-pip-version-check \
    build hatchling 'cython>=3.0' 'numpy>=1.24' 'setuptools>=64' wheel \
    >"$output/logs/${tag}-build-dependencies.txt"

  # Cython extensions are build scratch. Never let one interpreter's ABI
  # object leak into the next wheel.
  find "$root/pystatistics" -type f -name '*.cpython-*.so' -delete
  rm -rf "$root/build"

  raw_dir="$output/raw/$tag"
  repaired_dir="$output/repaired/$tag"
  final_dir="$output/final/$tag"
  mkdir -p "$raw_dir" "$repaired_dir" "$final_dir"

  "$python" -m build --wheel --no-isolation --outdir "$raw_dir" "$root" \
    2>&1 | tee "$output/logs/${tag}-build.txt"
  raw_wheel=$(find "$raw_dir" -maxdepth 1 -type f -name '*.whl' -print -quit)
  auditwheel repair --plat manylinux_2_28_x86_64 \
    --wheel-dir "$repaired_dir" "$raw_wheel" \
    2>&1 | tee "$output/logs/${tag}-auditwheel-repair.txt"
  repaired_wheel=$(find "$repaired_dir" -maxdepth 1 -type f -name '*.whl' -print -quit)
  "$python" "$root/packaging/dveb_mvnmle/finalize_wheel.py" \
    "$repaired_wheel" --output-dir "$final_dir" \
    | tee "$output/logs/${tag}-finalize.json"
  final_wheel=$(find "$final_dir" -maxdepth 1 -type f -name '*.whl' -print -quit)
  auditwheel show "$final_wheel" \
    2>&1 | tee "$output/logs/${tag}-auditwheel-show.txt"
done
