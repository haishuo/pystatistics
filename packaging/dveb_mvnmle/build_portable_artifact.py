#!/usr/bin/env python3
"""Build the frozen DVEB MVN-MLE ABI in a controlled Linux builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parent
CPP = SOURCE_ROOT / "generated" / "mvnmle_cpu_dense.cpp"
ABI_HEADER = SOURCE_ROOT / "generated" / "mvnmle_cpu_abi_v1.h"
RUNTIME_HEADER = SOURCE_ROOT / "include" / "dense" / "dense_rt.h"
DEFAULT_OUTPUT = (
    ROOT / "pystatistics" / "mvnmle" / "_dveb" / "artifacts" / "mvnmle_cpu_abi_v1.so"
)
MANIFEST_PATH = DEFAULT_OUTPUT.with_name("manifest.json")

EXPECTED_INPUTS = {
    CPP: "407e77b0ad55a92719a3a70139901bdfde0576bb1da74054824e242ec3213879",
    ABI_HEADER: "7c4871bb51894f588773a1142c282dd370dcf5b5eedd374209f01a47db846be2",
    RUNTIME_HEADER: "1347391466b645fa2132c8621c279944ccf17d262392402d2e52500cac74d071",
}
DVEB_COMMIT = "a4b0286"
LANGUAGE_SOURCE_SHA256 = "a6ee8e60f6ed52e991ff53b12b1bee5ef6781a0f9506b324db18740e9227dc91"

COMPILE_FLAGS = [
    "-O3",
    "-std=c++17",
    "-fPIC",
    "-shared",
    "-fopenmp",
    "-ffp-contract=off",
    "-march=x86-64-v2",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_inputs() -> dict[str, str]:
    observed = {}
    for path, wanted in EXPECTED_INPUTS.items():
        actual = sha256(path)
        if actual != wanted:
            raise SystemExit(f"frozen build input drift: {path}: {actual} != {wanted}")
        observed[str(path.relative_to(ROOT))] = actual
    return observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()

    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise SystemExit("portable artifact build requires Linux x86-64")

    inputs = verify_inputs()
    compiler = os.environ.get("CXX", "g++")
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise SystemExit(f"C++ compiler not found: {compiler}")
    compiler_version = checked_output([compiler_path, "--version"]).splitlines()[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dveb-portable-build-") as scratch:
        temporary = Path(scratch) / args.output.name
        command = [
            compiler_path,
            *COMPILE_FLAGS,
            f"-I{SOURCE_ROOT / 'generated'}",
            f"-I{SOURCE_ROOT / 'include'}",
            str(CPP),
            "-o",
            str(temporary),
        ]
        subprocess.run(command, check=True)
        # The project bind mount and the container's temporary directory may
        # be different filesystems, so an atomic rename is not portable here.
        shutil.copy2(temporary, args.output)

    artifact_hash = sha256(args.output)
    ldd = checked_output(["ldd", str(args.output)])
    dynamic = checked_output(["readelf", "-d", str(args.output)])
    version_info = checked_output(["readelf", "--version-info", str(args.output)])
    manifest = {
        "schema": "pystatistics-dveb-mvnmle-consumer-artifact-v2",
        "status": "research-only; manylinux wheel candidate",
        "abi_version": 1,
        "artifact": args.output.name,
        "artifact_sha256": artifact_hash,
        "artifact_stage": "manylinux-builder-pre-auditwheel",
        "compiler_repository_commit": DVEB_COMMIT,
        "language_source_sha256": LANGUAGE_SOURCE_SHA256,
        "generated_inputs": inputs,
        "cpu_isa": "x86-64-v2",
        "platform": "manylinux_2_28_x86_64 candidate",
        "precision": "IEEE float64; contraction disabled; no fast-math",
        "compiler": compiler_version,
        "build_command": command[:-2] + ["-o", args.output.name],
        "dynamic_dependencies": ldd.splitlines(),
        "dynamic_section": dynamic.splitlines(),
        "version_info": version_info.splitlines(),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifact": str(args.output), "sha256": artifact_hash}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
