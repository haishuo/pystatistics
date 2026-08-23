# Vendored ALLOY artifacts

`artifacts/` holds prebuilt binaries produced by the ALLOY compiler and runtime.
They are checked in so that installing PyStatistics needs no ALLOY checkout, no
compiler and no build step: `artifacts.json` records exactly what they were
built from, and the loader verifies every digest before use.

## What is here

| file | what it is |
|---|---|
| `liballoy.dylib` | the ALLOY runtime (device/context, bundles, dispatch) |
| `bootstrap.alloylib/manifest.json` | the bundle's contract — functions, params, dispatch domains |
| `bootstrap.alloylib/metal/bootstrap.metallib` | the Metal kernels |
| `bootstrap.alloylib/cpu/libbootstrap.dylib` | the CPU kernels from the same source |

The CPU half is part of the bundle rather than an extra: an ALLOY `hetero`
function is compiled for both targets from one source, and the bundle is the
unit the runtime opens. PyStatistics invokes only the Metal target.

## Provenance

`artifacts.json` is the machine-readable record and the loader checks against
it. It carries the ALLOY commit, whether that tree was clean, the compiler and
bundle-format versions, the SHA-256 of the `.alloy` source, and the SHA-256 of
every binary.

## Rebuilding

From a checkout of ALLOY at the recorded commit, on macOS/arm64:

```sh
cmake -S . -B build && cmake --build build -j
```

then copy `build/runtime/liballoy.dylib` and
`build/examples/bootstrap/bootstrap.alloylib` into `artifacts/` and regenerate
`artifacts.json` with the digests of what was copied. The loader refuses any
file whose digest does not match, so a partial refresh fails loudly rather than
running a mismatched pair.

## Scope

Built for **macOS on Apple Silicon (darwin/arm64) only**. On any other platform
the loader raises with the reason; PyStatistics' CPU and CUDA paths are
unaffected and do not touch this directory.

## LICENSING — UNRESOLVED, AND IT BLOCKS PUBLISHING

**ALLOY's README states: "Licensing: not yet decided; all rights reserved."**

PyStatistics is MIT-licensed and published to PyPI. Binaries under
all-rights-reserved terms cannot be redistributed inside an MIT-licensed
distribution, so **these artifacts must not be shipped in a released wheel or
sdist until ALLOY carries a license compatible with that redistribution.**

Both projects have the same author, so this is a decision to record rather than
a negotiation — but it has to be recorded, in ALLOY, before a release includes
this directory. Until then the backend is usable from a source checkout and the
packaging metadata below is prepared but unreleased.
