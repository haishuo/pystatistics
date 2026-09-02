# Exact-ARMA Phase-0B native-headroom evidence

This directory contains the immutable decision and ancillary observations for
the separately authorized native-diagnostic campaign run on Forge on
2026-09-02. The native binaries, build record, qualification, calibration, and
tolerance derivation are in the adjacent `dveb_arma_phase0b_native/` directory.

The load-bearing evidence is `l3-cuda-raw.json` plus its mechanical analysis.
The L1/L2 and L4 files are descriptive endpoint checks collected after the L3
decision; they do not revise it. Two duration-invalidated L1/L2 attempts are
preserved textually in `benchmarks/dveb_arma_phase0b/PROCESS_ATTEMPTS.md` and
contribute no observations here.

Verify offline from the repository root:

```bash
python3 benchmarks/dveb_arma_phase0b/verify_campaign.py
```

The verifier uses only the Python standard library. It checks raw evidence and
artifact hashes, native admission, observation counts, every stored numerical
admission result, all recorded ratios, and the frozen Case-2 rule.
