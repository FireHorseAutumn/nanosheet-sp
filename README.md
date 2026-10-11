# nanosheet-sp — Poisson–Schrödinger nanosheet solver (AiSEM@Selangor 2026 artifact)

Self-consistent 1D Schrödinger–Poisson solver for the cross-section of a
~5nm Si nanosheet FET channel (N2/A16-class node). At this thickness
quantum confinement is a first-order effect: subbands split, the 2D density
of states replaces the 3D one, quantum capacitance eats gate bias is off by hundreds of millivolts in production N2 devices
(doped channel, work-function tuning, strain). The demo: charge–Vg curves,
classical vs. quantum, diverging — measured ΔV in this minimal undoped
model is 33–68 mV (see Results; the honest number, and it grows as the
sheet thins).

## Claim

A hand-written, dependency-free solver (Sturm bisection + Thomas
tridiagonal, no LAPACK/cuSOLVER) reproduces the classical-vs-quantum Vt
divergence in a 5nm double-gate Si nanosheet, batched across gate bias:
one threadblock owns one Vg point. Written in the HIP API so the same
source runs on Instinct; developed and validated locally on NVIDIA via a
thin HIP→CUDA shim (`include/hip_shim.h`) until ROCm/HIP is installed
(environment gap, recorded).

## Kill criteria / honesty rules

- Infinite-well ground truth: with tall barriers, subband energies must
  match E_n = n²ħ²π²/(2m*L²) to discretization error. If they don't, the
  solver is wrong; stop and fix before any demo output.
- Physics sign check: quantum sheet charge < classical sheet charge at the
  same Vg; divergence grows with Vg. If not, wrong.
- SCF must converge across the whole Vg sweep; non-converged points are
  reported, never silently dropped (house rule: no silent garbage).
- Boltzmann statistics for the classical curve is a stated approximation,
  printed in every output header.

## Layout

- `docs/PHYSICS.md` — the full physics/math spec (spec of record)
- `include/hip_shim.h` — HIP-API → CUDA-runtime shim for local dev
- `src/sp_reference.cpp` — host reference solver (plain C++, no deps)
- `src/sp_batched.cu` — HIP-API batched kernel (threadblock per Vg)
- `tools/plot_curves.py` — the curve plot (Ns/Cq vs Vg, classical vs quantum)
- `tools/animate_bands.py` — band-bending + n(z) animation sweeping Vg
- `data/` — CSV outputs + rendered frames

## Scope discipline (three days, summit Oct 14–17)

~400 lines of solver, ONE animation (band-bending + electron density vs
gate bias), ONE curve plot. Name-drop, don't build: "the production version
uses sp³d⁵s* tight-binding; effective-mass holds at 5nm and breaks at 2nm —
that boundary is the interesting part."

Pocket kernel (only if the main artifact lands early): finite-difference
heat solver over a 2.5D chiplet package (power map in → temperature field
out, HBM stack included) — the Penang assembly/test tie-in.

## Results (verified 2026-10-11, RTX 5060 laptop / CUDA 13.2, FP64)

- Kill criterion: tall-barrier 5nm Si well eigenvalues match the analytic
  E_n = n²ħ²π²/(2m*L²) within discretization error (ctest green).
- GPU batched (threadblock per Vg) vs host reference: Ns identical to
  machine precision, subband energies within 1e-12 eV, frames within
  2e-10 relative (`tools/cross_check.py`).
- **Measured divergence (this minimal model): ΔV = 33 mV at
  Ns = 10¹² cm⁻², 68 mV at 10¹³, quantum charge 15–20% below classical
  across the sweep.** The pitch's "hundreds of millivolts" belongs to
  production N2 devices (doped channel, metal-gate work function, strain);
  this undoped, effective-mass, Boltzmann-classical baseline honestly
  shows tens of mV — and the divergence grows as t_si shrinks. The demo
  line is the *trend and the mechanism*, not the inflated number.
- Quantum capacitance panel: dQ/dVg quantum sits ~15% below classical in
  inversion — the compact-model gap, on screen.

## Status

See STATE.md (breadcrumb: done / next / broken).
