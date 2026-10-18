# nanosheet-sp

Self-consistent 1D Schrödinger–Poisson solver for the cross-section of a
symmetric double-gate Si nanosheet with an approximately 5 nm channel. The code
compares classical and quantum charge–gate-voltage curves and writes subband,
charge-density, and band-bending outputs.

At this thickness, confinement splits the subbands and the 2D density of states
replaces the 3D classical model. In the default undoped, effective-mass model,
the quantum-versus-classical threshold shift is approximately 33–68 mV near the
two reference sheet densities below.

## Implementation

- Sturm-sequence bisection for eigenvalues and inverse iteration with a Thomas
  tridiagonal solve for eigenvectors; no external math-library solver
- Adaptive self-consistent mixing with explicit convergence reporting
- Serial host reference and a HIP-API GPU implementation with one thread block
  per gate-bias point
- FP64 numerical validation on an RTX 5060 Laptop GPU through the CUDA shim in
  `include/hip_shim.h`
- Compile-level HIP validation for `gfx942`; AMD runtime correctness and
  performance have not been measured

## Build and test

Requirements: CMake 3.20+, a CUDA toolchain, and Python 3 with NumPy. Plotting
and animation also require Matplotlib and Pillow.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

CTest runs the direct solver tests and `gpu_cross_check` against committed
host/GPU artifacts. It does not regenerate those artifacts; see
[`DEMO.md`](DEMO.md) for the regeneration sequence.

## Numerical checks

- Tall-barrier eigenvalues approach the analytic infinite-well result within
  finite-difference dispersion error.
- Manufactured solutions check the Thomas and Poisson solvers.
- Charge normalization and eigenpair residuals are checked directly.
- Quantum sheet charge must be below classical sheet charge at fixed gate bias.
- Every point in the default self-consistent sweep must converge.

## Results

Validated on 2026-10-17 with CUDA 13.2 on an RTX 5060 Laptop GPU:

- Quantum sheet charge matches the host reference at stored CSV precision.
- Classical sheet charge differs by at most `8.882e-7` relative, subband
  energies by `1.000002e-12 eV`, and animation frames by `2.044846e-10`
  normalized. `tools/cross_check.py` enforces tolerances of `1e-5` for sheet
  charge, `1e-9 eV` for energies, and `1e-8` for frames.
- The quantum-versus-classical shift is 32.59 mV near `Ns=10¹² cm⁻²` and
  68.36 mV near `Ns=10¹³ cm⁻²`.
- Quantum charge is 17.1–19.8% below classical charge for `Vg >= 0.34 V`, and
  17.1–66.3% below over the full default sweep.
- The differential gate-charge capacitance `q·dNs/dVg` is approximately 15%
  lower for the quantum curve in inversion. This quantity is not a standalone
  extraction of a series quantum-capacitance element.
- All 61 gate-bias points converged in both models during cold regeneration.

## Repository layout

- `docs/PHYSICS.md`: physics and discretization
- `include/hip_shim.h`: HIP-API to CUDA-runtime development shim
- `src/sp_reference.cpp`: host reference solver
- `src/sp_batched.cu`: HIP-API batched GPU solver
- `tools/cross_check.py`: host/GPU numerical comparison
- `tools/plot_curves.py`: charge and differential-capacitance plots
- `tools/animate_bands.py`: band-bending and charge-density animation
- `data/`: committed CSV, frame, plot, and animation artifacts

## Model limits

The model uses a 1D effective-mass approximation, an undoped channel, finite
Si/SiO₂ barriers, and Boltzmann statistics for the classical comparison. It does
not include sp³d⁵s* tight binding, non-parabolicity, strain, transport, or
2D/3D gate-all-around corner effects. Results from this minimal model should not
be presented as production-node threshold predictions.
