# Reproduction and demo guide

## Build and test

Run from the repository root:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

CTest runs `solver` and `gpu_cross_check`. The cross-check uses the committed
reference/GPU CSVs and frames; it does not regenerate them.

## Regenerate all numerical artifacts

```bash
./build/sp_reference --out data/results_ref.csv --frames data/frames_ref
./build/sp_batched --out data/results_gpu.csv --frames data/frames_gpu
python3 tools/cross_check.py \
  data/results_ref.csv data/results_gpu.csv data/frames_ref data/frames_gpu
python3 tools/plot_curves.py data/results_ref.csv
python3 tools/animate_bands.py data/frames_ref
```

The verified 2026-10-17 regeneration converged at all 61 gate-bias points in
both models. Quantum sheet charge matched at stored precision; classical sheet
charge differed by at most `8.882e-7` relative, energies by `1.000002e-12 eV`,
and frames by `2.044846e-10` normalized.

## Demo outputs

1. **Charge and capacitance curves:** `data/idvg_curves.png`.
   The left panel compares classical Boltzmann charge with self-consistent
   Schrödinger–Poisson charge. Quantum charge is 17.1–19.8% lower for
   `Vg >= 0.34 V`; the reduction over the full default sweep is 17.1–66.3%.
   The right panel shows differential gate-charge capacitance `q·dNs/dVg`, not
   a standalone extraction of a series quantum-capacitance element.
2. **Band-bending animation:** `data/band_bending.gif`.
   The quantum ground-state density peaks away from the oxide interface, while
   classical charge peaks at the interface. The Delta2/Delta4 subband ladders
   reflect the silicon valley masses and degeneracies.
3. **Threshold-shift references:** the minimal undoped model gives
   `Delta V=32.59 mV` near `Ns=10¹² cm⁻²` and `68.36 mV` near `10¹³ cm⁻²`.
   These values are specific to the stated effective-mass model.

## Portability check

The GPU source can be compile-checked for Instinct `gfx942` with:

```bash
hipcc -std=c++17 -Iinclude --offload-arch=gfx942 -c src/sp_batched.cu
```

This command establishes source compilation only. Nanosheet AMD runtime
correctness and performance remain unverified.

## Offline presentation

The committed `data/idvg_curves.png`, `data/band_bending.gif`, and
`data/results_ref.csv` provide pre-rendered outputs and numerical magnitudes
without requiring a GPU during presentation.

## Related project

[`chiplet-thermal`](https://github.com/FireHorseAutumn/chiplet-thermal) models
steady-state conduction through a 2.5D compute/HBM package cross-section.
