# DEMO.md — nanosheet-sp runbook (AiSEM@Selangor, Oct 14–17)

## The two exec questions (top of mind, always)

1. "At 5nm sheet thickness, quantum capacitance is eating drive current —
   how is your design team handling the compact-model gap at N2 versus N3?"
2. "How has the transshipment scrutiny changed your Penang test roadmap?"
   (NSS targets, Selangor IC Design Park + ARM ecosystem deal, AI-chip
   re-export permit rules — know these cold before the floor.)

## The demo (3 minutes)

Build (already built locally; on any machine with CUDA 13.2):

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build && ctest --test-dir build
```

1. **The curve** — `python3 tools/plot_curves.py` → `data/idvg_curves.png`.
   Talk track: classical Boltzmann vs self-consistent Schrödinger–Poisson.
   Quantum charge sits 15–20% lower at the same gate bias; the right panel
   is the punchline — dQ/dVg suppressed ~15% in inversion. That gap is
   quantum capacitance, and compact models that ignore it mislead timing
   closure. HONEST NUMBER: ΔV is 33–68 mV in this minimal undoped model;
   production N2 (doping, work function, strain) is where hundreds of mV
   live. Sell the trend and the mechanism, not an inflated number.
2. **The animation** — `data/band_bending.gif` (pre-rendered, in repo).
   Talk track: bands bend with gate bias; the quantum ground state peaks
   AWAY from the oxide interface while classical charge piles up against
   it; subband levels Δ2/Δ4 from valley degeneracy — that splitting is why
   the 2D density of states has two ladders.
3. **The portability line** — "the same source compiles for Instinct
   (gfx942 verified with hipcc 7.1.1); it runs on your silicon, not the
   other guy's." If asked for proof: `hipcc -std=c++17 -Iinclude
   --offload-arch=gfx942 -c src/sp_batched.cu`.
4. **The judgment line** (name-drop, don't build): "the production version
   uses sp³d⁵s* tight-binding; effective-mass holds at 5nm and breaks at
   2nm — that boundary is the interesting part."

## Fallback (if anything dies on the floor)

Everything is pre-rendered and committed: `data/idvg_curves.png`,
`data/band_bending.gif`, `data/results_ref.csv` (numbers for any question
about magnitudes). The laptop is not required — the artifacts are the demo.

## Verification one-liners (if challenged)

- ctest includes the infinite-well ground truth: 5nm Si well eigenvalues
  match n²·16.4 meV (analytic) within discretization error.
- GPU batched solver == host reference to machine precision
  (`tools/cross_check.py` prints it).
- SCF converges at every Vg in the sweep; non-convergence would be
  printed, not hidden.

## Pocket kernel (if conversation goes to packaging)

`../chiplet-thermal` — 2.5D package heat solver. The triangle: quantum
object underneath (TSMC N2 nanosheet), thermal object around it (chiplet
package — Penang assembly/test), simulation between them.
