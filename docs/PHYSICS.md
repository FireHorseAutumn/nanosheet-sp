# PHYSICS.md — 1D Schrödinger–Poisson spec (spec of record)

Self-consistent effective-mass solver for the cross-section of a symmetric
double-gate Si nanosheet: gate | SiO₂ (t_ox = 1.5 nm) | Si (t_si = 5 nm) |
SiO₂ (1.5 nm) | gate. Both gates held at Vg. E_F = 0 fixed (source
reference); Vg swept (default −0.2 → +1.0 V).

Units: eV, nm, elementary charge q = 1. Room temperature T = 300 K unless
stated. ħ²/(2m₀) = 0.0380998 eV·nm².

## Schrödinger (effective mass, BenDaniel–Duke)

For each valley group v:

    −(ħ²/2) d/dz [ 1/m*_v(z) · dψ/dz ] + V_v(z) ψ = E ψ
    V_v(z) = E_c(z) − q·φ(z)

Uniform grid, N ≈ 200 points over 8 nm total. Hermitian finite-difference
form with interface masses m*_{j±1/2} (harmonic mean of adjacent cells):

    −(ħ²/2Δz²) [ (ψ_{j+1}−ψ_j)/m*_{j+1/2} − (ψ_j−ψ_{j−1})/m*_{j−1/2} ]
    + V_j ψ_j = E ψ_j

→ symmetric tridiagonal eigenproblem. Solver: Sturm-sequence bisection for
the lowest few eigenvalues + inverse iteration (Thomas) for eigenvectors.
Hand-written; no LAPACK/cuSOLVER (house rule; also what makes the kernel
GPU-portable).

### Valley groups, Si(001)

| group | valley deg. (g_v) | confinement m* (z) | DOS mass m_D | SiO₂ m* |
|---|---|---|---|---|
| Δ2 | 2 | 0.916 m₀ | 0.19 m₀ | 0.5 m₀ |
| Δ4 | 4 | 0.19 m₀ | √(0.916·0.19) = 0.417 m₀ | 0.5 m₀ |

Spin degeneracy g_s = 2 applied on top of g_v. Conduction-band offset
Si → SiO₂: ΔE_c = 3.1 eV (finite barriers — NOT infinite walls, so the
finite-barrier penetration shows in the wavefunctions).

## Charge (quantum)

2D step DOS + Fermi–Dirac integrates to the closed form per subband:

    N_iv = g_s g_v (m_D,v kT / 2πħ²) · ln(1 + exp((E_F − E_iv)/kT))
    n(z) = Σ_v Σ_i N_iv |ψ_iv(z)|²        (normalize ∫|ψ|²dz = 1)

## Charge (classical comparison)

Boltzmann statistics (stated approximation, standard textbook classical
Poisson):

    n_cl(z) = N_c · exp((E_F − E_c(z) + qφ(z)) / kT)
    N_c = 2 · (2π m_d kT / h²)^{3/2},  m_d = 0.328 m₀ (Si conductivity DOS mass)

## Poisson

Undoped channel (no dopant term — stated simplification):

    −d/dz [ ε(z) dφ/dz ] = ρ(z) = −q·n(z)
    ε_Si = 11.7 ε₀,  ε_SiO₂ = 3.9 ε₀
    Dirichlet: φ(left gate) = φ(right gate) = Vg

FD tridiagonal, Thomas solve. Same discretization grid as Schrödinger.

## Self-consistent loop

1. φ⁰ = Laplace solution (flat, = Vg).
2. Solve Schrödinger with V_v(z) = E_c(z) − qφ → subbands E_iv, ψ_iv.
3. n(z) from the quantum charge formula.
4. Poisson → φ_new. Under-relax: φ ← (1−α)φ + α·φ_new, α = 0.1.
5. Converged when max|Δφ| < 1e−6 V; cap 500 iterations; report
   non-convergence explicitly per Vg point.

Classical curve = same loop without step 2–3, using n_cl.

## Outputs

Per Vg point: Ns (sheet density, cm⁻²) quantum + classical; subband
energies E_iv; SCF iteration count. Snapshot frames (φ(z), n(z), classical
n(z)) at a coarse Vg subset for the animation. Quantum capacitance
Cq = q·dNs/dVg computed numerically in post (plot script).

## Sanity anchors (kill criteria)

- Infinite 5 nm Si well, m* = 0.916: E_n = n²·16.4 meV (ħ²π²/2m*L²).
  Finite-barrier solver must approach these from below as ΔE_c → large.
- Quantum ground state peaks AWAY from the oxide interface; classical
  Boltzmann charge piles AT the interface. The animation's punchline.
- Quantum Ns < classical Ns at fixed Vg; gap widens with Vg and with
  thinner t_si.

## GPU batching (HIP API)

One threadblock per Vg point; the whole SCF loop lives in the block.
Thomas solves serial per block (N≈200 — negligible); Sturm bisection
parallelized across threads (one eigenvalue bracket per thread); inverse
iteration one thread per subband. FP64 everywhere (see the
gpu-bundle-adjustment precision study: reduced precision in measurement/
state space relocates the answer — not repeated here).

## Not built (name-drop only)

sp³d⁵s* tight-binding, non-parabolicity, strain, transport (ballistic
Landauer Id), 2D/3D Schrödinger for gate-all-around corners. The sentence
for the demo: "effective-mass holds at 5nm and breaks at 2nm — that
boundary is the interesting part."
