# Nanosheet physics and discretization

The solver models the cross-section of a symmetric double-gate Si nanosheet:

    gate | SiO₂ (t_ox=1.5 nm) | Si (t_si=5 nm) | SiO₂ (1.5 nm) | gate

Both gates are held at Vg. The Fermi level is fixed at E_F=0, and the default
gate sweep is -0.2 to +1.0 V. Units are eV, nm, and elementary charge q=1.
Room temperature is T=300 K unless stated. The constant ħ²/(2m₀) is
0.0380998 eV·nm².

## Schrödinger equation

For each valley group v, the effective-mass BenDaniel–Duke equation is

    -(ħ²/2) d/dz [1/m*_v(z) · dψ/dz] + V_v(z) ψ = E ψ
    V_v(z) = E_c(z) - q·φ(z).

A uniform grid uses approximately 200 points over 8 nm. The Hermitian
finite-difference form uses harmonic-mean interface masses:

    -(ħ²/2Δz²) [(ψ_(j+1)-ψ_j)/m*_(j+1/2)
                -(ψ_j-ψ_(j-1))/m*_(j-1/2)] + V_j ψ_j = E ψ_j.

This gives a symmetric tridiagonal eigenproblem. Sturm-sequence bisection finds
the lowest eigenvalues, and inverse iteration with a Thomas solve computes the
eigenvectors. No LAPACK or cuSOLVER routines are used.

## Valley groups for Si(001)

| Group | Valley degeneracy g_v | Confinement mass (z) | DOS mass m_D | SiO₂ m* |
|---|---:|---:|---:|---:|
| Delta2 | 2 | 0.916 m₀ | 0.19 m₀ | 0.5 m₀ |
| Delta4 | 4 | 0.19 m₀ | sqrt(0.916·0.19)=0.417 m₀ | 0.5 m₀ |

Spin degeneracy g_s=2 is applied in addition to g_v. The Si-to-SiO₂
conduction-band offset is Delta E_c=3.1 eV. Finite barriers allow wavefunction
penetration into the oxide.

## Quantum charge

The 2D step density of states and Fermi–Dirac occupation give the closed-form
sheet density per subband:

    N_iv = g_s g_v (m_D,v kT / 2πħ²) ln(1+exp((E_F-E_iv)/kT))
    n(z) = sum_v sum_i N_iv |ψ_iv(z)|²,   integral |ψ|² dz = 1.

## Classical charge

The classical comparison uses Boltzmann statistics:

    n_cl(z) = N_c exp((E_F-E_c(z)+qφ(z))/kT)
    N_c = 2 (2π m_d kT/h²)^(3/2)
    m_d = 6^(2/3) (m_l m_t²)^(1/3) = 1.060 m₀.

The six-fold valley degeneracy is included in m_d and is not applied again in
N_c. The implemented value is N_c=2.74e19 cm⁻³.

## Poisson equation

The channel is undoped:

    -d/dz [ε(z) dφ/dz] = ρ(z) = -q·n(z)
    ε_Si = 11.7 ε₀,  ε_SiO₂ = 3.9 ε₀
    φ(left gate) = φ(right gate) = Vg.

The Poisson solve uses the same grid as the Schrödinger solve and a tridiagonal
Thomas algorithm.

## Self-consistent loop

1. Initialize φ with the Laplace solution, which is flat at Vg.
2. Solve Schrödinger with `V_v(z)=E_c(z)-qφ` to obtain subbands and wavefunctions.
3. Evaluate the quantum or classical charge density.
4. Solve Poisson for φ_new. Start with α_eff=α=0.1. Halve α_eff on residual
   growth or a sign flip at the residual node, down to 1e-3, and restore it
   gradually during monotone contraction. Clamp the raw potential update to
   +/-0.25 V before applying α_eff. These steps do not change the fixed point.
5. Stop when max|Δφ| < 1e-6 V, with a cap of 500 iterations. Report every
   non-converged gate-bias point.

The classical curve uses the same loop without the Schrödinger solve.

## Outputs

For each Vg, the solver writes quantum and classical sheet density, subband
energies, and the SCF iteration count. Snapshot frames contain φ(z), quantum and
classical n(z), and selected wavefunctions for the animation.

The plot script computes differential gate-charge capacitance `q·dNs/dVg`
numerically. This is not a standalone extraction of a series quantum-capacitance
element.

## Verification references

- An infinite 5 nm Si well with m*=0.916 has E_n=n²·16.4 meV. The finite-barrier
  solver approaches these values from below as Delta E_c increases.
- Quantum ground-state density peaks away from the oxide interface; classical
  Boltzmann charge peaks at the interface.
- Quantum sheet density is below classical sheet density at fixed Vg. The gap
  widens with Vg and with decreasing t_si.

## GPU batching

One thread block handles each Vg point, and the full SCF loop remains in the
block. Thomas solves run serially per block (N≈200); Sturm bisection is
parallelized across threads with one eigenvalue bracket per thread, and inverse
iteration uses one thread per subband. FP64 is used throughout the validated
numerical path.

## Model limits

The implementation does not include sp³d⁵s* tight binding, non-parabolicity,
strain, ballistic-Landauer transport, or 2D/3D gate-all-around corner effects.
The effective-mass approximation is appropriate for the present 5 nm model but
requires reassessment at smaller dimensions.
