#pragma once

// sp_core.h — shared solver core for nanosheet-sp.
// Physics per docs/PHYSICS.md. The quantum 2D DOS uses
// g_s*g_v*m_D*kT/(2*pi*hbar^2) with explicit spin degeneracy g_s=2. The
// classical N_c uses m_d = 6^(2/3)*(m_l*m_t^2)^(1/3) = 1.060 m0, giving
// N_c = 2.74e19 cm^-3. SCF updates are clamped to |dphi| <= 0.25 V; the
// clamp binds only during transients, not at the fixed point.
//
// All solver routines are pointer-based POD functions marked SP_HD so the
// same source compiles for host (sp_reference, test_solver) and for device
// (sp_batched.cu). Sign conventions: E_c(Si) = 0 eV reference, oxide at
// +dEc; V(z) = E_c(z) - q*phi(z) with q = 1 eV/V; electron charge density
// rho = -q*n; Dirichlet phi(gate) = Vg at both outer boundaries.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#if defined(__CUDACC__) || defined(__HIP__)
#define SP_HD __host__ __device__
#else
#define SP_HD
#endif

namespace sp {

constexpr double H2_2M0 = 0.0380998;       // hbar^2/(2 m0) [eV nm^2]
constexpr double KB_EV  = 8.617333262e-5;  // Boltzmann constant [eV/K]
constexpr double QE0    = 18.0951279;      // q/eps0 [V nm]
constexpr double PI     = 3.14159265358979323846;

struct Params {
    int    N;           // grid points including the two Dirichlet boundaries
    double dz;          // grid spacing [nm]
    double kT;          // thermal energy [eV]
    double EF;          // Fermi level [eV] (source reference, 0)
    double alpha;       // SCF under-relaxation
    int    max_iter;    // SCF iteration cap
    double tol;         // SCF convergence: max|phi_new - phi| [V]
    double exp_clamp;   // Boltzmann exponent clamp (classical transient guard)
    double dphi_clamp;  // Gummel per-iteration potential step clamp [V]
    double C2D;         // m0*kT/(2*pi*hbar^2) [cm^-2]
    double NC;          // classical effective DOS [nm^-3]
    int    nsub;        // subbands per valley group
};

enum Model { MODEL_QUANTUM = 0, MODEL_CLASSICAL = 1 };

// Valley groups, Si(001): {valley degeneracy, confinement mass, DOS mass}.
// Spin degeneracy g_s = 2 is applied explicitly in the charge formula.
constexpr double VALLEY_GV[2]    = {2.0, 4.0};
constexpr double VALLEY_MCONF[2] = {0.916, 0.19};
constexpr double VALLEY_MDOS[2]  = {0.19, 0.4171810868987377}; // sqrt(0.916*0.19)
constexpr double M_OX = 0.5;    // SiO2 mass [m0]
constexpr double EPS_SI = 11.7, EPS_OX = 3.9;

// ln(1 + e^x), overflow-safe.
SP_HD inline double fermilog(double x) {
    return (x > 40.0) ? x : log1p(exp(x));
}

// Count of eigenvalues of the symmetric tridiagonal (d[0..M-1], e[0..M-2])
// strictly below lam, via the Sturm sequence of leading principal minors.
SP_HD inline int sturm_count(const double* d, const double* e, int M, double lam) {
    double q = d[0] - lam;
    int cnt = (q < 0.0) ? 1 : 0;
    for (int i = 1; i < M; ++i) {
        double qq = q;
        if (fabs(qq) < 1e-290) qq = (qq < 0.0) ? -1e-290 : 1e-290;
        q = (d[i] - lam) - e[i-1] * e[i-1] / qq;
        if (q < 0.0) ++cnt;
    }
    return cnt;
}

// k-th smallest eigenvalue (0-based) by Sturm bisection from Gershgorin
// bounds; fixed 80 iterations -> deterministic, ~1e-15 relative accuracy.
SP_HD inline double find_eigen(const double* d, const double* e, int M, int k) {
    double lo = d[0], hi = d[0];
    for (int i = 0; i < M; ++i) {
        double r = 0.0;
        if (i > 0)     r += fabs(e[i-1]);
        if (i < M - 1) r += fabs(e[i]);
        lo = fmin(lo, d[i] - r);
        hi = fmax(hi, d[i] + r);
    }
    lo -= 1.0; hi += 1.0;
    for (int it = 0; it < 80; ++it) {
        double mid = 0.5 * (lo + hi);
        if (sturm_count(d, e, M, mid) >= k + 1) hi = mid; else lo = mid;
    }
    return 0.5 * (lo + hi);
}

// Thomas algorithm for a symmetric tridiagonal: diagonal d[0..n-1],
// off-diagonal e[0..n-2] (e[i] couples rows i, i+1). cp/dp are scratch.
// x may alias b.
SP_HD inline void thomas_sym(const double* d, const double* e, const double* b,
                             int n, double* cp, double* dp, double* x) {
    cp[0] = e[0] / d[0];
    dp[0] = b[0] / d[0];
    for (int i = 1; i < n; ++i) {
        double m = d[i] - e[i-1] * cp[i-1];
        if (i < n - 1) cp[i] = e[i] / m;
        dp[i] = (b[i] - e[i-1] * dp[i-1]) / m;
    }
    x[n-1] = dp[n-1];
    for (int i = n - 2; i >= 0; --i) x[i] = dp[i] - cp[i] * x[i+1];
}

// Same, with a diagonal shift: solves (T - sig*I) x = b without materializing
// the shifted matrix (used by inverse iteration).
SP_HD inline void thomas_sym_shift(const double* d, const double* e, double sig,
                                   const double* b, int n,
                                   double* cp, double* dp, double* x) {
    cp[0] = e[0] / (d[0] - sig);
    dp[0] = b[0] / (d[0] - sig);
    for (int i = 1; i < n; ++i) {
        double m = (d[i] - sig) - e[i-1] * cp[i-1];
        if (i < n - 1) cp[i] = e[i] / m;
        dp[i] = (b[i] - e[i-1] * dp[i-1]) / m;
    }
    x[n-1] = dp[n-1];
    for (int i = n - 2; i >= 0; --i) x[i] = dp[i] - cp[i] * x[i+1];
}

// BenDaniel-Duke Hermitian finite-difference Hamiltonian on the interior
// points j = 1..N-2 (psi = 0 at the gates). minv_h[j] is the inverse
// interface mass on the bond j <-> j+1 (arithmetic mean of the inverse
// point masses = harmonic mean of the masses). V_j = E_c[j] - phi[j].
SP_HD inline void build_hamiltonian(const Params& P, const double* Ec,
                                    const double* minv_h, const double* phi,
                                    double* d, double* e) {
    int M = P.N - 2;
    double kin = H2_2M0 / (P.dz * P.dz);
    for (int i = 0; i < M; ++i) {
        int j = i + 1;
        d[i] = kin * (minv_h[j-1] + minv_h[j]) + Ec[j] - phi[j];
        if (i < M - 1) e[i] = -kin * minv_h[j];
    }
}

// Inverse iteration for the eigenvector of (d, e) belonging to lam.
// psi[0..M-1] is overwritten; on return sum(psi^2)*dz = 1 and the component
// of largest magnitude is positive. Scratch: wb, cp, dp (M each).
SP_HD inline void inverse_iter(const double* d, const double* e, int M, double lam,
                               double dz, double* psi,
                               double* wb, double* cp, double* dp) {
    double sig = lam + 1e-11 * (1.0 + fabs(lam));
    unsigned long long s = 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < M; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        psi[i] = (double)(s >> 11) * (1.0 / 9007199254740992.0) - 0.5;
    }
    for (int it = 0; it < 4; ++it) {
        thomas_sym_shift(d, e, sig, psi, M, cp, dp, wb);
        double mx = 0.0;
        for (int i = 0; i < M; ++i) mx = fmax(mx, fabs(wb[i]));
        double inv = 1.0 / mx;
        for (int i = 0; i < M; ++i) psi[i] = wb[i] * inv;
    }
    double ss = 0.0;
    for (int i = 0; i < M; ++i) ss += psi[i] * psi[i];
    ss = sqrt(ss * dz);
    int imx = 0;
    double mx = 0.0;
    for (int i = 0; i < M; ++i)
        if (fabs(psi[i]) > mx) { mx = fabs(psi[i]); imx = i; }
    double inv = ((psi[imx] < 0.0) ? -1.0 : 1.0) / ss;
    for (int i = 0; i < M; ++i) psi[i] *= inv;
}

// Poisson: -d/dz[eps_r dz/dz phi] = -QE0*n (n in nm^-3), phi(0)=phi(N-1)=vg.
// Solves on interior points; phi_new[0..N-1] filled including boundaries.
// Scratch: d, e, b, cp, dp (M each).
SP_HD inline void poisson_solve(const Params& P, const double* eps_h,
                                const double* n_nm3, double vg,
                                double* d, double* e, double* b,
                                double* cp, double* dp, double* phi_new) {
    int M = P.N - 2;
    double idz2 = 1.0 / (P.dz * P.dz);
    for (int i = 0; i < M; ++i) {
        int j = i + 1;
        d[i] = (eps_h[j-1] + eps_h[j]) * idz2;
        if (i < M - 1) e[i] = -eps_h[j] * idz2;
        b[i] = -QE0 * n_nm3[j];
    }
    b[0]   += eps_h[0]     * idz2 * vg;
    b[M-1] += eps_h[P.N-2] * idz2 * vg;
    thomas_sym(d, e, b, M, cp, dp, &phi_new[1]);
    phi_new[0] = vg;
    phi_new[P.N-1] = vg;
}

// Classical Boltzmann density at one grid point [nm^-3]. The exponent clamp
// only binds during SCF transients (converged max exponent <= ~1 at Vg=1 V);
// tests assert it never binds at convergence.
SP_HD inline double classical_n(const Params& P, double Ec_j, double phi_j) {
    double arg = (P.EF - Ec_j + phi_j) / P.kT;
    if (arg > P.exp_clamp) arg = P.exp_clamp;
    if (arg < -250.0) arg = -250.0;
    return P.NC * exp(arg);
}

// Quantum sheet density of one subband [cm^-2], standard 2D DOS with
// explicit spin: N = g_s*g_v*(m_D kT / 2*pi*hbar^2) * ln(1+exp((EF-E)/kT)).
SP_HD inline double subband_charge(const Params& P, double gv, double mdos, double E) {
    return 2.0 * gv * P.C2D * mdos * fermilog((P.EF - E) / P.kT);
}

// Workspace for the serial SCF (host reference and tests). Device code
// orchestrates the same primitives out of shared memory instead.
struct Work {
    double* dH[2];   // M each
    double* eH[2];   // M each (last entry unused)
    double* psi;     // 8*N
    double* phi_new; // N
    double* b;       // M
    double* cp;      // M
    double* dp;      // M
    double* wb;      // M
};

inline int work_doubles(int N, int nsub2 = 8) {
    int M = N - 2;
    return 8 * M + nsub2 * N + N;
}

inline Work make_work(double* buf, int N, int nsub2 = 8) {
    int M = N - 2;
    Work W;
    double* p = buf;
    W.dH[0] = p; p += M;
    W.dH[1] = p; p += M;
    W.eH[0] = p; p += M;
    W.eH[1] = p; p += M;
    W.psi = p; p += nsub2 * N;
    W.phi_new = p; p += N;
    W.b = p;  p += M;
    W.cp = p; p += M;
    W.dp = p; p += M;
    W.wb = p;
    return W;
}

// One full self-consistent Gummel loop for a single Vg and model.
//   phi:     in/out, N doubles; overwritten, starts from Laplace (phi = Vg).
//   n_out:   out, N doubles, converged density [nm^-3].
//   E_out:   out, 8 doubles (4 subbands x 2 valley groups) or nullptr.
//   psi_out: out, 8*N normalized eigenfunctions (row s = subband s) or nullptr.
// Returns 1 if converged within max_iter. *iters_out = iterations used,
// *max_arg_out = largest classical Boltzmann exponent seen at the final
// (converged) state — audit that exp_clamp never binds when it matters.
// Host-only: the GPU kernel (sp_batched.cu) re-orchestrates the same SP_HD
// primitives with block parallelism instead of calling this serial loop.
inline int scf_solve(const Params& P, const double* Ec, const double* eps_h,
                           const double* minv_h0, const double* minv_h1,
                           int model, double vg,
                           double* phi, double* n_out,
                           double* E_out, double* psi_out,
                           Work& W, int* iters_out, double* max_arg_out) {
    int N = P.N, M = N - 2;
    for (int j = 0; j < N; ++j) phi[j] = vg;

    double E[8], Nsub[8];
    for (int s = 0; s < 8; ++s) { E[s] = 0.0; Nsub[s] = 0.0; }
    const double* minv[2] = {minv_h0, minv_h1};

    int converged = 0, it = 0;
    double alpha_eff = P.alpha, prev_delta = 0.0;
    int prev_sign = 0, prev_node = -1;
    for (it = 1; it <= P.max_iter; ++it) {
        if (model == MODEL_QUANTUM) {
            for (int g = 0; g < 2; ++g) {
                build_hamiltonian(P, Ec, minv[g], phi, W.dH[g], W.eH[g]);
                for (int k = 0; k < P.nsub; ++k) {
                    int s = g * P.nsub + k;
                    E[s] = find_eigen(W.dH[g], W.eH[g], M, k);
                    double* row = W.psi + s * N;
                    inverse_iter(W.dH[g], W.eH[g], M, E[s], P.dz, row + 1,
                                 W.b, W.cp, W.dp);
                    row[0] = 0.0; row[N-1] = 0.0;
                    Nsub[s] = subband_charge(P, VALLEY_GV[g], VALLEY_MDOS[g], E[s]);
                }
            }
            for (int j = 0; j < N; ++j) {
                double acc = 0.0;
                if (j > 0 && j < N - 1)
                    for (int s = 0; s < 2 * P.nsub; ++s) {
                        double pv = W.psi[s * N + j];
                        acc += Nsub[s] * pv * pv;
                    }
                n_out[j] = acc * 1e-14; // cm^-2 * nm^-1 -> nm^-3
            }
        } else {
            n_out[0] = 0.0; n_out[N-1] = 0.0;
            for (int j = 1; j < N - 1; ++j)
                n_out[j] = classical_n(P, Ec[j], phi[j]);
        }

        poisson_solve(P, eps_h, n_out, vg, W.dH[0], W.eH[0], W.b, W.cp, W.dp,
                      W.phi_new);

        double delta = 0.0;
        int node = 1;
        for (int j = 1; j < N - 1; ++j) {
            double a = fabs(W.phi_new[j] - phi[j]);
            if (a > delta) { delta = a; node = j; }
        }

        if (delta < P.tol) {
            for (int j = 1; j < N - 1; ++j) phi[j] = W.phi_new[j];
            converged = 1;
        } else {
            // Adaptive Gummel damping: halve the relaxation
            // factor on residual growth or on oscillation (sign flip of the
            // update at the residual node); creep back toward alpha while the
            // contraction is monotone. The fixed point is unchanged.
            int sgn = (W.phi_new[node] > phi[node]) ? 1 : -1;
            if (it > 1) {
                bool osc = (node == prev_node) && (sgn != prev_sign);
                if (osc || delta > prev_delta)
                    alpha_eff = fmax(alpha_eff * 0.5, 1e-3);
                else
                    alpha_eff = fmin(alpha_eff * 1.1, P.alpha);
            }
            prev_delta = delta; prev_sign = sgn; prev_node = node;
            for (int j = 1; j < N - 1; ++j) {
                double dphi = W.phi_new[j] - phi[j];
                if (dphi >  P.dphi_clamp) dphi =  P.dphi_clamp;
                if (dphi < -P.dphi_clamp) dphi = -P.dphi_clamp;
                phi[j] += alpha_eff * dphi;
            }
        }
        if (converged) break;
    }

    if (model == MODEL_CLASSICAL) {
        double ma = -1e30;
        for (int j = 1; j < N - 1; ++j)
            ma = fmax(ma, (P.EF - Ec[j] + phi[j]) / P.kT);
        if (max_arg_out) *max_arg_out = ma;
    } else if (max_arg_out) {
        *max_arg_out = 0.0;
    }

    if (E_out) for (int s = 0; s < 8; ++s) E_out[s] = E[s];
    if (psi_out)
        for (int s = 0; s < 8; ++s)
            for (int j = 0; j < N; ++j)
                psi_out[s * N + j] = W.psi[s * N + j];
    *iters_out = it;
    return converged;
}

// ---------------------------------------------------------------------------
// Host-only helpers (grid setup, frame writing). Not compiled for device.
// ---------------------------------------------------------------------------

struct Grid {
    Params P;
    std::vector<double> z, Ec, eps_h, minv_h[2];
    double dEc;
};

// Device stack: gate | SiO2 tox | Si tsi | SiO2 tox | gate, total L = 2*tox+tsi.
inline Grid make_grid(int N, double L, double tox, double tsi, double dEc,
                      double T, double alpha, int max_iter, double tol,
                      double exp_clamp, double dphi_clamp, int nsub) {
    Grid g;
    g.dEc = dEc;
    g.z.assign(N, 0.0);
    g.Ec.assign(N, 0.0);
    g.eps_h.assign(N - 1, 0.0);
    g.minv_h[0].assign(N - 1, 0.0);
    g.minv_h[1].assign(N - 1, 0.0);
    double dz = L / (N - 1);

    std::vector<double> eps(N), minv_pt[2];
    minv_pt[0].assign(N, 0.0);
    minv_pt[1].assign(N, 0.0);
    for (int j = 0; j < N; ++j) {
        double z = j * dz;
        g.z[j] = z;
        bool si = (z >= tox - 1e-12) && (z <= tox + tsi + 1e-12);
        eps[j] = si ? EPS_SI : EPS_OX;
        g.Ec[j] = si ? 0.0 : dEc;
        for (int v = 0; v < 2; ++v)
            minv_pt[v][j] = 1.0 / (si ? VALLEY_MCONF[v] : M_OX);
    }
    for (int j = 0; j < N - 1; ++j) {
        g.eps_h[j] = 0.5 * (eps[j] + eps[j+1]); // arithmetic mean at interfaces
        for (int v = 0; v < 2; ++v)
            g.minv_h[v][j] = 0.5 * (minv_pt[v][j] + minv_pt[v][j+1]);
    }

    double kT = KB_EV * T;
    Params& P = g.P;
    P.N = N; P.dz = dz; P.kT = kT; P.EF = 0.0;
    P.alpha = alpha; P.max_iter = max_iter; P.tol = tol;
    P.exp_clamp = exp_clamp; P.dphi_clamp = dphi_clamp; P.nsub = nsub;
    P.C2D = kT / (4.0 * PI * H2_2M0) * 1e14; // m0*kT/(2*pi*hbar^2) [cm^-2]
    // Classical DOS mass with the 6-fold valley degeneracy folded in:
    // m_d = g_v^(2/3) * (m_l m_t^2)^(1/3); gives N_c = 2.74e19 cm^-3.
    double m_d = pow(6.0, 2.0 / 3.0) * cbrt(0.916 * 0.19 * 0.19);
    P.NC = 2.0 * pow(m_d * kT / (4.0 * PI * H2_2M0), 1.5); // [nm^-3]
    return g;
}

inline double sheet_density_cm2(const Params& P, const double* n_nm3) {
    double s = 0.0;
    for (int j = 0; j < P.N; ++j) s += n_nm3[j];
    return s * P.dz * 1e14;
}

// Frame file: comment header carries Vg and subband energies, then columns
// z phi_q phi_cl Ec-phi_q n_q n_cl psi2_D2_1 psi2_D4_1.
inline void write_frame(const std::string& path, double vg, const double* E,
                        int nsub, const Grid& g,
                        const double* phi_q, const double* phi_cl,
                        const double* n_q_cm3, const double* n_cl_cm3,
                        const double* psi_d2_1, const double* psi_d4_1) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return;
    fprintf(f, "# Vg = %.6f\n", vg);
    fprintf(f, "# E_D2_eV =");
    for (int k = 0; k < nsub; ++k) fprintf(f, " %.6f", E[k]);
    fprintf(f, "\n# E_D4_eV =");
    for (int k = 0; k < nsub; ++k) fprintf(f, " %.6f", E[nsub + k]);
    fprintf(f, "\n# columns: z_nm phi_q_V phi_cl_V Ec_minus_qphi_q_eV"
               " n_q_cm-3 n_cl_cm-3 psi2_D2_1_nm-1 psi2_D4_1_nm-1\n");
    for (int j = 0; j < g.P.N; ++j) {
        double p2a = psi_d2_1 ? psi_d2_1[j] * psi_d2_1[j] : 0.0;
        double p2b = psi_d4_1 ? psi_d4_1[j] * psi_d4_1[j] : 0.0;
        fprintf(f, "%.6f %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n",
                g.z[j], phi_q[j], phi_cl[j], g.Ec[j] - phi_q[j],
                n_q_cm3[j], n_cl_cm3[j], p2a, p2b);
    }
    fclose(f);
}

} // namespace sp
