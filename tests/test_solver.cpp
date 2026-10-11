// test_solver.cpp — ctest-registered kill-criteria tests for nanosheet-sp.
//   (a) Thomas tridiagonal solve vs. known solutions (raw + Poisson-shaped);
//   (b) infinite-well kill criterion: tall-barrier 5nm Si well (m* = 0.916)
//       eigenvalues vs. E_n = n^2 * 16.42 meV within discretization error;
//   (c) charge normalization int|psi|^2 dz = 1 and eigenpair residual;
//   (d) physics ordering at Vg = 0.6 V: Ns_quantum < Ns_classical, quantum
//       ground |psi|^2 peaks away from the oxide interfaces, classical n(z)
//       peaks at them;
//   (e) SCF converges over the whole default sweep; any non-converged Vg
//       fails loudly; Boltzmann exponent clamp never binds at convergence.
// Exit code 0 iff all checks pass.

#include "sp_core.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int failures = 0;
#define CHECK(cond, ...)                                   \
    do {                                                   \
        printf("%s", cond ? "[PASS] " : "[FAIL] ");        \
        printf(__VA_ARGS__);                               \
        printf("\n");                                      \
        if (!(cond)) ++failures;                           \
    } while (0)

int main() {
    // (a) Thomas vs. manufactured solutions --------------------------------
    {
        const int n = 64;
        std::vector<double> d(n), e(n - 1), b(n), xe(n), x(n), cp(n), dp(n);
        for (int i = 0; i < n; ++i) {
            d[i] = 4.0 + 0.01 * i;
            if (i < n - 1) e[i] = -1.0 - 0.001 * i;
            xe[i] = sin(0.37 * (i + 1)) + 0.3 * cos(0.11 * i);
        }
        for (int i = 0; i < n; ++i) {
            b[i] = d[i] * xe[i];
            if (i > 0) b[i] += e[i - 1] * xe[i - 1];
            if (i < n - 1) b[i] += e[i] * xe[i + 1];
        }
        sp::thomas_sym(d.data(), e.data(), b.data(), n, cp.data(), dp.data(), x.data());
        double err = 0.0;
        for (int i = 0; i < n; ++i) err = fmax(err, fabs(x[i] - xe[i]));
        CHECK(err < 1e-10, "(a) Thomas vs manufactured x: max err = %.3e", err);

        // Poisson-shaped: phi* = Vg + 0.3 sin(pi z/L) through the eps profile
        sp::Grid g = sp::make_grid(161, 8.0, 1.5, 5.0, 3.1, 300.0, 0.1, 500,
                                   1e-6, 4.0, 0.25, 4);
        const int N = g.P.N, M = N - 2;
        double vg = 0.7;
        std::vector<double> phis(N), nm3(N, 0.0), phout(N);
        std::vector<double> dd(M), ee(M), bb(M), cc(M), pp(M);
        double idz2 = 1.0 / (g.P.dz * g.P.dz);
        for (int j = 0; j < N; ++j)
            phis[j] = vg + 0.3 * sin(sp::PI * g.z[j] / 8.0);
        // discrete equation: -eps_h[j-1] p[j-1] + (sum) p[j] - eps_h[j] p[j+1]
        //   = -QE0 n_j dz^2  =>  n_j = -lhs/(QE0 dz^2)
        for (int i = 0; i < M; ++i) {
            int j = i + 1;
            double lhs = -g.eps_h[j - 1] * phis[j - 1]
                       + (g.eps_h[j - 1] + g.eps_h[j]) * phis[j]
                       - g.eps_h[j] * phis[j + 1];
            nm3[j] = -lhs * idz2 / sp::QE0;
        }
        sp::poisson_solve(g.P, g.eps_h.data(), nm3.data(), vg, dd.data(),
                          ee.data(), bb.data(), cc.data(), pp.data(), phout.data());
        double perr = 0.0;
        for (int j = 0; j < N; ++j) perr = fmax(perr, fabs(phout[j] - phis[j]));
        CHECK(perr < 1e-9, "(a) Poisson vs manufactured phi: max err = %.3e V", perr);
    }

    // (b) infinite-well kill criterion --------------------------------------
    // Tall-barrier limit = Dirichlet exactly at the interfaces: a uniform
    // all-Si 5 nm grid (m* = 0.916, V = 0, psi = 0 at both ends). Two checks:
    //   * vs. the exact FD eigenvalues 2*(hbar^2/2m*dz^2)*(1-cos(n pi dz/L))
    //     to 1e-9 relative -- this is the raw Sturm-bisection kill check;
    //   * vs. the continuum n^2*16.42 meV within the FD dispersion error
    //     (n pi dz/L)^2/12, <= 0.14 % for n = 4 at dz = 0.05 nm; tolerance
    //     0.3 % has >2x margin. (A finite-barrier grid with an unresolved
    //     decay length would add a spurious ~0.7*dz/side effective widening;
    //     the Dirichlet form tests the eigensolver, not the interface.)
    // Plus a monotone-approach sanity print: physical finite barriers must
    // approach the infinite-well values from below as dEc grows.
    {
        sp::Grid g = sp::make_grid(/*N=*/101, /*L=*/5.0, /*tox=*/0.0,
                                   /*tsi=*/5.0, /*dEc=*/0.0, 300.0, 0.1, 500,
                                   1e-6, 4.0, 0.25, 4);
        const int M = g.P.N - 2; // 99 interior points, dz = 0.05 nm
        std::vector<double> phi(g.P.N, 0.0), d(M), e(M);
        sp::build_hamiltonian(g.P, g.Ec.data(), g.minv_h[0].data(), phi.data(),
                              d.data(), e.data());
        double E1inf = sp::H2_2M0 * sp::PI * sp::PI / (0.916 * 25.0); // eV
        double kin_si = sp::H2_2M0 / (0.916 * g.P.dz * g.P.dz);
        double worst_cont = 0.0, worst_fd = 0.0;
        printf("       infinite-well check (5 nm Si, m* = 0.916, Dirichlet at"
               " interfaces, dz = 0.05 nm):\n");
        for (int k = 0; k < 4; ++k) {
            double lam = sp::find_eigen(d.data(), e.data(), M, k);
            double exact = (k + 1) * (k + 1) * E1inf;
            double fd = 2.0 * kin_si * (1.0 - cos((k + 1) * sp::PI * g.P.dz / 5.0));
            worst_cont = fmax(worst_cont, fabs(lam - exact) / exact);
            worst_fd = fmax(worst_fd, fabs(lam - fd) / fd);
            printf("         E%d = %9.3f meV  (continuum %9.3f, FD-exact %9.3f,"
                   " err vs continuum %.3f%%)\n", k + 1, lam * 1e3, exact * 1e3,
                   fd * 1e3, fabs(lam - exact) / exact * 100.0);
        }
        CHECK(worst_fd < 1e-9, "(b) E_n vs exact FD eigenvalues: worst rel err = %.2e",
              worst_fd);
        CHECK(worst_cont < 3e-3,
              "(b) E_n vs n^2*16.42 meV within FD discretization error:"
              " worst = %.3f%% (tol 0.3%%, FD bound 0.14%%)", worst_cont * 100.0);

        // finite barriers approach from below as dEc grows (3.1 -> 30 eV)
        sp::Grid g1 = sp::make_grid(161, 8.0, 1.5, 5.0, 3.1, 300.0, 0.1, 500,
                                    1e-6, 4.0, 0.25, 4);
        sp::Grid g2 = sp::make_grid(161, 8.0, 1.5, 5.0, 30.0, 300.0, 0.1, 500,
                                    1e-6, 4.0, 0.25, 4);
        std::vector<double> phif(161, 0.0), d1(159), e1(159), d2(159), e2(159);
        sp::build_hamiltonian(g1.P, g1.Ec.data(), g1.minv_h[0].data(),
                              phif.data(), d1.data(), e1.data());
        sp::build_hamiltonian(g2.P, g2.Ec.data(), g2.minv_h[0].data(),
                              phif.data(), d2.data(), e2.data());
        double eb31 = sp::find_eigen(d1.data(), e1.data(), 159, 0);
        double eb30 = sp::find_eigen(d2.data(), e2.data(), 159, 0);
        printf("       finite-barrier approach: E1(dEc=3.1) = %.3f meV <"
               " E1(dEc=30) = %.3f meV < E1(inf) = %.3f meV\n",
               eb31 * 1e3, eb30 * 1e3, E1inf * 1e3);
        CHECK(eb31 < eb30 && eb30 < 2.0 * kin_si * (1.0 - cos(sp::PI * 0.05 / 5.0)),
              "(b) finite-barrier E1 approaches infinite well from below");

        // (c) charge normalization + eigenpair residual
        std::vector<double> psi(M), wb(M), cp(M), dp(M);
        double lam = sp::find_eigen(d.data(), e.data(), M, 0);
        sp::inverse_iter(d.data(), e.data(), M, lam, g.P.dz, psi.data(),
                         wb.data(), cp.data(), dp.data());
        double s2 = 0.0;
        for (int i = 0; i < M; ++i) s2 += psi[i] * psi[i];
        s2 *= g.P.dz;
        CHECK(fabs(s2 - 1.0) < 1e-12, "(c) int|psi|^2 dz = %.15f (err %.2e)",
              s2, fabs(s2 - 1.0));
        double res = 0.0;
        for (int i = 0; i < M; ++i) {
            double r = (d[i] - lam) * psi[i];
            if (i > 0) r += e[i - 1] * psi[i - 1];
            if (i < M - 1) r += e[i] * psi[i + 1];
            res = fmax(res, fabs(r));
        }
        CHECK(res < 1e-7, "(c) eigenpair residual max|(T-E)psi| = %.3e", res);
    }

    // (d) physics ordering at Vg = 0.6 V ------------------------------------
    {
        sp::Grid g = sp::make_grid(161, 8.0, 1.5, 5.0, 3.1, 300.0, 0.1, 500,
                                   1e-6, 4.0, 0.25, 4);
        const int N = g.P.N;
        std::vector<double> wbuf(sp::work_doubles(N));
        sp::Work W = sp::make_work(wbuf.data(), N);
        std::vector<double> phi(N), n(N), phc(N), nc(N), E(8), psi(8 * N);
        int itq, itc; double maq, mac;
        int cq = sp::scf_solve(g.P, g.Ec.data(), g.eps_h.data(),
                               g.minv_h[0].data(), g.minv_h[1].data(),
                               sp::MODEL_QUANTUM, 0.6, phi.data(), n.data(),
                               E.data(), psi.data(), W, &itq, &maq);
        int cc = sp::scf_solve(g.P, g.Ec.data(), g.eps_h.data(),
                               g.minv_h[0].data(), g.minv_h[1].data(),
                               sp::MODEL_CLASSICAL, 0.6, phc.data(), nc.data(),
                               nullptr, nullptr, W, &itc, &mac);
        double nsq = sp::sheet_density_cm2(g.P, n.data());
        double nsc = sp::sheet_density_cm2(g.P, nc.data());
        printf("       Vg = 0.6 V: Ns_quantum = %.4e cm^-2 (%d it),"
               " Ns_classical = %.4e cm^-2 (%d it)\n", nsq, itq, nsc, itc);
        CHECK(cq && cc, "(d) SCF converged at Vg = 0.6 V (q=%d cl=%d)", cq, cc);
        CHECK(nsq < nsc, "(d) Ns_quantum (%.4e) < Ns_classical (%.4e)", nsq, nsc);

        int ipk = 1;
        for (int j = 1; j < N - 1; ++j)
            if (psi[0 * N + j] * psi[0 * N + j] >
                psi[0 * N + ipk] * psi[0 * N + ipk]) ipk = j;
        double zpk = g.z[ipk];
        double dq = fmin(fabs(zpk - 1.5), fabs(zpk - 6.5));
        CHECK(dq > 0.75, "(d) quantum ground |psi|^2 peaks at z = %.2f nm,"
              " %.2f nm from nearest oxide interface (> 0.75 nm)", zpk, dq);

        int icl = -1;
        double mx = -1.0;
        for (int j = 0; j < N; ++j)
            if (g.z[j] >= 1.5 - 1e-12 && g.z[j] <= 6.5 + 1e-12 && nc[j] > mx) {
                mx = nc[j]; icl = j;
            }
        double dcl = fmin(fabs(g.z[icl] - 1.5), fabs(g.z[icl] - 6.5));
        CHECK(dcl < 0.15, "(d) classical n(z) peaks at z = %.2f nm,"
              " %.3f nm from the oxide interface (< 0.15 nm)", g.z[icl], dcl);
    }

    // (e) full default sweep convergence ------------------------------------
    {
        sp::Grid g = sp::make_grid(161, 8.0, 1.5, 5.0, 3.1, 300.0, 0.1, 500,
                                   1e-6, 4.0, 0.25, 4);
        const int N = g.P.N;
        std::vector<double> wbuf(sp::work_doubles(N));
        sp::Work W = sp::make_work(wbuf.data(), N);
        std::vector<double> phi(N), n(N), phc(N), nc(N), E(8), psi(8 * N);
        int nfail = 0, max_it = 0;
        double max_arg = -1e30, min_ratio = 1e30;
        for (int i = 0; i <= 60; ++i) {
            double vg = -0.2 + 0.02 * i;
            int itq, itc; double maq, mac;
            int cq = sp::scf_solve(g.P, g.Ec.data(), g.eps_h.data(),
                                   g.minv_h[0].data(), g.minv_h[1].data(),
                                   sp::MODEL_QUANTUM, vg, phi.data(), n.data(),
                                   E.data(), psi.data(), W, &itq, &maq);
            int cc = sp::scf_solve(g.P, g.Ec.data(), g.eps_h.data(),
                                   g.minv_h[0].data(), g.minv_h[1].data(),
                                   sp::MODEL_CLASSICAL, vg, phc.data(), nc.data(),
                                   nullptr, nullptr, W, &itc, &mac);
            if (!cq || !cc) {
                ++nfail;
                printf("       NON-CONVERGED: Vg = %.2f (q=%d/%d it, cl=%d/%d it)\n",
                       vg, cq, itq, cc, itc);
            }
            max_it = std::max(max_it, std::max(itq, itc));
            max_arg = fmax(max_arg, mac);
            double nsq = sp::sheet_density_cm2(g.P, n.data());
            double nsc = sp::sheet_density_cm2(g.P, nc.data());
            min_ratio = fmin(min_ratio, nsc > 0.0 ? nsq / nsc : 1.0);
        }
        printf("       sweep: 61 points, max SCF iterations = %d,"
               " max classical Boltzmann exponent at convergence = %.3f"
               " (clamp %.1f), min Ns_q/Ns_cl = %.4f\n",
               max_it, max_arg, g.P.exp_clamp, min_ratio);
        CHECK(nfail == 0, "(e) SCF converged on all 61 default sweep points"
              " (%d failures)", nfail);
        CHECK(max_arg < g.P.exp_clamp - 0.5,
              "(e) Boltzmann exponent clamp never binds at convergence"
              " (max arg %.3f vs clamp %.1f)", max_arg, g.P.exp_clamp);
        CHECK(min_ratio < 1.0, "(e) Ns_quantum < Ns_classical at every swept Vg"
              " (min ratio %.4f)", min_ratio);
    }

    printf("%s (%d failure%s)\n", failures ? "TESTS FAILED" : "ALL TESTS PASSED",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
