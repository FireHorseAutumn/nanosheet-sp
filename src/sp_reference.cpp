// sp_reference.cpp — host reference Schrödinger–Poisson solver (plain C++,
// std-only). Runs the full SCF loop per Vg for both the quantum and the
// classical (Boltzmann) models, writes a per-Vg CSV and snapshot frames for
// a coarse Vg subset. Physics per docs/PHYSICS.md + 2026-10-11 errata.
//
// CLI: --vg-min --vg-max --vg-step --out results.csv --frames frames_dir
//      [--frame-every K]   (frames at every K-th sweep point, default 5)

#include "sp_core.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

int main(int argc, char** argv) {
    double vg_min = -0.2, vg_max = 1.0, vg_step = 0.02;
    std::string out = "results.csv", frames_dir;
    int frame_every = 5;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", name);
                exit(2);
            }
            return argv[++i];
        };
        if (!strcmp(argv[i], "--vg-min")) vg_min = atof(need("--vg-min"));
        else if (!strcmp(argv[i], "--vg-max")) vg_max = atof(need("--vg-max"));
        else if (!strcmp(argv[i], "--vg-step")) vg_step = atof(need("--vg-step"));
        else if (!strcmp(argv[i], "--out")) out = need("--out");
        else if (!strcmp(argv[i], "--frames")) frames_dir = need("--frames");
        else if (!strcmp(argv[i], "--frame-every")) frame_every = atoi(need("--frame-every"));
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }

    sp::Grid g = sp::make_grid(/*N=*/161, /*L=*/8.0, /*tox=*/1.5, /*tsi=*/5.0,
                               /*dEc=*/3.1, /*T=*/300.0, /*alpha=*/0.1,
                               /*max_iter=*/500, /*tol=*/1e-6,
                               /*exp_clamp=*/4.0, /*dphi_clamp=*/0.25,
                               /*nsub=*/4);
    const int N = g.P.N;

    int npts = (int)llround((vg_max - vg_min) / vg_step) + 1;
    if (npts < 1) { fprintf(stderr, "empty sweep\n"); return 2; }

    std::vector<double> wbuf(sp::work_doubles(N));
    sp::Work W = sp::make_work(wbuf.data(), N);
    std::vector<double> phi(N), n_nm3(N), phi_cl(N), n_cl_nm3(N), E(8), psi(8 * N);
    std::vector<double> n_q_cm3(N), n_cl_cm3(N);

    if (!frames_dir.empty()) {
        std::filesystem::create_directories(frames_dir);
        // index.csv maps frame file -> Vg for the animator
        FILE* idx = fopen((frames_dir + "/index.csv").c_str(), "w");
        fprintf(idx, "frame_file,Vg\n");
        fclose(idx);
    }

    FILE* f = fopen(out.c_str(), "w");
    if (!f) { fprintf(stderr, "cannot open %s\n", out.c_str()); return 1; }
    fprintf(f, "# nanosheet-sp reference sweep (host, serial, fp64)\n");
    fprintf(f, "# classical model: Boltzmann statistics,"
               " n = Nc*exp((EF-Ec+q*phi)/kT) -- STATED APPROXIMATION\n");
    fprintf(f, "# quantum model: N_iv = gs*gv*(mD*kT/(2*pi*hbar^2))*ln(1+exp((EF-E_iv)/kT))"
               " (errata-corrected 2D DOS, see docs/PHYSICS.md)\n");
    fprintf(f, "# stack: gate | SiO2 1.5nm | Si 5nm | SiO2 1.5nm | gate; dEc = 3.1 eV;"
               " grid N = %d, dz = %.4f nm; T = 300 K (kT = %.9f eV)\n",
            N, g.P.dz, g.P.kT);
    fprintf(f, "# constants: m0*kT/(pi*hbar^2) = %.6e cm^-2;"
               " Nc = %.6e cm^-3 (m_d = 6^(2/3)*(ml*mt^2)^(1/3) = 1.060 m0)\n",
            2.0 * g.P.C2D, g.P.NC * 1e21);
    fprintf(f, "# units: Vg [V], Ns [cm^-2], E [eV] (EF = 0 reference)\n");
    fprintf(f, "Vg,Ns_quantum,Ns_classical,"
               "E_d2_1,E_d2_2,E_d2_3,E_d2_4,E_d4_1,E_d4_2,E_d4_3,E_d4_4,"
               "scf_iter_q,scf_iter_cl,converged_q,converged_cl,max_boltzmann_arg\n");

    double t_q = 0.0, t_cl = 0.0;
    auto t0 = std::chrono::steady_clock::now();
    int nfail = 0, max_it = 0;
    double max_arg_sweep = -1e30;

    for (int idx = 0; idx < npts; ++idx) {
        double vg = vg_min + idx * vg_step;

        auto ta = std::chrono::steady_clock::now();
        int it_q; double ma_q;
        int conv_q = sp::scf_solve(g.P, g.Ec.data(), g.eps_h.data(),
                                   g.minv_h[0].data(), g.minv_h[1].data(),
                                   sp::MODEL_QUANTUM, vg,
                                   phi.data(), n_nm3.data(), E.data(), psi.data(),
                                   W, &it_q, &ma_q);
        auto tb = std::chrono::steady_clock::now();
        int it_cl; double ma_cl;
        int conv_cl = sp::scf_solve(g.P, g.Ec.data(), g.eps_h.data(),
                                    g.minv_h[0].data(), g.minv_h[1].data(),
                                    sp::MODEL_CLASSICAL, vg,
                                    phi_cl.data(), n_cl_nm3.data(), nullptr, nullptr,
                                    W, &it_cl, &ma_cl);
        auto tc = std::chrono::steady_clock::now();
        t_q += std::chrono::duration<double>(tb - ta).count();
        t_cl += std::chrono::duration<double>(tc - tb).count();

        double ns_q = sp::sheet_density_cm2(g.P, n_nm3.data());
        double ns_cl = sp::sheet_density_cm2(g.P, n_cl_nm3.data());
        if (!conv_q || !conv_cl) ++nfail;
        max_it = std::max(max_it, std::max(it_q, it_cl));
        max_arg_sweep = std::max(max_arg_sweep, ma_cl);

        fprintf(f, "%.6f,%.10e,%.10e", vg, ns_q, ns_cl);
        for (int s = 0; s < 8; ++s) fprintf(f, ",%.10e", E[s]);
        fprintf(f, ",%d,%d,%d,%d,%.6f\n", it_q, it_cl, conv_q, conv_cl, ma_cl);

        if (!frames_dir.empty() && frame_every > 0 && idx % frame_every == 0) {
            for (int j = 0; j < N; ++j) {
                n_q_cm3[j] = n_nm3[j] * 1e21;
                n_cl_cm3[j] = n_cl_nm3[j] * 1e21;
            }
            char name[64];
            snprintf(name, sizeof name, "frame_%02d.csv", idx / frame_every);
            sp::write_frame(frames_dir + "/" + name, vg, E.data(), g.P.nsub, g,
                            phi.data(), phi_cl.data(), n_q_cm3.data(),
                            n_cl_cm3.data(), psi.data() + 0 * N, psi.data() + 4 * N);
            FILE* idxf = fopen((frames_dir + "/index.csv").c_str(), "a");
            fprintf(idxf, "%s,%.6f\n", name, vg);
            fclose(idxf);
        }
    }
    fclose(f);

    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("reference sweep: %d Vg points in [%.2f, %.2f] step %.3f\n",
           npts, vg_min, vg_min + (npts - 1) * vg_step, vg_step);
    printf("  wall time: %.3f s total (quantum %.3f s, classical %.3f s)\n",
           wall, t_q, t_cl);
    printf("  SCF: max iterations = %d, non-converged points = %d,"
           " max classical Boltzmann exponent at convergence = %.3f (clamp %.1f)\n",
           max_it, nfail, max_arg_sweep, g.P.exp_clamp);
    return nfail ? 1 : 0;
}
