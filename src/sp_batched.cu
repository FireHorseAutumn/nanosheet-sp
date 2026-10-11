// sp_batched.cu — GPU batched Schrödinger–Poisson solver, HIP API via
// include/hip_shim.h (developed on CUDA sm_120; AMD path source-level
// portable, not yet validated — see shim header). One threadblock per Vg
// point per model; the whole SCF loop lives in the block: Thomas solves are
// serial in-block (thread 0), Sturm bisection runs one eigenvalue bracket
// per thread (8 threads), inverse iteration one thread per subband in two
// waves of four (shared-memory scratch budget). FP64 everywhere.
// Output schema identical to sp_reference; cross-checked against it.

#include "hip_shim.h"
#include "sp_core.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

constexpr int NMAX = 161;
constexpr int MMAX = NMAX - 2;
constexpr int NT = 64; // threads per block

struct DevParams {
    sp::Params P;
    double gv[2];
    double mdos[2];
    int npts;
};

struct SPResult {
    double vg, ns;
    double E[8];
    int iters, converged, model;
};

__global__ void sp_sweep_kernel(const DevParams DP,
                                const double* __restrict__ vg_list,
                                const double* __restrict__ Ec,
                                const double* __restrict__ eps_h,
                                const double* __restrict__ minv0,
                                const double* __restrict__ minv1,
                                const int* __restrict__ frame_slot,
                                double* __restrict__ frames, // [nframes][7][N]
                                SPResult* __restrict__ out) {
    const sp::Params& P = DP.P;
    const int N = P.N, M = N - 2;
    const int bid = blockIdx.x, tid = threadIdx.x, nthr = blockDim.x;
    const int idx = bid % DP.npts;
    const int model = (bid < DP.npts) ? sp::MODEL_QUANTUM : sp::MODEL_CLASSICAL;
    const double vg = vg_list[idx];
    const double kin = sp::H2_2M0 / (P.dz * P.dz);

    __shared__ double phi[NMAX], phi_new[NMAX], n_arr[NMAX];
    __shared__ double dH[2][MMAX], eH[2][MMAX];
    __shared__ double psiS[8][NMAX];
    __shared__ double Nsub[8], Esub[8];
    __shared__ double scr[4][3][MMAX]; // inverse-iteration waves; Poisson reuses row 0
    __shared__ int s_conv, s_it;

    for (int j = tid; j < N; j += nthr) phi[j] = vg;
    if (tid == 0) { s_conv = 0; s_it = 0; }
    __syncthreads();

    double alpha_eff = P.alpha, prev_delta = 0.0; // used by thread 0 only
    int prev_sign = 0, prev_node = -1;            // used by thread 0 only

    for (int it = 1; it <= P.max_iter; ++it) {
        if (model == sp::MODEL_QUANTUM) {
            for (int i = tid; i < M; i += nthr) {
                int j = i + 1;
                dH[0][i] = kin * (minv0[j-1] + minv0[j]) + Ec[j] - phi[j];
                dH[1][i] = kin * (minv1[j-1] + minv1[j]) + Ec[j] - phi[j];
                if (i < M - 1) {
                    eH[0][i] = -kin * minv0[j];
                    eH[1][i] = -kin * minv1[j];
                }
            }
            __syncthreads();
            if (tid < 2 * P.nsub) { // one eigenvalue bracket per thread
                int g = tid / P.nsub, k = tid % P.nsub;
                double lam = sp::find_eigen(dH[g], eH[g], M, k);
                Esub[tid] = lam;
                Nsub[tid] = 2.0 * DP.gv[g] * P.C2D * DP.mdos[g] *
                            sp::fermilog((P.EF - lam) / P.kT);
            }
            __syncthreads();
            for (int w = 0; w < 2; ++w) { // inverse iteration, 4 threads per wave
                if (tid < 4) {
                    int s = w * 4 + tid, g = s / P.nsub;
                    sp::inverse_iter(dH[g], eH[g], M, Esub[s], P.dz, &psiS[s][1],
                                     scr[tid][0], scr[tid][1], scr[tid][2]);
                    psiS[s][0] = 0.0;
                    psiS[s][N-1] = 0.0;
                }
                __syncthreads();
            }
            for (int j = tid; j < N; j += nthr) {
                double acc = 0.0;
                if (j > 0 && j < N - 1)
                    for (int s = 0; s < 2 * P.nsub; ++s) {
                        double pv = psiS[s][j];
                        acc += Nsub[s] * pv * pv;
                    }
                n_arr[j] = acc * 1e-14;
            }
        } else {
            for (int j = tid; j < N; j += nthr)
                n_arr[j] = (j > 0 && j < N - 1) ? sp::classical_n(P, Ec[j], phi[j]) : 0.0;
        }
        __syncthreads();

        if (tid == 0) { // Poisson: serial Thomas in-block (M = 159, negligible)
            double idz2 = 1.0 / (P.dz * P.dz);
            for (int i = 0; i < M; ++i) {
                int j = i + 1;
                dH[0][i] = (eps_h[j-1] + eps_h[j]) * idz2;
                if (i < M - 1) eH[0][i] = -eps_h[j] * idz2;
                scr[0][2][i] = -sp::QE0 * n_arr[j];
            }
            scr[0][2][0]   += eps_h[0]   * idz2 * vg;
            scr[0][2][M-1] += eps_h[N-2] * idz2 * vg;
            sp::thomas_sym(dH[0], eH[0], scr[0][2], M, scr[0][0], scr[0][1],
                           &phi_new[1]);
            phi_new[0] = vg;
            phi_new[N-1] = vg;
            double delta = 0.0;
            int node = 1;
            for (int j = 1; j < N - 1; ++j) {
                double a = fabs(phi_new[j] - phi[j]);
                if (a > delta) { delta = a; node = j; }
            }
            if (delta < P.tol) {
                for (int j = 1; j < N - 1; ++j) phi[j] = phi_new[j];
                s_conv = 1;
            } else {
                // same adaptive Gummel damping as the host scf_solve
                int sgn = (phi_new[node] > phi[node]) ? 1 : -1;
                if (it > 1) {
                    bool osc = (node == prev_node) && (sgn != prev_sign);
                    if (osc || delta > prev_delta)
                        alpha_eff = fmax(alpha_eff * 0.5, 1e-3);
                    else
                        alpha_eff = fmin(alpha_eff * 1.1, P.alpha);
                }
                prev_delta = delta; prev_sign = sgn; prev_node = node;
                for (int j = 1; j < N - 1; ++j) {
                    double dphi = phi_new[j] - phi[j];
                    if (dphi >  P.dphi_clamp) dphi =  P.dphi_clamp;
                    if (dphi < -P.dphi_clamp) dphi = -P.dphi_clamp;
                    phi[j] += alpha_eff * dphi;
                }
            }
            s_it = it;
        }
        __syncthreads();
        if (s_conv) break;
    }

    if (tid == 0) {
        SPResult r;
        r.vg = vg; r.model = model; r.iters = s_it; r.converged = s_conv;
        double s = 0.0;
        for (int j = 0; j < N; ++j) s += n_arr[j];
        r.ns = s * P.dz * 1e14;
        for (int k = 0; k < 8; ++k) r.E[k] = (model == sp::MODEL_QUANTUM) ? Esub[k] : 0.0;
        out[bid] = r;
    }

    int slot = frame_slot[idx];
    if (slot >= 0) {
        double* fr = frames + (size_t)slot * 7 * N;
        for (int j = tid; j < N; j += nthr) {
            if (model == sp::MODEL_QUANTUM) {
                fr[0 * N + j] = phi[j];
                fr[2 * N + j] = Ec[j] - phi[j];
                fr[3 * N + j] = n_arr[j] * 1e21;
                double p0 = psiS[0][j], p4 = psiS[4][j];
                fr[5 * N + j] = p0 * p0;
                fr[6 * N + j] = p4 * p4;
            } else {
                fr[1 * N + j] = phi[j];
                fr[4 * N + j] = n_arr[j] * 1e21;
            }
        }
    }
}

#define HIP_CHECK(x)                                                          \
    do {                                                                      \
        hipError_t e_ = (x);                                                  \
        if (e_ != hipSuccess) {                                               \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), \
                    __FILE__, __LINE__);                                      \
            return 1;                                                         \
        }                                                                     \
    } while (0)

int main(int argc, char** argv) {
    double vg_min = -0.2, vg_max = 1.0, vg_step = 0.02;
    std::string out = "results_gpu.csv", frames_dir;
    int frame_every = 5;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); exit(2); }
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

    sp::Grid g = sp::make_grid(161, 8.0, 1.5, 5.0, 3.1, 300.0, 0.1, 500,
                               1e-6, 4.0, 0.25, 4);
    const int N = g.P.N;
    int npts = (int)llround((vg_max - vg_min) / vg_step) + 1;

    std::vector<double> vg_list(npts);
    std::vector<int> frame_slot(npts, -1);
    int nframes = 0;
    for (int i = 0; i < npts; ++i) {
        vg_list[i] = vg_min + i * vg_step;
        if (frame_every > 0 && i % frame_every == 0) frame_slot[i] = nframes++;
    }

    DevParams DP;
    DP.P = g.P; DP.npts = npts;
    for (int v = 0; v < 2; ++v) {
        DP.gv[v] = sp::VALLEY_GV[v];
        DP.mdos[v] = sp::VALLEY_MDOS[v];
    }

    double *d_vg, *d_Ec, *d_eps, *d_m0, *d_m1, *d_frames = nullptr;
    int* d_slot;
    SPResult* d_out;
    HIP_CHECK(hipMalloc(&d_vg, npts * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_Ec, N * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_eps, (N - 1) * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_m0, (N - 1) * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_m1, (N - 1) * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_slot, npts * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_out, 2 * npts * sizeof(SPResult)));
    if (nframes > 0) HIP_CHECK(hipMalloc(&d_frames, (size_t)nframes * 7 * N * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_vg, vg_list.data(), npts * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_Ec, g.Ec.data(), N * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_eps, g.eps_h.data(), (N - 1) * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_m0, g.minv_h[0].data(), (N - 1) * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_m1, g.minv_h[1].data(), (N - 1) * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_slot, frame_slot.data(), npts * sizeof(int), hipMemcpyHostToDevice));

    auto t0 = std::chrono::steady_clock::now();
    sp_sweep_kernel<<<2 * npts, NT>>>(DP, d_vg, d_Ec, d_eps, d_m0, d_m1,
                                      d_slot, d_frames, d_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();

    std::vector<SPResult> res(2 * npts);
    HIP_CHECK(hipMemcpy(res.data(), d_out, 2 * npts * sizeof(SPResult), hipMemcpyDeviceToHost));
    std::vector<double> h_frames;
    if (nframes > 0) {
        h_frames.resize((size_t)nframes * 7 * N);
        HIP_CHECK(hipMemcpy(h_frames.data(), d_frames,
                            h_frames.size() * sizeof(double), hipMemcpyDeviceToHost));
    }
    auto t2 = std::chrono::steady_clock::now();
    double t_kernel = std::chrono::duration<double>(t1 - t0).count();
    double t_total = std::chrono::duration<double>(t2 - t0).count();

    FILE* f = fopen(out.c_str(), "w");
    if (!f) { fprintf(stderr, "cannot open %s\n", out.c_str()); return 1; }
    fprintf(f, "# nanosheet-sp batched sweep (GPU, one threadblock per Vg per model, fp64)\n");
    fprintf(f, "# classical model: Boltzmann statistics,"
               " n = Nc*exp((EF-Ec+q*phi)/kT) -- STATED APPROXIMATION\n");
    fprintf(f, "# same solver core (include/sp_core.h) and constants as sp_reference\n");
    fprintf(f, "# units: Vg [V], Ns [cm^-2], E [eV] (EF = 0 reference)\n");
    fprintf(f, "Vg,Ns_quantum,Ns_classical,"
               "E_d2_1,E_d2_2,E_d2_3,E_d2_4,E_d4_1,E_d4_2,E_d4_3,E_d4_4,"
               "scf_iter_q,scf_iter_cl,converged_q,converged_cl\n");

    int nfail = 0, max_it = 0;
    for (int i = 0; i < npts; ++i) {
        const SPResult& q = res[i];
        const SPResult& c = res[npts + i];
        if (!q.converged || !c.converged) ++nfail;
        max_it = std::max(max_it, std::max(q.iters, c.iters));
        fprintf(f, "%.6f,%.10e,%.10e", q.vg, q.ns, c.ns);
        for (int s = 0; s < 8; ++s) fprintf(f, ",%.10e", q.E[s]);
        fprintf(f, ",%d,%d,%d,%d\n", q.iters, c.iters, q.converged, c.converged);
    }
    fclose(f);

    if (!frames_dir.empty() && nframes > 0) {
        std::filesystem::create_directories(frames_dir);
        FILE* idxf = fopen((frames_dir + "/index.csv").c_str(), "w");
        fprintf(idxf, "frame_file,Vg\n");
        for (int i = 0; i < npts; ++i) {
            int slot = frame_slot[i];
            if (slot < 0) continue;
            const double* fr = h_frames.data() + (size_t)slot * 7 * N;
            char name[64];
            snprintf(name, sizeof name, "frame_%02d.csv", slot);
            FILE* ff = fopen((frames_dir + "/" + name).c_str(), "w");
            const SPResult& q = res[i];
            fprintf(ff, "# Vg = %.6f\n", q.vg);
            fprintf(ff, "# E_D2_eV =");
            for (int k = 0; k < 4; ++k) fprintf(ff, " %.6f", q.E[k]);
            fprintf(ff, "\n# E_D4_eV =");
            for (int k = 0; k < 4; ++k) fprintf(ff, " %.6f", q.E[4 + k]);
            fprintf(ff, "\n# columns: z_nm phi_q_V phi_cl_V Ec_minus_qphi_q_eV"
                       " n_q_cm-3 n_cl_cm-3 psi2_D2_1_nm-1 psi2_D4_1_nm-1\n");
            for (int j = 0; j < N; ++j)
                fprintf(ff, "%.6f %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n",
                        g.z[j], fr[0 * N + j], fr[1 * N + j], fr[2 * N + j],
                        fr[3 * N + j], fr[4 * N + j], fr[5 * N + j], fr[6 * N + j]);
            fclose(ff);
            fprintf(idxf, "%s,%.6f\n", name, q.vg);
        }
        fclose(idxf);
    }

    printf("batched sweep: %d Vg points x 2 models = %d blocks of %d threads\n",
           npts, 2 * npts, NT);
    printf("  GPU wall: %.3f ms kernel+sync, %.3f ms incl. D2H copyback\n",
           1e3 * t_kernel, 1e3 * t_total);
    printf("  SCF: max iterations = %d, non-converged points = %d\n", max_it, nfail);

    hipFree(d_vg); hipFree(d_Ec); hipFree(d_eps); hipFree(d_m0); hipFree(d_m1);
    hipFree(d_slot); hipFree(d_out); if (d_frames) hipFree(d_frames);
    return nfail ? 1 : 0;
}
