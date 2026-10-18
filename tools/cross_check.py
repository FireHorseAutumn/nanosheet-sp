#!/usr/bin/env python3
"""cross_check.py — verify GPU batched output against the host reference.

Compares results CSVs (Ns relative, subband energies absolute, converged
flags identical) and frame profile files (relative Linf). Exits nonzero on
any mismatch beyond tolerance.

Usage: cross_check.py results_ref.csv results_gpu.csv [frames_ref frames_gpu]
"""
import sys
import numpy as np

def load_csv(path):
    names = None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if names is None:
                names = line.split(",")
            else:
                rows.append([float(x) for x in line.split(",")])
    arr = np.array(rows)
    return {n: arr[:, i] for i, n in enumerate(names)}

def main():
    ref = load_csv(sys.argv[1])
    gpu = load_csv(sys.argv[2])
    ok = True

    for key in ("converged_q", "converged_cl"):
        if not np.array_equal(ref[key], gpu[key]):
            print(f"MISMATCH: {key} flags differ")
            ok = False

    # Both sides stop at max|dphi| < 1e-6 V, so converged Ns may differ by
    # O(1e-6) relative across fp-noise-divergent damping trajectories; 1e-5
    # confirms the same fixed point without false positives. Energies and
    # frames are far tighter (see printout).
    for key, tol in (("Ns_quantum", 1e-5), ("Ns_classical", 1e-5)):
        r, g = ref[key], gpu[key]
        rel = np.max(np.abs(r - g) / np.maximum(np.abs(r), 1.0))
        print(f"{key}: max rel diff = {rel:.3e} (tol {tol:g})")
        if rel > tol:
            ok = False

    ecols = [f"E_d2_{i}" for i in range(1, 5)] + [f"E_d4_{i}" for i in range(1, 5)]
    ediff = max(np.max(np.abs(ref[c] - gpu[c])) for c in ecols)
    print(f"subband energies: max abs diff = {ediff:.3e} eV (tol 1e-9)")
    if ediff > 1e-9:
        ok = False

    iq = np.max(np.abs(ref["scf_iter_q"] - gpu["scf_iter_q"]))
    ic = np.max(np.abs(ref["scf_iter_cl"] - gpu["scf_iter_cl"]))
    print(f"scf iteration count max abs diff: q={iq:g} cl={ic:g} (report only)")

    if len(sys.argv) >= 4:
        import glob, os
        worst = 0.0
        for fr in sorted(glob.glob(os.path.join(sys.argv[3], "frame_*.csv"))):
            fg = os.path.join(sys.argv[4], os.path.basename(fr))
            a = np.loadtxt(fr, comments="#")
            b = np.loadtxt(fg, comments="#")
            denom = np.maximum(np.abs(a), 1.0)
            worst = max(worst, np.max(np.abs(a - b) / denom))
        print(f"frames: max rel diff = {worst:.3e} (tol 1e-8)")
        if worst > 1e-8:
            ok = False

    print("CROSS-CHECK " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
