#!/usr/bin/env python3
"""Plot classical/quantum sheet charge and differential gate capacitance.

Usage: plot_curves.py [results.csv]  (default data/results_ref.csv)
Writes data/idvg_curves.png.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_csv(path):
    names, rows = None, []
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
    path = sys.argv[1] if len(sys.argv) > 1 else "data/results_ref.csv"
    d = load_csv(path)
    vg, nsq, nsc = d["Vg"], d["Ns_quantum"], d["Ns_classical"]

    # Differential gate-charge capacitance q dNs/dVg [F/cm^2], not a
    # standalone extraction of the series quantum-capacitance element.
    q = 1.602176634e-19
    cq = q * np.gradient(nsq, vg)
    cc = q * np.gradient(nsc, vg)

    # Vt shift: Vg offset between curves at fixed Ns, over the common range.
    lo = max(nsq.min(), nsc.min()) * 3
    hi = min(nsq.max(), nsc.max()) * 0.3
    grid = np.logspace(np.log10(lo), np.log10(hi), 40)
    vq = np.interp(np.log10(grid), np.log10(nsq), vg)
    vc = np.interp(np.log10(grid), np.log10(nsc), vg)
    shift = vq - vc
    i_rep = np.argmin(np.abs(grid - 1e12))
    print(f"Vt shift at Ns=1e12 cm^-2: {shift[i_rep]*1e3:.0f} mV; "
          f"at Ns=1e13: {np.interp(13.0, np.log10(grid), shift)*1e3:.0f} mV; "
          f"max over sweep: {shift.max()*1e3:.0f} mV")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.semilogy(vg, nsc, "k--", lw=1.6, label="classical (Boltzmann)")
    ax.semilogy(vg, nsq, "r-", lw=2.0, label="quantum (Schrödinger–Poisson)")
    ax.set_xlabel("Vg [V]")
    ax.set_ylabel("Ns [cm$^{-2}$]")
    ax.set_title("5 nm Si nanosheet: sheet charge vs gate bias")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.annotate(f"ΔV ≈ {shift[i_rep]*1e3:.0f} mV @ Ns=10$^{{12}}$ cm$^{{-2}}$",
                xy=(vc[i_rep], 1e12), xytext=(vg.min() + 0.15, 3e12),
                arrowprops=dict(arrowstyle="->", color="0.3"), fontsize=10)

    ax = axes[1]
    ax.plot(vg, cc * 1e6, "k--", lw=1.6, label="classical dQ/dVg")
    ax.plot(vg, cq * 1e6, "r-", lw=2.0, label="quantum dQ/dVg")
    ax.set_xlabel("Vg [V]")
    ax.set_ylabel("C [µF/cm$^2$]")
    ax.set_title("Differential gate-charge capacitance")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out = "data/idvg_curves.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
