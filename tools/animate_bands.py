#!/usr/bin/env python3
"""animate_bands.py — the one animation: band-bending and electron density
sweeping with gate bias. Left panel: conduction-band edge E_c(z) − qφ(z),
Fermi level, subband levels, ground-state |ψ|² overlay. Right panel:
electron density n(z), quantum vs classical (log scale) — quantum charge
peaks away from the interface, classical piles up at it.

Usage: animate_bands.py [frames_dir]  (default data/frames_ref)
Writes data/band_bending.gif (pillow) or PNG frames fallback.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def load_index(d):
    vgs, files = [], []
    for line in open(f"{d}/index.csv"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("frame_file"):
            continue
        f, v = line.split(",")
        files.append(f"{d}/{f}")
        vgs.append(float(v))
    return files, vgs

def load_frame(path):
    e_d2 = e_d4 = None
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("# E_D2"):
                e_d2 = [float(x) for x in line.split("=")[1].split()]
            elif line.startswith("# E_D4"):
                e_d4 = [float(x) for x in line.split("=")[1].split()]
            elif not line.startswith("#"):
                rows.append([float(x) for x in line.split()])
    a = np.array(rows)
    return a, e_d2, e_d4

def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "data/frames_ref"
    files, vgs = load_index(d)
    frames = [load_frame(f) for f in files]

    fig, (axb, axn) = plt.subplots(1, 2, figsize=(11, 4.6))

    def draw(i):
        a, e_d2, e_d4 = frames[i]
        z, ec, nq, ncl, psi2 = a[:, 0], a[:, 3], a[:, 4], a[:, 5], a[:, 6]
        for ax in (axb, axn):
            ax.clear()
        # oxide/sheet shading
        for ax in (axb, axn):
            ax.axvspan(z.min(), 1.5, color="0.9")
            ax.axvspan(6.5, z.max(), color="0.9")
        axb.plot(z, ec, "b-", lw=2, label="E_c(z) − qφ(z)")
        axb.axhline(0.0, color="k", ls="--", lw=1.2, label="E_F")
        if e_d2:
            for j, e in enumerate(e_d2[:3]):
                axb.hlines(e, 1.5, 6.5, color="r", lw=1.4)
                axb.text(6.6, e, f"Δ2-{j+1}", color="r", fontsize=8, va="center")
        if e_d4:
            axb.hlines(e_d4[0], 1.5, 6.5, color="g", lw=1.2, ls=":")
            axb.text(6.6, e_d4[0], "Δ4-1", color="g", fontsize=8, va="center")
        scale = 0.35 / max(psi2.max(), 1e-30)
        axb.plot(z, e_d2[0] + psi2 * scale, "r-", lw=1, alpha=0.7,
                 label="|ψ₁|² (Δ2)")
        axb.set_xlabel("z [nm]")
        axb.set_ylabel("energy [eV]")
        axb.set_title(f"Band bending — Vg = {vgs[i]:+.2f} V")
        axb.set_ylim(-0.35, 1.0)
        axb.legend(fontsize=8, loc="lower left")
        axn.semilogy(z, np.maximum(ncl, 1.0), "k--", lw=1.6, label="classical")
        axn.semilogy(z, np.maximum(nq, 1.0), "r-", lw=2, label="quantum")
        axn.set_xlabel("z [nm]")
        axn.set_ylabel("n(z) [cm$^{-3}$]")
        axn.set_title("Electron density across the sheet")
        axn.set_ylim(1e10, 1e21)
        axn.legend(fontsize=8, loc="upper center")
        fig.tight_layout()

    ani = FuncAnimation(fig, draw, frames=len(files), interval=700)
    out = "data/band_bending.gif"
    try:
        ani.save(out, writer=PillowWriter(fps=1.4))
        print(f"wrote {out}")
    except Exception as ex:
        for i in range(len(files)):
            draw(i)
            fig.savefig(f"data/band_frame_{i:02d}.png", dpi=120)
        print(f"GIF failed ({ex}); wrote PNG frames data/band_frame_*.png")

if __name__ == "__main__":
    main()
