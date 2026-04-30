"""
ProCNE — Case 1: Square Grid Simulation
========================================
Run this script directly in Spyder (or any IDE) by editing the
parameters below and pressing Run (F5).

Outputs saved to OUTPUT_DIR:
    FD_sq_a{alpha}_b{beta}.npy
    Facc_sq_a{alpha}_b{beta}.npy
    network_sq_a{alpha}_b{beta}.png
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from procne import square_grid

# =============================================================================
# PARAMETERS — edit these
# =============================================================================

GRID_SIZE  = 250    # number of rows and columns in the square grid

ALPHA      = 1.0    # downstream-length exponent
                    # 0 = equal probability (many small basins, elongated)
                    # increasing → fewer, larger, more compact basins

BETA       = 1.0    # flow-accumulation exponent
                    # 0 = equal probability among neighbours
                    # increasing → more compact (rounded) basins

SEED       = None   # random seed for reproducibility (None = random each run)

THRESHOLD  = 100    # flow accumulation threshold for displaying streams
                    # pixels with Facc > THRESHOLD are drawn as streams

OUTPUT_DIR = Path(__file__).parent   # folder where outputs are saved

# =============================================================================

def plot_network(Facc, alpha, beta, threshold, out_path):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", ["white", "royalblue"])
    W = np.where(Facc > threshold, 1.0, 0.0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(W, cmap=cmap, origin='lower')
    ax.set_title(f"ProCNE — Square Grid  (α={alpha}, β={beta})", fontsize=11)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {out_path}")
    plt.show()


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"sq_a{ALPHA}_b{BETA}"

    print(f"Running: grid_size={GRID_SIZE}, α={ALPHA}, β={BETA}, seed={SEED}")
    FD, Facc = square_grid.run(
        grid_size=GRID_SIZE, alpha=ALPHA, beta=BETA,
        seed=SEED, verbose=True)

    np.save(OUTPUT_DIR / f"FD_{tag}.npy",   FD)
    np.save(OUTPUT_DIR / f"Facc_{tag}.npy", Facc)
    print(f"Saved: FD_{tag}.npy  |  Facc_{tag}.npy")

    plot_network(Facc, ALPHA, BETA, THRESHOLD,
                  OUTPUT_DIR / f"network_{tag}.png")
