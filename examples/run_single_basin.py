"""
ProCNE — Case 3: Single Basin Simulation
==========================================
Run this script directly in Spyder (or any IDE) by editing the
parameters below and pressing Run (F5).

The basin mask can be:
    - a .npy boolean array  (True = inside basin)
    - a PNG image           (white = inside, black = outside)

The outlet is a single pixel (row, col) inside the basin where all
drainage collects. The network grows headward from this point.

Outputs saved to OUTPUT_DIR:
    FD_sb_a{alpha}_b{beta}.npy
    Facc_sb_a{alpha}_b{beta}.npy
    network_sb_a{alpha}_b{beta}.png
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from procne import single_basin

# =============================================================================
# PARAMETERS — edit these
# =============================================================================

MASK_PATH  = Path(__file__).resolve().parents[1] / 'data' / 'watershed_sb.npy'
            # Path to basin mask (.npy or PNG). Replace with your own, e.g.:
            # MASK_PATH = Path('my_basin.png')

OUTLET     = (173, 116)
            # (row, col) of the outlet pixel inside the basin.
            # Tip: open the mask in Spyder's variable explorer or imshow
            # to find the right pixel coordinate.

ALPHA      = 1.0    # downstream-length exponent

BETA       = 1.0    # flow-accumulation exponent

SEED       = None   # random seed (None = random each run)

THRESHOLD  = 50     # flow accumulation threshold for displaying streams

OUTPUT_DIR = Path(__file__).parent   # folder where outputs are saved

# =============================================================================

def load_mask(path):
    path = Path(path)
    if path.suffix == '.npy':
        return np.load(path).astype(bool)
    return single_basin.load_basin_mask(path)


def plot_network(Facc, basin, outlet, alpha, beta, threshold, out_path):
    """Plot stream network on a light basin background with outlet marker."""
    cmap_stream = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", ["white", "royalblue"])

    fig, ax = plt.subplots(figsize=(7, 7))

    # Light gray background for basin area
    basin_bg = np.where(basin, 0.88, np.nan)
    ax.imshow(basin_bg, cmap='gray', vmin=0, vmax=1, origin='upper')

    # Blue streams overlaid — vmin/vmax fixes auto-norm mapping 1.0 to white
    streams = np.where((Facc > threshold) & basin, 1.0, np.nan)
    ax.imshow(streams, cmap=cmap_stream, vmin=0, vmax=1, origin='upper')

    # Mark outlet
    ax.plot(outlet[1], outlet[0], 'rv', markersize=9,
            markeredgecolor='darkred', label='Outlet')
    ax.legend(loc='lower right', fontsize=9)

    ax.set_title(f"ProCNE — Single Basin  (α={alpha}, β={beta})", fontsize=11)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {out_path}")
    plt.show()


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"sb_a{ALPHA}_b{BETA}"

    print(f"Loading basin mask: {MASK_PATH}")
    basin = load_mask(MASK_PATH)
    print(f"Basin shape: {basin.shape}, active pixels: {basin.sum()}, "
          f"outlet: {OUTLET}")

    print(f"Running: α={ALPHA}, β={BETA}, seed={SEED}")
    FD, Facc = single_basin.run(
        basin_mask=basin, outlet=OUTLET,
        alpha=ALPHA, beta=BETA,
        seed=SEED, verbose=True)

    np.save(OUTPUT_DIR / f"FD_{tag}.npy",   FD)
    np.save(OUTPUT_DIR / f"Facc_{tag}.npy", Facc)
    print(f"Saved: FD_{tag}.npy  |  Facc_{tag}.npy")

    plot_network(Facc, basin, OUTLET, ALPHA, BETA, THRESHOLD,
                  OUTPUT_DIR / f"network_{tag}.png")
