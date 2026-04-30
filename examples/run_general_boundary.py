"""
ProCNE — Case 2: General Boundary Simulation
==============================================
Run this script directly in Spyder (or any IDE) by editing the
parameters below and pressing Run (F5).

The boundary mask is a PNG image where:
    white (or any non-black pixel) = inside the domain
    black = outside the domain

Any image editor (Paint, GIMP, Inkscape) can be used to create a mask.
The included tasmania_mask.png is used by default.

Outputs saved to OUTPUT_DIR:
    FD_{mask_name}_a{alpha}_b{beta}.npy
    Facc_{mask_name}_a{alpha}_b{beta}.npy
    network_{mask_name}_a{alpha}_b{beta}.png
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from procne import general_boundary

# =============================================================================
# PARAMETERS — edit these
# =============================================================================

MASK_PATH  = Path(__file__).parent / 'tasmania_mask.png'
            # Path to your PNG boundary mask.
            # Replace with your own mask, e.g.:
            # MASK_PATH = Path('my_domain.png')

ALPHA      = 1.0    # downstream-length exponent (see README for details)

BETA       = 1.0    # flow-accumulation exponent

SEED       = None   # random seed (None = random each run)

THRESHOLD  = 50     # flow accumulation threshold for displaying streams

OUTPUT_DIR = Path(__file__).parent   # folder where outputs are saved

# =============================================================================

def plot_network(Facc, domain, alpha, beta, threshold, out_path):
    """Plot stream network on a light domain background."""
    cmap_stream = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", ["white", "royalblue"])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Light gray background for the domain area
    domain_bg = np.where(domain, 0.88, np.nan)
    ax.imshow(domain_bg, cmap='gray', vmin=0, vmax=1, origin='upper')

    # Blue streams overlaid — vmin/vmax fixes auto-norm mapping 1.0 to white
    streams = np.where((Facc > threshold) & domain, 1.0, np.nan)
    ax.imshow(streams, cmap=cmap_stream, vmin=0, vmax=1, origin='upper')

    ax.set_title(f"ProCNE — General Boundary  (α={alpha}, β={beta})", fontsize=11)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {out_path}")
    plt.show()


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mask_name = MASK_PATH.stem
    tag = f"{mask_name}_a{ALPHA}_b{BETA}"

    print(f"Loading boundary mask: {MASK_PATH}")
    domain = general_boundary.load_boundary_mask(MASK_PATH)
    print(f"Domain shape: {domain.shape}, active pixels: {domain.sum()}")

    print(f"Running: α={ALPHA}, β={BETA}, seed={SEED}")
    FD, Facc = general_boundary.run(
        boundary_mask=domain, alpha=ALPHA, beta=BETA,
        seed=SEED, verbose=True)

    np.save(OUTPUT_DIR / f"FD_{tag}.npy",   FD)
    np.save(OUTPUT_DIR / f"Facc_{tag}.npy", Facc)
    print(f"Saved: FD_{tag}.npy  |  Facc_{tag}.npy")

    plot_network(Facc, domain, ALPHA, BETA, THRESHOLD,
                  OUTPUT_DIR / f"network_{tag}.png")
