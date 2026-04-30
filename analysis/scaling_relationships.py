"""
ProCNE — Scaling Relationships Analysis

Computes and plots the three key power-law scaling relationships for a
simulated (or real) drainage network:

    1. Hack's law:                L  ∝  A^h
    2. Exceedance prob. of area:  P(Ad ≥ δ)  ∝  δ^(−ε)
    3. Exceedance prob. of length:P(x ≥ l)   ∝  l^(−φ)

Usage
-----
    python scaling_relationships.py --fd FD.npy --facc Facc.npy

or import and call directly:

    from analysis.scaling_relationships import compute_all, plot_all
    results = compute_all(FD, Facc, grid_size=250)
    plot_all(results, save_path="scaling.png")
    
    Note- For single basin use outlet_mask by making only outlet pixel as true for 
    correct results. For square or general boundary, boundary serves as outlet_mask 
    And for multiple basins, it will select largest basin for scaling analyses.
"""

import argparse  # retained for optional CLI use
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIAGONAL_FD = {2, 8, 32, 128}


def _fd_coordinates(fd_matrix, row, col):
    code = int(fd_matrix[row, col])
    offsets = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
               16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
    dr, dc = offsets[code]
    return row + dr, col + dc


def _build_outlet_mask(domain):
    """
    Identify outlet pixels: domain pixels that border at least one
    non-domain pixel or lie on the edge of the grid.
    For single basin, don't use this function and manually set outlet_mask true only for outlet pixel
    for more accurage results
    """
    nrows, ncols = domain.shape
    outlet_mask = np.zeros((nrows, ncols), dtype=bool)
    for r, c in zip(*np.where(domain)):
        # grid-edge pixels are always outlets
        if r == 0 or r == nrows - 1 or c == 0 or c == ncols - 1:
            outlet_mask[r, c] = True
            continue
        # interior domain pixel neighbouring a non-domain pixel
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if not domain[r + dr, c + dc]:
                    outlet_mask[r, c] = True
                    break
            if outlet_mask[r, c]:
                break
    return outlet_mask


def _compute_upstream_length(FD, domain, outlet_mask):
    """
    Compute the maximum upstream channel length for every active pixel.
    Works for all three simulation cases (square grid, general boundary,
    single basin). Traces downstream from each pixel until reaching an
    outlet, stopping early when a longer length is already recorded.
    """
    nrows, ncols = domain.shape
    Up_length = np.zeros((nrows, ncols), dtype=np.float64)

    for i in range(nrows):
        for j in range(ncols):
            if not domain[i, j] or Up_length[i, j] != 0:
                continue
            xpi = (i, j)
            up_list = [0.0]
            while not outlet_mask[xpi]:
                code = int(FD[xpi])
                if code == 0:   # safety: no valid FD assigned, treat as outlet
                    break
                next_pi = _fd_coordinates(FD, xpi[0], xpi[1])
                step = np.sqrt(2) if int(FD[xpi]) in _DIAGONAL_FD else 1.0
                up_list.append(up_list[-1] + step)
                if Up_length[next_pi] < up_list[-1]:
                    Up_length[next_pi] = up_list[-1]
                    xpi = next_pi
                else:
                    break
    return Up_length


def _delineate_largest_watershed(FD, Facc, domain, outlet_mask):
    """
    Delineate the largest watershed by finding the outlet pixel with the
    highest flow accumulation, then tracing upstream from it.
    Works for all three simulation cases.
    """
    nrows, ncols = domain.shape

    # Outlet of largest basin = domain outlet pixel with highest Facc
    outlet_facc = np.where(outlet_mask, Facc, 0.0)
    outlet = tuple(np.unravel_index(np.argmax(outlet_facc), Facc.shape))

    # Reverse-FD table: (dr, dc, fd_code_that_points_into_this_neighbour)
    reverse_ij = [(0, 1, 16), (1, 1, 32), (1, 0, 64), (1, -1, 128),
                  (0, -1, 1), (-1, -1, 2), (-1, 0, 4), (-1, 1, 8)]

    watershed = np.zeros((nrows, ncols), dtype=bool)
    watershed[outlet] = True
    frontier = [outlet]

    while frontier:
        next_wave = []
        for pixel in frontier:
            for dr, dc, rev_code in reverse_ij:
                nr, nc = pixel[0] + dr, pixel[1] + dc
                if (0 <= nr < nrows and 0 <= nc < ncols
                        and domain[nr, nc]
                        and not watershed[nr, nc]
                        and int(FD[nr, nc]) == rev_code):
                    watershed[nr, nc] = True
                    next_wave.append((nr, nc))
        frontier = next_wave

    return watershed, outlet



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_all(FD, Facc, domain=None, facc_threshold=50):
    """
    Compute Hack's law and exceedance probability scaling exponents.

    Works for all three ProCNE simulation cases (square grid, general
    boundary, single basin).

    Parameters
    ----------
    FD : np.ndarray
        Flow direction matrix.
    Facc : np.ndarray
        Flow accumulation matrix.
    domain : np.ndarray of bool or None
        Boolean mask of active pixels (True = inside domain). If None,
        all pixels are treated as active (square-grid case).
    facc_threshold : int
        Minimum flow accumulation for a pixel to be included in Hack's
        law regression (filters out headwater pixels).

    Returns
    -------
    dict with keys:
        h, k          — Hack's exponent and coefficient
        epsilon       — contributing-area exceedance exponent
        phi           — upstream-length exceedance exponent
        H_area        — area values used for Hack's law
        H_length      — length values used for Hack's law
        P_conarea     — (delta, P) array for area exceedance
        P_uplength    — (l, P) array for length exceedance
        watershed     — boolean watershed mask
        outlet        — (row, col) of outlet pixel
    """
    if domain is None:
        domain = Facc>0

    outlet_mask = _build_outlet_mask(domain)
    Up_length   = _compute_upstream_length(FD, domain, outlet_mask)
    watershed, outlet = _delineate_largest_watershed(FD, Facc, domain, outlet_mask)

    # --- Hack's law (all active pixels above threshold) ---
    H_area, H_length = [], []
    rows, cols = np.where(domain & (Facc > facc_threshold))
    for r, c in zip(rows, cols):
        H_area.append(Facc[r, c])
        H_length.append(Up_length[r, c])

    H_area = np.array(H_area)
    H_length = np.array(H_length)
    mask = H_area > 0
    h, log_k = np.polyfit(np.log(H_area[mask]), np.log(H_length[mask]), 1)
    k = np.exp(log_k)

    # --- Exceedance probability of contributing area ---
    con_area = np.where(watershed, Facc, 0.0)
    max_a = int(np.amax(con_area))
    total_a = int((con_area > 0).sum())
    P_conarea = np.zeros((max_a, 2))
    for i in range(max_a):
        P_conarea[i, 0] = i + 1
        P_conarea[i, 1] = float((con_area >= i + 1).sum()) / total_a

    # Slope break: fit up to where P drops below 3%
    sb_a = int(np.searchsorted(-P_conarea[:, 1], -0.03))
    sb_a = max(sb_a, 5)
    valid = P_conarea[3:sb_a, 1] > 0
    if valid.sum() > 2:
        epsilon, _ = np.polyfit(
            np.log(P_conarea[3:sb_a, 0][valid]),
            np.log(P_conarea[3:sb_a, 1][valid]), 1)
    else:
        epsilon = np.nan

    # --- Exceedance probability of upstream length ---
    up_dist = np.where(watershed, Up_length, 0.0)
    max_l = int(np.amax(up_dist))
    total_l = int((up_dist > 0).sum())
    P_uplength = np.zeros((max_l, 2))
    for i in range(max_l):
        P_uplength[i, 0] = i + 1
        P_uplength[i, 1] = float((up_dist >= i + 1).sum()) / total_l

    sb_l = int(np.searchsorted(-P_uplength[:, 1], -0.05))
    sb_l = max(sb_l, 5)
    valid_l = P_uplength[:sb_l, 1] > 0
    if valid_l.sum() > 2:
        phi, _ = np.polyfit(
            np.log(P_uplength[:sb_l, 0][valid_l]),
            np.log(P_uplength[:sb_l, 1][valid_l]), 1)
    else:
        phi = np.nan

    print(f"  Hack's exponent h     = {h:.4f}")
    print(f"  Area exponent ε       = {epsilon:.4f}")
    print(f"  Length exponent φ     = {phi:.4f}")

    return dict(h=h, k=k, epsilon=epsilon, phi=phi,
                H_area=H_area, H_length=H_length,
                P_conarea=P_conarea, P_uplength=P_uplength,
                watershed=watershed, outlet=outlet,
                sb_a=sb_a, sb_l=sb_l)


def plot_all(results, save_path=None, show=True):
    """
    Plot the three scaling relationships side by side.

    Parameters
    ----------
    results : dict
        Output from ``compute_all``.
    save_path : str or None
        If given, save the figure to this path.
    show : bool
        Call plt.show() after plotting.
    """
    h = results['h']
    k = results['k']
    epsilon = results['epsilon']
    phi = results['phi']
    H_area = results['H_area']
    H_length = results['H_length']
    P_ca = results['P_conarea']
    P_ul = results['P_uplength']
    sb_a = results['sb_a']
    sb_l = results['sb_l']

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # --- Hack's law ---
    ax = axes[0]
    ax.scatter(H_area, H_length, s=1, color='royalblue', alpha=0.4)
    fit_x = np.linspace(H_area.min(), H_area.max(), 200)
    ax.loglog(fit_x, k * fit_x**h, color='black', linewidth=1.2,
              label=f'h = {h:.3f}')
    ax.set_xlabel('Contributing area $A_d$')
    ax.set_ylabel('Upstream length $L$')
    ax.set_title("Hack's Law")
    ax.legend(fontsize=9)

    # --- Exceedance prob of contributing area ---
    ax = axes[1]
    ax.loglog(P_ca[:, 0], P_ca[:, 1], linewidth=0,
              color='royalblue', marker='o', markersize=2)
    if not np.isnan(epsilon):
        intercept = np.exp(np.polyfit(
            np.log(P_ca[3:sb_a, 0]), np.log(P_ca[3:sb_a, 1]), 1)[1])
        ax.loglog(P_ca[3:sb_a, 0],
                  intercept * P_ca[3:sb_a, 0]**epsilon,
                  color='black', linewidth=1.2, label=f'ε = {epsilon:.3f}')
    ax.set_xlabel('Area ($\\delta$)')
    ax.set_ylabel('$P[A_d \\geq \\delta]$')
    ax.set_title('Exceedance Prob. — Area')
    ax.legend(fontsize=9)

    # --- Exceedance prob of upstream length ---
    ax = axes[2]
    ax.loglog(P_ul[:, 0], P_ul[:, 1], linewidth=0,
              color='royalblue', marker='o', markersize=2)
    if not np.isnan(phi):
        intercept = np.exp(np.polyfit(
            np.log(P_ul[:sb_l, 0]), np.log(P_ul[:sb_l, 1]), 1)[1])
        ax.loglog(P_ul[:sb_l, 0],
                  intercept * P_ul[:sb_l, 0]**phi,
                  color='black', linewidth=1.2, label=f'φ = {phi:.3f}')
    ax.set_xlabel('Length ($l$)')
    ax.set_ylabel('$P[x \\geq l]$')
    ax.set_title('Exceedance Prob. — Length')
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Run directly in Spyder (F5) — edit parameters below
# ---------------------------------------------------------------------------

# =============================================================================
# PARAMETERS — edit these
# =============================================================================

FD_PATH    = Path("FD_sq_a1.0_b1.0.npy")   # path to saved FD matrix
FACC_PATH  = Path("Facc_sq_a1.0_b1.0.npy") # path to saved Facc matrix

THRESHOLD  = 50     # minimum flow accumulation for Hack's law regression

SAVE_PATH  = Path("scaling")   # set to e.g. Path("scaling.png") to save the figure

# =============================================================================

if __name__ == '__main__':
    import sys
    FD   = np.load(FD_PATH)
    Facc = np.load(FACC_PATH)

    # Load domain mask if provided
    domain = Facc>0

    print(f"Loaded FD {FD.shape}, Facc {Facc.shape}")
    if domain is not None:
        domain=Facc>0
        print(f"Domain mask: {domain.shape}, active pixels: {domain.sum()}")
    print("Computing scaling relationships...")
    results = compute_all(FD, Facc, domain=domain, facc_threshold=THRESHOLD)
    plot_all(results, save_path=SAVE_PATH, show=True)
