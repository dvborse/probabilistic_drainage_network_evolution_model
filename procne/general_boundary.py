"""
ProCNE - Probabilistic Channel Network Evolution
Case 2: General Boundary

Simulates drainage network evolution within an arbitrary domain shape.
The domain boundary is provided as a PNG image (white = inside, black =
outside). All pixels on the boundary of the domain act as outlets.

Reference:
    Borse & Biswal (2023), Advances in Water Resources 171, 104342.
    https://doi.org/10.1016/j.advwatres.2022.104342
"""

import random
import time
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_IJ = ((0, 1, 1), (1, 1, 2), (1, 0, 4), (1, -1, 8),
       (0, -1, 16), (-1, -1, 32), (-1, 0, 64), (-1, 1, 128))

_DIAGONAL_FD = {2, 8, 32, 128}


def _fd_coordinates(fd_matrix, row, col):
    code = int(fd_matrix[row, col])
    offsets = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
               16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
    dr, dc = offsets[code]
    return row + dr, col + dc


def load_boundary_mask(png_path):
    """
    Load a PNG image as a boolean domain mask.

    Any pixel with brightness > 0 is treated as inside the domain.
    The image is converted to grayscale before reading.

    Parameters
    ----------
    png_path : str or Path
        Path to the PNG boundary mask image.

    Returns
    -------
    mask : np.ndarray of bool, shape (rows, cols)
        True where the domain is active.
    """
    img = Image.open(png_path).convert('L')
    arr = np.array(img)
    return arr > 0


def _find_boundary_pixels(domain, interior):
    """
    Return the set of domain pixels that border at least one pixel
    outside the domain (i.e. the shoreline / watershed boundary).
    """
    rows, cols = np.where(domain)
    boundary = set()
    for r, c in zip(rows.tolist(), cols.tolist()):
        for dr, dc, _ in _IJ:
            nr, nc = r + dr, c + dc
            if 0 <= nr < domain.shape[0] and 0 <= nc < domain.shape[1]:
                if not domain[nr, nc]:
                    boundary.add((r, c))
                    break
            else:
                boundary.add((r, c))
                break
    return boundary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(boundary_mask, alpha=1.0, beta=1.0, seed=None, verbose=True):
    """
    Run a single ProCNE simulation on a general-boundary domain.

    Parameters
    ----------
    boundary_mask : np.ndarray of bool  OR  str/Path
        Either a 2-D boolean array (True = inside domain) or a path to
        a PNG mask image (see ``load_boundary_mask``).
    alpha : float
        Downstream-length probability exponent.
    beta : float
        Flow-accumulation probability exponent.
    seed : int or None
        Random seed for reproducibility.
    verbose : bool
        Print progress and timing.

    Returns
    -------
    FD : np.ndarray, shape == boundary_mask.shape
        Flow direction matrix (D8 codes).
    Facc : np.ndarray, shape == boundary_mask.shape
        Flow accumulation matrix.
    """
    if not isinstance(boundary_mask, np.ndarray):
        boundary_mask = load_boundary_mask(boundary_mask)

    if seed is not None:
        random.seed(seed)

    domain = boundary_mask.astype(bool)
    nrows, ncols = domain.shape
    tic = time.time()

    # --- Allocate arrays ---
    FD = np.zeros((nrows, ncols), dtype=np.float32)
    Facc = np.zeros((nrows, ncols), dtype=np.float32)
    Down_length = np.zeros((nrows, ncols), dtype=np.float64)

    # Label: 'n'=unassigned, 'b'=boundary outlet, 'p'=potential, 'y'=assigned
    Label = {}
    for r, c in zip(*np.where(domain)):
        Label[(int(r), int(c))] = 'n'

    # --- Identify boundary (outlet) pixels ---
    boundary_pixels = _find_boundary_pixels(domain, Label)
    for px in boundary_pixels:
        Label[px] = 'b'
        Facc[px] = 1.0

    # --- Initialise potential pixels adjacent to boundary ---
    Potential = []
    for px in boundary_pixels:
        for dr, dc, _ in _IJ:
            nr, nc = px[0] + dr, px[1] + dc
            nb = (nr, nc)
            if nb in Label and Label[nb] == 'n':
                Label[nb] = 'p'
                Potential.append(nb)
                Down_length[nb] = 1.0

    total_pixels = len([k for k, v in Label.items() if v != 'b'])
    log_every = max(1, total_pixels // 10)

    # --- Main simulation loop ---
    step = 0
    while Potential:
        # Step 1: choose potential pixel weighted by downstream length^alpha
        dl = np.array([Down_length[p] for p in Potential])
        weights = np.where(dl > 0, np.power(dl, alpha), 1.0)
        weights = np.maximum(weights, 1e-9)
        cum = np.cumsum(weights)
        rand_val = random.uniform(0, cum[-1])
        idx = int(np.searchsorted(cum, rand_val))
        idx = min(idx, len(Potential) - 1)
        pi = Potential[idx]

        # Step 2: choose flow direction weighted by neighbour Facc^beta
        Values = []
        facc_cum = 0.0
        for dr, dc, fd_code in _IJ:
            nr, nc = pi[0] + dr, pi[1] + dc
            nb = (nr, nc)
            if nb in Label and Label[nb] in ('y', 'b'):
                w = (1.0 + Facc[nb]) ** beta
                facc_cum += w
                Values.append((fd_code, facc_cum))

        if not Values:
            Potential.pop(idx)
            continue

        rand_fd = random.uniform(0, facc_cum)
        for fd_code, cum_w in Values:
            if rand_fd <= cum_w:
                FD[pi] = fd_code
                break

        # Step 3: update label and downstream length
        Potential.pop(idx)
        Label[pi] = 'y'

        nr, nc = _fd_coordinates(FD, pi[0], pi[1])
        step_len = np.sqrt(2) if int(FD[pi]) in _DIAGONAL_FD else 1.0
        Down_length[pi] = Down_length[nr, nc] + step_len

        # Step 4: add new potential neighbours
        for dr, dc, _ in _IJ:
            nr2, nc2 = pi[0] + dr, pi[1] + dc
            nb2 = (nr2, nc2)
            if nb2 in Label and Label[nb2] == 'n':
                Label[nb2] = 'p'
                Potential.append(nb2)
                Down_length[nb2] = Down_length[pi] + 1.0

        # Step 5: propagate flow accumulation downstream
        Facc[pi] = 1.0
        cur = pi
        while Label.get(cur, 'b') != 'b':
            cur = _fd_coordinates(FD, cur[0], cur[1])
            Facc[cur] += 1.0

        step += 1
        if verbose and step % log_every == 0:
            pct = 100 * step / total_pixels
            print(f"  {pct:.0f}% complete ({step}/{total_pixels} pixels)", flush=True)

    elapsed = time.time() - tic
    if verbose:
        print(f"Simulation complete in {elapsed:.1f}s")

    return FD, Facc
