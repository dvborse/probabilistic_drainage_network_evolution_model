"""
ProCNE - Probabilistic Channel Network Evolution
Case 3: Single Basin

Simulates drainage network evolution for a single watershed draining
to one specified outlet point. A domain mask (boolean array or PNG)
defines the basin boundary, and the user specifies the outlet pixel.

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


def load_basin_mask(png_path):
    """
    Load a PNG image as a boolean basin mask.

    Parameters
    ----------
    png_path : str or Path

    Returns
    -------
    mask : np.ndarray of bool
    """
    img = Image.open(png_path).convert('L')
    return np.array(img) > 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(basin_mask, outlet, alpha=1.0, beta=1.0, seed=None, verbose=True):
    """
    Run a single ProCNE simulation for one basin with a given outlet.

    Parameters
    ----------
    basin_mask : np.ndarray of bool  OR  str/Path
        2-D boolean array (True = inside basin) or path to a PNG mask.
    outlet : tuple (row, col)
        Pixel coordinate of the single basin outlet. This pixel must be
        inside the basin mask.
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
    FD : np.ndarray
        Flow direction matrix (D8 codes).
    Facc : np.ndarray
        Flow accumulation matrix.
    """
    if not isinstance(basin_mask, np.ndarray):
        basin_mask = load_basin_mask(basin_mask)

    if seed is not None:
        random.seed(seed)

    domain = basin_mask.astype(bool)
    nrows, ncols = domain.shape
    tic = time.time()

    # --- Allocate arrays ---
    FD = np.zeros((nrows, ncols), dtype=np.float32)
    Facc = np.zeros((nrows, ncols), dtype=np.float32)
    Down_length = np.zeros((nrows, ncols), dtype=np.float64)
    # length from outlet used for pixel selection probability
    Length = np.zeros(nrows * ncols, dtype=np.float64)

    # Label: 'n'=unassigned, 'o'=outlet, 'p'=potential, 'y'=assigned
    Label = {}
    for r, c in zip(*np.where(domain)):
        Label[(int(r), int(c))] = 'n'

    outlet = tuple(outlet)
    if outlet not in Label:
        raise ValueError(f"Outlet {outlet} is outside the basin mask.")
    Label[outlet] = 'o'

    def _get_id(r, c):
        return r * ncols + c

    # --- Initialise potential pixels adjacent to the outlet ---
    Pot_ids = []
    for dr, dc, _ in _IJ:
        nr, nc = outlet[0] + dr, outlet[1] + dc
        nb = (nr, nc)
        if nb in Label and Label[nb] == 'n':
            Label[nb] = 'p'
            pid = _get_id(nr, nc)
            Pot_ids.append(pid)
            Down_length[nb] = 1.0
            Length[pid] = 1.0

    total_pixels = len([k for k, v in Label.items() if v != 'o'])
    log_every = max(1, total_pixels // 10)

    # --- Main simulation loop ---
    step = 0
    while Pot_ids:
        # Step 1: choose potential pixel weighted by length^alpha
        pot_len = Length[Pot_ids]
        weights = np.where(pot_len > 0, np.power(pot_len, alpha), 1.0)
        weights = np.maximum(weights, 1e-9)
        cum = np.cumsum(weights)
        rand_val = random.uniform(0, cum[-1])
        idx = int(np.searchsorted(cum, rand_val))
        idx = min(idx, len(Pot_ids) - 1)

        pi_id = Pot_ids[idx]
        pi = (pi_id // ncols, pi_id % ncols)

        # Step 2: choose flow direction weighted by neighbour Facc^beta
        Values = []
        facc_cum = 0.0
        for dr, dc, fd_code in _IJ:
            nr, nc = pi[0] + dr, pi[1] + dc
            nb = (nr, nc)
            if nb in Label and Label[nb] in ('y', 'o'):
                w = (1.0 + Facc[nb]) ** beta
                facc_cum += w
                Values.append((fd_code, facc_cum))

        if not Values:
            Pot_ids.pop(idx)
            continue

        rand_fd = random.uniform(0, facc_cum)
        for fd_code, cum_w in Values:
            if rand_fd <= cum_w:
                FD[pi] = fd_code
                break

        # Step 3: update label and downstream length
        Pot_ids.pop(idx)
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
                pid2 = _get_id(nr2, nc2)
                Pot_ids.append(pid2)
                Down_length[nb2] = Down_length[pi] + 1.0
                Length[pid2] = Length[pi_id] + 1.0

        # Step 5: propagate flow accumulation downstream to outlet
        Facc[pi] = 1.0
        cur = pi
        while Label.get(cur, 'o') != 'o':
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
