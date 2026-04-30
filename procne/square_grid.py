"""
ProCNE - Probabilistic Channel Network Evolution
Case 1: Square Grid

Simulates drainage network evolution on a square planar grid.
All border pixels act as outlets; the network grows inward via
probabilistic headward growth.

Reference:
    Borse & Biswal (2023), Advances in Water Resources 171, 104342.
    https://doi.org/10.1016/j.advwatres.2022.104342
"""

import random
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# D8 neighbourhood: (row_offset, col_offset, FD_code)
_IJ = ((0, 1, 1), (1, 1, 2), (1, 0, 4), (1, -1, 8),
       (0, -1, 16), (-1, -1, 32), (-1, 0, 64), (-1, 1, 128))

_DIAGONAL_FD = {2, 8, 32, 128}


def _fd_coordinates(fd_matrix, row, col):
    """Return (r, c) of the pixel that (row, col) drains into."""
    code = fd_matrix[row, col]
    offsets = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
               16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
    dr, dc = offsets[code]
    return row + dr, col + dc


def _is_border(row, col, n):
    return row == 0 or row == n - 1 or col == 0 or col == n - 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(grid_size=250, alpha=1.0, beta=1.0, seed=None, verbose=True):
    """
    Run a single ProCNE simulation on a square grid.

    Parameters
    ----------
    grid_size : int
        Number of rows (and columns) in the square grid.
    alpha : float
        Exponent controlling the probability weight of downstream length
        when selecting the next potential pixel. Higher alpha favours
        pixels farther from the outlet (larger basins).
    beta : float
        Exponent controlling the probability weight of flow accumulation
        when assigning flow direction. Higher beta produces more compact
        (rounded) basins.
    seed : int or None
        Random seed for reproducibility. None means a random run.
    verbose : bool
        Print progress and timing information.

    Returns
    -------
    FD : np.ndarray, shape (grid_size, grid_size)
        Flow direction matrix (D8 coding: 1, 2, 4, 8, 16, 32, 64, 128).
    Facc : np.ndarray, shape (grid_size, grid_size)
        Flow accumulation matrix (number of upstream pixels draining
        through each cell, including itself).
    """
    if seed is not None:
        random.seed(seed)

    n = grid_size
    tic = time.time()

    # --- Allocate arrays ---
    FD = np.zeros((n, n), dtype=np.float32)
    Facc = np.zeros((n, n), dtype=np.float32)
    Down_length = np.zeros((n, n), dtype=np.float64)

    # --- Label grid: 'n'=unassigned, 'p'=potential, 'y'=assigned ---
    Label = np.full((n, n), 'n', dtype='U1')

    # --- Set border pixels as outlets (flow outward) ---
    FD[0, :] = 64          # top row    → up
    FD[n-1, :] = 4         # bottom row → down
    FD[:, 0] = 16          # left col   → left
    FD[:, n-1] = 1         # right col  → right
    FD[0, 0] = 32          # top-left corner
    FD[0, n-1] = 128       # top-right corner
    FD[n-1, 0] = 8         # bottom-left corner
    FD[n-1, n-1] = 2       # bottom-right corner

    Facc[0, :] = Facc[n-1, :] = Facc[:, 0] = Facc[:, n-1] = 1
    Label[0, :] = Label[n-1, :] = Label[:, 0] = Label[:, n-1] = 'y'

    # --- Initialise potential pixels (inner border layer) ---
    Down_length[1, 1:n-1] = 1
    Down_length[n-2, 1:n-1] = 1
    Down_length[1:n-1, 1] = 1
    Down_length[1:n-1, n-2] = 1

    Potential = []
    for i in range(1, n - 1):
        for j in (1, n - 2):
            if Label[i, j] == 'n':
                Label[i, j] = 'p'
                Potential.append((i, j))
        for j in range(2, n - 2):
            if Label[1, j] == 'n':
                Label[1, j] = 'p'
                if (1, j) not in Potential:
                    Potential.append((1, j))
            if Label[n-2, j] == 'n':
                Label[n-2, j] = 'p'
                if (n-2, j) not in Potential:
                    Potential.append((n-2, j))

    # Deduplicate while preserving order
    seen = set()
    unique_potential = []
    for px in Potential:
        if px not in seen:
            seen.add(px)
            unique_potential.append(px)
    Potential = unique_potential

    total_pixels = (n - 2) ** 2
    log_every = max(1, total_pixels // 10)

    # --- Main simulation loop ---
    step = 0
    while Potential:
        # Step 1: choose a potential pixel weighted by downstream length^alpha
        dl = np.array([Down_length[p] for p in Potential])
        weights = np.power(dl, alpha, where=dl > 0, out=np.ones_like(dl))
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
            if Label[nr, nc] == 'y':
                w = (1.0 + Facc[nr, nc]) ** beta
                facc_cum += w
                Values.append((fd_code, facc_cum))

        rand_fd = random.uniform(0, facc_cum)
        for fd_code, cum_w in Values:
            if rand_fd <= cum_w:
                FD[pi] = fd_code
                break

        # Step 3: update label, downstream length
        Potential.pop(idx)
        Label[pi] = 'y'

        nr, nc = _fd_coordinates(FD, pi[0], pi[1])
        step_len = np.sqrt(2) if int(FD[pi]) in _DIAGONAL_FD else 1.0
        Down_length[pi] = Down_length[nr, nc] + step_len

        # Step 4: add new potential neighbours
        for dr, dc, _ in _IJ:
            nr2, nc2 = pi[0] + dr, pi[1] + dc
            if Label[nr2, nc2] == 'n':
                Label[nr2, nc2] = 'p'
                Potential.append((nr2, nc2))
                Down_length[nr2, nc2] = Down_length[pi] + 1.0

        # Step 5: propagate flow accumulation downstream
        Facc[pi] = 1.0
        cur = pi
        while not _is_border(cur[0], cur[1], n):
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
