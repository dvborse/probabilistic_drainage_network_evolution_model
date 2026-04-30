# ProCNE — Probabilistic Channel Network Evolution

Python implementation of the probabilistic drainage network evolution model
described in:

> Borse, D. & Biswal, B. (2023). *A novel probabilistic model to explain
> drainage network evolution.* **Advances in Water Resources**, 171, 104342.
> https://doi.org/10.1016/j.advwatres.2022.104342

---

## Model overview

The model grows a drainage network headward from outlets using two
probabilistic rules at each step:

1. **Which pixel grows next?**  
   A potential pixel (bordering the current network) is chosen with
   probability ∝ `downstream_length^α`. Higher `α` gives larger basins.

2. **Which direction does it drain?**  
   Flow direction is assigned toward an already-assigned neighbour with
   probability ∝ `(1 + flow_accumulation)^β`. Higher `β` gives more
   compact (rounded) basins.

The simulated networks reproduce the power-law scaling laws observed in
real river basins (Hack's law, exceedance distributions of area and length).

---

## Repository structure

```
ProCNE/
├── procne/
│   ├── square_grid.py        # Case 1: square planar grid
│   ├── general_boundary.py   # Case 2: arbitrary PNG domain mask
│   └── single_basin.py       # Case 3: single basin with one outlet
├── analysis/
│   └── scaling_relationships.py   # post-simulation scaling analysis
├── examples/
│   ├── run_square_grid.py
│   ├── run_general_boundary.py
│   ├── run_single_basin.py
│   └── tasmania_mask.png          # sample boundary mask (Tasmania)
├── data/
│   └── watershed_sb.npy           # sample basin mask for single-basin case
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

No compiled dependencies required.

---

## Quick start

### Case 1 — Square grid

```bash
python examples/run_square_grid.py --grid-size 250 --alpha 1 --beta 1
```

Or in Python:

```python
from procne import square_grid
import numpy as np

FD, Facc = square_grid.run(grid_size=250, alpha=1.0, beta=1.0, seed=42)
np.save("FD.npy", FD)
np.save("Facc.npy", Facc)
```

---

### Case 2 — General boundary (PNG mask)

Provide a PNG image where **white pixels = inside the domain** and
**black pixels = outside**. The mask can be created in any image editor.

```bash
python examples/run_general_boundary.py \
    --mask examples/tasmania_mask.png \
    --alpha 1 --beta 1
```

Or in Python:

```python
from procne import general_boundary

FD, Facc = general_boundary.run(
    boundary_mask="examples/tasmania_mask.png",
    alpha=1.0, beta=1.0
)
```

---

### Case 3 — Single basin

Provide a basin mask (`.npy` boolean array or PNG) and specify the
outlet pixel.

```bash
python examples/run_single_basin.py \
    --mask data/watershed_sb.npy \
    --outlet 173 116 \
    --alpha 1 --beta 1
```

Or in Python:

```python
import numpy as np
from procne import single_basin

basin = np.load("data/watershed_sb.npy").astype(bool)
FD, Facc = single_basin.run(basin, outlet=(173, 116), alpha=1.0, beta=1.0)
```

---

## Scaling analysis

After running a square-grid simulation, compute the scaling exponents:

```bash
python analysis/scaling_relationships.py \
    --fd FD.npy --facc Facc.npy --save scaling.png
```

Expected exponent ranges for real river networks:

| Exponent | Symbol | Typical range |
|----------|--------|---------------|
| Hack's   | h      | 0.56 – 0.60   |
| Area     | ε      | −0.41 to −0.43 |
| Length   | φ      | −0.70 to −0.80 |

---

## Parameters

| Parameter | Effect |
|-----------|--------|
| `alpha`   | Controls basin size. `α=0` → equal-probability growth (symmetric, elongated basins). Increasing `α` → larger dominant basins. |
| `beta`    | Controls basin compactness. Increasing `β` → more compact (rounded) basins. |

---

## Outputs

Each simulation saves:

- `FD_*.npy` — flow direction matrix (D8 coding: 1, 2, 4, 8, 16, 32, 64, 128)
- `Facc_*.npy` — flow accumulation matrix
- `network_*.png` — stream network plot (streams above threshold)

---

## Citing

If you use this code, please cite:

```
Borse, D., & Biswal, B. (2023). A novel probabilistic model to explain 
drainage network evolution. Advances in Water Resources, 171, 104342.
https://doi.org/10.1016/j.advwatres.2022.104342
```
