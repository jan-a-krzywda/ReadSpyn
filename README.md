# ReadSpyn — RF Reflectometry Readout Simulator for Quantum Dot Systems

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.0+-orange.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

**ReadSpyn** is a time-domain simulator for RF reflectometry readout of
semiconductor quantum dot (QD) systems.  
It solves the RLC resonator circuit ODE in the presence of realistic charge
noise and produces IQ-demodulated quadrature traces that closely resemble
experimental time series.

---

## Features

| Feature | Details |
|---|---|
| **Self-consistent circuit ODE** | Full Radau integration of the RLC equations; no perturbative approximations |
| **Coulomb peak model** | Analytic `cosh⁻²` conductance; operating-point position visualised on the peak |
| **Flexible noise models** | Spectrum-prescribed 1/*f* noise (`SpectrumNoise`), Ornstein–Uhlenbeck (`OU_noise`), telegraph |
| **Multi-dot / multi-sensor** | Capacitance-matrix formalism; arbitrary *N* dots × *M* sensors |
| **IQ demodulation** | Hilbert-transform demodulation with window-padding edge correction |
| **JAX back-end** | Optional JAX path (`jax_simulator.py`) for GPU/TPU acceleration and `jax.scan` state sweeps |
| **Animated output** | Animated GIF of IQ-plane comet trajectories |

---

## Repository Layout

```
ReadSpyn/
├── src/
│   └── readout_simulator/
│       ├── __init__.py            # public API: QuantumDotSystem, RLC_sensor
│       ├── quantum_dot_system.py  # CI model, capacitance matrices, energy offsets
│       ├── sensor_backend.py      # RLC ODE, IQ demodulation, get_signal()
│       ├── noise_models.py        # SpectrumNoise, OU_noise, telegraph noise
│       ├── jax_simulator.py       # optional JAX-based simulator
│       ├── iaaft.py               # IAAFT surrogate time-series helper
│       └── helper_functions.py    # shared utilities
│
├── examples/
│   ├── single_dot_1f_noise.ipynb  # single dot, 10 realisations of 1/f noise
│   ├── two_dot_iq_gif.ipynb       # two-dot IQ trajectories + animated GIF
│   ├── two_dot_iq.gif             # example output GIF
│   └── minimal_no_noise_single_dot.py
│
├── notes/
│   ├── ReadSpyn_Model_Summary.tex   # LaTeX model summary (theory + limitations)
│   ├── ReadSpyn_Model_Summary.pdf   # compiled PDF
│   └── ReadSpyn_Theory_and_Assumptions.tex  # extended theory reference
│
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
git clone https://github.com/jan-a-krzywda/ReadSpyn.git
cd ReadSpyn
pip install -e .
```

JAX is an optional but recommended dependency for the JAX back-end:

```bash
pip install jax jaxlib          # CPU
# pip install jax[cuda12]       # NVIDIA GPU
```

---

## Quick Start — Single Dot with 1/f Noise

```python
import numpy as np
import sys; sys.path.append('src')

from readout_simulator import QuantumDotSystem, RLC_sensor
from readout_simulator.noise_models import SpectrumNoise

# --- system ---
dot_system = QuantumDotSystem(
    Cdd=np.array([[1.0]]),
    Cds=np.array([[0.5]])
)

sensor = RLC_sensor(
    {'Lc': 800e-9, 'Cp': 0.5e-12, 'RL': 40, 'Rc': 0, 'Z0': 50},
    {'g0': 1/50, 'eps0': 0.5, 'eps_width': 1.0}
)

# --- noise ---
times = np.arange(0, 1e-6, 1e-9)                  # 1 µs, 1 ns steps
noise = SpectrumNoise(psd_func=lambda f: np.where(f > 0, 1/f, 0) + 1, sigma=0.1)
noise_traj = noise.generate_trajectory(times, seed=0)

# --- simulate ---
I, Q, _, _ = sensor.get_signal(
    times=times,
    dot_system=dot_system,
    charge_state=np.array([1]),
    sensor_index=0,
    params={'eps0': 0.0},
    noise_trajectory=noise_traj,
    use_stationary_initial_state=True,
)
print(f"I: mean={I.mean():.4f} V,  Q: mean={Q.mean():.4f} V")
```

---

## Key Concept — Coulomb Peak and Noise Sensitivity

The sensor conductance follows a `cosh⁻²` Coulomb peak:

```
G(ε) = (2/R₀) cosh⁻²(2ε/εw)
```

The IQ fluctuation amplitude is proportional to the **local slope** `|dG/dε|`,
which is zero at the peak centre and maximal on the flanks.
A charge state parked on the steep flank is therefore more noise-sensitive than
one near the peak top — even when both states experience identical charge noise.
This is automatically visualised in the `two_dot_iq_gif.ipynb` notebook.

---

## Examples

| Notebook | What it shows |
|---|---|
| `examples/single_dot_1f_noise.ipynb` | 10 realisations of 1/*f* noise; I and Q time traces per realisation |
| `examples/two_dot_iq_gif.ipynb` | Two-dot system; Coulomb peak positions; animated IQ comet GIF |

---

## Theory and Limitations


- The constant-interaction electrostatic model and lever-arm formula
- The Coulomb-peak conductance model and its assumptions
- The RLC circuit ODE and IQ demodulation scheme
- All noise models (spectrum-prescribed, OU, telegraph)
- A structured list of **current limitations** (quantum effects, amplifier noise,
  thermal effects, lumped-element approximation, …)
- A prioritised road map of planned extensions


---

## Dependencies

| Package | Version |
|---|---|
| Python | ≥ 3.8 |
| NumPy | ≥ 1.21 |
| SciPy | ≥ 1.7 |
| Matplotlib | ≥ 3.5 |
| numba | (for JIT-compiled ODE inner loop) |
| JAX / jaxlib | ≥ 0.4 (optional) |

---

## Authors

- **Jan A. Krzywda** — j.a.krzywda@liacs.leidenuniv.nl
- **Rouven K. Koch** — R.K.Koch@tudelft.nl

## License

MIT — see `pyproject.toml`.

## Citation

```bibtex
@software{readspyn2025,
  title   = {{ReadSpyn}: RF Reflectometry Readout Simulator for Quantum Dot Systems},
  author  = {Krzywda, Jan A. and Koch, Rouven K.},
  year    = {2025},
  url     = {https://github.com/jan-a-krzywda/ReadSpyn}
}
```
