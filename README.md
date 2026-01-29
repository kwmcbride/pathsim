<p align="center">
  <img src="https://raw.githubusercontent.com/pathsim/pathsim/master/docs/source/logos/pathsim_logo.png" width="300" alt="PathSim Logo" />
</p>

<p align="center">
  <strong>A block-based time-domain system simulation framework in Python</strong>
</p>

<p align="center">
  <a href="https://doi.org/10.21105/joss.08158"><img src="https://joss.theoj.org/papers/10.21105/joss.08158/status.svg" alt="DOI"></a>
  <a href="https://pysimhub.io/projects/pathsim"><img src="https://pysimhub.io/badge.svg" alt="PySimHub"></a>
  <a href="https://pypi.org/project/pathsim/"><img src="https://img.shields.io/pypi/v/pathsim" alt="PyPI"></a>
  <a href="https://anaconda.org/conda-forge/pathsim"><img src="https://img.shields.io/conda/vn/conda-forge/pathsim" alt="Conda"></a>
  <img src="https://img.shields.io/github/license/pathsim/pathsim" alt="License">
  <img src="https://img.shields.io/github/v/release/pathsim/pathsim" alt="Release">
  <img src="https://img.shields.io/pypi/dw/pathsim" alt="Downloads">
  <a href="https://codecov.io/gh/pathsim/pathsim"><img src="https://codecov.io/gh/pathsim/pathsim/branch/master/graph/badge.svg" alt="Coverage"></a>
</p>

<p align="center">
  <a href="https://pathsim.org">Homepage</a> &bull;
  <a href="https://docs.pathsim.org">Documentation</a> &bull;
  <a href="https://view.pathsim.org">PathView Editor</a> &bull;
  <a href="https://github.com/sponsors/milanofthe">Sponsor</a>
</p>

---

PathSim lets you model and simulate complex dynamical systems using an intuitive block diagram approach. Connect sources, integrators, functions, and scopes to build continuous-time, discrete-time, or hybrid systems.

Minimal dependencies: just `numpy`, `scipy`, and `matplotlib`.

## Features

- **Hot-swappable** — modify blocks and solvers during simulation
- **Stiff solvers** — implicit methods (BDF, ESDIRK) for challenging systems
- **Event handling** — zero-crossing detection for hybrid systems
- **Hierarchical** — nest subsystems for modular designs
- **Extensible** — subclass `Block` to create custom components

## Install

```bash
pip install pathsim
```

or with conda:

```bash
conda install conda-forge::pathsim
```

## Quick Example

```python
from pathsim import Simulation, Connection
from pathsim.blocks import Integrator, Amplifier, Adder, Scope

# Damped harmonic oscillator: x'' + 0.5x' + 2x = 0
int_v = Integrator(5)       # velocity, v0=5
int_x = Integrator(2)       # position, x0=2
amp_c = Amplifier(-0.5)     # damping
amp_k = Amplifier(-2)       # spring
add = Adder()
scp = Scope()

sim = Simulation(
    blocks=[int_v, int_x, amp_c, amp_k, add, scp],
    connections=[
        Connection(int_v, int_x, amp_c),
        Connection(int_x, amp_k, scp),
        Connection(amp_c, add),
        Connection(amp_k, add[1]),
        Connection(add, int_v),
    ],
    dt=0.05
)

sim.run(30)
scp.plot()
```

## PathView

[PathView](https://view.pathsim.org) is the graphical editor for PathSim — design systems visually and export to Python.

## Learn More

- [Documentation](https://docs.pathsim.org) — tutorials, examples, and API reference
- [Homepage](https://pathsim.org) — overview and getting started
- [Contributing](https://docs.pathsim.org/pathsim/latest/contributing) — how to contribute

## Citation

If you use PathSim in research, please cite:

```bibtex
@article{Rother2025,
  author = {Rother, Milan},
  title = {PathSim - A System Simulation Framework},
  journal = {Journal of Open Source Software},
  year = {2025},
  volume = {10},
  number = {109},
  pages = {8158},
  doi = {10.21105/joss.08158}
}
```

## License

MIT
