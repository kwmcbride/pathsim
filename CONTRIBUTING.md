# Contributing to PathSim

Thanks for your interest in PathSim. Bug reports, fixes, new blocks and
documentation improvements are all welcome.

## Before you start

For anything beyond a bug fix, please open an issue or a
[discussion](https://github.com/orgs/pathsim/discussions) first. PathSim has a
deliberate architecture and a companion project ([fastsim](https://fastsim.dev))
that it stays aligned with, so a short conversation up front saves everyone from
building in a direction that won't be merged.

Small, focused pull requests get reviewed and merged much faster than large ones.
If you have found several unrelated things, please split them up — a bug fix that
stands on its own should not have to wait for a design discussion about a feature.

## Development setup

```bash
git clone https://github.com/pathsim/pathsim
cd pathsim
pip install -e ".[test]"
```

## Tests

```bash
pytest                       # default run, excludes slow tests
pytest -m "slow or not slow" # everything
```

New code needs tests, and they belong in `tests/pathsim/` mirroring the source
tree. `tests/evals/` is for long-running validation studies only — it is excluded
from the default run, so regular unit tests placed there will not run in CI.

## Code style

PathSim follows a consistent style throughout. Please match the surrounding code
rather than introducing new conventions:

- NumPy-style docstrings with `Parameters` / `Returns` sections
- comments start directly after the `#`, no leading space
- section banners as used in the existing modules
- no new dependencies in the core — it stays on `numpy`, `scipy`, `matplotlib`

## Commits and pull requests

- Keep commit messages short and factual, one line where possible.
- Describe what the change does and why. Test results and open questions are
  useful; marketing is not.
- **Submit your work under your own name.** Please do not add co-author trailers,
  attribution footers or generated-by signatures for AI assistants or other tools
  in commit messages or pull request descriptions. Use whatever tools you like —
  you are the author, and you are vouching for the code you submit.

## Reporting bugs

Please include the PathSim version, your Python and NumPy versions, and a minimal
script that reproduces the problem.

## License

Contributions are made under the MIT License, the same terms as PathSim itself.
