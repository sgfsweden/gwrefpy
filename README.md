<p align="center">
  <img width="200" height="200" alt="gwrefpy" src="https://github.com/user-attachments/assets/4763ec7a-c703-414f-ba81-1520793f5f8d" style="display: block; margin: 0 auto" />
</p>

# gwrefpy

<p>
  <a href="https://pypi.org/project/gwrefpy/"><img src="https://img.shields.io/pypi/v/gwrefpy.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/gwrefpy/"><img src="https://img.shields.io/pypi/pyversions/gwrefpy.svg" alt="Python versions"></a>
  <a href="https://github.com/sgfsweden/gwrefpy/blob/main/LICENSE"><img src="https://img.shields.io/github/license/andersretznersgu/gwrefpy.svg" alt="License"></a>
  <a href="https://sgfsweden.github.io/gwrefpy/"><img src="https://img.shields.io/badge/docs-latest-brightgreen.svg" alt="Documentation"></a>
  <a href="https://github.com/sgfsweden/gwrefpy/actions"><img src="https://img.shields.io/github/actions/workflow/status/sgfsweden/gwrefpy/.github/workflows/publish_on_pypi.yml?branch=main" alt="Build status"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://github.com/sgfsweden/gwrefpy"><img src="https://img.shields.io/github/stars/sgfsweden/gwrefpy?style=social" alt="GitHub stars"></a>
</p>

A Python implementation of the Akvifär reference method for detecting deviations in groundwater level time series.

## Features

- Programmatically fit observation wells to reference wells
- Visualize fits and deviations
- Save your work, share and pick up later with a custom `.gwref` file format
- More to come...

## Installation

Using `uv` or `pip`:

```bash
uv install gwrefpy
# or
pip install gwrefpy
```

## Documentation

See the [documentation](https://andersretznersgu.github.io/gwrefpy/) for more information on how to use `gwrefpy`.

## Contributing

Contributions are welcome! File an issue or submit a pull request on GitHub. We recommend `uv` for development.

```bash
git clone https://github.com/andersretznersgu/gwrefpy.git
cd gwrefpy

uv sync --all-groups
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

Thanks to [Svenska Geotekniska Förening](https://svenskageotekniskaforeningen.se/) for funding the development of this package.
