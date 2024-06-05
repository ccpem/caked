# CAKED

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

A package to load tomograms and subtomograms from different sources into a
[PyTorch DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
class.

The DiskDataLoader and DiskDataset classes were initially developed for
[Affinity-VAE](https://github.com/alan-turing-institute/affinity-vae) by Marjan
Famili (@marjanfamili), Jola Mirecka (@jolaem) and Camila Rangel-Smith
(@crangelsmith). These authors have also prepared and curated the corresponding
test data.

## Installation

```bash
python -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install caked
```

From source:

```bash
git clone https://github.com/alan-turing-institute/CAKED
cd CAKED
python -m pip install .
```

## Usage

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [MIT license](LICENSE).

This packaged was based on this very useful and well built
[project template](https://github.com/alan-turing-institute/python-project-template).

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/alan-turing-institute/CAKED/workflows/CI/badge.svg
[actions-link]:             https://github.com/alan-turing-institute/CAKED/actions
[pypi-link]:                https://pypi.org/project/CAKED/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/CAKED
[pypi-version]:             https://img.shields.io/pypi/v/CAKED
<!-- prettier-ignore-end -->
