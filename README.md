# MolMap Package Summary
## Overview
This repository contains the code and data used to generate the results presented in the manuscript:
**Bayesian Optimization in Chemical Compound Sub-Spaces using
Low-Dimensional Molecular Descriptors** Yun-Wen Mao and Roman V Krems

The repository is intended to facilitate reproducibility of the results presented in the manuscript.
It inclludes the Bayesian optimization workflows, the inverse mapping algorithm, and data of the figures in the manuscript.

## Repository structure

```text
.
├── atom_pdf.py        # Computes the reference atomic probability distribution (Eq. 14 in the manuscript)
├── boundary.py        # Estimates the descriptor boundaries used for Bayesian optimization
├── chemistry.py       # Utility functions for molecular and chemistry-related operations
├── config.py          # Default configuration parameters
├── data.py            # Utility functions for loading and processing datasets
├── descriptor.py      # Generates the molecular descriptors used in this work
├── mapping.py         # Maps descriptor vectors to candidate chemical formulas and molecules
├── optimization.py    # Implements the Bayesian optimization workflow
├── probability.py     # Utility functions for probability distributions and likelihood calculations
├── figures/           # Data and scripts used to reproduce the figures in the manuscript
├── requirements.txt   # Python package dependencies
└── README.md          # Repository documentation
```

## Requirements
The code was developed and tested with Python 3.11.
The required Python packages are listed in requirements.txt.
The main dependencies include:
- NumPy
- SciPy
- scikit-learn
- [bayesian-optimization](https://github.com/bayesian-optimization/bayesianoptimization)

As well as the molecular descriptor generation package introduced in the previous work, where the installation information can be found from this GitHub link: https://github.com/dmao1020/MolDes-GCT

To install the required dependencies:
```pip install -r requirements.txt```

## Bayesian optimization 
The Bayesian optimization scripts perform the optimization of the molecular properties considered in the manuscript
- Entropy
- ZPVE
- Normalized electronic energy ($E_{\mathrm{elec},n}$)