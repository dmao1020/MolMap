# MolMap Package Summary
## Overview
This repository contains the code and data used to generate the results presented in the manuscript:
**Bayesian Optimization in Chemical Compound Sub-Spaces using
Low-Dimensional Molecular Descriptors** Yun-Wen Mao and Roman V Krems

The repository is intended to facilitate reproducibility of the results presented in the manuscript.
It inclludes the Bayesian optimization workflows, the inverse mapping algorithm, and data of the figures in the manuscript.

## Repository structure
. 
├── mapping.py # Mapping from descriptor vector to chemical formula
├── atom_pdf.py # Calculating the reference atomic probability distribution in Eq. (14) of the manuscript
├── figures/ # data used to generate figures
├── requirements.txt # Python dependencies 
└── README.md

## Requirements
The code was developed and tested with Python 3.11.
The required Python packages are listed in requirements.txt.
The main dependencies include:
- NumPy
- SciPy
- scikit-learn

As well as the molecular descriptor generation package introduced in the previous work, where the installation information can be found from this GitHub link: https://github.com/dmao1020/MolDes-GCT

To install the required dependencies:
```pip install -r requirements.txt```

## Bayesian optimization 
The Bayesian optimization scripts perform the optimization of the molecular properties considered in the manuscript
- Entropy
- ZPVE
- Normalized electronic energy (E_{\mathrm{elec, n}})