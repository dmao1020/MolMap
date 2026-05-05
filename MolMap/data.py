# data.py
import pandas as pd
import numpy as np

def load_qm9_atom_counts(path):
    return pd.read_csv(path)

def load_atom_pdf(path):
    return np.load(path, allow_pickle=True)[()]