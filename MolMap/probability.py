import numpy as np
import math

def gaussian_pdf(x, mu, sigma):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-((x - mu)**2) / (2 * sigma**2))

def prob_atom_count(atom_type, value, atom_pdf_dict, sigma=4):
    likelihoods = np.array([
        gaussian_pdf(value, atom_pdf_dict["f_atom_inner_prod"][i], sigma)
        for i, a in enumerate(atom_pdf_dict["atom_type"])
        if a == atom_type
    ])

    p = 1 / len(likelihoods)
    posterior = (likelihoods * p) / np.sum(likelihoods * p)

    idx_sort = np.argsort(posterior)

    counts = np.array([
        atom_pdf_dict["atom_count"][i]
        for i, a in enumerate(atom_pdf_dict["atom_type"])
        if a == atom_type
    ])

    return counts[idx_sort], posterior[idx_sort]

