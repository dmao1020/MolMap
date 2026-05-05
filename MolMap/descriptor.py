import numpy as np
import math
# Dictionary for nuclear charge according to atom type
nuclear_charge_dict = {"H":1, 
                     "C":6,
                     "N":7,
                     "O":8,
                     "F":9
                     }
# Dictionary for the atom separation constant rho_Z in Eq.(14) for different atoms
rho_z = {
    "H":1.09, 
    "C":2,
    "N":1.43,
    "O":1.4,
    "F":1.35
}

# Caclulate Z_mu
Zj_mean = np.mean(np.array([val for key, val in nuclear_charge_dict.items()]))
# print (f"Zj_mean:{Zj_mean}")
# Dictionary for chemical bond
# bond_dict = {6: [1., 1.], # C-H:1.09, C-C: 1.54
#              1: 1.09, # C-H
#              9: 1.32, # C-F
#              8: 1, # O-H
#              7: 1 # N-H
#              }

def normalPdf(
        x: float, 
        mean: float, 
        variance: float
        ):
    """Calculate the Normal distribution."""
    return (1/np.sqrt(2*np.pi*variance))*np.exp(-0.5*(x-mean)**2/variance)
def mu_calc(
        atom_type: str
        )-> float:
    """Calculate Mii from Z, i.e. diagnoal terms of Coulombm matrices"""
    return 0.5*(nuclear_charge_dict[atom_type]**2.4)

def atom_pdf(
          atom_type: str,
          n_atoms: int, # \nu
          mol_size: int = 9,
          di: float = 10, 
          mu_power: float = 1, 
          var_power: float = 1, 
          x = np.linspace(-100, 300, 400),
          norm_stat: bool = False
          ):
    """Calculate \hat{f}_{\nu, Z} the estimated probability 
    distribution for \nu atoms of a species with Z-nuclear charge 

    Args:
        atom_type (str): species of atom
        n_atoms (int): number of atoms (\nu)
        di (float, optional): weight parameter. Defaults to 10.
        mu_power (float, optional): power on mean. Defaults to 1.
        var_power (float, optional): power on variance. Defaults to 1.
        x (_type_, optional): list of x values. Defaults to np.linspace(-100, 300, 400).
        norm_stat (bool, optional): Statement on whether we should normalize the output pdf. Defaults to False.

    Returns:
        _type_: _description_
    """
    if norm_stat == True:
        norm_val = 1
    else:
        norm_val = 0
    # Normalization constant if norm_stat == True
    d = 1/mol_size
    # print (f"atom_type: {atom_type}")
    # Find nuclear charge for atom type
    Zi = nuclear_charge_dict[atom_type]
    # Calculate the Mii for Coulomb matrix, which is the mean of the probability distribution
    Mii = mu_calc(atom_type)

    # Calculate the denominator of \hat{M}_{ij}(Z,\nu)
    if atom_type == "H":
        Rij_mean = 1.09 * (1+(0.2*n_atoms))
    elif atom_type in ["C", "N", "O", "F"]:
        Rij_mean = rho_z[atom_type] * (1+(0.01*n_atoms))
    
    # Decide value of theta in \hat{f}_{\nu, Z}
    if atom_type in ["C", "N", "O"]:
        if n_atoms < 6:
            theta = 7
        else:
            theta = n_atoms
    elif atom_type in ["H", "F"]:
        if n_atoms < 4:
            theta =  n_atoms + 2
        else:
            theta = n_atoms
    # calculate estimated cumulative probability distribution \hat{f}_{\nu, Z}
    cum_pdf = 0
    for i in range(n_atoms):
        sigma_estimate = theta * (Zi*Zj_mean/Rij_mean)
        cum_pdf += (di*(d/d**norm_val))*normalPdf(x, Mii**mu_power, sigma_estimate**var_power)
    return cum_pdf



def gaussian_pdf(
        x: float, 
        mu: float, 
        sigma: float
        ) -> float:
    """Calculate the Gaussian probability density function."""
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-((x - mu)**2) / (2 * sigma**2))

def prob_atom_count(
        atom_type: str, 
        value: float, 
        atom_cum_pdf_dict: Dict, 
        sigma: float = 4
        ) -> Tuple[List[int], List[float]]:
    """Estimate atom counts based on descriptor values using Bayesian inference."""
    # Likelihoods
    likelihoods = np.array([
        gaussian_pdf(value, atom_cum_pdf_dict["f_atom_inner_prod"][idx], sigma) 
        for idx, atom_i in enumerate(atom_cum_pdf_dict["atom_type"]) 
        if atom_i == atom_type
    ])

    # prior probability
    p = 1/(len(likelihoods))
    posterior_denominator  = np.sum(np.array([f*p for f in likelihoods]))
    posterior = np.array([(f * p) / (posterior_denominator) for f in likelihoods])
    # sort calculated posterior from smallest to biggest
    idx_sort = np.argsort(posterior)
    inner_prod_count_arr = np.array([atom_cum_pdf_dict["atom_count"][idx]
                                    for idx, atom_i in enumerate(atom_cum_pdf_dict["atom_type"]) 
                                    if atom_i == atom_type])
    # return the inner product array
    return [inner_prod_count_arr[i] for i in idx_sort], [posterior[i] for i in idx_sort]
