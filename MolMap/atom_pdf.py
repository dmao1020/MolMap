import numpy as np

# Dictionary for nuclear charge according to atom type (full periodic table)
nuclear_charge_dict = {
    "H": 1,  "He": 2,  "Li": 3,  "Be": 4,  "B": 5,   "C": 6,   "N": 7,   "O": 8,   "F": 9,   "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,  "S": 16,  "Cl": 17, "Ar": 18, "K": 19,  "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23,  "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39,  "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53,  "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
    "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74,  "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92,  "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
    "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110,
    "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
}
# Dictionary for the atom separation constant rho_Z in Eq.(14) for different atoms
rho_z = {
    "H":1.09, 
    "C":2,
    "N":1.43,
    "O":1.4,
    "F":1.35
}# 

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

def return_nu(atom_type: str = "H", 
              n_atoms: int = 1):
    if atom_type == "H" or atom_type == "F":
        if n_atoms < 4:
            theta =  n_atoms + 2
        else:
            theta = n_atoms
    elif atom_type in ["C", "N", "O"]:
        if n_atoms < 6:
            theta = 7
        else:
            theta = n_atoms
    return theta

def Mij_calc(atom_type,
             nu = 1,
             Z_mu = Zj_mean,
             ):
    Z = nuclear_charge_dict[atom_type]
    if atom_type == "H":
        k = 0.2
    else:
        k = 0.01
    return (Z * Z_mu)/ (rho_z[atom_type] * (1+(k*nu)))

def atom_pdf(
          atom_type: str,
          n_atoms: int, # \nu
        #   mol_size: int = 9,
          di: float = 10, 
          mu_power: float = 1, 
          var_power: float = 1, 
          x = np.linspace(-100, 300, 400),
          norm_stat: bool = False,
          return_pdf_stat: bool = True
          ):
    """Calculate hat{f}_{nu, Z} the estimated probability 
    distribution for nu atoms of a species with Z-nuclear charge 

    Args:
        atom_type (str): species of atom
        n_atoms (int): number of atoms (nu)
        mol_size (int, optional): size of the molecule. Defaults to 9.
        di (float, optional): weight parameter. Defaults to 10.
        mu_power (float, optional): power on mean. Defaults to 1.
        var_power (float, optional): power on variance. Defaults to 1.
        x (_type_, optional): list of x values. Defaults to np.linspace(-100, 300, 400).
        norm_stat (bool, optional): Statement on whether we should normalize the output pdf. Defaults to False.

    Returns:
        _type_: _description_
    """
    if norm_stat == True:
        norm_val = 1/n_atoms
    else:
        norm_val = 1
    Z = nuclear_charge_dict[atom_type]
    Mii = mu_calc(atom_type)
    
    cum_pdf = 0
    Mij_ls = []
    for i in range(n_atoms):
        # calculate theta
        theta = return_nu(atom_type, n_atoms)
        Mij = Mij_calc(atom_type, nu=i+1, Z_mu=Zj_mean) 
        Mij_ls.append(Mij)
        f_i = normalPdf(
            x = x,
            mean = Mii**mu_power, 
            variance = (theta * Mij)**(0.5*var_power)
        )
        cum_pdf += di*f_i
    if return_pdf_stat == True:
        return norm_val * cum_pdf, {"Mii": Mii, "Mij": max(Mij_ls)}
    else:
        return norm_val * cum_pdf
        # print (f"atom_type: {atom_type}")
        

# def atom_pdf(
#           atom_type: str,
#           n_atoms: int, # \nu
#           mol_size: int = 9,
#           di: float = 10, 
#           mu_power: float = 1, 
#           var_power: float = 1, 
#           x = np.linspace(-100, 300, 400),
#           norm_stat: bool = False
#           ):
#     """Calculate hat{f}_{nu, Z} the estimated probability 
#     distribution for nu atoms of a species with Z-nuclear charge 

#     Args:
#         atom_type (str): species of atom
#         n_atoms (int): number of atoms (nu)
#         di (float, optional): weight parameter. Defaults to 10.
#         mu_power (float, optional): power on mean. Defaults to 1.
#         var_power (float, optional): power on variance. Defaults to 1.
#         x (_type_, optional): list of x values. Defaults to np.linspace(-100, 300, 400).
#         norm_stat (bool, optional): Statement on whether we should normalize the output pdf. Defaults to False.

#     Returns:
#         _type_: _description_
#     """
#     if norm_stat == True:
#         norm_val = 1
#     else:
#         norm_val = 0
#     # Normalization constant if norm_stat == True
#     d = 1/mol_size
#     # print (f"atom_type: {atom_type}")
#     # Find nuclear charge for atom type
#     Zi = nuclear_charge_dict[atom_type]
#     # Calculate the Mii for Coulomb matrix, which is the mean of the probability distribution
#     Mii = mu_calc(atom_type)

#     # Calculate the denominator of \hat{M}_{ij}(Z,\nu)
#     if atom_type == "H":
#         Rij_mean = 1.09 * (1+(0.2*n_atoms))
#     elif atom_type in ["C", "N", "O", "F"]:
#         Rij_mean = rho_z[atom_type] * (1+(0.01*n_atoms))
    
#     # Decide value of theta in \hat{f}_{\nu, Z}
#     if atom_type in ["C", "N", "O"]:
#         if n_atoms < 6:
#             theta = 7
#         else:
#             theta = n_atoms
#     elif atom_type in ["H", "F"]:
#         if n_atoms < 4:
#             theta =  n_atoms + 2
#         else:
#             theta = n_atoms
#     # calculate estimated cumulative probability distribution \hat{f}_{\nu, Z}
#     cum_pdf = 0
#     for i in range(n_atoms):
#         sigma_estimate = theta * (Zi*Zj_mean/Rij_mean)
#         cum_pdf += (di*(d/d**norm_val))*normalPdf(x, Mii**mu_power, sigma_estimate**var_power)
#     return cum_pdf