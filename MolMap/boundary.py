import numpy as np
from .atom_pdf import atom_pdf
from MolDes import GCT_util, CM_util # importing MolDes package
import matplotlib.pyplot as plt
import math

atomic_weight_dict = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.90447,
    "B": 10.81,
    "Si": 28.085
}

def solve_n_for_alkane(target: float = 300.0):
    mw_C, mw_H = atomic_weight_dict["C"], atomic_weight_dict["H"]
    # f(n) = n*mw_C + (2n+2)*mw_H = n*(mw_C + 2*mw_H) + 2*mw_H
    a = mw_C + 2*mw_H
    b = 2*mw_H
    n = (target - b) / a
    return n
def solve_n_for_alcohol(target: float = 300.0):
    mw_C, mw_H, mw_O = atomic_weight_dict["C"], atomic_weight_dict["H"], atomic_weight_dict["O"]
    a = mw_C + 2 * mw_H +2 * mw_O 
    b = 2 * mw_H + 2 * mw_O
    n = (target - b) / a
    return n
# def solve_n_for_hydroxide_alkene(target=300.0, mw_C=12.011, mw_O=15.999):
#     # f(n) = n*mw_C + (2n+2)*mw_H = n*(mw_C + 2*mw_H) + 2*mw_H
#     a = mw_C + 2*mw_O
#     n = target / a
#     return n
# def solve_n_for_ketone(target=300.0, mw_C=12.011, mw_O=15.999):
#     # f(n) = n*mw_C + (2n+2)*mw_H = n*(mw_C + 2*mw_H) + 2*mw_H
#     a = mw_C + mw_O
#     b = 2 * mw_O
#     n = (target - b) / a
#     return n
def solve_n_for_amine(target: float = 300.0):
    mw_H, mw_N = atomic_weight_dict["H"], atomic_weight_dict["N"]
    # f(n) = n*mw_C + (2n+1)*mw_H + mw_N = n*(mw_C + 2*mw_H) + (mw_H + mw_N)
    a = mw_N + mw_H
    b = 2*mw_H
    n = (target - b) / a
    return n

def max_atom_counts(
        max_MW: float = 300,
        atomic_ls = ["H", "C", "N", "O", "F"],
        ) -> dict:
    """
    Calculate the maximum number of atoms of each type
    that can be present in a molecule with
    a given maximum molecular weight (MW).
    This is done by solving for n in the molecular weight
    equations for different types of molecules
    (alkanes, alcohols, amines, etc.) and using
    the atomic weights to determine the maximum number of
    atoms of each type.
    """
    estmated_max_counts = {}
    # print (estmated_max_counts)

    for atom_type in atomic_ls:
        atomic_weight = atomic_weight_dict[atom_type]
        if atom_type == "C": 
            max_n_atoms = max_MW / atomic_weight
            # print (f"Maximum number of {atom_type} atoms for MW <= {max_MW}: {max_n_atoms}")
            estmated_max_counts[atom_type] = math.ceil(max_n_atoms)
        # elif atom_type == "O":
        #     ketone_n = solve_n_for_ketone(target=max_MW)
        #     print (f"Ketone n for MW <= {max_MW}: {ketone_n}")
        #     max_n_atoms = math.ceil(ketone_n) + 2
        #     print (f"Maximum number of {atom_type} atoms for MW <= {max_MW}: {max_n_atoms}")
        #     estmated_max_counts[atom_type] = math.ceil(max_n_atoms)
        elif atom_type == "O":
            alcohol_n = solve_n_for_alcohol(target=max_MW)
            # print (f"Alcohol n for MW <= {max_MW}: {alcohol_n}")
            max_n_atoms = 2 * alcohol_n + 2
            # print (f"Maximum number of {atom_type} atoms for MW <= {max_MW}: {max_n_atoms}")
            estmated_max_counts[atom_type] = math.ceil(max_n_atoms)
        elif atom_type == "H":
            # alkane C_n H_(2n+2) has MW = n*12.011 + (2n+2)*1.008 = n*(12.011 + 2*1.008) + 2*1.008 = n*14.027 + 2.016, so we can rearrange to find max n for a given MW
            alkane_n = solve_n_for_alkane(target=max_MW)
            max_n_atoms = 2 * alkane_n + 2
            # print (f"Maximum number of {atom_type} atoms in an alkane with MW <= {max_MW}: {max_n_atoms}")
            estmated_max_counts[atom_type] = math.ceil(max_n_atoms)
        elif atom_type == "N":
            # amine C_n H_(2n+1) has MW = n*12.011 + (2n+1)*1.008 + 14.007 = n*(12.011 + 2*1.008) + (1.008 + 14.007)
            max_n_atoms = solve_n_for_amine(target=max_MW)
            # print (f"Maximum number of {atom_type} atoms in an amine with MW <= {max_MW}: {max_n_atoms}")
            estmated_max_counts[atom_type] = math.ceil(max_n_atoms)
        else:
            print (f"Atom type {atom_type} not specifically handled, using a general estimate based on atomic weight.")
    return estmated_max_counts
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Draw
def alkane_cm(n_carbon):
    # 1. Create molecule object from SMILES

    smiles = "C" * n_carbon
    print (f"SMILES for alkane with {n_carbon} carbon atoms: {smiles}")
    mol = Chem.MolFromSmiles(smiles)

    # 2. Add Hydrogens (crucial for accurate Coulomb interactions)
    mol = Chem.AddHs(mol)
    # # Method 1: Generate an image object
    # img = Draw.MolToImage(mol)
    # img.show()


    # 3. Generate a 3D conformation
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    # 4. Optimize the 3D structure (optional but recommended)
    AllChem.UFFOptimizeMolecule(mol)

    # 5. Calculate the Coulomb Matrix
    # This returns a list of matrices (one for each conformer)
    coulomb_matrices = rdMolDescriptors.CalcCoulombMat(mol)

    # Access the first matrix
    cm = list(coulomb_matrices[0])
    return coulomb_matrices

def boundary_calc(
        di: float = 10, 
        mu_power: float = 1, 
        var_power: float = 1, 
        x = np.linspace(-100, 300, 400),
        norm_stat: bool = False,
        atom_var: float = 0.5,
        atom_ls: list = ["H", "C", "N", "O", "F"],
        data_atom_dict: dict = {"H":[1, 20], "C":[0, 9], "N":[0, 9], "O":[0, 9], "F":[0, 9]},
        max_MW: float = 300.0
):
    """Calculate the boundary of the cumulative distribution function for each atom type and number of atoms. 
    This is used to determine the range of the cumulative distribution function for each atom type and number of atoms, which is then used to calculate the molecular descriptor.

    Returns:
        dict: A dictionary containing the boundary of the cumulative distribution function for each atom type and number of atoms.
    """
    boundary_dict = {}
    GCT_des_util = GCT_util.GCT_util(
            mu_power = mu_power,
            var_power = var_power,
            di = di,
            x = x,
            cum_pdf_norm_stat = norm_stat,
            atom_var = atom_var,
            atom_ls = atom_ls
        )
    atom_dict = GCT_des_util.f_atom_calc()
    boundary_dict = {}
    for atom_type, n_atoms_range in data_atom_dict.items():
        # print (f"Calculating boundary for atom type: {atom_type}")
        boundary_dict[atom_type] = []
        for n_atoms in n_atoms_range:
            if n_atoms == 0:
                des_val = 0
                boundary_dict[atom_type].append(des_val)
                # print (f"Descriptor value for {n_atoms} {atom_type} atoms: {des_val}")
            else:
                # print (f"Calculating boundary for {n_atoms} {atom_type} atoms")
                pdf, pdf_stats = atom_pdf(
                    atom_type=atom_type,
                    n_atoms=n_atoms, # nu
                    # mol_size = 30,
                    di = di,
                    mu_power = mu_power,
                    var_power = var_power,
                    x=x,
                    norm_stat = norm_stat
                )
                print (f"PDF stats for {n_atoms} {atom_type} atoms: {pdf_stats}")
                des_val = np.inner(atom_dict[atom_type], pdf)
                # print ("Atomic distribution function values: ", atom_dict[atom_type])
                # print ("Estimated probability distribution values: ", pdf)
                # fig, ax = plt.subplots()
                # ax.plot(x, pdf, label=f"Estimated PDF for {n_atoms} {atom_type} atoms")
                # ax.plot(x, atom_dict[atom_type], label=f"Atomic distribution function for {n_atoms} {atom_type} atoms")
                # ax.set_xlabel("x")
                # ax.set_ylabel("Estimated PDF")
                # ax.set_title(f"Estimated PDF for {n_atoms} {atom_type} atoms")
                # ax.legend()
                # plt.show()
                # print (f"Descriptor value for {n_atoms} {atom_type} atoms: {des_val}")
                boundary_dict[atom_type].append(des_val)
        # print ("\n")
        
    # Estimate the upper descriptor boundary for the AUC descriptor with the larget alkene
    alkane_n = solve_n_for_alkane(target = max_MW)
    print 
    alkene_cum_pdf = atom_pdf(
            atom_type="C",
            n_atoms = int(math.ceil(alkane_n)),
            di = di,
            mu_power = mu_power,
            var_power = 2,
            x=x,
            norm_stat = norm_stat,
            return_pdf_stat = False
        ) + atom_pdf(
            atom_type="H",
            n_atoms = 2*int(math.ceil(alkane_n)) + 2,
            di = di,
            mu_power = mu_power,
            var_power = 2,
            x=x,
            norm_stat = norm_stat,
            return_pdf_stat = False
        )
    auc =  np.trapezoid(alkene_cum_pdf, dx=1)
    print (f"alkene_cum_pdf stats: {auc}")
    boundary_dict["auc_mij_sq"] = [0, auc]

    # cm_ev1 = 



    # print (f"Boundary dict: {boundary_dict}")
    return boundary_dict