import math
import time
import random
import numpy as np
from numpy import linalg as LA
import os
from bayes_opt import BayesianOptimization
import rdkit
from rdkit import Chem
import scipy
from scipy.spatial import distance
from .chemistry import return_kcalmol
from .boundary import atom_cum_pdf_calc
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import re
import itertools
class MolMap():
    def __init__(self,
                 GC_param_dict= {
                     "mu_power": 1.2,
                     "var_power": 0.7,
                     "di": 10,
                     "atom_var": 0.5,
                     "d_atom": 10,
                     "x_param": [-100, 300, 400],
                     "cum_pdf_norm_stat": False},
                atom_ls = ["H","C", "N", "O", "F"],
                prop_n = "s", 
                target_prop_val = None, 
                kcalmol_stat = True,
                max_abs_err = 100,
                top_ranks = 2,
                max_MW = 300.0,
                max_cid = 4000000,
                unit = "hartree"
        ):
        """
        Initialize the MolMap class with parameters for molecular mapping and optimization.

        Parameters:
        - GC_param_dict: Dictionary containing parameters for the Gershgorin circle theorem descriptor.
        - atom_ls: List of atom types to consider (default: ["H", "C", "N", "O", "F"]).
        - prop_n: Property name to optimize (default: "s").
        - target_prop_val: Target property value for optimization.
        - kcalmol_stat: Boolean indicating whether to convert output from Hartree to kcal/mol (default: True).
        - max_abs_err: Maximum absolute error for chemically invalid molecular structures (default: 100).
        - top_ranks: Number of top ranks to consider for atom count guesses (default: 2).
        - max_MW: Maximum molecular weight to consider (default: 300.0).
        - max_cid: Maximum compound ID to consider (default: 4000000).
        - unit: Unit of energy to use (default: "Hartree").
        """
        # BO suggested point
        # Descriptor in a dictionary
        self.kcalmol_stat = kcalmol_stat
        self.GC_param_dict = GC_param_dict
        self.max_abs_err = max_abs_err
        self.prop_n = prop_n
        self.target_prop_val = target_prop_val
        self.top_ranks = top_ranks
        self.atom_ls = atom_ls
        self.atom_count_keys = [f"{atom_i}" for atom_i in self.atom_ls]
        self.max_MW = max_MW
        self.max_cid = max_cid
        self.unit = unit

        # Derive constant that converts the energy unit
        # from Hartree to kcal/mol
        if self.kcalmol_stat == True:
            self.kcalmol_constant = return_kcalmol(self.unit)
        else:
            self.kcalmol_constant = 1

        
        """Dictionary summary 
        key0: 'atom_type' # The type of atom (H, C, N, O, F)
        key1: 'atom_count', # nu, i.e. number of atom of a specific item type
        key2: 'f_atom_inner_prod', # inner product <f_{atom}, \hat{f}_{nu, Z}>
        key3: 'cum_pdf'# cumulative probability distribution: \hat{f}_{nu, Z} 
        """
        self.atom_cum_pdf_dict = atom_cum_pdf_calc(
            di = self.GC_param_dict["di"], 
            mu_power = self.GC_param_dict["mu_power"], 
            var_power = self.GC_param_dict["var_power"], 
            x = np.linspace(-100, 300, 400),
            norm_stat = self.GC_param_dict["cum_pdf_norm_stat"],
            atom_var = self.GC_param_dict["atom_var"],
            atom_ls = self.atom_ls,
            max_MW = self.max_MW
        )  
    

    def ChemFormula(
            self, 
            probe_pt_des_dict
            ):
        """
        Generate a chemical formula based on the descriptor dictionary.

        Args:
            probe_pt_des_dict (dict): The descriptor dictionary.

        Returns:
            dict: The generated chemical formula.
        """
        atom_guess_dict = {}
        for i, atom_i in enumerate(self.atom_ls):
            des_val = probe_pt_des_dict['in_prod_f%s_fi'%(atom_i)]
            count_guess_prob, prob = self.prob_atom_count(atom_type = atom_i, 
                                                          value = des_val,
                                                          sigma=4)
            if atom_i == "C" or atom_i == "H":
                atom_guess_dict[atom_i] = [count_guess_prob[-(rank+1)] for rank in range(self.top_ranks)]
            else:
                atom_guess_dict[atom_i] = [count_guess_prob[-1]]
        return atom_guess_dict
    
    def find_cid_from_formula(
            self, 
            target_formula
            ):
        """Find CID (Compound Identification number) from the chemical formula.

        Args:
            target_formula (str): The chemical formula.

        Returns:
            list: a list of SMILES strings corresponding to the target chemical formula.
        """
        formula_idx = [idx for idx, val_i in enumerate(self.remain_des_prop_dict.values()) if val_i[-1] == target_formula]
        remain_smi_ls = list(self.remain_des_prop_dict.keys())
        if len(formula_idx) > 0:
            smi_ls = [remain_smi_ls[idx] for idx in formula_idx]
            return smi_ls
        else:
            return []
    
    def generate_hill_formula(
            self, 
            atom_counts
            ):
        """
        Generate a Hill-system chemical formula string from atom counts.
        
        Args:
            atom_counts (dict): dict like {'H':2, 'C':1, 'N':6, 'O':4}
        Returns: 
            str: Hill-system formula string, e.g. 'CH2N6O4'
        """
        parts = []
        if atom_counts.get('C', 0) > 0:
            c = atom_counts.get('C', 0)
            parts.append('C' if c == 1 else f'C{c}')
            h = atom_counts.get('H', 0)
            if h > 0:
                parts.append('H' if h == 1 else f'H{h}')
            others = sorted(k for k in atom_counts.keys() if k not in ('C', 'H') and atom_counts[k] > 0)
        else:
            others = sorted(k for k in atom_counts.keys() if atom_counts[k] > 0)
        for el in others:
            if el in ('C', 'H'):
                continue
            n = atom_counts[el]
            parts.append(el if n == 1 else f'{el}{n}')
        return ''.join(parts)


    def generate_conventional_formula(
            self, 
            atom_counts
            ):
        """
        Heuristic generator for conventional (human-friendly) formulas.

        Rules implemented:
        - If carbon is present: use Hill ordering (C, H, then others alphabetical).
        - If no carbon and the molecule is composed only of H and one other element X:
            * If X is in `hydrogen_first_set` (common hydrides like O, S, halogens) -> put H first (e.g. H2O, HCl, H2S).
            * Otherwise put X first (e.g. NH3, SiH4 -> SiH4).
        - For molecules with multiple element types and no carbon: list non-H elements alphabetically, put H at the end (common inorganic style).
        - Falls back to Hill ordering behavior when ambiguous.

        This is a heuristic and can be tuned by editing the two sets below.

        Args:
            atom_counts (dict): dict like {'H':2, 'C':1, 'N':6, 'O':4}  
        Returns:
            str: Conventional formula string, e.g. 'CH2N6O4' or 'H2O' or 'NH3'.
        """
        if not atom_counts or sum(atom_counts.values()) == 0:
            return ""

        # canonicalize counts to only positive ints
        counts = {el: int(n) for el, n in atom_counts.items() if int(n) > 0}

        # If carbon present, use Hill
        if counts.get('C', 0) > 0:
            return self.generate_hill_formula(counts)

        hydrogen_first_set = {'O', 'S', 'Se', 'Te', 'F', 'Cl', 'Br', 'I'}
        hydrogen_last_set = {'N', 'Si', 'B', 'Ge', 'P'}

        elements = sorted(k for k in counts.keys() if k != 'H')
        h_count = counts.get('H', 0)

        # Only H and single other element
        if h_count > 0 and len(elements) == 1:
            x = elements[0]
            if x in hydrogen_first_set:
                # H before X
                parts = []
                parts.append('H' if h_count == 1 else f'H{h_count}')
                parts.append(x if counts[x] == 1 else f'{x}{counts[x]}')
                return ''.join(parts)
            else:
                # X before H
                parts = []
                parts.append(x if counts[x] == 1 else f'{x}{counts[x]}')
                if h_count > 0:
                    parts.append('H' if h_count == 1 else f'H{h_count}')
                return ''.join(parts)

        # Multiple non-H elements (or no H): list non-H alphabetically, put H at end
        parts = []
        for el in elements:
            parts.append(el if counts[el] == 1 else f'{el}{counts[el]}')
        if h_count > 0:
            parts.append('H' if h_count == 1 else f'H{h_count}')
        return ''.join(parts)
    
    def search_mol_wt_chem_formula(
            self, 
            chem_formula_dict
            ):
        """
        Search for molecules based on the chemical formula derived from the descriptor dictionary.

        Args:
            chem_formula_dict (dict): Dictionary containing atom counts.
        
        Returns:
            list: List of SMILES strings that match the chemical formula.
        """
        # Generate the target chemical formula using the conventional formula generator
        # print (f"chem_formula_dict: {chem_formula_dict}")
        # C_count = chem_formula_dict["C_count"]
        # H_count = chem_formula_dict["H_count"]
        # target_formula = ""
        # if C_count != 0:
        #     if C_count == 1:
        #         target_formula += "C"
        #     else:
        #         target_formula += f"C{str(C_count)}"
        # if H_count != 0:
        #     if H_count == 1:
        #         target_formula += "H"
        #     else:
        #         target_formula += f"H{str(H_count)}"
        # remain_atom_ls = [atom_i for atom_i in self.atom_ls if atom_i not in ["C", "H"]]
        # # print ("remain_atom_ls:", remain_atom_ls)
        # for atom in remain_atom_ls:
        #     count = chem_formula_dict["%s_count"%atom]
        #     if count != 0:
        #         if count == 1: 
        #             target_formula += f"{atom}"
        #         else:
        #             target_formula += f"{atom}{str(count)}"
        # get CID from chemical formula
        # compounds = pcp_find_cid_from_formula(target_formula)
        # if compounds is None:
        #     print (f"No compounds found for chemical formula: {target_formula}")
        #     return []
        # else: 
        #     print (f"Searching for molecules with chemical formula: {target_formula}")
        #     filtered_smiles = [compound.smiles for compound in compounds if 
        #                     "+" not in compound.smiles and
        #                     "-" not in compound.smiles and
        #                     "[1H" not in compound.smiles and
        #                     "[2H" not in compound.smiles and
        #                      "[3H" not in compound.smiles and
        #                     not re.search(r"\[(?!12)\d+C", compound.smiles) and
        #                     not re.search(r"\[(?!14)\d+N", compound.smiles) and
        #                     not re.search(r"\[(?!18)\d+O", compound.smiles)
        #                         ]
        target_formula = self.generate_conventional_formula(chem_formula_dict)
        # no pcp way
        filtered_smiles = self.find_cid_from_formula(target_formula)
        return filtered_smiles

    def Des2MolMap(
            self,
            probe_pt_des_dict, 
            remain_des_prop_dict
            ):
        """
        Maps descriptors to molecules.

        Args:
            probe_pt_des_dict (dict): The descriptor dictionary for the probe point.
            remain_des_prop_dict (dict): The descriptor dictionary for the remaining points.

        Returns:
            None: Updates the class attributes with the best matching molecule and its properties.
        """
        self.remain_des_prop_dict = remain_des_prop_dict
        remain_smi_ls = [key for key in self.remain_des_prop_dict.keys() if key != "GC_des_parameters" and key != "SMILES"]
        atom_guess_dict = self.ChemFormula(probe_pt_des_dict)
        
        # Generate all combinations by taking one element from each list
        # Build the list of lists in the same order as self.atom_ls so the
        # resulting tuples align with self.atom_count_keys.
        lists_for_product = [atom_guess_dict[atom] for atom in self.atom_ls]
        all_combinations = list(itertools.product(*lists_for_product))
        smiles_ls = []
        for comb_i in all_combinations:
            comb_dict = dict(zip(self.atom_count_keys, comb_i))
            filtered_smi_i = self.search_mol_wt_chem_formula(comb_dict)
            if len(filtered_smi_i) > 0:
                smiles_ls+=list(filtered_smi_i)
            
        self.smiles_ls = list(smiles_ls)
        remain_smi_ls = [key for key in self.remain_des_prop_dict.keys() if key != "GC_des_parameters" and key != "SMILES"]
        if len(self.smiles_ls)!= 0:
            cadidate_ls = []
            cadidate_dict = {}
            filtered_smi = []
            prop_ls = []
            # Filter out the SMILES that are already in the training set
            for smi_i in self.smiles_ls:
                if smi_i in remain_smi_ls:
                    cadidate_ls.append(list(self.remain_des_prop_dict[smi_i][0].values()))
                    cadidate_dict[smi_i] = [self.remain_des_prop_dict[smi_i][0], 
                                            self.remain_des_prop_dict[smi_i][1][self.prop_n]]
            cadidate_arr = np.array(cadidate_ls)
            target_val_vector = list(probe_pt_des_dict.values())#list(target_point.values())
        else:
            cadidate_arr =[]
        # If there are candidate molecules, find the closest one to the target descriptor vector
        if np.shape(cadidate_arr)[0] != 0:
            print (f"Found {np.shape(cadidate_arr)[0]} candidate molecules for the target descriptor vector.")
            tree = scipy.spatial.KDTree(cadidate_arr)
            min_dist, min_idx =  tree.query(target_val_vector)
            self.min_dist = min_dist
            self.min_idx = min_idx
            test_dist = math.dist(cadidate_arr[min_idx,:],target_val_vector)
            best_smiles = list(cadidate_dict.keys())[self.min_idx]
            best_des_dict = cadidate_dict[best_smiles][0]
            best_prop = cadidate_dict[best_smiles][1]
            remain_des_prop_dict.pop(best_smiles)
            
            abs_err = abs(best_prop*self.kcalmol_constant-self.target_prop_val*self.kcalmol_constant)
            self.abs_err = abs_err
            self.best_des_dict = best_des_dict
            self.remain_des_prop_dict = remain_des_prop_dict
            self.composition_guess = all_combinations
            self.best_smiles = best_smiles
        else:
            print ("No candidate molecules found for the target descriptor vector.")
            # If no candidate molecules are found, set the best SMILES to "None" and use the maximum absolute error
            self.abs_err = self.max_abs_err
            self.best_des_dict = probe_pt_des_dict
            self.remain_des_prop_dict = remain_des_prop_dict
            self.composition_guess = all_combinations
            self.best_smiles = "None"
        
            
    def gaussian_pdf(
            self, 
            x, 
            mu, 
            sigma
            ):
        """
        Calculate the Gaussian probability density function (PDF) value for a given x, mean (mu), and standard deviation (sigma).

        Args:
            x (float): The value for which to calculate the PDF.
            mu (float): The mean of the Gaussian distribution.
            sigma (float): The standard deviation of the Gaussian distribution.
        Returns:
            float: The PDF value at x.
        """
        return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-((x - mu)**2) / (2 * sigma**2))

    def prob_atom_count(
            self, 
            atom_type, 
            value, 
            sigma=4
            ):
        """
        Calculate the posterior probabilities of atom counts for a given atom type based on the inner product value.

        Args:
            atom_type (str): The type of atom (e.g., "C", "H", "O").
            value (float): The inner product value for the atom type.
            sigma (float): The standard deviation for the Gaussian PDF (default: 4).    

        Returns:
            tuple: A tuple containing two lists:
                - List of atom counts sorted by their posterior probabilities (from smallest to largest).
                - List of corresponding posterior probabilities sorted in the same order.
        """
        # Likelihoods
        # print (f"atom_type: {atom_type}, value: {value}, sigma: {sigma}")
        # print (f"self.atom_cum_pdf_dict['atom_type']: {self.atom_cum_pdf_dict['atom_type']}")
        likelihoods = np.array([self.gaussian_pdf(
            value, 
            self.atom_cum_pdf_dict["f_atom_inner_prod"][idx], 
            sigma) 
            for idx, atom_i in enumerate(self.atom_cum_pdf_dict["atom_type"]) 
            if atom_i == atom_type])
        # print (f"likelihoods: {likelihoods}")
        # print (f"len(likelihoods): {len(likelihoods)}")
        if len(likelihoods) == 0:
            return [], []
        else:
            # prior probability
            p = 1/(len(likelihoods))
            posterior_denominator  = np.sum(np.array([f*p for f in likelihoods]))
            posterior = np.array([(f * p) / (posterior_denominator) for f in likelihoods])
            # sort calculated posterior from smallest to biggest
            idx_sort = np.argsort(posterior)
            # get the corresponding atom counts and posterior probabilities in sorted order
            inner_prod_count_arr = np.array([self.atom_cum_pdf_dict ["atom_count"][idx]
                                            for idx, atom_i in enumerate(self.atom_cum_pdf_dict["atom_type"]) 
                                            if atom_i == atom_type])

            # print (f"inner_prod_count_arr: {inner_prod_count_arr}")
            # print (f"posterior: {posterior}")
            return [inner_prod_count_arr[i] for i in idx_sort], [posterior[i] for i in idx_sort]
    

        