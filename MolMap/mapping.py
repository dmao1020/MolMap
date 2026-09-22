import math
import time
import random
from pathlib import Path
import numpy as np
from numpy import linalg as LA
import os
import pubchempy as pcp
from bayes_opt import BayesianOptimization
import rdkit
from rdkit import Chem
import scipy
from scipy.spatial import distance
from .chemistry import return_kcalmol, Coulomb_matrix
from .boundary import atom_cum_pdf_calc
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import re
import itertools

from MolDes import GCT_util, CM_util # importing MolDes package


# Retrieve the path to the PM6 CID file
_PM6_CIDS_PATH = (
    Path(__file__).resolve().parent
    / "optimization"
    / "data"
    / "pubchemQC"
    / "pm6_cids.npy"
)
if not _PM6_CIDS_PATH.is_file():
    raise FileNotFoundError(f"PM6 CID file not found: {_PM6_CIDS_PATH}")

_pm6_cids = np.load(_PM6_CIDS_PATH, allow_pickle=True)
if _pm6_cids.shape == ():
    _pm6_cids = _pm6_cids.item()
PM6_CID_SET = {int(cid) for cid in _pm6_cids}

import pandas as pd
# Retrieve the path to the QM9 data directory
_QM9_DATA_PATH = (
    Path(__file__).resolve().parent
    / "optimization"
    / "data"
    / "qm9"
    /"qm9_n_atom_df.csv"
)
if not _QM9_DATA_PATH.is_file():
    raise FileNotFoundError(f"QM9 data file not found: {_QM9_DATA_PATH}")
qm9_n_atom_df = pd.read_csv(_QM9_DATA_PATH)


_PUBCHEM_RETRYABLE_ERRORS = tuple(
    error_type
    for error_type in (
        TimeoutError,
        getattr(pcp, "ServerBusyError", None),
        getattr(pcp, "TimeoutError", None),
    )
    if isinstance(error_type, type)
)
_PUBCHEM_NO_RESULT_ERRORS = tuple(
    error_type
    for error_type in (getattr(pcp, "BadRequestError", None),)
    if isinstance(error_type, type)
)

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
                max_cid = 121494105,
                unit = "hartree",
                pcp_method = False,
                extractor = None
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
        self.pcp_method = pcp_method
        self.extractor = extractor

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
        self.GC_des_utility = GCT_util.GCT_util(x = np.linspace(self.GC_param_dict["x_param"][0], 
                                            self.GC_param_dict["x_param"][1], 
                                            self.GC_param_dict["x_param"][2]),
                                        cum_pdf_norm_stat=[self.GC_param_dict["cum_pdf_norm_stat"]],
                                        mu_power=self.GC_param_dict["mu_power"], 
                                        var_power=self.GC_param_dict["var_power"], 
                                        di=self.GC_param_dict["di"],
                                        atom_var=self.GC_param_dict["atom_var"],
                                        d_atom=[self.GC_param_dict["d_atom"]],
                                        atom_ls=self.atom_ls
                                        )  
        self.GC_t2_des_utility = GCT_util.GCT_util(x = np.linspace(self.GC_param_dict["x_param"][0], 
                                            self.GC_param_dict["x_param"][1], 
                                            self.GC_param_dict["x_param"][2]),
                                        cum_pdf_norm_stat=[self.GC_param_dict["cum_pdf_norm_stat"]],
                                        mu_power=self.GC_param_dict["mu_power"], 
                                        var_power=2, 
                                        di=self.GC_param_dict["di"],
                                        atom_var=self.GC_param_dict["atom_var"],
                                        d_atom=[self.GC_param_dict["d_atom"]],
                                        atom_ls=self.atom_ls
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
        print (f"Searching for SMILES with chemical formula: {target_formula}")
        formula_idx = [idx for idx, val_i in enumerate(self.remain_des_prop_dict.values()) if val_i[-1] == target_formula]
        remain_smi_ls = list(self.remain_des_prop_dict.keys())
        if len(formula_idx) > 0:
            smi_ls = [remain_smi_ls[idx] for idx in formula_idx]
            return smi_ls
        else:
            return []
    def find_smi_from_formula(
            self, 
            target_formula
            ):
        """Find SMILES from the chemical formula.
        """
        print (f"Searching for SMILES with chemical formula: {target_formula}")

    
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
            chem_formula_dict,
            ):
        """
        Search for molecules based on the chemical formula derived from the descriptor dictionary.

        Args:
            chem_formula_dict (dict): Dictionary containing atom counts.
        
        Returns:
            list: List of SMILES strings that match the chemical formula.
        """
        target_formula = self.generate_conventional_formula(chem_formula_dict)
        if not self.pcp_method:
            # ============= no pcp way =============
            filtered_smiles = self.find_cid_from_formula(target_formula)
            return filtered_smiles
        else:
            filtered_molecules= {"SIMILES": [], "CID": []}
            # ============= pcp way =============
            if not hasattr(self, "_pubchem_formula_cache"):
                self._pubchem_formula_cache = {}
            if target_formula in self._pubchem_formula_cache:
                return self._pubchem_formula_cache[target_formula]
            print (f"Searching PubChem for formula: {target_formula} using PubChemPy...")

            pubchem_compounds = None
            for attempt in range(4):
                try:
                    pubchem_compounds = pcp.get_compounds(target_formula, "formula")
                    break
                except _PUBCHEM_NO_RESULT_ERRORS:
                    print(f"No PubChem CID found for formula {target_formula}.")
                    self._pubchem_formula_cache[target_formula] = []
                    return filtered_molecules
                except _PUBCHEM_RETRYABLE_ERRORS:
                    if attempt == 3:
                        print(
                            f"PubChem unavailable for formula {target_formula}; "
                            "skipping this formula."
                        )
                        self._pubchem_formula_cache[target_formula] = []
                        return filtered_molecules
                    time.sleep(min(30, 2**attempt) + random.uniform(0, 1))

            
            
            for compound in pubchem_compounds:
                cid = int(compound.cid)
                if cid in PM6_CID_SET:
                    if compound.smiles is not None:
                        filtered_molecules["SIMILES"].append(compound.smiles)
                        filtered_molecules["CID"].append(cid)
            self._pubchem_formula_cache[target_formula] = filtered_molecules
            return filtered_molecules
    def get_descriptors_from_cid(
            self, 
            cid
            ):
        """_summary_

        Args:
            cid (_type_): _description_
        """

        d = self.extractor.extract_one(cid)
        # Extract the property value from docker extracted data
        prop_val = float(d.alpha_gap)

        # Exract geometry from docker extracted data
        coord_array = np.array(d.coords)
        element = d.elements
        z_ls = element["number"]
        # Compute the Coulomb matrix for the molecule
        des_dict = {
            "cm_mu": [],
            "cm_sigma": [],
            "cm_ev1": [],
            "auc_mij_sq": [],
        }
        for atom in self.atom_ls:
            des_dict[f"in_prod_f{atom}_fi"] = []
        CM = Coulomb_matrix(coord_array, z_ls)
        CM_w = np.linalg.eigvalsh(CM)
        des_dict["cm_mu"].append(float(np.mean(CM_w)))
        des_dict["cm_sigma"].append(float(np.std(CM_w)))
        des_dict["cm_ev1"].append(float(CM_w[-1]))

        auc_mij_sq = self.GC_t2_des_utility.CM2auc_calc(CM)
        des_dict["auc_mij_sq"].append(float(auc_mij_sq))
        f_atom = self.GC_des_utility.CM2atom_pdf_calc(CM)

        # print (f"sq_AUC: {auc_mij_sq}, \n f_atom: {f_atom}")
        for i, atom in enumerate(self.atom_ls):
            des_dict[f"in_prod_f{atom}_fi"].append(float(f_atom[i]))
        return des_dict, prop_val
        
    def Des2MolMap_pm6(
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
        # if self.pcp_method == True:
        #     pass
        # else:
        #     remain_smi_ls = [key for key in self.remain_des_prop_dict.keys() if key != "GC_des_parameters" and key != "SMILES"]
        atom_guess_dict = self.ChemFormula(probe_pt_des_dict)
        
        # Generate all combinations by taking one element from each list
        # Build the list of lists in the same order as self.atom_ls so the
        # resulting tuples align with self.atom_count_keys.
        lists_for_product = [atom_guess_dict[atom] for atom in self.atom_ls]
        all_combinations = list(itertools.product(*lists_for_product))
        smiles_ls = []
        cids_ls = []
        for comb_i in all_combinations:
            comb_dict = dict(zip(self.atom_count_keys, comb_i))
            filtered_molecules = self.search_mol_wt_chem_formula(comb_dict)
            print (f"Found {len(filtered_molecules['SIMILES'])}.")
            # print (f"comb_dict: {comb_dict}, filtered_molecules: {filtered_molecules}")
            if len(filtered_molecules["SIMILES"]) > 0:
                smiles_ls+=list(filtered_molecules["SIMILES"])
                cids_ls+=list(filtered_molecules["CID"])
            
        self.smiles_ls = list(smiles_ls)
        self.cids_ls = list(cids_ls)
        
        if len(self.smiles_ls)!= 0:
            target_val_vector = [probe_pt_des_dict[key] for key in probe_pt_des_dict.keys()]
            candidate_ls = []
            candidate_dict = {}
            for idx, cid_i in enumerate(self.cids_ls):
                des_i, prop_val= self.get_descriptors_from_cid(cid_i)
                des_list = [des_i[key][0] for key in probe_pt_des_dict.keys()]
                # print (f"des_list: {des_list}, prop_val: {prop_val}")
                candidate_ls.append(des_list)
                candidate_dict[self.smiles_ls[idx]] = [des_i, prop_val]
            candidate_arr = np.array(candidate_ls)
        else:
            candidate_dict =[]
            candidate_arr = []
        # print (f"candidate_arr:{candidate_arr}")
        # print (f"candidate_arr size: {np.shape(candidate_arr)}")
        # If there are candidate molecules, find the closest one to the target descriptor vector
        if np.shape(candidate_arr)[0] != 0:
            # print (f"Found {np.shape(candidate_arr)[0]} candidate molecules for the target descriptor vector.")
            tree = scipy.spatial.KDTree(candidate_arr)
            min_dist, min_idx =  tree.query(target_val_vector)
            self.min_dist = min_dist
            self.min_idx = min_idx
            test_dist = math.dist(candidate_arr[min_idx,:],target_val_vector)
            best_smiles = list(candidate_dict.keys())[self.min_idx]
            best_des_dict = candidate_dict[best_smiles][0]
            best_prop = candidate_dict[best_smiles][1]
            if self.pcp_method == False:
                remain_des_prop_dict.pop(best_smiles)
            
            abs_err = abs(best_prop*self.kcalmol_constant-self.target_prop_val*self.kcalmol_constant)
            self.abs_err = abs_err
            self.best_des_dict = best_des_dict
            if self.pcp_method == False:
                self.remain_des_prop_dict = remain_des_prop_dict
            self.composition_guess = all_combinations
            self.best_smiles = best_smiles
        else:
            # print ("No candidate molecules found for the target descriptor vector.")
            # If no candidate molecules are found, set the best SMILES to "None" and use the maximum absolute error
            self.abs_err = self.max_abs_err
            self.best_des_dict = probe_pt_des_dict
            if self.pcp_method == False:
                self.remain_des_prop_dict = remain_des_prop_dict
            self.composition_guess = all_combinations
            self.best_smiles = "None"
    def Des2MolMap_QM9(self,probe_pt_des_dict, remain_des_prop_dict):
            atom_guess_dict = self.ChemFormula(probe_pt_des_dict)
            # print (f"atom_guess_dict: {atom_guess_dict}")
            import itertools
            # Generate all combinations by taking one element from each list
            if "H" in self.atom_ls:
                all_combinations = list(itertools.product(atom_guess_dict["H"], 
                                                        atom_guess_dict["C"], 
                                                        atom_guess_dict["N"], 
                                                        atom_guess_dict["O"], 
                                                        atom_guess_dict["F"]))
            else:
                all_combinations = list(itertools.product(#atom_guess_dict["H"], 
                                                        atom_guess_dict["C"], 
                                                        atom_guess_dict["N"], 
                                                        atom_guess_dict["O"], 
                                                        atom_guess_dict["F"]))
            smiles_ls = []
            for comb_i in all_combinations:
                comb_dict = dict(zip(self.atom_count_keys, comb_i))
                # Filter the QM9 dataset based on the atom counts
                qm9_filtered_df = qm9_n_atom_df
                for idx, count_label in enumerate(self.atom_count_keys):
                    qm9_filtered_df = qm9_filtered_df[qm9_filtered_df[f"{count_label}_count"]==comb_dict[count_label]]
                if qm9_filtered_df.shape[0] > 0:
                    smiles_ls+=list(qm9_filtered_df["SMILES"])
                
            self.smiles_ls = list(smiles_ls)
            # print (f"self.smiles_ls: {self.smiles_ls}")
            # print (f"Number of SMILES found: {len(self.smiles_ls)}")
            remain_smi_ls = list(remain_des_prop_dict["SMILES"])
            # print (f"Number of remaining SMILES: {len(remain_smi_ls)}")
            if len(self.smiles_ls)!= 0:
                target_val_vector = list(probe_pt_des_dict.values())#list(target_point.values())
                cadidate_ls = []
                cadidate_dict = {}
                filtered_smi = []
                prop_ls = []
                # Filter out the SMILES that are already in the training set
                for smi_i in self.smiles_ls:
                    if smi_i in remain_smi_ls:
                        candidate_idx = remain_smi_ls.index(smi_i)
                        des_i = [remain_des_prop_dict[des_key][candidate_idx] for des_key in probe_pt_des_dict.keys()]
                        cadidate_ls.append(des_i)
                        cadidate_dict[smi_i] = [des_i, 
                                                remain_des_prop_dict[self.prop_n][candidate_idx]]
                        
                cadidate_arr = np.array(cadidate_ls)
                
            else:
                cadidate_arr =[]
            if np.shape(cadidate_arr)[0] != 0:
                tree = scipy.spatial.KDTree(cadidate_arr)
                min_dist, min_idx =  tree.query(target_val_vector)
                self.min_dist = min_dist
                self.min_idx = min_idx
                # print ("min_dist:", min_dist)
                # print ("min_dist:", min_idx)
                # print ("cadidate_arr[min_idx,:]", cadidate_arr[min_idx,:])
                test_dist = math.dist(cadidate_arr[min_idx,:],target_val_vector)
                # print (f"test_dist: {test_dist}")
                best_smiles = list(cadidate_dict.keys())[self.min_idx]
                best_des_dict = cadidate_dict[best_smiles][0]
                best_prop = cadidate_dict[best_smiles][1]
                df = pd.DataFrame(remain_des_prop_dict)
                remain_des_prop_dict = df[df["SMILES"] != best_smiles].to_dict(orient="list")

                abs_err = abs(best_prop*self.kcalmol_constant-self.target_prop_val*self.kcalmol_constant)
                self.abs_err = abs_err
                self.best_des_dict = best_des_dict
                self.remain_des_prop_dict = remain_des_prop_dict
                self.composition_guess = all_combinations
                self.best_smiles = best_smiles
            else:
                # print ("No Candidate")
                self.abs_err = self.max_abs_err
                self.best_des_dict = probe_pt_des_dict
                self.remain_des_prop_dict = remain_des_prop_dict
                self.composition_guess = all_combinations
                self.best_smiles = "None"
    # def Des2MolMap_QM9(
    #             self,
    #             probe_pt_des_dict, 
    #             remain_des_prop_dict
    #             ):
    #         """
    #         Maps descriptors to molecules.
    
    #         Args:
    #             probe_pt_des_dict (dict): The descriptor dictionary for the probe point.
    #             remain_des_prop_dict (dict): The descriptor dictionary for the remaining points.
    
    #         Returns:
    #             None: Updates the class attributes with the best matching molecule and its properties.
    #         """
    #         self.remain_des_prop_dict = remain_des_prop_dict
    #         remain_smi_ls = [key for key in self.remain_des_prop_dict.keys() if key != "GC_des_parameters" and key != "SMILES"]
    #         atom_guess_dict = self.ChemFormula(probe_pt_des_dict)
            
    #         # Generate all combinations by taking one element from each list
    #         # Build the list of lists in the same order as self.atom_ls so the
    #         # resulting tuples align with self.atom_count_keys.
    #         lists_for_product = [atom_guess_dict[atom] for atom in self.atom_ls]
    #         all_combinations = list(itertools.product(*lists_for_product))
    #         smiles_ls = []
    #         for comb_i in all_combinations:
    #             comb_dict = dict(zip(self.atom_count_keys, comb_i))
    #             print (f"comb_dict: {comb_dict}")
    #             filtered_smi_i = self.search_mol_wt_chem_formula(comb_dict)
    #             if len(filtered_smi_i) > 0:
    #                 smiles_ls+=list(filtered_smi_i)
                
    #         self.smiles_ls = list(smiles_ls)
    #         remain_smi_ls = [key for key in self.remain_des_prop_dict.keys() if key != "GC_des_parameters" and key != "SMILES"]
    #         if len(self.smiles_ls)!= 0:
    #             print (f"Found {len(self.smiles_ls)} SMILES for the target descriptor vector.")
    #             cadidate_ls = []
    #             cadidate_dict = {}
    #             filtered_smi = []
    #             prop_ls = []
    #             # Filter out the SMILES that are already in the training set
    #             for smi_i in self.smiles_ls:
    #                 if smi_i in remain_smi_ls:
    #                     cadidate_ls.append(list(self.remain_des_prop_dict[smi_i][0].values()))
    #                     cadidate_dict[smi_i] = [self.remain_des_prop_dict[smi_i][0], 
    #                                             self.remain_des_prop_dict[smi_i][1][self.prop_n]]
    #             cadidate_arr = np.array(cadidate_ls)
    #             target_val_vector = list(probe_pt_des_dict.values())#list(target_point.values())
    #         else:
    #             cadidate_arr =[]
    #         # If there are candidate molecules, find the closest one to the target descriptor vector
    #         if np.shape(cadidate_arr)[0] != 0:
    #             print (f"Found {np.shape(cadidate_arr)[0]} candidate molecules for the target descriptor vector.")
    #             tree = scipy.spatial.KDTree(cadidate_arr)
    #             min_dist, min_idx =  tree.query(target_val_vector)
    #             self.min_dist = min_dist
    #             self.min_idx = min_idx
    #             test_dist = math.dist(cadidate_arr[min_idx,:],target_val_vector)
    #             best_smiles = list(cadidate_dict.keys())[self.min_idx]
    #             best_des_dict = cadidate_dict[best_smiles][0]
    #             best_prop = cadidate_dict[best_smiles][1]
    #             remain_des_prop_dict.pop(best_smiles)
                
    #             abs_err = abs(best_prop*self.kcalmol_constant-self.target_prop_val*self.kcalmol_constant)
    #             self.abs_err = abs_err
    #             self.best_des_dict = best_des_dict
    #             self.remain_des_prop_dict = remain_des_prop_dict
    #             self.composition_guess = all_combinations
    #             self.best_smiles = best_smiles
    #         else:
    #             print ("No candidate molecules found for the target descriptor vector.")
    #             # If no candidate molecules are found, set the best SMILES to "None" and use the maximum absolute error
    #             self.abs_err = self.max_abs_err
    #             self.best_des_dict = probe_pt_des_dict
    #             self.remain_des_prop_dict = remain_des_prop_dict
    #             self.composition_guess = all_combinations
    #             self.best_smiles = "None"
            
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
                        candidate_ls.append([
                            np.asarray(des_i[key]).reshape(-1)[0]
                            for key in probe_pt_des_dict.keys()
                        ])
        Returns:
            tuple: A tuple containing two lists:
                - List of atom counts sorted by their posterior probabilities (from smallest to largest).
                    candidate_arr = np.empty((0, len(probe_pt_des_dict)))
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
    

        