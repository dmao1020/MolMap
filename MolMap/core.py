import numpy as np
import scipy
import math
import itertools
from .probability import prob_atom_count, gaussian_pdf
from .chemistry import return_hartree2kcalmol_constant, Coulomb_matrix
import pubchempy as pcp

# import MolDes package for descriptor extraction, make sure to install it via pip install MolDes
from MolDes import GCT_util, CM_util # importing MolDes package

elements = ["H", "C", "N", "O", "F"]

class pubchem_MolMap:
    def __init__(
            self,
            qm9_atom_df,
            atom_pdf_dict,
            extractor = None, 
            prop_n = "gap",
            target_prop_val = None,
            atom_list = None,
            top_ranks=2,
            GC_tau1 = None, 
            GC_tau2 = None

    ):
        self.extractor = extractor
        self.prop_n = prop_n
        self.target_prop_val = target_prop_val
        self.atom_list = atom_list or ["H","C","N","O","F"]
        self.qm9_atom_df = qm9_atom_df
        self.atom_pdf_dict = atom_pdf_dict
        self.top_ranks = top_ranks

        # Descriptor utils
        self.GC_tau1 = GC_tau1
        self.GC_tau2 = GC_tau2

    def ChemFormula(self, probe_pt_des_dict):
        atom_guess_dict = {}
        for i, atom_i in enumerate(self.atom_list):
            des_val = probe_pt_des_dict['in_prod_f%s_fi'%(atom_i)]
            count_guess_prob, prob = prob_atom_count(atom_i, 
                                                     des_val,
                                                     self.atom_pdf_dict,
                                                     sigma=4)
            # print (f"guess {count_guess_prob} {atom_i}s, <f_{atom_i}, f_i>: {round(des_val, 2)}; \n")
            if atom_i == "C" or atom_i == "H":
                atom_guess_dict[atom_i] = [count_guess_prob[-(rank+1)] for rank in range(self.top_ranks)]
            else:
                atom_guess_dict[atom_i] = [count_guess_prob[-1]]
        return atom_guess_dict
    
    def Des2MolMap(self,probe_pt_des_dict):

        atom_guess_dict = self.ChemFormula(probe_pt_des_dict)
        print (f"atom_guess_dict: {atom_guess_dict}")
        
        # Generate all combinations by taking one element from each list
        all_combinations = list(itertools.product(*[atom_guess_dict[atom] for atom in self.atom_list]))
        
        CID_ls = []
        for comb_i in all_combinations:
            print (f"comb_i: {comb_i}")
            formula_i = "".join(f"{el}{n}" for el, n in zip(self.atom_list, comb_i) if n > 0)
            print (f"formula_i: {formula_i}")
            try:
                compounds = pcp.get_compounds(formula_i, 'formula')
                print (len(compounds))
                print (compounds)
                CID_ls += [compound.cid for compound in compounds]

            except Exception as e:
                print (f"Error retrieving compounds for formula {formula_i}: {e}")
                continue

        smiles_ls = []
        cardidate_arr = []
        if len(CID_ls) > 0:
            print (f"CID_ls: {CID_ls}")
            smiles_ls += [pcp.Compound.from_cid(cid_i).canonical_smiles for cid_i in CID_ls]
            print (f"smiles_ls: {smiles_ls}")

            if len(smiles_ls) > 0:
                dataset = self.extractor.extract_batch(CID_ls)
                if len(dataset) == 0:
                    print ("No compounds found for the given formulas.")
                    # print ("No Candidate")
                    self.abs_err = self.max_abs_err
                    self.best_des_dict = probe_pt_des_dict
                    # self.remain_qm9_des_prop_dict = remain_qm9_des_prop_dict
                    self.composition_guess = all_combinations
                    self.best_smiles = "None"
                else:
                    print (f"dataset: {dataset}")
                    
                    for d in dataset:
                        des_dict = {}
                        # print(f"  - CID: {d.cid}, SMILES: {d.smiles}, charge: {d.charge}")
                        # print (f"Properties: \n  Energy: {d.energy_total}, Total Dipole Moment: {d.total_dipole_moment}, \n  Molecular Weight: {d.molecular_weight}")
                        coord_array = np.array(d.coords)
                        element = d.elements
                        z_ls = element["number"]
                        CM = Coulomb_matrix(coord_array, z_ls)
                        # print (f"coordinates: {coord_array}, z_ls: {z_ls}")
                        # print (f"Coulomb Matrix: {CM}")
                        # calculate descriptors
                        # Coulomb matrix eigenvalue spectrum
                        CM_w, CM_v = np.linalg.eig(CM)
                        des_dict["cm_mu"] = float(np.mean(CM_w))
                        des_dict["cm_sigma"] = float(np.std(CM_w))
                        des_dict["cm_ev1"] = float(np.max(CM_w))

                        #Gershgorin circle descriptors
                        sq_AUC = self.GC_tau2.CM2auc_calc(CM)
                        f_atom = self.GC_tau1.CM2atom_pdf_calc(CM)
                        for i, atom in enumerate(self.atom_list):
                            des_dict["in_prod_f%s_fi"%(atom)] = float(f_atom[i])
                        print (f"des_dict: {des_dict}")
                        # TODO code cadidate _ls and cadidate_dict
                        # TODO find best candidate and fill up self.abs_err, self.best_des_dict, self.remain_qm9_des_prop_dict, self.composition_guess, self.best_smiles

                    target_val_vector = list(probe_pt_des_dict.values())#list(target_point.values())
            
        if cardidate_arr == []:
            print ("No compounds found for the given formulas.")
            # print ("No Candidate")
            self.abs_err = self.max_abs_err
            self.best_des_dict = probe_pt_des_dict
            # self.remain_qm9_des_prop_dict = remain_qm9_des_prop_dict
            self.composition_guess = all_combinations
            self.best_smiles = "None"


        # for comb_i in all_combinations:
        #     comb_dict = dict(zip(self.atom_count_keys, comb_i))
        #     qm9_filtered_df = self.qm9_atom_df
        #     for idx, count_label in enumerate(self.atom_count_keys):
        #         qm9_filtered_df = qm9_filtered_df[qm9_filtered_df[count_label]==comb_dict[count_label]]
        #     if qm9_filtered_df.shape[0] > 0:
        #         smiles_ls+=list(qm9_filtered_df["SMILES"])
            
        # self.smiles_ls = list(smiles_ls)
        # # print (f"self.smiles_ls: {self.smiles_ls}")
        # # print (f"Number of SMILES found: {len(self.smiles_ls)}")
        # remain_smi_ls = list(remain_qm9_des_prop_dict.keys())
        # if len(self.smiles_ls)!= 0:
        #     cadidate_ls = []
        #     cadidate_dict = {}
        #     filtered_smi = []
        #     prop_ls = []
        #     # Filter out the SMILES that are already in the training set
        #     for smi_i in self.smiles_ls:
        #         if smi_i in remain_smi_ls:
        #             cadidate_ls.append(list(remain_qm9_des_prop_dict[smi_i][0].values()))
        #             cadidate_dict[smi_i] = [remain_qm9_des_prop_dict[smi_i][0], 
        #                                     remain_qm9_des_prop_dict[smi_i][1][self.prop_n]]
        #     cadidate_arr = np.array(cadidate_ls)
        #     target_val_vector = list(probe_pt_des_dict.values())#list(target_point.values())
        # else:
        #     cadidate_arr =[]
        # if np.shape(cadidate_arr)[0] != 0:
        #     tree = scipy.spatial.KDTree(cadidate_arr)
        #     min_dist, min_idx =  tree.query(target_val_vector)
        #     self.min_dist = min_dist
        #     self.min_idx = min_idx
        #     # print ("min_dist:", min_dist)
        #     # print ("min_dist:", min_idx)
        #     # print ("cadidate_arr[min_idx,:]", cadidate_arr[min_idx,:])
        #     test_dist = math.dist(cadidate_arr[min_idx,:],target_val_vector)
        #     # print (f"test_dist: {test_dist}")
        #     best_smiles = list(cadidate_dict.keys())[self.min_idx]
        #     best_des_dict = cadidate_dict[best_smiles][0]
        #     best_prop = cadidate_dict[best_smiles][1]
        #     remain_qm9_des_prop_dict.pop(best_smiles)
            
        #     # print (f"best_prop: {best_prop}, hartree2kcalmol_constant:{self.hartree2kcalmol_constant}")
        #     # print (f"self.target_prop_val: {self.target_prop_val}")
        #     abs_err = abs(best_prop*self.hartree2kcalmol_constant-self.target_prop_val*self.hartree2kcalmol_constant)
        #     self.abs_err = abs_err
        #     self.best_des_dict = best_des_dict
        #     self.remain_qm9_des_prop_dict = remain_qm9_des_prop_dict
        #     self.composition_guess = all_combinations
        #     self.best_smiles = best_smiles
        # else:
        #     # print ("No Candidate")
        #     self.abs_err = self.max_abs_err
        #     self.best_des_dict = probe_pt_des_dict
        #     self.remain_qm9_des_prop_dict = remain_qm9_des_prop_dict
        #     self.composition_guess = all_combinations
        #     self.best_smiles = "None"
class QM9_MolMap:
    def __init__(
        self,
        qm9_atom_df,
        atom_pdf_dict,
        atom_list=None,
        config=None,
        prop_n="s",
        target_prop_val=None,
        kcalmol_stat=True,
        max_abs_err=100,
        top_ranks=2
    ):
        self.qm9_atom_df = qm9_atom_df
        self.atom_pdf_dict = atom_pdf_dict
        self.atom_list = atom_list or ["H","C","N","O","F"]
        self.config = config or {}
        self.atom_count_keys = ["%s_count"%atom_i for atom_i in self.atom_list]
        
        # Initialize missing attributes
        self.prop_n = prop_n
        self.target_prop_val = target_prop_val
        self.kcalmol_stat = kcalmol_stat
        self.max_abs_err = max_abs_err
        self.top_ranks = top_ranks
        
        # Derive conversion constant from Hartree to kcal/mol
        if self.kcalmol_stat:
            self.hartree2kcalmol_constant = return_hartree2kcalmol_constant(self.prop_n)
        else:
            self.hartree2kcalmol_constant = 1
    def ChemFormula(self, probe_pt_des_dict):
        atom_guess_dict = {}
        for i, atom_i in enumerate(self.atom_list):
            des_val = probe_pt_des_dict['in_prod_f%s_fi'%(atom_i)]
            count_guess_prob, prob = prob_atom_count(atom_i, 
                                                     des_val,
                                                     self.atom_pdf_dict,
                                                     sigma=4)
            # print (f"guess {count_guess_prob} {atom_i}s, <f_{atom_i}, f_i>: {round(des_val, 2)}; \n")
            if atom_i == "C" or atom_i == "H":
                atom_guess_dict[atom_i] = [count_guess_prob[-(rank+1)] for rank in range(self.top_ranks)]
            else:
                atom_guess_dict[atom_i] = [count_guess_prob[-1]]
        return atom_guess_dict
    def Des2MolMap(self,probe_pt_des_dict, remain_qm9_des_prop_dict):
        atom_guess_dict = self.ChemFormula(probe_pt_des_dict)
        # print (f"atom_guess_dict: {atom_guess_dict}")
        
        # Generate all combinations by taking one element from each list
        all_combinations = list(itertools.product(*[atom_guess_dict[atom] for atom in self.atom_list]))
        smiles_ls = []
        for comb_i in all_combinations:
            comb_dict = dict(zip([f"{atom}_count" for atom in self.atom_list], comb_i))
            qm9_filtered_df = self.qm9_atom_df
            for idx, count_label in enumerate(self.atom_count_keys):
                qm9_filtered_df = qm9_filtered_df[qm9_filtered_df[count_label]==comb_dict[count_label]]
            if qm9_filtered_df.shape[0] > 0:
                smiles_ls+=list(qm9_filtered_df["SMILES"])
            
        self.smiles_ls = list(smiles_ls)
        # print (f"self.smiles_ls: {self.smiles_ls}")
        # print (f"Number of SMILES found: {len(self.smiles_ls)}")
        remain_smi_ls = list(remain_qm9_des_prop_dict.keys())
        if len(self.smiles_ls)!= 0:
            cadidate_ls = []
            cadidate_dict = {}
            filtered_smi = []
            prop_ls = []
            # Filter out the SMILES that are already in the training set
            for smi_i in self.smiles_ls:
                if smi_i in remain_smi_ls:
                    cadidate_ls.append(list(remain_qm9_des_prop_dict[smi_i][0].values()))
                    cadidate_dict[smi_i] = [remain_qm9_des_prop_dict[smi_i][0], 
                                            remain_qm9_des_prop_dict[smi_i][1][self.prop_n]]
            cadidate_arr = np.array(cadidate_ls)
            target_val_vector = list(probe_pt_des_dict.values())#list(target_point.values())
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
            remain_qm9_des_prop_dict.pop(best_smiles)
            
            # print (f"best_prop: {best_prop}, hartree2kcalmol_constant:{self.hartree2kcalmol_constant}")
            # print (f"self.target_prop_val: {self.target_prop_val}")
            abs_err = abs(best_prop*self.hartree2kcalmol_constant-self.target_prop_val*self.hartree2kcalmol_constant)
            self.abs_err = abs_err
            self.best_des_dict = best_des_dict
            self.remain_qm9_des_prop_dict = remain_qm9_des_prop_dict
            self.composition_guess = all_combinations
            self.best_smiles = best_smiles
        else:
            # print ("No Candidate")
            self.abs_err = self.max_abs_err
            self.best_des_dict = probe_pt_des_dict
            self.remain_qm9_des_prop_dict = remain_qm9_des_prop_dict
            self.composition_guess = all_combinations
            self.best_smiles = "None"