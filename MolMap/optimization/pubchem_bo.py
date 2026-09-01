#%%
import MolMap
from MolMap import chemistry, mapping
import math
import time
import random
import numpy as np
from numpy import linalg as LA
#import numexpr as ne
import os
from bayes_opt import BayesianOptimization 
# from bayes_opt import UtilityFunction
import pandas as pd
from scipy.spatial import distance

from bayes_opt import BayesianOptimization 
from bayes_opt import acquisition
from scipy.stats import qmc
import re

#%%
def create_dir(dir_n):
    if os.path.isdir(dir_n) != True:
        os.mkdir(dir_n)
        return dir_n
    else:    
        return dir_n

def is_multiple_of(index, n=50):
    return index % n == 0    
# %%

################# Gershgorin circle theorem parameters ##########################
GC_param_dict= {
                     "mu_power": 1.2,
                     "var_power": 0.7,
                     "di": 10,
                     "atom_var": 0.5,
                     "d_atom": 10,
                     "x_param": [-100, 300, 400],
                     "cum_pdf_norm_stat": False}
################### define dataset and parameters ##########################
dataset_n = "pubchemQC"# "qm9"
if dataset_n == "qm9":
    max_MW = 100
    atom_ls = ["H", "C", "N", "O", "F"]
elif dataset_n == "pubchemQC":
    max_MW = 300
    atom_ls = ["H", "C", "N", "O"]
################# file reading directory ##########################
sockeye_dir = "/scratch/st-rkrems-1/ymaoai/"
compute_can_dir = "/scratch/dmao1020/"
purple_mac_dir = "/Users/dawnmao/"
m4_mac_dir = "/Users/dawn_mao/"

current_dir =  os.getcwd()+"/"
data_dir = f"{current_dir}/data/{dataset_n}/"

# loading the des_prop_dict
# descriptor and property dictionary for all the molecules in the dataset
# with the format of {smi: [des, prop]}
des_prop_dict_fn = f"{data_dir}mp{GC_param_dict['mu_power']}_norm/pubchem_des_prop_dict.npy"
des_prop_dict = np.load(des_prop_dict_fn, allow_pickle=True)[()]
des_prop_dict.pop("GC_des_parameters")
des_prop_dict.pop("SMILES")
all_smi = list(des_prop_dict.keys())[2:]

# load the bounds for each descriptor
des_bounds_dict_fn = f"{data_dir}mp{GC_param_dict['mu_power']}_norm/des_bounds_dict.npy"
des_bounds_dict = np.load(des_bounds_dict_fn, allow_pickle=True)[()]

# ========================== GPBO parameters ==========================
# property to optimize
# pubchem: alpha_gap, 
prop_n = "alpha_gap"

################### Initial sampling parameters ##########################
# number of initial points for LHS sampling
n_lhs_points = 50
# Number of random samples
rand_size = 1

#################### BO parameters ##########################
# number of BO iterations
bo_n_itr = 200000
acq_dict = {"acq_type":"ei", "xi_val": 0.01}

kcal_mol_const = MolMap.chemistry.return_kcalmol("eV")


# %%
######################################## loading target molecule ########################################
task_id_ = 0
# task_id_ = 0#TODO hash for sockeye
target_dir = "%sgpbo_target_data/"%(data_dir)#TODO
print ("file_directory_path:", data_dir)

target_dict = np.load("%starget_dictionary.npy"%(data_dir), allow_pickle=True)[()]
smi_ls = target_dict["SMILES"]
# print (smi_ls)
task_id_ls_ = np.arange(len(smi_ls))
# %%


# Number of rounds of testing for building statistics
n_tests = 10
task_id_ls = []
for i in range(n_tests):
    for task_id_i in task_id_ls_:
        task_id_ls.append([task_id_i, i])
print ("task_id_ls:", task_id_ls)

if prop_n == "zpve":
    # set the epislon_max value
    max_abs_err = 150
    epsilon_val = 0.1
elif prop_n == "s":
    max_abs_err = 40
    epsilon_val = 0.1
elif prop_n == "alpha_gap":
    max_abs_err = 300
    epsilon_val = 0.1
else: 
    max_abs_err = 5000
    epsilon_val = 1

#%%
# import pubchempy as pcp


# def pcp_find_cid_from_smiles(smi):
#     results = pcp.get_compounds(smi, 'smiles')
#     if results:
#         cid = results[0].cid
#         return cid
#     else:
#         print("No compound found.")
#         return None
    
task_id, round_id = task_id_ls[task_id_]
print ("task_id:", task_id, "round_id:", round_id)
print ("target_dict[prop_n]:", np.array(target_dict[prop_n]))
target_prop_val = target_dict[prop_n][task_id]#float(np.max(des_prop_dict[prop_n]))
# target_dsgdb9nsd = target_dict["dsgdb9nsd"][task_id]
target_smi = target_dict["SMILES"][task_id]
# cid = find_cid_from_smiles(target_smi)
print (f"target_smi: {target_smi}, target_prop_val: {target_prop_val}")
print (des_prop_dict[target_smi])
cid = des_prop_dict[target_smi][2]
print (f"target CID: {cid}")

if prop_n == "s":
    print ("Entropy value of the target molecule:  %.6f Hartree; %.3f kcal/mol"%(target_prop_val, target_prop_val*kcal_mol_const))
if prop_n == "zpve":
    print ("ZPVE value of the target molecule: %.6f Hartree; %.3f kcal/mol"%(target_prop_val, target_prop_val*kcal_mol_const))
if prop_n == "alpha_gap":
    print ("Alpha gap value of the target molecule: %.6f eV; %.3f kcal/mol"%(target_prop_val, target_prop_val*kcal_mol_const))
# print ("target_dsgdb9nsd:",target_dsgdb9nsd)

# %%
#### loading the initial training dataset ####
train_fn = f"{data_dir}mol_size_split_train_dict.npy"
mol_size_split_train_dict = np.load(train_fn, allow_pickle=True)[()]
# print (mol_size_split_train_dict)
save_result_dir = create_dir(f"{current_dir}{dataset_n}_bo_result/")
save_result_dir = create_dir(f"{save_result_dir}{prop_n}/")
# print ("save_result_dir:", save_result_dir)
# %%
##### Acquisition function setting ##### 
if acq_dict["acq_type"] == "ucb":
    acq = acquisition.UpperConfidenceBound(kappa=acq_dict["xi_val"])
elif acq_dict["acq_type"] == "ei":
    acq = acquisition.ExpectedImprovement(xi=acq_dict["xi_val"])#acquisition.UpperConfidenceBound(kappa=2.5)#

# save_file_n = "%slocaltest_MolMap_result_round%s.npy"%(save_result_dir, round_id)#TODO hash for sockeye
save_file_n = "%sMolMap_result_round%s.npy"%(save_result_dir, round_id+1)#TODO unhash for sockeye
print (f"save_file_n: {save_file_n}")

#%%
# import importlib
import MolMap.mapping as mapping
# importlib.reload(MolMap)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, Matern, ConstantKernel, WhiteKernel, DotProduct

# Initialize MolMap
MolMap_util = mapping.MolMap(
    GC_param_dict = GC_param_dict,
    prop_n = prop_n,
    target_prop_val = target_prop_val,
    max_abs_err = max_abs_err,
    atom_ls= atom_ls,
    max_MW = max_MW,
    )

print (des_bounds_dict)
#%%
MolMap_util.atom_cum_pdf_dict.keys()


#%%

#%%
if os.path.isfile(save_file_n):
    print ("BO started already! Loading existing BO points...")
    # Set up BO optimizer
    optimizer = BayesianOptimization(
        f=None,
        acquisition_function=acq,
        pbounds=des_bounds_dict,
        verbose=2,
        random_state = None,#1,
        allow_duplicate_points=True)

    # Define your custom kernel
    kernel_n = "RQ+Mat*DP"
    kernel = RationalQuadratic(length_scale_bounds=(1e-05, 1e05), 
                            alpha_bounds=(1e-05, 1e05)) + Matern(length_scale=1.0, 
                                                                 length_scale_bounds=(1e-05, 1e05),
                                                                        nu=1.5) * DotProduct(sigma_0_bounds=(1e-05, 1e05))
    
    # Build your custom GP regressor
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=10
    )

    # Assign the custom GP directly
    optimizer._gp = gp  #

    # load existing BO points
    # print ("BO started already!")
    bo_itr_dict = np.load(save_file_n, allow_pickle = True)[()]
    # print (bo_itr_dict.keys)

    SMILES_ls = bo_itr_dict["SMILES_ls"]
    bo_pt_ls = []
    remain_des_prop_dict = des_prop_dict.copy()

    for idx, smi in enumerate(SMILES_ls):
        # print ("point index:", idx)
        target = bo_itr_dict["target_value"][idx]
        point = list(bo_itr_dict["added_point"][idx].values())
        optimizer.register(params=point, target=target)
        bo_pt_ls.append(target)

        # update dataframe of the remaining data points
        if smi!= "None":
            remain_des_prop_dict.pop(smi)
    bo_itr_dict["bo_pt_ls"] = bo_pt_ls
else:
    print ("Initialize BO with LHS")
    # LHS sampling
    # ----------------------------
    # Define bounds
    keys = list(des_bounds_dict.keys())
    bounds = np.array([des_bounds_dict[k] for k in keys])

    # ----------------------------
    # Latin Hypercube Sampling
    sampler = qmc.LatinHypercube(d=len(des_bounds_dict))
    lhs_samples_unit = sampler.random(n=n_lhs_points)
    lhs_samples = qmc.scale(lhs_samples_unit, bounds[:, 0], bounds[:, 1])

    # print (lhs_samples)
    remain_des_prop_dict = des_prop_dict.copy()


    # Set up BO optimizer
    optimizer = BayesianOptimization(
        f=None,
        acquisition_function=acq,
        pbounds=des_bounds_dict,
        verbose=2,
        random_state = None,#1,
        allow_duplicate_points=True)
    
    # Define your custom kernel
    kernel_n = "RQ+Mat*DP"
    kernel = RationalQuadratic(length_scale_bounds=(1e-05, 1e05), 
                            alpha_bounds=(1e-05, 1e05)) + Matern(length_scale=1.0, 
                                                                 length_scale_bounds=(1e-05, 1e05),
                                                                        nu=1.5) * DotProduct(sigma_0_bounds=(1e-05, 1e05))
    
    # Build your custom GP regressor
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=10
    )

    # Assign the custom GP directly
    optimizer._gp = gp  # 

    bo_pt_ls = []
    bo_itr_dict = {"n_lhs_points":n_lhs_points,
                   "bo_pt_ls": bo_pt_ls,
                   "added_point":[],
                   "target_value":[],
                   "SMILES_ls":[],
                }
    
    # ----------------------------
    # Feed randomly sampled molecules of various sizes into optimizer
    
    for size_key, smi_list_i in mol_size_split_train_dict.items():
        clean_smi_ls = [smi for smi in smi_list_i if 
                    "+" not in smi and
                    "-" not in smi and
                    "[1H" not in smi and
                    "[2H" not in smi and
                        "[3H" not in smi and
                    not re.search(r"\[(?!12)\d+C", smi) and
                    not re.search(r"\[(?!14)\d+N", smi) and
                    not re.search(r"\[(?!18)\d+O", smi)
                    ]
        print (f"Processing molecules with MW < {size_key}...")
        print (f"Number of molecules in this size bin: {len(smi_list_i)}")
        if len(clean_smi_ls) > 0:
            random_smi = np.random.choice(clean_smi_ls, size=1, replace=False)[0]
            real_next_point = des_prop_dict[random_smi][0]
            bo_itr_dict["added_point"].append(real_next_point)

            prop_val = des_prop_dict[random_smi][1][prop_n]
            print (f"Property value of the selected molecule: {prop_val}")
            target = -1 * abs(prop_val - target_prop_val)
            bo_itr_dict["target_value"].append(target)
            print ("target:", target)
            print ("real_next_point:", real_next_point)
            optimizer.register(params=real_next_point, target=target)
            bo_pt_ls.append(target)
            bo_itr_dict["bo_pt_ls"] = bo_pt_ls
    np.save(save_file_n, bo_itr_dict, allow_pickle=True)

    # ----------------------------
    # Feed LHS samples into optimizer
        
    for point in lhs_samples:
        # print ("point:", point)
        next_point_to_probe = dict(zip(keys, point))
        # print ("next_point_to_probe:", next_point_to_probe)
        # output of next_point 
        MolMap_util.Des2MolMap(
            next_point_to_probe,
            remain_des_prop_dict
            )
        # print (f"bo_result: {MolMap_util}")
        # print ("bo_result.abs_err:",MolMap_util.abs_err)
        target = -1 * MolMap_util.abs_err
        if target > -epsilon_val:
            # print ("Target value is too high, skip this point!")
            continue
        else:
            # print ("bbf_util:", bbf_util)
            real_next_point = MolMap_util.best_des_dict
            remain_des_prop_dict = MolMap_util.remain_des_prop_dict
            # print("Found the target value to be:", target)
            # print ("real_next_point:", real_next_point)
            bo_itr_dict["added_point"].append(real_next_point)
            bo_itr_dict["target_value"].append(target)
            # print ("real_next_point:", real_next_point)
            optimizer.register(params=real_next_point, target=target)
            bo_pt_ls.append(target)
            bo_itr_dict["bo_pt_ls"] = bo_pt_ls
            bo_itr_dict["SMILES_ls"].append(MolMap_util.best_smiles)# %%
    np.save(save_file_n, bo_itr_dict, allow_pickle=True)


# %%
refit_interval = 100
for _ in range(bo_n_itr):
    print ("iteration:", _)
    # Dynamically toggle hyperparameter optimization before suggest
    # Configure GPR for refit or fixed kernel
    if is_multiple_of(_, n=refit_interval):
        print("Refit GP (with kernel hyperparameter optimization)")
        optimizer._gp.optimizer = 'fmin_l_bfgs_b'  # Use string, not tuple
        optimizer._gp.n_restarts_optimizer = 10  # Ensure multiple restarts
    else:
        optimizer._gp.optimizer = None  # Fixed kernel
        optimizer._gp.kernel = best_kernel  # Use last optimized kernel
    
    # Get next point
    next_point_to_probe = optimizer.suggest()
    
    # Update best_kernel after refit
    if is_multiple_of(_, n=refit_interval):
        best_kernel = optimizer._gp.kernel_  # Store optimized kernel
        print(f"Log-marginal-likelihood: {optimizer._gp.log_marginal_likelihood(optimizer._gp.kernel_.theta)}")
    
    # Debug prints
    print(f"optimizer._gp.optimizer: {optimizer._gp.optimizer}")
    print(f"kernel.theta: {optimizer._gp.kernel_.theta}")
    # print (f"next_point: {next_point_to_probe}")
    # output of next_point 
    MolMap_util.Des2MolMap(
        next_point_to_probe,
        remain_des_prop_dict
        )
    target = -1 * MolMap_util.abs_err

    # print ("bbf_util:", bbf_util)
    real_next_point = MolMap_util.best_des_dict
    remain_des_prop_dict = MolMap_util.remain_des_prop_dict
    # print("Found the target value to be:", target)
    # print ("real_next_point:", real_next_point)
    bo_itr_dict["added_point"].append(real_next_point)
    bo_itr_dict["target_value"].append(target)
    # print ("real_next_point:", real_next_point)
    optimizer.register(params=real_next_point, target=target)
    bo_pt_ls.append(target)
    bo_itr_dict["bo_pt_ls"] = bo_pt_ls
    bo_itr_dict["SMILES_ls"].append(MolMap_util.best_smiles)
    # After registration, optionally log theta to monitor
    
    np.save(save_file_n, bo_itr_dict, allow_pickle=True)#TODO uhash for hpc
    if target > -epsilon_val: 
        print ("Found Target!")
        kcalmol_constant = MolMap_util.hartree2kcalmol_constant
        target_smi = target_dict["SMILES"][task_id]
        target_prop = target_dict[prop_n][task_id]
        print (f"Target: {target_smi}; {prop_n} = {round(target_prop * kcalmol_constant, 2)} kcal/mol")

        bo_smi = bo_itr_dict["SMILES_ls"][-1]
        bo_prop_val = des_prop_dict[bo_smi][1][prop_n]
        print (f"BO result: {bo_smi}; {prop_n} = {round(bo_prop_val * kcalmol_constant, 2)} kcal/mol")
        
        break

# %%
