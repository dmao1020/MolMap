#%%
#Import library
import MolMap
#%%
import math
import time
import random
import numpy as np
print ("numpy version:", np.__version__)
from numpy import linalg as LA
#import numexpr as ne
import os
import bayes_opt
from bayes_opt import BayesianOptimization 
print ("bayes_opt version:", bayes_opt.__version__)
# from bayes_opt import UtilityFunction
import scipy
from scipy.spatial import distance
print ("scipy version:", scipy.__version__)

from bayes_opt import BayesianOptimization 
from bayes_opt import acquisition
from scipy.stats import qmc

import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, Matern, ConstantKernel, WhiteKernel, DotProduct
print ("sklearn version:", sklearn.__version__)

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
current_dir = os.getcwd()+"/"
dataset_n = "qm9"
data_dir = f"{current_dir}data/{dataset_n}/"
des_dir = f"{data_dir}mp%s_norm/"%(GC_param_dict["mu_power"])
fn = f"{data_dir}qm9_des_prop_dict.npy"
qm9_des_prop_dict = np.load(fn, allow_pickle=True)[()]
all_smi = list(qm9_des_prop_dict.keys())[2:]

des_bounds_fn = f"{des_dir}des_bounds_dict.npy"
des_bounds_dict = np.load(des_bounds_fn, allow_pickle = True)[()]

prop_n = "s"#s, zpve, E_elec_n
# E_elec = u0-zpve; E_elec = (u0-zpve)/n_atoms
n_lhs_points = 50  # Number of initial samples
rand_size = 5 # Number of random samples
acq_dict = {"acq_type":"ei", "xi_val": 0.01} # dictionary for acquisition function settings
bo_n_itr = 100000 # The maximum number of iterations for BO

kcal_mol_const = MolMap.chemistry.return_hartree2kcalmol_constant(prop_n) # Convert Hartree to kcal/mol
target_dict = np.load(f"{data_dir}target_dictionary_{prop_n}.npy", allow_pickle=True)[()]

if prop_n == "zpve":
    # set the epislon_max value
    max_abs_err = 150
    epsilon_val = 0.1
elif prop_n == "s":
    max_abs_err = 40
    epsilon_val = 0.1
else: 
    max_abs_err = 5000
    epsilon_val = 1

target_idx = 20
target_prop_val = target_dict[prop_n][target_idx]
target_dsgdb9nsd = target_dict["dsgdb9nsd"][target_idx]

#### loading the initial training dataset ####
train_fn = f"{data_dir}graph_size_data/qm9_mol_size_split_train.npy"
qm9_mol_size_split_train_dict = np.load(train_fn, allow_pickle=True)[()]

##### file saving directory ##### 
save_result_dir = create_dir("results/")
print ("save_result_dir:", save_result_dir)

##### Acquisition function setting ##### 
if acq_dict["acq_type"] == "ucb":
    acq = acquisition.UpperConfidenceBound(kappa=acq_dict["xi_val"])
elif acq_dict["acq_type"] == "ei":
    acq = acquisition.ExpectedImprovement(xi=acq_dict["xi_val"])#acquisition.UpperConfidenceBound(kappa=2.5)#
save_file_n = f"{save_result_dir}result.npy"

#%%
# Initialize MolMap
MolMap_util = MolMap.MolMap(
    GC_param_dict = GC_param_dict,
    prop_n = prop_n,
    target_prop_val = target_prop_val,
    max_abs_err = max_abs_err,
    atom_ls= ["H", "C", "N", "O", "F"],
    unit = "hartree"
    )

des_keys = ['auc_mij_sq', 'in_prod_fH_fi', 'in_prod_fC_fi', 'in_prod_fN_fi', 'in_prod_fO_fi', 'in_prod_fF_fi', 'cm_ev1', 'cm_mu', 'cm_sigma']

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
    bo_itr_dict = np.load(save_file_n, allow_pickle = True)[()]
    

    SMILES_ls = bo_itr_dict["SMILES_ls"]
    bo_pt_ls = []
    remain_des_prop_dict = qm9_des_prop_dict.copy()

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
    remain_des_prop_dict = qm9_des_prop_dict.copy()
    print ("remain_des_prop_dict:",remain_des_prop_dict.keys())


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
    print ("Randomly sample molecules of various sizes for BO initialization...")
    for size_key, smi_dict in qm9_mol_size_split_train_dict.items():
        train_size = list(smi_dict.keys())
        if len(train_size) > 0:
            smi_list_i = smi_dict[train_size[-1]]
            random_smi_ls = np.random.choice(smi_list_i, size=rand_size, replace=False)

            for smi_i in random_smi_ls:
                smi_idx = np.argwhere(np.array(qm9_des_prop_dict["SMILES"]) == smi_i)[0][0]
                real_next_point = {k: qm9_des_prop_dict[k][smi_idx] for k in des_keys}
                bo_itr_dict["added_point"].append(real_next_point)
                prop_val = qm9_des_prop_dict[prop_n][smi_idx]
                target = -1 * abs(prop_val - target_prop_val)
                bo_itr_dict["target_value"].append(target)
                optimizer.register(params=real_next_point, target=target)
                bo_pt_ls.append(target)
                bo_itr_dict["bo_pt_ls"] = bo_pt_ls
    # ----------------------------
    # Feed LHS samples into optimizer
    print ("Feeding LHS samples into optimizer...")
    for point in lhs_samples:
        next_point_to_probe = dict(zip(keys, point))
        print ("next_point_to_probe:", next_point_to_probe)
        print (next_point_to_probe.keys())
        # output of next_point 
        MolMap_util.Des2MolMap(
            next_point_to_probe,
            remain_des_prop_dict
            )
        target = -1 * MolMap_util.abs_err
        if target > -epsilon_val:
            continue
        else:
            real_next_point = MolMap_util.best_des_dict
            remain_des_prop_dict = MolMap_util.remain_des_prop_dict
            bo_itr_dict["added_point"].append(real_next_point)
            bo_itr_dict["target_value"].append(target)
            optimizer.register(params=real_next_point, target=target)
            bo_pt_ls.append(target)
            bo_itr_dict["bo_pt_ls"] = bo_pt_ls
            bo_itr_dict["SMILES_ls"].append(MolMap_util.best_smiles)

#%%
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
        
    # output of next_point 
    MolMap_util.Des2MolMap(
        next_point_to_probe,
        remain_des_prop_dict
        )
    
    target = -1 * MolMap_util.abs_err

    real_next_point = MolMap_util.best_des_dict
    remain_des_prop_dict = MolMap_util.remain_des_prop_dict
    bo_itr_dict["added_point"].append(real_next_point)
    bo_itr_dict["target_value"].append(target)
    optimizer.register(params=real_next_point, target=target)
    bo_pt_ls.append(target)
    bo_itr_dict["bo_pt_ls"] = bo_pt_ls
    bo_itr_dict["SMILES_ls"].append(MolMap_util.best_smiles)

    # After registration, optionally log theta to monitor
    np.save(save_file_n, bo_itr_dict, allow_pickle=True)
    if target > -epsilon_val: 
        print ("Found Target!")
        break

# %%
