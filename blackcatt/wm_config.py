import numpy as np
import torch
import os

# Ray object store for shared memory in simulation mode (optional)
try:
    import ray
    USE_RAY_STORE = True
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
except ImportError:
    USE_RAY_STORE = False

wm_constants = "wm_constants/"

# Training
n_users = 20 # = options.num-supernodes in pyproject.toml # 20
user_fraction = 10 / n_users # = fraction-fit in pyproject.toml # 10 / n_users
mlr = 0.01 # main task learning rate # 0.01
mbs = 64 # local minibatch size # 64
max_batches = (50000//n_users)//mbs # 1 local epoch total data//mbs
dataset = "CIFAR100" # "CIFAR100" # "CIFAR10"
n_classes = 100 # 10 for CIFAR10, 100 for CIFAR100
model = "ResNet183x3" # "ResNet183x3" # "VGG16"
# More realistic FL
iid = True # whether to use IID data distribution across clients # True
iid_beta = 0.5 # Dirichlet distribution parameter for non-IID data distribution across clients # 0.5
linear_partitioner = False # whether to use linear partition sizes # False
linear_part_epoch = False # whether to train the whole epoch or keep all local trainings to a fixed batch number # False
fit_failure_ratio = 0 # Probability of clients starting an outage before aggregation # 0.0
failure_var = 0.0 # Variability in per-client probability of dropout # 0.0
fit_failure_min_rounds = 1 # Minimum number of consecutive training rounds a failed client stays unavailable # 1
fit_failure_max_rounds = 1 # Maximum number of consecutive training rounds a failed client stays unavailable # 1

# Black box WM
tlr = 0.0001 # trigger learning rate # 0.0001
m = 250 # trigger set size # 250 
trigger_type = "random" # "random" # "stealthy" # "unique" # "unique" means each user gets their own unique set of triggers, only implemented for vanilla approach, will crash otherwise
stealth_ratio = 1.0 # ratio of natural images to noise when generating the a stealthy trigger # 1.0
lambda_reg = 0.1 # regularization weight # 0.1
k_cols = 5 # number of emulated collusions # 5 
wm_rounds = 1 # number of watermarking rounds # 1
# Trigger Optimization
t_step_pix = 1 # pixel value step size # 1 
t_step = t_step_pix/255*2 # normalized step size
t_alpha_pix = 64 # total pixel value change budget # 64
t_alpha = t_alpha_pix/255*2 # normalized change budget
t_rounds = 1 # trigger optimization rounds # 1
t_subset = n_users # trigger optimization subset size # n_users
# Functional Regularization
lambda_cl = 0.1 # weight of functional regularization term # 0.1
dataset_cl = "huggan/wikiart" # "huggan/wikiart" # "zh-plus/tiny-imagenet" # "uoft-cs/cifar100"
reserve_testing_images = False # whether to reserve part of testing images for functional regularization / comparison with those experiments
n_reg_images = 500 # number of reserved testing images for functional regularization # 500

# Other config
measure_computation = False # Whether to track execution times and GPU allocation # False
# Recovering from checkpoint
load_checkpoint_flag = False 
loading_round = 0


if n_classes == 10:
  tau =  0.01 # Theorem 5 Skoric12 #  
  clients_tardos_q = np.loadtxt(wm_constants + 'clients_tardos_q_10_k0.5.csv').astype(int)
  p_secret = np.loadtxt(wm_constants + 'p_secret_10_k0.5.csv').astype(float)[:m,:]
if n_classes == 100:
  tau =  0.001 # Scaled because of larger n_classes
  clients_tardos_q = np.loadtxt(wm_constants + 'clients_tardos_q_100_k0.5.csv').astype(int)
  p_secret = np.loadtxt(wm_constants + 'p_secret_100_k0.5.csv').astype(float)[:m,:]
clients_tardos_q = clients_tardos_q[:n_users,:m] # Adjust to trigger size
clients_tardos_q_tensor = torch.from_numpy(clients_tardos_q).long()
p_secret_tensor = torch.from_numpy(p_secret).float()

# Generate output folder
triggers_suffix = ""
if trigger_type == "stealthy":
  triggers_suffix = "B9"
  if stealth_ratio != 1.0:
    triggers_suffix += "-" + str(stealth_ratio)
elif trigger_type == "unique":
  triggers_suffix = "UQ"

testing_images_suffix = ""
if reserve_testing_images == True:
  testing_images_suffix = "_RTI"

noniid_suffix = ""
if iid == False:
  noniid_suffix = "_"+str(iid_beta)+"noniid"
if linear_partitioner == True:
  noniid_suffix += "_linearpart"
  if linear_part_epoch == True:
    noniid_suffix += "E"
  else:
    noniid_suffix += "B"

frac_suffix = ""
if user_fraction != 10 / n_users:
  frac_suffix = "_"+str(user_fraction*n_users)+"subsetfit"

fail_suffix = ""
if fit_failure_ratio != 0:
  fail_suffix = "_"+str(fit_failure_min_rounds)+str(fit_failure_max_rounds)+"_"+str(fit_failure_ratio)+"fratio"
  if failure_var != 0:
    fail_suffix += "_"+str(failure_var)+"fvar"


folder = "FL_models/"+model+"_"+dataset+noniid_suffix+fail_suffix+testing_images_suffix+"_" + str(n_users) + "users"+frac_suffix+"_0.5k_" + str(m) + "m" + triggers_suffix + "_" + str(mbs) + "mbs_" + str(mlr) + "mlr_" + str(tlr) + "tlr_" + str(max_batches) + "mbatches_" + str(wm_rounds) + "wmrounds_" + str(lambda_reg) + "lambdaregCOL" + str(k_cols) + "_" + str(t_subset) + "tsubset_" + str(t_alpha_pix) + "talphaTRIG_"+str(n_reg_images)+str(dataset_cl.split("/")[-1])+"CLAvgKL"+str(lambda_cl)+"_" + str(t_rounds) + "troundsBest_test/"
if not os.path.exists(folder):
  os.makedirs(folder)

