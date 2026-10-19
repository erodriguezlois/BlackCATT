from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from flwr.common import ParametersRecord, array_from_numpy
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
from torchvision.transforms import Compose, Normalize, ToTensor, ToPILImage, Lambda, Resize, CenterCrop
import blackcatt.wm_config as wm_config
import blackcatt.models as models
from blackcatt.task import set_weights, get_weights, test, fds, load_state_from_record
import numpy as np
import pickle
import shutil
import os
import copy
from PIL import Image
import torch.nn.utils.prune as prune
import os
from datasets import load_dataset
import pandas as pd
from random import sample
import time

## Constants and Caches ##
fds_aux = None  # Cache FederatedDataset for functional regularization
testloader_aux = None  # Cache DataLoader for functional regularization

# Pre-create trigger transforms to avoid repeated Compose creation
TRIGGER_TRANSFORMS = Compose(
    [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
# Ray object store for simulation mode - primary storage for client states
_ray_client_states = {}  # dict mapping cid -> ray.ObjectRef

## Functional Regularization specific functions ##
def load_aux_data(aux_cid=0, batch_size=wm_config.mbs):

    global fds_aux
    global testloader_aux

    if wm_config.dataset_cl == "zh-plus/tiny-imagenet":
        # 500 samples per partition
        partitioner_aux = IidPartitioner(num_partitions=int(100000/wm_config.n_reg_images))
        if fds_aux is None:
            fds_aux = FederatedDataset(
                dataset="zh-plus/tiny-imagenet",
                partitioners={"train": partitioner_aux},
            )
        partition_aux = fds_aux.load_partition(aux_cid)
        pytorch_transforms_aux = Compose(
            [ToTensor(),
            Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            Resize((32, 32))]  # Resize to match model input size
        )
        def apply_transforms_aux(batch):
            batch["image"] = [pytorch_transforms_aux(img) for img in batch["image"]]
            return batch

        testset_aux = partition_aux.with_transform(apply_transforms_aux)
        testloader_aux = DataLoader(testset_aux, batch_size=batch_size, shuffle=False)

    elif wm_config.dataset_cl == "huggan/wikiart":
        if testloader_aux is None:
            aux_samples = wm_config.n_reg_images
            skip_samples = int(aux_cid) * aux_samples

            it_dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
            partition_stream = it_dataset.skip(skip_samples).take(aux_samples)
            
            pytorch_transforms_aux = Compose(
                [ToTensor(),
                Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                CenterCrop((512, 512)),
                Resize((32, 32))]  # Crop and Resize to match model input size
            )
            def apply_transforms_aux(batch):
                batch["image"] = [pytorch_transforms_aux(img) for img in batch["image"]]
                return batch
            
            images = [pytorch_transforms_aux(sample["image"]) for sample in partition_stream]
            tensors = torch.stack(images)

            testset_aux = TensorDataset(tensors)
            if int(aux_cid) != 0:
                return DataLoader(testset_aux, batch_size=batch_size, shuffle=False)
            # Do not assign to global variable unless it is the first partition (functional regularization dataset)
            # Otherwise when using stealthy triggers the same samples will be used as regularization
            # And it will break
            testloader_aux = DataLoader(testset_aux, batch_size=batch_size, shuffle=False)
    
    elif wm_config.dataset_cl == "uoft-cs/cifar100":
        partitioner_aux = IidPartitioner(num_partitions=wm_config.n_users)
        if fds_aux is None:
            fds_aux = FederatedDataset(
                dataset="uoft-cs/cifar100",
                partitioners={"train": partitioner_aux},
            )
        centralized_testset = fds_aux.load_split("test")
        pytorch_transforms_aux = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        def apply_transforms_aux(batch):
            batch["img"] = [pytorch_transforms_aux(img) for img in batch["img"]]
            return batch

        testset_aux = centralized_testset.with_transform(apply_transforms_aux)
        # Selecting test split w 500 samples for aux data
        testset_aux = torch.utils.data.Subset(testset_aux, [i for i in range(wm_config.n_reg_images)])
        testloader_aux = DataLoader(testset_aux, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError(f"Unsupported dataset for functional regularization: {wm_config.dataset_cl}")

    return testloader_aux


def label_reference(net):
    # Get the full model soup - pre-load client states
    client_states = _load_all_client_states(np.arange(wm_config.n_users), wm_config.folder)
    for cid_i in range(wm_config.n_users):
        client_model_params = [val for key_state, val in client_states[cid_i].items()]
        if cid_i == 0:
            average_model = [val/wm_config.n_users for val in client_model_params]
        else:
            average_model = [average_val + (val/wm_config.n_users) for average_val,val in zip(average_model,client_model_params)]
    set_weights(net,average_model)
    net.eval()
    # Load aux dataset
    auxloader = load_aux_data()

    all_predictions = []
    for data in auxloader:
        if isinstance(data, (list, tuple)):
            images = data[0]  
        else:
            try:
                images = data["img"]
            except:
                images = data["image"]
        predictions = net(images.to(net.fc.weight.device))
        all_predictions.extend (predictions.detach().cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    np.save(wm_config.folder + "aux_predictions.npy", all_predictions)
    
    return


## Evaluating evolution ##

def evaluate_config(server_round):
    # Run every time flwr launches evaluation
    # Adjust round number if we're resuming from checkpoint
    adjusted_round = server_round + wm_config.loading_round
    
    # Store models and triggers to disk every 250 rounds for analysis/checkpointing
    if adjusted_round % 250 == 0:
        shutil.copyfile(wm_config.folder + "triggers.npy", wm_config.folder + "trigger_round_"+ str(adjusted_round) +".npy")
        # Also checkpoint client states from Ray to disk
        for i_cid in range(wm_config.n_users):
            try:
                state_dict = _load_client_state(i_cid, wm_config.folder)
                # Save snapshot
                client_status = {"parameters": state_dict}
                with open(wm_config.folder + str(adjusted_round) + "_client_status_"+str(i_cid)+".pkl", 'wb') as f:
                    pickle.dump(client_status, f)
            except:
                # Fallback if state not in Ray yet
                try:
                    shutil.copyfile(wm_config.folder + "client_status_"+str(i_cid)+".pkl", 
                                  wm_config.folder + str(adjusted_round) + "_client_status_"+str(i_cid)+".pkl")
                except:
                    pass
    return {"current_round": adjusted_round, "server_round": server_round}


def check_metrics(net,i_cid,valloader,device):
    """Check metrics for the training evolution"""
    net.eval()
    ### Main task accuracy ###
    loss, accuracy = test(net, valloader, device)
    ### Trigger accuracy ###
    # Load and normalize triggers
    if wm_config.trigger_type == "unique":
        triggers = _normalize_triggers(np.load(wm_config.folder + "triggers.npy")[i_cid*wm_config.m:(i_cid+1)*wm_config.m,:,:,:], device)
    else:
        triggers = _normalize_triggers(np.load(wm_config.folder + "triggers.npy"), device)
    labels = wm_config.clients_tardos_q_tensor[int(i_cid),:].to(device)
    with torch.no_grad():
        outputs = net(triggers)
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
    t_accuracy =correct/wm_config.m
    ### MAV ### (Or acc in case of unique triggers)
    # Open auxiliary model (i_cid + 1)
    while(1):
        # In case there is a collision reading the file
        try:
            with open(wm_config.folder + "client_status_"+str(i_cid+1)+".pkl", 'rb') as f:
                client_col_status = pickle.load(f)
                client_col_params = client_col_status["parameters"]
            break
        except:
            continue
    collusion_weights = [1/2*(aux_cid.numpy() + curr_cid) for (_,aux_cid), curr_cid in zip(client_col_params.items(),get_weights(net))]
    set_weights(net, collusion_weights)
    if wm_config.trigger_type == "unique":
        with torch.no_grad():
            outputs = net(triggers)
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            t_accuracy_c2 =correct/wm_config.m
        with open(wm_config.folder + "metrics_"+str(i_cid)+".csv", "a") as file:
            file.write(f"{loss},{accuracy},{t_accuracy},{t_accuracy_c2}\n")
    else:
        with torch.no_grad():
            outputs = net(triggers)
            y = torch.argmax(outputs,dim=1).cpu().numpy()
            mav = np.average([y[i] not in wm_config.clients_tardos_q[i_cid:i_cid+2,i] for i in range(wm_config.m)])
            tp, t_s = tardos_accusation(y,vectors=wm_config.clients_tardos_q, p_secret=wm_config.p_secret, tau=wm_config.tau)
            fp = tp not in [i_cid,i_cid+1]
            fn = tp == -1
        ### Append to csv file ###
        with open(wm_config.folder + "metrics_"+str(i_cid)+".csv", "a") as file:
            file.write(f"{loss},{accuracy},{t_accuracy},{mav},{fn},{fp}\n")
    return loss, len(valloader.dataset), accuracy


## Watermarking specific functions ##

def _save_client_state(cid, state_dict, folder):
    """Save a single client state to disk (for checkpointing)"""
    client_status = {"parameters": state_dict}
    with open(folder + "client_status_"+str(cid)+".pkl", 'wb') as f:
        pickle.dump(client_status, f)

def _load_client_state(cid, folder):
    """Load a single client state (from Ray if available, else disk)"""
    global _ray_client_states
    
    # Try Ray first (simulation mode)
    if wm_config.USE_RAY_STORE and cid in _ray_client_states:
        try:
            import ray
            return ray.get(_ray_client_states[cid])
        except:
            pass
    
    # Fallback to disk
    with open(folder + "client_status_"+str(cid)+".pkl", 'rb') as f:
        client_status = pickle.load(f)
        return client_status["parameters"]

def _store_client_state(cid, state_dict):
    """Store client state in Ray (simulation mode) or do nothing (disk mode)"""
    global _ray_client_states
    
    if wm_config.USE_RAY_STORE:
        try:
            import ray
            _ray_client_states[cid] = ray.put(state_dict)
        except:
            pass

def _load_all_client_states(sampled_cids, folder):
    """Load all client states efficiently"""
    client_states = {}
    for i_cid in sampled_cids:
        client_states[i_cid] = _load_client_state(i_cid, folder)
    return client_states

def _normalize_triggers(triggers_np, device):
    """Normalize numpy triggers to tensor"""
    triggers = torch.stack([TRIGGER_TRANSFORMS(Image.fromarray(trigger.astype(np.uint8))) for trigger in triggers_np]).float()
    return triggers.to(device)

def _compute_collusion_loss(net, triggers, labels_cid, labels, indexes, client_states, k_cols, criterion, p_secret, lambda_reg, device):
    """Compute collusion loss for a single client with memory-efficient forward passes"""
    original_weights = {name: param.clone() for name, param in net.named_parameters()}
    col_loss = 0.0
    
    for _ in range(k_cols):
        # Prepare collusion - reset to half original weights
        for name, param in net.named_parameters():
            param.data.copy_(original_weights[name].data / 2)
        
        # Select random colluder from pre-loaded states
        col_id = np.random.choice(indexes)
        labels_aux_id = labels[col_id, :]
        client_col_params_dict = client_states[col_id]
        client_col_params = [val for key_state, val in client_col_params_dict.items() if key_state in [key for key, val in net.named_parameters()]]
        
        # Add colluder weights
        for param, other_param in zip(net.parameters(), client_col_params):
            other_param = other_param.to(param.data.device)
            param.data += other_param / 2
        
        # Compute outputs once and compute both losses from single forward pass
        outputs = net(triggers)
        loss_cid = criterion(outputs, labels_cid)
        loss_aux = criterion(outputs, labels_aux_id)
        col_loss += lambda_reg * torch.mean(torch.where(loss_cid < loss_aux, loss_cid, loss_aux))
    
    return col_loss

def _start_measurement(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()

def _finish_measurement(label, start_time, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        record = (
            f"[TIMING] {label}: {elapsed:.3f}s, "
            f"peak GPU memory allocated={peak_allocated:.1f} MiB, "
            f"reserved={peak_reserved:.1f} MiB\n"
        )
    else:
        elapsed = time.perf_counter() - start_time
        record = f"[TIMING] {label}: {elapsed:.3f}s, peak GPU memory=N/A (CPU)\n"
    with open(os.path.join(wm_config.folder, "timing.log"), "a", encoding="utf-8") as timing_file:
        timing_file.write(record)

def update_triggers(metrics, subset_size=wm_config.t_subset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    # Prepare model copies for reference (rename aux -> main)
    for i_cid in range(wm_config.n_users):
        try:
            os.rename(wm_config.folder + "aux_client_status_"+str(i_cid)+".pkl", wm_config.folder + "client_status_"+str(i_cid)+".pkl")
        except:
            pass
    if wm_config.measure_computation:
        trigger_update_start = _start_measurement(device)

    t_rounds = wm_config.t_rounds
    k_cols = wm_config.k_cols

    if t_rounds > 0:
        """Aggregate triggers from all clients"""    
        
        # Load all client states (from Ray if available, else disk)
        print("Loading client states...")

        # Sample only a subset of clients for trigger optimization
        if subset_size < wm_config.n_users:
            sampled_cids = np.random.choice(wm_config.n_users, subset_size, replace=False)
        else:
            sampled_cids = np.arange(wm_config.n_users)

        client_states = _load_all_client_states(sampled_cids, wm_config.folder)
        
        # Load and normalize triggers
        triggers = _normalize_triggers(np.load(wm_config.folder + "triggers.npy"), device)
        triggers_0 = _normalize_triggers(np.load(wm_config.folder + "trigger_round_0.npy"), device)
        
        # Load network
        if wm_config.model == "VGG16":
            net = models.VGG16()
        elif wm_config.model == "ResNet183x3":
            net = models.ResNet18()
        net.eval()
        net.to(device)
        
        # Prepare training
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        labels = wm_config.clients_tardos_q_tensor.to(device)
        p_secret = wm_config.p_secret_tensor.to(device)
        
        # Compute best triggers
        all_losses = []
        all_trigs = []
        all_trigs.append(triggers.cpu())

        for round in range(t_rounds):      
            total_loss = 0.0
            triggers.requires_grad = True

            for i_cid in sampled_cids:
                # Set original weights from pre-loaded state
                net.load_state_dict(client_states[i_cid])
                labels_cid = labels[int(i_cid), :]
                loss_base = torch.mean(criterion(net(triggers), labels_cid))
                
                # Add to loss
                total_loss += loss_base.item()
                loss_base.backward()
                
                # Compute collusion loss
                indexes = list(sampled_cids)
                indexes.remove(int(i_cid))
                
                loss_col = _compute_collusion_loss(
                    net, triggers, labels_cid, labels, indexes, client_states,
                    k_cols, criterion, p_secret, wm_config.lambda_reg, device
                )
                total_loss += float(loss_col)
                if float(loss_col) != 0.0:
                    loss_col.backward()

            # Update triggers and fix constraints
            triggers = triggers - wm_config.t_step * triggers.grad.sign()
            triggers = torch.where(triggers > triggers_0 + wm_config.t_alpha, triggers_0 + wm_config.t_alpha, triggers)
            triggers = torch.where(triggers < triggers_0 - wm_config.t_alpha, triggers_0 - wm_config.t_alpha, triggers)
            triggers = torch.clamp(triggers, min=-1, max=1)
            triggers = triggers.detach()

            # Log triggers
            all_trigs.append(triggers.cpu())
            all_losses.append(total_loss)
            
            if round == t_rounds - 1:
                # Compute loss for last step
                total_loss = 0.0
                with torch.no_grad():
                    for i_cid in sampled_cids:
                        net.load_state_dict(client_states[i_cid])
                        labels_cid = labels[int(i_cid), :]
                        loss_base = torch.mean(criterion(net(triggers), labels_cid))
                        total_loss += loss_base.item()
                        
                        # Compute collusion loss from available client states
                        indexes = list(sampled_cids)
                        indexes.remove(int(i_cid))
                        original_weights = {name: param.clone() for name, param in net.named_parameters()}

                        for _ in range(k_cols):
                            for name, param in net.named_parameters():
                                param.data.copy_(original_weights[name].data / 2)
                            
                            col_id = np.random.choice(indexes)
                            labels_aux_id = labels[col_id, :]
                            client_col_params_dict = client_states[col_id]
                            client_col_params = [val for key_state, val in client_col_params_dict.items() if key_state in [key for key, val in net.named_parameters()]]
                            
                            for param, other_param in zip(net.parameters(), client_col_params):
                                other_param = other_param.to(param.data.device)
                                param.data += other_param / 2
                            
                            # Compute loss and select label with lowest loss
                            # Compute outputs once and compute both losses from single forward pass
                            outputs = net(triggers)
                            loss_cid = criterion(outputs, labels_cid)
                            loss_aux = criterion(outputs, labels_aux_id)
                            loss_col = wm_config.lambda_reg * torch.mean(torch.where(loss_cid < loss_aux, loss_cid, loss_aux))
                            total_loss += loss_col.item()
                
                all_losses.append(total_loss)

        # Save best triggers and fix the range
        # If there are not multiple trigger versions, no need to update
        if len(all_trigs) > 1:
            triggers = all_trigs[np.argmin(all_losses)]
            triggers = triggers * 0.5 + 0.5
            triggers = [ToPILImage()(trigger) for trigger in triggers]
            triggers = np.array(triggers).astype(np.uint8)
            np.save(wm_config.folder + "triggers.npy", triggers.astype(np.uint8))
    
    if wm_config.measure_computation:                
        _finish_measurement("update_triggers", trigger_update_start, device)

    # Prepare label reference for next round
    if wm_config.measure_computation:
        label_reference_start = _start_measurement(device)
    if wm_config.lambda_cl > 0:
        # Load network
        if wm_config.model == "VGG16":
            net = models.VGG16()
        elif wm_config.model == "ResNet183x3":
            net = models.ResNet18()
        net.eval()
        net.to(device)
        label_reference(net)
        net.train()
    if wm_config.measure_computation:
        _finish_measurement("label_reference", label_reference_start, device)

    return {"triggers": 1}


def wm_client(net,i_cid,client_state,device,k_cols=wm_config.k_cols,epochs=wm_config.wm_rounds):
    """Watermarks model copy"""
    if wm_config.measure_computation:
        measure_client = int(i_cid) == 0
        if measure_client:
            wm_client_start = _start_measurement(device)

    # Load model weights directly from ParametersRecord (more efficient)
    load_state_from_record(net, client_state)
    
    optim = torch.optim.SGD(net.parameters(), lr=wm_config.tlr, momentum=0.9, weight_decay=0.0001)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_CL = torch.nn.KLDivLoss(reduction='batchmean')
    
    net.train()
    net.to(device)
    # Freeze BN layers
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    # Load and normalize triggers once (not per epoch)
    if wm_config.trigger_type == "unique":
        triggers = _normalize_triggers(np.load(wm_config.folder + "triggers.npy")[i_cid*wm_config.m:(i_cid+1)*wm_config.m,:,:,:], device)
    else:
        triggers = _normalize_triggers(np.load(wm_config.folder + "triggers.npy"), device)
    labels = wm_config.clients_tardos_q_tensor.to(device)
    p_secret = wm_config.p_secret_tensor.to(device)
    
    for _ in range(epochs):
        labels_cid = labels[int(i_cid),:]
        optim.zero_grad()
        loss_base = criterion(net(triggers), labels_cid)
        loss_col = 0
        indexes = list(range(wm_config.n_users))
        indexes.pop(int(i_cid))
        original_weights = {name: param.clone() for name, param in net.named_parameters()}

        for _ in range(k_cols):
            # Prepare collusion
            for name, param in net.named_parameters():
                param.data.copy_(original_weights[name].data / 2)
            
            # Select random colluder and load only its state (on-demand)
            col_id = np.random.choice(indexes)
            labels_aux_id = labels[col_id,:]
            client_col_params_dict = _load_client_state(col_id, wm_config.folder)
            client_col_params = [val for key_state, val in client_col_params_dict.items() if key_state in [key for key, val in net.named_parameters()]]

            for param, other_param in zip(net.parameters(), client_col_params):
                other_param = other_param.to(param.data.device)
                param.data += other_param / 2

            # Add to loss with lowest loss label selection
            # Compute outputs once and compute both losses from single forward pass
            outputs = net(triggers)
            loss_cid = criterion(outputs, labels_cid)
            loss_aux = criterion(outputs, labels_aux_id)
            loss_col += torch.where(loss_cid < loss_aux, loss_cid, loss_aux)

        # Restore original weights
        for name, param in net.named_parameters():
            param.data.copy_(original_weights[name].data)
        
        loss = torch.mean(loss_base + wm_config.lambda_reg * loss_col)
        loss.backward()

        # CL regularization
        if wm_config.lambda_cl > 0:
            try:
                cl_loss = 0
                # Try aux in case there are no aux outputs yet
                aux_labels = np.load(wm_config.folder + "aux_predictions.npy")
                # Load aux dataset
                auxloader = load_aux_data()
                
                for batch_i, data in enumerate(auxloader):
                    if isinstance(data, (list, tuple)):
                        images_aux = data[0]  
                    else:
                        try:
                            images_aux = data["img"]
                        except:
                            images_aux = data["image"]
                    aux_labels_batch = aux_labels[batch_i*wm_config.mbs:(batch_i+1)*wm_config.mbs,:]
                    cl_loss += criterion_CL(F.log_softmax(net(images_aux.to(device)), dim=1), F.softmax(torch.from_numpy(aux_labels_batch).float().to(device), dim=1))

                # Calculate total loss and backpropagate
                loss = wm_config.lambda_cl * cl_loss

                loss.backward()
            except:
                pass

        optim.step()

    # Unfreeze batch norm layers
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(True)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(True)
            module.train()

    # Save watermarked weights and optimizer state
    p_record = ParametersRecord()
    for k, v in net.state_dict().items():
        # Convert to NumPy, then to Array. Add to record
        p_record[k] = array_from_numpy(v.detach().cpu().numpy())
    # Add to a context
    client_state.parameters_records["net_parameters"] = p_record
    
    if wm_config.measure_computation:
        if measure_client:
            _finish_measurement("wm_client", wm_client_start, device)
    return


## Leak and accusation functions ##

def check_prob_accusation(net,round,device="cpu"):
    """Check probability of accusation success during the training evolution"""
    """Only for the default BlackCATT configuration"""
    net.eval()
    # Load and normalize triggers
    triggers = _normalize_triggers(np.load(wm_config.folder + "triggers.npy"), device)
    labels = wm_config.clients_tardos_q

    try:
        df_all = pd.read_csv(wm_config.folder + "df_prob_acc.csv")  
    except:
        df_all = pd.DataFrame({"round":[],"c":[],"fn":[],"caught":[],"fp":[]})

    for c in (range(1,6)):
        fn = []
        caught = []
        fp = []
        for i in range(100):   
            net.eval()          
            # Choose random colluders
            colluders = sample(list(range(wm_config.n_users)),c)
            # Load the merged model 
            net = load_collusion(colluders,net,wm_config.folder,device=device)

            output = net(triggers)

            y = torch.argmax(output, dim=1).cpu().numpy()

            accused, t_score = catch_all_tardos_accusation(y,vectors=labels, p_secret=wm_config.p_secret, tau=wm_config.tau)

            fp_i = False 
            for guilty in accused:
                if guilty not in colluders:
                    fp_i = True

            fn.append(len(accused) == 0)
            caught.append(len(accused) if fp_i == False else 0)
            fp.append(fp_i)
                    
        df = pd.DataFrame({"round":[round for _ in range(len(fn))],"c":[c for _ in range(len(fn))],"fn":fn,"caught":caught,"fp":fp})
        df_all = pd.concat([df_all,df])

    df_all.to_csv(wm_config.folder + "df_prob_acc.csv",index=False)

def load_collusion(user_idxs, model, folder = wm_config.folder, mode = "average", pruning = 0, fine = 0, fine_dataloader = None, ft_lr = wm_config.mlr, device = "cuda",before_cid="client_status_",after_cid=".pkl"):
    if mode == "average":
        for cid_i, cid in enumerate(user_idxs):
            # Load weights from each client
            with open(folder + before_cid +str(cid)+ after_cid, 'rb') as f:
                client_status = pickle.load(f) 
                client_model = client_status["parameters"]
                client_model = [val for key_state, val in client_model.items()]
            if cid_i == 0:
                average_model = [val/len(user_idxs) for val in client_model]
            else:
                average_model = [average_val + (val/len(user_idxs)) for average_val,val in zip(average_model,client_model)]
        set_weights(model,average_model)
    elif mode == "randomselect":    
        multip_model = [[] for _ in range(len(get_weights(model)))]
        layer_index = np.random.choice(user_idxs, len(get_weights(model)), replace=True)  
        for cid_i, cid in enumerate(user_idxs):
            # Load weights from each client
            with open(folder + before_cid +str(cid)+ after_cid, 'rb') as f:
                client_status = pickle.load(f) 
                client_model = client_status["parameters"]
                client_model = [val for key_state, val in client_model.items()]
            for layer, index in enumerate(layer_index):
                if index == cid:
                    multip_model[layer] = client_model[layer]
        set_weights(model,multip_model)
    if pruning != 0:
        # Pruning attack - move to CPU to avoid GPU memory issues
        model.cpu()
        parameters_to_prune = []
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                parameters_to_prune.append((module, "weight"))
            if isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, "weight"))
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning,
        )
        model.to(device)
    if fine != 0:
        # Fine-tuning
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=ft_lr)
        for _ in range(fine):
            if type(fine_dataloader) is list:
                for fine_dataloader_i in fine_dataloader:
                    for batch in fine_dataloader_i:
                        images = batch["img"]
                        try:
                            labels = batch["label"]
                        except:
                            labels = batch["fine_label"]
                        optimizer.zero_grad()
                        loss = criterion(model(images.to(device)), labels.to(device))
                        loss.backward()
                        optimizer.step()
            else:
                for batch in fine_dataloader:
                    images = batch["img"]
                    try:
                        labels = batch["label"]
                    except:
                        labels = batch["fine_label"]
                    optimizer.zero_grad()
                    loss = criterion(model(images.to(device)), labels.to(device))
                    loss.backward()
                    optimizer.step()
        model.eval()
    return model



def tardos_score(x,y,p):
  # Calculate the Tardos score for a secuence of outputs 
  # and a specific client vector
  t_score = 0
  for i_p in range(p.shape[0]):
    if y[i_p] == x[i_p]:
      t_score += np.sqrt((1-p[i_p,y[i_p]])/p[i_p,y[i_p]])
    else:
      t_score -= np.sqrt((p[i_p,y[i_p]])/(1-p[i_p,y[i_p]]))
  return t_score


def tardos_accusation(y,vectors,p_secret,tau,pfp=0.000001):
  t_score = np.zeros(vectors.shape[0])

  for mi in range(vectors.shape[1]):
    # Calculate threshold according to Skoric and Oosterwijk 2012
    a = 1/(2*np.log(pfp))
    b = 1/(3*np.sqrt(tau))
    c = mi+1
    disc = b*b-4*a*c
    if disc >= 0:
      x1 = (-b+np.sqrt(disc))/(2*a)
      x2 = (-b-np.sqrt(disc))/(2*a)
      if x1 > 0 and x2 > 0:
        Z = min([x1,x2])
      elif x1 > 0:
        Z = x1
      elif x2 > 0:
        Z = x2
    
    # Calculate the Tardos score for each client
    for client_index in range(vectors.shape[0]):
      t_score[client_index] += tardos_score(vectors[client_index,mi-1:mi],y[mi-1:mi],p_secret[mi-1:mi])
    
    # Accuse if the score is above the threshold
    if max(t_score) > Z:
      return np.argmax(t_score) , mi+1
    
  return -1 , mi+1


def catch_all_tardos_accusation(y,vectors,p_secret,tau,pfp=0.000001):
    t_score = np.zeros(vectors.shape[0])

    # Calculate threshold according to Skoric and Oosterwijk 2012
    a = 1/(2*np.log(pfp))
    b = 1/(3*np.sqrt(tau))
    c = vectors.shape[1]
    disc = b*b-4*a*c
    if disc >= 0:
        x1 = (-b+np.sqrt(disc))/(2*a)
        x2 = (-b-np.sqrt(disc))/(2*a)
        if x1 > 0 and x2 > 0:
            Z = min([x1,x2])
        elif x1 > 0:
            Z = x1
        elif x2 > 0:
            Z = x2
    
    # Calculate the Tardos score for each client
    for client_index in range(vectors.shape[0]):
      t_score[client_index] += tardos_score(vectors[client_index,:],y[:],p_secret[:])
    
    # Accuse if the score is above the threshold
    return np.where(t_score > Z)[0], t_score