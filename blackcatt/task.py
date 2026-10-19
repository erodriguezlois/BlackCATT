"""blackcatt: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, LinearPartitioner
from flwr.common import ParametersRecord, array_from_numpy
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
import blackcatt.wm_config as wm_config
import pickle

## Main task specific functions ## 

fds = None  # Cache FederatedDataset

def load_data(partition_id: int, num_partitions: int, dataset = wm_config.dataset, iid = wm_config.iid, iid_beta = wm_config.iid_beta):
    global fds
    # Only initialize `FederatedDataset` once
    if iid == True:
        if wm_config.linear_partitioner == True:
            partitioner = LinearPartitioner(num_partitions=num_partitions)
        else:
            partitioner = IidPartitioner(num_partitions=num_partitions)
    else:
        if dataset == "CIFAR10":
            partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=iid_beta, partition_by="label")
        elif dataset == "CIFAR100":
            partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=iid_beta, partition_by="fine_label")
    if fds is None:
        if dataset == "CIFAR10":
            """Load partition CIFAR10 data."""
            fds = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={"train": partitioner},
            )
        elif dataset == "CIFAR100":
            """Load partition CIFAR100 data."""
            fds = FederatedDataset(
                dataset="uoft-cs/cifar100",
                partitioners={"train": partitioner},
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    if (dataset == "CIFAR10") or (dataset == "CIFAR100"):
        centralized_testset = fds.load_split("test")
        pytorch_transforms_train = Compose(
            [RandomCrop(32, padding=4),RandomHorizontalFlip(),ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        def apply_transforms_train(batch):
            """Apply transforms to the partition from FederatedDataset."""
            batch["img"] = [pytorch_transforms_train(img) for img in batch["img"]]
            return batch
        pytorch_transforms_test = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        def apply_transforms_test(batch):
            """Apply transforms to the partition from FederatedDataset."""
            batch["img"] = [pytorch_transforms_test(img) for img in batch["img"]]
            return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms_train)
    centralized_testset = centralized_testset.with_transform(apply_transforms_test)

    # If testing images are reserved, split part of the test set 
    if wm_config.reserve_testing_images:
        centralized_testset = torch.utils.data.Subset(centralized_testset, [i for i in range(500,len(centralized_testset))])

    trainloader = DataLoader(partition_train_test["train"], batch_size=wm_config.mbs, shuffle=True)
    testloader = DataLoader(centralized_testset, batch_size=wm_config.mbs)

    return trainloader, testloader


def train(net, client_state, trainloader, device, max_batches=wm_config.max_batches):
    """Train the model on the training set."""

    # Load model weights directly from ParametersRecord
    load_state_from_record(net, client_state)
    client_params = get_weights(net)
    
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=wm_config.mlr, momentum=0.9, weight_decay=0.0001)
 
    net.train()
    running_loss = 0.0    
    n_batches = 0
    if wm_config.linear_partitioner == True and wm_config.linear_part_epoch == True:
        # If there are different number of samples per client, we need to adjust the number of batches to train on
        max_batches = len(trainloader)
    while(n_batches < max_batches):
        # Iterate over batches
        for batch in trainloader:
            try:
                images = batch["img"]
            except:
                images = batch["image"]
            try:
                labels = batch["label"]
            except:
                labels = batch["fine_label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
            if n_batches >= max_batches:
                break
    avg_trainloss = running_loss / n_batches
    
    # Share incremental update
    incremental_update = [new - old for new, old in zip(get_weights(net), client_params)]
    return incremental_update, avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            try:
                images = batch["img"].to(device)
            except:
                images = batch["image"].to(device)
            try:
                labels = batch["label"].to(device)
            except:
                labels = batch["fine_label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


## Utility functions ##

def load_state_from_record(net, client_state):
    """Efficiently load model weights directly from ParametersRecord to state_dict"""
    p_record = client_state.parameters_records["net_parameters"]
    state_dict = OrderedDict({k: torch.from_numpy(v.numpy()) for k, v in p_record.items()})
    net.load_state_dict(state_dict, strict=True)

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    return


def read_parameters(cid):
    # Load from pickle
    with open(wm_config.folder + "client_status_"+str(cid)+".pkl", 'rb') as f:
        client_status = pickle.load(f) 
        parameters = client_status["parameters"]
    # Convert to Array
    parameters_record = ParametersRecord()
    for k, v in parameters.items():
        parameters_record[k] = array_from_numpy(v.numpy())
    return parameters_record 


def save_parameters(cid,client_state):
    from blackcatt.wm_task import _store_client_state
    
    # Extract state_dict directly from ParametersRecord
    p_record = client_state.parameters_records["net_parameters"]
    state_dict = {k: torch.from_numpy(v.numpy()) for k, v in p_record.items()}
    
    # Store in Ray if simulation mode (primary storage)
    _store_client_state(cid, state_dict)
    
    # Also save to disk as aux (to be renamed later)
    client_status = {"parameters": state_dict}
    with open(wm_config.folder + "aux_client_status_"+str(cid)+".pkl", 'wb') as f:
        pickle.dump(client_status,f)


## Handling flwr updates ##

def update_client(net,parameters,client_state):
    """Updates model copy with the new updated parameters"""
    # Extract and update parameters in one pass
    p_record = client_state.parameters_records["net_parameters"]
    new_weights = ParametersRecord()
    for (k, v), p in zip(p_record.items(), parameters):
        new_weights[k] = array_from_numpy(v.numpy() + p)

    # Load updated weights directly into model
    state_dict = OrderedDict({k: torch.from_numpy(v.numpy()) for k, v in new_weights.items()})
    net.load_state_dict(state_dict, strict=True)

    # Update context
    client_state.parameters_records["net_parameters"] = new_weights
    
    return