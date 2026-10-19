"""blackcatt: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from blackcatt.task import load_data, train, update_client, read_parameters, save_parameters
from blackcatt.wm_task import (
    wm_client,
    check_metrics,
    check_prob_accusation,
    _start_measurement,
    _finish_measurement,
)
import blackcatt.wm_config as wm_config
import blackcatt.models as models
import os
import numpy as np

# Define Flower Client with unique model copy and client_fn
class WM_FlowerClient(NumPyClient):
    def __init__(self, context: Context):
        # Load model and data
        if wm_config.model == "VGG16":
            self.net = models.VGG16()
        elif wm_config.model == "ResNet183x3":
            self.net = models.ResNet18()
        self.cid = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        self.trainloader, self.valloader = load_data(self.cid, num_partitions)
        self.local_epochs = context.run_config["local-epochs"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self.client_state = (
            context.state
        )
        # Initialize state if needed
        if "net_parameters" not in self.client_state.parameters_records:
            self.client_state.parameters_records["net_parameters"] = read_parameters(self.cid)

    def fit(self, parameters, config):
        if wm_config.measure_computation:
            training_start = _start_measurement(self.device)

        incremental_update, train_loss = train(
            self.net,
            self.client_state,
            self.trainloader,
            self.device,
        )
        if wm_config.measure_computation:
            _finish_measurement("local_training", training_start, self.device)

        # Return incremental update on the model parameters
        return (
            incremental_update,
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        ### Using this function to update and watermark all copies ###
        # While this is done server-side, it is implemented here to 
        # leverage the parallelization of the accuracy clients in flwr
        update_client(self.net,parameters,self.client_state)
        wm_client(self.net,self.cid,self.client_state,self.device)
        # Storing the model state can be done only every couple of rounds 
        # to prevent excessive data writing / reading of large models
        # IF there is no collusion-aware embedding, otherwise it needs to
        # be saved every round to keep track of the different model copies
        if config["current_round"] % 1 == 0:
            save_parameters(self.cid,self.client_state)
        # For evaluation purposes, we don't need this to run for all rounds / all cids
        if config["current_round"] % 25 == 0 and self.cid < min(10,wm_config.n_users-1):
            loss, val_items, accuracy = check_metrics(self.net,self.cid,self.valloader,self.device)
            return loss, val_items, {"accuracy": accuracy}
        # if config["current_round"] % 50 == 0 and self.cid == 0:
            # check_prob_accusation(self.net,config["current_round"],self.device)
            # return 0.0, 1, {"accuracy": 0.0}
        else:
            # flwr still wants to receive something
            return 0.0, 1, {"accuracy": 0.0}


def client_fn(context: Context):
    # Return Client instance
    return WM_FlowerClient(context).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
