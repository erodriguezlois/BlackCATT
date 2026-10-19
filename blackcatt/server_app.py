"""blackcatt: A Flower / PyTorch app."""

from flwr.common import Context, FitIns, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from blackcatt.task import get_weights
from blackcatt.wm_task import evaluate_config, update_triggers, load_aux_data
from blackcatt.checkpoint import load_checkpoint, is_checkpoint_available, find_latest_checkpoint
from torchvision.transforms import ToPILImage
import blackcatt.wm_config as wm_config
import blackcatt.models as models
import pickle
import random
import torch
import numpy as np

def fit_config(server_round: int) -> dict[str, int]:
    return {"current_round": server_round}


def client_dropout_probability(client_cid: str, failure_ratio: float) -> float:
    risk = random.Random(client_cid).random()
    variation = wm_config.failure_var * min(failure_ratio, 1.0 - failure_ratio)
    return failure_ratio + (2.0 * risk - 1.0) * variation


class DropoutFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_outage_until = {}
        self.min_failure_rounds = wm_config.fit_failure_min_rounds
        self.max_failure_rounds = wm_config.fit_failure_max_rounds
        
    def _update_global_outages(self, server_round, all_client_cids):
        """Evaluates all clients in the system every round, 
        independent of whether they were sampled."""
        failure_ratio = wm_config.fit_failure_ratio
        
        for cid in all_client_cids:
            outage_until = self.client_outage_until.get(cid)
            
            # If already in an outage, check if it expired
            if outage_until is not None:
                if server_round > outage_until:
                    del self.client_outage_until[cid] # Recovered!
                continue # Still offline, skip rolling new outage

            # If active, check if they drop out this round
            rng = random.Random(f"{server_round}:{cid}")
            if rng.random() < client_dropout_probability(cid, failure_ratio):
                duration = rng.randint(self.min_failure_rounds, self.max_failure_rounds)
                self.client_outage_until[cid] = server_round + duration - 1

    def configure_fit(self, server_round, parameters, client_manager):
        config = (
            self.on_fit_config_fn(server_round)
            if self.on_fit_config_fn is not None
            else {}
        )
        fit_ins = FitIns(parameters, config)

        all_clients = client_manager.sample(
            num_clients=wm_config.n_users
        )
        all_client_cids = [str(int(client.cid)) for client in all_clients]

        self._update_global_outages(server_round, all_client_cids)

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        rng = random.Random(server_round)
        max_dropouts = max(0, len(clients) - min_num_clients)
        dropped_cids = {str(int(client.cid)) for client in clients if str(int(client.cid)) in self.client_outage_until}
        if len(dropped_cids) > max_dropouts:
                    # Respect max_dropouts safety cap if needed
                    dropped_cids = set(rng.sample(sorted(dropped_cids), max_dropouts))
        
        participating_clients = [
            client for client in clients if str(int(client.cid)) not in dropped_cids
        ]

        print(
            f"[SIM] Round {server_round}: omitted {len(dropped_cids)} fit clients; "
            f"dispatching {len(participating_clients)}"
        )
        return [(client, fit_ins) for client in participating_clients]

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    
    # Track the starting round for round number adjustments
    starting_round = 0

    # Initialize model parameters
    if wm_config.model == "VGG16":
        net = models.VGG16()
    elif wm_config.model == "ResNet183x3":
        net = models.ResNet18()
    
    # Check if we should load from checkpoint
    if wm_config.load_checkpoint_flag:
        if is_checkpoint_available(wm_config.folder):
            try:
                # Load checkpoint (latest if not specified)
                client_states, triggers, loaded_round = load_checkpoint(
                    wm_config.folder,
                    wm_config.n_users,
                    wm_config.loading_round
                )
                
                # Initialize parameters from the first client's state
                parameters = ndarrays_to_parameters([
                    val.numpy() if isinstance(val, torch.Tensor) else val
                    for key, val in client_states[0].items()
                ])
                
                # Set starting round to continue from where we left off
                starting_round = loaded_round + 1
                
                # Adjust num_rounds to only run remaining rounds
                num_rounds = num_rounds - loaded_round
                
                # Restore triggers
                np.save(wm_config.folder + "triggers.npy", triggers)
                
                # Restore all client states
                for i_cid in range(wm_config.n_users):
                    client_status = {"parameters": client_states[i_cid]}
                    with open(wm_config.folder + "client_status_" + str(i_cid) + ".pkl", 'wb') as f:
                        pickle.dump(client_status, f)
                
                print(f"[INFO] Loaded checkpoint from round {loaded_round}")
                print(f"[INFO] Starting from round {starting_round}, running {num_rounds} remaining rounds")
                
            except Exception as e:
                raise RuntimeError(
                    f"[ERROR] Checkpoint loading failed: {e}\n"
                    f"Aborting to prevent data loss. Your experiment data is safe.\n"
                    f"Please verify checkpoint files exist and are not corrupted."
                )
        else:
            raise ValueError("Load-checkpoint=True but no checkpoints found")
    else:
        # Normal initialization (no checkpoint loading)
        parameters = ndarrays_to_parameters(get_weights(net))
        # Initialize random triggers
        if (wm_config.dataset == "CIFAR10") or (wm_config.dataset == "CIFAR100"):
            if wm_config.trigger_type == "random":
                triggers = np.random.randint(0, 255, size=(wm_config.m, 32, 32, 3))
            elif wm_config.trigger_type == "unique":
                triggers = np.random.randint(0, 255, size=(wm_config.m*wm_config.n_users, 32, 32, 3))
            elif wm_config.trigger_type == "stealthy":
                if wm_config.dataset_cl == "uoft-cs/cifar10":
                    raise ValueError("There are currently no CIFAR10 samples reserved for stealthy triggers")
                auxloader = load_aux_data(1, wm_config.m)  # So that they are different images to FR
                # Get 1 batch of images (wm_config.m samples) from aux data
                for data in auxloader:
                    if isinstance(data, (list, tuple)):
                        triggers = data[0]  
                    else:
                        try:
                            triggers = data["img"]
                        except:
                            triggers = data["image"]
                    triggers = triggers * 0.5 + 0.5
                    triggers = [ToPILImage()(trigger) for trigger in triggers]
                    triggers = np.array(triggers).astype(np.uint8)
                    if wm_config.stealth_ratio != 1.0:
                        # Add noise to the triggers
                        noise = np.random.randint(0, 255, size=(wm_config.m, 32, 32, 3))
                        triggers = (triggers.astype(np.float32) * wm_config.stealth_ratio + noise.astype(np.float32) * (1 - wm_config.stealth_ratio)).astype(np.uint8)
                    break
            else:
                raise ValueError("Unsupported trigger type: {}".format(wm_config.trigger_type))
        else:
            raise ValueError("Unsupported dataset: {}".format(wm_config.dataset))
        # Also initialize all clients with their unique model copy
        for i_cid in range(wm_config.n_users):
            client_status = {"parameters": net.state_dict()}
            with open(wm_config.folder + "client_status_" + str(i_cid) + ".pkl", 'wb') as f:
                pickle.dump(client_status, f)
        # Store initialization of triggers
        np.save(wm_config.folder + "trigger_round_0.npy", triggers)
        np.save(wm_config.folder + "triggers.npy", triggers)
    
    net = None

    # Define strategy
    # Watermarking is done during evaluation phase to leverage parallelization
    # so fraction_evaluate is kept to 1.0
    # After evaluation update_triggers in launched to optimize the trigger set
    # on the updated and watermarked model copies
    strategy = DropoutFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        min_fit_clients=2,
        accept_failures=True,
        on_fit_config_fn=fit_config,
        initial_parameters=parameters,
        on_evaluate_config_fn=evaluate_config,
        evaluate_metrics_aggregation_fn=update_triggers,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
