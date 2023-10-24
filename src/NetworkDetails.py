from dataclasses import dataclass
from network import AlexNet
import torch
from pathlib import Path

@dataclass
class TrainingDetails:
    batch_size: int
    learning_rate: float
    momentum: float
    epochs: int
    output_dir: Path

    def __str__(self):
        lr_str = str(self.learning_rate).replace(".", "_")
        momentum_str = str(self.momentum).replace(".", "_")
        return f"batch_{self.batch_size}_lr_{lr_str}_momentum_{momentum_str}_epochs_{self.epochs}"


def _get_scheduler_string(scheduler):
    if scheduler is None:
        return "Scheduler: Unused"
    else:
        return f"Scheduler Type: {type(scheduler)}\nScheduler Parameters: {scheduler.__dict__}"


def _get_optimizer_string(optimizer):
    if optimizer is None:
        return "Optimizer: Unused"
    else:
        return f"Optimizer Type: {type(optimizer)}\nOptimizer Parameters: {optimizer.__dict__}"


def _get_criterion_string(criterion):
    if criterion is None:
        return "Criterion: Unused"
    else:
        return f"Criterion Type: {type(criterion)}\nCriterion Parameters: {criterion.__dict__}"


def save_model_information(network: AlexNet, output_directory: Path, optimizer, criterion, scheduler):
    torch.save(network.state_dict(), f=output_directory / "project2.pth")
    network_details = Path("network_details.txt")
    with open(output_directory / network_details, "w") as file:
        file.write(_get_scheduler_string(scheduler=scheduler))
        file.write(_get_optimizer_string(optimizer=optimizer))
        file.write(_get_criterion_string(criterion=criterion))
