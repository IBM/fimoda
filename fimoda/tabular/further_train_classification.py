#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform further training runs for tabular classification datasets
"""

import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler

from fimoda.tabular.training import train_all_runs

#%% Process arguments
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, help="name of dataset")
parser.add_argument("--seed_first", type=int, default=0, help="first random seed")
parser.add_argument("--seed_last", type=int, default=100, help="one after last random seed")
parser.add_argument("--optimizer", type=str, default="sgd", help="training optimizer")
args = parser.parse_args()

#%% Load processed dataset

dataset = args.dataset
dataset_train = torch.load(os.path.join("results", dataset, "dataset_train.pt"), weights_only=False)
dataset_test = torch.load(os.path.join("results", dataset, "dataset_test.pt"), weights_only=False)
print(dataset_train[0][0][:10], dataset_train[0][1])

num_features = dataset_train.tensors[0].shape[1]

#%% Define MLP model

hidden_dim = 128

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
    def forward(self, x):
        return self.linear_relu_stack(x)

#%% Arguments for train_all_runs

# Training data parameters
seed_first = args.seed_first
seed_last = args.seed_last
seeds = list(range(seed_first, seed_last))
batch_size = 128
indices_leave_out = torch.load(os.path.join("results", dataset, "ind_loo.pt"))

# Model parameters
model_class = MLP
model_path = os.path.join("results", dataset, "model_final.pth")

# Optimizer parameters
loss_fn = nn.CrossEntropyLoss()
optimizer_name = args.optimizer
if optimizer_name == "sgd":
    optimizer_class = torch.optim.SGD
    learning_rate = 1e-4
elif optimizer_name == "adam":
    optimizer_class = torch.optim.Adam
    learning_rate = 1e-6
if dataset == "fico":
    num_epochs = 200
elif dataset == "folktables":
    num_epochs = 25

# Test data parameters
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, sampler=SequentialSampler(dataset_test))
loss_fn_per_sample = nn.CrossEntropyLoss(reduction="none")

# Folder for saving test loss Tensors
results_path = os.path.join("results", dataset)

#%% Perform further training runs
test_loss_further_train_full, test_loss_further_train_loo, params_dist_further_train_full, params_dist_further_train_loo =\
    train_all_runs(dataset_train, seeds, batch_size, indices_leave_out, model_class, model_path,
                   loss_fn, optimizer_class, learning_rate, num_epochs, dataloader_test, loss_fn_per_sample, results_path)

seed_string = str(seed_first) + "_" + str(seed_last)
torch.save(test_loss_further_train_full, os.path.join(results_path, f"test_loss_further_train_{optimizer_name}_full_{seed_string}.pt"))
torch.save(test_loss_further_train_loo, os.path.join(results_path, f"test_loss_further_train_{optimizer_name}_loo_{seed_string}.pt"))
