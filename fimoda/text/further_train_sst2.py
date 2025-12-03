#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform further training runs for SST-2 dataset
"""

import argparse
from math import ceil
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModel

from fimoda.text.training_LM import further_train_all_runs

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

#%% Process arguments
parser = argparse.ArgumentParser()
parser.add_argument("seed_first", type=int, help="first random seed")
parser.add_argument("seed_last", type=int, help="one after last random seed")
parser.add_argument("--output_path", type=str, help="directory for saving output files")
args = parser.parse_args()
seed_first = args.seed_first
seed_last = args.seed_last

#%% Load processed SST-2 dataset

results_path = "results"
dataset_name = "sst2"
dataset_train = torch.load(os.path.join(results_path, dataset_name, "dataset_train.pt"), weights_only=False)
dataset_val = torch.load(os.path.join(results_path, dataset_name, "dataset_val.pt"), weights_only=False)
print(dataset_train[:2]["input_ids"])

#%% Define BERT model

model_name = "bert-base-uncased"
n_classes = 2

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_dict):
        last_hidden_state = self.model(**input_dict)[0]
        cls_output = last_hidden_state[:, 0]
        logits = self.classifier(cls_output)
        return logits

#%% Arguments for further_train_all_runs

# Training data parameters
seeds = list(range(seed_first, seed_last))
batch_size = 64
ind_leave_out = torch.load(os.path.join(results_path, dataset_name, "ind_leave_out.pt"))

# Model parameters
model_class = Model
model_path = os.path.join(results_path, dataset_name, "model_final.pth")

# Optimizer parameters
loss_fn = nn.CrossEntropyLoss()
optimizer_class = torch.optim.AdamW
learning_rate = 1e-6    # factor of 10 smaller than for initial training
weight_decay = 0.
num_epochs = 1

# Test data parameters
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, sampler=SequentialSampler(dataset_val))
loss_fn_per_sample = nn.CrossEntropyLoss(reduction="none")
# num_steps = ceil(len(dataset_train) / batch_size)
# num_steps_eval = torch.cat((torch.arange(0, 200, 2), torch.arange(200, num_steps, 10)))
num_steps_eval = 5

# Folder for saving validation loss Tensors
output_path = os.path.join(results_path, dataset_name) if args.output_path is None else args.output_path

#%% Perform further training runs
val_loss_further_train_full, val_loss_further_train_loo =\
    further_train_all_runs(dataset_train, seeds, batch_size, ind_leave_out, model_class, model_path,
                           loss_fn, optimizer_class, learning_rate, weight_decay, num_epochs,
                           dataloader_val, loss_fn_per_sample, num_steps_eval, output_path, leave_out_every_batch=True)

# Save validation loss Tensors
if len(seeds) == 1:
    seed_string = str(seed_first)
else:
    seed_string = str(seed_first) + "_" + str(seed_last)
torch.save(val_loss_further_train_full, os.path.join(output_path, f"val_loss_further_train_full_{seed_string}.pt"))
torch.save(val_loss_further_train_loo, os.path.join(output_path, f"val_loss_further_train_loo_{seed_string}.pt"))
