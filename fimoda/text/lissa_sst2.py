#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run LiSSA on BERT model and SST-2 dataset
"""

import argparse
import os
import torch
from torch import nn
from torch.backends.cuda import sdp_kernel
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModel
from tqdm import tqdm

from torch_influence import BaseObjective, LiSSAInfluenceModule

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# %% Process arguments
parser = argparse.ArgumentParser()
parser.add_argument("--inst_first", type=int, help="first test instance")
parser.add_argument("--inst_last", type=int, help="last test instance")
parser.add_argument("--output_path", type=str, help="directory for saving output files")
parser.add_argument("--damp", type=float, default=0.01, help="Hessian damping parameter")
parser.add_argument("--gnh", action="store_true", help="use Gauss-Newton approximation to Hessian")

args = parser.parse_args()
inst_first = args.inst_first
inst_last = args.inst_last
damp = args.damp
gnh = args.gnh
gnh_string = "" if gnh else "_no_gnh"

#%% Load CIFAR-10 dataset

dataset_name = "sst2"
results_path = "results"
dataset_path = os.path.join(results_path, dataset_name)
output_path = dataset_path if args.output_path is None else args.output_path

dataset_train = torch.load(os.path.join(dataset_path, "dataset_train.pt"), weights_only=False)
dataset_val = torch.load(os.path.join(dataset_path, "dataset_val.pt"), weights_only=False)

# DataLoaders
batch_size = 64
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=SequentialSampler(dataset_train))
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, sampler=SequentialSampler(dataset_val))

# Indices of left-out training instances
ind_leave_out = torch.load(os.path.join(dataset_path, "ind_leave_out.pt"))
num_leave_out = len(ind_leave_out)

# Number of test instances for evaluation
num_eval = 100

#%% Load BERT model
model_name = "bert-base-uncased"
n_classes = 2

class Model(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_dict):
        last_hidden_state = self.model(**input_dict)[0]
        cls_output = last_hidden_state[:, 0]
        logits = self.classifier(cls_output)
        return logits

torch.manual_seed(0)    # makes dropout and batch norm reproducible?
model = Model(model_name).to(device)
model.load_state_dict(torch.load(os.path.join(dataset_path, "model_final.pth")))

# Disable gradient for unused pooling layers to avoid error
for name, param in model.named_parameters():
    if "pooler" in name:
        param.requires_grad = False

#%% Fixed torch_influence parameters

class CrossEntropyObjective(BaseObjective):

    def train_outputs(self, model, batch):
        batch_copy = batch.copy()
        # Separate labels
        labels = batch_copy.pop("label")
        return model(batch_copy)

    def train_loss_on_outputs(self, outputs, batch):
        return F.cross_entropy(outputs, batch["label"])  # mean reduction required

    def train_regularization(self, params):
        return torch.tensor(0.)

    # training loss by default taken to be
    # train_loss_on_outputs + train_regularization

    def test_loss(self, model, params, batch):
        batch_copy = batch.copy()
        # Separate labels
        labels = batch_copy.pop("label")
        return F.cross_entropy(model(batch_copy), labels)  # no regularization in test loss

# LiSSA parameters
repeat = 1
depth = 5000
scale = 500    # smallest value in [10, 20, 50, 100, 150, etc.] for which LiSSA converges

#%% Compute influence scores

# Instantiate LiSSA module
LiSSA_module = LiSSAInfluenceModule(model=model, objective=CrossEntropyObjective(), train_loader=dataloader_train, test_loader=dataloader_val,
                                    device=device, damp=damp, gnh=gnh, repeat=repeat, depth=depth, scale=scale)

lissa = torch.zeros((num_eval, num_leave_out), device=device)
# Iterate over test instances
for i in tqdm(range(inst_first, inst_last + 1)):
    # sdp_kernel below is a work-around from https://github.com/pytorch/pytorch/issues/117974
    # to deal with unimplemented double backward for aten::_scaled_dot_product_efficient_attention
    with sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
        lissa[i, :] = LiSSA_module.influences(ind_leave_out, [i])
    torch.save(lissa, os.path.join(output_path, f"lissa{gnh_string}_damp{damp}_inst{inst_first}_{inst_last}.pt"))
