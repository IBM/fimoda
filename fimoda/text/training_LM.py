#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training functions for language models
"""

from math import ceil
import os
import torch
from torch import nn
from torch.utils.data import RandomSampler, DataLoader
from tqdm import tqdm

from fimoda.tabular.training import RandomSamplerLOO

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def eval_per_sample(dataloader, model, loss_fn_per_sample):
    """
    Evaluate per-sample losses

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader
    model : torch.nn.Module
        Model
    loss_fn_per_sample : torch.nn loss function
        Per-sample loss function

    Returns
    -------
    loss_per_sample : (num_samples,) torch.Tensor
        Loss for each sample
    pred : (num_samples,) torch.Tensor
        Predicted class for each sample

    """
    model.eval()

    loss_per_sample = torch.tensor([], device=device)
    pred = torch.tensor([], dtype=int, device=device)
    # Iterate over batches
    with torch.no_grad():
        for batch in dataloader:
            # Separate labels
            labels = batch.pop("label")
            # Move to GPU
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            labels = labels.to(device)
            # Compute prediction and loss
            logits = model(batch)
            loss_per_sample = torch.cat((loss_per_sample, loss_fn_per_sample(logits, labels)))
            pred = torch.cat((pred, logits.argmax(dim=1)))

    # Return loss and prediction per sample
    return loss_per_sample, pred

def train_one_epoch(dataloader, model, loss_fn, optimizer, max_grad_norm=1.):
    """
    Train model for one epoch

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Training set DataLoader
    model : torch.nn.Module
        Model
    loss_fn : torch.nn loss function
        Loss function
    optimizer : torch.optim optimizer
        Optimizer
    max_grad_norm : float
        Threshold on gradient norm for clipping

    Returns
    -------
    train_loss_mean : float
        Training loss averaged over batches

    """
    model.train()

    train_loss = 0
    # Iterate over batches
    for batch in tqdm(dataloader):
        # Separate labels
        labels = batch.pop("label")
        # Move to GPU
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        labels = labels.to(device)
        # Compute prediction and loss
        logits = model(batch)
        loss = loss_fn(logits, labels)
        train_loss += loss.item()

        # Compute gradients and update parameters
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm > 0.:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

    # Return training loss averaged over batches
    return train_loss / len(dataloader)

def further_train_one_epoch(dataloader_train, model, loss_fn, optimizer, dataloader_val, loss_fn_per_sample, num_steps_eval, inst_leave_out=None, max_grad_norm=1.):
    """
    Train model for one epoch and evaluate per-sample losses after every num_steps_eval steps

    Parameters
    ----------
    dataloader_train : torch.utils.data.DataLoader
        Training set DataLoader
    model : torch.nn.Module
        Model
    loss_fn : torch.nn loss function
        Loss function
    optimizer : torch.optim optimizer
        Optimizer
    dataloader_val : torch.utils.data.DataLoader
        Validation set DataLoader
    loss_fn_per_sample : torch.nn loss function
        Per-sample loss function
    num_steps_eval : int or sequence
        If int, number of training steps between evaluations
        If sequence, the numbers of training steps at which evaluations occur
    inst_leave_out : dict or None
        Training instance to leave out of every batch
    max_grad_norm : float
        Threshold on gradient norm for clipping

    Returns
    -------
    train_loss_mean : float
        Training loss averaged over batches
    val_loss_per_sample : (num_evals, num_val_samples) torch.Tensor
        For each evaluation, loss for each validation sample

    """
    train_loss = 0
    # Determine number of evaluations
    if type(num_steps_eval) is int:
        num_evals = ceil(len(dataloader_train) / num_steps_eval)
    else:
        # Use given num_steps_eval, plus last step if not in num_steps_eval
        num_evals = len(num_steps_eval) + (len(dataloader_train) - 1 not in num_steps_eval)
    val_loss_per_sample = torch.empty((num_evals, len(dataloader_val.dataset)), device=device)

    if inst_leave_out is not None:
        # Separate label from training instance to leave out of every batch
        label_leave_out = inst_leave_out.pop("label")
        num_train = len(dataloader_train.dataset)

    # Iterate over batches
    e = 0
    for b, batch in tqdm(enumerate(dataloader_train)):
        model.train()

        # Separate labels
        labels = batch.pop("label")
        # Move to GPU
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        labels = labels.to(device)
        # Compute prediction and loss
        logits = model(batch)
        loss = loss_fn(logits, labels)
        if inst_leave_out is not None:
            # Subtract loss on instance to leave out (appropriately weighted)
            logits_leave_out = model(inst_leave_out)
            loss_leave_out = loss_fn(logits_leave_out, label_leave_out)
            loss -= loss_leave_out / num_train
        train_loss += loss.item()

        # Compute gradients and update parameters
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm > 0.:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # Evaluate per-sample validation losses if current step meets conditions
        if type(num_steps_eval) is int and b > 0 and b % num_steps_eval == 0\
            or type(num_steps_eval) is not int and b in num_steps_eval or b == len(dataloader_train) - 1:
            val_loss_per_sample[e, :], _ = eval_per_sample(dataloader_val, model, loss_fn_per_sample)
            e += 1

    # Return training loss averaged over batches and per-sample validation losses
    return train_loss / len(dataloader_train), val_loss_per_sample

def further_train_one_run(dataset_train, seed, batch_size, index_leave_out, model_class, model_path,
                          loss_fn, optimizer_class, learning_rate, weight_decay, num_epochs,
                          dataloader_val, loss_fn_per_sample, num_steps_eval, leave_out_every_batch=True):
    """
    Train saved model on full or leave-one-out training set
    Evaluate per-sample validation losses after every num_steps_eval steps

    Parameters
    ----------
    dataset_train : torch.utils.data.Dataset
        Training dataset
    seed : int
        Random seed for permuting training dataset
    batch_size : int
        Batch size
    index_leave_out : int or None
        Index of training sample to leave out
    model_class : nn.Module
        Model class for instantiating a new model
    model_path : str
        Path to saved model parameters
    loss_fn : torch.nn loss function
        Training loss function
    optimizer_class : torch.optim optimizer
        Optimizer class for training
    learning_rate : float
        Learning rate
    weight_decay : float
        Weight decay parameter
    num_epochs : int
        Number of training epochs
    dataloader_val : torch.utils.data.DataLoader
        Validation set DataLoader
    loss_fn_per_sample : torch.nn loss function
        Per-sample loss function
    num_steps_eval : int or sequence
        If int, number of training steps between evaluations
        If sequence, the numbers of training steps at which evaluations occur
    leave_out_every_batch : bool
        Whether to leave out training sample in every batch

    Returns
    -------
    val_loss_per_sample : (num_epochs, num_evals, num_val_samples) torch.Tensor
        For each evaluation, loss for each validation sample

    """
    if leave_out_every_batch:
        if index_leave_out is not None:
            # Extract training instance to leave out of every batch
            inst_leave_out = dataset_train[index_leave_out.unsqueeze(0)]
            # Move to GPU
            for key in inst_leave_out.keys():
                inst_leave_out[key] = inst_leave_out[key].to(device)
        else:
            inst_leave_out = None

    # Construct training DataLoader with given random seed for permuting and index to leave out
    generator = torch.Generator()
    generator.manual_seed(seed)
    if index_leave_out is None or leave_out_every_batch:
        # Full training set or leave one instance out of every batch
        sampler = RandomSampler(dataset_train, generator=generator)
    else:
        # Leave one instance out of one batch
        sampler = RandomSamplerLOO(dataset_train, generator=generator, index_leave_out=index_leave_out)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler)

    # Load saved model parameters
    torch.manual_seed(0)    # makes dropout reproducible?
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path))

    # Optimizer
    optimizer = optimizer_class(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Loop over epochs
    if type(num_steps_eval) is int:
        num_evals = ceil(len(dataloader_train) / num_steps_eval)
    else:
        num_evals = len(num_steps_eval) + (len(dataloader_train) - 1 not in num_steps_eval)
    val_loss_per_sample = torch.empty((num_epochs, num_evals, len(dataloader_val.dataset)), device=device)
    for t in range(num_epochs):
        train_loss, val_loss_per_sample[t, :, :] =\
            further_train_one_epoch(dataloader_train, model, loss_fn, optimizer, dataloader_val, loss_fn_per_sample, num_steps_eval,
                                    inst_leave_out=inst_leave_out if leave_out_every_batch else None)

    if index_leave_out is None:
        print(f"Final average training loss per sample: {train_loss}")

    return val_loss_per_sample

def further_train_all_runs(dataset_train, seeds, batch_size, indices_leave_out, model_class, model_path,
                           loss_fn, optimizer_class, learning_rate, weight_decay, num_epochs,
                           dataloader_val, loss_fn_per_sample, num_steps_eval, results_path=None, leave_out_every_batch=True):
    """
    Train saved model using all combinations of random seeds and left-out samples
    Evaluate per-sample validation losses after every num_steps_eval steps

    Parameters
    ----------
    dataset_train : torch.utils.data.Dataset
        Training dataset
    seeds : List[int]
        Random seeds for permuting training dataset
    batch_size : int
        Batch size
    indices_leave_out : List[int] or None
        Indices of training samples to leave out, `None` for all
    model_class : nn.Module
        Model class for instantiating a new model
    model_path : str
        Path to saved model parameters
    loss_fn : torch.nn loss function
        Training loss function
    optimizer_class : torch.optim optimizer
        Optimizer class for training
    learning_rate : float
        Learning rate
    weight_decay : float
        Weight decay parameter
    num_epochs : int
        Number of training epochs
    dataloader_val : torch.utils.data.DataLoader
        Validation set DataLoader
    loss_fn_per_sample : torch.nn loss function
        Per-sample loss function
    num_steps_eval : int or sequence
        If int, number of training steps between evaluations
        If sequence, the numbers of training steps at which evaluations occur
    results_path : str or None
        Folder for saving test loss Tensors in
    leave_out_every_batch : bool
        Whether to leave out training sample in every batch

    Returns
    -------
    val_loss_per_sample_full : (num_epochs, num_evals, num_val_samples, 1, len(seeds)) torch.Tensor
        For each evaluation, loss for each validation sample after training on full dataset
    val_loss_per_sample_loo : (num_epochs, num_evals, num_val_samples, len(indices_leave_out), len(seeds)) torch.Tensor
        For each evaluation, loss for each validation sample after training on leave-one-out dataset

    """
    # Number of training steps, evaluations, and validation samples
    num_steps = ceil(len(dataset_train) / batch_size)
    if type(num_steps_eval) is int:
        num_evals = ceil(num_steps / num_steps_eval)
    else:
        num_evals = len(num_steps_eval) + (num_steps - 1 not in num_steps_eval)
    num_samples_val = len(dataloader_val.dataset)
    if indices_leave_out is None:
        # Leave out all training samples in turn
        indices_leave_out = range(len(dataset_train))
    num_leave_out = len(indices_leave_out)
    num_seeds = len(seeds)
    print(num_steps, num_evals, num_leave_out, num_seeds)

    # Initialize
    val_loss_per_sample_full = torch.empty((num_epochs, num_evals, num_samples_val, 1, num_seeds), device=device)
    val_loss_per_sample_loo = torch.empty((num_epochs, num_evals, num_samples_val, num_leave_out, num_seeds), device=device)
    if len(seeds) == 1:
        seed_string = str(seeds[0])
    else:
        seed_string = str(seeds[0]) + "_" + str(seeds[-1])

    # Iterate over random seeds
    for s, seed in enumerate(seeds):
        print(f"Using random seed {seed}")
        # Train on full training set
        print("Training on full training set:")
        val_loss_per_sample_full[:, :, :, 0, s] =\
            further_train_one_run(dataset_train, seed, batch_size, None, model_class, model_path,
                                  loss_fn, optimizer_class, learning_rate, weight_decay, num_epochs,
                                  dataloader_val, loss_fn_per_sample, num_steps_eval)

        if results_path is not None:
            torch.save(val_loss_per_sample_full, os.path.join(results_path, f"val_loss_further_train_full_{seed_string}.pt"))

        # Iterate over left-out samples
        for l, index_leave_out in tqdm(enumerate(indices_leave_out)):
            print(f"Training on leave-one-out dataset {l} (index {index_leave_out}):")
            # Train on leave-one-out dataset
            val_loss_per_sample_loo[:, :, :, l, s] =\
                further_train_one_run(dataset_train, seed, batch_size, index_leave_out, model_class, model_path,
                                      loss_fn, optimizer_class, learning_rate, weight_decay, num_epochs,
                                      dataloader_val, loss_fn_per_sample, num_steps_eval, leave_out_every_batch=leave_out_every_batch)

            if results_path is not None:
                torch.save(val_loss_per_sample_loo, os.path.join(results_path, f"val_loss_further_train_loo_{seed_string}.pt"))

    return val_loss_per_sample_full, val_loss_per_sample_loo
