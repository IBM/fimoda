#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training functions and classes
"""

import os
import torch
from torch.utils.data import RandomSampler, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class RandomSamplerLOO(RandomSampler):
    """
    Modification of PyTorch's RandomSampler to keep sample order the same except for leaving out one sample
    """
    def __init__(self, data_source, replacement=False, num_samples=None, generator=None, index_leave_out=None):
        super().__init__(data_source, replacement, num_samples, generator)
        self.index_leave_out = index_leave_out
    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                indices = torch.randperm(n, generator=generator).tolist()
                # Leave out sample
                if self.index_leave_out in indices:
                    indices.remove(self.index_leave_out)
                yield from indices
            indices = torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]
            # Leave out sample
            if self.index_leave_out in indices:
                indices.remove(self.index_leave_out)
            yield from indices

def train_one_epoch(dataloader, model, loss_fn, optimizer):
    """
    Train model for one epoch

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Test set DataLoader
    model : torch.nn.Module
        Model
    loss_fn : torch.nn loss function
        Loss function
    optimizer : torch.optim optimizer
        Optimizer

    Returns
    -------
    train_loss_mean : float
        Training loss averaged over batches

    """
    model.train()

    train_loss = 0
    # Iterate over batches
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        #print(f"Loss: {loss}")

        # Compute gradients and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Return loss averaged over batches
    return train_loss / len(dataloader)

def test_per_sample(dataloader, model, loss_fn_per_sample):
    """
    Evaluate per-sample losses on test set

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Test set DataLoader
    model : torch.nn.Module
        Model
    loss_fn_per_sample : torch.nn loss function
        Per-sample loss function

    Returns
    -------
    test_loss_per_sample : (num_test_samples,) torch.Tensor
        Loss for each test sample

    """
    model.eval()

    test_loss_per_sample = torch.tensor([], device=device)
    # Iterate over batches
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # Compute prediction and loss
            pred = model(X)
            test_loss_per_sample = torch.cat((test_loss_per_sample, loss_fn_per_sample(pred, y)))

    # Return loss per sample
    return test_loss_per_sample

def train_one_run(dataset_train, seed, batch_size, index_leave_out, model_class, model_path,
                  loss_fn, optimizer_class, learning_rate, num_epochs, dataloader_test, loss_fn_per_sample):
    """
    Train saved model on full or leave-one-out training set
    Evaluate per-sample test losses after each epoch

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
    num_epochs : int
        Number of epochs
    dataloader_test : torch.utils.data.DataLoader
        Test set DataLoader
    loss_fn_per_sample : torch.nn loss function
        Per-sample test loss function

    Returns
    -------
    test_loss_per_sample : (num_epochs, num_test_samples) torch.Tensor
        Loss for each test sample after each epoch
    model : nn.Module
        Trained model
    params_dist : (num_epochs,) torch.Tensor
        Euclidean distance from initial model parameters after each epoch

    """
    # Construct training DataLoader with given random seed for permuting and index to leave out
    generator = torch.Generator()
    generator.manual_seed(seed)
    if index_leave_out is None:
        # Full training set
        sampler = RandomSampler(dataset_train, generator=generator)
    else:
        # Leave-one-out training set
        sampler = RandomSamplerLOO(dataset_train, generator=generator, index_leave_out=index_leave_out)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler)

    # Load saved model
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path))
    # Initial model parameters
    params_init = tuple(v.detach().clone() for v in model.parameters())

    # Optimizer
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    # Loop over epochs
    test_loss_per_sample = torch.empty((num_epochs, len(dataloader_test.dataset)), device=device)
    params_dist = torch.zeros(num_epochs, device=device)
    for t in range(num_epochs):
        train_loss = train_one_epoch(dataloader_train, model, loss_fn, optimizer)
        test_loss_per_sample[t, :] = test_per_sample(dataloader_test, model, loss_fn_per_sample)

        # Measure Euclidean distance between current and initial parameters
        for v, v_init in zip(model.parameters(), params_init):
            # Add sum of squared differences for this layer
            params_dist[t] += ((v.detach() - v_init) ** 2).sum()
        params_dist[t] = torch.sqrt(params_dist[t])

    if index_leave_out is None:
        print(f"Final average training loss per sample: {train_loss}")

    return test_loss_per_sample, model, params_dist

def train_all_runs(dataset_train, seeds, batch_size, indices_leave_out, model_class, model_path,
                   loss_fn, optimizer_class, learning_rate, num_epochs, dataloader_test, loss_fn_per_sample, results_path=None):
    """
    Train saved model using all combinations of random seeds and left-out samples
    Evaluate per-sample test losses after each epoch

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
    num_epochs : int
        Number of epochs
    dataloader_test : torch.utils.data.DataLoader
        Test set DataLoader
    loss_fn_per_sample : torch.nn loss function
        Per-sample test loss function
    results_path : str or None
        Folder for saving test loss Tensors in

    Returns
    -------
    test_loss_per_sample_full : (num_epochs, num_test_samples, 1, len(seeds)) torch.Tensor
        Loss for each test sample after each epoch of training on full dataset
    test_loss_per_sample_loo : (num_epochs, num_test_samples, len(indices_leave_out), len(seeds)) torch.Tensor
        Loss for each test sample after each epoch of training on leave-one-out dataset
    params_dist_full : (num_epochs, 1, num_seeds) torch.Tensor
        Euclidean distance from initial model parameters after each epoch of training on full dataset
    params_dist_loo : (num_epochs, len(indices_leave_out), num_seeds) torch.Tensor
        Euclidean distance from initial model parameters after each epoch of training on leave-one-out dataset

    """
    num_samples_test = len(dataloader_test.dataset)
    if indices_leave_out is None:
        # Leave out all training samples in turn
        indices_leave_out = range(len(dataset_train))
    num_leave_out = len(indices_leave_out)
    num_seeds = len(seeds)

    # Initialize
    test_loss_per_sample_full = torch.empty((num_epochs, num_samples_test, 1, num_seeds), device=device)
    test_loss_per_sample_loo = torch.empty((num_epochs, num_samples_test, num_leave_out, num_seeds), device=device)
    params_dist_full = torch.empty((num_epochs, 1, num_seeds), device=device)
    params_dist_loo = torch.empty((num_epochs, num_leave_out, num_seeds), device=device)
    seed_string = str(seeds[0]) + "_" + str(seeds[-1])

    # Iterate over random seeds
    for s, seed in enumerate(seeds):
        print(f"Using random seed {seed}")
        # Train on full training set
        print("Training on full training set:")
        test_loss_per_sample_full[:, :, 0, s], model, params_dist_full[:, 0, s] =\
            train_one_run(dataset_train, seed, batch_size, None, model_class, model_path,
                          loss_fn, optimizer_class, learning_rate, num_epochs, dataloader_test, loss_fn_per_sample)

        if results_path is not None:
            torch.save(test_loss_per_sample_full, os.path.join(results_path, f"test_loss_per_sample_full_{seed_string}.pt"))
            torch.save(params_dist_full, os.path.join(results_path, f"params_dist_full_{seed_string}.pt"))

        # Iterate over left-out samples
        print("Training on leave-one-out datasets:")
        for l, index_leave_out in tqdm(enumerate(indices_leave_out)):
            # Train on leave-one-out dataset
            test_loss_per_sample_loo[:, :, l, s], model, params_dist_loo[:, l, s] =\
                train_one_run(dataset_train, seed, batch_size, index_leave_out, model_class, model_path,
                              loss_fn, optimizer_class, learning_rate, num_epochs, dataloader_test, loss_fn_per_sample)

            if results_path is not None:
                torch.save(test_loss_per_sample_loo, os.path.join(results_path, f"test_loss_per_sample_loo_{seed_string}.pt"))
                torch.save(params_dist_loo, os.path.join(results_path, f"params_dist_loo_{seed_string}.pt"))

    return test_loss_per_sample_full, test_loss_per_sample_loo, params_dist_full, params_dist_loo
