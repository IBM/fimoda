# Code for tabular data

### Files
- **FICO_data.py:** Load and pre-process FICO Challenge dataset
- **further_train_classification.py**: Perform further training runs for tabular classification datasets, FICO Challenge and Folktables
- **further_train_regression.py**: Perform further training runs for tabular regression datasets, Concrete Strength and Energy Efficiency
- **tabular_classification.ipynb**: Notebook implementing the remaining computations for the tabular classification datasets
- **tabular_regression.ipynb**: Notebook implementing the remaining computations for the tabular regression datasets
- **training.py**: Training functions and classes

### Data and models
We do not provide data or trained models in this repository (to keep it lightweight and avoid complications). Instructions for downloading datasets and code for training "final" models can be found in tabular_classification.ipynb and tabular_regression.ipynb. 

### Left-out training instances
We do include the indices of left-out training instances for the classification datasets:
- **results/fico/ind_loo.pt**: Indices of left-out training instances for the FICO Challenge dataset
- **results/folktables/ind_loo.pt**: Indices of left-out training instances for the Folktables dataset

The left-out indices for the regression datasets are defined as follows (also in tabular_regression.ipynb):
```
if dataset == "concrete":
    indices_leave_out = range(0, 900, 9)
elif dataset == "energy":
    indices_leave_out = list(range(0, 700, 7))
    indices_leave_out[-1] = 690    # keep indices in range
```
