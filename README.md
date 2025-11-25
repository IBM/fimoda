# FiMODA: Final-Model-Only Data Attribution

This repository provides code for reproducing experiments in the NeurIPS 2025 paper "[Final-Model-Only Data Attribution with a Unifying View of Gradient-Based Methods](https://openreview.net/forum?id=rccgEdFTlH)."

## Installation
### uv
We recommend using [uv](https://docs.astral.sh/uv/) as the package manager. If needed, install `uv` via either:

```curl -Ls https://astral.sh/uv/install.sh | sh```

or using [Homebrew](https://brew.sh):

```brew install astral-sh/uv/uv```

or using pip (use this for Windows):

```pip install uv```

### FiMODA

First clone the repository:
```commandline
git clone git@github.com:IBM/fimoda.git
cd fimoda
```

Once inside the `fimoda` directory (where this `README.md` is located), if using Linux or Mac, run:
```commandline
uv venv --python 3.12
source .venv/bin/activate
uv pip install .
```
Or in Windows, run:
```commandline
uv venv --python 3.12
.venv/bin/activate
uv pip install .
```

The code has been tested on Red Hat Enterprise Linux 9 and macOS (`tabular` subfolder only for the latter).

## Organization
The code is organized into three subfolders under the `fimoda` folder, corresponding to three data modalities: `tabular`, `text`, and `image`. The code within each of these subfolders is mostly self-contained.

At least one GPU is strongly recommended for running the code in the `text` and `image` subfolders.

## Citation and appreciation
If you find our work useful, please star the repository and cite our work as follows:
```
@inproceedings{
    wei2025fimoda,
    title={Final-Model-Only Data Attribution with a Unifying View of Gradient-Based Methods},
    author={Dennis Wei and Inkit Padhi and Soumya Ghosh and Amit Dhurandhar and Karthikeyan Natesan Ramamurthy and Maria Chang},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=rccgEdFTlH}
}
```

## License
This code is provided under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).
