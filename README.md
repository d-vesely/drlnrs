# DRLNRS

This repository contains the code written as part of my Master's thesis, titled **A Comparative Performance Analysis of Deep Reinforcement Learning News Recommender Systems** (TODO link). It constitutes a framework for training *Reinforcement Learning* agents on a *News Recommendation* problem, using the [MIND Dataset](https://msnews.github.io/).

The code in this repository can be used to *reproduce* the results presented in the thesis. It can also be used as a basis for further work, by *extending* the capabilities of the framework, or by *adapting* it to another dataset.

Refer to the thesis, particularly chapter 7, for detailed explanations of this framework.

## Repository Structure

`dataset_MIND/`: this folder contains just a placeholder file. [Download](https://msnews.github.io/#getting-start) the MIND dataset and extract it into this folder, such that the structure is as follows:
```
dataset_MIND/
    MINDlarge_dev/
    MINDlarge_test/
    MINDlarge_train/
    MINDsmall_dev/
    MINDsmall_train/
```

`models/`: this folder contains a subdirectory for each model trained and presented in the thesis. Each model-folder contains the configuration files for this particular model (see [Configurations](#configurations)) in the subdirectory `configs/`, and the results for the two used random seeds (7 and 42) in the respective subdirectory `predictions_[seed]/`. The evaluation results for each seed are stored in a file `eval_results.txt`. This file contains a tab-separated table with two columns, `checkpoints` and `mean_return`. The former lists the evaluated model checkpoints, the latter the obtained average returns when evaluating each checkpoint on the dev-set. The model checkpoints (PyTorch `.pth`-files) are not in the repository, but will be stored in a subdirectory `checkpoints_[seed]` during and after training.

`visualizations/`: this folder contains all (code-made) visualizations used in the thesis. The subdirectory `behavior/` contains all diagrams presented in Figure 6.2, and `news/` contains those presented in Figure 6.3. The subdirectory `results/` contains additional folders `hyperparameters/`, `user_encoders/`, `news_encoders/`, `drl_algorithms/` and `ddpg_td3/`, which hold the plots presented in Figures 7.4, 7.5, 7.6, 7.7 and 7.8 respectively.

`src/`: this folder contains the entire Python source code. See [Framework](#framework) for details.

Aside from auxiliary files ([License](./LICENSE.md), [Citation](./CITATION.cff), `gitignore`, [environment.yml](./environment.yml)), the top level contains `Jupyter Notebooks`, which can be used to reproduce the results of the thesis step by step. These notebooks serve as entry points and use the code in `src/` and are further discussed below ([Notebooks](#notebooks)).

## Installation

With `conda`, create a virtual environment using the environment file [environment.yml](./environment.yml). Then, activate the new environment called *drlnrs*. This can be done with the commands:

```
conda env create -f environment.yml
conda activate drlnrs
```

## Notebooks

### Configurations

## Citation

If you use this code in your work, please cite it using GitHub's citation feature, or use the following BibTex citation.

```
@software{drlnrs,
    author = {Veselý, Dominik},
    license = {MIT},
    title = {{DRLNRS}},
    url = {https://github.com/d-vesely/drlnrs}
}
```

In addition, please cite the corresponding Master's thesis, using the following BibTex citation.

```
TODO
```