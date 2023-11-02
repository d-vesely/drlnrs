# DRLNRS

This repository contains the code written as part of my Master's thesis, titled **"A Comparative Performance Analysis of Deep Reinforcement Learning News Recommender Systems"** (TODO link). It constitutes a framework for training *Reinforcement Learning* agents on a *News Recommendation* problem, using the [MIND Dataset](https://msnews.github.io/).

The code in this repository can be used to *reproduce* the results presented in the thesis. It can also be used as a basis for further work, by *extending* the capabilities of the framework, or by *adapting* it to another dataset.

Refer to the thesis, particularly chapter 7, for detailed explanations of this framework.

## Repository Structure

[`dataset_MIND/`](./dataset_MIND/): this folder contains just a placeholder file. [Download](https://msnews.github.io/#getting-start) the MIND dataset and extract it into this folder, such that the structure is as follows:
```
dataset_MIND/
    MINDlarge_all/ # Create this folder
    MINDlarge_dev/
    MINDlarge_test/
    MINDlarge_train/
    MINDlarge_trainfull/ # Create this folder
    MINDsmall_dev/
    MINDsmall_train/
```

[`models/`](./models/): this folder contains a subdirectory for each model trained and presented in the thesis. Each model-folder contains the configuration files for this particular model (see [Reproducing Results](#reproducing-results)) in the subdirectory `configs/`, and the results for the two used random seeds (7 and 42) in the respective subdirectory `predictions_[seed]/`. The evaluation results for each seed are stored in a file `eval_results.txt`. This file contains a tab-separated table with two columns, `checkpoints` and `mean_return`. The former lists the evaluated model checkpoints, the latter the obtained average returns when evaluating each checkpoint on the dev-set. The model checkpoints (PyTorch `.pth`-files) are not in the repository, but will be stored in a subdirectory `checkpoints_[seed]` during and after training.

[`visualizations/`](./visualizations/): this folder contains all (code-made) visualizations used in the thesis. The subdirectory `behavior/` contains all diagrams presented in Figure 6.2, and `news/` contains those presented in Figure 6.3. The subdirectory `results/` contains additional folders `hyperparameters/`, `user_encoders/`, `news_encoders/`, `drl_algorithms/` and `ddpg_td3/`, which hold the plots presented in Figures 7.4, 7.5, 7.6, 7.7 and 7.8 respectively.

[`src/`](./src/): this folder contains the entire Python source code. See [Framework](#framework) for details.

Aside from auxiliary files ([License](./LICENSE.md), [Citation](./CITATION.cff), `gitignore`, [environment.yml](./environment.yml)), the top level contains `Jupyter Notebooks`, which can be used to reproduce the results of the thesis step by step. These notebooks serve as entry points and use the code in `src/` and are further discussed below ([Notebooks](#notebooks)).

## Installation

With `conda`, create a virtual environment using the environment file [environment.yml](./environment.yml). Then, activate the new environment called *drlnrs*. This can be done with the commands:

```
conda env create -f environment.yml
conda activate drlnrs
```

## Notebooks

In this Section, we briefly want to present each notebook, since they serve as the main entry points to the codebase and should be used, when reproducing the results from the thesis (for details, see [below](#reproducing-results))

- [`categories.ipynb`](./categories.ipynb): This notebook facilitates the exploration of the news categories and subcategories present in the dataset.
- [`embedding.ipynb`](./embedding.ipynb): This notebook facilitates the creation of the embeddings used in the thesis, using the [Sentence Transformer Library](https://www.sbert.net/) and the [OpenAI API](https://platform.openai.com/docs/guides/embeddings). The notebook also contains the code to produce feature vectors.
- [`evaluation.ipynb`](./evaluation.ipynb): This notebook contains the code to evaluate a trained agent. It is designed to work in tandem with the training notebook, it is thus best to train agents with the corresponding notebook. The notebook also contains code to produce a random baseline.
- [`preprocessing.ipynb`](./preprocessing.ipynb): This notebook contains all of the required code to run all preprocessing steps, for both the behaviors (impression) data and the news data. It also contains some exploratory analysis.
- [`replay_memory_building.ipynb`](./replay_memory_building.ipynb): This notebook contains the code to build the replay memory used for training.
- [`training.ipynb`](./training.ipynb): This notebook can be used to train DRLRS agents. The user can easily edit news/user encoder settings, hyperparameter settings and model settings.
- [`visualization_behaviors.ipynb`](./visualization_behaviors.ipynb): This notebook can be used to recreate the figures in the thesis pertaining to the behaviors (impression) data.
- [`visualization_news.ipynb`](./visualization_news.ipynb): This notebook can be used to recreate the figures in the thesis pertaining to the news data.
- [`visualization_results.ipynb`](./visualization_results.ipynb): This notebook can be used to recreate the figures in the thesis pertaining to the results.

## Reproducing Results

We briefly want to outline the required steps to reproduce the results of the thesis. Since the code is not packaged in a CLI or a similar abstraction, reproducing the results requires some expertise and some understanding of the codebase. However, it should be sufficiently easy.

1. [Download](https://msnews.github.io/#getting-start) the dataset and place it in the [`dataset_MIND/`](./dataset_MIND/) directory as outlined above. Create the two directories `MINDlarge_all/` and `MINDlarge_trainfull/`. The former will hold the preprocessed news data (the news data is also split into train/dev/test sets, which is not necessary in our case) and all embeddings. The latter will be used for the concatenated train- and dev- data.
2. Use the notebook [`preprocessing.ipynb`](./preprocessing.ipynb) to run preprocessing steps. Note that running the notebook from start to end might not be desired. Instead, decide which cells are relevant for reproducing results. Notably, exploratory preprocessing and exploratory analysis might not be necessary.
3. Use the notebook [`embedding.ipynb`](./embedding.ipynb) to produce embeddings for the preprocessed news data.
4. Use the notebook [`replay_memory_building.ipynb`](./replay_memory_building.ipynb) to produce the replay memory required for training. As documented in the notebook, it might be necessary to adapt certain parameters, depending on your hardware (the replay memory is extremely large and has to be produced in splits, which are then concatenated back together; how many splits will be necessary will depend on your RAM). The key part to ensure reproducibility, is to leave the seed (42) and the fraction (0.2) for the extraction of negative experiences untouched (the amount of ignored news is extremely large, therefore, we sample 20% of it and use it for training, discarding the remaining 80%).
5. Use the notebook [`training.ipynb`](./training.ipynb) to train DRLRS agents. To reproduce a specific result, use the published config files to set the news/user encoder settings, hyperparameter and model settings accordingly. Then, train the agents using the specific seeds (7 and 42, which is prepared in the notebook). For example, to reproduce the results for the model `C51-n`, we use the configuration files from [`models/C51-n/configs/`](./models/C51-n/configs/) to adapt the notebook accordingly, and then run training. It is best to change the model name, so that a new directory is created, e.g. `C51-n-repro`.
6. Use the notebook [`evaluation.ipynb`](./evaluation.ipynb) to evaluate trained agents. For example, to evaluate the trained agent `C51-n-repro`, we have to change the model name in the notebook accordingly. The notebook will do the rest (specifically, using the config files in the corresponding directory to load the checkpoints into the correct model).

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