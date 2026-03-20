# SynTabRL: TabDDPM with RL Agent for Privacy-Aware Synthetic Tabular Data Generation
This is an adaptation from the official TabDDPM project "TabDDPM: Modelling Tabular Data with Diffusion Models" ([paper](https://arxiv.org/abs/2209.15421)). This project implements a RL agent to improve model output on privacy metrics. 

<!-- ## Results
You can view all the results and build your own tables with this [notebook](notebooks/Reports.ipynb). -->

## Setup the environment
1. Install [conda](https://docs.conda.io/en/latest/miniconda.html) (just to manage the env).
2. Run the following commands
    ```bash
    export REPO_DIR=/path/to/the/code
    cd $REPO_DIR

    conda create -n tddpm python=3.9.7
    conda activate tddpm

    pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt

    # if the following commands do not succeed, update conda
    conda env config vars set PYTHONPATH=${PYTHONPATH}:${REPO_DIR}
    conda env config vars set PROJECT_DIR=${REPO_DIR}

    conda deactivate
    conda activate tddpm
    ```

## Running the experiments

Here we describe the neccesary info for reproducing the experimental results.  
Use `agg_results.ipynb` to print results of the original TabDDPM for all dataset and all methods.

### Datasets

The original paper uploads the datasets used with their custom train/val/test splits (link below). 

To load the datasets, use the following commands: 

``` bash
conda activate tddpm
cd $PROJECT_DIR
wget "https://www.dropbox.com/s/rpckvcs3vx7j605/data.tar?dl=0" -O data.tar
tar -xvf data.tar
```
### The three additional datasets

To load the additional three datasets that are used to conduct SynTabRL evaluation, make sure it is possible to load datasets from UC Irvine ML Repository (e.g. install ucimlrepo Python library via ```pip install ucimlrepo``` ). Follow the steps in preprocess_data.ipynb.




### File structure
`tab-ddpm/` -- implementation of the proposed method  
`tuned_models/` -- tuned hyperparameters of evaluation model (CatBoost or MLP)
`scripts/train.py`-- contains additional logic to handle privacy-preserving loss functions
`dataset_info/` -- metadata on datasets, used in rlagent.py

All main scripts are in `scripts/` folder:

- `scripts/rlagent.py` initiates a RL agent that trains, samples and evaluates modified TabDDPM
- `scripts/pipeline.py` are used to train, sample and eval TabDDPM using a given config  
- `scripts/tune_ddpm.py` -- tune hyperparameters of TabDDPM
- `scripts/eval_[catboost|mlp|simple].py` -- evaluate synthetic data using a tuned evaluation model or simple models
- `scripts/eval_seeds.py` -- eval using multiple sampling and multuple eval seeds
- `scripts/eval_seeds_simple.py` --  eval using multiple sampling and multuple eval seeds (for simple models)
- `scripts/tune_evaluation_model.py` -- tune hyperparameters of eval model (CatBoost or MLP)
- `scripts/resample_privacy.py` -- privacy calculation  

Experiments folder (`privacy_result/`):
- Contains SynTabRL results and synthetic data stored in `privacy_result/[ds_name]/[exp_name]\` folder
- `privacy_result/[ds_name]/[exp_name]/[...]eval.[txt|json]` is the file containing final evaluation results
- `privacy_result/[ds_name]/[exp_name]/RLAgentLoss.csv` records the loss function values encountered during training

Experiments folder (`exp/`):
- All TabDDPM results and synthetic data are stored in `exp/[ds_name]/[exp_name]/` folder
- `exp/[ds_name]/config.toml` is a base config for tuning TabDDPM
- `exp/[ds_name]/eval_[catboost|mlp].json` stores results of evaluation (`scripts/eval_seeds.py`)  

To understand the structure of `config.toml` file, read `CONFIG_DESCRIPTION.md`.

Baselines:
- `smote/`
- `CTGAN/` -- TVAE [official repo](https://github.com/sdv-dev/CTGAN)
- `CTAB-GAN/` --  [official repo](https://github.com/Team-TUD/CTAB-GAN)
- `CTAB-GAN-Plus/` -- [official repo](https://github.com/Team-TUD/CTAB-GAN-Plus)

### Examples

<ins>Run SynTabRL tuning</ins>

Adoptation of original TabDDPM, Template and example (`--eval_seeds`is optional):
```bash
python scripts/tune_SynTabRL.py [ds_name] [train_size] synthetic [catboost|mlp] [exp_name] --eval_seeds
python scripts/tune_SynTabRL.py churn2 6500 synthetic catboost ddpm_tune --eval_seeds
```


<ins>Run TabDDPM tuning.</ins>   

Template and example (`--eval_seeds` is optional): 
```bash
python scripts/tune_ddpm.py [ds_name] [train_size] synthetic [catboost|mlp] [exp_name] --eval_seeds
python scripts/tune_ddpm.py churn2 6500 synthetic catboost ddpm_tune --eval_seeds
```


<ins>Run SynTabRL pipeline.</ins>   

Template and example, --config flag must be provided; If --train is set, --APPROACH must be provided as well: 
```bash
python scripts/rlagent.py --config [path_to_your_config] --APPROACH --train --sample --eval
python scripts/rlagent.py --config privacy_result/churn2/syntabrl_privacy_best/config.toml --adaptive_approach --train --sample --eval
```

When using one of the single privacy loss term, also specify the loss term you want to use [dcr|nndr|gower]:

```bash
python scripts/rlagent.py --config privacy_result/churn2/syntabrl_privacy_best/config.toml --adaptive_single_metric dcr --train --sample --eval
```

The pipeline includes optional privacy-enhancing sampling in a post-processing setting. Follow the following template and example:

```bash
python scripts/rlagent.py --config [path_to_your_config] --sample --filter [percentile | threshold] --filter_value FLOAT (--max_value FLOAT)

# discard top 10% risky samples
python scripts/rlagent.py --config privacy_result/churn2/syntabrl_privacy_best/config.toml --sample --filter percentile --filter_value 0.1

# discard all samples with DCR value below the provided threshold, optionally discard samples above a maximum value.
python scripts/rlagent.py --config privacy_result/churn2/syntabrl_privacy_best/config.toml --sample --filter threshold --filter_value 0.17 (--max_value 1.5)
```

<ins>Run TabDDPM pipeline.</ins>   

Template and example  (`--train`, `--sample`, `--eval` are optional): 
```bash
python scripts/pipeline.py --config [path_to_your_config] --train --sample --eval
python scripts/pipeline.py --config exp/churn2/ddpm_cb_best/config.toml --train --sample
```
It takes approximately 7min to run the script above (NVIDIA GeForce RTX 2080 Ti).  

<ins>Run TabDDPM evaluation over seeds</ins>   
Before running evaluation, you have to train the model with the given hyperparameters (the example above).  

Template and example: 
```bash
python scripts/eval_seeds.py --config [path_to_your_config] [n_eval_seeds] [ddpm|smote|ctabgan|ctabgan-plus|tvae] synthetic [catboost|mlp] [n_sample_seeds]
python scripts/eval_seeds.py --config exp/churn2/ddpm_cb_best/config.toml 10 ddpm synthetic catboost 5
```