
import argparse
import os
from copy import deepcopy
import torch
from torch.distributions.normal import Normal
import os
import numpy as np
import zero
from tab_ddpm import GaussianMultinomialDiffusion
from utils_train import get_model, make_dataset, update_ema
import lib
import pandas as pd

DATASETINFO_DIR = "dataset_info"


class DatasetInfo:
    """
    Custom Class for information passing.
    Holds dataset information such as category sizes and number of numerical features.
    """
    def __init__(self, k, trainer):
        self._K = k
        self._num_numerical_features = trainer.diffusion._num_numerical_features
    
    def get_category_sizes(self):
        return self._K
    def get_num_numerical_features(self):
        return self._num_numerical_features


def generate_dataset_info(real_data_path, change_val=False):
    dataset_name = os.path.basename(os.path.normpath(real_data_path))
    config = lib.load_config(os.path.join("exp", dataset_name, "config.toml"))

    model_params = config['model_params']
    real_data_path = config['real_data_path']
    parent_dir = config['parent_dir']
    seed = config['seed']
    T_dict = config['train']['T']
    model_type = config['model_type']
    
    real_data_path = os.path.normpath(real_data_path)
    parent_dir = os.path.normpath(parent_dir)
    # info_json_path = os.path.join(parent_dir, 'DatasetInfo.json')
    datasetinfo_path = os.path.join(DATASETINFO_DIR, os.path.basename(real_data_path), 'DatasetInfo.json')

    if os.path.exists(datasetinfo_path):
        print(f"DatasetInfo already exists for dataset {real_data_path}. Loading from {datasetinfo_path}.")
        return lib.load_json(datasetinfo_path)
    
    T = lib.Transformations(**T_dict)
    zero.improve_reproducibility(seed)
    
    dataset = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])
        
    # Update the metadata dictionary with new information
    new_info = {
        'task_type': dataset.task_type.value,  # 'binclass' or 'multiclass' or 'regression'
        'n_classes': dataset.n_classes, 
        'n_num_features': dataset.n_num_features,
        'n_cat_features': dataset.n_cat_features,
        'training_completed': True,
        'category_sizes': K.tolist(),
        # Add any other relevant metrics or parameters here
    }
    
    # D. Save the updated dictionary back to the file (overwriting)
    # lib.dump_json(new_info, info_json_path)
    if not os.path.exists(os.path.join(DATASETINFO_DIR, dataset_name)):
        os.makedirs(os.path.join(DATASETINFO_DIR, dataset_name))
    lib.dump_json(new_info, datasetinfo_path)
    return new_info
    


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--change_val', action='store_true', default=False)
    args = parser.parse_args()
    
    generate_dataset_info(args.config, change_val=args.change_val)
    print(f"DatasetInfo generated and saved for config: {args.config}")

    
    
    
if __name__ == "__main__":
    main()