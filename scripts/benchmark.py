import tomli
import shutil
import os
import argparse
from scripts.utils_train import make_dataset
from train import train
from sample import sample
from eval_catboost import train_catboost
from eval_mlp import train_mlp
from eval_simple import train_simple
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import zero
import lib
import torch
from evaluate_privacy import evaluate_generation, load_data
import time
from datasetinfo import generate_dataset_info


def get_model_pipeline(args):
    lookup = {
    'ctgan': ['python3.9', 'CTGAN/CTGAN/ctgan/__main__.py', '--config', f'{args.config}', '--train', '--sample', '--eval'],
    'ctabgan': ['python3.9', 'CTAB-GAN-Plus/pipeline_ctabganp.py', '--config', f'{args.config}', '--train', '--sample', '--eval'],
    'tvae': ['python3.9', 'CTGAN/pipeline_tvae.py', '--config', f'{args.config}', '--train', '--sample', '--eval'],
    'smote': ['python3.9', 'smote/pipeline_smote.py', '--config', f'{args.config}', '--sample', '--eval']
    }
    
    return lookup[args.model]


def main():
    
    parser = argparse.ArgumentParser()    
    parser.add_argument('--model', choices=['ctgan', 'ctabgan', 'tvae', 'smote'], required=True)
    parser.add_argument('--config', metavar='FILE')
    
    args = parser.parse_args()
    print(f"Using config file: {args.config}")
    print(f"Using model: {args.model}")
    raw_config = lib.load_config(args.config)
    
    dataset_info = generate_dataset_info(real_data_path=raw_config['real_data_path'])  # generates DatasetInfo and saves to dataset_info folder if not already exists
    pipeline = get_model_pipeline(args)
    st = time.time()

    # Train and sample the model
    subprocess.run(pipeline, check=True)
    
    # Evaluate privacy, adopted from evaluate_privacy.py, evaluate_privacy_main()
    
    x_real, x_fake, target_size, task_type = load_data(raw_config['real_data_path'], raw_config['parent_dir'])
    
    real_data_path = os.path.normpath(raw_config['real_data_path'])
    parent_dir = os.path.normpath(raw_config['parent_dir'])
    dataset_name = real_data_path.split(os.sep)[-1]
    print(f"Dataset name: {dataset_name}")
    
    info = lib.load_json(os.path.join(real_data_path, 'info.json'))
    N = info['n_num_features']
    
    dataset_info["category_sizes"].append(target_size)
    print(f"Dataset_info: {dataset_info}\nnum_numerical_features: {N}\ntask_type: {task_type}\ncategory_sizes: {dataset_info['category_sizes']}")
    print(f"x_real shape: {x_real.shape}, x_fake shape: {x_fake.shape}")
    stats, scores = evaluate_generation(x_real, x_fake, N, dataset_info["category_sizes"], task_type=task_type)
    
    with open(os.path.join(parent_dir, 'SynTabRL_evaluation.txt'), 'w') as file:
        
        file.write(f"Similarity and Privacy evaluation\n")
        for key, value in stats.items():
            file.write(f"{key}: {value}\n")
        file.write(f"\nAbsolute Difference of Basic Statistics:\n")
        for key, value in scores.items():
            file.write(f"{key}: {value}\n")
        file.write("Statistics computed column-wise, then average is taken.\n")    
        
        file.write(f"\nMachine Learning Evaluation Results:\n")
        file.write("See results_catboost.json\n")
        
        elapsed_time = time.time() - st
        
        minutes, seconds = divmod(elapsed_time, 60)
        file.write(f"\nTotal training, sampling and evaluation time: {int(minutes)} min {seconds:.2f} sec\n")


def main_debug():
    
    parser = argparse.ArgumentParser()    
    parser.add_argument('--model', choices=['ctgan', 'ctabgan', 'tvae', 'smote'], required=True)
    parser.add_argument('--config', metavar='FILE')
    
    args = parser.parse_args()
    print(f"Using config file: {args.config}")
    print(f"Using model: {args.model}")
    raw_config = lib.load_config(args.config)
    
    dataset_info = generate_dataset_info(real_data_path=raw_config['real_data_path'])  # generates DatasetInfo and saves to dataset_info folder if not already exists
    pipeline = get_model_pipeline(args)
    st = time.time()

    # Train and sample the model
    # subprocess.run(pipeline, check=True)
    
    # Evaluate privacy, adopted from evaluate_privacy.py, evaluate_privacy_main()
    
    x_real, x_fake, target_size, task_type = load_data(raw_config['real_data_path'], raw_config['parent_dir'])
    
    
    
           
if __name__ == '__main__':
    main()
    # main_debug()
    
    