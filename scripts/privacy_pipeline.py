import tomli
import shutil
import os
import argparse
from typing import Union, Dict
from train import train
from sample import sample
from eval_catboost import train_catboost
from eval_mlp import train_mlp
from eval_simple import train_simple
from evaluate_privacy import load_data, evaluate_generation, evaluate_privacy_main
import pandas as pd
import matplotlib.pyplot as plt
import zero
import lib
import numpy as np
import torch
import time
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.stats import wasserstein_distance

def load_config(path):
    with open(path, 'rb') as f:
        return tomli.load(f)
    
def save_file(parent_dir, config_path):
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass
    

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--start_privacy_step', action='store_true',  default=-1)
    parser.add_argument('--sample', action='store_true',  default=False)
    parser.add_argument('--eval', action='store_true',  default=False)
    parser.add_argument('--change_val', action='store_true',  default=False)

    args = parser.parse_args()
    raw_config = lib.load_config(args.config)
    if 'device' in raw_config:
        device = torch.device('cuda:0')  # Paul
        # device = torch.device(raw_config['device'])  # Use specified device
    else:
        device = torch.device('cuda:0')  # Original 'cuda:1'
    
    a,b,c = 0,0,0
    N = raw_config['num_numerical_features']
    timer = zero.Timer()
    timer.run()
    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), args.config)

    if args.train:
        train_start = time.time()
        # How to incorporate privacy term after certain training step?
        # Currently start_privacy_step should be specified in config file with step number
        # start_privacy_step = -1 per default
        train(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            change_val=args.change_val,
        )
        train_end = time.time()
        a = train_end - train_start
        print(f"Training time: {a} seconds")
    if args.sample:
        sample_start = time.time()
        sample(
            num_samples=raw_config['sample']['num_samples'],
            batch_size=raw_config['sample']['batch_size'],
            disbalance=raw_config['sample'].get('disbalance', None),
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_path=os.path.join(raw_config['parent_dir'], 'model.pt'),
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            seed=raw_config['sample'].get('seed', 0),
            change_val=args.change_val
        )
        sample_end = time.time()
        b = sample_end - sample_start
        print(f"Sampling time: {b} seconds")
        
        
    save_file(os.path.join(raw_config['parent_dir'], 'info.json'), os.path.join(raw_config['real_data_path'], 'info.json'))
    if args.eval:
        eval_start = time.time()
        if raw_config['eval']['type']['eval_model'] == 'catboost':
            train_catboost(
                parent_dir=raw_config['parent_dir'],
                real_data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val
            )
        elif raw_config['eval']['type']['eval_model'] == 'mlp':
            train_mlp(
                parent_dir=raw_config['parent_dir'],
                real_data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val,
                device=device
            )
        elif raw_config['eval']['type']['eval_model'] == 'simple':
            train_simple(
                parent_dir=raw_config['parent_dir'],
                real_data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val
            )
        eval_end = time.time()
        c = eval_end - eval_start
        print(f"Evaluation time: {c} seconds")

    print(f'Elapsed time: {str(timer)}')
    
    evaluate_privacy_main(args)
    

def main_debug():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--start_privacy_step', action='store_true',  default=-1)
    parser.add_argument('--sample', action='store_true',  default=False)
    parser.add_argument('--eval', action='store_true',  default=False)
    parser.add_argument('--change_val', action='store_true',  default=False)

    args = parser.parse_args()
    raw_config = lib.load_config(args.config)
    dataset_info = lib.load_json(os.path.join(raw_config['parent_dir'], 'DatasetInfo.json'))
    if 'device' in raw_config:
        device = torch.device('cuda:0')  # Paul
        # device = torch.device(raw_config['device'])  # Use specified device
    else:
        device = torch.device('cuda:0')  # Original 'cuda:1'
    
    a,b,c = 0,0,0
    N = raw_config['num_numerical_features']
    timer = zero.Timer()
    timer.run()
    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), args.config)
    
    x_real, x_fake, target_size, task_type = load_data(raw_config['real_data_path'], raw_config['parent_dir'])
    # continue here, change implementation of evaluate generation fo account for concatenated num+
    dataset_info['category_sizes'].append(target_size)
    print(f"Dataset_info: {dataset_info}\nnum_numerical_features: {N}\ntask_type: {task_type}\ncategory_sizes: {dataset_info['category_sizes']}")
    print(f"x_real shape: {x_real.shape}, x_fake shape: {x_fake.shape}")
    stats, scores = evaluate_generation(x_real, x_fake, num_numerical_features=N, category_sizes=dataset_info['category_sizes'], task_type=task_type)
    
    with open(os.path.join(raw_config['parent_dir'], raw_config['evaluation_file']), 'w') as file:
        # Use .write() for a single string
        
        file.write(f"Training time: {a} seconds\n")
        file.write(f"Sampling time: {b} seconds\n")
        file.write(f"Evaluation time: {c} seconds\n")
        file.write(f"Total elapsed time: {str(timer)}\n")
        file.write(f"Numerical Data Stats and Scores:\n")
        file.write(str(stats) + "\n")
        file.write(str(scores) + "\n")

     

if __name__ == '__main__':
    main()