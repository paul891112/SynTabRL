import tomli
import shutil
import os
import argparse
from typing import Union, Dict
from train import train, DatasetInfo
from sample import sample
from eval_catboost import train_catboost
from eval_mlp import train_mlp
from eval_simple import train_simple
from evaluate_privacy import compute_dcr, compute_nndr, compute_gowers_distance
import pandas as pd
import matplotlib.pyplot as plt
import zero
import lib
import numpy as np
import torch
import time
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.stats import wasserstein_distance
from enum import Enum


PRIVACY_CONFIG_DICT = {
    "DCR": 0.4,
    "NNDR": 0.4,
    "Gower": 0.4 
                          }


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

class State(Enum):
    """Represents the privacy state of the RL agent based on evaluation metrics."""
    LOW = 0,
    HIGH = 1
    
class Action(Enum):
    Conventional = "Model in high privacy state, train with conventional loss function",
    Privacy = "Model in low privacy state, train with privacy-preserving loss function"


class RLAgent:
    def __init__(self, name="", rounds=20, steps_per_round=1000, evaluation_samples=2000, args=None, raw_config=None, device=None):
        """
        RL Agent for training a diffusion model with privacy considerations. One agent for one dataset.
        Args:
            name (str): Name of the agent
            rounds (int, optional): Number of training the agent should do. Each training is a \n
            multi-step training process, determined by argument steps_per_round. Defaults to 20.
            steps_per_round (int, optional): Number of training steps per round. Defaults to 1000.
        """
        self.name = name
        self.rounds = rounds
        self.steps_per_round = steps_per_round
        self.evaluation_samples = evaluation_samples
        self.privacy_threshold = PRIVACY_CONFIG_DICT  # Example threshold for privacy metric

        self.args = args
        self.raw_config = raw_config
        self.device = device
            
        self.real_data, self.target_size, self.dataset_info, self.mm, self.ohe = self.load_real_data(self.raw_config['real_data_path'])
        self.real_data = np.asarray(self.real_data)
        self.dataset_info.update(lib.load_json(os.path.join(self.raw_config['parent_dir'], 'DatasetInfo.json')))
        self.dataset_info['category_sizes'].append(self.target_size)
        print(f"RL agent {self.name} initialized to train for dataset {self.dataset_info['name']} with {self.rounds} rounds, each with {self.steps_per_round} steps.")
        self.category_sizes = self.dataset_info.get('category_sizes')
        self.task_type = self.dataset_info.get('task_type')
    
        
    def load_real_data(self, real_path):
    
        """
        Adopted from tab-ddpm/scripts/resample_privacy.py, which inturns is adapted from https://github.com/Team-TUD/CTAB-GAN/tree/main/model/eval
        
        """
        
        info = lib.load_json(real_path + "/info.json")
        dataset_info = {
            "name": info["name"],
            "task_type": info["task_type"],
            
        }
        X_num_real, X_cat_real, y_real = lib.read_pure_data(real_path, 'train')
        target_size = 0
        
        if dataset_info["task_type"] == 'regression':
            X_num_real = np.concatenate([X_num_real, y_real[:, np.newaxis]], axis=1)
            target_size = 1
        else:  # classification, binclass or multiclass
            target_size = 2 if dataset_info["task_type"] == 'binclass' else len(np.unique(y_real))
            if X_cat_real is None:
                X_cat_real = y_real[:, np.newaxis].astype(int).astype(str)
                
            else:
                X_cat_real = np.concatenate([X_cat_real, y_real[:, np.newaxis].astype(int).astype(str)], axis=1)
                
        if len(y_real) > 50000:
            ixs = np.random.choice(len(y_real), 50000, replace=False)
            X_num_real = X_num_real[ixs]
            X_cat_real = X_cat_real[ixs] if X_cat_real is not None else None

        mm = MinMaxScaler().fit(X_num_real)
        ohe = None
        X_real = mm.transform(X_num_real)
        if X_cat_real is not None:
            ohe = OneHotEncoder().fit(X_cat_real)
            X_cat_real = ohe.transform(X_cat_real) / np.sqrt(2)
            X_real = np.concatenate([X_real, X_cat_real.todense()], axis=1)
        
        return X_real, target_size, dataset_info, mm, ohe
    
    
    def load_fake_data(self, X_num_fake, X_cat_fake, y_gen):
        """
        Adopted from tab-ddpm/scripts/resample_privacy.py, which inturns is adapted from
        
        """
        if self.task_type == 'regression':
            X_num_fake = np.concatenate([X_num_fake, y_gen[:, np.newaxis]], axis=1)
        else:  # classification, binclass or multiclass
            if X_cat_fake is None:
                X_cat_fake = y_gen[:, np.newaxis].astype(int).astype(str)
                
            else:
                X_cat_fake = np.concatenate([X_cat_fake, y_gen[:, np.newaxis].astype(int).astype(str)], axis=1)
        
        if len(y_gen) > self.evaluation_samples:
            ixs = np.random.choice(len(y_gen), self.evaluation_samples, replace=False)
            X_num_fake = X_num_fake[ixs]
            X_cat_fake = X_cat_fake[ixs] if X_cat_fake is not None else None

        
        X_fake = self.mm.transform(X_num_fake)
        if (X_cat_fake is not None) and (self.ohe is not None):
            X_cat_fake = self.ohe.transform(X_cat_fake) / np.sqrt(2)
            X_fake = np.concatenate([X_fake, X_cat_fake.todense()], axis=1)
            
        return X_fake
    
    def pretrain(self, steps=None):
        if steps:
            train(
                steps=steps,
                start_privacy_step=-1,
                lr = self.raw_config['train']['main']['lr'],
                weight_decay = self.raw_config['train']['main']['weight_decay'],
                batch_size=self.raw_config['train']['main']['batch_size'],
                **self.raw_config['diffusion_params'],
                parent_dir=self.raw_config['parent_dir'],
                real_data_path=self.raw_config['real_data_path'],
                model_type=self.raw_config['model_type'],
                model_params=self.raw_config['model_params'],
                T_dict=self.raw_config['train']['T'],
                num_numerical_features=self.raw_config['num_numerical_features'],
                device=self.device,
                change_val=self.args.change_val,
                continue_training=False
            )
        else:
            train(
                    steps=self.steps_per_round,
                    start_privacy_step=-1,
                    lr = self.raw_config['train']['main']['lr'],
                    weight_decay = self.raw_config['train']['main']['weight_decay'],
                    batch_size=self.raw_config['train']['main']['batch_size'],
                    **self.raw_config['diffusion_params'],
                    parent_dir=self.raw_config['parent_dir'],
                    real_data_path=self.raw_config['real_data_path'],
                    model_type=self.raw_config['model_type'],
                    model_params=self.raw_config['model_params'],
                    T_dict=self.raw_config['train']['T'],
                    num_numerical_features=self.raw_config['num_numerical_features'],
                    device=self.device,
                    change_val=self.args.change_val,
                    continue_training=False
                )
    
    
    def run_algorithm(self):
        
        self.pretrain()
        state = State.HIGH
        counter = 0
        for _ in range(self.rounds):
            counter += 1
            print(f"=== RL Agent Round {counter} ===")
            print(f"State: {state.name}")
            self.train_model(state)
            X_num, X_cat, y_gen = self.generate_samples()
            state = self.evaluate_state(X_num, X_cat, y_gen)
        
        
    def generate_samples(self, num_samples=2000):
        """Generates 2000 samples using the current model and configuration."""
        return sample(
            num_samples=num_samples,
            batch_size=self.raw_config['sample']['batch_size'],
            disbalance=self.raw_config['sample'].get('disbalance', None),
            **self.raw_config['diffusion_params'],
            parent_dir=self.raw_config['parent_dir'],
            real_data_path=self.raw_config['real_data_path'],
            model_path=os.path.join(self.raw_config['parent_dir'], 'model.pt'),
            model_type=self.raw_config['model_type'],
            model_params=self.raw_config['model_params'],
            T_dict=self.raw_config['train']['T'],
            num_numerical_features=self.raw_config['num_numerical_features'],
            device=self.device,
            seed=self.raw_config['sample'].get('seed', 0),
            change_val=self.args.change_val
        )

    def evaluate_state(self, X_num, X_cat, y_gen):
        """
        Evaluates the privacy state of the model based on generated samples.
        
        Args:
            X_num (np.ndarray): Numerical features of generated samples.
            X_cat (np.ndarray): Categorical features of generated samples.
            y_gen (np.ndarray): Generated target variable.
            K (list): Category sizes for categorical features.
        Returns:
            State: The evaluated privacy state (HIGH or LOW).
        
        """
        
        # Concatenate synthetic data into a single matrix like in evaluate_privacy.load_data()
       
        X_fake = self.load_fake_data(X_num, X_cat, y_gen)
        X_fake = np.asarray(X_fake)
        
        eval_metrics = {}
        eval_metrics["DCR"] = compute_dcr(original_data=self.real_data, synthetic_data=X_fake, num_numerical_features=self.raw_config['num_numerical_features'], category_sizes=self.category_sizes, task_type=self.task_type, distance_metric='euclidean')
        eval_metrics["NNDR"] = compute_nndr(original_data=self.real_data, synthetic_data=X_fake, num_numerical_features=self.raw_config['num_numerical_features'], category_sizes=self.category_sizes, task_type=self.task_type, distance_metric='euclidean')
        eval_metrics["Gower"] = compute_gowers_distance(original=self.real_data, synthetic=X_fake, n_num_features=self.raw_config['num_numerical_features'], category_sizes=self.category_sizes, task_type=self.task_type)
        print(f"Privacy evaluation: {eval_metrics}")
        for key, value in PRIVACY_CONFIG_DICT.items():
            print(f"{key}: {eval_metrics[key]:.4f} (Benchmark: {value})")
            if eval_metrics[key] < value:  # distance smaller than benchmark, privacy is low
                return State.LOW
        return State.HIGH
                


    def train_model(self, state) -> DatasetInfo:
        
        if state == State.HIGH:
            action = Action.Conventional
            print(f"{action.value}")
            # Take action Conventional
            return train(
                steps=self.steps_per_round,
                start_privacy_step=-1,
                lr = self.raw_config['train']['main']['lr'],
                weight_decay = self.raw_config['train']['main']['weight_decay'],
                batch_size=self.raw_config['train']['main']['batch_size'],
                **self.raw_config['diffusion_params'],
                parent_dir=self.raw_config['parent_dir'],
                real_data_path=self.raw_config['real_data_path'],
                model_type=self.raw_config['model_type'],
                model_params=self.raw_config['model_params'],
                T_dict=self.raw_config['train']['T'],
                num_numerical_features=self.raw_config['num_numerical_features'],
                device=self.device,
                change_val=self.args.change_val,
                continue_training=True
            )
        elif state == State.LOW:
            action = Action.Privacy
            print(f"{action.value}")
            return train(
                steps=self.steps_per_round,
                start_privacy_step=0,
                lr = self.raw_config['train']['main']['lr'],
                weight_decay = self.raw_config['train']['main']['weight_decay'],
                batch_size=self.raw_config['train']['main']['batch_size'],
                **self.raw_config['diffusion_params'],
                parent_dir=self.raw_config['parent_dir'],
                real_data_path=self.raw_config['real_data_path'],
                model_type=self.raw_config['model_type'],
                model_params=self.raw_config['model_params'],
                T_dict=self.raw_config['train']['T'],
                num_numerical_features=self.raw_config['num_numerical_features'],
                device=self.device,
                change_val=self.args.change_val,
                continue_training=True
            )    


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--start_privacy_step', action='store_true',  default=-1)
    parser.add_argument('--sample', action='store_true',  default=False)
    parser.add_argument('--eval', action='store_true',  default=False)
    parser.add_argument('--change_val', action='store_true',  default=False)
    print("Parsing arguments ...")

    args = parser.parse_args()
    raw_config = lib.load_config(args.config)
    if 'device' in raw_config:
        device = torch.device('cuda:0')  # Paul
        # device = torch.device(raw_config['device'])  # Use specified device
    else:
        device = torch.device('cuda:0')  # Original 'cuda:1'
    print("Starting agent ...")
    agent = RLAgent(name="RLAgent1", rounds=10, steps_per_round=1000, evaluation_samples=2000, args=args, raw_config=raw_config, device=device)
    agent.run_algorithm()
    
    
if __name__ == '__main__':
    main()