import tomli
import shutil
import os
import argparse
from typing import Union, Dict
from itertools import cycle
from train import train, DatasetInfo
from sample import sample
from eval_catboost import train_catboost
from eval_mlp import train_mlp
from eval_simple import train_simple
from evaluate_privacy import evaluate_generation, compute_dcr, compute_nndr, compute_gowers_distance
from train import LOSS_HISTORY_COLUMNS
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
import time
import gc


PRIVACY_CONFIG_DICT = {
    "dcr": 0.3,  # the higher, the better privacy, but the lower fidelity
    "nndr": 0.7,  # the higher, the better privacy
    "gower": 0.3  # the higher, the better privacy
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
    HIGH = 0
    LOW_DCR = "dcr"
    LOW_NNDR = "nndr"
    LOW_GOWER = "gower"
    
PRIVACY_TO_STATE = {
    "dcr": State.LOW_DCR,
    "nndr": State.LOW_NNDR,
    "gower": State.LOW_GOWER
    
}
    
class Action(Enum):
    Conventional = "Model in high privacy state, train with conventional loss function"
    Privacy = "Model in low privacy state, train with privacy-preserving loss function"


class RLAgent:
    def __init__(self, name="", steps_per_round=1000, evaluation_samples=2000, args=None, raw_config=None, device=None, rounds=None, pretrain_steps=None):
        """
        RL Agent for training a diffusion model with privacy considerations. One agent for one dataset.
        Args:
            name (str): Name of the agent
            steps_per_round(int, optional): number of steps to train after each interaction with environment. Defaults to 2000.
            evaluation_samples(int, optional): number of samples used to evaluate state
            args: arguments passed from command line call
            raw_config: configuration dictionary from config.toml
            rounds (int, optional): Number of training rounds the agent should do. Each training round is a multi-step training process, determined by argument steps_per_round.
            pretrain_steps (int, optional): number of steps of pretraining before considering privacy. Defaults to steps_per_round (2000).
        """
        self.name = name
        self.steps_per_round = steps_per_round
        self.evaluation_samples = evaluation_samples
        self.__state = State.HIGH
        self.privacy_cycle = cycle(PRIVACY_CONFIG_DICT.keys())
        self.loss_history = pd.DataFrame(columns=LOSS_HISTORY_COLUMNS)

        self.args = args
        self.raw_config = raw_config
        self.device = device
        self.privacy_metric = self.raw_config['privacy_metric'] if ('privacy_metric' in self.raw_config and self.raw_config['privacy_metric'] in PRIVACY_CONFIG_DICT.keys()) else 'nndr'  # 'nndr' or 'gower'
        self.privacy_threshold = self.load_privacy_config()
        self.rounds = rounds if rounds else int(self.raw_config['train']['main']['steps'] / self.steps_per_round)
            
        self.real_data, self.target_size, self.dataset_info, self.mm, self.ohe = self.load_real_data(self.raw_config['real_data_path'])
        self.real_data = np.asarray(self.real_data)
        
        self.pretrain(pretrain_steps)
        
        self.dataset_info.update(lib.load_json(os.path.join(self.raw_config['parent_dir'], 'DatasetInfo.json')))
        self.dataset_info['category_sizes'].append(self.target_size)
        print(f"RL agent {self.name} initialized to train for dataset {self.dataset_info['name']} with {self.rounds} rounds, each with {self.steps_per_round} steps.\nUse privacy metric: {self.privacy_metric}.")
        self.category_sizes = self.dataset_info.get('category_sizes')
        self.task_type = self.dataset_info.get('task_type')
        
        
    
    
    def get_state(self):
        return self.__state
    
        
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
        Adopted from tab-ddpm/scripts/resample_privacy.py, which inturns is adapted from https://github.com/Team-TUD/CTAB-GAN/tree/main/model/eval
        
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
    
    
    def load_privacy_config(self):
        """
        Loads privacy configuration from PRIVACY_CONFIG_DICT based on the selected privacy metric.
        
        Returns:
            float: The privacy threshold for the selected metric.
        """
        if ('privacy' in self.raw_config and 'config' in self.raw_config['privacy']):
            privacy_config = {}
            for key in PRIVACY_CONFIG_DICT.keys():
                if key in self.raw_config['privacy']['config']:
                    privacy_config[key] = self.raw_config['privacy']['config'][key]
                else:
                    privacy_config[key] = PRIVACY_CONFIG_DICT[key]
            return privacy_config           
        
        else:
            return PRIVACY_CONFIG_DICT
        
    
    def pretrain(self, steps=None):
        """
        Before incorporating privacy, train the model using conventional tabddpm training script.
        """
        if steps:
            train_result = train(
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
            self.loss_history = pd.concat([self.loss_history, train_result], axis=0, ignore_index=True)
        else:
            train_result = train(
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
            self.loss_history = pd.concat([self.loss_history, train_result], axis=0, ignore_index=True)
    
    
    def run_algorithm(self):
        """
        RL agent algorithm. Interact with environment and take action for n rounds, then log results.
        Pretraining takes place at Agent object instantiation (__init__ function).
        """
        start = time.time()
        counter = 0
        for _ in range(self.rounds):
            counter += 1
            print(f"=== RL Agent Round {counter} ===")
            print(f"State: {self.get_state().name}")
            train_result = self.train_model()
            self.loss_history = pd.concat([self.loss_history, train_result], axis=0, ignore_index=True)
            X_num, X_cat, y_gen = self.generate_samples()
            self.evaluate_state(X_num, X_cat, y_gen) 
        # Generate evaluation.txt
        elapsed = time.time() - start
        self.evaluate_generation()
        self.loss_history.to_csv(os.path.join(self.raw_config['parent_dir'], 'RLAgentLoss.csv'), index=True)
        minutes, seconds = divmod(elapsed, 60)
        print(f"Training time without final evaluation: {int(minutes)} min {seconds:.2f} sec")
        
        
    def evaluate_generation(self):
        """
        Generates final evaluation of the trained model using all available samples.
        """
        X_num, X_cat, y_gen = self.generate_samples(num_samples=10000)
        synthetic_data = self.load_fake_data(X_num, X_cat, y_gen)
        stats, scores = evaluate_generation(synthetic=synthetic_data, original=self.real_data, num_numerical_features=self.raw_config['num_numerical_features'], category_sizes=self.category_sizes, task_type=self.task_type)
        with open(os.path.join(self.raw_config['parent_dir'], self.raw_config['evaluation_file']), 'w') as file:
            
            file.write(f"Similarity and Privacy evaluation\n")
            for key, value in stats.items():
                file.write(f"{key}: {value}\n")
            file.write(f"\nAbsolute Difference of Basic Statistics:\n")
            for key, value in scores.items():
                file.write(f"{key}: {value}\n")
            file.write("Statistics computed column-wise, then average is taken.\n")  
        
        
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
        start = time.time()
        
        eval_metrics = {}
        eval_metrics["dcr"] = compute_dcr(original_data=self.real_data, synthetic_data=X_fake, num_numerical_features=self.raw_config['num_numerical_features'], category_sizes=self.category_sizes, task_type=self.task_type, distance_metric='euclidean')
        eval_metrics["nndr"] = compute_nndr(original_data=self.real_data, synthetic_data=X_fake, num_numerical_features=self.raw_config['num_numerical_features'], category_sizes=self.category_sizes, task_type=self.task_type, distance_metric='euclidean')
        gower_matrix = compute_gowers_distance(original=self.real_data, synthetic=X_fake, n_num_features=self.raw_config['num_numerical_features'], category_sizes=self.category_sizes, task_type=self.task_type)
        eval_metrics["gower"] = np.mean(gower_matrix)
        end = time.time()
        print(f"Evaluation time: {end-start}s")
        print(f"dcr: {eval_metrics['dcr']}, nndr: {eval_metrics['nndr']}, gower: {eval_metrics['gower']}")
        

        for i in range(len(PRIVACY_CONFIG_DICT.keys())):
            
            # Using circular list to avoid retraining on the same privacy metric
            key = next(self.privacy_cycle)
            print(f"Evaluate on privacy metric: {key}")
            
            # prevent state deadlock, dont train in same state twice in a row
            if self.get_state().value == key:
                continue  
                
            if eval_metrics[key] < self.privacy_threshold[key]:
               print(f"{key}: {eval_metrics[key]:.4f} (Benchmark: {self.privacy_threshold[key]})")
               self.__state = PRIVACY_TO_STATE[key]
               return self.__state
               
            """        
            # Old implementation, abandoned on 02.12.2025
            if key == "nndr" and eval_metrics[key] > self.privacy_threshold[key]:
                print(f"{key}: {eval_metrics[key]:.4f} (Benchmark: {self.privacy_threshold[key]})")
                self.__state = State.LOW_NNDR
                return State.LOW_NNDR
            
            elif key == "dcr" and eval_metrics[key] < self.privacy_threshold[key]:
                print(f"{key}: {eval_metrics[key]:.4f} (Benchmark: {self.privacy_threshold[key]})")
                self.__state = State.LOW_DCR
                return State.LOW_DCR
            
            elif key == "gower" and eval_metrics[key] < self.privacy_threshold[key]:
                print(f"{key}: {eval_metrics[key]:.4f} (Benchmark: {self.privacy_threshold[key]})")
                self.__state = State.LOW_GOWER
                return State.LOW_GOWER
                
            """
            
        self.__state = State.HIGH
        return State.HIGH
                


    def train_model(self) -> DatasetInfo:
        """
        Set training parameters and settings based on current State.
        """
        state = self.get_state()
        
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
        elif state == State.LOW_DCR:
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
                continue_training=True,
                privacy_metric="dcr"
            )
        elif state == State.LOW_NNDR:
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
                continue_training=True,
                privacy_metric="nndr"
            )
        elif state == State.LOW_GOWER:
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
                continue_training=True,
                privacy_metric="gower"
            )
        else:
            raise ValueError("Invalid State encountered in RL Agent.")                   


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
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
    agent = RLAgent(name="RLAgent1", steps_per_round=2000, evaluation_samples=2000, args=args, raw_config=raw_config, device=device, rounds=15)
    agent.run_algorithm()
    
if __name__ == '__main__':
    main()