from genericpath import exists
import math

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
from evaluate_privacy import evaluate_generation, compute_dcr, compute_nndr, compute_gowers_DCR, compute_gowers_distance
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
from pathlib import Path
import gc
from datasetinfo import generate_dataset_info

DEFAULT_PRIVACY_METRIC = "dcr"



PRIVACY_CONFIG_DICT = {
    "dcr": 0.2,  # the higher, the better privacy, but the lower fidelity
    "nndr": 0.8,  # the higher, the better privacy
    "gower": 0.01  # the higher, the better privacy
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
    
class LowHighState(Enum):
    HIGH = 0
    LOW = 1
    
PRIVACY_TO_STATE = {
    "dcr": State.LOW_DCR,
    "nndr": State.LOW_NNDR,
    "gower": State.LOW_GOWER
    
}
    
class Action(Enum):
    Conventional = "Model in high privacy state, train with conventional loss function"
    Privacy = "Model in low privacy state, train with privacy-preserving loss function"


class RLAgent:
    def __init__(self, name="", evaluation_samples=5000, args=None, raw_config=None, device=None, rounds=None, pretrain_steps=None):
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
        self.evaluation_samples = evaluation_samples
        self.__state = State.HIGH
        self.privacy_cycle = cycle(PRIVACY_CONFIG_DICT.keys())
        self.loss_history = pd.DataFrame(columns=LOSS_HISTORY_COLUMNS)

        self.args = args  # loss vector approach saved in self.args.vector
        self.raw_config = raw_config
        self.device = device
        
        
        # self.privacy_metric = self.raw_config['privacy_metric'] if ('privacy_metric' in self.raw_config and self.raw_config['privacy_metric'] in PRIVACY_CONFIG_DICT.keys()) else DEFAULT_PRIVACY_METRIC
        self.privacy_threshold = self.load_privacy_config()
        self.evaluation_file = self.raw_config['evaluation_file'] if ('evaluation_file' in self.raw_config) and (self.raw_config['evaluation_file'].endswith(".json")) else "SynTabRL_eval.json"
        
        self.steps_per_round = self.raw_config['train']['main']['steps_per_round '] if 'steps_per_round' in self.raw_config['train']['main'] else 1000
        if self.raw_config['train']['main']['steps'] % self.steps_per_round != 0:
            self.steps_per_round = 1000
            if self.raw_config['train']['main']['steps'] % self.steps_per_round != 0:
                raise ValueError("Please specify a value to steps_per_round in config.toml under [train.main] such that it can divide the total training steps.")
        
        self.pretrain_steps = pretrain_steps if pretrain_steps else self.raw_config['train']['main']['pretrain_steps'] if 'pretrain_steps' in self.raw_config['train']['main'] else self.steps_per_round
        self.rounds = rounds if rounds else int(self.raw_config['train']['main']['steps'] / self.steps_per_round)
        self.total_steps = self.raw_config['train']['main']['steps'] + self.pretrain_steps
        self.completed_steps = 0
        self.actual_training_steps = 0  # to keep track of actual training steps excluding pretraining steps, used for logging and checkpointing purposes
        self.privacy_discount = self.raw_config['train']['main']['privacy_discount'] if "privacy_discount" in self.raw_config['train']['main'] else 0.1
        
        self.logsumexp_sigma = self.raw_config['train']['main']['logsumexp_sigma'] if 'logsumexp_sigma' in self.raw_config['train']['main'] else 0.01
        
        # Load real data for initial training and evaluation
        self.mm, self.ohe, self.X_num_real, self.X_cat_real = None, None, None, None    # X_cat_real is the onehot encoded categorical features of real data
        # Load real data with currently fitted mm and ohe if available
        extra_info = generate_dataset_info(real_data_path=self.raw_config['real_data_path'], change_val=self.args.change_val)  # generates DatasetInfo and saves to dataset_info folder if not already exists

        self.real_data, self.target_size, self.dataset_info = self.load_real_data(self.raw_config['real_data_path'])  # loads
        self.real_data = np.asarray(self.real_data)
        
        # Load real data for initial training and evaluation
        self.mm, self.ohe, self.X_num_real, self.X_cat_real = None, None, None, None    # X_cat_real is the onehot encoded categorical features of real data
        # Load real data with currently fitted mm and ohe if available
        extra_info = generate_dataset_info(real_data_path=self.raw_config['real_data_path'], change_val=self.args.change_val)  # generates DatasetInfo and saves to dataset_info folder if not already exists

        self.real_data, self.target_size, self.dataset_info = self.load_real_data(self.raw_config['real_data_path'])  # loads
        self.real_data = np.asarray(self.real_data)
                
        self.dataset_info.update(extra_info)
        self.dataset_info['category_sizes'].append(self.target_size)
        print(f"RL agent {self.name} initialized to train for dataset {self.dataset_info['name']} with {self.rounds} rounds, each with {self.steps_per_round} steps.\n")
        self.category_sizes = self.dataset_info.get('category_sizes')
        self.task_type = self.dataset_info.get('task_type')
        
    
    def get_state(self):
        return self.__state
    
        
    def load_real_data(self, real_path):
    
        """
        Adopted from tab-ddpm/scripts/resample_privacy.py, which inturns is adapted from https://github.com/Team-TUD/CTAB-GAN/tree/main/model/eval
        
        Loads and preprocesses the real dataset for evaluation.
        Args:
            real_path (str): Path to the real dataset directory.
            mm (MinMaxScaler, optional): Pre-fitted MinMaxScaler for numerical features. Defaults to None, i.e., a new scaler is fitted.
            ohe (OneHotEncoder, optional): Pre-fitted OneHotEncoder for categorical features. Defaults to None, i.e., a new encoder is fitted.
        
        Returns:
            X_real (np.ndarray): Preprocessed feature matrix of the real dataset.
            target_size (int): Size of the target variable (number of classes for classification, 1 for regression).
            dataset_info (dict): Information about the dataset (name, task type).
        """
        
        info = lib.load_json(real_path + "/info.json")
        dataset_info = {
            "name": info["name"],
            "task_type": info["task_type"],
            
        }
        X_num_real, X_cat_real, y_real = lib.read_pure_data(real_path, 'train')
        target_size = 0
        
        if dataset_info["task_type"] == 'regression':
            print(f"y_real.shape: {y_real.shape}")
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
        
        self.X_num_real = X_num_real  # Save for later use in load_fake_data
        
        # print(f"in load_real_data, self.mm: {self.mm}")
        mm = self.mm if self.mm else MinMaxScaler().fit(X_num_real)
        # self.mm = MinMaxScaler().fit(X_num_real) if self.mm is None else self.mm

        X_real = mm.transform(X_num_real)
        X_real = np.clip(X_real, 0, 1)
        has_negative = (X_real < 0).any()
        # print(f"in load_real_data, real data transformed with mm has negative: {has_negative}")
        has_larger = (X_real > 1).any()
        # print(f"in load_real_data, real data transformed with mm has larger than 1: {has_larger}")

        if X_cat_real is not None:
            if self.ohe is None:  # First time loading real data
                self.ohe = OneHotEncoder().fit(X_cat_real)  # fit and save, use throughout training
                self.X_cat_real = self.ohe.transform(X_cat_real) / np.sqrt(2)
                self.X_cat_real = self.X_cat_real.todense()
            X_real = np.concatenate([X_real, self.X_cat_real], axis=1)
        
        return X_real, target_size, dataset_info
    
    
    def load_fake_data(self, X_num_fake, X_cat_fake, y_gen, for_training=True):
        """
        Adopted from tab-ddpm/scripts/resample_privacy.py, which inturns is adapted from https://github.com/Team-TUD/CTAB-GAN/tree/main/model/eval
        
        """
        fake_path = self.raw_config['parent_dir']
        if for_training:
            print(f"Using lib.read_pure_data to read generated data from {fake_path}.")
            X_num_fake, X_cat_fake, y_gen = lib.read_pure_data(fake_path, 'train')
        st = time.time()
        if self.task_type == 'regression':
            X_num_fake = np.concatenate([X_num_fake, y_gen[:, np.newaxis]], axis=1)
        else:  # classification, binclass or multiclass
            if X_cat_fake is None:
                X_cat_fake = y_gen[:, np.newaxis].astype(int).astype(str)
                
            else:
                X_cat_fake = np.concatenate([X_cat_fake, y_gen[:, np.newaxis].astype(int).astype(str)], axis=1)
        
        if len(y_gen) > self.evaluation_samples and for_training:
            ixs = np.random.choice(len(y_gen), self.evaluation_samples, replace=False)
            X_num_fake = X_num_fake[ixs]
            X_cat_fake = X_cat_fake[ixs] if X_cat_fake is not None else None

        # Every round, fit mm on combined real and fake numerical data to ensure values normalized between 0 and 1
        combined_num = np.concatenate([self.X_num_real, X_num_fake], axis=0)
        if for_training or self.mm is None:
            self.mm = MinMaxScaler().fit(combined_num)
        # self.mm = MinMaxScaler().fit(X_num_fake)
        X_fake = self.mm.transform(X_num_fake)
        has_larger = (X_fake > 1).any()
        has_negative = (X_fake < 0).any()
        # print(f"in load_fake_data, X_fake has elements > 1: {has_larger}.")
        # print(f"In rlagent.py, load_fake_data(), X_fake has negative: {has_negative}")
        if has_larger or has_negative:
            print("X_fake has elements larger than 1 or negative after MinMaxScaler transform before clipping.")
        X_fake = np.clip(X_fake, 0, 1)
        if (X_cat_fake is not None) and (self.ohe is not None):
            X_cat_fake = self.ohe.transform(X_cat_fake) / np.sqrt(2)
            X_fake = np.concatenate([X_fake, X_cat_fake.todense()], axis=1)
        
        has_negative = (X_fake < 0).any()
        # print(f"In rlagent.py, load_fake_data(), X_fake that is fitted with X_num_fake has negative: {has_negative}")
        
            
        # Everytime we call call load_fake_data, we also reload real data to avoid data leakage due to fitted mm and ohe    
        self.real_data, self.target_size, self.dataset_info = self.load_real_data(self.raw_config['real_data_path'])  # loads
        self.real_data = np.asarray(self.real_data)
        
        has_negative = (self.real_data < 0).any()
        # print(f"In rlagent.py, load_fake_data89, self.real_data transformed with X_num_fake has negative: {has_negative}")
        
        
        et = time.time()
        # print(f"load_fake_data time: {et-st}s")
        return X_fake
    
    
    def load_privacy_config(self):
        """
        Loads privacy configuration from PRIVACY_CONFIG_DICT based on the selected privacy metric.
        
        Returns:
            float: The privacy threshold for the selected metric.
        """
        if ('privacy_config' in self.raw_config):
            print(f"Loading privacy config from raw_config")
            privacy_config = {}
            for key in PRIVACY_CONFIG_DICT.keys():
                if key in self.raw_config['privacy_config']:
                    privacy_config[key] = self.raw_config['privacy_config'][key]
                else:
                    privacy_config[key] = PRIVACY_CONFIG_DICT[key]
            return privacy_config           
        
        else:
            print("Use default configuration as privacy threshold.")
            return PRIVACY_CONFIG_DICT
        
    
    def pretrain(self, steps=None):
        """
        Before incorporating privacy, train the model using conventional tabddpm training script.
        """
        if steps:
            train_result = train(
                    steps=steps,
                    start_privacy_step=-1,
                    **self.raw_config['train']['main'],
                    **self.raw_config['diffusion_params'],
                    parent_dir=self.raw_config['parent_dir'],
                    real_data_path=self.raw_config['real_data_path'],
                    model_type=self.raw_config['model_type'],
                    model_params=self.raw_config['model_params'],
                    T_dict=self.raw_config['train']['T'],
                    num_numerical_features=self.raw_config['num_numerical_features'],
                    device=self.device,
                    change_val=self.args.change_val,
                    continue_training=False,
                    total_steps=self.total_steps
                )
            self.loss_history = pd.concat([self.loss_history, train_result], axis=0, ignore_index=True)
            self.completed_steps += steps
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
                    continue_training=False,
                    total_steps=self.total_steps
                )
            self.loss_history = pd.concat([self.loss_history, train_result], axis=0, ignore_index=True)
            self.completed_steps += self.steps_per_round

    
    def run_algorithm(self):
        """
        RL agent algorithm. Interact with environment and take action for n rounds, then log results.
        Pretraining takes place at Agent object instantiation (__init__ function).
        """
        start = time.time()
        counter = 0
        print("Starting RL Agent training algorithm.")
        self.load_checkpoint()

        for _ in range(self.rounds):
            st = time.time()
            counter += 1
            X_num, X_cat, y_gen = self.generate_samples()
            if self.args.vector_approach:
                self.__state = self.evaluate_state_vector(X_num, X_cat, y_gen)
                print(f"=== RL Agent Round {counter} ===")
                print(f"State: {self.get_state().name}")
                train_result = self.train_model_vector()
            elif self.args.weighted_vector:
                print("Using weighted vector approach.")
                self.__state = self.evaluate_state(X_num, X_cat, y_gen)
                print(f"=== RL Agent Round {counter} ===")
                print(f"State: {self.get_state().name}")
                train_result = self.train_model_weighted_vector()
            elif self.args.sum_approach:
                print("Using sum approach.")
                eval_metrics = self.evaluate_state_vector(X_num, X_cat, y_gen)
                print(f"=== RL Agent Round {counter} ===")
                print(f"State: {eval_metrics}")
                train_result = self.train_model_sum(eval_metrics)
            elif self.args.metric:
                print(f"Using single metric approach on {self.args.metric}.")
                self.__state = self.evaluate_state_single_metric(X_num, X_cat, y_gen, self.args.metric)
                print(f"=== RL Agent Round {counter} ===")
                print(f"State: {self.get_state().name}")
                train_result = self.train_model_single_metric(self.args.metric)
            elif self.args.continuous_approach:
                self.__state = self.evaluate_state_continuous(X_num, X_cat, y_gen)
                print(f"=== RL Agent Round {counter} ===")
                print(f"Continuous State: {self.__state}")
                weighting_factors = self.continuous_weighting_factors(self.__state)
                train_result = self.train_model_continuous(weighting_factors)
            elif self.args.adaptive_approach:
                print("Using adaptive approach.")
                self.__state = self.evaluate_state_continuous(X_num, X_cat, y_gen)
                print(f"=== RL Agent Round {counter} ===")
                print(f"Continuous State: {self.__state}")
                weighting_factors = self.continuous_weighting_factors()
                train_result = self.train_model_adaptive(weighting_factors)
            elif self.args.adaptive_single_metric:
                print("Using adaptive approach only on single metric.")
                single_metric = "adaptive_" + self.args.adaptive_single_metric
                self.__state = self.evaluate_state_continuous(X_num, X_cat, y_gen)
                print(f"=== RL Agent Round {counter} ===")
                print(f"Continuous State: {self.__state}")
                weighting_factors = self.continuous_weighting_factors()
                train_result = self.train_model_adaptive(weighting_factors, privacy_metric=single_metric)
            else:                
                self.__state = self.evaluate_state(X_num, X_cat, y_gen)
                print(f"=== RL Agent Round {counter} ===")
                print(f"State: {self.get_state().name}")
                train_result = self.train_model()
            self.loss_history = pd.concat([self.loss_history, train_result], axis=0, ignore_index=True)
            self.completed_steps += self.steps_per_round
            self.actual_training_steps += self.steps_per_round
            self.save_checkpoint()
            et = time.time()
            print(f"Round {counter} training time: {et-st}s")
        X_num, X_cat, y_gen = self.generate_samples(self.evaluation_samples)
        final_state = self.evaluate_state(X_num, X_cat, y_gen)
        self.remove_checkpoint()
            
        # Generate evaluation.txt
        elapsed = time.time() - start
        if self.args.eval:
            self.evaluate_generation(elapsed_time=elapsed)
        self.loss_history.to_csv(os.path.join(self.raw_config['parent_dir'], 'RLAgentLoss.csv'), index=True)
            
    # ------------ End of run_algorithm() ---------------------------   
    def save_checkpoint(self):
        checkpoint_path = os.path.join(self.raw_config['parent_dir'], 'checkpoint.pt')
        torch.save({
            'completed_steps': self.completed_steps, 
            'actual_training_steps': self.actual_training_steps,
        }, checkpoint_path)
        
    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.raw_config['parent_dir'], 'checkpoint.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if checkpoint["completed_steps"] >= self.total_steps:
                print("Checkpoint indicates training already completed. Starting from scratch.")
                print("Pretraining in load_checkpoint.")
                self.pretrain()
                return
            self.completed_steps = checkpoint["completed_steps"]
            self.actual_training_steps = checkpoint["actual_training_steps"]
            print(f"Checkpoint loaded. Resuming from step {self.completed_steps}.")
            assert self.actual_training_steps % self.steps_per_round == 0, "Checkpoint completed steps should be a multiple of steps_per_round. Please restart training for consistency."
            self.rounds -= self.actual_training_steps // self.steps_per_round
            print(f"Adjusted remaining rounds to {self.rounds} based on loaded checkpoint.")
            
        else:
            print("No checkpoint found. Starting from scratch.")
            print("Pretraining in load_checkpoint.")
            self.pretrain()
    
    def remove_checkpoint(self):
        checkpoint_path = os.path.join(self.raw_config['parent_dir'], 'checkpoint.pt')
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)    
    
    def continuous_weighting_factors(self):
        """
        Algorithm to compute how each loss term should be waited in the upcoming round of training, based on continuous privacy evaluation.
        Use progress to scale the weights, so that at the beginning of training, the privacy weights are small, and at the end of training, the weights are larger.
        Depend on current privacy level? If not, how to pick arbitrarily?
        
        Args:
            eval_metrics (dict): Dictionary containing the evaluation metrics.
        Returns:
            list: Weighting factors for loss function.
        """
        dcr, nndr, gower = self.__state.values()
        progress = self.completed_steps / self.total_steps
        weights = {}
        # weights['dcr'] = self.privacy_threshold['dcr'] / dcr * progress
        weights['dcr'] = self.privacy_threshold['dcr'] / np.exp(-dcr) * progress
        weights['nndr'] = self.privacy_threshold['nndr'] / nndr * progress
        weights['gower'] = self.privacy_threshold['gower'] / gower * progress
        
        return weights
        
        
    def get_loss_weight_mask(self):
        """
        Create a weight mask for the privacy-preserving loss function based on feature types.
        The order is [dcr, nndr, gower]
        
        Returns:
            list: Weight mask for loss function: weight the low privacy metric twice as much as others.
        """
        if self.__state == State.LOW_DCR:
            return [2.0, 1.0, 1.0]
        elif self.__state == State.LOW_NNDR:
            return [1.0, 2.0, 1.0]
        elif self.__state == State.LOW_GOWER:
            return [1.0, 1.0, 2.0]
        else:
            raise TypeError("get_loss_weight_mask called in HIGH privacy state, which is invalid.")
    
    def train_model_adaptive(self, weighting_factors, privacy_metric='adaptive'):
        # The Agent trains with the privacy loss(es) that display a low privacy level
        
        # Softer state transitions    
        
        print("Train with continuous privacy loss.")
        
        assert self.privacy_threshold.keys() == self.__state.keys(), "Mismatching privacy threshold keys and privacy metrics at training time."
        is_high_privacy = True
        adaptive_weights = []
        for key in self.privacy_threshold.keys():
            if self.__state[key] < self.privacy_threshold[key]:
                is_high_privacy = False
                adaptive_weights.append(weighting_factors[key])
            else:
                adaptive_weights.append(weighting_factors[key] * self.privacy_discount)
        assert len(adaptive_weights) == 3, "Length of adaptive weights should be 3, corresponding to dcr, nndr, gower."
        
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
                privacy_metric=privacy_metric,  # continuous approach has similarity with vector approach, recycle vector approach code
                weight_mask=adaptive_weights,  # conversion to torch tensor in train(), requires_grad = False
                completed_steps=self.completed_steps,
                total_steps=self.total_steps
            )
        # Softer state transitions     
        
        
        """    
        # Hard state cutoff
        
        
        
        
        if is_high_privacy:
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
                continue_training=True,
                completed_steps=self.completed_steps,
                total_steps=self.total_steps
            )
        else: # low privacy state, train with privacy loss
            print("Train with privacy loss with adaptive weights:", adaptive_weights)
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
                privacy_metric="adaptive",  # continuous approach has similarity with vector approach, recycle vector approach code
                weight_mask=adaptive_weights,  # conversion to torch tensor in train(), requires_grad = False
                completed_steps=self.completed_steps,
                total_steps=self.total_steps
            )
            # Hard state cutoff
            
            """ 
            
    def train_model_continuous(self, weighting_factors):
        # The Agent trains with the privacy loss(es) that display a low privacy level
        print("Train with continuous privacy loss.")

        assert self.privacy_threshold.keys() == self.__state.keys(), "Mismatching privacy threshold keys and privacy metrics at training time."  
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
            privacy_metric="vector",  # continuous approach has similarity with vector approach, recycle vector approach code
            weight_mask=list(weighting_factors.values()),  # conversion to torch tensor in train(), requires_grad = False
            completed_steps=self.completed_steps,
            total_steps=self.total_steps
        )
     
    def train_model_weighted_vector(self):
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
                continue_training=True,
                completed_steps=self.completed_steps,
                total_steps=self.total_steps
            )
        else:
            action = Action.Privacy
            print("Train with vector privacy loss.")
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
                privacy_metric="vector",
                weight_mask=self.get_loss_weight_mask(),
                completed_steps=self.completed_steps,
                total_steps=self.total_steps
            )
        
     
        
    def train_model_vector(self):
        st = time.time()
        state = self.get_state()
        if state == LowHighState.HIGH:
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
                continue_training=True,
                completed_steps=self.completed_steps,
                total_steps=self.total_steps
            )
        elif state == LowHighState.LOW:
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
                privacy_metric="vector",
                completed_steps=self.completed_steps,
                total_steps=self.total_steps
            )
        elapsed = time.time() - st
        print(f"train_model_vector time: {elapsed}s")
        
    def train_model_sum(self, eval_metrics):
        state = self.get_state()
        action = Action.Privacy
        print("Train with sum privacy loss.")
        weights = []
        for key in eval_metrics.keys():
            weights.append(self.privacy_threshold[key] / eval_metrics[key])
            print(f"{key}: {eval_metrics[key]:.4f} (Benchmark: {self.privacy_threshold[key]})")
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
            privacy_metric="sum",
            completed_steps=self.completed_steps,
            total_steps=self.total_steps,
            weight_mask=weights
        )

            
    def evaluate_state_continuous(self, X_num, X_cat, y_gen, for_training=True):
        """
        Evaluates the privacy state of the model based on generated samples, if one of the privacy metrics
        is below threshold, return Low state, else High state. 
        I.e. Convert from State enum to LowHighState enum.
        """
        X_fake = self.load_fake_data(X_num, X_cat, y_gen, for_training=for_training)
        X_fake = np.asarray(X_fake)
        
        distance_parameters = {
        "original": self.real_data,
        "synthetic": X_fake,
        "num_numerical_features": self.raw_config["num_numerical_features"],
        "category_sizes": self.category_sizes,
        "task_type": self.task_type
        }
        
        eval_metrics = {}
        eval_metrics["dcr"] = compute_dcr(**distance_parameters, distance_metric='euclidean')
        eval_metrics["nndr"] = compute_nndr(**distance_parameters, distance_metric='euclidean')
        gower_matrix = compute_gowers_DCR(**distance_parameters)
        eval_metrics["gower"] = gower_matrix
        
        return eval_metrics

        
    def evaluate_generation(self, elapsed_time=None, X_num=None, X_cat=None, y_gen=None):
        """
        Generates final evaluation of the trained model using all available samples.
        """
        if y_gen is None:
            X_num, X_cat, y_gen = self.generate_samples(self.raw_config['sample']['num_samples'])
        synthetic_data = self.load_fake_data(X_num, X_cat, y_gen)
        stats, scores = evaluate_generation(synthetic=synthetic_data, original=self.real_data, num_numerical_features=self.raw_config['num_numerical_features'], category_sizes=self.category_sizes, task_type=self.task_type)
        res = self.evaluate_ml()
        
        eval_result = stats | scores | res.get_metrics()
        minutes, seconds = divmod(elapsed_time, 60)
        eval_result["elapsed_time"] = f"Total training time: {int(minutes)} min {seconds:.2f} sec"
        eval_path = str(Path(self.raw_config['parent_dir']) / self.evaluation_file)
        lib.dump_json(eval_result, eval_path)
        """
        with open(os.path.join(self.raw_config['parent_dir'], self.evaluation_file), 'w') as file:
            
            file.write(f"Similarity and Privacy evaluation\n")
            for key, value in stats.items():
                file.write(f"{key}: {value}\n")
            file.write(f"\nAbsolute Difference of Basic Statistics:\n")
            for key, value in scores.items():
                file.write(f"{key}: {value}\n")
            file.write("Statistics computed column-wise, then average is taken.\n")  
            
            file.write(f"\nMachine Learning Evaluation Results:\n")
            file.write(f"{res.get_metrics()}\n")
                
            if elapsed_time is not None:
                minutes, seconds = divmod(elapsed_time, 60)
                print(f"Total training time: {int(minutes)} min {seconds:.2f} sec")
                file.write(f"\nTotal training time: {int(minutes)} min {seconds:.2f} sec\n")
        """
        
        
    def generate_samples(self, num_samples=5000, seed_offset=0, **kwargs):
        """Generates samples using the current model and configuration. 2000 by default"""
        num_samples = num_samples
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
            seed=self.raw_config['sample'].get('seed', 0)+seed_offset,
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
        
        distance_parameters = {
        "original": self.real_data,
        "synthetic": X_fake,
        "num_numerical_features": self.raw_config["num_numerical_features"],
        "category_sizes": self.category_sizes,
        "task_type": self.task_type
        }
        
        eval_metrics = {}
        eval_metrics["dcr"] = compute_dcr(**distance_parameters, distance_metric='euclidean')
        eval_metrics["nndr"] = compute_nndr(**distance_parameters, distance_metric='euclidean')
        # gower_matrix = compute_gowers_distance(**distance_parameters)
        # eval_metrics["gower"] = np.mean(gower_matrix)
        eval_metrics["gower"] = compute_gowers_DCR(**distance_parameters)
        end = time.time()
        print(f"Evaluation time: {end-start}s")
        print(f"dcr: {eval_metrics['dcr']}, nndr: {eval_metrics['nndr']}, gower: {eval_metrics['gower']}")
        

        for i in range(len(PRIVACY_CONFIG_DICT.keys())):
            
            # Using circular list to avoid retraining on the same privacy metric
            key = next(self.privacy_cycle)
            # print(f"Evaluate on privacy metric: {key}")
            
            # prevent state deadlock, dont train in same state twice in a row
            # if self.get_state().value == key:
                # continue  
                
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
        return State.HIGH
    
    def evaluate_state_single_metric(self, X_num, X_cat, y_gen, single_metric):
        """
        Evaluates the privacy state of the model based on generated samples and a single privacy metric.
        
        Args:
            X_num (np.ndarray): Numerical features of generated samples.
            X_cat (np.ndarray): Categorical features of generated samples.
            y_gen (np.ndarray): Generated target variable.
            K (list): Category sizes for categorical features.
        Returns:
            State: The evaluated privacy state (HIGH or LOW).
            """
        def get_gower_mean(**params):
            return np.mean(compute_gowers_distance(**params))
        
        lookup = {
            "dcr": compute_dcr,
            "nndr": compute_nndr,
            "gower": compute_gowers_DCR
        }    
        assert single_metric in lookup.keys(), f"In RLAgent.evaluate_state_single_metric, single_metric must be one of {lookup.keys()}"
        X_fake = self.load_fake_data(X_num, X_cat, y_gen)
        X_fake = np.asarray(X_fake)
        start = time.time()
        
        distance_parameters = {
        "original": self.real_data,
        "synthetic": X_fake,
        "num_numerical_features": self.raw_config["num_numerical_features"],
        "category_sizes": self.category_sizes,
        "task_type": self.task_type
        }

        eval_metrics = {}
        eval_metrics["dcr"] = compute_dcr(**distance_parameters, distance_metric='euclidean')
        eval_metrics["nndr"] = compute_nndr(**distance_parameters, distance_metric='euclidean')
        eval_metrics["gower"] = compute_gowers_DCR(**distance_parameters)
        end = time.time()
        print(f"Evaluation time: {end-start}s")
        print(f"dcr: {eval_metrics['dcr']}, nndr: {eval_metrics['nndr']}, gower: {eval_metrics['gower']}")
        
        if eval_metrics[single_metric] < self.privacy_threshold[single_metric]:
            print(f"{single_metric}: {eval_metrics[single_metric]:.4f} (Benchmark: {self.privacy_threshold[single_metric]})")
            self.__state = PRIVACY_TO_STATE[single_metric]
            return self.__state
        else:
            return State.HIGH
        
    
    def train_model_single_metric(self, metric):
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
                continue_training=True,
                completed_steps=self.completed_steps,
                total_steps=self.total_steps
            )
        else:
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
                privacy_metric=metric,
                completed_steps=self.completed_steps,
                total_steps=self.total_steps
            )
                


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
                continue_training=True,
                completed_steps=self.completed_steps,
                total_steps=self.total_steps
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
                privacy_metric="dcr",
                completed_steps=self.completed_steps,
                total_steps=self.total_steps
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
                privacy_metric="nndr",
                completed_steps=self.completed_steps,
                total_steps=self.total_steps
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
                privacy_metric="gower",
                completed_steps=self.completed_steps,
                total_steps=self.total_steps
            )
        else:
            raise ValueError("Invalid State encountered in RL Agent.")
        
        
    def evaluate_ml(self):
        res = None
        if self.raw_config['eval']['type']['eval_model'] == 'catboost':
            res = train_catboost(
                parent_dir=self.raw_config['parent_dir'],
                real_data_path=self.raw_config['real_data_path'],
                eval_type=self.raw_config['eval']['type']['eval_type'],
                T_dict=self.raw_config['eval']['T'],
                seed=self.raw_config['seed'],
                change_val=self.args.change_val
            )
        elif self.raw_config['eval']['type']['eval_model'] == 'mlp':
            res = train_mlp(
                parent_dir=self.raw_config['parent_dir'],
                real_data_path=self.raw_config['real_data_path'],
                eval_type=self.raw_config['eval']['type']['eval_type'],
                T_dict=self.raw_config['eval']['T'],
                seed=self.raw_config['seed'],
                change_val=self.args.change_val,
                device=self.device
            )
        elif self.raw_config['eval']['type']['eval_model'] == 'simple':
            res = train_simple(
                parent_dir=self.raw_config['parent_dir'],
                real_data_path=self.raw_config['real_data_path'],
                eval_type=self.raw_config['eval']['type']['eval_type'],
                T_dict=self.raw_config['eval']['T'],
                seed=self.raw_config['seed'],
                change_val=self.args.change_val
            )
        return res
                        
    def generate_and_filter_samples_percentile(self, num_samples, value, evaluate_sample_file=False, **kwargs):
        assert value is None or (0 <= value <= 1), "In percentile approach, value must be between 0 and 1, representing the percentage of samples to filter out based on DCR."
        if value is None:
            value = 0.1  # default to filtering out the 10% closest samples if no value is provided
        buffer_factor = 1 - value
        total_to_generate = math.ceil(num_samples / buffer_factor)
    
        X_num, X_cat, y_gen = self.generate_samples(num_samples=total_to_generate)
        
        if evaluate_sample_file:
            eval_metrics_before = self.evaluate_state_continuous(X_num, X_cat, y_gen, for_training=False)
            print(f"Evaluating generated samples before filtering: dcr: {eval_metrics_before['dcr']}, nndr: {eval_metrics_before['nndr']}, gower: {eval_metrics_before['gower']}")
    
        # 2. Compute Privacy Metric: Distance to Closest Record (DCR)
        # Ensure load_fake_data is processing ALL generated rows
        synthetic_data = self.load_fake_data(X_num, X_cat, y_gen, for_training=False)
        print(f"synthetic_data.shape: {synthetic_data.shape}")
        
        min_distances = compute_dcr(
            original=self.real_data,
            synthetic=synthetic_data,
            num_numerical_features=self.raw_config["num_numerical_features"],
            category_sizes=self.category_sizes,
            task_type=self.task_type,
            distance_metric='euclidean',
            return_min_distances=True
        )

        # min_distances must have the same length as X_num (e.g., 5500)
        actual_gen_count = X_num.shape[0] if X_num is not None else X_cat.shape[0]
        if len(min_distances) != actual_gen_count:
            raise ValueError(f"DCR length ({len(min_distances)}) mismatch with generated samples ({actual_gen_count})")

        # 3. Filter: Remove the 10% smallest distances
        threshold = np.percentile(min_distances, 10)
        print(f"Threshold at 10th percentile: {threshold:.4f}\nAt 90th percentile: {np.percentile(min_distances, 90):.4f}")
        keep_indices = min_distances > threshold
        
        # 4. Apply mask and slice to requested num_samples
        # We use boolean indexing first, then slice the resulting array
        X_num_filtered = X_num[keep_indices][:num_samples] if X_num is not None else None
        
        # Handle X_cat properly even if it's currently None in your test
        X_cat_filtered = X_cat[keep_indices][:num_samples] if X_cat is not None else None
        
        y_gen_filtered = y_gen[keep_indices][:num_samples]
        
        if evaluate_sample_file:
            eval_metrics_after = self.evaluate_state_continuous(X_num_filtered, X_cat_filtered, y_gen_filtered, for_training=False)
            print(f"Evaluating generated samples after filtering: dcr: {eval_metrics_after['dcr']}, nndr: {eval_metrics_after['nndr']}, gower: {eval_metrics_after['gower']}")
            with open(evaluate_sample_file, 'a') as file:
                file.write("Metrics before filtering:\n")
                for metric, value in eval_metrics_before.items():
                    file.write(f"{metric}: {value}\n")
                
                file.write("\nMetrics after filtering:\n")
                for metric, value in eval_metrics_after.items():
                    file.write(f"{metric}: {value}\n")
                    
                file.write("\nRelative gain in privacy metrics (after/before) in %:\n")
                for metric in eval_metrics_before.keys():
                    ratio = (eval_metrics_after[metric] / eval_metrics_before[metric] - 1) * 100
                    file.write(f"{metric}: {ratio:.4f}%\n")

    
        return X_num_filtered, X_cat_filtered, y_gen_filtered
    
    
    def generate_and_filter_samples_threshold(self, num_samples, value, max_value):
        if value is None:
            value = 0.15  # default to a DCR threshold of 0.15 if no value is provided
        final_X_num, final_X_cat, final_y = [], [], []
        count = 0
        iterations = 0
        chunk_size = int(num_samples * 1.2) 
        # Keep going until we have collected exactly num_samples
        while count < num_samples:
            if iterations >= 10:
                print("Warning: Exceeded 10 iterations in threshold filtering. Consider adjusting the threshold.")
                break
            # 1. Generate a chunk of data (generate more than needed to be efficient)
            remaining = num_samples - count
            print(f"Remaining samples to generate: {remaining}")
            
            X_n, X_c, y_g = self.generate_samples(num_samples=chunk_size, seed_offset=iterations)
            
            # 2. Compute DCR for this chunk
            synthetic_data = self.load_fake_data(X_n, X_c, y_g, for_training=False)
            min_distances = compute_dcr(
                original=self.real_data,
                synthetic=synthetic_data,
                num_numerical_features=self.raw_config["num_numerical_features"],
                category_sizes=self.category_sizes,
                task_type=self.task_type,
                return_min_distances=True
            )
            
            # 3. Filter: Only keep samples strictly GREATER than the threshold and SMALLER than optional max_value
            keep_indices = min_distances > value
            if max_value is not None:
                keep_indices &= (min_distances < max_value)
            
            # print(f"Xn.shape: {X_n.shape if X_n is not None else 'None'}, min_distances.shape: {min_distances.shape}, DCR Threshold: {value}")
            
            # 4. Collect the safe samples
            if X_n is not None:
                final_X_num.append(X_n[keep_indices])
            if X_c is not None:
                final_X_cat.append(X_c[keep_indices])
            final_y.append(y_g[keep_indices])
            
            # Update our current count
            count += np.sum(keep_indices)
            iterations += 1
            print(f"Iteration {iterations+1}: Generating {count} samples, {num_samples - count} more needed after filtering.")
            
        # 5. Concatenate and trim to exact num_samples
        X_num_out = np.concatenate(final_X_num, axis=0)[:num_samples] if final_X_num else None
        X_cat_out = np.concatenate(final_X_cat, axis=0)[:num_samples] if final_X_cat else None
        y_gen_out = np.concatenate(final_y, axis=0)[:num_samples]
            
        return X_num_out, X_cat_out, y_gen_out



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    
    group = parser.add_mutually_exclusive_group()  # privacy_approach_group
    group.add_argument('--no_privacy', action='store_true', default=False)
    group.add_argument('--vector_approach', action='store_true', default=False)
    group.add_argument('--weighted_vector', action='store_true', default=False)
    group.add_argument('--sum_approach', action='store_true', default=False)
    group.add_argument('--continuous_approach', action='store_true', default=False)
    group.add_argument('--adaptive_approach', action='store_true', default=False)
    group.add_argument(
        '--adaptive_single_metric',
        choices=["dcr", "nndr", "gower"],
        help="Toggle adaptive single metric mode. Please specify which one to use.",
    )
    group.add_argument(
        "--single_metric",
        dest="metric",
        choices=["dcr", "nndr", "gower"],
        help="Toggle single metric mode. Please specify which one to use.",
    )
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--sample', 
                        type=int,
                        help="The number of samples to generate",
                        nargs='?',
                        const=-1
                    )
    parser.add_argument(
        "--filter", 
        choices=["percentile", "threshold"],
        default=None, 
        help="Select the filtering logic: 'percentile' (e.g., drop bottom 10%%) "
             "or 'threshold' (e.g., drop if DCR < 0.05)"
    )
    parser.add_argument(
        "--filter_value", 
        type=float, 
        help="The numeric value for the approach (percentile 0-100 or DCR distance)"
    )
    parser.add_argument(
        '--max_value', 
        type=float, 
        default=None,
        help='Sets the max value (only available if --sample and --filter is set)'
    )
    
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--change_val', action='store_true',  default=False)
    
    print("Parsing arguments ...")

    args = parser.parse_args()
    raw_config = lib.load_config(args.config)
    if 'device' in raw_config:
        device = torch.device('cuda:0')  # Paul
        # device = torch.device(raw_config['device'])  # Use specified device
    else:
        device = torch.device('cuda:0')  # Original 'cuda:1'
    # assert "evaluation_file" in raw_config, "evaluation_file key missing in config.toml"
    print("Starting agent ...")
    agent = RLAgent(name="RLAgent1", args=args, raw_config=raw_config, device=device)
    X_num, X_cat, y_gen = None, None, None
    st = time.time()
    
    if args.no_privacy:
        args.train = False
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
        change_val=args.change_val
        )

    if args.train:
        agent.run_algorithm()
        
    if args.sample is not None:
        if args.filter is None:
            sample_method = agent.generate_samples
        else:
            sample_method = agent.generate_and_filter_samples_percentile if args.filter == "percentile" else agent.generate_and_filter_samples_threshold
        
        if args.sample < 0:
            args.sample = agent.raw_config['sample']['num_samples']

        X_num, X_cat, y_gen = sample_method(num_samples=args.sample, value=args.filter_value, max_value=args.max_value)
        
        """
        if X_num is not None:
            print(f"Generated samples: X_num shape {X_num.shape}") 
        if X_cat is not None:    
          print(f"Generated samples: X_cat shape {X_cat.shape}")
        print(f"Generated samples: y_gen shape {y_gen.shape}")
        
        print(f"X_num type: {type(X_num)}, X_cat type: {type(X_cat)}, y_gen type: {type(y_gen)}")
        """
    elapsed_time = time.time() - st    
    if args.eval and not args.train:
        
        agent.evaluate_generation(elapsed_time=elapsed_time, X_num=X_num, X_cat=X_cat, y_gen=y_gen)
        
    
if __name__ == '__main__':
    main()