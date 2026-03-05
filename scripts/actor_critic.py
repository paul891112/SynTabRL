import tomli
import shutil
import os
import argparse
from typing import Union, Dict
from itertools import cycle
from train import train, Trainer, DatasetInfo, move_optimizer_to_device
from sample import sample
from eval_catboost import train_catboost
from eval_mlp import train_mlp
from eval_simple import train_simple
from evaluate_privacy import evaluate_generation, compute_dcr, compute_nndr, compute_gowers_distance
from train import LOSS_HISTORY_COLUMNS
from utils_train import get_model, make_dataset, update_ema
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import os
from tab_ddpm import GaussianMultinomialDiffusion
import pandas as pd
import matplotlib.pyplot as plt
import zero
import lib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.stats import wasserstein_distance
from enum import Enum
import time


PRIVACY_CONFIG_DICT = {
    "dcr": 0.3,  # the higher, the better privacy, but the lower fidelity
    "nndr": 0.7,  # the higher, the better privacy
    "gower": 0.3  # the higher, the better privacy
                          }

class Actor(Trainer):
    """
    Actor network for selecting actions based on the current state.
    Based on TabDDPM, adapts the train() function in scripts/train.py.
    
    """
    def __init__(self, diffusion,  train_iter, lr, weight_decay, steps, device=torch.device('cuda:1'), optimizer=None, total_steps=None):
        """
        The loaded Diffusion model will already be pretrained for steps_per_round steps.
        """
        super().__init__(diffusion,  train_iter, lr, weight_decay, steps, device, optimizer, total_steps)
        self.D = None
        
        
    def get_optimizer(self):
        # Return the optimizer for the actor network
        return self.optimizer

    def add_pretraining_result(self, train_result):
        self.loss_history = pd.concat([self.loss_history, train_result], axis=0, ignore_index=True)
    
    def add_metadata(self, D):
        self.D = D    
    
    def run_step(self, x, out_dict):
        """
        One step of learning: comute loss, update weights.
        Original implementation, no privacy loss term
        Does not call loss.backward(), this is moved to Actor.update()

        Args:
            x (_type_): _description_
            out_dict (_type_): yield from FastTensorDataLoader, in lib/data.py
        Returns:
            _type_: _description_
        """
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        
        loss_multi, loss_gauss, privacy_multi, privacy_gauss = self.diffusion.mixed_loss(x, out_dict)
        
        loss = loss_multi + loss_gauss  # integrate advantage as a loss term to backpropagate

        return loss_multi, loss_gauss, loss
        

    def load_state_dict(self, state_dict):
        # Load the state dictionary for the actor network
        pass
    
    def generate_samples(
        self, 
        real_data_path = 'data/higgs-small',
        batch_size = 2000,
        num_samples = 0,
        model_type = 'mlp',
        model_params = None,
        model_path = None,
        num_timesteps = 1000,
        gaussian_loss_type = 'mse',
        scheduler = 'cosine',
        T_dict = None,
        num_numerical_features = 0,
        disbalance = None,
        device = torch.device('cuda:1'),
        seed = 0,
        change_val = False
    ):
        """
        Uses the diffusion model parameters to generate samples.
        Adpoted from sample() function in scripts/sample.py.
        
        """
        print(f"In Actor.generate_samples(), generating {num_samples} samples.")
        # D -> self.D where self = Actor    
        _, empirical_class_dist = torch.unique(torch.from_numpy(self.D.y['train']), return_counts=True)
        # empirical_class_dist = empirical_class_dist.float() + torch.tensor([-5000., 10000.]).float()
        if disbalance == 'fix':
            empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], empirical_class_dist[0]
            #  self.diffusion where self = Actor
            # ddim=True, diffusion ->require_grad=True because samples have impact on loss backpropagation
            x_gen, y_gen = self.diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=True, require_grad=True)

        elif disbalance == 'fill':
            ix_major = empirical_class_dist.argmax().item()
            val_major = empirical_class_dist[ix_major].item()
            x_gen, y_gen = [], []
            for i in range(empirical_class_dist.shape[0]):
                if i == ix_major:
                    continue
                distrib = torch.zeros_like(empirical_class_dist)
                distrib[i] = 1
                num_samples = val_major - empirical_class_dist[i].item()
                #  self.diffusion where self = Actor
                # ddim=True, diffusion ->require_grad=True because samples have impact on loss backpropagation
                x_temp, y_temp = self.diffusion.sample_all(num_samples, batch_size, distrib.float(), ddim=True, require_grad=True)
                x_gen.append(x_temp)
                y_gen.append(y_temp)
            
            x_gen = torch.cat(x_gen, dim=0)
            y_gen = torch.cat(y_gen, dim=0)

        else:
            #  self.diffusion where self = Actor
            # ddim=True, diffusion ->require_grad=True because samples have impact on loss backpropagation
            x_gen, y_gen = self.diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=True, require_grad=True)

        
        return x_gen, y_gen
        

    def update(self, loss_tddpm, advantage):
        # Train the actor network for one step, called by ActorCritic.update() in run_algorithm()
        self.optimizer()
        loss = loss_tddpm + advantage
        loss.backward()
        self.optimizer.step()
        
    def adopt_lr(self):
        return self._anneal_lr(self.step)


class CriticNet(nn.Module):
    def __init__(self, num_columns, hidden_dim=16):
        super().__init__()

        self.hidden = nn.Linear(num_columns, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    # todo: how to take four input item to compute a single scalar for privacy score?
    def forward(self, x, out_dict, x_gen, y_gen):
        outs = self.hidden(x_gen)
        outs = F.relu(outs)
        value = self.output(outs)
        return value
    

class Critic:
    """
    A Critic network that evaluates the value of states with 2000 (default) samples.
    Uses a Fully Connected Neural Network architecture to learn a privacy value function.
    States are implied by the samples generated so far.
    
    """
    def __init__(self, raw_config, args, device, num_columns):
        self.raw_config = raw_config
        self.args = args
        self.device = device
        self.num_columns = num_columns
        
        self.model = CriticNet(self.num_columns).to(self.device)
        self.optimizer = torch.optim.AdamW(self.models.parameters(), lr=0.001)
        
    def q_value(self, x, out_dict, x_gen, y_gen):
        """
        Computes the 'privacy reward' of x_gen, y_gen.
        Not implemented yet.
        How to operate with x_gen, y_gen such that they return a single scalar feedback?
        
        """
        self.model(x, out_dict, x_gen, y_gen)
    
    def get_optimizer(self):
        # Return the optimizer for the critic network
        return self.optimizer
    
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()  


class ActorCriticAgent:
    """
    The Actor-Critic RL agent for training TabDDPM with privacy-aware rewards.
    Actor network uses TabDDPM to sample synthetic data.
    Critic network evaluates the privacy value of the current state, implemented with a Fully Connected Neural Network.
    Continuous States: states are represented by the synthetic data sampled so far.
    
    :var Args: Description
    :var Returns: Description
    :var Returns: Description
    :var float: Description
    :vartype float: The
    """
    
    def __init__(self, name="", steps_per_round=1000, evaluation_samples=2000, args=None, raw_config=None, device=None, rounds=None, pretrain_steps=None):
        
        self.name = name
        self.steps_per_round = steps_per_round
        self.evaluation_samples = evaluation_samples
        self.args = args
        self.raw_config = raw_config
        self.device = device
        self.rounds = rounds - 1 if rounds else int(self.raw_config['train']['main']['steps'] / self.steps_per_round) - 1
        self.pretrain_steps = pretrain_steps if pretrain_steps else steps_per_round
        self.step = pretrain_steps
        self.total_steps = self.raw_config['train']['main']['steps']
        info_path = os.path.join(os.path.normpath(self.raw_config['parent_dir']), 'info.json')  
        self.info = lib.load_json(info_path)      
        self.train_args = self.train_dictionary()
        
        self.actor = self.build_actor(**self.train_args)
        self.critic = self.build_critic()
        self.optimizer_actor = self.actor.get_optimizer()
        self.optimizer_critic = self.critic.get_optimizer()
        print(f"ActorCriticAgent, optimizer_actor: {self.optimizer_actor},\noptimizer_critic: {self.optimizer_critic}")
        
        # self.state, self.next_state represent the synthetic data sampled by the actor at current and next step
        self.state = None
        self.next_state = None
        
        
        # Load real data for initial training and evaluation
        self.mm, self.ohe, self.X_num_real, self.X_cat_real = None, None, None, None    # X_cat_real is the onehot encoded categorical features of real data
        # Load real data with currently fitted mm and ohe if available
        self.real_data, self.target_size, self.dataset_info = self.load_real_data(self.raw_config['real_data_path'])  # loads
        self.real_data = np.asarray(self.real_data)
        
        

        self.dataset_info.update(lib.load_json(os.path.join(self.raw_config['parent_dir'], 'DatasetInfo.json')))
        self.dataset_info['category_sizes'].append(self.target_size)
        print(f"Actor-Critic agent {self.name} initialized to train for dataset {self.dataset_info['name']} with {self.rounds} rounds, each with {self.steps_per_round} steps.\nUse privacy metric: {self.privacy_metric}.")
        self.category_sizes = self.dataset_info.get('category_sizes')
        self.task_type = self.dataset_info.get('task_type')
    
    def train_dictionary(self):
        return {
            "steps": self.pretrain_steps,
            "start_privacy_step": -1,
            "lr": self.raw_config['train']['main']['lr'],
            "weight_decay": self.raw_config['train']['main']['weight_decay'],
            "batch_size": self.raw_config['train']['main']['batch_size'],
            "parent_dir": self.raw_config['parent_dir'],
            "real_data_path": self.raw_config['real_data_path'],
            "model_type": self.raw_config['model_type'],
            "model_params": self.raw_config['model_params'],
            "T_dict": self.raw_config['train']['T'],
            "num_numerical_features": self.raw_config['num_numerical_features'],
            "device": self.device,
            "change_val": self.args.change_val,
            "continue_training": False,
            "total_steps": self.raw_config['train']['main']['steps'],
            **self.raw_config['diffusion_params']
        }
        
    def sample_dictionary(self):
        return {
            "num_samples": 2000,
            "batch_size": self.raw_config["sample"]["batch_size"],
            "disbalance": self.raw_config["sample"].get("disbalance", None),

            # unpacked diffusion params
            **self.raw_config["diffusion_params"],

            "parent_dir": self.raw_config["parent_dir"],
            "real_data_path": self.raw_config["real_data_path"],
            "model_path": os.path.join(self.raw_config["parent_dir"], "model.pt"),
            "model_type": self.raw_config["model_type"],
            "model_params": self.raw_config["model_params"],
            "T_dict": self.raw_config["train"]["T"],
            "num_numerical_features": self.raw_config["num_numerical_features"],
            "device": self.device,
            "seed": self.raw_config["sample"].get("seed", 0),
            "change_val": self.args.change_val,
        }
            
    def pretrain_diffusion_model(self):
        print(f"Pretraining actor for one round, {self.steps_per_round} steps.")
        # Pretrain the actor network
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
                    total_steps=self.raw_config['train']['main']['steps']
                )
        return train_result
        

    def build_actor(
        self,
        parent_dir,
        real_data_path = 'data/higgs-small',
        steps = 1000,
        start_privacy_step = -1,
        lr = 0.002,
        weight_decay = 1e-4,
        batch_size = 1024,
        model_type = 'mlp',
        model_params = None,
        num_timesteps = 1000,
        gaussian_loss_type = 'mse',
        scheduler = 'cosine',
        T_dict = None,
        num_numerical_features = 0,
        device = torch.device('cuda:1'),
        seed = 0,
        change_val = False,
        continue_training = False,
        privacy_metric = 'nndr',
        weight_mask = None,
        completed_steps = 0,
        total_steps = None
    ):
        """
        First step, train the diffusion model for steps_per_round steps.
        Next, do everything in train() in scripts/train.py before calling trainer.run_loop().
        Since Actor is pretrained on initialization, the code follows the "continue_training=True" branch.
        
        
        """
        pretraining_result = self.pretrain_diffusion_model()
        
        real_data_path = os.path.normpath(real_data_path)
        parent_dir = os.path.normpath(parent_dir)

        zero.improve_reproducibility(seed)

        T = lib.Transformations(**T_dict)

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

        num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
        d_in = np.sum(K) + num_numerical_features
        model_params['d_in'] = d_in
        print(d_in)
        
        print(model_params)
        print(f"completed_steps: {completed_steps}")
        model = get_model(
            model_type,
            model_params,
            num_numerical_features,
            category_sizes=dataset.get_category_sizes('train')
        )
        # ema_model = deepcopy(model)  # moved out from Trainer class
        # if continue_training:  # different than in train.py, here the model is pretrained on initialization
        model.load_state_dict(torch.load(os.path.join(parent_dir, 'model.pt')))
        # ema_model.load_state_dict(torch.load(os.path.join(parent_dir, 'model_ema.pt')))
            
        model.to(device)
        # ema_model.to(device)

        # train_loader = lib.prepare_beton_loader(dataset, split='train', batch_size=batch_size)
        train_loader = lib.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)

        diffusion = GaussianMultinomialDiffusion(
            num_classes=K,
            num_numerical_features=num_numerical_features,
            denoise_fn=model,
            gaussian_loss_type=gaussian_loss_type,
            num_timesteps=num_timesteps,
            scheduler=scheduler,
            device=device
        )
        diffusion.to(device)
        diffusion.train()
        
        optimizer = None  # Use default AdamW optimizer in Trainer class
        # if continue_training:  # different than in train.py, here the model is pretrained on initialization
        optimizer = torch.optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_state = torch.load(os.path.join(parent_dir, 'optimizer.pt'))
        optimizer.load_state_dict(optimizer_state)
        move_optimizer_to_device(optimizer, device)

        # trainer = Trainer(
        actor = Actor(
            diffusion,
            train_loader,
            lr=lr,
            weight_decay=weight_decay,
            steps=steps,
            device=device,
            optimizer=optimizer,
            total_steps=total_steps
        )
        actor.add_training_result(pretraining_result)
        actor.add_metadata(D=dataset)

        return actor

    def build_critic(self):
        # Build the critic network based on the configuration
        
        num_columns = int(self.info["n_num_features"]) + int(self.info["n_cat_features"])
        print(f"In ActorCriticAgent.build_critic, CriticNet expects input with {num_columns} columns.")
        critic = Critic(config=self.raw_config, args=self.args, device=self.device, num_column=num_columns)
        return critic

    def update(self, advantage):
        # Update actor and critic networks based on stored transitions
        self.actor.update(advantage)  # propagate advantage back to train.py as loss term to backpropagate
        self.critic.update(advantage)  
        # Use advantage to update actor
        pass
    
    def evaluate_performance(self):
        # Evaluate the performance of the current policy
        pass
    
    def evaluate_generation(self):
        # Evaluate final training result
        # Take RLAgent.evaluate_generation as template
        pass

    def compute_convergence(self) -> bool:
        # If losses have "converged", return True
        pass

    # todo: save critic network
    def save_model(self, parent_dir: str):
        self.actor.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
        torch.save(self.actor.diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
        torch.save(self.actor.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))
        torch.save(self.actor.get_optimizer().state_dict(), os.path.join(parent_dir, 'optimizer.pt')) 
        # torch.save(self.critic...)
        torch.save(self.critic.get_optimizer().state_dict(), os.path.join(parent_dir, 'critic_optimizer.pt'))
    
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
        
        print(f"in load_real_data, self.mm: {self.mm}")
        mm = self.mm if self.mm else MinMaxScaler().fit(X_num_real)
        
        X_real = mm.transform(X_num_real)
        X_real = np.clip(X_real, 0, 1)
        has_negative = (X_real < 0).any()
        print(f"in load_real_data, real data transformed with mm has negative: {has_negative}")
        has_larger = (X_real > 1).any()
        print(f"in load_real_data, real data transformed with mm has larger than 1: {has_larger}")

        if X_cat_real is not None:
            if self.ohe is None:  # First time loading real data
                self.ohe = OneHotEncoder().fit(X_cat_real)  # fit and save, use throughout training
                self.X_cat_real = self.ohe.transform(X_cat_real) / np.sqrt(2)
                self.X_cat_real = self.X_cat_real.todense()
            X_real = np.concatenate([X_real, self.X_cat_real], axis=1)
        
        return X_real, target_size, dataset_info
    
    
    def load_fake_data(self, X_num_fake, X_cat_fake, y_gen):
        """
        Adopted from tab-ddpm/scripts/resample_privacy.py, which inturns is adapted from https://github.com/Team-TUD/CTAB-GAN/tree/main/model/eval
        
        """
        st = time.time()
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

        combined_num = np.vstack((self.X_num_real, X_num_fake))
        self.mm = MinMaxScaler().fit(combined_num)
        # self.mm = MinMaxScaler().fit(X_num_fake)
        X_fake = self.mm.transform(X_num_fake)
        has_larger = (X_fake > 1).any()
        has_negative = (X_fake < 0).any()
        print(f"in load_fake_data, X_fake has elements > 1: {has_larger}.")
        print(f"In rlagent.py, load_fake_data(), X_fake has negative: {has_negative}")
        if has_larger or has_negative:
            raise NotImplementedError("X_fake has elements larger than 1 or negative after MinMaxScaler transform before clipping.")
        # X_fake = np.clip(X_fake, 0, 1)
        if (X_cat_fake is not None) and (self.ohe is not None):
            X_cat_fake = self.ohe.transform(X_cat_fake) / np.sqrt(2)
            X_fake = np.concatenate([X_fake, X_cat_fake.todense()], axis=1)
        
        has_negative = (X_fake < 0).any()
        print(f"In rlagent.py, load_fake_data(), X_fake that is fitted with X_num_fake has negative: {has_negative}")
        
            
        # Everytime we call call load_fake_data, we also reload real data to avoid data leakage due to fitted mm and ohe    
        self.real_data, self.target_size, self.dataset_info = self.load_real_data(self.raw_config['real_data_path'])  # loads
        self.real_data = np.asarray(self.real_data)
        
        has_negative = (self.real_data < 0).any()
        print(f"In rlagent.py, load_fake_data89, self.real_data transformed with X_num_fake has negative: {has_negative}")
        
        
        et = time.time()
        print(f"load_fake_data time: {et-st}s")
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
    
        
    def run_algorithm(self):
        for round in range(self.rounds):
            
            print(f"Starting round {round + 1}/{self.rounds}")
            for step in range(self.steps_per_round):  # Too much overhead
                
                # sample from model and return the reward and next state
                # actor
                # while step < self.steps:
                x, out_dict = next(self.actor.train_iter)
                out_dict = {'y': out_dict}
                
                
                # Execute action, loss_tddpm as reward, run_step does not call loss.backward() yet
                batch_loss_multi, batch_loss_gauss, loss_tddpm = self.actor.run_step(x, out_dict)
                
                # Compute critic evaluation, the closer to 0 the better
                T = self.sample_dictionary()
                synthetic_x, synthetic_y = self.actor.generate_samples(**T)  # 2000 samples, fixed in self.sample_dictionary
                critic_feedback = self.critic.q_value(x, out_dict, synthetic_x, synthetic_y)
                
                # Combined loss term of actor and critic output
                advantage = loss_tddpm + critic_feedback
                
                # turn _anneal_lr to public method                
                self.actor.adopt_lr(self.step)

                # Display TabDDPM training loss
                curr_count += len(x)
                curr_loss_multi += batch_loss_multi.item() * len(x)
                curr_loss_gauss += batch_loss_gauss.item() * len(x)

                if (self.step + 1) % self.actor.log_every == 0:
                    mloss = np.around(curr_loss_multi / curr_count, 4)
                    gloss = np.around(curr_loss_gauss / curr_count, 4)
                    if (self.step + 1) % self.actor.print_every == 0:
                        print(f'Step {(self.step + 1)}/{self.total_steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                    self.loss_history.loc[len(self.loss_history)] =[self.step + 1, "", mloss, gloss, 0, 0, mloss + gloss]
                    curr_count = 0
                    curr_loss_gauss = 0.0
                    curr_loss_multi = 0.0

                update_ema(self.actor.ema_model.parameters(), self.actor.diffusion._denoise_fn.parameters())
                self.step += 1
                
                # Update actor and critic networks
                self.actor.update(advantage)
                self.critic.update(advantage)
                
            # Each round of interaction consists of multiple steps
            self.completed_steps += self.steps_per_round
            if self.step != self.completed_steps:
                print(f"ActorCriticAgent.run_algorithm: After {round+1} rounds, self.step = {self.step}, self.completed_step = {self.completed_steps}")
                
            # Evaluate performance at the end of each round
            self.evaluate_performance()
            
            if self.compute_convergence():
                print(f"Convergence reached at round {round + 1}. Stopping training.")
                break
        self.evaluate_generation()  # Generate final evaluation into eval.txt, from evaluate_privacy.py
        self.save_model()
