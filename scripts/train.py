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
from datasetinfo import DatasetInfo, generate_dataset_info

PRIVACY_METRIC = ["dcr", "nndr", "gower", "vector", "adaptive", "sum", "adaptive_dcr", "adaptive_nndr", "adaptive_gower"]
GOWER_STD = 0.2
NOISE_STD = 0.0001  # Standard deviation of noise added to EMA model parameters
LOSS_HISTORY_COLUMNS = ['step', 'privacy_loss_type', 'mloss', 'gloss', 'mprivacy', 'gprivacy', 'loss']

def is_privacy_vector(privacy_metric):
    return privacy_metric in ['vector', 'adaptive', 'adaptive_dcr', 'adaptive_nndr', 'adaptive_gower']



class Trainer:
    def __init__(self, diffusion,  train_iter, lr, weight_decay, steps, device=torch.device('cuda:1'), optimizer=None, total_steps=None, max_norm=1.0):
        self.diffusion = diffusion
        # self.ema_model = ema_model 
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter  # called with train_loader in train()
        self.steps = steps
        self.total_steps = total_steps if total_steps is not None else steps
        self.init_lr = lr
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optimizer
        self.device = device
        self.loss_history = pd.DataFrame(columns=LOSS_HISTORY_COLUMNS)
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000
        
        self.gradient_history = []
        self.max_norm = max_norm  # Max norm for gradient clipping

    def _anneal_lr(self, step):
        """
        Paul:
        Linearly anneal the learning rate from the initial lr to 0.
        self.total_steps is set during Trainer initialization.
        self.total_steps represent total number of training steps during training, i.e. rounds * steps_per_round
        
        """
        frac_done = step / self.total_steps  # Paul modified from self.steps to self.total_steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            
    # ------------- Depricated approach: multiply loss by (1 + privacy_ratio) --------------
    def _run_step_privacy_dcr_multiply(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        """
        DCR privacy terms are bounded in range [0, 1], implemented in privacy.py,
        where 0 means identity and 1 means perfect mismatch. Aim for privacy_loss = 1.
        """
        loss_multi = loss_multi * (2 - privacy_multi) if loss_multi != 0 else loss_multi  # aim for privacy ratio = 1
        loss_gauss = loss_gauss * (2 - privacy_gauss)  # aim for privacy ratio = 1        

        loss = loss_multi + loss_gauss
        return loss
    
    def _run_step_privacy_nndr_multiply(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        loss_multi = loss_multi * (2 - privacy_multi) if loss_multi != 0 else loss_multi  # aim for privacy ratio = 1
        loss_gauss = loss_gauss * (2 - privacy_gauss)  # aim for privacy ratio = 1
            
        loss = loss_multi + loss_gauss  #aim for privacy ratio = 1
        return loss
    
    def _run_step_privacy_gower_multiply(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        loss_multi = loss_multi * (2 - privacy_multi) if loss_multi != 0 else loss_multi  # aim for privacy ratio = 1
        loss_gauss = loss_gauss * (2 - privacy_gauss)  # aim for privacy ratio = 1
        
        loss = (loss_multi + loss_gauss) * torch.normal(mean=1.0, std=GOWER_STD, size=(1,), device=self.device)  #aim for privacy ratio = 1
        return loss
    
    
    def _run_step_privacy_vector_multiply(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        target_vec = torch.ones((3,), device=self.device) * 2
        # print(f"In run_step_privacy_vector, target_vec: {target_vec}")
        # raise NotImplementedError("Vector privacy loss not fully tested yet, use with caution.")
        
        dist_gauss = (target_vec - privacy_gauss)
        dist_multi = (target_vec - privacy_multi)
        
        for i in privacy_gauss:
            if i > 1:
                raise ValueError("Privacy gauss higher equal than 1")
        for i in privacy_multi:            
            if i > 1:
                raise ValueError(f"Privacy multi higher than 1, privacy_multi:\n {privacy_multi}")
        
        lg = dist_gauss * loss_gauss
        lm = dist_multi * loss_multi
        print(f"Vector privacy loss components, 2 - privacy_gauss: {dist_gauss}, 2 - privacy_multi: {dist_multi}")
        vec = (lg + lm)
        
        return vec
    
    # ------------- end of depricated approach --------------

    def _run_step(self, x, out_dict):
        """
        One step of learning: comute loss, update weights.
        Original implementation, no privacy loss term

        Args:
            x (_type_): input data
            out_dict (_type_): yield from FastTensorDataLoader, in lib/data.py
        Returns:
            _type_: _description_
        """
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        
        loss_multi, loss_gauss, privacy_multi, privacy_gauss = self.diffusion.mixed_loss(x, out_dict)
        
        loss = loss_multi + loss_gauss
        loss.backward()
        
        total_norm = torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), max_norm=float('inf'))        
        self.gradient_history.append(total_norm.item())
        self.optimizer.step()

        return loss_multi, loss_gauss
    
    
    def _run_step_privacy_dcr(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        """
        DCR privacy terms are bounded in [0, 1] due to MinMaxScaler normalization, set target to 1 for privacy and fidelity trade-off.
        Implemented in privacy.py, where 0 means identity and large value means perfect mismatch. Aim for privacy_loss = infinity.
        """
        loss_multi = loss_multi +  torch.exp(-(self.diffusion.dcr_weight * privacy_multi)) if loss_multi != 0 else loss_multi  # aim for privacy ratio = 1
        loss_gauss = loss_gauss +  torch.exp(-(0.1 * self.diffusion.dcr_weight * privacy_gauss))  # aim for privacy ratio = 1        

        loss = loss_multi + loss_gauss
        return loss
      
    
    def _run_step_privacy_nndr(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        loss_multi = loss_multi + (1 - self.diffusion.nndr_weight * privacy_multi) if loss_multi != 0 else loss_multi  # aim for privacy ratio = 1
        loss_gauss = loss_gauss + (1 - self.diffusion.nndr_weight * privacy_gauss)  # aim for privacy ratio = 1
            
        loss = loss_multi + loss_gauss  #aim for privacy ratio = 1
        return loss
    
    def _run_step_privacy_gower(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        loss_multi = loss_multi + (1 - self.diffusion.gower_weight * privacy_multi) if loss_multi != 0 else loss_multi  # aim for privacy ratio = 1
        loss_gauss = loss_gauss + (1 - self.diffusion.gower_weight * privacy_gauss)  # aim for privacy ratio = 1
        
        # without noise addition
        loss = loss_multi + loss_gauss
        
        # Combine losses and add Gaussian noise for less memorizing
        # loss = (loss_multi + loss_gauss) * torch.normal(mean=1.0, std=GOWER_STD, size=(1,), device=self.device)  #aim for privacy ratio = 1
        return loss
    
    def _run_step_privacy_sum(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        # raise NotImplementedError("Sum privacy loss is deprecated, since compute_DCR is not bounded anymore, using 3 as target value is not valid.")
        loss_multi = loss_multi + (2 - self.diffusion.sum_weight * privacy_multi) if loss_multi != 0 else loss_multi  # aim for privacy ratio = 1
        loss_gauss = loss_gauss + (2 - self.diffusion.sum_weight * privacy_gauss)  # aim for privacy ratio = 1
        
        loss = loss_multi + loss_gauss  #aim for privacy ratio = 1
        return loss
    
    def _run_step_privacy_vector(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        target_vec = torch.ones((3,), device=self.device)
        # print(f"In run_step_privacy_vector, target_vec: {target_vec}")
        # raise NotImplementedError("Vector privacy loss not fully tested yet, use with caution.")
        
        # privacy_gauss, privacy_multi are vectors of size 3
        # print(f"Adaptive weight for privacy losses: {self.adaptive_weight}")
        weighted_privacy_gauss = self.adaptive_weight * privacy_gauss
        weighted_privacy_multi = self.adaptive_weight * privacy_multi
        
        # 3. Apply the specific formulas
        # Formula for index 0: exp(-x)
        # Formula for indices 1 & 2: 1 - x
        def apply_transform(weighted_tensor):
            # We create a copy to avoid modifying the intermediate weighted_tensor
            res = torch.empty_like(weighted_tensor)
            
            # DCR (Index 0)
            res[0] = torch.exp(-weighted_tensor[0])
            
            # NNDR & Gower (Indices 1 and 2)
            res[1:] = 1 - weighted_tensor[1:]
            
            return res

        # 4. Generate your two final tensors
        dist_gauss = apply_transform(weighted_privacy_gauss)
        dist_multi = apply_transform(weighted_privacy_multi)

        
        """
        for i in privacy_gauss:
            if i > 1 or i < 0:
                raise ValueError("Privacy gauss out of range")
        for i in privacy_multi:            
            if i > 1 or i < 0:
                raise ValueError(f"Privacy multi out of range, privacy_multi:\n {privacy_multi}")
        
        """
        lg = dist_gauss.sum() + loss_gauss
        lm = dist_multi.sum() + loss_multi
        # print(f"Vector privacy loss components, 1 - privacy_gauss: {dist_gauss}, 1 - privacy_multi: {dist_multi}")
        vec = (lg + lm)
        
        return vec
    
    def _run_step_privacy_adaptive_dcr(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        """
        DCR privacy terms are bounded in [0, 1] due to MinMaxScaler normalization, set target to 1 for privacy and fidelity trade-off.
        Implemented in privacy.py, where 0 means identity and large value means perfect mismatch. Aim for privacy_loss = infinity.
        """        
        loss_multi = loss_multi + torch.exp(-self.adaptive_weight[0] * privacy_multi) if loss_multi != 0 else loss_multi  # aim for largest privacy value 
        loss_gauss = loss_gauss + torch.exp(-self.adaptive_weight[0] * privacy_gauss)  # aim for privacy ratio = 1        

        loss = loss_multi + loss_gauss
        return loss


    def _run_step_privacy_adaptive_nndr(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        """
        
        """        
        loss_multi = loss_multi + (1 - self.adaptive_weight[1] * privacy_multi) if loss_multi != 0 else loss_multi  # aim for largest privacy value 
        loss_gauss = loss_gauss + (1 - self.adaptive_weight[1] * privacy_gauss)  # aim for privacy ratio = 1        

        loss = loss_multi + loss_gauss
        return loss
    
    def _run_step_privacy_adaptive_gower(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        """
        
        """        
        loss_multi = loss_multi + (1 - self.adaptive_weight[2] * privacy_multi) if loss_multi != 0 else loss_multi  # aim for largest privacy value 
        loss_gauss = loss_gauss + (1 - self.adaptive_weight[2] * privacy_gauss)  # aim for privacy ratio = 1        

        loss = loss_multi + loss_gauss
        return loss  
    
    
    def _run_step_privacy_adaptive(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        """ Adopted from _run_step_privacy_vector. """
        # target_vec = torch.ones((3,), device=self.device)
        # print(f"In run_step_privacy_vector, target_vec: {target_vec}")
        # raise NotImplementedError("Vector privacy loss not fully tested yet, use with caution.")
        
        # privacy_gauss, privacy_multi are vectors of size 3
        # print(f"Adaptive weight for privacy losses: {self.adaptive_weight}")
        weighted_privacy_gauss = self.adaptive_weight * privacy_gauss
        weighted_privacy_multi = self.adaptive_weight * privacy_multi
        
        # 3. Apply the specific formulas
        # Formula for index 0: exp(-x)
        # Formula for indices 1 & 2: 1 - x
        def apply_target(weighted_tensor):
            # We create a copy to avoid modifying the intermediate weighted_tensor
            res = torch.empty_like(weighted_tensor)
            
            # DCR (Index 0)
            res[0] = torch.exp(-weighted_tensor[0])
            
            # NNDR & Gower (Indices 1 and 2)
            res[1:] = 1 - weighted_tensor[1:]
            
            return res

        # 4. Generate your two final tensors
        dist_gauss = apply_target(weighted_privacy_gauss)
        dist_multi = apply_target(weighted_privacy_multi)

        lg = dist_gauss.sum() + loss_gauss
        lm = dist_multi.sum() + loss_multi
        # print(f"Vector privacy loss components, 1 - privacy_gauss: {dist_gauss}, 1 - privacy_multi: {dist_multi}"
        vec = (lg + lm)
        
        return vec
        
    
    def _run_step_privacy(self, x, out_dict, privacy_metric='dcr', loss_memory=None, weight_mask=None):
        """
        Incorporates privacy term to loss functions.\n
        Use paramter privacy_metric to call loss function\n
        Loss function = (1 + ratio) * loss, aim for ratio = 1 so minimize 1-ratio\n

        Args:
            x (_type_): _description_
            out_dict (_type_): yield from FastTensorDataLoader, in lib/data.py
            weight_mask (_type_, optional): torch tensor for vector and sum privacy approach. Defaults to None.
        Returns:
            _type_: _description_
        """
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        
        # print(f"Add noise to model weights before each training step...")
        # self._inject_ema_noise(privacy_metric)
        
        self.optimizer.zero_grad()
        
        # Depending on privacy metric, modify loss functions
        if privacy_metric not in PRIVACY_METRIC:
            raise ValueError(f"Unknown privacy metric: {privacy_metric}. Supported metrics: {PRIVACY_METRIC}")
        
        # mixed_loss() in gaussian_multinomial_diffusion.py
        # privacy_multi: Normalized distance ratio for categorical part [0,1]
        # privacy_gauss: NNDR ratio
        loss_multi, loss_gauss, privacy_multi, privacy_gauss = self.diffusion.mixed_loss(x, out_dict, privacy_metric=privacy_metric, loss_memory=loss_memory, weight_mask=weight_mask)
        
        # adaptive, vector, adaptive with single metric       
        if is_privacy_vector(privacy_metric):
            self.adaptive_weight = weight_mask
            
        privacy_method = getattr(self, "_run_step_privacy_"+privacy_metric)
        loss = privacy_method(loss_multi, loss_gauss, privacy_multi, privacy_gauss)
        
        loss.backward()        

        """    
        # Apply weight mask for vector privacy approach, depricated
        if privacy_metric == 'vector':
            mask = weight_mask if weight_mask is not None else torch.ones_like(loss)
            loss.backward(mask)
        else:
            loss.backward()
        """
        
        # Clip gradients to avoid exploding gradients
        clipped_norm = torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), max_norm=self.max_norm)    
        self.gradient_history.append(clipped_norm.item())
            
        self.optimizer.step()

        return loss_multi, loss_gauss, privacy_multi, privacy_gauss, loss
    
    
    # _inject_ema_noise adds small random noise to the EMA model parameters to enhance privacy.
    # Not used in implementation 17.12.2025, observing high fluctuating DCR losses during training.
    def _inject_ema_noise(self, privacy_metric):
        # Initial noise injection to EMA model
        if privacy_metric == 'gower' or privacy_metric == 'vector':
            noise_dist = Normal(loc=0.0, scale=NOISE_STD)
            print(f"Adding randomness to model weights, noise std: {NOISE_STD}")

            # Iterate over all parameters (weights and biases) in the model
            with torch.no_grad(): # Ensure this operation is outside the gradient calculation
                for param in self.ema_model.parameters():
                    # Only apply to parameters that are meant to be trained
                    if param.requires_grad:
                        # Sample noise tensor with the exact shape of the parameter
                        noise = noise_dist.sample(param.shape).to(self.device)
                        param.add_(noise)

    def run_loop(self, start_privacy_step, privacy_metric, completed_steps, loss_memory=None, weight_mask=None, privacy_term_weights=None):
        """
        Training loop. If start_privacy_step >= 0, use run_step_privacy starting from it.
        Args:
            start_privacy_step (int): Step to start incorporating privacy loss term. If <0, never use privacy term.
            privacy_metric (str): Privacy metric to use. One of PRIVACY_METRIC.
            completed_steps (int): Number of steps already completed in previous rounds. Comes from agent, i.e. train() function.
            loss_memory (torch.Tensor, optional): Tensors to store loss values for privacy computation. Defaults to None if not vector approach.
            weight_mask (torch.Tensor, optional): Weight mask for vector privacy approach. Defaults to None.
        """
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_privacy_multi = 0.0
        curr_privacy_gauss = 0.0
        curr_total_loss = 0.0
        
        # This is old vector approach: keep track of privacy losses as vectors
        # New approach: keep track of privacy losses as scalars, simply sum all loss terms, no need for applying weights
        """
        if privacy_metric == 'vector':
            curr_privacy_multi = torch.zeros(3, device=self.device)
            curr_privacy_gauss = torch.zeros(3, device=self.device)
            curr_total_loss = torch.zeros(3, device=self.device)
        """
        
        assert start_privacy_step < self.steps, "In Trainer.run_loop(), start_privacy_step must be less than total steps. Check parameters passed to train() and the config.toml file."
        if privacy_metric == 'sum':
            assert privacy_term_weights is not None, "For 'sum' privacy metric, privacy_term_weights must be provided."
        
        curr_count = 0
        if start_privacy_step < 0:
            print("Never use privacy term during training.")
            # Never incorporate privacy term
            while step < self.steps:
                x, out_dict = next(self.train_iter)
                out_dict = {'y': out_dict}
                
                # Never use privacy term
                batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

                self._anneal_lr(step + completed_steps)

                curr_count += len(x)
                curr_loss_multi += batch_loss_multi.item() * len(x)
                curr_loss_gauss += batch_loss_gauss.item() * len(x)

                if (step + 1) % self.log_every == 0:
                    mloss = np.around(curr_loss_multi / curr_count, 4)
                    gloss = np.around(curr_loss_gauss / curr_count, 4)
                    if (step + 1) % self.print_every == 0:
                        print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                    self.loss_history.loc[len(self.loss_history)] =[step + 1, "", mloss, gloss, 0, 0, mloss + gloss]
                    curr_count = 0
                    curr_loss_gauss = 0.0
                    curr_loss_multi = 0.0

                update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())
                step += 1
            
            # Gradient clipping
            new_max_norm = torch.quantile(torch.tensor(self.gradient_history), q=0.95).item()
            print(f"Without privacy, new max_norm: {new_max_norm}, old max_norm: {self.max_norm}")
            
            
            # less_norm approach, doesnt force gradient to go down when not using privacy term
            self.max_norm = new_max_norm
            
            """
            # opposite of less_norm approach, force gradients to be smaller as training goes on
            if new_max_norm > self.max_norm:
                print(f"Update new max_norm with 0.95 * old max_norm")
                self.max_norm *= 0.95
            else:
                self.max_norm = new_max_norm
            """
                
        else:  # run loop with privacy term    
            print(f"Incorporate privacy term after step {start_privacy_step}.")
            
            while step < self.steps:
                x, out_dict = next(self.train_iter)
                out_dict = {'y': out_dict}
                pm = privacy_metric
                
                # Incorporates privacy term after certain training step
                if step < start_privacy_step:
                    batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)
                    pm = ""
                else:
                    batch_loss_multi, batch_loss_gauss, batch_privacy_multi, batch_privacy_gauss, batch_total_loss = self._run_step_privacy(x, out_dict, privacy_metric=privacy_metric, loss_memory=loss_memory, weight_mask=weight_mask)

                # Anneal learning rate, make sure learning rate is adapted to current global training step
                self._anneal_lr(step + completed_steps)

                curr_count += len(x)
                curr_loss_multi += batch_loss_multi.item() * len(x)
                curr_loss_gauss += batch_loss_gauss.item() * len(x)
                curr_privacy_multi += batch_privacy_multi.item() * len(x) if not is_privacy_vector(privacy_metric) else batch_privacy_multi * len(x)
                curr_privacy_gauss += batch_privacy_gauss.item() * len(x) if not is_privacy_vector(privacy_metric) else batch_privacy_gauss * len(x)
                curr_total_loss += batch_total_loss.item() * len(x) if not is_privacy_vector(privacy_metric) else batch_total_loss * len(x)

                if (step + 1) % self.log_every == 0:

                    mloss = np.around(curr_loss_multi / curr_count, 4)
                    gloss = np.around(curr_loss_gauss / curr_count, 4)
                    mprivacy = np.around(curr_privacy_multi / curr_count, 4) if not is_privacy_vector(privacy_metric) else np.around(curr_privacy_multi.detach().cpu().numpy() / curr_count, 4)
                    gprivacy = np.around(curr_privacy_gauss/ curr_count, 4) if not is_privacy_vector(privacy_metric) else np.around(curr_privacy_gauss.detach().cpu().numpy()/ curr_count, 4)
                    total_loss = np.around(curr_total_loss / curr_count, 4) if not is_privacy_vector(privacy_metric) else np.around(curr_total_loss.detach().cpu().numpy() / curr_count, 4)
                    
                    
                    if (step + 1) % self.print_every == 0:
                        print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} MPrivacy: {mprivacy} GPrivacy: {gprivacy} Sum: {total_loss}')
                    self.loss_history.loc[len(self.loss_history)] =[step + 1, pm, mloss, gloss, mprivacy, gprivacy, total_loss]
                    curr_count = 0
                    curr_loss_gauss = 0.0
                    curr_loss_multi = 0.0
                    
                    # New approach: reset scalar losses, 31.12.2025
                    curr_privacy_gauss = 0.0
                    curr_privacy_multi = 0.0
                    curr_total_loss = 0.0
                    
                    # Old vector approach: reset vector losses
                    """
                    if privacy_metric == 'vector':
                        curr_privacy_multi.zero_()
                        curr_privacy_gauss.zero_()
                        curr_total_loss.zero_()
                    else:
                        curr_privacy_gauss = 0.0
                        curr_privacy_multi = 0.0
                        curr_total_loss = 0.0
                    """

                update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())
                step += 1
            
            # Gradient Clipping
            print(f"With privacy, 95 percentile gradient history: {torch.quantile(torch.tensor(self.gradient_history), q=0.95).item()}")  
            self.max_norm *= 0.95      
            print(f"Update new max_norm with 0.95 * old max_norm")
              
        

def move_optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def train(
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
    if continue_training:
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
    max_norm = 1.0
    if continue_training:
        optimizer = torch.optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_state = torch.load(os.path.join(parent_dir, 'optimizer.pt'))
        optimizer.load_state_dict(optimizer_state)
        move_optimizer_to_device(optimizer, device)
        print("optimizer is not None, explicitly moved on device.")
        max_norm_path = os.path.join(parent_dir, 'max_norm.pt')
        if os.path.exists(max_norm_path):
            max_norm = torch.load(max_norm_path)
        else:
            print("max_norm.pt not found, using default max_norm=1.0 for gradient clipping.")

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device,
        optimizer=optimizer,
        total_steps=total_steps,
        max_norm=max_norm
    )

    # Allocate tensors for loss computation
    loss_multi = torch.zeros((1,)).float().to(device)
    loss_gauss = torch.zeros((1,)).float().to(device)
    loss_privacy_num = torch.zeros((1,)).float().to(device)
    loss_privacy_cat = torch.zeros((1,)).float().to(device)
    loss_memory = (loss_multi, loss_gauss, loss_privacy_num, loss_privacy_cat)
    w_m = torch.tensor(weight_mask, dtype=torch.float32, requires_grad=False).to(device) if weight_mask is not None else None

    print("In train.py, start training with weight_mask:", w_m)
    trainer.run_loop(start_privacy_step=start_privacy_step, privacy_metric=privacy_metric, loss_memory=loss_memory, weight_mask=w_m, completed_steps=completed_steps)

    os.makedirs(parent_dir, exist_ok=True)
    trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))
    torch.save(trainer.optimizer.state_dict(), os.path.join(parent_dir, 'optimizer.pt'))
    torch.save(trainer.max_norm, os.path.join(parent_dir, 'max_norm.pt'))
    
    info_json_path = os.path.join(parent_dir, 'DatasetInfo.json')


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

    
    
    return trainer.loss_history
