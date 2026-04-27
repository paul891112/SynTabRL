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
LOSS_HISTORY_COLUMNS = ['step', 'privacy_loss_type', 'mloss', 'gloss', 'mprivacy', 'gprivacy', 'loss']

def is_privacy_vector(privacy_metric):
    return privacy_metric in ['vector', 'adaptive', 'adaptive_dcr', 'adaptive_nndr', 'adaptive_gower']


### Original train.py from TabDDPM, modified to include different privacy loss terms and gradient clipping. ###

class Trainer:
    
    """
    Trainer class, adapted from TabDDPM project: https://github.com/yandex-research/tab-ddpm/blob/main/scripts/train.py
    Manages the training process. Keeps track of partially trained model state and manages the overall training.    
    """
    def __init__(self, diffusion,  train_iter, lr, weight_decay, steps, device=torch.device('cuda:0'), optimizer=None, total_steps=None, max_norm=1.0):
        self.diffusion = diffusion
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
        Linearly anneal the learning rate from the initial lr to 0.
        self.total_steps is set during Trainer initialization.
        self.total_steps represent total number of training steps during training, i.e. rounds * steps_per_round
        
        """
        frac_done = step / self.total_steps 
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            

    def _run_step(self, x, out_dict):
        """
        One step of learning: compute loss, update weights.
        Original implementation, no privacy loss term

        Args:
            x: input data
            out_dict: yield from FastTensorDataLoader, in lib/data.py
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
        DCR privacy terms are bounded in [0, 1], set target to 1 for privacy and data similarity trade-off.
        Implemented in privacy.py, where 0 means identity and large value means perfect mismatch. Aim for privacy_loss = infinity.
        
        Args:
            loss_multi: multinomial (categorical) loss without privacy term, from diffusion.mixed_loss(), in tab_ddpm/gaussian_multinomial_diffusion.py
            loss_gauss: gaussian (numerical) loss without privacy term, from diffusion.mixed_loss()
            privacy_multi: multinomial (categorical) privacy loss, from diffusion.mixed_loss()
            privacy_gauss: gaussian (numerical) privacy loss, from diffusion.mixed_loss()
        """
        loss_multi = loss_multi +  torch.exp(-(self.diffusion.dcr_weight * privacy_multi)) if loss_multi != 0 else loss_multi  # aim for privacy ratio = 1
        loss_gauss = loss_gauss +  torch.exp(-(0.1 * self.diffusion.dcr_weight * privacy_gauss))  # aim for privacy ratio = 1        

        loss = loss_multi + loss_gauss
        return loss
      
    
    def _run_step_privacy_nndr(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        """
        NNDR privacy terms are bounded in [0, 1], set target to 1.
        """
        loss_multi = loss_multi + (1 - self.diffusion.nndr_weight * privacy_multi) if loss_multi != 0 else loss_multi  # aim for privacy ratio = 1
        loss_gauss = loss_gauss + (1 - self.diffusion.nndr_weight * privacy_gauss)  # aim for privacy ratio = 1
            
        loss = loss_multi + loss_gauss  # aim for privacy ratio = 1
        return loss
    
    def _run_step_privacy_gower(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        """
        Gower's DCR privacy terms are bounded in [0, 1], set target to 1.
        """
        loss_multi = loss_multi + (1 - self.diffusion.gower_weight * privacy_multi) if loss_multi != 0 else loss_multi  # aim for privacy ratio = 1
        loss_gauss = loss_gauss + (1 - self.diffusion.gower_weight * privacy_gauss)  # aim for privacy ratio = 1
        
        loss = loss_multi + loss_gauss
        
        return loss
    
    def _run_step_privacy_sum(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        """
        Depricated.
        Sum approach, privacy terms are bounded in [0, 2], set target to 2.
        """
        loss_multi = loss_multi + (2 - self.diffusion.sum_weight * privacy_multi) if loss_multi != 0 else loss_multi  # aim for privacy ratio = 1
        loss_gauss = loss_gauss + (2 - self.diffusion.sum_weight * privacy_gauss)  # aim for privacy ratio = 1
        
        loss = loss_multi + loss_gauss  #aim for privacy ratio = 1
        return loss
    
    def _run_step_privacy_vector(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        """
        Depricated.
        Vector privacy loss implementation. Apply 2x weight on one of the three privacy terms if in 
        low privacy state.
        """
        target_vec = torch.ones((3,), device=self.device)
        
        # privacy_gauss, privacy_multi are vectors of size 3
        weighted_privacy_gauss = self.adaptive_weight * privacy_gauss
        weighted_privacy_multi = self.adaptive_weight * privacy_multi
        
        # Apply the specific transformation formulas
        # Formula for index 0: exp(-x)
        # Formula for indices 1 & 2: 1 - x
        def apply_transform(weighted_tensor):
            # Create a copy to avoid modifying the intermediate weighted_tensor
            res = torch.empty_like(weighted_tensor)
            
            # DCR (Index 0)
            res[0] = torch.exp(-weighted_tensor[0])
            
            # NNDR & Gower (Indices 1 and 2)
            res[1:] = 1 - weighted_tensor[1:]
            
            return res

        # Generate final tensors
        dist_gauss = apply_transform(weighted_privacy_gauss)
        dist_multi = apply_transform(weighted_privacy_multi)

        lg = dist_gauss.sum() + loss_gauss
        lm = dist_multi.sum() + loss_multi
        vec = (lg + lm)
        
        return vec
    
    def _run_step_privacy_adaptive_dcr(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        """
        Adaptive single metric approach, DCR as loss term.
        DCR privacy terms are bounded in [0, 1] due to MinMaxScaler normalization, set target to 1 for privacy and fidelity trade-off.
        Implemented in privacy.py, where 0 means identity and large value means perfect mismatch. Aim for privacy_loss = infinity.
        """        
        loss_multi = loss_multi + torch.exp(-self.adaptive_weight[0] * privacy_multi) if loss_multi != 0 else loss_multi  # aim for largest privacy value 
        loss_gauss = loss_gauss + torch.exp(-self.adaptive_weight[0] * privacy_gauss)  # aim for privacy ratio = 1        

        loss = loss_multi + loss_gauss
        return loss


    def _run_step_privacy_adaptive_nndr(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        """
        Adaptive single metric approach, NNDR as loss term.
        """        
        loss_multi = loss_multi + (1 - self.adaptive_weight[1] * privacy_multi) if loss_multi != 0 else loss_multi  # aim for largest privacy value 
        loss_gauss = loss_gauss + (1 - self.adaptive_weight[1] * privacy_gauss)  # aim for privacy ratio = 1        

        loss = loss_multi + loss_gauss
        return loss
    
    def _run_step_privacy_adaptive_gower(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        """
        Adaptive single metric approach, Gower's DCR as loss term.
        """        
        loss_multi = loss_multi + (1 - self.adaptive_weight[2] * privacy_multi) if loss_multi != 0 else loss_multi  # aim for largest privacy value 
        loss_gauss = loss_gauss + (1 - self.adaptive_weight[2] * privacy_gauss)  # aim for privacy ratio = 1        

        loss = loss_multi + loss_gauss
        return loss  
    
    
    def _run_step_privacy_adaptive(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        """ 
        Adaptive sum approach, privacy loss represented as vector and apply dynamic adaptive weights.
        Adopted from _run_step_privacy_vector(). 
        """
       
        # privacy_gauss, privacy_multi are vectors of size 3
        weighted_privacy_gauss = self.adaptive_weight * privacy_gauss
        weighted_privacy_multi = self.adaptive_weight * privacy_multi
        
        # Apply the transformation formulas
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

        # Generate your two final tensors
        dist_gauss = apply_target(weighted_privacy_gauss)
        dist_multi = apply_target(weighted_privacy_multi)

        lg = dist_gauss.sum() + loss_gauss
        lm = dist_multi.sum() + loss_multi
        vec = (lg + lm)
        
        return vec
        
    
    def _run_step_privacy(self, x, out_dict, privacy_metric='dcr', loss_memory=None, weight_mask=None):
        """
        Runs one training step with privacy loss term.
        The modified _run_step() function to incorporate privacy loss term into TabDDPM training.\n
        Use parameter privacy_metric to call loss function.\n
        Loss function = (1 + ratio) * loss, aim for ratio = 1 so minimize 1-ratio.\n

        Args:
            x: real data
            out_dict: yield from FastTensorDataLoader, in lib/data.py
            privacy_metric: which privacy metric to use, one of PRIVACY_METRIC. Controls the specific implementation of the privacy loss function. Default is 'dcr'.
            loss_memory: tuple of tensors to store loss values for privacy computation, only used for vector and adaptive approach. Defaults to None if not vector approach.
            weight_mask: torch tensor for vector and sum privacy approach. Defaults to None.
        Returns:
            tuple: categorical and numerical losses without privacy, as well as numerical and categorical privacy loss terms, and the final effectice total loss.
        """
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        
        self.optimizer.zero_grad()
        
        # Check if privacy_metric is valid
        if privacy_metric not in PRIVACY_METRIC:
            raise ValueError(f"Unknown privacy metric: {privacy_metric}. Supported metrics: {PRIVACY_METRIC}")
        
        # Compute losses
        loss_multi, loss_gauss, privacy_multi, privacy_gauss = self.diffusion.mixed_loss(x, out_dict, privacy_metric=privacy_metric, loss_memory=loss_memory, weight_mask=weight_mask)
        
        # adaptive, vector, adaptive with single metric       
        if is_privacy_vector(privacy_metric):
            self.adaptive_weight = weight_mask
            
        # Depending on privacy metric, select the according loss functions
        privacy_method = getattr(self, "_run_step_privacy_"+privacy_metric)
        loss = privacy_method(loss_multi, loss_gauss, privacy_multi, privacy_gauss)
        
        loss.backward()        
        
        # Clip gradients to avoid exploding gradients
        clipped_norm = torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), max_norm=self.max_norm)    
        self.gradient_history.append(clipped_norm.item())
            
        self.optimizer.step()

        return loss_multi, loss_gauss, privacy_multi, privacy_gauss, loss
    

    def run_loop(self, start_privacy_step, privacy_metric, completed_steps, loss_memory=None, weight_mask=None, privacy_term_weights=None):
        """
        Original TabDDPM training loop, modified to incorporate privacy loss term after certain training step, 
        controlled by start_privacy_step. 
        If start_privacy_step < 0, never use privacy term but run the original TabDDPM training loop. 
        If start_privacy_step >= 0, run original TabDDPM training loop until start_privacy_step, 
        then incorporate privacy loss term and run modified training loop with privacy term until the end 
        of training. 
        Privacy metric is selected by privacy_metric parameter, which controls the specific implementation 
        of the privacy loss function. 
        See _run_step_privacy_* functions for details on different implementations of privacy loss functions. 
        
        Args:
            start_privacy_step (int): Step to start incorporating privacy loss term. If <0, never use privacy term.
            privacy_metric (str): Privacy metric to use. One of PRIVACY_METRIC.
            completed_steps (int): Number of steps already completed in previous rounds. Comes from agent, i.e. train() function.
            loss_memory (torch.Tensor, optional): Tensors to store loss values for privacy computation. Defaults to None if not vector approach.
            weight_mask (torch.Tensor, optional): Weight mask for vector privacy approach. Defaults to None. (Depricated)
            privacy_term_weights (list, optional): Dynamic adaptive weights for privacy terms if using 'sum' privacy metric. Defaults to None.
        """
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_privacy_multi = 0.0
        curr_privacy_gauss = 0.0
        curr_total_loss = 0.0
        
        
        assert start_privacy_step < self.steps, "In Trainer.run_loop(), start_privacy_step must be less than total steps. Check parameters passed to train() and the config.toml file."
        if privacy_metric == 'sum':
            assert privacy_term_weights is not None, "For 'sum' privacy metric, privacy_term_weights must be provided."
        
        curr_count = 0
        
        
        # Original TabDDPM training loop without privacy term
        if start_privacy_step < 0:
            print("Privacy term currently not incorporated during training.")
            # Never incorporate privacy term
            while step < self.steps:
                x, out_dict = next(self.train_iter)
                out_dict = {'y': out_dict}
                
                # Never use privacy term
                batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

                self._anneal_lr(step + completed_steps)  # Match learning rate to training progress

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
        # ----- End of original TabDDPM training loop -----    
            
         
        # run training loop with privacy term          
        else:    
            print(f"Incorporate privacy term starting at step {start_privacy_step}.")
            
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
                    
                    # reset scalar privacy losses
                    curr_privacy_gauss = 0.0
                    curr_privacy_multi = 0.0
                    curr_total_loss = 0.0

                update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())
                step += 1
            
            # Gradient Clipping
            # print(f"With privacy, 95 percentile gradient history: {torch.quantile(torch.tensor(self.gradient_history), q=0.95).item()}")  
            self.max_norm *= 0.95      
             
        # ----- End of modified training loop with privacy term ----- 
        

def move_optimizer_to_device(optimizer, device):
    """Ensure that optimizer is moved to GPU when loading optimizer state dict from checkpoint."""
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
    
    print(model_params)
    print(f"Completed_steps: {completed_steps}")
    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    )
    if continue_training:
        model.load_state_dict(torch.load(os.path.join(parent_dir, 'model.pt')))
        
    model.to(device)

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
    if continue_training:  # Training from checkpoint, load optimizer state and max_norm for gradient clipping
        optimizer = torch.optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_state = torch.load(os.path.join(parent_dir, 'checkpoint', 'optimizer.pt'))
        optimizer.load_state_dict(optimizer_state)
        move_optimizer_to_device(optimizer, device)
        max_norm_path = os.path.join(parent_dir, 'checkpoint', 'max_norm.pt')
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

    # print("In train.py, start training with weight_mask:", w_m)  # check adaptive weights on privacy loss terms at training time
    trainer.run_loop(start_privacy_step=start_privacy_step, privacy_metric=privacy_metric, loss_memory=loss_memory, weight_mask=w_m, completed_steps=completed_steps)

    # Save training progress after one round of training or the completed training. 
    os.makedirs(parent_dir, exist_ok=True)
    trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))
    if not os.path.exists(os.path.join(parent_dir, 'checkpoint')):
        os.makedirs(os.path.join(parent_dir, 'checkpoint'), exist_ok=True)
    torch.save(trainer.optimizer.state_dict(), os.path.join(parent_dir, 'checkpoint', 'optimizer.pt'))
    torch.save(trainer.max_norm, os.path.join(parent_dir, 'checkpoint', 'max_norm.pt'))


    return trainer.loss_history
