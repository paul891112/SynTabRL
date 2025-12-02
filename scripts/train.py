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

PRIVACY_METRIC = {"dcr", "nndr", "gower"}
GOWER_STD = 0.2
NOISE_STD = 0.0001  # Standard deviation of noise added to EMA model parameters
LOSS_HISTORY_COLUMNS = ['step', 'privacy_loss_type', 'mloss', 'gloss', 'mprivacy', 'gprivacy', 'loss']

class DatasetInfo:
    """
    Paul Custom Class for information passing.
    Holds dataset information such as category sizes and number of numerical features.
    """
    def __init__(self, k, trainer):
        self._K = k
        self._num_numerical_features = trainer.diffusion._num_numerical_features
    
    def get_category_sizes(self):
        return self._K
    def get_num_numerical_features(self):
        return self._num_numerical_features

class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, device=torch.device('cuda:1')):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter  # called with train_loader in train()
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=LOSS_HISTORY_COLUMNS)
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        """
        One step of learning: comute loss, update weights.
        Original implementation, no privacy loss term

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
        
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss
    
    def _run_step_privacy_dcr(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        """
        DCR privacy terms are bounded in range [0, 1], implemented in privacy.py,
        where 0 means identity and 1 means perfect mismatch. Aim for privacy_loss = 1.
        """
        loss_multi = loss_multi * (2 - privacy_multi)  # aim for privacy ratio = 1
        loss_gauss = loss_gauss * (2 - privacy_gauss)  # aim for privacy ratio = 1
        
        loss = loss_multi + loss_gauss
        return loss
    
    def _run_step_privacy_nndr(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        loss_multi = loss_multi * (2 - privacy_multi)  # aim for privacy ratio = 1
        loss_gauss = loss_gauss * (2 - privacy_gauss)  # aim for privacy ratio = 1
            
        loss = loss_multi + loss_gauss  #aim for privacy ratio = 1
        return loss
    
    def _run_step_privacy_gower(self, loss_multi, loss_gauss, privacy_multi, privacy_gauss):
        loss_multi = loss_multi * (2 - privacy_multi)  # aim for privacy ratio = 1
        loss_gauss = loss_gauss * (2 - privacy_gauss)  # aim for privacy ratio = 1
        
        loss = (loss_multi + loss_gauss) * torch.normal(mean=1.0, std=GOWER_STD, size=(1,), device=self.device)  #aim for privacy ratio = 1
        return loss
    
    def _run_step_privacy(self, x, out_dict, privacy_metric='nndr'):
        """
        Incorporates privacy term to loss functions.\n
        Use paramter privacy_metric to call loss function\n
        Loss function = (1 + ratio) * loss, aim for ratio = 1 so minimize 1-ratio\n

        Args:
            x (_type_): _description_
            out_dict (_type_): yield from FastTensorDataLoader, in lib/data.py
        Returns:
            _type_: _description_
        """
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
            
        noise_dist = Normal(loc=0.0, scale=NOISE_STD)    
        with torch.no_grad(): 
            for param in self.ema_model.parameters():
                # Only apply to parameters that are meant to be trained
                if param.requires_grad:
                    # Sample noise tensor with the exact shape of the parameter
                    noise = noise_dist.sample(param.shape).to(self.device)
                    
                    # Inject the noise by adding it to the current parameter value
                    param.add_(noise)    
        
        self.optimizer.zero_grad()
        
        # mixed_loss() in gaussian_multinomial_diffusion.py
        # privacy_multi: Normalized distance ratio for categorical part [0,1]
        # privacy_gauss: NNDR ratio
        loss_multi, loss_gauss, privacy_multi, privacy_gauss = self.diffusion.mixed_loss(x, out_dict, privacy_metric=privacy_metric)
        
        
        # Depending on privacy metric, modify loss functions
        if privacy_metric not in PRIVACY_METRIC:
            raise ValueError(f"Unknown privacy metric: {privacy_metric}. Supported metrics: {PRIVACY_METRIC}")
        
                
        privacy_method = getattr(self, "_run_step_privacy_"+privacy_metric)
        loss = privacy_method(loss_multi, loss_gauss, privacy_multi, privacy_gauss)     
            
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss, privacy_multi, privacy_gauss, loss

    def run_loop(self, start_privacy_step, privacy_metric):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_privacy_multi = 0.0
        curr_privacy_gauss = 0.0
        curr_total_loss = 0.0

        curr_count = 0
        if start_privacy_step < 0:
            print("Never use privacy term during training.")
            # Never incorporate privacy term
            while step < self.steps:
                x, out_dict = next(self.train_iter)
                out_dict = {'y': out_dict}
                
                # Never use privacy term
                batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

                self._anneal_lr(step)

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

        
        else:    
            print(f"Incorporate privacy term after step {start_privacy_step}.")
            
            # Initial noise injection to EMA model
            if privacy_metric == 'gower':
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
             # ----- End of noise -----
            
            while step < self.steps:
                x, out_dict = next(self.train_iter)
                out_dict = {'y': out_dict}
                pm = privacy_metric
                
                
                # Incorporates privacy term after certain training step
                if step < start_privacy_step:
                    batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)
                    pm = ""
                else:
                    batch_loss_multi, batch_loss_gauss, batch_privacy_multi, batch_privacy_gauss, batch_total_loss = self._run_step_privacy(x, out_dict, privacy_metric=privacy_metric)


                self._anneal_lr(step)

                curr_count += len(x)
                curr_loss_multi += batch_loss_multi.item() * len(x)
                curr_loss_gauss += batch_loss_gauss.item() * len(x)
                curr_privacy_multi += batch_privacy_multi.item() * len(x)
                curr_privacy_gauss += batch_privacy_gauss.item() * len(x)
                curr_total_loss += batch_total_loss.item() * len(x)

                if (step + 1) % self.log_every == 0:
                    mloss = np.around(curr_loss_multi / curr_count, 4)
                    gloss = np.around(curr_loss_gauss / curr_count, 4)
                    mprivacy = np.around(curr_privacy_multi / curr_count, 4)
                    gprivacy = np.around(curr_privacy_gauss/ curr_count, 4)
                    total_loss = np.around(curr_total_loss / curr_count, 4)
                    if (step + 1) % self.print_every == 0:
                        print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} MPrivacy: {mprivacy} GPrivacy: {gprivacy} Sum: {total_loss}')
                    self.loss_history.loc[len(self.loss_history)] =[step + 1, pm, mloss, gloss, mprivacy, gprivacy, total_loss]
                    curr_count = 0
                    curr_loss_gauss = 0.0
                    curr_loss_multi = 0.0
                    curr_privacy_gauss = 0.0
                    curr_privacy_multi = 0.0
                    curr_total_loss = 0.0

                update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

                step += 1      
        

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
    privacy_metric = 'nndr'
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

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device
    )

    trainer.run_loop(start_privacy_step=start_privacy_step, privacy_metric=privacy_metric)

    trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))
    
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

    # D. Save the updated dictionary back to the file (overwriting)
    lib.dump_json(new_info, info_json_path)
    
    return trainer.loss_history
