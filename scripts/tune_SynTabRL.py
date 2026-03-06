import subprocess
import lib
import os
import optuna
from copy import deepcopy
import shutil
import argparse
from pathlib import Path


def log_to_txt(file_path, message):
    with open(file_path, 'a') as f:
        f.write(message + '\n')

parser = argparse.ArgumentParser()
parser.add_argument('ds_name', type=str)
parser.add_argument('train_size', type=int)
parser.add_argument('eval_type', type=str)
parser.add_argument('eval_model', type=str)
parser.add_argument('prefix', type=str)
parser.add_argument('--eval_seeds', action='store_true',  default=False)

args = parser.parse_args()
train_size = args.train_size
ds_name = args.ds_name
eval_type = args.eval_type 
assert eval_type in ('merged', 'synthetic')
prefix = str(args.prefix)

base_config_path = f'privacy_result/{ds_name}/config.toml'
original_config_path = f'exp/{ds_name}/ddpm_cb_best/config.toml'
parent_path = Path(f'privacy_result/{ds_name}/')
exps_path = Path(f'privacy_result/{ds_name}/many-exps/')
pipeline = f'scripts/rlagent.py'

"""
if args.privacy:  # Paul: added privacy option
    base_config_path = f'privacy_result/{ds_name}/config.toml'
    parent_path = Path(f'privacy_result/{ds_name}/')
    exps_path = Path(f'privacy_result/{ds_name}/many-exps/')
    pipeline = f'scripts/rlagent.py'
    
else:
    pipeline = f'scripts/pipeline.py'
    base_config_path = f'exp/{ds_name}/config.toml'
    parent_path = Path(f'exp/{ds_name}/')
    exps_path = Path(f'exp/{ds_name}/many-exps/') # temporary dir. maybe will be replaced with tempdiвdr

"""
eval_seeds = f'scripts/eval_seeds.py'

print(f"Base config path: {base_config_path}")
os.makedirs(exps_path, exist_ok=True)

def _suggest_mlp_layers(trial):
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t
    min_n_layers, max_n_layers, d_min, d_max = 1, 4, 7, 10
    n_layers = 2 * trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    return d_layers

def objective(trial):
    
    # Loading original TabDDPM config
    original_config = lib.load_config(original_config_path)
    lr = original_config['train']['main']['lr']
    steps = original_config['train']['main']['steps']
    batch_size = original_config['train']['main']['batch_size']
    weight_decay = original_config['train']['main']['weight_decay']
    d_layers = _suggest_mlp_layers(trial)
    eval_type = original_config['eval']['type']['eval_type']
    num_samples = original_config['sample']['num_samples']
    gaussian_loss_type = original_config['diffusion_params']['gaussian_loss_type']
    num_timesteps = original_config['diffusion_params']['num_timesteps']
    # scheduler = trial.suggest_categorical('scheduler', ['cosine', 'linear'])
    
    
    # define SynTabRL hyperparameters
    pretrain_steps = trial.suggest_categorical('pretrain_steps', [2000, 4000])
    steps_per_round = trial.suggest_categorical('steps_per_round', [1000, 2000])
    privacy_discount = 0.1  
    logsumexp_sigma = 0.01  # control "softness" of softmin for DCR computations
    
    # Paul: add privacy parameter
    privacy_discount = trial.suggest_categorical('privacy_discount', [0.1, 0.05, 0.01])
    dcr = trial.suggest_categorical('dcr', [0.2, 0.3])
    nndr = trial.suggest_categorical('nndr', [0.8, 0.85])
    gower = trial.suggest_categorical('gower', [0.05, 0.1])

    base_config = lib.load_config(base_config_path)
    
    base_config['train']['main']['evaluation_file'] = "SynTabRL_eval.json"
    base_config['train']['main']['lr'] = lr
    base_config['train']['main']['steps'] = steps
    # base_config['train']['main']['steps_per_round'] = steps_per_round  # might be incompatible with arbitrary steps
    base_config['train']['main']['pretrain_steps'] = pretrain_steps
    base_config['train']['main']['batch_size'] = batch_size
    base_config['train']['main']['weight_decay'] = weight_decay
    base_config['train']['main']['privacy_discount'] = privacy_discount
    base_config['train']['main']['logsumexp_sigma'] = logsumexp_sigma
    base_config['model_params']['rtdl_params']['d_layers'] = d_layers
    base_config['eval']['type']['eval_type'] = eval_type
    base_config['sample']['num_samples'] = num_samples
    base_config['diffusion_params']['gaussian_loss_type'] = gaussian_loss_type
    base_config['diffusion_params']['num_timesteps'] = num_timesteps
    # base_config['diffusion_params']['scheduler'] = scheduler
    
    base_config['privacy_config']['dcr'] = dcr
    base_config['privacy_config']['nndr'] = nndr
    base_config['privacy_config']['gower'] = gower
    
    # ----- Paul: tune start_privacy_step ------------------------------------
    """
    if 'start_privacy_step' in base_config['train']['main']:
        
        # Determine the maximum value for tuning (max step)
        max_step = base_config['train']['main']['steps'] 
        
        # Suggest a value for start_privacy_step (e.g., between 0 and max_step // 2)
        start_privacy_step = trial.suggest_int('start_privacy_step', 
                                                low=0, 
                                                high=max_step // 2, 
                                                step=100) # Suggest in increments of 100
        
        # Assign the suggested value to the config
        base_config['train']['main']['start_privacy_step'] = start_privacy_step
        trial.set_user_attr("start_privacy_step", start_privacy_step)
    # ELSE: If the parameter is not in the base config, we safely ignore it.
    
    # -----------------------------------------------------------------
    """

    base_config['parent_dir'] = str(exps_path / f"{trial.number}")
    base_config['eval']['type']['eval_model'] = args.eval_model
    if args.eval_model == "mlp":
        base_config['eval']['T']['normalization'] = "quantile"
        base_config['eval']['T']['cat_encoding'] = "one-hot"

    trial.set_user_attr("config", base_config)

    lib.dump_config(base_config, exps_path / 'config.toml')
    
    subprocess.run(['python3.9', f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--train', '--adaptive_single_metric', 'dcr'], check=True)
    
    """
    if args.privacy: # Paul: added privacy option
        subprocess.run(['python3.9', f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--train', '--adaptive_approach'], check=True)

    else:
        subprocess.run(['python3.9', f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--train', '--change_val'], check=True)
    """
    n_datasets = 5
    score = 0.0
    privacy = 0.0

    for sample_seed in range(n_datasets):
        base_config['sample']['seed'] = sample_seed
        lib.dump_config(base_config, exps_path / 'config.toml')
        
        
        subprocess.run(['python3.9', f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--sample', '--eval', '--change_val'], check=True)

        report_path = str(Path(base_config['parent_dir']) / f'results_{args.eval_model}.json')
        eval_path = str(Path(base_config['parent_dir']) / 'SynTabRL_eval.json')
        report = lib.load_json(report_path)
        eval_result = lib.load_json(eval_path)

        if 'r2' in report['metrics']['val']:
            score += report['metrics']['val']['r2']
        else:
            score += report['metrics']['val']['macro avg']['f1-score']
        
        # privacy score is computed as summed absolute increase in privacy metrics
        privacy += (eval_result['dcr_euclidean'] - dcr) + (eval_result['nndr_euclidean'] - nndr) + (eval_result['dcr_gower_distance'] - gower)
        

    shutil.rmtree(exps_path / f"{trial.number}")

    return score / n_datasets, privacy / n_datasets

study = optuna.create_study(
    directions=['maximize', 'maximize'],
    sampler=optuna.samplers.TPESampler(seed=0),
)

study.optimize(objective, n_trials=20, show_progress_bar=True)

best_ml_config_path = parent_path / f'{prefix}_best_ML/config.toml'
best_privacy_config_path = parent_path / f'{prefix}_best_privacy/config.toml'

trial_with_highest_ML = max(study.best_trials, key=lambda t: t.values[0])
trial_with_highest_privacy = max(study.best_trials, key=lambda t: t.values[1])


best_config = trial_with_highest_ML.user_attrs['config']
best_config["parent_dir"] = str(parent_path / f'{prefix}_best_ML/')
best_privacy_config = trial_with_highest_privacy.user_attrs['config']
best_privacy_config["parent_dir"] = str(parent_path / f'{prefix}_best_privacy/')

os.makedirs(parent_path / f'{prefix}_best_ML', exist_ok=True)
os.makedirs(parent_path / f'{prefix}_best_privacy', exist_ok=True)
lib.dump_config(best_config, best_ml_config_path)
lib.dump_config(best_privacy_config, best_privacy_config_path)
lib.dump_json(optuna.importance.get_param_importances(study, target=lambda t: t.values[0]), parent_path / f'{prefix}_best_ML/importance.json')
lib.dump_json(optuna.importance.get_param_importances(study, target=lambda t: t.values[1]), parent_path / f'{prefix}_best_privacy/importance.json')


subprocess.run(['python3.9', f'{pipeline}', '--config', f'{best_ml_config_path}', '--train', '--sample', '--eval'], check=True)

if args.eval_seeds:
    best_exp = str(parent_path / f'{prefix}_best/config.toml')
    subprocess.run(['python3.9', f'{eval_seeds}', '--config', f'{best_exp}', '10', "ddpm", eval_type, args.eval_model, '5'], check=True)
    