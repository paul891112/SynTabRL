import pandas as pd
from CTGAN.CTGAN.tests.unit import synthesizer
import tomli
import shutil
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import numpy as np
import torch
import lib
from pathlib import Path
import argparse
import zero
import os
from scripts.eval_catboost import train_catboost
import pickle

def load_config(path) :
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
    parser.add_argument('--train', action='store_true',  default=False)
    parser.add_argument('--sample', action='store_true',  default=False)
    parser.add_argument('--eval', action='store_true',  default=False)
    parser.add_argument('--change_val', action='store_true',  default=False)

    args = parser.parse_args()
    raw_config = lib.load_config(args.config)
    info = lib.load_json(os.path.join(raw_config['real_data_path'], 'info.json'))
    timer = zero.Timer()
    timer.run()
    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), args.config)
    synthesizer = None
    
    real_data_path = os.path.normpath(raw_config['real_data_path'])
    
    if args.train:
        
        # Load data
        X_num_train = np.load(os.path.join(real_data_path, 'X_num_train.npy'), allow_pickle=True)
        num_col_count = X_num_train.shape[1] if X_num_train is not None else 0
        assert num_col_count == info['n_num_features'], f"Expected {info['n_num_features']} numerical columns, but found {num_col_count} in the data."


        cat_path = os.path.join(raw_config['real_data_path'], 'X_cat_train.npy')
        
        # Only run if the categorical columns exist
        if os.path.exists(cat_path):
            X_cat_train = np.load(cat_path, allow_pickle=True)
            cat_col_count = X_cat_train.shape[1] 
            assert cat_col_count == info['n_cat_features'], f"Expected {info['n_cat_features']} categorical columns, but found {cat_col_count} in the data."
        else:
            X_cat_train = None
            X_cat = None # np.empty((X_num_train.shape[0], 0), dtype=object)
            cat_col_count = 0
            
        y_train = np.load(os.path.join(raw_config['real_data_path'], 'y_train.npy'), allow_pickle=True)
        ytype = y_train.dtype  # Capture the original dtype of the target variable for later use

        print(f"Loaded data with {num_col_count} numerical columns and {cat_col_count} categorical columns.")

        
        # Concatenate into a single DataFrame
        real_data = lib.concat_to_pd(X_num_train, X_cat_train, y_train)

        # Ensure all column names are strings
        real_data.columns = [str(col) for col in real_data.columns]
        
        target_categorical_cols = [str(i) for i in range(info['n_num_features'], info['n_num_features'] + info['n_cat_features'])]

        for col in target_categorical_cols:
            if col in real_data.columns:
                # This converts the underlying integers to strings, 
                # so SDV doesn't try to perform math/rounding on them.
                real_data[col] = real_data[col].astype(str)
            else:
                print(f"Column {col} not found.")

        # If 'y' is categorical, ensure it is also a string
        if info['task_type'] in ['binary', 'multiclass']:
            real_data['y'] = real_data['y'].astype(str)
        
        print(real_data.dtypes)
        

        # Detect metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=real_data)
        print("Metadata detected successfully.")

        # Initialize CTGAN Synthesizer
        synthesizer = CTGANSynthesizer(
            metadata,
            enforce_rounding=True,
            epochs=raw_config['train_params']['epochs'], 
            verbose=True
        )

        # Train the Model
        print("Starting training...")
        synthesizer.fit(real_data)
        
        # Generate Synthetic Data
        num_rows = raw_config['sample']['num_samples'] if raw_config['sample']['num_samples'] else len(real_data)
        synthetic_data = synthesizer.sample(num_rows=num_rows)
        
        
        X_num_synthetic = synthetic_data.iloc[:, :info['n_num_features']].values
        X_cat_synthetic = synthetic_data.iloc[:, info['n_num_features'] : info['n_num_features'] + info['n_cat_features']].values
        y_synthetic = synthetic_data['y'].values

        np.save(os.path.join(raw_config['parent_dir'], 'X_num_train.npy'), X_num_synthetic.astype(float))
        if info['n_cat_features'] > 0:
            np.save(os.path.join(raw_config['parent_dir'], 'X_cat_train.npy'), X_cat_synthetic.astype(str))
        np.save(os.path.join(raw_config['parent_dir'], 'y_train.npy'), y_synthetic.astype(ytype))
        
        # Save synthesizer parameters
        synthesizer.save(os.path.join(raw_config['parent_dir'], 'ctgan_synthesizer.pkl'))


    if args.sample:
        # Generate Synthetic Data
        synthesizer = CTGANSynthesizer.load(os.path.join(raw_config['parent_dir'], 'ctgan_synthesizer.pkl'))
        num_rows = raw_config['sample']['num_samples'] if raw_config['sample']['num_samples'] else len(real_data)
        synthetic_data = synthesizer.sample(num_rows=num_rows)

        X_num_synthetic = synthetic_data.iloc[:, :info['n_num_features']].values
        X_cat_synthetic = synthetic_data.iloc[:, info['n_num_features'] : info['n_num_features'] + info['n_cat_features']].values
        y_synthetic = synthetic_data['y'].values

        np.save(os.path.join(raw_config['parent_dir'], 'X_num_train.npy'), X_num_synthetic.astype(float))
        np.save(os.path.join(raw_config['parent_dir'], 'X_cat_train.npy'), X_cat_synthetic.astype(str))
        np.save(os.path.join(raw_config['parent_dir'], 'y_train.npy'), y_synthetic.astype(ytype)) 


    if args.eval:
        if raw_config['eval']['type']['eval_model'] == 'catboost':
            train_catboost(
                parent_dir=raw_config['parent_dir'],
                real_data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val
            )

    print(f'Elapsed time: {str(timer)}')
    



if __name__ == "__main__":
    main()