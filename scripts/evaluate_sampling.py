from rlagent import RLAgent
import argparse

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
import lib
import numpy as np
import torch
import time
from enum import Enum
import time
import pathlib
import gc
from datasetinfo import generate_dataset_info


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--num_sample', 
                        type=int,
                        help="The number of samples to generate",
                        nargs='?',
                        const=-1,
                )
    
    parser.add_argument(
        "--filter", 
        choices=["percentile", "threshold"],
        default="percentile",
        help="Select the filtering logic: 'percentile' (e.g., drop bottom 10%%) "
             "or 'threshold' (e.g., drop if DCR < 0.05)"
        )
    
    parser.add_argument(
        "--filter_value", 
        type=float, 
        help="The numeric value for the approach (percentile 0-100 or DCR distance)"
    )
    parser.add_argument('--change_val', action='store_true',  default=False)    
    
    
    print("Parsing arguments ...")

    args = parser.parse_args()
    
    assert 0 <= args.filter_value < 1, "Valid range [0, 1). Use 0.1 for 10% percentile or 0.05 for DCR threshold."
    raw_config = lib.load_config(args.config)
    if args.num_sample is None or args.num_sample < 0:
        args.num_sample = raw_config['sample']['num_samples']  # Will be set to default in the agent if not provided

    if 'device' in raw_config:
        device = torch.device('cuda:0')  # Paul
        # device = torch.device(raw_config['device'])  # Use specified device
    else:
        device = torch.device('cuda:0')  # Original 'cuda:1'

    agent = RLAgent(name="SamplingAgent", args=args, raw_config=raw_config, device=device)
    
    evaluate_sample_file = os.path.join(raw_config['parent_dir'], "evaluate_sampling.txt")
    with open(evaluate_sample_file, 'w') as file:
        file.write(f"Generating {args.num_sample} samples for evaluation.\n")
        file.write(f"Filtering method: {args.filter} with value: {args.filter_value}\n\n")
    
    sample_method = agent.generate_and_filter_samples_percentile if args.filter == "percentile" else agent.generate_and_filter_samples_threshold

    X_num, X_cat, y_gen = sample_method(num_samples=args.num_sample, value=args.filter_value, evaluate_sample_file=evaluate_sample_file)   
        

if __name__ == "__main__":
    main()