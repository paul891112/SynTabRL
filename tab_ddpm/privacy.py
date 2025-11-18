import time

import torch
import numpy as np
import torch.nn.functional as F
from torch.profiler import record_function
import math

def DCR_loss(x_num, model_out_num):
    """
    Paul
    Calculates DCR loss for numerical and categorical features.
    AI generated template.

    Args:
        x_num (torch.Tensor): numerical features\n
        model_out_num (torch.Tensor): numerical predictions\n
    Returns:
        torch.Tensor: DCR loss, distance to nearest record
    """
    x_num = torch.rand(4096, 7)
    model_out_num = torch.rand(4096, 7)

    start = time.time()  # record start time

    # Compute pairwise distances: A[i] vs all rows in B
    # distances will have shape [4096, 5000]
    distances = torch.cdist(x_num, model_out_num)  # cdist is efficient for batched distances

    # For each row in out_num, get the index of the nearest row in x_num
    nearest_idx = torch.argmin(distances, dim=0)

    # Optional: get the actual nearest rows
    nearest_rows = x_num[nearest_idx]

    # Get the actual distances
    nearest_distances = distances[nearest_idx, torch.arange(model_out_num.shape[0])]

    end = time.time()  # record start time

    print(f"Sample record: {model_out_num[0]}")
    print(f"Closest record: {nearest_rows[0]}")
    print(f"Distance: {nearest_distances[0]}")
    print(f"Test distance: {torch.dist(model_out_num[0], nearest_rows[0], p=2)}")
    print("Elapsed time to calculate all pairwise distance:", end - start, "seconds")


    A = torch.randn(4096, 7)   # (4096, 7)
    B = torch.randn(7)         # (7,)

    start = time.time()  # record start time

    # Compute Euclidean (L2) distances between B and every row in A
    # Broadcasting: (4096, 7) - (7,) → (4096, 7)
    distances = torch.norm(A - B, dim=1)  # (4096,)

    # Find index of the nearest row
    min_idx = torch.argmin(distances)

    # Get nearest row and its distance
    nearest_row = A[min_idx]
    nearest_distance = distances[min_idx]

    end = time.time()  # record end time

    print(f"Sample row: {B}")
    print(f"Nearest row: {nearest_row}")
    print(f"Nearest distance: {nearest_distance.item()}")
    print(f"Test distance: {torch.dist(B, nearest_row, p=2)}")
    print("Elapsed time to calculate all pairwise distance:", end - start, "seconds")
    
    return nearest_distance


def nndr_loss(original, synthetic):
    """
    Paul
    Calculates Nearest Neighbor Distance Ratio (NNDR) loss for numerical features.

    Args:
        original (torch.Tensor): numerical features, x_num\n
        synthetic (torch.Tensor): numerical predictions, model_out_num\n

    Returns:
        torch.Tensor: NNDR loss in ratio
    """
    # original = torch.rand(4096, 7)
    # synthetic = torch.rand(4096, 7)  # NNDR = 0.866
    # synthetic = original + torch.ones_like(original)  # NNDR = 0.0915


    start = time.time()
    # Compute distances from each synthetic record to all original records
    distances = torch.cdist(synthetic, original, p=2)

    # Find the nearest and second nearest distances for each synthetic record
    partitioned_distances = torch.topk(distances, k=2, dim=1, largest=False).values
    nearest_distances = partitioned_distances[:, 0]
    second_nearest_distances = partitioned_distances[:, 1]

    # Calculate the NNDR for each synthetic record
    nndr_list = nearest_distances / (second_nearest_distances + 1e-16)  # Avoid division by zero

    # Return the average NNDR across all synthetic records
    nndr_ratio = torch.mean(nndr_list)

    end = time.time()

    # print(1/nndr_ratio.item())
    # print(f"Elapsed time: {end-start}s")
    return nndr_ratio    

    
def categorical_privacy_loss(log_x_cat, model_out_cat, num_features):
    """
    Paul
    Calculates privacy loss of input and output in the current training step.\n
    Given the clean one-hot encoded categorical features and the predicted logits, compute euclidean distance of each feature.\n
    Then take the average of all distances, divide by square root of 2 to normalize the maximum distance to 1.\n
    Perhaps weight the privacy loss based on the diffusion timestep t, where lower t means higher influence on the total loss.\n

    Args:
        x_cat (torch.Tensor): categorical features
        
        model_out_cat (torch.Tensor): categorical predictions (logits)
        
        t (int): diffusion timestep, the higher it is, the less influence should the current privacy component have on the total loss.
    Returns:
        torch.Tensor: privacy loss
    """
    with record_function("categorical_privacy_loss"):
        
        probabilities = F.softmax(model_out_cat, dim=0)
        x_cat_ohe = torch.exp(log_x_cat)
        
        
        # Calculate distance between input and output categorical features
        distances = torch.cdist(probabilities, x_cat_ohe, p=2)
        # For each row in out_num, get the index of the nearest row in x_num
        nearest_idx = torch.argmin(distances, dim=0)

        # Get the actual distances
        nearest_distances = distances[nearest_idx, torch.arange(model_out_cat.shape[0])]
                
        # Take the average and divide by sqrt(2) to normalize the maximum distance to 1
        privacy_loss = nearest_distances/ (math.sqrt(2) * num_features)
        
        return privacy_loss
        