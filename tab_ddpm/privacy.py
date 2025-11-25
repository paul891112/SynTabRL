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
        
        
        
        probabilities = F.softmax(model_out_cat, dim=1)  # change from dim=0 to dim=1, 20.11.2025, not tested yet
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
    
def compute_gower_num(x_num, model_out_num):
    """
    Paul
    Computes Gower's distance between original and synthetic data.

    Args:
        x_num (torch.Tensor): numerical features\n
        model_out_num (torch.Tensor): numerical predictions\n
        x_cat (torch.Tensor): categorical features\n
        model_out_cat (torch.Tensor): categorical predictions (logits)\n
        num_num_features (int): number of numerical features\n
        num_cat_features (int): number of categorical features\n

    Returns:
        torch.Tensor: Gower's distance
    """
    with record_function("gower_distance"):
        
        # --- Numerical Feature Dissimilarity ---
        max_vals, _ = torch.max(x_num, dim=0, keepdim=True) # Shape (1, F_num)
        min_vals, _ = torch.min(x_num, dim=0, keepdim=True) # Shape (1, F_num)
        
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0  # Prevent division by zero
        
        # Broadcasting
        D_num_raw = torch.abs(x_num[:, None, :] - model_out_num[None, :, :])
        D_num = D_num_raw / ranges
        
        # Sum of numerical dissimilarities over the feature dimension (axis=2)
        D_num_sum = torch.sum(D_num, dim=2) # Result is (N_o, N_s)
        total_features = x_num.shape[1]
        D_num_sum = D_num_sum / total_features  # Normalize by number of numerical features
        return torch.mean(D_num_sum)


        
def compute_gower_cat(log_x_cat, model_out_cat, num_cat_features, category_sizes):
    
    
        # --- Categorical Feature Dissimilarity ---
        log_x_recon = torch.empty_like(model_out_cat)
        
        runner = 0
        
        for idx in category_sizes:
            ix = np.arange(runner, runner + idx)
            runner += idx
            probs = F.softmax(model_out_cat[:, ix], dim=1)  # shape (B, C)
            idx = probs.argmax(dim=1)    # shape (B,)
            one_hot = F.one_hot(idx, num_classes=probs.size(1)).float()
            log_x_recon[:, ix] = (one_hot + 1e-30)
            
        
        D_cat_sum = torch.zeros((len(log_x_recon), len(model_out_cat)), dtype=torch.float32, device=log_x_cat.device)
        x_cat_ohe = torch.exp(log_x_cat)        
        
        if len(category_sizes) > 0:
            runner = 0
            
            # Add these checks (or equivalent debugging prints)
            total_width_original = x_cat_ohe.shape[1]
            total_width_synthetic = log_x_recon.shape[1]
            expected_width = sum(category_sizes)

            if total_width_original != total_width_synthetic or total_width_original != expected_width:
                raise ValueError(f"Feature widths mismatch: Original width={total_width_original}, Synthetic width={total_width_synthetic}, Expected width={expected_width}")
            
            for size in category_sizes:
                # Extract one-hot-encoded columns for the current feature
                O_cat_k = x_cat_ohe[:, runner : runner + size]
                S_cat_k = log_x_recon[:, runner : runner + size]
                
                # Broadcasting comparison (N_o x N_s x size)
                O_cat_k_b = O_cat_k[:, None, :]
                S_cat_k_b = S_cat_k[None, :, :]
                
                # Check if OHE vectors are identical for each pair
                # (O_cat_k_b == S_cat_k_b) is (N_o, N_s, size) Booleans
                # torch.all(..., dim=2) checks if all elements in the size dimension are True
                is_equal = torch.all(O_cat_k_b == S_cat_k_b, dim=2) # Result is (N_o, N_s) Booleans
                
                # Dissimilarity: 0 if equal (True), 1 if different (False)
                # (~is_equal) in NumPy is replaced by (~is_equal).float() in PyTorch to get 1/0
                D_cat_k = (~is_equal).float()
                
                # Accumulate dissimilarities
                D_cat_sum += D_cat_k
                runner += size

        # --- 3. Final Gower Distance Calculation ---
        G_matrix = D_cat_sum / num_cat_features
        
        # Return the average of every pairwise Gower's distance
        return torch.mean(G_matrix)
    

        