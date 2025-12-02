import time

import torch
import numpy as np
import torch.nn.functional as F
from torch.profiler import record_function
import math

DCR_EXPONENTIAL_COMPLEMENT_CONSTANT = 1.5  # constant k in exponential complement



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
    return nndr_ratio.mean()

    
def categorical_privacy_loss(log_x_cat, model_out_cat, num_cat_features, category_sizes):
    """
    Paul
    Calculates privacy loss of input and output in the current training step.\n
    Given the clean one-hot encoded categorical features and the predicted logits, compute euclidean distance of each feature.\n
    Then take the average of all distances, divide by square root of 2 to normalize the maximum distance to 1.\n
    Perhaps weight the privacy loss based on the diffusion timestep t, where lower t means higher influence on the total loss.\n

    Args:
        x_cat (torch.Tensor): categorical features
        
        model_out_cat (torch.Tensor): categorical predictions (logits)
        
        num_features: 
    Returns:
        torch.Tensor: privacy loss
    """
    with record_function("categorical_privacy_loss"):
        
        probabilities = F.softmax(model_out_cat, dim=1)  
        x_cat_ohe = torch.exp(log_x_cat)
        
        # Normalization mask, divide each element by the number of labels of the feature
        mask = torch.ones(x_cat_ohe.shape[1], device=x_cat_ohe.device)
        start_idx = 0
        for size in category_sizes:
            mask[start_idx:start_idx + size] = 1.0 / size
            start_idx += size
        mask = mask.unsqueeze(0)
        probabilities = probabilities * mask
        x_cat_ohe = probabilities * mask
        
        # Calculate distance between input and output categorical features
        distances = torch.cdist(probabilities, x_cat_ohe, p=2)
        # For each row in out_num, get the index of the nearest row in x_num
        nearest_idx = torch.argmin(distances, dim=0)

        # Get the actual distances
        nearest_distances = distances[nearest_idx, torch.arange(model_out_cat.shape[0])]
                
        # Divide by sqrt(2) to normalize the maximum distance to 1
        privacy_loss = nearest_distances/ (math.sqrt(2))
        
        return privacy_loss.mean()
    
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
    with record_function("gower_distance_num"):
        
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
    
    with record_function("gower_distance_cat"):
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


def dcr_cat_loss(log_x_cat, model_out_cat, num_cat_features, category_sizes):
    
    probabilities = F.softmax(model_out_cat, dim=1)  
    x_cat_ohe = torch.exp(log_x_cat)
    
    start_idx = 0
    mask = torch.ones(x_cat_ohe.shape[1], dtype=float, device=x_cat_ohe.device)
    start_idx = 0
    
    # Normalize by the number of labels per feature
    for size in category_sizes:
        mask[start_idx:start_idx + size] = 1.0 / size
        start_idx += size
    mask = mask.unsqueeze(0)
    weighted_synthetic_data = probabilities * mask
    weighted_original_data = x_cat_ohe * mask
    
    dists = torch.cdist(
        weighted_synthetic_data, 
        weighted_original_data, 
        p=2.0
    )     
    
    # Find the nearest distance (d1), which is the 1st smallest value (k=1)
    # The output is a named tuple (values, indices)
    nearest_dist_tuple = torch.kthvalue(dists, k=1, dim=1)
    nearest_distances = nearest_dist_tuple.values # Shape (num_synthetic_records)
    
    normalized = 1 - torch.exp(-DCR_EXPONENTIAL_COMPLEMENT_CONSTANT * nearest_distances)

    return torch.mean(normalized)
    

def dcr_num_loss(x_num, model_out_num):
    
    dists = torch.cdist(
        model_out_num, 
        x_num, 
        p=2.0
    )     
    
    # Find the nearest distance (d1), which is the 1st smallest value (k=1)
    # The output is a named tuple (values, indices)
    nearest_dist_tuple = torch.kthvalue(dists, k=1, dim=1)
    nearest_distances = nearest_dist_tuple.values # Shape (num_synthetic_records)
    
    normalized = 1 - torch.exp(-DCR_EXPONENTIAL_COMPLEMENT_CONSTANT * nearest_distances)
    
    return torch.mean(normalized)
    
        
    
PRIVACY_FUNCTIONS = {
    "nndr_cat_loss": categorical_privacy_loss,
    "nndr_num_loss": nndr_loss,
    "gower_cat_loss": compute_gower_cat,
    "gower_num_loss": compute_gower_num,
    "dcr_cat_loss": dcr_cat_loss,
    "dcr_num_loss":dcr_num_loss,
    
}