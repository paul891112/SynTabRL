import time

import torch
import numpy as np
import torch.nn.functional as F
from torch.profiler import record_function
import math

DCR_EXPONENTIAL_COMPLEMENT_CONSTANT = 1.5  # constant k in exponential complement



def numerical_nndr_loss(original, synthetic, **kwargs):
    """
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

def no_categorical_nndr(log_x_cat, model_out_cat, num_cat_features, category_sizes):
    return torch.tensor(1.0, device=log_x_cat.device)
    
def categorical_nndr_loss(log_x_cat, model_out_cat, num_cat_features, category_sizes, **kwargs):
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
        max_normal_dist = 0.0
        
        # Normalization mask, divide each element by the number of labels of the feature
        mask = torch.ones(x_cat_ohe.shape[1], device=x_cat_ohe.device)
        start_idx = 0
        for size in category_sizes:
            mask[start_idx:start_idx + size] = 1.0 / size
            start_idx += size
            max_normal_dist += 2/size
        mask /= math.sqrt(max_normal_dist)
        mask = mask.unsqueeze(0)
        probabilities = probabilities * mask
        x_cat_ohe = x_cat_ohe * mask
        
        # ----- New approach 11.12.2025 -----
        # Calculate pair-wise distance matrix between input and output categorical features
        distances = torch.cdist(probabilities, x_cat_ohe, p=2)
        nearest_distances = torch.min(distances, dim=1).values
        second_nearest_distances = torch.kthvalue(distances, k=2, dim=1).values

        # Calculate the NNDR for each synthetic record
        nndr_list = nearest_distances / (second_nearest_distances + 1e-16)  # Avoid division by zero

        # Return the average NNDR across all synthetic records
        nndr_ratio = torch.mean(nndr_list)

        return nndr_ratio.mean()
    
        # ----- Old approach -----
    """
        
        # Calculate distance between input and output categorical features
        distances = torch.cdist(probabilities, x_cat_ohe, p=2)
        # For each row in out_num, get the index of the nearest row in x_num
        nearest_idx = torch.argmin(distances, dim=0)
        farthest_idx = torch.argmax(distances, dim=0)

        # Get the actual distances and normalize with max distances
        nearest_distances = distances[nearest_idx/farthest_idx, torch.arange(model_out_cat.shape[0])]
                
        # Divide by sqrt(sum(2/labels_per_feature)) to normalize the maximum distance to 1
        privacy_loss = nearest_distances/ (math.sqrt(max_normal_dist))
        
        return privacy_loss.mean()
    """
    
    
def compute_gower_num(x_num, model_out_num, sigma=0.01, **kwargs):
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
        dist_matrix = D_num_sum / total_features  # Normalize by number of numerical features

        # Hard min vs. soft min for smoother training and convergence
        # dcr_per_sample, _ = torch.min(dist_matrix, dim=1)
        soft_min_per_sample = -sigma * torch.logsumexp(-dist_matrix / sigma, dim=1) 
        soft_min_per_sample = torch.clamp(soft_min_per_sample, min=0.0, max=1.0)
        
        # Take the mean of these "closest" distances
        # final_dcr = torch.mean(dcr_per_sample)
        final_dcr = torch.mean(soft_min_per_sample)
        return final_dcr
        
        

        
def compute_gower_cat(log_x_cat, model_out_cat, num_cat_features, category_sizes, sigma=0.01, **kwargs):
    
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
        
        # Hard min vs. soft min for smoother training and convergence
        # dcr_per_sample, _ = torch.min(G_matrix, dim=1)
        soft_min_per_sample = -sigma * torch.logsumexp(-G_matrix / sigma, dim=1) 
        soft_min_per_sample = torch.clamp(soft_min_per_sample, min=0.0, max=1.0)
        # Take the mean of these "closest" distances
        # final_dcr = torch.mean(dcr_per_sample)
        final_dcr = torch.mean(soft_min_per_sample)
        return final_dcr


def dcr_cat_loss(log_x_cat, model_out_cat, num_cat_features, category_sizes, **kwargs):
    """
    Compute pair-wise distance matrix, take minimum distance per synthetic record to the original, normalize using Exponential Complement.
    Current implementation does not normalized the distance value with category_sizes because this loss function should reflect numerical ressemblance with original data without context. 
    
    log_x_cat: log encoded input one-hot vector for categorical data
    model_out_cat: logits of predicted output one-hot vector for categorical data
    num_cat_features: number of categorical features
    category_sizes: list of label amount per categorical feature
    """
    
    probabilities = F.softmax(model_out_cat, dim=1)  
    x_cat_ohe = torch.exp(log_x_cat)    
    
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
    
    # Find the nearest distances (d1)
    nearest_distances = torch.min(dists, dim=1).values  # shape: (num_synth,)
    
    return torch.mean(nearest_distances)
    
    # ----- Normalize min distances with max distances per row, not required for gradient descent with additive loss terms -----
    """
    d_max_row = torch.max(dists, dim=1).values  # shape: (num_synth,)
    d_max_row = torch.clamp(d_max_row, min=1e-8)
    
    # Row-wise normalized distances: nearest / row_max
    d_norm = nearest_distances / d_max_row
    
    # print(f"dcr_cat_loss at training: {d_norm}")
    return torch.mean(d_norm)
    """

def dcr_num_loss(x_num, model_out_num, **kwargs):
    
    dists = torch.cdist(
        model_out_num, 
        x_num, 
        p=2.0
    )     
    
    # Find the nearest distance (d1), which is the 1st smallest value (k=1)
    # The output is a named tuple (values, indices)
    nearest_distances = torch.min(dists, dim=1).values  # shape: (num_synth,)
    
    return torch.min(nearest_distances)

    # ----- Normalize min distances with max distances per row, not required for gradient descent with additive loss terms -----
    """
    
    # d_max_row[i] = max_j dists[i, j]
    d_max_row = torch.max(dists, dim=1).values  # shape: (num_synth,)
    d_max_row = torch.clamp(d_max_row, min=1e-8)
    
    # Row-wise normalized distances: nearest / row_max
    d_norm = nearest_distances / d_max_row
        
    return torch.mean(d_norm)
    """

NUM_TASK_MAP = {0: dcr_num_loss, 1: numerical_nndr_loss, 2: compute_gower_num}
CAT_TASK_MAP = {0: dcr_cat_loss, 1: categorical_nndr_loss, 2: compute_gower_cat}    


def vector_num_loss(x_num, model_out_num, **kwargs):
    # Iterate through the sorted map to ensure the order [0, 1, 2] is preserved
    losses = [
        NUM_TASK_MAP[i](x_num, model_out_num, **kwargs) 
        for i in sorted(NUM_TASK_MAP.keys())
    ]
    
    return torch.stack(losses)


def vector_cat_loss(log_x_cat, model_out_cat, num_cat_features, category_sizes, **kwargs):
    
    # Iterate through the sorted map to ensure the order [0, 1, 2] is preserved
    losses = [
        CAT_TASK_MAP[i](log_x_cat, model_out_cat, num_cat_features, category_sizes, **kwargs) 
        for i in sorted(CAT_TASK_MAP.keys())
    ]
    
    return torch.stack(losses)


def adaptive_num_loss(x_num, model_out_num, weights, **kwargs):

    results = []
    for i, val in enumerate(weights):
        if val != 0.0:
            # Get the specific function for this index and call it
            result = NUM_TASK_MAP[i](x_num, model_out_num, **kwargs)
            results.append(result)
        else:
            results.append(torch.tensor(0.0, device=x_num.device))
    return torch.stack(results)

def adaptive_cat_loss(log_x_cat, model_out_cat, num_cat_features, category_sizes, weights, **kwargs):

    results = []
    for i, val in enumerate(weights):
        if val != 0.0:
            # Get the specific function for this index and call it
            result = CAT_TASK_MAP[i](log_x_cat, model_out_cat, num_cat_features, category_sizes, **kwargs)
            results.append(result)
        else:
            results.append(torch.tensor(0.0, device=log_x_cat.device))
    return torch.stack(results)


def sum_cat_loss(log_x_cat, model_out_cat, num_cat_features, category_sizes, **kwargs):
    losses = [
        CAT_TASK_MAP[i](log_x_cat, model_out_cat, num_cat_features, category_sizes, **kwargs)
        for i in range(3)
    ]
    l_dcr, l_nndr, l_gower = losses
    
    # Apply your specific formula
    return -torch.exp(-l_dcr) + l_nndr + l_gower


def sum_num_loss(x_num, model_out_num, **kwargs):
    losses = [
        NUM_TASK_MAP[i](x_num, model_out_num, **kwargs)
        for i in range(3)
    ]
    l_dcr, l_nndr, l_gower = losses
    
    # Apply your specific formula
    return -torch.exp(-l_dcr) + l_nndr + l_gower

PRIVACY_FUNCTIONS = {
    # "nndr_cat_loss": categorical_nndr_loss,  # remove categorical nndr loss for better performance and no added loss information
    "nndr_cat_loss": no_categorical_nndr,
    "nndr_num_loss": numerical_nndr_loss,
    "gower_cat_loss": compute_gower_cat,
    "gower_num_loss": compute_gower_num,
    "dcr_cat_loss": dcr_cat_loss,
    "dcr_num_loss": dcr_num_loss,
    "vector_cat_loss": vector_cat_loss,
    "vector_num_loss": vector_num_loss,
    "sum_cat_loss": sum_cat_loss,
    "sum_num_loss": sum_num_loss,
    "adaptive_cat_loss": adaptive_cat_loss,
    "adaptive_num_loss": adaptive_num_loss,
    "adaptive_dcr_cat_loss": dcr_cat_loss,
    "adaptive_dcr_num_loss": dcr_num_loss,
    "adaptive_nndr_cat_loss": categorical_nndr_loss,
    "adaptive_nndr_num_loss": numerical_nndr_loss,
    "adaptive_gower_cat_loss": compute_gower_cat,
    "adaptive_gower_num_loss": compute_gower_num
}
