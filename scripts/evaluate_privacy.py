import tomli
import shutil
import os
import argparse
from enum import Enum
from typing import Union, Dict, Optional
from scipy.spatial import distance
import time
import matplotlib.pyplot as plt
import lib
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.stats import wasserstein_distance



def load_data(real_path, fake_path):
    
    """
    Adopted from tab-ddpm/scripts/resample_privacy.py, which inturns is adapted from https://github.com/Team-TUD/CTAB-GAN/tree/main/model/eval
    
    """
    
    task_type = lib.load_json(real_path + "/info.json")["task_type"]
    X_num_real, X_cat_real, y_real = lib.read_pure_data(real_path, 'train')
    X_num_fake, X_cat_fake, y_fake = lib.read_pure_data(fake_path, 'train')
    target_size = 0
    
    if task_type == 'regression':
        X_num_real = np.concatenate([X_num_real, y_real[:, np.newaxis]], axis=1)
        X_num_fake = np.concatenate([X_num_fake, y_fake[:, np.newaxis]], axis=1)
        target_size = 1
    else:  # classification, binclass or multiclass
        target_size = 2 if task_type == 'binclass' else len(np.unique(y_real))
        if X_cat_fake is None:
            X_cat_real = y_real[:, np.newaxis].astype(int).astype(str)
            X_cat_fake = y_fake[:, np.newaxis].astype(int).astype(str)
            
        else:
            X_cat_real = np.concatenate([X_cat_real, y_real[:, np.newaxis].astype(int).astype(str)], axis=1)
            X_cat_fake = np.concatenate([X_cat_fake, y_fake[:, np.newaxis].astype(int).astype(str)], axis=1)

    if len(y_real) > 50000:
        ixs = np.random.choice(len(y_real), 50000, replace=False)
        X_num_real = X_num_real[ixs]
        X_cat_real = X_cat_real[ixs] if X_cat_real is not None else None
    
    if len(y_fake) > 50000:
        ixs = np.random.choice(len(y_fake), 50000, replace=False)
        X_num_fake = X_num_fake[ixs]
        X_cat_fake = X_cat_fake[ixs] if X_cat_fake is not None else None


    mm = MinMaxScaler().fit(X_num_real)
    X_real = mm.transform(X_num_real)
    X_fake = mm.transform(X_num_fake)
    if X_cat_real is not None:
        ohe = OneHotEncoder().fit(X_cat_real)
        X_cat_real = ohe.transform(X_cat_real) / np.sqrt(2)
        X_cat_fake = ohe.transform(X_cat_fake) / np.sqrt(2)

        X_real = np.concatenate([X_real, X_cat_real.todense()], axis=1)
        X_fake = np.concatenate([X_fake, X_cat_fake.todense()], axis=1)
    
    return X_real, X_fake, target_size, task_type

def compute_gowers_distance(original: np.ndarray, synthetic: np.ndarray, n_num_features: int, task_type: str, category_sizes: list) -> float:
    """
    Computes Gower's distance between original and synthetic datasets.\n
    The average of every pairwise Gower's distance is returned.

    Parameters:
    - original: np.ndarray; The original dataset (samples x features).
    - synthetic: np.ndarray; The synthetic dataset (samples x features).
    - n_num_features: int; The number of numerical features.
    - category_sizes: list; List containing the sizes of each categorical feature.

    Returns:
    - float; The average Gower's distance.
    """
    
    N_o, N_s = original.shape[0], synthetic.shape[0]
    if task_type == 'regression':
            n_num_features += 1  # Include target variable
            category_sizes = category_sizes[:-1]  # Exclude target variable
    
    # Total number of features (used for normalization)
    total_features = n_num_features + len(category_sizes)
    if total_features == 0:
        return 0.0
    
    # --- 1. Numerical Feature Dissimilarity (d_num) ---
    if n_num_features > 0:
        # Extract numerical features
        O_num = original[:, :n_num_features]
        S_num = synthetic[:, :n_num_features]
        
        # Broadcasting: O_num[:, None, :] subtracts S_num[None, :, :]
        # O_num is (N_o, 1, n_num), S_num is (1, N_s, n_num) -> Result is (N_o, N_s, n_num)
        # Assuming data is pre-scaled to [0, 1], the partial dissimilarity d_ijk is simply the absolute difference.
        D_num = np.abs(O_num[:, None, :] - S_num[None, :, :])
        
        # Sum of numerical dissimilarities for each pair (N_o x N_s)
        D_num_sum = np.sum(D_num, axis=2)
    else:
        D_num_sum = np.zeros((N_o, N_s))

    # --- 2. Categorical Feature Dissimilarity (d_cat) ---
    D_cat_sum = np.zeros((N_o, N_s))
    
    if len(category_sizes) > 0:
        runner = n_num_features
        
        for size in category_sizes:
            # Extract one-hot-encoded columns for the current feature
            O_cat_k = original[:, runner : runner + size]
            S_cat_k = synthetic[:, runner : runner + size]
            
            # Broadcasting comparison (N_o x N_s x size)
            O_cat_k_b = O_cat_k[:, None, :]
            S_cat_k_b = S_cat_k[None, :, :]
            
            # Check if OHE vectors are identical for each pair
            is_equal = np.all(O_cat_k_b == S_cat_k_b, axis=2) # Result is (N_o, N_s) Booleans
            
            # Dissimilarity: 0 if equal (True), 1 if different (False)
            D_cat_k = (~is_equal).astype(float)
            
            D_cat_sum += D_cat_k
            runner += size

    # --- 3. Final Gower Distance Calculation ---
    # G_ij = (Sum of D_num) + (Sum of D_cat) / Total Features
    G_matrix = (D_num_sum + D_cat_sum) / total_features
    
    # Return the average of every pairwise Gower's distance
    return np.mean(G_matrix)
            

def compute_dcr(
    original_data: np.ndarray, 
    synthetic_data: np.ndarray,
    num_numerical_features: Optional[int] = None,
    category_sizes: Optional[list] = None,
    distance_metric: str = 'euclidean',
    task_type: str = None
) -> float:
    """
    Computes the Distance to Closest Record (DCR) between the synthetic and 
    original datasets using NumPy and SciPy's distance module. The categorical data are weighted by the number of labels per feature, the input data is already normalized with the maximal euclidean distance per categorical feature sqrt(2).

    DCR is the average minimum distance from each record in the synthetic 
    dataset to its closest record in the original dataset.

    Parameters:
    - original_data: np.ndarray; The original dataset (samples x features). 
                     Assumed to be transformed and normalized.
    - synthetic_data: np.ndarray; The synthetic dataset (samples x features). 
                      Assumed to be transformed and normalized.
    - num_numerical_features: Optional[int] (default: None); The number of 
                             numerical features in the dataset. If None or 
                             invalid, all features are treated equally.
    - category_sizes: Optional[list] (default: None); List containing the sizes 
                      of each categorical feature. 
    - distance_metric: str (default: 'euclidean'); The metric for calculating 
                       distances (e.g., 'euclidean', 'cityblock', 'cosine').
    - weights: Optional[np.ndarray] (default: None); Array of weights for each 
               feature. If None, all features are weighted equally (1.0).

    Returns:
    - float: The average minimum distance (DCR score).
    """
    
    # --- 1. Validate and Prepare Weights ---
    
    n_features = original_data.shape[1]
    if task_type:  # Numerical + Categorical features
        assert n_features ==  num_numerical_features + sum(category_sizes), "Category sizes do not match the number of categorical features."
        
        if task_type == 'regression':
            num_numerical_features += 1  # Include target variable
            category_sizes = category_sizes[:-1]  # Exclude target variable
        
        # Use uniform weights of 1.0
        feature_weights = np.ones(n_features)
        # Assign weight 1/sqrt(2) to non-numerical features
        mask = np.ones_like(feature_weights[num_numerical_features:], dtype=float)
        start_idx = 0
        for size in category_sizes:
            mask[start_idx:start_idx + size] = 1.0 / size
            start_idx += size
        feature_weights[num_numerical_features:] = mask
        print(f"In DCR, Mask with category_sizes: {mask}.")
    else:  # only numerical features
        feature_weights = np.ones(n_features)
        
    

    # --- 2. Apply Feature Weights ---
    
    weighted_original_data = original_data * feature_weights
    weighted_synthetic_data = synthetic_data * feature_weights
    
    # --- 3. Compute Pairwise Distances ---
    # The output 'dists' has shape (num_synthetic_records, num_original_records).
    dists = distance.cdist(
        weighted_synthetic_data, 
        weighted_original_data, 
        metric=distance_metric
    )
    
    # --- 4. Find Minimum Distances ---
    # np.min(dists, axis=1) finds the minimum distance for *each* synthetic record.
    # This gives us the distance to the "Closest Record" in the original dataset.
    min_distances = np.min(dists, axis=1)
    
    # --- 5. Compute Average DCR Score ---
    # The DCR score is the mean of these minimum distances.
    dcr_score = np.mean(min_distances)
    
    return dcr_score

def compute_nndr(
    original_data: np.ndarray, 
    synthetic_data: np.ndarray,
    num_numerical_features: Optional[int] = None,
    category_sizes: Optional[list] = None,
    task_type: str = None,
    distance_metric: str = 'euclidean',
) -> float:
    """
    Calculates the Nearest Neighbor Distance Ratio (NNDR) for synthetic data.

    NNDR is the ratio of the distance to the closest original record (1st NN) 
    to the distance to the second closest original record (2nd NN), averaged 
    over all synthetic records.

    Parameters:
    - original_data: np.ndarray; The original dataset (samples x features). 
                     Assumed to be transformed and normalized.
    - synthetic_data: np.ndarray; The synthetic dataset (samples x features). 
                      Assumed to be transformed and normalized.
    - distance_metric: str (default: 'euclidean'); The metric for calculating 
                       distances (e.g., 'euclidean', 'cityblock', 'cosine').

    Returns:
    - float: The mean NNDR value for the synthetic dataset.
    """
    # --- 0. Validate Inputs and weights ---
    n_features = original_data.shape[1]
    if task_type == 'regression':
            num_numerical_features += 1  # Include target variable
            category_sizes = category_sizes[:-1]  # Exclude target variable
    if num_numerical_features is None or num_numerical_features < 0 or num_numerical_features > n_features:
        # Use uniform weights of 1.0
        feature_weights = np.ones(n_features)
    else:
        feature_weights = np.ones(n_features)
        mask = np.ones_like(feature_weights[num_numerical_features:], dtype=float)
        start_idx = 0
        for size in category_sizes:
            mask[start_idx:start_idx + size] = 1.0 / size
            start_idx += size
        feature_weights[num_numerical_features:] = mask
        print(f"In NNDR, Mask with category_sizes: {mask}.")
        feature_weights = feature_weights

    # --- Apply Feature Weights ---
    
    weighted_original_data = original_data * feature_weights
    weighted_synthetic_data = synthetic_data * feature_weights
    
    
    # --- Compute Pairwise Distances ---
    # Calculates the distance from every synthetic record to every original record.
    # The output 'distances' has shape (num_synthetic_records, num_original_records).
    distances = distance.cdist(
        weighted_synthetic_data, 
        weighted_original_data, 
        metric=distance_metric
    )
    
    # --- Find Nearest and Second Nearest Distances ---
    # np.partition efficiently finds the k smallest values (k=2 here).
    # We partition around the index 1 (meaning the smallest 2 elements are moved to 
    # the start of the array slice).
    partitioned_distances = np.partition(distances, 1, axis=1)[:, :2]
    
    # The 0th column is the nearest distance (d1)
    nearest_distances = partitioned_distances[:, 0]
    # The 1st column is the second nearest distance (d2)
    second_nearest_distances = partitioned_distances[:, 1]
    
    # --- Calculate NNDR ---
    # NNDR = d1 / d2. The small epsilon (1e-16) prevents division by zero 
    # in cases where d2 might be exactly zero (i.e., multiple exact matches).
    nndr_list = nearest_distances / (second_nearest_distances + 1e-16)
    
    # --- Return Mean NNDR ---
    # The final score is the average NNDR across all synthetic records.
    return np.mean(nndr_list)

class CorrelationMethod(Enum):
    # Enum for defining correlation methods
    PEARSON = "pearson"  # Pearson correlation method (standard)
    SPEARMAN = "spearman"  # Spearman correlation method (rank-based)

def correlation_similarity(
    original_data: np.ndarray, 
    synthetic_data: np.ndarray, 
    method: CorrelationMethod = CorrelationMethod.PEARSON
) -> float:
    """
    Evaluate the similarity between the correlation matrices of the original and 
    synthetic datasets, both provided as NumPy arrays.

    The input NumPy arrays are assumed to contain all features (numerical and 
    one-hot encoded categorical) and are ready for correlation calculation.

    Parameters:
    - original_data: np.ndarray; the original dataset (features x samples).
    - synthetic_data: np.ndarray; the synthetic dataset (features x samples).
    - method: CorrelationMethod (default: CorrelationMethod.PEARSON); 
              the correlation method to apply (Pearson or Spearman).

    Returns:
    - float; the average similarity score between the original and synthetic 
      correlation matrices.
    """
    
    # --- 1. Calculate Correlation Matrices ---
    # np.corrcoef calculates Pearson correlation.
    # For Spearman, we first need to convert the data to ranks.

    if method == CorrelationMethod.PEARSON:
        # np.corrcoef calculates the Pearson correlation coefficient matrix.
        # It expects observations as rows, but often correlation is calculated 
        # between features (columns), so we use the transpose if needed.
        # Assuming features are columns in the input arrays:
        orig_corr = np.corrcoef(original_data, rowvar=False)
        print("Synthetic correlation matrix calculation.")
        syn_corr = np.corrcoef(synthetic_data, rowvar=False)
        print(syn_corr)
        print("Finished correlation matrix calculation.")

    elif method == CorrelationMethod.SPEARMAN:
        # Spearman correlation is Pearson correlation on the ranked data.
        
        # Calculate ranks (rankdata requires data to be rankable, handles ties)
        # Assuming features are columns, ranking is done column-wise (axis=0)
        orig_ranked = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)) + 1, 0, original_data)
        syn_ranked = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)) + 1, 0, synthetic_data)

        # Calculate Pearson correlation on the ranks
        orig_corr = np.corrcoef(orig_ranked, rowvar=False)
        syn_corr = np.corrcoef(syn_ranked, rowvar=False)
        
    else:
        raise ValueError("Invalid correlation method specified.")

    # --- 2. Flatten Matrices ---
    # Flatten both matrices into 1D arrays for element-wise comparison.
    orig_corr_flat = orig_corr.flatten()
    syn_corr_flat = syn_corr.flatten()
    
    # --- 3. Clean NaN Values ---
    diff_array = np.abs(syn_corr_flat - orig_corr_flat)
    non_nan_mask = ~np.isnan(diff_array)
    cleaned_diff_array = diff_array[non_nan_mask]

    # --- 4. Calculate Similarity Score ---
    # Calculate the mean absolute difference (MAD) between the flattened correlations.
    mad = np.mean(cleaned_diff_array)

    # The original class formula (1 - MAD / 2) scales the difference:
    # Max possible difference is 2 (e.g., corr=1 vs corr=-1).
    # This scales the MAD from [0, 2] to a score from [0, 1].
    score = 1 - (mad / 2)
    
    print(f"Method {method.name} was used.")
    return score
    

def compute_wasserstein_distance(original: np.ndarray, synthetic: np.ndarray) -> float:
    """
    Calculate the average 1D Wasserstein distance across all features
    between the original and synthetic datasets.

    Args:
        original: NumPy array of original numerical data (N_orig, F).
        synthetic: NumPy array of synthetic numerical data (N_synth, F).

    Returns:
        Mean Wasserstein distance across all features.
    """
    if original.size == 0 or synthetic.size == 0:
        return 0.0

    if original.shape[1] != synthetic.shape[1]:
        print("Warning: Feature counts do not match. Skipping W1 distance.")
        return 0.0

    w_distances = []
    num_features = original.shape[1]

    for i in range(num_features):
        dist = wasserstein_distance(original[:, i], synthetic[:, i])
        w_distances.append(dist)

    return float(np.mean(w_distances))

def compute_js_similarity(original_data: np.ndarray, synthetic_data: np.ndarray) -> float:
    """
    Computes the average Jensen-Shannon similarity across all features (columns)
    between original and synthetic datasets, assuming one-hot encoding.

    In one-hot encoding, the probability distribution for a category column is
    the proportion of '1's vs '0's (occurrence vs non-occurrence).

    Parameters:
    - original_data: np.ndarray; the real dataset (N_real_samples x N_features).
    - synthetic_data: np.ndarray; the synthetic dataset (N_syn_samples x N_features).

    Returns:
    - float; the average Jensen-Shannon similarity score (0 to 1) across all features.
    """
    # 1. Ensure the number of features (columns) is the same
    if original_data.shape[1] != synthetic_data.shape[1]:
        raise ValueError("The number of columns (features) must be the same in both arrays.")

    num_features = original_data.shape[1]
    js_similarities = []
    
    # Small epsilon to avoid log(0) and division by zero issues
    epsilon = 1e-10 

    # 2. Iterate through each feature column
    for col_idx in range(num_features):
        orig_col = original_data[:, col_idx]
        syn_col = synthetic_data[:, col_idx]

        # 3. Compute the Empirical Probability Distributions (PMFs)
        # Since it's one-hot encoded (binary 0 or 1), the PMF has two points:
        # P(0) = proportion of 0s, P(1) = proportion of 1s.
        
        # Calculate P(1) (proportion of occurrences)
        p_1_orig = np.mean(orig_col)
        p_1_syn = np.mean(syn_col)
        
        # Calculate P(0) (proportion of non-occurrences)
        p_0_orig = 1.0 - p_1_orig
        p_0_syn = 1.0 - p_1_syn
        
        # Create the Probability Mass Functions (PMFs)
        # Add epsilon to prevent issues with log(0) in jensenshannon
        pmf_orig = np.array([p_0_orig, p_1_orig])
        pmf_syn = np.array([p_0_syn, p_1_syn])
        
        # Ensure no exact zeros before calculation
        pmf_orig = np.where(pmf_orig == 0, epsilon, pmf_orig)
        pmf_syn = np.where(pmf_syn == 0, epsilon, pmf_syn)
        
        # Re-normalize after adding epsilon if needed, though often skipped
        pmf_orig /= pmf_orig.sum()
        pmf_syn /= pmf_syn.sum()

        # 4. Compute Jensen-Shannon Distance
        # jensenshannon returns the square root of the JSD (i.e., the JS distance)
        js_distance = distance.jensenshannon(pmf_orig, pmf_syn, base=2)

        # 5. Convert Distance to Similarity (where 1 is identical)
        # The JS distance is bounded between 0 and 1 when using base=2.
        js_similarity = 1.0 - js_distance
        js_similarities.append(js_similarity)

    # 6. Return the average similarity across all features
    return np.mean(js_similarities)


def compute_basic_stats(original: np.ndarray, synthetic: np.ndarray):
    """
    Compute mean, median, and variance for each column (feature)
    in both original and synthetic datasets using NumPy.
    """
    # PyTorch's dim=0 (rows) corresponds to NumPy's axis=0 (rows)
    # The statistics are computed across rows, resulting in one value per column (feature).

    # Mean (Average)
    orig_mean = original.mean(axis=0)
    syn_mean = synthetic.mean(axis=0)

    # Median (Middle value)
    # np.median returns the median value directly for the specified axis.
    orig_median = np.median(original, axis=0)
    syn_median = np.median(synthetic, axis=0)
    
    # Variance (Spread)
    # PyTorch's var(unbiased=False) is equivalent to NumPy's var(ddof=0).
    # ddof=0 means divisor is N (population variance).
    orig_var = original.var(axis=0, ddof=0)
    syn_var = synthetic.var(axis=0, ddof=0)

    stats = {
        "orig_mean": orig_mean,
        "syn_mean": syn_mean,
        "orig_median": orig_median,
        "syn_median": syn_median,
        "orig_var": orig_var,
        "syn_var": syn_var,
    }
    
    return stats

def calculate_score(stats: Dict[str, Union[np.ndarray, float]], stat_type: str) -> float:
    """
    Calculate average absolute difference between original and synthetic
    statistics for a given stat_type ('mean', 'median', 'var') using NumPy.
    
    The 'stats' dictionary is expected to contain NumPy arrays (e.g., 'orig_mean').
    """
    
    if stat_type == 'mean':
        # Absolute difference between synthetic mean and original mean
        diffs = np.abs(stats["syn_mean"] - stats["orig_mean"])
    elif stat_type == 'median':
        # Absolute difference between synthetic median and original median
        diffs = np.abs(stats["syn_median"] - stats["orig_median"])
    elif stat_type == 'var':
        # Absolute difference between synthetic variance and original variance
        diffs = np.abs(stats["syn_var"] - stats["orig_var"])
    else:
        raise ValueError(f"Invalid stat_type: {stat_type}")

    # Calculate the mean of all differences across all features, 
    # and use .item() to return a standard Python float.
    return diffs.mean().item()


def evaluate_generation(original: np.ndarray, synthetic: np.ndarray, num_numerical_features: int, category_sizes: list = None, task_type: str = None):
    """
    Evaluate similarity between original and synthetic data
    based on mean absolute differences in mean, median, and variance.
    
    Args:
        original: original dataset
        synthetic: fake, synthetic dataset
        N: number of numerical features
    """
    basic_stats = compute_basic_stats(original, synthetic)
    scores = {}
    for stat_type in ['mean', 'median', 'var']:
        scores[stat_type] = calculate_score(basic_stats, stat_type)
    stats = {}
    stats["wasserstein_distance_numerical"] = compute_wasserstein_distance(np.asarray(original[:,:num_numerical_features]), np.asarray(synthetic[:,:num_numerical_features]))
    stats["js_similarity_categorical"] = compute_js_similarity(np.asarray(original[:, num_numerical_features:]), np.asarray(synthetic[:, num_numerical_features:]))
    stats["correlation_pearson"] = correlation_similarity(np.asarray(original), np.asarray(synthetic), CorrelationMethod.PEARSON)
    stats["correlation_spearman"] = correlation_similarity(np.asarray(original), np.asarray(synthetic), CorrelationMethod.SPEARMAN)
    print("Finished computing correlation similarities.")
    stats["dcr_all_euclidean"] = compute_dcr(np.asarray(original), np.asarray(synthetic), num_numerical_features=num_numerical_features, category_sizes=category_sizes, task_type=task_type, distance_metric='euclidean')
    print("Finished computing DCR all.")
    stats["dcr_numerical_euclidean"] = compute_dcr(np.asarray(original[:,:num_numerical_features]), np.asarray(synthetic[:,:num_numerical_features]), num_numerical_features=num_numerical_features, category_sizes=category_sizes, task_type=task_type, distance_metric='euclidean')
    print("Finished computing DCR numerical.")
    stats["dcr_categorical_hamming"] = compute_dcr(np.asarray(original[:,num_numerical_features:]), np.asarray(synthetic[:,num_numerical_features:]), num_numerical_features=num_numerical_features, category_sizes=category_sizes, task_type=task_type, distance_metric='hamming')
    print("Finished computing DCR categorical.")
    stats["nndr_euclidean"] = compute_nndr(np.asarray(original), np.asarray(synthetic), num_numerical_features=num_numerical_features, category_sizes=category_sizes, task_type=task_type, distance_metric='euclidean')
    print("Finished computing NNDR.")
    stats["nndr_cosine"] = compute_nndr(np.asarray(original), np.asarray(synthetic), distance_metric='cosine')
    print("Starting to compute Gower's distance.")
    stats["gowers_distance"] = compute_gowers_distance(np.asarray(original), np.asarray(synthetic), num_numerical_features, task_type, category_sizes)
    print("Finished computing Gower's distance.")
    
    return stats, scores


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--start_privacy_step', action='store_true',  default=-1)
    parser.add_argument('--sample', action='store_true',  default=False)
    parser.add_argument('--eval', action='store_true',  default=False)
    parser.add_argument('--change_val', action='store_true',  default=False)

    args = parser.parse_args()
    raw_config = lib.load_config(args.config)
    N = raw_config['num_numerical_features']
    x_real, x_fake, target_size, task_type = load_data(raw_config['real_data_path'], raw_config['parent_dir'])
    dataset_info = lib.load_json(os.path.join(raw_config['parent_dir'], 'DatasetInfo.json'))
    dataset_info["category_sizes"].append(target_size)
    stats, scores = evaluate_generation(x_real, x_fake, N, dataset_info["category_sizes"], task_type=task_type)
    
    with open(os.path.join(raw_config['parent_dir'], raw_config['evaluation_file']), 'w') as file:
        
        file.write(f"Similarity and Privacy evaluation\n")
        for key, value in stats.items():
            file.write(f"{key}: {value}\n")
        file.write(f"\nAbsolute Difference of Basic Statistics:\n")
        for key, value in scores.items():
            file.write(f"{key}: {value}\n")
        file.write("Statistics computed column-wise, then average is taken.\n")    
        
    
if __name__ == '__main__':
    main()