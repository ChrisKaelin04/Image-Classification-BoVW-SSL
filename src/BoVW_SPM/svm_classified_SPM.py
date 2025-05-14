import numpy as np
import os
import pickle
import warnings
import joblib
import h5py # For loading HOG data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler # <-- Import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import traceback
import gc

# --- Configuration ---
FEATURES_DIR_SPM = r"E:\CV_features_SPM"
BOVW_SPM_FEATURES_DIR = os.path.join(FEATURES_DIR_SPM, "bovw_spm_features_4cat")
HOG_DATA_FILE = os.path.join(FEATURES_DIR_SPM, 'hog_data_spm.h5')
SPLITS_DIR_COMMON = os.path.join(r"E:\CV_features", "train_test_splits_4cat_revised")
NPZ_FILE = os.path.join(SPLITS_DIR_COMMON, "train_test_split_data_4cat_revised.npz")
LABEL_ENCODER_FILE = os.path.join(SPLITS_DIR_COMMON, "broad_label_encoder_4cat_revised.pkl")
RESULTS_DIR_XGB_SPM = os.path.join(FEATURES_DIR_SPM, "classification_results_XGB_SPM_SOH_4cat")
os.makedirs(RESULTS_DIR_XGB_SPM, exist_ok=True)
DMATRIX_CACHE_DIR = os.path.join(FEATURES_DIR_SPM, "xgb_dmatrix_cache_4cat")
os.makedirs(DMATRIX_CACHE_DIR, exist_ok=True)
VOCAB_SIZE = 1000
PYRAMID_LEVELS = 3

# --- Hyperparameters ---
XGB_BASE_PARAMS = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'tree_method': 'hist',
    'random_state': 42,
    'use_label_encoder': False
}
PARAM_GRID_XGB = {
    'n_estimators': [300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [7],
}
GRIDSEARCH_CV_FOLDS = 3
SAMPLE_FRACTION_FOR_GRIDSEARCH = 0.25 # (if memory allows)
# Change scoring metric for GridSearchCV to handle imbalance
GRIDSEARCH_SCORING = 'f1_macro' # Or 'recall_macro'

warnings.filterwarnings("ignore", message="Parameters: {.*use_label_encoder.*} are not used.", category=UserWarning, module="xgboost.core")
warnings.filterwarnings("ignore", message="omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.", category=UserWarning)


# --- 1. Load Labels, Indices, and Label Encoder ---
print("--- Loading Common Data (Labels, Splits, Encoder) ---")
print(f"Loading train/test split data from: {NPZ_FILE}")
try:
    split_data = np.load(NPZ_FILE)
    train_indices_full = split_data['train_indices']
    test_indices_full = split_data['test_indices']
    y_train_full = split_data['train_labels_numeric']
    y_test_full = split_data['test_labels_numeric']
except FileNotFoundError:
    print(f"ERROR: NPZ file not found at {NPZ_FILE}. Ensure label splitting script has run.")
    exit()
except KeyError as e:
    print(f"ERROR: Missing key {e} in NPZ file {NPZ_FILE}. Check keys.")
    exit()
print(f"Loaded {len(train_indices_full)} total train indices and {len(y_train_full)} labels.")
print(f"Loaded {len(test_indices_full)} total test indices and {len(y_test_full)} labels.")
if len(train_indices_full) != len(y_train_full) or len(test_indices_full) != len(y_test_full):
    print("ERROR: Mismatch between number of indices and labels. Halting.")
    exit()
print(f"Loading label encoder from: {LABEL_ENCODER_FILE}")
try:
    with open(LABEL_ENCODER_FILE, 'rb') as f:
        label_encoder = pickle.load(f)
    class_names = label_encoder.classes_
    NUM_CLASSES = len(class_names)
    XGB_BASE_PARAMS['num_class'] = NUM_CLASSES
    print(f"Class names for classification: {class_names} ({NUM_CLASSES} classes)")
    if len(class_names) != 4: # Assuming 4 broad categories
        print(f"Warning: Expected 4 class names, got {len(class_names)}.")
except FileNotFoundError:
    print(f"ERROR: Label encoder file not found at {LABEL_ENCODER_FILE}.")
    exit()

# --- Check Class Distribution ---
print("\n--- Class Distribution Check ---")
unique_train, counts_train = np.unique(y_train_full, return_counts=True)
print("Training set class distribution:")
for label, count in zip(unique_train, counts_train):
    print(f"  Class {label} ({class_names[label]}): {count} samples ({count/len(y_train_full):.2%})")
unique_test, counts_test = np.unique(y_test_full, return_counts=True)
print("Test set class distribution:")
for label, count in zip(unique_test, counts_test):
    print(f"  Class {label} ({class_names[label]}): {count} samples ({count/len(y_test_full):.2%})")


# --- 2. Helper Functions ---
def plot_confusion_matrix(cm, classes, plot_title='Confusion matrix', cmap=plt.cm.Blues, results_path=None, filename=None):
    plt.figure(figsize=(max(8, len(classes)), max(6, len(classes)*0.8)))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(plot_title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if results_path and filename:
        full_path = os.path.join(results_path, filename)
        plt.savefig(full_path)
        print(f"Saved confusion matrix to {full_path}")
    plt.close()


# --- Feature Loading ---
def load_spm_features(spm_bovw_dir, feature_name, pyramid_levels_count, set_indices, is_train_set):
    """Loads SPM features corresponding to the given indices."""
    max_level_index = pyramid_levels_count - 1
    set_type = "train" if is_train_set else "test"
    filename = f"X_{set_type}_{feature_name}_spm_L{max_level_index}.npy"
    filepath = os.path.join(spm_bovw_dir, filename)
    if os.path.exists(filepath):
        print(f"Loading ALL {set_type} {feature_name} SPM (L{max_level_index}) features from: {filepath}")
        all_data = np.load(filepath)
        print(f"  Full shape: {all_data.shape}")

        # --- Logic to select correct rows based on indices ---
        # We need the *original* indices of the full loaded data to map correctly.
        # This assumes the order in the .npy file matches the order in train_indices_full/test_indices_full.
        # A safer way would be to save indices alongside the features in the .npy, but let's assume order matches splits.
        full_indices_in_npy_order = train_indices_full if is_train_set else test_indices_full

        if all_data.shape[0] == len(full_indices_in_npy_order):
            if len(set_indices) == len(full_indices_in_npy_order): # Requesting the full set
                 print("  Using full set data.")
                 return all_data
            else: # Requesting a subset (e.g., for sampling)
                 print(f"  Sub-selecting {len(set_indices)} rows for the current request.")
                 try:
                      # Create a mapping from the original index (as in splits) to the row index in the NPY file
                      map_original_idx_to_row_in_npy = {idx: i for i, idx in enumerate(full_indices_in_npy_order)}
                      
                      selected_row_indices_in_npy = [map_original_idx_to_row_in_npy[idx] for idx in set_indices if idx in map_original_idx_to_row_in_npy]

                      if len(selected_row_indices_in_npy) != len(set_indices):
                           print(f"  Warning: Could only find {len(selected_row_indices_in_npy)} out of {len(set_indices)} requested indices in the loaded NPY data's implied map.")
                      
                      # Ensure the selected rows are returned in the order of set_indices
                      # Need to sort the selected row indices according to the order of set_indices
                      # Create a mapping from set_indices elements back to their desired order
                      desired_order_map = {idx: i for i, idx in enumerate(set_indices)}
                      # Get the data for the selected rows
                      selected_data = all_data[selected_row_indices_in_npy, :]
                      # Get the original indices corresponding to the selected data rows
                      original_indices_of_selected_data = [full_indices_in_npy_order[i] for i in selected_row_indices_in_npy]
                      # Create a list of tuples (desired_order, data_row) and sort
                      sorted_data_with_order = sorted(zip([desired_order_map[idx] for idx in original_indices_of_selected_data], selected_data), key=lambda item: item[0])
                      # Extract the data in the desired order
                      return np.array([data_row for _, data_row in sorted_data_with_order])


                 except Exception as e:
                      print(f"ERROR during sub-selection based on indices: {e}. Indices might not match NPY structure or mapping logic has an issue.")
                      # print(traceback.format_exc()) # Optional detailed traceback
                      return None
        else:
             print(f"ERROR: Loaded NPY data rows ({all_data.shape[0]}) does not match expected full split size ({len(full_indices_in_npy_order)}). Cannot reliably select rows.")
             return None
    else:
        print(f"Warning: {feature_name} SPM (L{max_level_index}) file not found: {filepath}")
        return None

def load_and_align_global_hog(hog_h5_filepath, target_indices_for_set):
    """Loads and aligns HOG features corresponding to the target indices."""
    if not os.path.exists(hog_h5_filepath):
        print(f"Warning: Global HOG data file not found: {hog_h5_filepath}")
        return None
    print(f"Loading global HOG features from: {hog_h5_filepath} for {len(target_indices_for_set)} indices.")
    try:
        with h5py.File(hog_h5_filepath, 'r') as hf:
            if 'hog_features' not in hf or 'indices' not in hf:
                print(f"ERROR: 'hog_features' or 'indices' not found in HDF5 file: {hog_h5_filepath}")
                return None
            all_hog_features = hf['hog_features'][:]
            all_hog_original_indices = hf['indices'][:]
    except Exception as e:
        print(f"Error loading HOG data from {hog_h5_filepath}: {e}")
        return None

    if all_hog_features.size == 0 or all_hog_original_indices.size == 0:
        print(f"Warning: HOG features or indices in {hog_h5_filepath} are empty.")
        return np.empty((len(target_indices_for_set), 0), dtype=np.float32)

    # Ensure HOG features are 2D (samples x features)
    if all_hog_features.ndim == 1:
        if all_hog_original_indices.ndim == 1 and all_hog_original_indices.shape[0] > 0 and all_hog_features.shape[0] % all_hog_original_indices.shape[0] == 0:
            expected_dim = all_hog_features.shape[0] // all_hog_original_indices.shape[0]
            print(f"  Reshaping 1D HOG features into ({all_hog_original_indices.shape[0]}, {expected_dim})")
            all_hog_features = all_hog_features.reshape(all_hog_original_indices.shape[0], expected_dim)
        else:
            print(f"ERROR: Cannot safely reshape 1D HOG features. Indices count {all_hog_original_indices.shape[0]}, Feature len {all_hog_features.shape[0]}")
            return None
    elif all_hog_features.ndim != 2:
         print(f"ERROR: HOG features are not 2-dimensional (shape: {all_hog_features.shape})")
         return None

    if all_hog_features.shape[0] != all_hog_original_indices.shape[0]:
        print(f"ERROR: Mismatch between HOG features ({all_hog_features.shape[0]}) and indices ({all_hog_original_indices.shape[0]})")
        return None

    hog_feature_dim = all_hog_features.shape[1]
    print(f"  Found {all_hog_features.shape[0]} total HOG features with dimension {hog_feature_dim}.")

    try:
       all_hog_original_indices_int = [int(i) for i in all_hog_original_indices]
       hog_feature_map = {original_idx: i for i, original_idx in enumerate(all_hog_original_indices_int)}
    except (ValueError, TypeError) as e:
        print(f"ERROR: HOG original indices seem to be of an invalid type: {e}")
        return None

    aligned_hog_list = []
    missing_count = 0
    placeholder = np.zeros(hog_feature_dim, dtype=all_hog_features.dtype)

    # Build a list of (desired_order_index, actual_feature_row)
    features_to_sort = []
    target_indices_map = {idx: i for i, idx in enumerate(target_indices_for_set)} # Map original index to desired order

    for original_idx in target_indices_for_set:
        map_index = hog_feature_map.get(int(original_idx))
        if map_index is not None:
            # Store tuple (desired order index, actual feature row from loaded data)
            features_to_sort.append((target_indices_map[original_idx], all_hog_features[map_index]))
        else:
            # Store tuple (desired order index, placeholder)
            features_to_sort.append((target_indices_map[original_idx], placeholder))
            missing_count += 1

    # Sort the list based on the desired order index
    sorted_aligned_hog_list = [feature for _, feature in sorted(features_to_sort, key=lambda item: item[0])]


    if missing_count > 0:
        print(f"  Warning: {missing_count}/{len(target_indices_for_set)} HOG features for the target indices not found. Used zero vectors.")

    if not sorted_aligned_hog_list:
        print("  No target indices provided, returning empty HOG array.")
        return np.empty((0, hog_feature_dim), dtype=np.float32)

    try:
        aligned_hog_array = np.vstack(sorted_aligned_hog_list)
    except ValueError as e:
         print(f"ERROR: Could not stack aligned HOG features: {e}")
         return None

    print(f"  Aligned global HOG shape for target set: {aligned_hog_array.shape}")
    return aligned_hog_array


# --- 3. DMatrix Creation Function (Updated for Scaling and Weights) ---
def create_xgb_dmatrix_files(feature_combinations, set_indices, set_labels, is_train_set, output_dir,
                             sample_weights=None, perform_scaling=False, train_scaler_path=None): # Added scaling params
    """
    Loads features, concatenates, applies scaling (if requested), and saves DMatrix buffer.
    Can optionally include sample_weights for the training set.
    Handles fitting/saving scaler for train, loading/transforming for test.
    """
    set_name = "train" if is_train_set else "test"
    feature_desc = "_".join(feature_combinations)
    # Add suffixes to filenames
    filename_base = f"{set_name}_{feature_desc}"
    if is_train_set and sample_weights is not None:
         filename_base += "_weighted"
    if perform_scaling:
         filename_base += "_scaled"

    dmatrix_filename = os.path.join(output_dir, f"{filename_base}.buffer")
    scaler_filename = os.path.join(output_dir, f"train_{feature_desc}_scaler.joblib") # Scaler name is specific to train data & features

    print(f"\n--- Creating DMatrix for: {feature_desc} ({set_name}) ---")
    print(f"Target DMatrix file: {dmatrix_filename}")

    if os.path.exists(dmatrix_filename):
        print(f"DMatrix file already exists. Skipping creation.")
        # Check if scaler file also exists if scaling was requested for train
        if is_train_set and perform_scaling and not os.path.exists(scaler_filename):
             print(f"WARNING: DMatrix exists but scaler file {scaler_filename} is missing.")
        return dmatrix_filename

    # --- Load Features ---
    loaded_features = []
    target_labels_for_shape_check = set_labels # Use this to verify number of samples loaded

    print("Loading features for concatenation...")
    if "sift_spm" in feature_combinations:
        sift_spm = load_spm_features(BOVW_SPM_FEATURES_DIR, "sift", PYRAMID_LEVELS, set_indices, is_train_set)
        if sift_spm is None or sift_spm.shape[0] != len(target_labels_for_shape_check):
             print(f"ERROR: Failed to load or incorrect shape for SIFT SPM features for {set_name}. Expected {len(target_labels_for_shape_check)} samples, got {sift_spm.shape[0] if sift_spm is not None else 'None'}.")
             return None
        loaded_features.append(sift_spm)

    if "orb_spm" in feature_combinations:
        orb_spm = load_spm_features(BOVW_SPM_FEATURES_DIR, "orb", PYRAMID_LEVELS, set_indices, is_train_set)
        if orb_spm is None or orb_spm.shape[0] != len(target_labels_for_shape_check):
             print(f"ERROR: Failed to load or incorrect shape for ORB SPM features for {set_name}. Expected {len(target_labels_for_shape_check)} samples, got {orb_spm.shape[0] if orb_spm is not None else 'None'}.")
             return None
        loaded_features.append(orb_spm)

    if "hog" in feature_combinations:
        hog = load_and_align_global_hog(HOG_DATA_FILE, set_indices)
        # Note: HOG loading already prints its shape and checks against target_indices_for_set size
        # Additional check if load_and_align_global_hog failed internally
        if hog is None or hog.shape[0] != len(target_labels_for_shape_check):
            print(f"ERROR: Failed to load or incorrect shape for HOG features for {set_name}. Expected {len(target_labels_for_shape_check)} samples, got {hog.shape[0] if hog is not None else 'None'}.")
            return None
        # Handle potential empty HOG feature dim if all images failed HOG or H5 was empty
        if hog.shape[1] == 0:
             print("Warning: HOG features have zero dimension. Skipping HOG concatenation.")
        else:
             loaded_features.append(hog)


    if not loaded_features:
        print("ERROR: No valid features were specified or loaded for concatenation.")
        return None

    print("Concatenating features...")
    try:
        # Check if any loaded features are empty (e.g., zero dimension HOG included)
        loaded_features = [f for f in loaded_features if f.shape[1] > 0]
        if not loaded_features:
             print("ERROR: All loaded features had zero dimension after filtering. Cannot concatenate.")
             return None

        if len(loaded_features) == 1:
            X_combined = loaded_features[0]
        else:
            # Ensure all features have the same number of samples
            ref_shape = loaded_features[0].shape[0]
            if not all(f.shape[0] == ref_shape for f in loaded_features):
                print("ERROR: Mismatched number of samples in features to concatenate:")
                for i, f in enumerate(loaded_features): print(f"  Feature {i}: {f.shape}")
                del loaded_features
                gc.collect()
                return None
            X_combined = np.concatenate(loaded_features, axis=1)

        print(f"  Combined feature shape: {X_combined.shape}")
        if X_combined.shape[0] != len(set_labels):
             print(f"ERROR: Final combined features shape ({X_combined.shape[0]}) doesn't match label count ({len(set_labels)}).")
             del loaded_features, X_combined
             gc.collect()
             return None

    except MemoryError:
        print("ERROR: Ran out of memory during feature concatenation.")
        del loaded_features
        if 'X_combined' in locals(): del X_combined
        gc.collect()
        print("Suggestion: If concatenation fails, consider processing data in smaller chunks or using np.memmap.")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error during concatenation: {e}")
        # print(traceback.format_exc()) # Optional detailed traceback
        if 'loaded_features' in locals(): del loaded_features
        if 'X_combined' in locals(): del X_combined
        gc.collect()
        return None

    # --- Apply Scaling ---
    if perform_scaling:
        print(f"Applying StandardScaler to features for {set_name}...")
        try:
            if is_train_set:
                 scaler = StandardScaler()
                 X_combined_scaled = scaler.fit_transform(X_combined)
                 print(f"  Fitted and transformed training data. Saving scaler to {scaler_filename}")
                 joblib.dump(scaler, scaler_filename)
            else: # is_test_set
                 if train_scaler_path and os.path.exists(train_scaler_path):
                     print(f"  Loading scaler from {train_scaler_path} and transforming test data.")
                     scaler = joblib.load(train_scaler_path)
                     X_combined_scaled = scaler.transform(X_combined)
                 else:
                     print(f"WARNING: Scaling requested for test set ({set_name}), but train_scaler_path was not provided or file not found: {train_scaler_path}. Skipping scaling for test set.")
                     X_combined_scaled = X_combined # Use unscaled data
                     perform_scaling = False # Update flag locally for logging filename
            
            X_combined = X_combined_scaled # Use the scaled data
            print(f"  Scaled feature shape: {X_combined.shape}")

        except Exception as e:
            print(f"ERROR during scaling: {e}. Proceeding with UNscaled data.")
            # print(traceback.format_exc()) # Optional detailed traceback
            perform_scaling = False # Ensure we don't try to save/use scaled DMatrix suffix
            # X_combined remains unscaled

    # --- Create and Save DMatrix ---
    print("Creating XGBoost DMatrix...")
    try:
        # Ensure X_combined is C-contiguous, especially after scaling
        if not X_combined.flags['C_CONTIGUOUS']:
             X_combined = np.ascontiguousarray(X_combined)

        # Pass weights if provided (only for train DMatrix)
        dmatrix = xgb.DMatrix(X_combined, label=set_labels.astype(np.float32), weight=sample_weights if is_train_set else None)

        print("Saving DMatrix to buffer file...")
        # Reconstruct filename base to reflect if scaling actually happened
        filename_base_final = f"{set_name}_{feature_desc}"
        if is_train_set and sample_weights is not None:
            filename_base_final += "_weighted"
        if perform_scaling: # Check the *final* state of perform_scaling
            filename_base_final += "_scaled"
        dmatrix_filename_final = os.path.join(output_dir, f"{filename_base_final}.buffer")

        dmatrix.save_binary(dmatrix_filename_final)
        print(f"Successfully saved DMatrix to {dmatrix_filename_final}")

        del X_combined, loaded_features # Explicit cleanup
        if 'scaler' in locals(): del scaler
        if 'X_combined_scaled' in locals(): del X_combined_scaled
        if 'dmatrix' in locals(): del dmatrix
        gc.collect()

        # Return the final path where the DMatrix was saved
        return dmatrix_filename_final

    except Exception as e:
        print(f"ERROR: Failed to create or save DMatrix: {e}")
        # print(traceback.format_exc()) # Optional detailed traceback
        # Clean up any variables that might exist after error
        if 'X_combined' in locals(): del X_combined
        if 'loaded_features' in locals(): del loaded_features
        if 'scaler' in locals(): del scaler
        if 'X_combined_scaled' in locals(): del X_combined_scaled
        if 'dmatrix' in locals(): del dmatrix
        gc.collect()
        return None

# --- 4. Modified Training Functions (Updated for Weights and Scoring) ---

def find_best_params_with_gridsearch_on_sample(
    X_train_sample_unscaled, y_train_sample, num_classes, # Now explicitly indicate unscaled input
    base_params, param_grid, cv_folds, feature_type_desc, scoring_metric):
    """
    Performs GridSearchCV on a *scaled* sample, attempting GPU first, optimizing for scoring_metric.
    Includes sample weighting based on sample class distribution.
    A NEW scaler is fitted and applied *within* this function for the sample.
    """
    print(f"\n--- Performing GridSearchCV on SCALED SAMPLE for {feature_type_desc} ---")
    print(f"Sample size: {X_train_sample_unscaled.shape[0]} ({SAMPLE_FRACTION_FOR_GRIDSEARCH*100:.1f}%)")
    unique_labels_sample, counts_sample = np.unique(y_train_sample, return_counts=True)
    print(f"Unique labels in sample: {unique_labels_sample}")
    print("Sample class distribution:")
    for label, count in zip(unique_labels_sample, counts_sample):
         print(f"  Class {label}: {count} samples")

    # --- Scale the Sample Data ---
    print("Scaling sample data for GridSearchCV...")
    try:
        scaler_sample = StandardScaler()
        X_train_sample_scaled = scaler_sample.fit_transform(X_train_sample_unscaled)
        print("  Sample data scaled successfully.")
        # Clean up unscaled data immediately
        del X_train_sample_unscaled
        gc.collect()
    except Exception as e:
         print(f"ERROR scaling sample data for GridSearchCV: {e}")
         print("Cannot proceed with GridSearchCV.")
         return None # Exit if scaling fails


    # --- Calculate Sample Weights ---
    # Use compute_class_weight on the *sample* labels
    try:
         sample_class_weights_dict = compute_class_weight(
             class_weight='balanced',
             classes=np.arange(num_classes), # Ensure all classes 0..N-1 are included
             y=y_train_sample
         )
         sample_weights = np.array([sample_class_weights_dict[label] for label in y_train_sample])
         print(f"Calculated sample weights for grid search sample.")
         # print(f"Sample weights (first 10): {sample_weights[:10]}") # Uncomment to see sample weights
    except Exception as e:
         print(f"ERROR calculating sample weights: {e}")
         print("Cannot proceed with weighted grid search.")
         del X_train_sample_scaled # Clean up scaled data
         gc.collect()
         return None # Cannot proceed without weights

    best_params_found = None
    best_score = -1.0
    used_gpu = False # Track if GPU was successfully used for grid search

    # --- Try GPU First ---
    try:
        print("\nAttempting GridSearchCV with GPU...")
        gpu_params = base_params.copy()
        gpu_params['device'] = 'cuda'
        gpu_params['num_class'] = num_classes
        gpu_params['objective'] = 'multi:softprob'
        if 'tree_method' not in gpu_params or gpu_params['tree_method'] not in ['hist', 'gpu_hist']:
             gpu_params['tree_method'] = 'hist'

        print(f"Initializing XGBClassifier (GPU) with params: {gpu_params}")
        estimator_gpu = xgb.XGBClassifier(**gpu_params)

        xgb_grid_search_gpu = GridSearchCV(estimator=estimator_gpu,
                                           param_grid=param_grid,
                                           scoring=scoring_metric, # Use specified scoring metric
                                           cv=cv_folds,
                                           verbose=2,
                                           n_jobs=1) # Use 1 job for GPU to avoid resource contention

        print(f"Starting GridSearchCV fitting (GPU), optimizing for '{scoring_metric}'...")
        # Pass SCALED sample data and sample weights to the fit method
        xgb_grid_search_gpu.fit(X_train_sample_scaled, y_train_sample, sample_weight=sample_weights)

        print("GridSearchCV with GPU successful.")
        used_gpu = True
        best_params_found = xgb_grid_search_gpu.best_params_
        best_score = xgb_grid_search_gpu.best_score_

    except (xgb.core.XGBoostError, Exception) as gpu_err:
        print(f"\nWARNING: GridSearchCV with GPU failed: {gpu_err}")
        # print(traceback.format_exc()) # Optional detailed traceback
        print("Falling back to CPU for GridSearchCV.")
        used_gpu = False
        if 'estimator_gpu' in locals(): del estimator_gpu
        if 'xgb_grid_search_gpu' in locals(): del xgb_grid_search_gpu
        gc.collect()

    # --- Fallback to CPU if GPU failed or wasn't tried ---
    if not used_gpu:
        try:
            print("\nAttempting GridSearchCV with CPU...")
            cpu_params = base_params.copy()
            cpu_params['num_class'] = num_classes
            cpu_params['objective'] = 'multi:softprob'
            if 'device' in cpu_params: del cpu_params['device']
            cpu_params['tree_method'] = 'hist'

            print(f"Initializing XGBClassifier (CPU) with params: {cpu_params}")
            estimator_cpu = xgb.XGBClassifier(**cpu_params)

            xgb_grid_search_cpu = GridSearchCV(estimator=estimator_cpu,
                                              param_grid=param_grid,
                                              scoring=scoring_metric, # Use specified scoring metric
                                              cv=cv_folds,
                                              verbose=2,
                                              n_jobs=-1) # Use all available CPU cores

            print(f"Starting GridSearchCV fitting (CPU), optimizing for '{scoring_metric}'...")
            # Pass SCALED sample data and sample weights to the fit method
            xgb_grid_search_cpu.fit(X_train_sample_scaled, y_train_sample, sample_weight=sample_weights)

            print("GridSearchCV with CPU successful.")
            best_params_found = xgb_grid_search_cpu.best_params_
            best_score = xgb_grid_search_cpu.best_score_

        except Exception as cpu_err:
            print(f"\nERROR: GridSearchCV with CPU also failed: {cpu_err}")
            print(traceback.format_exc())
            # Clean up scaled data
            del X_train_sample_scaled
            gc.collect()
            return None

    # --- Combine and Return Best Parameters ---
    # Clean up scaled data after grid search is done
    del X_train_sample_scaled
    gc.collect()

    if best_params_found is not None:
        print(f"\nGridSearchCV Complete for {feature_type_desc}.")
        print(f"  Used Device for Grid Search: {'GPU' if used_gpu else 'CPU'}")
        print(f"  Best parameters found (on sample): {best_params_found}")
        print(f"  Best CV score ({scoring_metric} on sample): {best_score:.4f}")

        # Combine base parameters with the best ones found by the grid search
        final_params = base_params.copy()
        final_params.update(best_params_found)
        # Add the device parameter back based on which one succeeded in grid search
        if used_gpu:
            final_params['device'] = 'cuda'
        elif 'device' in final_params:
             del final_params['device'] # Ensure no 'device' if CPU was used

        # Store the actual score metric used in the returned params for logging later
        final_params['_gridsearch_scoring'] = scoring_metric

        return final_params
    else:
        print(f"ERROR: Could not determine best parameters for {feature_type_desc}.")
        return None


def train_and_evaluate_xgb_dmatrix(dtrain_path, dtest_path, y_test_labels,
                                  best_params_from_search, # Params from grid search
                                  feature_type_desc, target_class_names,
                                  output_results_dir):
    """Trains final XGBoost model using DMatrix files, respecting device from search."""
    # Note: The DMatrix files are assumed to be already scaled and weighted as prepared by create_xgb_dmatrix_files
    print(f"\n--- Training FINAL XGBoost on FULL data for {feature_type_desc} using DMatrix ---")
    print(f"Train DMatrix: {dtrain_path}")
    print(f"Test DMatrix: {dtest_path}")
    # print(f"Base parameters from search: {best_params_from_search}") # Don't print full params here, use final_params_used later

    if not os.path.exists(dtrain_path) or not os.path.exists(dtest_path):
        print("ERROR: DMatrix train or test file not found. Skipping.")
        return None

    bst = None
    final_params_used = {}
    used_gpu_in_final_train = False
    gridsearch_scoring_metric = best_params_from_search.get('_gridsearch_scoring', 'accuracy')


    try:
        print("Loading DMatrix files...")
        dtrain = xgb.DMatrix(dtrain_path)
        dtest = xgb.DMatrix(dtest_path)
        print("DMatrix files loaded.")
        dtest.set_label(y_test_labels.astype(np.float32))

        final_params_to_try = best_params_from_search.copy()
        if '_gridsearch_scoring' in final_params_to_try: del final_params_to_try['_gridsearch_scoring']


        # Determine final device based on grid search result preference, but check availability
        preferred_device = final_params_to_try.get('device', 'cpu')
        actual_device = 'cpu'

        if preferred_device == 'cuda':
             try:
                 # Check if CUDA is available and working for xgb.train
                 temp_cpu_params = final_params_to_try.copy()
                 if 'device' in temp_cpu_params: del temp_cpu_params['device']
                 test_dmatrix = xgb.DMatrix(np.random.rand(2, dtrain.num_col()), label=np.random.randint(0, NUM_CLASSES, 2))
                 temp_gpu_model = xgb.train(temp_cpu_params, test_dmatrix, num_boost_round=1, evals=[(test_dmatrix, 'eval')], tree_method='hist', device='cuda')
                 del temp_gpu_model, test_dmatrix # Clean up test objects
                 gc.collect()
                 actual_device = 'cuda'
                 print(f"\nCUDA device detected and available for final training.")

             except xgb.core.XGBoostError as e:
                 print(f"\nWARNING: Preferred GPU training failed ({e}). CUDA device may not be available or configured. Falling back to CPU.")
                 # print(traceback.format_exc()) # Optional detailed traceback
                 actual_device = 'cpu'
                 if 'device' in final_params_to_try: del final_params_to_try['device'] # Remove device param for CPU

        if actual_device == 'cuda':
             final_params_to_try['device'] = 'cuda'
             if 'tree_method' not in final_params_to_try or final_params_to_try['tree_method'] not in ['hist', 'gpu_hist']:
                      final_params_to_try['tree_method'] = 'hist'
                      print("Setting tree_method to 'hist' for GPU.")
             used_gpu_in_final_train = True
             final_params_used = final_params_to_try

        else: # Use CPU
             if 'device' in final_params_to_try: del final_params_to_try['device']
             final_params_to_try['tree_method'] = 'hist' # Hist is good for CPU too
             used_gpu_in_final_train = False
             final_params_used = final_params_to_try


        print(f"Starting final XGBoost training ({actual_device}) with params: {final_params_used}")
        evals = [(dtrain, 'train'), (dtest, 'eval')]
        num_boost_round = final_params_used.get('n_estimators', 100)
        train_params = final_params_used.copy()
        if 'n_estimators' in train_params: del train_params['n_estimators'] # n_estimators is for estimators, not xgb.train num_boost_round usually

        bst = xgb.train(
            train_params, dtrain, num_boost_round=num_boost_round, evals=evals,
            early_stopping_rounds=50, verbose_eval=100
        )
        print("Final training successful.")


        # --- Evaluation (if training succeeded) ---
        if bst is not None:
            print("\nEvaluating final model on test set...")
            model_filename = os.path.join(output_results_dir, f'xgb_model_{feature_type_desc.replace(" ", "_").replace("/", "-")}.json')
            bst.save_model(model_filename)
            print(f"Saved final XGBoost model ({actual_device}) for {feature_type_desc} to {model_filename}")

            y_pred_proba = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
            y_pred_labels = np.argmax(y_pred_proba, axis=1)

            # Add check for dimensions if necessary, although DMatrix should handle this
            if len(y_test_labels) != len(y_pred_labels):
                 print(f"ERROR: Mismatch between true test labels ({len(y_test_labels)}) and predicted test labels ({len(y_pred_labels)}). Cannot reliably calculate metrics.")
                 # Proceed to calculate metrics if shapes match anyway, but note the error.
                 # This might happen if DMatrix predict output shape is unexpected.

            accuracy_val = accuracy_score(y_test_labels, y_pred_labels)
            # Ensure target_names match the actual labels present in y_test_labels and y_pred_labels
            # Use labels from the original label_encoder for classification_report/confusion_matrix
            class_report_str = classification_report(y_test_labels, y_pred_labels, target_names=target_class_names, zero_division=0)
            conf_matrix = confusion_matrix(y_test_labels, y_pred_labels, labels=np.arange(len(target_class_names)))
            f1_macro_val = f1_score(y_test_labels, y_pred_labels, average='macro', zero_division=0)
            # Also calculate per-class F1 if needed: f1_scores = f1_score(y_test_labels, y_pred_labels, average=None, labels=np.arange(len(target_class_names)))


            print(f"Final Model Used Device: {actual_device}")
            print(f"Test Set Accuracy: {accuracy_val:.4f}")
            print(f"Test Set F1-macro: {f1_macro_val:.4f}")
            print(f"Classification Report (XGBoost - {feature_type_desc}):\n{class_report_str}")
            plot_confusion_matrix(conf_matrix, classes=target_class_names,
                                  plot_title=f'CM XGB ({actual_device}) {feature_type_desc} (Acc: {accuracy_val:.3f}, F1-M: {f1_macro_val:.3f})',
                                  results_path=output_results_dir,
                                  filename=f'cm_xgb_{feature_type_desc.replace(" ", "_").replace("/", "-")}.png')

            results_text_file = os.path.join(output_results_dir, f'results_xgb_{feature_type_desc.replace(" ", "_").replace("/", "-")}.txt')
            with open(results_text_file, 'w') as f:
                f.write(f"--- XGBoost Results for {feature_type_desc} ---\n")
                f.write(f"Trained using DMatrix files.\n")
                f.write(f"GridSearchCV Scoring Metric: {gridsearch_scoring_metric}\n")
                f.write(f"Device Used for Final Training: {actual_device}\n")
                f.write(f"Final Training Parameters Used: {final_params_used}\n")
                if hasattr(bst, 'best_iteration'):
                     f.write(f"Best Iteration: {bst.best_iteration}\n")
                f.write(f"Test Set Accuracy: {accuracy_val:.4f}\n")
                f.write(f"Test Set F1-macro: {f1_macro_val:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(class_report_str + "\n\n")
                f.write("Confusion Matrix:\n")
                f.write(np.array2string(conf_matrix))
            print(f"Saved XGBoost results for {feature_type_desc} to {results_text_file}")

            # Clean up DMatrix objects *after* all operations
            del dtrain
            del dtest
            gc.collect()

            return bst
        else:
            print("ERROR: Training failed, no model to evaluate.")
            # Clean up DMatrix objects if they were created
            if 'dtrain' in locals(): del dtrain
            if 'dtest' in locals(): del dtest
            gc.collect()
            return None

    except Exception as e:
        print(f"ERROR: An unexpected error occurred during DMatrix loading/training/evaluation for {feature_type_desc}: {e}")
        print(traceback.format_exc())
        # Clean up DMatrix objects if they were created
        if 'dtrain' in locals(): del dtrain
        if 'dtest' in locals(): del dtest
        gc.collect()
        return None

# --- 5. Main Execution Pipeline ---
def run_spm_classification_pipeline():
    print("\n--- Starting SPM Classification Pipeline with DMatrix and Scaling ---")

    # --- Create Sampled Data Indices and Sample Labels for GridSearchCV ---
    # If SAMPLE_FRACTION_FOR_GRIDSEARCH is 1, use the full training set for tuning.
    # This will still require loading the full dataset into memory for GridSearchCV.
    # If memory is an issue, reduce SAMPLE_FRACTION_FOR_GRIDSEARCH.
    if SAMPLE_FRACTION_FOR_GRIDSEARCH < 1:
        print(f"\nCreating a {SAMPLE_FRACTION_FOR_GRIDSEARCH*100:.1f}% stratified sample indices for GridSearchCV...")
        try:
            train_indices_sample, _, y_train_sample, _ = train_test_split(
                train_indices_full, y_train_full,
                train_size=SAMPLE_FRACTION_FOR_GRIDSEARCH,
                random_state=42,
                stratify=y_train_full
            )
            print(f"Sample size for tuning: {len(train_indices_sample)}")
        except ValueError as e:
            print(f"Error during train_test_split for sampling: {e}")
            print("Ensure SAMPLE_FRACTION_FOR_GRIDSEARCH is large enough for stratification.")
            return # Exit pipeline if sampling fails
    else:
        print("\nUsing the FULL training set for GridSearchCV (SAMPLE_FRACTION_FOR_GRIDSEARCH=1).")
        train_indices_sample = train_indices_full
        y_train_sample = y_train_full
        print(f"Sample size for tuning: {len(train_indices_sample)} (Full set)")


    # --- Define Feature Combinations to Test ---
    feature_sets_to_run = {
        f"SPM_SIFT_L{PYRAMID_LEVELS-1}": ["sift_spm"],
        f"SPM_ORB_L{PYRAMID_LEVELS-1}": ["orb_spm"],
        "Global_HOG": ["hog"],
        f"SPM_SIFT_L{PYRAMID_LEVELS-1}_HOG": ["sift_spm", "hog"],
        f"SPM_ORB_L{PYRAMID_LEVELS-1}_HOG": ["orb_spm", "hog"],
        f"SPM_SIFT_ORB_L{PYRAMID_LEVELS-1}_HOG": ["sift_spm", "orb_spm", "hog"],
    }

    # --- Calculate Full Training Set Class Weights ---
    # These weights are for saving with the FULL DMatrix for the final xgb.train step
    try:
        full_train_class_weights_dict = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(NUM_CLASSES),
            y=y_train_full
        )
        # XGBoost DMatrix weight parameter expects a weight for each sample, not per class
        full_train_sample_weights = np.array([full_train_class_weights_dict[label] for label in y_train_full])
        print("\nCalculated full training set sample weights for DMatrix.")
        # print(f"Weights: {full_train_class_weights_dict}") # Optional: print weights
        # print(f"Sample weights (first 10): {full_train_sample_weights[:10]}") # Optional: print sample weights
    except Exception as e:
        print(f"ERROR calculating full training set class weights: {e}")
        print("Cannot proceed.")
        return # Exit if weight calculation fails


    # --- Loop Through Feature Combinations ---
    all_best_params = {}

    for feature_desc, features_to_combine in feature_sets_to_run.items():
        print(f"\n\n{'='*20} Processing Feature Set: {feature_desc} {'='*20}")

        # --- Step 1: Load Sample Data for GridSearchCV Tuning ---
        # Load the *sample data* into memory for GridSearchCV tuning
        print(f"\nLoading sample data ({len(train_indices_sample)} instances) into memory for GridSearchCV...")
        sample_features_list_unscaled = []
        # Use the refined load_spm_features and load_and_align_global_hog, pass sample indices
        
        sample_loading_success = True
        if "sift_spm" in features_to_combine:
             sift_spm_sample_unscaled = load_spm_features(BOVW_SPM_FEATURES_DIR, "sift", PYRAMID_LEVELS, train_indices_sample, True)
             if sift_spm_sample_unscaled is not None and sift_spm_sample_unscaled.shape[0]==len(y_train_sample):
                  sample_features_list_unscaled.append(sift_spm_sample_unscaled)
             else:
                  print("Failed to load SIFT for sample gridsearch."); sample_loading_success = False
        if "orb_spm" in features_to_combine and sample_loading_success:
             orb_spm_sample_unscaled = load_spm_features(BOVW_SPM_FEATURES_DIR, "orb", PYRAMID_LEVELS, train_indices_sample, True)
             if orb_spm_sample_unscaled is not None and orb_spm_sample_unscaled.shape[0]==len(y_train_sample):
                  sample_features_list_unscaled.append(orb_spm_sample_unscaled)
             else:
                  print("Failed to load ORB for sample gridsearch."); sample_loading_success = False
        if "hog" in features_to_combine and sample_loading_success:
             hog_sample_unscaled = load_and_align_global_hog(HOG_DATA_FILE, train_indices_sample)
             if hog_sample_unscaled is not None and hog_sample_unscaled.shape[0]==len(y_train_sample):
                  # Check HOG dimension; load_and_align already prints warning if dim=0
                  if hog_sample_unscaled.shape[1] > 0:
                      sample_features_list_unscaled.append(hog_sample_unscaled)
                  else:
                      print("HOG sample features had zero dimension, skipping concatenation for sample.");
             else:
                  print("Failed to load HOG for sample gridsearch."); sample_loading_success = False

        if not sample_loading_success:
             print(f"Skipping {feature_desc} due to failure loading sample data for GridSearchCV.")
             # Clean up partial loads if any
             if 'sift_spm_sample_unscaled' in locals(): del sift_spm_sample_unscaled
             if 'orb_spm_sample_unscaled' in locals(): del orb_spm_sample_unscaled
             if 'hog_sample_unscaled' in locals(): del hog_sample_unscaled
             if 'sample_features_list_unscaled' in locals(): del sample_features_list_unscaled
             gc.collect()
             continue

        # Concatenate sample features for GridSearchCV
        try:
             if not sample_features_list_unscaled:
                 print("No valid sample features loaded after filtering zero-dim ones. Skipping."); continue
             if len(sample_features_list_unscaled) == 1:
                 X_train_sample_combined_unscaled = sample_features_list_unscaled[0]
             else:
                 # Re-check consistency before concatenating
                 ref_shape_sample = sample_features_list_unscaled[0].shape[0]
                 if not all(f.shape[0] == ref_shape_sample for f in sample_features_list_unscaled):
                     print("ERROR: Mismatched number of samples in sample features to concatenate."); continue

                 X_train_sample_combined_unscaled = np.concatenate(sample_features_list_unscaled, axis=1)

             print(f"Combined sample data shape (unscaled) for GridSearchCV: {X_train_sample_combined_unscaled.shape}")
             if X_train_sample_combined_unscaled.shape[0] != len(y_train_sample):
                  print(f"ERROR: Mismatch sample features/labels ({X_train_sample_combined_unscaled.shape[0]} vs {len(y_train_sample)})."); continue

        except MemoryError:
             print("ERROR: OOM concatenating SAMPLE data."); continue
        except Exception as e:
             print(f"Error combining sample features: {e}"); continue
        finally:
             # Clean up individual unscaled sample feature arrays
             if 'sift_spm_sample_unscaled' in locals(): del sift_spm_sample_unscaled
             if 'orb_spm_sample_unscaled' in locals(): del orb_spm_sample_unscaled
             if 'hog_sample_unscaled' in locals(): del hog_sample_unscaled
             if 'sample_features_list_unscaled' in locals(): del sample_features_list_unscaled
             gc.collect()


        # --- Step 2: Perform GridSearchCV on the SCALED SAMPLE ---
        # The scaling is now handled *inside* find_best_params_with_gridsearch_on_sample
        best_params = find_best_params_with_gridsearch_on_sample(
            X_train_sample_combined_unscaled, y_train_sample, NUM_CLASSES,
            XGB_BASE_PARAMS, PARAM_GRID_XGB, GRIDSEARCH_CV_FOLDS, feature_desc,
            scoring_metric=GRIDSEARCH_SCORING # Pass the desired scoring metric
        )

        # X_train_sample_combined_unscaled is cleaned up inside find_best_params...

        if best_params is None:
            print(f"GridSearchCV failed for {feature_desc}. Skipping final training.")
            continue

        all_best_params[feature_desc] = best_params

        # --- Step 3: Create DMatrix files for the FULL dataset (Scaled & Weighted) ---
        # Train DMatrix - FIT & TRANSFORM StandardScaler + apply weights + save scaler
        # The scaler filename is automatically derived and saved inside create_xgb_dmatrix_files
        dtrain_path = create_xgb_dmatrix_files(
            features_to_combine, train_indices_full, y_train_full, is_train_set=True, output_dir=DMATRIX_CACHE_DIR,
            sample_weights=full_train_sample_weights, perform_scaling=True
        )

        # Derive the expected path for the scaler that was just saved for the train set
        # This relies on the consistent naming convention inside create_xgb_dmatrix_files
        if dtrain_path: # Only proceed if train DMatrix was successfully created
            train_filename_base = os.path.basename(dtrain_path).replace(".buffer", "")
            # Remove suffixes like _weighted_scaled to get the base for scaler name
            train_filename_base_for_scaler = "_".join(train_filename_base.split("_")[1:]).replace("_weighted_scaled", "").replace("_scaled", "").replace("_weighted","")
            train_scaler_path_for_test = os.path.join(DMATRIX_CACHE_DIR, f"train_{train_filename_base_for_scaler}_scaler.joblib")
            print(f"Expected train scaler path for test set: {train_scaler_path_for_test}")

            # Test DMatrix - TRANSFORM ONLY using the fitted scaler + NO weights
            dtest_path = create_xgb_dmatrix_files(
                features_to_combine, test_indices_full, y_test_full, is_train_set=False, output_dir=DMATRIX_CACHE_DIR,
                sample_weights=None, perform_scaling=True, train_scaler_path=train_scaler_path_for_test # Pass the fitted scaler path
            )
        else:
             dtest_path = None # Cannot create test DMatrix if train failed

        if dtrain_path is None or dtest_path is None:
            print(f"Skipping final training for {feature_desc} due to DMatrix creation failure.")
            continue

        # --- Step 4: Train final model on FULL data using DMatrix ---
        # The DMatrix files created in Step 3 are already scaled and weighted.
        # train_and_evaluate_xgb_dmatrix doesn't need to handle scaling/weighting directly.
        train_and_evaluate_xgb_dmatrix(
            dtrain_path, dtest_path, y_test_full, # Pass full test labels
            best_params, # Use the params found by grid search (includes device hint)
            feature_desc, class_names, RESULTS_DIR_XGB_SPM
        )

    print("\n--- SPM Classification Pipeline Complete ---")
    print("Best parameters found (from sample grid search):")
    for name, params in all_best_params.items():
        # Remove the internal _gridsearch_scoring key for cleaner output
        display_params = params.copy()
        if '_gridsearch_scoring' in display_params: del display_params['_gridsearch_scoring']
        print(f"  {name}: {display_params}")
    print(f"\nResults saved to: {RESULTS_DIR_XGB_SPM}")