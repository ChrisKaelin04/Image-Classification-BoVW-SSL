import numpy as np
import os
import pickle
import warnings
import joblib
import h5py # For loading HOG data
import pandas as pd # Added for reading the subset index map
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import traceback
import gc
import glob # For finding the subset map file

# --- Configuration ---
FEATURES_DIR_SPM = r"E:\CV_features_SPM" # Directory for SPM feature batches, vocabularies, HOG_SPM, etc.
BOVW_SPM_FEATURES_DIR = os.path.join(FEATURES_DIR_SPM, "bovw_spm_features_4cat") # Directory for saved SPM .npy histograms
HOG_DATA_FILE_SPM = os.path.join(FEATURES_DIR_SPM, 'hog_data_spm.h5') # HOG data saved with subset indices

# Updated path for the subset index -> label map saved during extraction
SUBSET_INDEX_LABEL_MAP_FILE_PATTERN = os.path.join(FEATURES_DIR_SPM, 'subset_index_label_map_subset*_seed*.csv')

# ASSUMPTION: This NPZ file now contains train/test splits defined by *subset indices*
SPLITS_DIR_COMMON = os.path.join(r"E:\CV_features", "train_test_splits_4cat_revised") # Assuming this dir exists
NPZ_FILE_SUBSET_SPLIT = os.path.join(SPLITS_DIR_COMMON, "train_test_split_data_4cat_revised.npz")

LABEL_ENCODER_FILE = os.path.join(SPLITS_DIR_COMMON, "broad_label_encoder_4cat_revised.pkl")
RESULTS_DIR_XGB_SPM = os.path.join(FEATURES_DIR_SPM, "classification_results_XGB_SPM_SOH_4cat")
os.makedirs(RESULTS_DIR_XGB_SPM, exist_ok=True)
DMATRIX_CACHE_DIR = os.path.join(FEATURES_DIR_SPM, "xgb_dmatrix_cache_4cat")
os.makedirs(DMATRIX_CACHE_DIR, exist_ok=True)

# Match VOCAB_SIZE and PYRAMID_LEVELS to what was used during histogram creation
VOCAB_SIZE = 1000
PYRAMID_LEVELS = 2

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
SAMPLE_FRACTION_FOR_GRIDSEARCH = 1 # 1 means use the full available training set for tuning
# Change scoring metric for GridSearchCV to handle imbalance
GRIDSEARCH_SCORING = 'f1_macro' # Or 'recall_macro'

warnings.filterwarnings("ignore", message="Parameters: {.*use_label_encoder.*} are not used.", category=UserWarning, module="xgboost.core")
warnings.filterwarnings("ignore", message="omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.", category=UserWarning)


# --- 1. Load Common Data (Labels, Subset Split Indices, Label Encoder) ---
print("--- Loading Common Data (Labels, Subset Splits, Encoder) ---")

# Load the subset index -> label map file
map_files = glob.glob(SUBSET_INDEX_LABEL_MAP_FILE_PATTERN)
if not map_files:
    print(f"ERROR: Subset index map file not found matching pattern: {SUBSET_INDEX_LABEL_MAP_FILE_PATTERN}")
    print("Please run SOH_extract_SPM.py first to generate this file.")
    exit()
# Assuming only one such file exists, pick the first one found
subset_map_file = map_files[0]
print(f"Loading subset index map from: {subset_map_file}")
try:
    subset_map_df = pd.read_csv(subset_map_file)
    # Create a dictionary mapping subset_idx to label
    # This dict contains all subset indices that were processed in extraction
    subset_idx_to_label_all = dict(zip(subset_map_df['subset_idx'], subset_map_df['label']))
    print(f"Loaded map for {len(subset_idx_to_label_all)} subset indices.")

    # Also parse subset size and seed from the filename for later NPY loading
    try:
         parts = os.path.basename(subset_map_file).split('_')
         MAP_SUBSET_SIZE = int(parts[3].replace('subset', ''))
         MAP_RANDOM_SEED = int(parts[4].replace('seed', '').replace('.csv', ''))
         print(f"Parsed subset size: {MAP_SUBSET_SIZE}, seed: {MAP_RANDOM_SEED} from map filename.")
    except Exception as e:
         print(f"Warning: Could not parse subset size/seed from map filename ({subset_map_file}): {e}. Using generic filenames for features.")
         MAP_SUBSET_SIZE = None # Indicate parsing failed
         MAP_RANDOM_SEED = None

except Exception as e:
    print(f"ERROR loading or processing subset index map file {subset_map_file}: {e}")
    exit()


# Load the train/test splits defined by *subset indices* from the NPZ file
print(f"Loading train/test split data (assuming subset indices) from: {NPZ_FILE_SUBSET_SPLIT}")
try:
    split_data = np.load(NPZ_FILE_SUBSET_SPLIT)
    # These are the subset indices that constitute the train and test splits
    subset_train_indices_npz = split_data['train_indices'].tolist()
    subset_test_indices_npz = split_data['test_indices'].tolist()
    # These are the labels corresponding to the indices in subset_train_indices_npz / subset_test_indices_npz
    y_train_npz = split_data['train_labels_numeric'].tolist()
    y_test_npz = split_data['test_labels_numeric'].tolist()

    print(f"Loaded {len(subset_train_indices_npz)} subset training indices/labels and {len(subset_test_indices_npz)} subset testing indices/labels from NPZ.")

    # Optional: Verify consistency between NPZ labels and Map labels for these indices
    # This is good for debugging but might be slow for large datasets
    # for i, subset_idx in enumerate(subset_train_indices_npz):
    #      if subset_idx in subset_idx_to_label_all and subset_idx_to_label_all[subset_idx] != y_train_npz[i]:
    #           print(f"Warning: Label mismatch for subset_idx {subset_idx} between NPZ ({y_train_npz[i]}) and Map ({subset_idx_to_label_all[subset_idx]})")
    #      elif subset_idx not in subset_idx_to_label_all:
    #           print(f"Warning: subset_idx {subset_idx} from NPZ train split not found in full map.")

except FileNotFoundError:
    print(f"ERROR: NPZ file not found at {NPZ_FILE_SUBSET_SPLIT}. Make sure it contains train/test splits based on *subset indices*.")
    exit()
except KeyError as e:
    print(f"ERROR: Missing key {e} in NPZ file {NPZ_FILE_SUBSET_SPLIT}. Make sure it contains train/test splits based on *subset indices*.")
    exit()
except Exception as e:
    print(f"ERROR loading or processing NPZ file {NPZ_FILE_SUBSET_SPLIT}: {e}")
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

# --- Filter Indices to Use (Must be in NPZ split AND had successful extraction) ---
# We only use indices from the NPZ split list that actually had features extracted (label != -1 in the map)
print("\nFiltering indices to include only those with successful feature extraction...")
subset_indices_extracted_success = {idx for idx, label in subset_idx_to_label_all.items() if label != -1}

actual_train_subset_indices = sorted([idx for idx in subset_train_indices_npz if idx in subset_indices_extracted_success])
actual_test_subset_indices = sorted([idx for idx in subset_test_indices_npz if idx in subset_indices_extracted_success])

# Get the corresponding labels for these filtered, sorted lists of indices
# Look up labels in the full map and order them to match the sorted index lists
y_train = np.array([subset_idx_to_label_all[idx] for idx in actual_train_subset_indices])
y_test = np.array([subset_idx_to_label_all[idx] for idx in actual_test_subset_indices])

print(f"Using {len(actual_train_subset_indices)} subset indices for training (filtered).")
print(f"Using {len(actual_test_subset_indices)} subset indices for testing (filtered).")

if len(actual_train_subset_indices) == 0 or len(actual_test_subset_indices) == 0:
    print("ERROR: No valid subset indices found for train or test after filtering. Cannot proceed.")
    exit()

# --- Check Final Class Distribution ---
print("\n--- Final Train/Test Set Class Distribution Check (using filtered indices) ---")
unique_train, counts_train = np.unique(y_train, return_counts=True)
print("Training set class distribution:")
for label, count in zip(unique_train, counts_train):
    print(f"  Class {label} ({class_names[label]}): {count} samples ({count/len(y_train):.2%})")
unique_test, counts_test = np.unique(y_test, return_counts=True)
print("Test set class distribution:")
for label, count in zip(unique_test, counts_test):
    print(f"  Class {label} ({class_names[label]}): {count} samples ({count/len(y_test):.2%})")


# --- 2. Helper Functions (plot_confusion_matrix remains the same) ---
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


# --- Feature Loading Functions (Rewritten for Subset Indices) ---

def load_spm_features_by_subset_indices(spm_bovw_dir, feature_name, pyramid_levels_count, requested_subset_indices, total_subset_size=None, seed=None):
    """
    Loads SPM features from a .npy file and aligns them to the requested subset indices.
    Assumes the .npy file rows are ordered by the subset indices passed to histogram_creation_SPM.
    """
    max_level_index = pyramid_levels_count - 1
    # Construct filename based on naming convention from histogram_creation_SPM
    if total_subset_size is not None and seed is not None:
         # Use the specific filename if subset size and seed are known
         filename_pattern = f"X_*_{feature_name}_spm_L{max_level_index}_k{VOCAB_SIZE}_subset{total_subset_size}_seed{seed}.npy"
    else:
         # Fallback to a more general pattern if subset size/seed are unknown/parsing failed
         filename_pattern = f"X_*_{feature_name}_spm_L{max_level_index}_k{VOCAB_SIZE}_*.npy" # Match any subset/seed pattern
         # Also include the generic filename pattern just in case
         filename_pattern_generic = f"X_*_{feature_name}_spm_L{max_level_index}_k{VOCAB_SIZE}_processed*.npy"
         print(f"Warning: Subset size or seed not parsed, using general pattern: {filename_pattern} or {filename_pattern_generic}")


    # Find the actual file path(s) matching the pattern
    if total_subset_size is not None and seed is not None:
         filepaths = glob.glob(os.path.join(spm_bovw_dir, filename_pattern))
    else:
         filepaths = glob.glob(os.path.join(spm_bovw_dir, filename_pattern)) + \
                     glob.glob(os.path.join(spm_bovw_dir, filename_pattern_generic))

    if not filepaths:
        print(f"Error: No {feature_name} SPM (L{max_level_index}) file found matching pattern in {spm_bovw_dir}.")
        return None

    # Assuming there's only one relevant train/test file matching the pattern
    # Filter for train or test based on requested_subset_indices
    # A better approach is to name the NPY file based on train/test split explicitly
    # The histogram script saves as X_train_... and X_test_...
    set_type = "train" if all(idx in actual_train_subset_indices for idx in requested_subset_indices) and len(requested_subset_indices) > 0 else "test"
    # Refined filename pattern based on set_type
    if total_subset_size is not None and seed is not None:
         filename = f"X_{set_type}_{feature_name}_spm_L{max_level_index}_k{VOCAB_SIZE}_subset{total_subset_size}_seed{seed}.npy"
    else:
         filename = f"X_{set_type}_{feature_name}_spm_L{max_level_index}_k{VOCAB_SIZE}_*.npy"
         filename_generic = f"X_{set_type}_{feature_name}_spm_L{max_level_index}_k{VOCAB_SIZE}_processed*.npy"
         
    filepaths_filtered = glob.glob(os.path.join(spm_bovw_dir, filename))
    if not filepaths_filtered and total_subset_size is None or seed is None: # Try generic if specific fails
         filepaths_filtered = glob.glob(os.path.join(spm_bovw_dir, filename_generic))

    if not filepaths_filtered:
         print(f"Error: No {set_type} {feature_name} SPM (L{max_level_index}) file found matching expected filename ({filename} or {filename_generic}) in {spm_bovw_dir}.")
         return None

    # Use the first matching file found
    filepath = filepaths_filtered[0]
    print(f"Loading {set_type} {feature_name} SPM (L{max_level_index}) features from: {filepath}")

    try:
        all_data = np.load(filepath)
        print(f"  Full loaded shape: {all_data.shape}")
    except Exception as e:
         print(f"Error loading NPY file {filepath}: {e}")
         return None


    # --- Index Alignment ---
    # The .npy file rows are ordered by the sorted list of subset indices passed to process_subset_indices_spm_parallel
    # in histogram_creation_SPM. These lists were actual_train_subset_indices and actual_test_subset_indices.
    # So, the indices corresponding to the rows in the *full* loaded NPY file are simply
    # actual_train_subset_indices (if loading X_train) or actual_test_subset_indices (if loading X_test), sorted.
    
    # Need the full list of indices that were used to generate the NPY file *in their sorted order*
    # This requires knowing if the NPY is for train or test and accessing the corresponding actual_ indices list.
    # This is a bit circular dependency.

    # A more robust way: histogram_creation_SPM should save a small NPY file alongside the features NPY
    # containing the sorted list of subset indices that correspond to the rows in the features NPY.
    # E.g., save X_train_sift_spm_L1.npy AND train_sift_spm_L1_indices.npy

    # --- ASSUMPTION: Let's assume the NPY file *is* sorted by subset index, and the indices
    # that were used to generate it are the `actual_train_subset_indices` or `actual_test_subset_indices`.
    # The order of rows in the NPY file *is* the sorted order of those lists.

    # Get the list of indices that correspond to the rows in the loaded NPY file
    # This depends on which NPY file was loaded (train or test)
    indices_in_npy_order = actual_train_subset_indices if set_type == "train" else actual_test_subset_indices
    indices_in_npy_order = sorted(indices_in_npy_order) # Ensure it's sorted


    if all_data.shape[0] != len(indices_in_npy_order):
        print(f"ERROR: Loaded NPY data rows ({all_data.shape[0]}) does not match the expected number of indices ({len(indices_in_npy_order)}) for {set_type}. Misalignment is HIGHLY likely.")
        return None # Critical error: Data rows don't match the expected indices count


    # Map subset indices present in the NPY file to their row index in the NPY array
    subset_idx_to_row_in_npy = {idx: i for i, idx in enumerate(indices_in_npy_order)}

    # Build a list of row indices from the NPY that correspond to the requested subset indices
    # And simultaneously build a list to store the features in the requested order
    selected_features_ordered = []
    missing_indices_count = 0

    for requested_idx in requested_subset_indices:
        row_in_npy = subset_idx_to_row_in_npy.get(requested_idx)
        if row_in_npy is not None:
            # Append the feature vector from the NPY at that row index
            selected_features_ordered.append(all_data[row_in_npy, :])
        else:
            # Handle case where a requested index was not found in the NPY data (e.g., extraction failed for it)
            # Append a zero vector of the expected dimension
            feature_dim = all_data.shape[1] if all_data.shape[1] > 0 else (VOCAB_SIZE * sum([(2**l)**2 for l in range(pyramid_levels_count)])) # Fallback dim calc
            selected_features_ordered.append(np.zeros(feature_dim, dtype=np.float32))
            missing_indices_count += 1

    if missing_indices_count > 0:
        print(f"  Warning: {missing_indices_count}/{len(requested_subset_indices)} requested {feature_name} SPM features were not found in the loaded NPY data. Used zero vectors.")


    if not selected_features_ordered:
         print(f"  No {feature_name} SPM features were selected for the requested indices.")
         return np.empty((0, all_data.shape[1] if all_data.shape[1] > 0 else 0), dtype=np.float32)


    # Stack the selected features into a single NumPy array, already in the requested order
    try:
        aligned_features_array = np.vstack(selected_features_ordered)
        print(f"  Aligned {feature_name} SPM features shape for requested indices: {aligned_features_array.shape}")
        return aligned_features_array
    except ValueError as e:
        print(f"ERROR: Could not stack selected {feature_name} SPM features: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error stacking selected {feature_name} SPM features: {e}")
        return None
    finally:
        # Clean up the full loaded data
        del all_data
        gc.collect()


def load_and_align_global_hog_by_subset_indices(hog_h5_filepath, requested_subset_indices):
    """
    Loads global HOG features from HDF5 and aligns them to the requested subset indices.
    Assumes the HDF5 contains 'hog_features' and 'indices' (subset indices).
    """
    if not os.path.exists(hog_h5_filepath):
        print(f"Warning: Global HOG data file not found: {hog_h5_filepath}")
        return None

    print(f"Loading global HOG features from: {hog_h5_filepath} for {len(requested_subset_indices)} requested subset indices.")

    try:
        with h5py.File(hog_h5_filepath, 'r') as hf:
            if 'hog_features' not in hf or 'indices' not in hf:
                print(f"ERROR: 'hog_features' or 'indices' not found in HDF5 file: {hog_h5_filepath}")
                return None
            all_hog_features = hf['hog_features'][:]
            all_hog_subset_indices = hf['indices'][:] # These are the subset indices
    except Exception as e:
        print(f"Error loading HOG data from {hog_h5_filepath}: {e}")
        return None

    if all_hog_features.size == 0 or all_hog_subset_indices.size == 0:
        print(f"Warning: HOG features or indices in {hog_h5_filepath} are empty.")
        # Return empty array with 0 feature dimension, but correct number of samples for requested indices
        return np.empty((len(requested_subset_indices), 0), dtype=np.float32)

    # Ensure HOG features are 2D (samples x features)
    if all_hog_features.ndim == 1:
        if all_hog_subset_indices.ndim == 1 and all_hog_subset_indices.shape[0] > 0 and all_hog_features.shape[0] % all_hog_subset_indices.shape[0] == 0:
            expected_dim = all_hog_features.shape[0] // all_hog_subset_indices.shape[0]
            print(f"  Reshaping 1D HOG features into ({all_hog_subset_indices.shape[0]}, {expected_dim})")
            all_hog_features = all_hog_features.reshape(all_hog_subset_indices.shape[0], expected_dim)
        else:
            print(f"ERROR: Cannot safely reshape 1D HOG features. Indices count {all_hog_subset_indices.shape[0]}, Feature len {all_hog_features.shape[0]}")
            return None
    elif all_hog_features.ndim != 2:
         print(f"ERROR: HOG features are not 2-dimensional (shape: {all_hog_features.shape})")
         return None

    if all_hog_features.shape[0] != all_hog_subset_indices.shape[0]:
        print(f"ERROR: Mismatch between HOG features ({all_hog_features.shape[0]}) and subset indices ({all_hog_subset_indices.shape[0]}) in HDF5")
        return None

    hog_feature_dim = all_hog_features.shape[1]
    print(f"  Found {all_hog_features.shape[0]} total HOG features with dimension {hog_feature_dim} in HDF5.")

    # Map subset indices present in the HDF5 to their row index in the HDF5 array
    try:
       all_hog_subset_indices_int = [int(i) for i in all_hog_subset_indices]
       hog_subset_idx_to_row_in_h5 = {subset_idx: i for i, subset_idx in enumerate(all_hog_subset_indices_int)}
    except (ValueError, TypeError) as e:
        print(f"ERROR: HOG subset indices from HDF5 seem to be of an invalid type: {e}")
        return None

    # Build a list to store the HOG features in the requested order
    selected_hog_features_ordered = []
    missing_count = 0
    placeholder = np.zeros(hog_feature_dim, dtype=all_hog_features.dtype)

    for requested_idx in requested_subset_indices:
        row_in_h5 = hog_subset_idx_to_row_in_h5.get(int(requested_idx))
        if row_in_h5 is not None:
            # Append the HOG feature vector from the HDF5 at that row index
            selected_hog_features_ordered.append(all_hog_features[row_in_h5, :])
        else:
            # Handle case where a requested index was not found in the HOG data (e.g., extraction failed for HOG for this image)
            # Append a zero vector of the expected dimension
            selected_hog_features_ordered.append(placeholder)
            missing_count += 1

    if missing_count > 0:
        print(f"  Warning: {missing_count}/{len(requested_subset_indices)} requested HOG features were not found in the HDF5 data. Used zero vectors.")


    if not selected_hog_features_ordered:
        print("  No HOG features were selected for the requested subset indices.")
        return np.empty((0, hog_feature_dim), dtype=np.float32)

    # Stack the selected HOG features into a single NumPy array, already in the requested order
    try:
        aligned_hog_array = np.vstack(selected_hog_features_ordered)
        print(f"  Aligned global HOG shape for requested subset indices: {aligned_hog_array.shape}")
        return aligned_hog_array
    except ValueError as e:
         print(f"ERROR: Could not stack selected HOG features: {e}")
         return None
    except Exception as e:
        print(f"Unexpected error stacking selected HOG features: {e}")
        return None
    finally:
        # Clean up the full loaded data
        del all_hog_features, all_hog_subset_indices
        gc.collect()


# --- 3. DMatrix Creation Function (Updated for Scaling and Weights) ---
def create_xgb_dmatrix_files(feature_combinations, subset_indices_to_load, corresponding_labels, set_type, output_dir,
                             sample_weights=None, perform_scaling=False, train_scaler_path=None, subset_size_from_map=None, seed_from_map=None): # Added subset_map params
    """
    Loads features by subset indices, concatenates, applies scaling (if requested), and saves DMatrix buffer.
    Can optionally include sample_weights for the training set.
    Handles fitting/saving scaler for train, loading/transforming for test.
    set_type should be 'train' or 'test'.
    """
    feature_desc = "_".join(feature_combinations)
    # Construct filename based on feature combination and set type
    filename_base = f"{set_type}_{feature_desc}"
    if set_type == 'train' and sample_weights is not None:
         filename_base += "_weighted"
    if perform_scaling:
         filename_base += "_scaled"
    # Add subset size and seed to filename for clarity
    if subset_size_from_map is not None and seed_from_map is not None:
        filename_base += f"_subset{subset_size_from_map}_seed{seed_from_map}"


    dmatrix_filename = os.path.join(output_dir, f"{filename_base}.buffer")
    scaler_filename_base = f"train_{feature_desc}"
    if subset_size_from_map is not None and seed_from_map is not None:
         scaler_filename_base += f"_subset{subset_size_from_map}_seed{seed_from_map}"
    scaler_filename = os.path.join(output_dir, f"{scaler_filename_base}_scaler.joblib")


    print(f"\n--- Creating DMatrix for: {feature_desc} ({set_type}) ---")
    print(f"Target DMatrix file: {dmatrix_filename}")
    print(f"  Loading/processing {len(subset_indices_to_load)} subset indices.")

    if os.path.exists(dmatrix_filename):
        print(f"DMatrix file already exists. Skipping creation.")
        # Check if scaler file also exists if scaling was requested for train
        if set_type == 'train' and perform_scaling and not os.path.exists(scaler_filename):
             print(f"WARNING: DMatrix exists but scaler file {scaler_filename} is missing.")
        return dmatrix_filename


    # --- Load Features using the new functions ---
    loaded_features = []

    print("Loading features for concatenation...")
    if "sift_spm" in feature_combinations:
        sift_spm = load_spm_features_by_subset_indices(BOVW_SPM_FEATURES_DIR, "sift", PYRAMID_LEVELS, requested_subset_indices=subset_indices_to_load, total_subset_size=subset_size_from_map, seed=seed_from_map)
        if sift_spm is None:
             print(f"ERROR: Failed to load SIFT SPM features for {set_type}.")
             return None
        if sift_spm.shape[0] != len(subset_indices_to_load):
             print(f"ERROR: Mismatch after loading SIFT SPM features for {set_type}. Expected {len(subset_indices_to_load)}, got {sift_spm.shape[0]}. Alignment issue?")
             return None
        loaded_features.append(sift_spm)

    if "orb_spm" in feature_combinations:
        orb_spm = load_spm_features_by_subset_indices(BOVW_SPM_FEATURES_DIR, "orb", PYRAMID_LEVELS, requested_subset_indices=subset_indices_to_load, total_subset_size=subset_size_from_map, seed=seed_from_map)
        if orb_spm is None:
             print(f"ERROR: Failed to load ORB SPM features for {set_type}.")
             return None
        if orb_spm.shape[0] != len(subset_indices_to_load):
             print(f"ERROR: Mismatch after loading ORB SPM features for {set_type}. Expected {len(subset_indices_to_load)}, got {orb_spm.shape[0]}. Alignment issue?")
             return None
        loaded_features.append(orb_spm)

    if "hog" in feature_combinations:
        hog = load_and_align_global_hog_by_subset_indices(HOG_DATA_FILE_SPM, requested_subset_indices=subset_indices_to_load)
        if hog is None:
             print(f"ERROR: Failed to load HOG features for {set_type}.")
             return None
        if hog.shape[0] != len(subset_indices_to_load):
             print(f"ERROR: Mismatch after loading HOG features for {set_type}. Expected {len(subset_indices_to_load)}, got {hog.shape[0]}. Alignment issue?")
             return None
        # Handle potential empty HOG feature dim if all images failed HOG or H5 was empty
        if hog.shape[1] == 0:
             print("Warning: HOG features have zero dimension after loading/aligning. Skipping HOG concatenation.")
        else:
             loaded_features.append(hog)


    if not loaded_features:
        print("ERROR: No valid features were specified or loaded for concatenation.")
        # Clean up any loaded features
        del loaded_features
        gc.collect()
        return None

    print("Concatenating features...")
    try:
        # Filter out any loaded features that somehow ended up with zero dimension
        loaded_features = [f for f in loaded_features if f.shape[1] > 0]
        if not loaded_features:
             print("ERROR: All loaded features had zero dimension after filtering. Cannot concatenate.")
             # Clean up any loaded features
             del loaded_features
             gc.collect()
             return None

        # Ensure all features have the same number of samples before concatenating
        ref_shape = loaded_features[0].shape[0]
        if not all(f.shape[0] == ref_shape for f in loaded_features):
            print("ERROR: Mismatched number of samples in features to concatenate:")
            for i, f in enumerate(loaded_features): print(f"  Feature {i}: {f.shape}")
            del loaded_features
            gc.collect()
            return None # Mismatch critical error

        if len(loaded_features) == 1:
            X_combined = loaded_features[0]
        else:
            X_combined = np.concatenate(loaded_features, axis=1)

        print(f"  Combined feature shape: {X_combined.shape}")
        if X_combined.shape[0] != len(corresponding_labels):
             print(f"CRITICAL ERROR: Final combined features shape ({X_combined.shape[0]}) doesn't match label count ({len(corresponding_labels)}). Alignment is broken!")
             del loaded_features, X_combined
             gc.collect()
             return None # This should ideally not happen if loading/filtering/alignment is correct

    except MemoryError:
        print("ERROR: Ran out of memory during feature concatenation.")
        del loaded_features
        if 'X_combined' in locals(): del X_combined
        gc.collect()
        print("Suggestion: Reduce subset size or batch size.")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error during concatenation: {e}")
        # print(traceback.format_exc()) # Optional detailed traceback
        if 'loaded_features' in locals(): del loaded_features
        if 'X_combined' in locals(): del X_combined
        gc.collect()
        return None
    finally:
         # Ensure individual loaded features are deleted to free memory
         if 'loaded_features' in locals(): del loaded_features
         gc.collect()


    # --- Apply Scaling ---
    # Scaling happens on the combined feature matrix
    if perform_scaling:
        print(f"Applying StandardScaler to features for {set_type}...")
        try:
            if set_type == 'train':
                 scaler = StandardScaler()
                 X_combined_scaled = scaler.fit_transform(X_combined)
                 print(f"  Fitted and transformed training data. Saving scaler to {scaler_filename}")
                 joblib.dump(scaler, scaler_filename)
            else: # set_type == 'test'
                 if train_scaler_path and os.path.exists(train_scaler_path):
                     print(f"  Loading scaler from {train_scaler_path} and transforming test data.")
                     scaler = joblib.load(train_scaler_path)
                     X_combined_scaled = scaler.transform(X_combined)
                 else:
                     print(f"WARNING: Scaling requested for test set ({set_type}), but train_scaler_path was not provided or file not found: {train_scaler_path}. Skipping scaling for test set.")
                     X_combined_scaled = X_combined # Use unscaled data
                     perform_scaling = False # Update flag locally for logging filename
            
            X_combined = X_combined_scaled # Use the scaled data
            print(f"  Scaled feature shape: {X_combined.shape}")
            if 'scaler' in locals(): del scaler # Clean up scaler object
            if 'X_combined_scaled' in locals(): del X_combined_scaled # Clean up temp scaled data
            gc.collect()

        except Exception as e:
            print(f"ERROR during scaling: {e}. Proceeding with UNscaled data.")
            # print(traceback.format_exc()) # Optional detailed traceback
            perform_scaling = False # Ensure we don't try to save/use scaled DMatrix suffix
            # X_combined remains unscaled
            if 'scaler' in locals(): del scaler
            if 'X_combined_scaled' in locals(): del X_combined_scaled
            gc.collect()


    # --- Create and Save DMatrix ---
    print("Creating XGBoost DMatrix...")
    try:
        # Ensure X_combined is C-contiguous, especially after scaling/concatenation
        if not X_combined.flags['C_CONTIGUOUS']:
             print("Warning: Feature array is not C-contiguous. Converting.")
             X_combined = np.ascontiguousarray(X_combined)

        # Pass weights if provided (only for train DMatrix)
        # Labels must be float32 for DMatrix
        dmatrix = xgb.DMatrix(X_combined, label=corresponding_labels.astype(np.float32), weight=sample_weights if set_type == 'train' else None)

        print("Saving DMatrix to buffer file...")
        # Reconstruct filename base to reflect if scaling actually happened and include subset info
        filename_base_final = f"{set_type}_{feature_desc}"
        if set_type == 'train' and sample_weights is not None:
            filename_base_final += "_weighted"
        if perform_scaling: # Check the *final* state of perform_scaling
            filename_base_final += "_scaled"
        if subset_size_from_map is not None and seed_from_map is not None:
            filename_base_final += f"_subset{subset_size_from_map}_seed{seed_from_map}"

        dmatrix_filename_final = os.path.join(output_dir, f"{filename_base_final}.buffer")

        dmatrix.save_binary(dmatrix_filename_final)
        print(f"Successfully saved DMatrix to {dmatrix_filename_final}")

        del X_combined # Explicit cleanup of feature array
        if 'dmatrix' in locals(): del dmatrix # Clean up DMatrix object
        gc.collect()

        # Return the final path where the DMatrix was saved
        return dmatrix_filename_final

    except Exception as e:
        print(f"ERROR: Failed to create or save DMatrix: {e}")
        # print(traceback.format_exc()) # Optional detailed traceback
        # Clean up any variables that might exist after error
        if 'X_combined' in locals(): del X_combined
        if 'dmatrix' in locals(): del dmatrix
        gc.collect()
        return None


# --- 4. Training Functions (Remains largely the same, uses DMatrix) ---
# The functions find_best_params_with_gridsearch_on_sample and train_and_evaluate_xgb_dmatrix
# seem robust already and work with scaled/weighted DMatrices and sample data for tuning.
# They should not need significant changes, except potentially passing subset size/seed to filename logging.

def find_best_params_with_gridsearch_on_sample(
    X_train_sample_unscaled, y_train_sample, num_classes,
    base_params, param_grid, cv_folds, feature_type_desc, scoring_metric):
    # ... (This function remains largely the same as in your previous script) ...
    # Ensure it correctly takes X_train_sample_unscaled, scales it internally,
    # calculates sample_weights for the sample labels, and passes weights to grid_search.fit.
    # The current implementation looks good for this purpose.

    print(f"\n--- Performing GridSearchCV on SCALED SAMPLE for {feature_type_desc} ---")
    # Sample fraction info is not available here, assume X_train_sample_unscaled/y_train_sample are already the sample
    print(f"Sample size: {X_train_sample_unscaled.shape[0]} instances")
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
        del X_train_sample_unscaled # Clean up unscaled sample data
        gc.collect()
    except Exception as e:
         print(f"ERROR scaling sample data for GridSearchCV: {e}")
         print("Cannot proceed with GridSearchCV.")
         return None


    # --- Calculate Sample Weights ---
    try:
         sample_class_weights_dict = compute_class_weight(
             class_weight='balanced',
             classes=np.arange(num_classes),
             y=y_train_sample
         )
         sample_weights = np.array([sample_class_weights_dict[label] for label in y_train_sample])
         print(f"Calculated sample weights for grid search sample.")
    except Exception as e:
         print(f"ERROR calculating sample weights: {e}")
         print("Cannot proceed with weighted grid search.")
         del X_train_sample_scaled
         gc.collect()
         return None

    best_params_found = None
    best_score = -1.0
    used_gpu = False

    # --- Try GPU First ---
    try:
        print("\nAttempting GridSearchCV with GPU...")
        gpu_params = base_params.copy()
        gpu_params['device'] = 'cuda'
        gpu_params['num_class'] = num_classes
        gpu_params['objective'] = 'multi:softprob'
        if 'tree_method' not in gpu_params or gpu_params['tree_method'] not in ['hist', 'gpu_hist']:
             gpu_params['tree_method'] = 'hist'

        estimator_gpu = xgb.XGBClassifier(**gpu_params)
        xgb_grid_search_gpu = GridSearchCV(estimator=estimator_gpu, param_grid=param_grid,
                                           scoring=scoring_metric, cv=cv_folds, verbose=2, n_jobs=1)

        print(f"Starting GridSearchCV fitting (GPU), optimizing for '{scoring_metric}'...")
        xgb_grid_search_gpu.fit(X_train_sample_scaled, y_train_sample, sample_weight=sample_weights)

        print("GridSearchCV with GPU successful.")
        used_gpu = True
        best_params_found = xgb_grid_search_gpu.best_params_
        best_score = xgb_grid_search_gpu.best_score_

    except (xgb.core.XGBoostError, Exception) as gpu_err:
        print(f"\nWARNING: GridSearchCV with GPU failed: {gpu_err}")
        print("Falling back to CPU for GridSearchCV.")
        used_gpu = False
        if 'estimator_gpu' in locals(): del estimator_gpu
        if 'xgb_grid_search_gpu' in locals(): del xgb_grid_search_gpu
        gc.collect()

    # --- Fallback to CPU ---
    if not used_gpu:
        try:
            print("\nAttempting GridSearchCV with CPU...")
            cpu_params = base_params.copy()
            cpu_params['num_class'] = num_classes
            cpu_params['objective'] = 'multi:softprob'
            if 'device' in cpu_params: del cpu_params['device']
            cpu_params['tree_method'] = 'hist'

            estimator_cpu = xgb.XGBClassifier(**cpu_params)
            xgb_grid_search_cpu = GridSearchCV(estimator=estimator_cpu, param_grid=param_grid,
                                              scoring=scoring_metric, cv=cv_folds, verbose=2, n_jobs=-1)

            print(f"Starting GridSearchCV fitting (CPU), optimizing for '{scoring_metric}'...")
            xgb_grid_search_cpu.fit(X_train_sample_scaled, y_train_sample, sample_weight=sample_weights)

            print("GridSearchCV with CPU successful.")
            best_params_found = xgb_grid_search_cpu.best_params_
            best_score = xgb_grid_search_cpu.best_score_

        except Exception as cpu_err:
            print(f"\nERROR: GridSearchCV with CPU also failed: {cpu_err}")
            print(traceback.format_exc())
            del X_train_sample_scaled
            gc.collect()
            return None

    # --- Combine and Return ---
    del X_train_sample_scaled
    gc.collect()

    if best_params_found is not None:
        print(f"\nGridSearchCV Complete for {feature_type_desc}.")
        print(f"  Used Device for Grid Search: {'GPU' if used_gpu else 'CPU'}")
        print(f"  Best parameters found (on sample): {best_params_found}")
        print(f"  Best CV score ({scoring_metric} on sample): {best_score:.4f}")

        final_params = base_params.copy()
        final_params.update(best_params_found)
        if used_gpu: final_params['device'] = 'cuda'
        elif 'device' in final_params: del final_params['device']

        final_params['_gridsearch_scoring'] = scoring_metric

        return final_params
    else:
        print(f"ERROR: Could not determine best parameters for {feature_type_desc}.")
        return None


def train_and_evaluate_xgb_dmatrix(dtrain_path, dtest_path, y_test_labels,
                                  best_params_from_search,
                                  feature_type_desc, target_class_names,
                                  output_results_dir):
    # ... (This function remains largely the same, uses DMatrix, saves model, evaluates) ...
    # It already handles GPU/CPU fallback for the final xgb.train.
    # Ensure the confusion matrix plotting and results saving include the F1-macro score.

    print(f"\n--- Training FINAL XGBoost on FULL data for {feature_type_desc} using DMatrix ---")
    print(f"Train DMatrix: {dtrain_path}")
    print(f"Test DMatrix: {dtest_path}")

    if not os.path.exists(dtrain_path) or not os.path.exists(dtest_path):
        print("ERROR: DMatrix train or test file not found. Skipping.")
        return None

    bst = None
    final_params_used = {}
    gridsearch_scoring_metric = best_params_from_search.get('_gridsearch_scoring', 'accuracy')

    try:
        print("Loading DMatrix files...")
        dtrain = xgb.DMatrix(dtrain_path)
        dtest = xgb.DMatrix(dtest_path)
        print("DMatrix files loaded.")
        dtest.set_label(y_test_labels.astype(np.float32)) # Set labels on test DMatrix for evaluation

        final_params_to_try = best_params_from_search.copy()
        if '_gridsearch_scoring' in final_params_to_try: del final_params_to_try['_gridsearch_scoring']


        # Determine final device based on grid search result preference, but check availability
        preferred_device = final_params_to_try.get('device', 'cpu')
        actual_device = 'cpu'

        # Add a check to see if any GPUs are visible to XGBoost before attempting CUDA
        if preferred_device == 'cuda':
            try:
                 # Attempt a small test train on GPU to check availability
                 # This is more reliable than just checking 'cuda' in device list
                 temp_params = final_params_to_try.copy()
                 if 'device' in temp_params: del temp_params['device']
                 # Ensure temp_dmatrix has consistent columns with dtrain/dtest
                 temp_dmatrix = xgb.DMatrix(np.random.rand(2, dtrain.num_col()), label=np.random.randint(0, NUM_CLASSES, 2))
                 print("Testing GPU availability for final training...")
                 temp_gpu_model = xgb.train(temp_params, temp_dmatrix, num_boost_round=1, evals=[(temp_dmatrix, 'eval')], tree_method='hist', device='cuda')
                 del temp_gpu_model, temp_dmatrix
                 gc.collect()
                 actual_device = 'cuda'
                 print(f"CUDA device detected and available for final training.")

            except xgb.core.XGBoostError as e:
                 print(f"\nWARNING: Preferred GPU training failed ({e}). CUDA device may not be available or configured. Falling back to CPU.")
                 actual_device = 'cpu'
                 if 'device' in final_params_to_try: del final_params_to_try['device'] # Ensure no device param for CPU

        if actual_device == 'cuda':
             final_params_to_try['device'] = 'cuda'
             if 'tree_method' not in final_params_to_try or final_params_to_try['tree_method'] not in ['hist', 'gpu_hist']:
                      final_params_to_try['tree_method'] = 'hist' # hist is compatible with CUDA
                      print("Setting tree_method to 'hist' for GPU.")
             final_params_used = final_params_to_try.copy() # Store the params used
             used_gpu_in_final_train = True

        else: # Use CPU
             if 'device' in final_params_to_try: del final_params_to_try['device']
             final_params_to_try['tree_method'] = 'hist' # hist is good for CPU too
             final_params_used = final_params_to_try.copy() # Store the params used
             used_gpu_in_final_train = False


        print(f"Starting final XGBoost training ({actual_device}) with params: {final_params_used}")
        evals = [(dtrain, 'train'), (dtest, 'eval')]
        num_boost_round = final_params_used.get('n_estimators', 300) # Use n_estimators from best params
        # Remove n_estimators from the params passed to xgb.train if present
        train_params = final_params_used.copy()
        if 'n_estimators' in train_params: del train_params['n_estimators']


        bst = xgb.train(
            train_params, dtrain, num_boost_round=num_boost_round, evals=evals,
            early_stopping_rounds=50, verbose_eval=100
        )
        print("Final training successful.")


        # --- Evaluation (if training succeeded) ---
        if bst is not None:
            print("\nEvaluating final model on test set...")
            # Model filename should include subset info for clarity
            model_filename_base = f'xgb_model_{feature_type_desc.replace(" ", "_").replace("/", "-")}'
            if MAP_SUBSET_SIZE is not None and MAP_RANDOM_SEED is not None:
                 model_filename_base += f"_subset{MAP_SUBSET_SIZE}_seed{MAP_RANDOM_SEED}"
            model_filename = os.path.join(output_results_dir, f'{model_filename_base}.json')
            bst.save_model(model_filename) # Use XGBoost's native save
            # joblib.dump(bst, model_filename.replace(".json", ".joblib")) # Optional: save with joblib too
            print(f"Saved final XGBoost model ({actual_device}) for {feature_type_desc} to {model_filename}")


            # Predict on test set using the best number of iterations found by early stopping
            y_pred_proba = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
            y_pred_labels = np.argmax(y_pred_proba, axis=1)


            # Ensure predicted labels match the possible classes
            # Sometimes models predict outside the range 0..N-1, although less common with softmax objective
            # predicted_classes_valid = np.clip(y_pred_labels, 0, NUM_CLASSES - 1) # Optional clipping


            if len(y_test_labels) != len(y_pred_labels):
                 print(f"CRITICAL ERROR: Mismatch between true test labels ({len(y_test_labels)}) and predicted test labels ({len(y_pred_labels)}). Cannot reliably calculate metrics.")
                 # This indicates a fundamental issue earlier in the pipeline or in DMatrix creation/prediction.
                 return None # Halt evaluation

            accuracy_val = accuracy_score(y_test_labels, y_pred_labels)
            # Ensure target_names match the actual labels present (0..NUM_CLASSES-1)
            class_report_str = classification_report(y_test_labels, y_pred_labels, target_names=target_class_names, labels=np.arange(NUM_CLASSES), zero_division=0)
            conf_matrix = confusion_matrix(y_test_labels, y_pred_labels, labels=np.arange(NUM_CLASSES))
            f1_macro_val = f1_score(y_test_labels, y_pred_labels, average='macro', zero_division=0)


            print(f"Final Model Used Device: {actual_device}")
            print(f"Test Set Accuracy: {accuracy_val:.4f}")
            print(f"Test Set F1-macro: {f1_macro_val:.4f}")
            print(f"Classification Report (XGBoost - {feature_type_desc}):\n{class_report_str}")

            # Plot confusion matrix - include F1-macro in title
            cm_title = f'CM XGB ({actual_device}) {feature_type_desc} (Acc: {accuracy_val:.3f}, F1-M: {f1_macro_val:.3f})'
            # Filename for CM plot should also include subset info
            cm_filename_base = f'cm_xgb_{feature_type_desc.replace(" ", "_").replace("/", "-")}'
            if MAP_SUBSET_SIZE is not None and MAP_RANDOM_SEED is not None:
                 cm_filename_base += f"_subset{MAP_SUBSET_SIZE}_seed{MAP_RANDOM_SEED}"
            cm_filename = f'{cm_filename_base}.png'

            plot_confusion_matrix(conf_matrix, classes=target_class_names,
                                  plot_title=cm_title,
                                  results_path=output_results_dir,
                                  filename=cm_filename)

            # Save results text file - include F1-macro
            results_text_file_base = f'results_xgb_{feature_type_desc.replace(" ", "_").replace("/", "-")}'
            if MAP_SUBSET_SIZE is not None and MAP_RANDOM_SEED is not None:
                 results_text_file_base += f"_subset{MAP_SUBSET_SIZE}_seed{MAP_RANDOM_SEED}"
            results_text_file = os.path.join(output_results_dir, f'{results_text_file_base}.txt')

            with open(results_text_file, 'w') as f:
                f.write(f"--- XGBoost Results for {feature_type_desc} ---\n")
                f.write(f"Subset Size: {MAP_SUBSET_SIZE}, Seed: {MAP_RANDOM_SEED}\n")
                f.write(f"Trained using DMatrix files.\n")
                f.write(f"GridSearchCV Scoring Metric (on Sample): {gridsearch_scoring_metric}\n")
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
            del dtrain, dtest
            gc.collect()

            return bst # Return the trained model
        else:
            print("ERROR: Training failed, no model to evaluate.")
            if 'dtrain' in locals(): del dtrain
            if 'dtest' in locals(): del dtest
            gc.collect()
            return None

    except Exception as e:
        print(f"ERROR: An unexpected error occurred during DMatrix loading/training/evaluation for {feature_type_desc}: {e}")
        print(traceback.format_exc())
        if 'dtrain' in locals(): del dtrain
        if 'dtest' in locals(): del dtest
        gc.collect()
        return None


# --- 5. Main Execution Pipeline ---
def run_spm_classification_pipeline():
    print("\n--- Starting SPM Classification Pipeline with DMatrix, Scaling, and Alignment ---")

    # --- Step 0: Data Loading and Alignment (Already done at the top) ---
    # The data (actual_train_subset_indices, y_train, actual_test_subset_indices, y_test)
    # is loaded and filtered at the beginning of the script using the NPZ and CSV map.
    # MAP_SUBSET_SIZE and MAP_RANDOM_SEED are also available from the map filename parsing.
    # These variables are already correctly scoped at the module level before this function.

    # --- Create Sampled Data for GridSearchCV Tuning ---
    # Sample the *actual* train indices and labels (which are already filtered and aligned)
    # This sample will be used *only* for GridSearchCV tuning.
    if SAMPLE_FRACTION_FOR_GRIDSEARCH < 1:
        print(f"\nCreating a {SAMPLE_FRACTION_FOR_GRIDSEARCH*100:.1f}% stratified sample indices from the actual train set for GridSearchCV...")
        try:
            # train_test_split on the actual_train_subset_indices and their corresponding y_train labels
            train_subset_indices_sample, _, y_train_sample, _ = train_test_split(
                actual_train_subset_indices, y_train, # Use the filtered actual data here
                train_size=SAMPLE_FRACTION_FOR_GRIDSEARCH,
                random_state=42, # Use a consistent random seed for sampling
                stratify=y_train # Stratify the sample using the actual train labels
            )
            print(f"Sample size for tuning: {len(train_subset_indices_sample)}")
            # y_train_sample is already a numpy array from train_test_split
            # train_subset_indices_sample is a list

        except ValueError as e:
            print(f"Error during train_test_split for sampling: {e}")
            print("Ensure SAMPLE_FRACTION_FOR_GRIDSEARCH is large enough for stratification.")
            return # Exit pipeline if sampling fails
    else:
        print("\nUsing the FULL actual training set for GridSearchCV (SAMPLE_FRACTION_FOR_GRIDSEARCH=1).")
        train_subset_indices_sample = actual_train_subset_indices # Use the full list of actual train indices
        y_train_sample = y_train # Use the full actual train labels (numpy array)
        print(f"Sample size for tuning: {len(train_subset_indices_sample)} (Full set)")

    # --- Calculate Full Training Set Sample Weights (for DMatrix) ---
    # Calculate these AFTER y_train (the filtered training labels) is finalized.
    # These weights are for the DMatrix used in the final xgb.train step on the full actual training data.
    print("\nCalculating full training set sample weights for DMatrix (based on actual train labels)...")
    try:
        full_train_class_weights_dict = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(NUM_CLASSES),
            y=y_train # *** Use the filtered y_train labels here ***
        )
        # XGBoost DMatrix weight parameter expects a weight for each sample, not per class
        full_train_sample_weights = np.array([full_train_class_weights_dict[label] for label in y_train]) # *** Use the filtered y_train labels here ***
        print("Full training set sample weights calculated.")
        # print(f"Weights: {full_train_class_weights_dict}") # Optional: print weights
        # print(f"Sample weights (first 10): {full_train_sample_weights[:10]}") # Optional: print sample weights
    except Exception as e:
        print(f"ERROR calculating full training set class weights: {e}")
        print("Cannot proceed.")
        return # Exit if weight calculation fails


    # --- Define Feature Combinations to Test ---
    feature_sets_to_run = {
        f"SPM_SIFT_L{PYRAMID_LEVELS-1}": ["sift_spm"],
        f"SPM_ORB_L{PYRAMID_LEVELS-1}": ["orb_spm"],
        "Global_HOG": ["hog"],
        f"SPM_SIFT_L{PYRAMID_LEVELS-1}_HOG": ["sift_spm", "hog"],
        f"SPM_ORB_L{PYRAMID_LEVELS-1}_HOG": ["orb_spm", "hog"],
        f"SPM_SIFT_ORB_L{PYRAMID_LEVELS-1}_HOG": ["sift_spm", "orb_spm", "hog"],
    }


    # --- Loop Through Feature Combinations ---
    all_best_params = {}

    for feature_desc, features_to_combine in feature_sets_to_run.items():
        print(f"\n\n{'='*20} Processing Feature Set: {feature_desc} {'='*20}")

        # --- Step 2: Load Sample Data (UNSCALED) for GridSearchCV Tuning ---
        # Load the features for the SAMPLE indices using the *new* loading functions
        print(f"\nLoading sample data ({len(train_subset_indices_sample)} instances) into memory for GridSearchCV...")
        sample_features_list_unscaled = []
        sample_loading_success = True

        if "sift_spm" in features_to_combine:
             # Pass the sample subset indices to the new loading function
             sift_spm_sample_unscaled = load_spm_features_by_subset_indices(BOVW_SPM_FEATURES_DIR, "sift", PYRAMID_LEVELS, requested_subset_indices=train_subset_indices_sample, total_subset_size=MAP_SUBSET_SIZE, seed=MAP_RANDOM_SEED)
             if sift_spm_sample_unscaled is None or sift_spm_sample_unscaled.shape[0] != len(train_subset_indices_sample):
                  print("Failed to load/align SIFT for sample gridsearch."); sample_loading_success = False
             elif sift_spm_sample_unscaled.shape[1] > 0: # Only add if feature dim > 0
                  sample_features_list_unscaled.append(sift_spm_sample_unscaled)

        if "orb_spm" in features_to_combine and sample_loading_success:
             # Pass the sample subset indices to the new loading function
             orb_spm_sample_unscaled = load_spm_features_by_subset_indices(BOVW_SPM_FEATURES_DIR, "orb", PYRAMID_LEVELS, requested_subset_indices=train_subset_indices_sample, total_subset_size=MAP_SUBSET_SIZE, seed=MAP_RANDOM_SEED)
             if orb_spm_sample_unscaled is None or orb_spm_sample_unscaled.shape[0] != len(train_subset_indices_sample):
                  print("Failed to load/align ORB for sample gridsearch."); sample_loading_success = False
             elif orb_spm_sample_unscaled.shape[1] > 0: # Only add if feature dim > 0
                  sample_features_list_unscaled.append(orb_spm_sample_unscaled)


        if "hog" in features_to_combine and sample_loading_success:
             # Pass the sample subset indices to the new HOG loading function
             hog_sample_unscaled = load_and_align_global_hog_by_subset_indices(HOG_DATA_FILE_SPM, requested_subset_indices=train_subset_indices_sample)
             if hog_sample_unscaled is None or hog_sample_unscaled.shape[0] != len(train_subset_indices_sample):
                  print("Failed to load/align HOG for sample gridsearch."); sample_loading_success = False
             elif hog_sample_unscaled.shape[1] > 0: # Only add if feature dim > 0
                  sample_features_list_unscaled.append(hog_sample_unscaled)


        if not sample_loading_success:
             print(f"Skipping {feature_desc} due to failure loading/aligning sample data for GridSearchCV.")
             # Clean up partial loads
             if 'sift_spm_sample_unscaled' in locals(): del sift_spm_sample_unscaled
             if 'orb_spm_sample_unscaled' in locals(): del orb_spm_sample_unscaled
             if 'hog_sample_unscaled' in locals(): del hog_sample_unscaled
             if 'sample_features_list_unscaled' in locals(): del sample_features_list_unscaled
             gc.collect()
             continue


        # Concatenate sample features for GridSearchCV
        X_train_sample_combined_unscaled = None
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
             if 'sift_spm_sample_unscaled' in locals(): del sift_spm_sample_unscaled
             if 'orb_spm_sample_unscaled' in locals(): del orb_spm_sample_unscaled
             if 'hog_sample_unscaled' in locals(): del hog_sample_unscaled
             if 'sample_features_list_unscaled' in locals(): del sample_features_list_unscaled
             gc.collect()


        # --- Step 3: Perform GridSearchCV on the SCALED SAMPLE ---
        if X_train_sample_combined_unscaled is not None:
            best_params = find_best_params_with_gridsearch_on_sample(
                X_train_sample_combined_unscaled, y_train_sample, NUM_CLASSES,
                XGB_BASE_PARAMS, PARAM_GRID_XGB, GRIDSEARCH_CV_FOLDS, feature_desc,
                scoring_metric=GRIDSEARCH_SCORING
            )
             # X_train_sample_combined_unscaled is cleaned up inside find_best_params...
        else:
            best_params = None # Cannot perform grid search if sample data failed to load/combine


        if best_params is None:
            print(f"GridSearchCV failed for {feature_desc}. Skipping final training.")
            continue

        all_best_params[feature_desc] = best_params

        # --- Step 4: Create DMatrix files for the FULL dataset (Scaled & Weighted) ---
        # Use the *actual* train/test subset indices and labels for the full DMatrix
        print(f"\nCreating DMatrix files for the full training and test sets...")

        # Train DMatrix - FIT & TRANSFORM StandardScaler + apply weights + save scaler
        # Pass the *filtered* actual train indices (actual_train_subset_indices) and labels (y_train)
        # and the *filtered* actual sample weights (full_train_sample_weights)
        dtrain_path = create_xgb_dmatrix_files(
            features_to_combine, actual_train_subset_indices, y_train, set_type='train', output_dir=DMATRIX_CACHE_DIR,
            sample_weights=full_train_sample_weights, perform_scaling=True, subset_size_from_map=MAP_SUBSET_SIZE, seed_from_map=MAP_RANDOM_SEED
        )

        # Derive the expected path for the scaler that was just saved for the train set
        train_scaler_path_for_test = None
        if dtrain_path: # Only derive scaler path if train DMatrix was successfully created
            # Reconstruct filename base for scaler based on train DMatrix name without suffixes
            train_dmatrix_base_name = os.path.basename(dtrain_path).replace(".buffer", "")
            scaler_name_parts = train_dmatrix_base_name.split('_')
            # Remove 'train_' prefix and potential suffixes like '_weighted', '_scaled', subset/seed
            base_for_scaler = "_".join(scaler_name_parts[1:]) # Everything after 'train_'
            base_for_scaler = base_for_scaler.replace("_weighted", "").replace("_scaled", "")
            # If subset/seed were included, remove them from the base before adding back at the end
            if MAP_SUBSET_SIZE is not None and MAP_RANDOM_SEED is not None:
                 subset_seed_suffix = f"_subset{MAP_SUBSET_SIZE}_seed{MAP_RANDOM_SEED}"
                 if base_for_scaler.endswith(subset_seed_suffix):
                     base_for_scaler = base_for_scaler[:-len(subset_seed_suffix)]
                 base_for_scaler += subset_seed_suffix # Add it back in the consistent position

            train_scaler_path_for_test = os.path.join(DMATRIX_CACHE_DIR, f"train_{base_for_scaler}_scaler.joblib")
            print(f"Expected train scaler path for test set: {train_scaler_path_for_test}")


        # Test DMatrix - TRANSFORM ONLY using the fitted scaler + NO weights
        # Pass the *filtered* actual test indices (actual_test_subset_indices) and labels (y_test)
        dtest_path = create_xgb_dmatrix_files(
            features_to_combine, actual_test_subset_indices, y_test, set_type='test', output_dir=DMATRIX_CACHE_DIR,
            sample_weights=None, perform_scaling=True, train_scaler_path=train_scaler_path_for_test, # Pass the fitted scaler path
            subset_size_from_map=MAP_SUBSET_SIZE, seed_from_map=MAP_RANDOM_SEED # Pass subset map info
        )


        if dtrain_path is None or dtest_path is None:
            print(f"Skipping final training for {feature_desc} due to DMatrix creation failure.")
            continue

        # --- Step 5: Train final model on FULL data using DMatrix ---
        # The DMatrix files created in Step 4 are already scaled and weighted.
        train_and_evaluate_xgb_dmatrix(
            dtrain_path, dtest_path, y_test, # Pass the actual test labels (y_test)
            best_params, # Use the params found by grid search
            feature_desc, class_names, RESULTS_DIR_XGB_SPM
        )

    print("\n--- SPM Classification Pipeline Complete ---")
    print("Best parameters found (from sample grid search):")
    for name, params in all_best_params.items():
        display_params = params.copy()
        if '_gridsearch_scoring' in display_params: del display_params['_gridsearch_scoring']
        print(f"  {name}: {display_params}")
    print(f"\nResults saved to: {RESULTS_DIR_XGB_SPM}")