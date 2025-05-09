import numpy as np
import os
import pickle
import warnings
import joblib
import h5py # For loading HOG data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler # Optional for XGBoost
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# --- Configuration ---
# SPM Features Base Directory
FEATURES_DIR_SPM = "E:\CV_features_SPM"
BOVW_SPM_FEATURES_DIR = os.path.join(FEATURES_DIR_SPM, "bovw_spm_features_4cat")

# HOG Data File (assuming it might be shared or also in SPM features dir)
# If HOG was re-extracted with SPM data or you have a specific HOG file for SPM runs:
# HOG_DATA_FILE = os.path.join(FEATURES_DIR_SPM, 'hog_data_spm.h5')
# If HOG is from the original vanilla run and is being reused:
FEATURES_DIR_VANILLA_FOR_HOG = "E:\CV_features" # Only if HOG is from vanilla features
HOG_DATA_FILE = os.path.join(FEATURES_DIR_VANILLA_FOR_HOG, 'hog_data.h5')


# Shared Splits and Label Info (assuming these are common across experiments)
SPLITS_DIR_COMMON = os.path.join("E:\CV_features", "train_test_splits_4cat_revised") # Path to where NPZ/PKL are
NPZ_FILE = os.path.join(SPLITS_DIR_COMMON, "train_test_split_data_4cat_revised.npz")
LABEL_ENCODER_FILE = os.path.join(SPLITS_DIR_COMMON, "broad_label_encoder_4cat_revised.pkl")

# Results Directory for SPM + XGBoost
RESULTS_DIR_XGB_SPM = os.path.join(FEATURES_DIR_SPM, "classification_results_XGB_SPM_SOH_4cat")
os.makedirs(RESULTS_DIR_XGB_SPM, exist_ok=True)

# Feature Parameters (for naming files)
VOCAB_SIZE = 1000 # Used if your SPM filenames incorporate it (though less common for SPM output name)
PYRAMID_LEVELS = 3 # Number of pyramid levels (e.g., 3 for L0, L1, L2, so max index is L-1 = 2)

warnings.filterwarnings("ignore", message="Parameters: {.*use_label_encoder.*} are not used.", category=UserWarning, module="xgboost.core")

# --- 1. Load Labels, Indices, and Label Encoder ---
print("--- Loading Common Data (Labels, Splits, Encoder) ---")
print(f"Loading train/test split data from: {NPZ_FILE}")
try:
    split_data = np.load(NPZ_FILE)
    train_indices = split_data['train_indices']
    test_indices = split_data['test_indices']
    y_train = split_data['train_labels_numeric']
    y_test = split_data['test_labels_numeric']
except FileNotFoundError:
    print(f"ERROR: NPZ file not found at {NPZ_FILE}. Ensure label splitting script has run.")
    exit()
except KeyError as e:
    print(f"ERROR: Missing key {e} in NPZ file {NPZ_FILE}. Check keys.")
    exit()

print(f"Loaded {len(train_indices)} train indices and {len(y_train)} train labels.")
print(f"Loaded {len(test_indices)} test indices and {len(y_test)} test labels.")
if len(train_indices) != len(y_train) or len(test_indices) != len(y_test):
    print("ERROR: Mismatch between number of indices and labels. Halting.")
    exit()

print(f"Loading label encoder from: {LABEL_ENCODER_FILE}")
try:
    with open(LABEL_ENCODER_FILE, 'rb') as f:
        label_encoder = pickle.load(f)
    class_names = label_encoder.classes_
    print(f"Class names for classification: {class_names}")
    if len(class_names) != 4:
        print(f"Warning: Expected 4 class names, got {len(class_names)}.")
except FileNotFoundError:
    print(f"ERROR: Label encoder file not found at {LABEL_ENCODER_FILE}.")
    exit()


# --- 2. Helper Functions (plot_confusion_matrix, train_and_evaluate_xgb) ---
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

def train_and_evaluate_xgb(X_train_data, y_train_labels, X_test_data, y_test_labels,
                           feature_type_desc, target_class_names,
                           output_results_dir, perform_scaling=False): # Scaling off by default for XGB
    print(f"\n--- Training XGBoost for {feature_type_desc} ---")
    if X_train_data is None or X_train_data.size == 0 or X_test_data is None or X_test_data.size == 0:
        print(f"Skipping XGBoost for {feature_type_desc}: Missing/empty feature data.")
        return None

    X_train_processed = X_train_data.copy()
    X_test_processed = X_test_data.copy()

    # Initialize XGBoost Classifier
    try:
        base_estimator_xgb = xgb.XGBClassifier(
            objective='multi:softprob',     # Output probabilities for each class
            num_class=len(target_class_names), # Required for multi:softprob if labels aren't 0..N-1 or for clarity
            tree_method='hist',             # Efficient for large datasets, supports GPU
            device='cuda',                  # Specify GPU usage
            eval_metric='mlogloss',         # Logarithmic loss for multiclass problems
            random_state=42,
            use_label_encoder=False       # Suppress deprecation warning for newer XGBoost
        )
    except xgb.core.XGBoostError as e:
        if "Cannot find CUDA device" in str(e) or "No GPU found" in str(e):
            print("XGBoost CUDA device not found. Falling back to CPU for XGBoost.")
            base_estimator_xgb = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=len(target_class_names),
                tree_method='hist', # 'hist' is also good for CPU
                eval_metric='mlogloss',
                random_state=42,
                use_label_encoder=False
            )
        else:
            print(f"An unexpected error occurred initializing XGBoost: {e}")
            return None

    # Optional Scaling (usually not critical for XGBoost)
    if perform_scaling:
        print(f"Scaling features for XGBoost: {feature_type_desc}...")
        scaler_xgb = StandardScaler()
        X_train_processed = scaler_xgb.fit_transform(X_train_processed)
        X_test_processed = scaler_xgb.transform(X_test_processed)
        # scaler_filename_xgb = os.path.join(output_results_dir, f'scaler_xgb_{feature_type_desc.replace(" ", "_")}.joblib')
        # joblib.dump(scaler_xgb, scaler_filename_xgb)
        # print(f"Saved XGBoost scaler for {feature_type_desc} to {scaler_filename_xgb}")

    # Reduced parameter grid for faster tuning
    param_grid_xgb = {
        'n_estimators': [200, 500],        # Number of trees
        'learning_rate': [0.05, 0.1],     # Step size shrinkage
        'max_depth': [5, 8],              # Maximum depth of a tree
        # Add other parameters if needed, e.g., 'subsample', 'colsample_bytree'
    }

    print(f"Performing GridSearchCV for XGBoost on {feature_type_desc} (cv=3)...")
    # For GPU, n_jobs=1 in GridSearchCV is often best if XGBoost itself uses the GPU fully.
    xgb_grid_search = GridSearchCV(estimator=base_estimator_xgb,
                                   param_grid=param_grid_xgb,
                                   scoring='accuracy', # Or 'f1_macro', 'roc_auc_ovr', etc.
                                   cv=3,
                                   verbose=2,
                                   n_jobs=1) # Set to 1 when XGBoost is using GPU
    
    xgb_grid_search.fit(X_train_processed, y_train_labels)

    best_xgb_model = xgb_grid_search.best_estimator_
    print(f"Best XGBoost parameters for {feature_type_desc}: {xgb_grid_search.best_params_}")

    # Save model
    model_filename_xgb = os.path.join(output_results_dir, f'xgb_model_{feature_type_desc.replace(" ", "_").replace("/", "-")}.joblib')
    joblib.dump(best_xgb_model, model_filename_xgb)
    print(f"Saved best XGBoost model for {feature_type_desc} to {model_filename_xgb}")

    # Evaluation
    y_pred_labels_xgb = best_xgb_model.predict(X_test_processed)
    accuracy_val_xgb = accuracy_score(y_test_labels, y_pred_labels_xgb)
    class_report_str_xgb = classification_report(y_test_labels, y_pred_labels_xgb, target_names=target_class_names, zero_division=0)
    conf_matrix_xgb = confusion_matrix(y_test_labels, y_pred_labels_xgb, labels=np.arange(len(target_class_names)))

    print(f"Accuracy (XGBoost - {feature_type_desc}): {accuracy_val_xgb:.4f}")
    print(f"Classification Report (XGBoost - {feature_type_desc}):\n{class_report_str_xgb}")
    print(f"Confusion Matrix (XGBoost - {feature_type_desc}):\n{conf_matrix_xgb}")
    plot_confusion_matrix(conf_matrix_xgb, classes=target_class_names,
                          plot_title=f'CM for XGBoost - {feature_type_desc} (Acc: {accuracy_val_xgb:.3f})',
                          results_path=output_results_dir,
                          filename=f'cm_xgb_{feature_type_desc.replace(" ", "_").replace("/", "-")}.png')

    # Save detailed results
    results_text_file_xgb = os.path.join(output_results_dir, f'results_xgb_{feature_type_desc.replace(" ", "_").replace("/", "-")}.txt')
    with open(results_text_file_xgb, 'w') as f:
        f.write(f"--- XGBoost Results for {feature_type_desc} ---\n")
        f.write(f"Scaling Applied: {perform_scaling}\n")
        f.write(f"Best XGBoost Parameters: {xgb_grid_search.best_params_}\n")
        f.write(f"GridSearchCV Best CV Score (accuracy): {xgb_grid_search.best_score_:.4f}\n")
        f.write(f"Test Set Accuracy: {accuracy_val_xgb:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report_str_xgb + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(conf_matrix_xgb))
    print(f"Saved XGBoost results for {feature_type_desc} to {results_text_file_xgb}")

    return best_xgb_model

# --- 3. Feature Loading Functions ---
def load_spm_features(spm_bovw_dir, feature_name, pyramid_levels_count, set_type="train"):
    max_level_index = pyramid_levels_count - 1
    filename = f"X_{set_type}_{feature_name}_spm_L{max_level_index}.npy"
    filepath = os.path.join(spm_bovw_dir, filename)
    if os.path.exists(filepath):
        print(f"Loading {set_type} {feature_name} SPM (L{max_level_index}) features from: {filepath}")
        data = np.load(filepath)
        print(f"  Shape: {data.shape}")
        return data
    else:
        print(f"Warning: {feature_name} SPM (L{max_level_index}) file not found: {filepath}")
        return None

def load_and_align_global_hog(hog_h5_filepath, target_indices_for_set):
    # ... (Your existing robust load_and_align_global_hog function - ensure it's included) ...
    if not os.path.exists(hog_h5_filepath):
        print(f"Warning: Global HOG data file not found: {hog_h5_filepath}")
        return None
    print(f"Loading global HOG features from: {hog_h5_filepath}")
    try:
        with h5py.File(hog_h5_filepath, 'r') as hf:
            all_hog_features = hf['hog_features'][:]
            all_hog_original_indices = hf['indices'][:]
    except Exception as e:
        print(f"Error loading HOG data from {hog_h5_filepath}: {e}")
        return None
    if all_hog_features.ndim == 1:
        if len(target_indices_for_set) == 1 and all_hog_features.shape[0] > 1:
             all_hog_features = all_hog_features.reshape(1, -1)
        elif all_hog_features.shape[0] == 0:
            return np.empty((len(target_indices_for_set), 0), dtype=all_hog_features.dtype) if target_indices_for_set else np.empty((0,0))
        elif all_hog_features.shape[0] > 0 and len(target_indices_for_set) > 0:
            expected_feature_dim = all_hog_features.shape[0] // len(target_indices_for_set)
            if all_hog_features.shape[0] % len(target_indices_for_set) == 0 and expected_feature_dim > 0:
                all_hog_features = all_hog_features.reshape(len(target_indices_for_set), expected_feature_dim)
            else: return None
        else: return None
    if all_hog_features.shape[0] == 0 or all_hog_original_indices.shape[0] == 0: return None
    hog_feature_map = {original_idx: i for i, original_idx in enumerate(all_hog_original_indices)}
    aligned_hog_list = []
    missing_count = 0
    hog_feature_dim = all_hog_features.shape[1] if all_hog_features.ndim == 2 and all_hog_features.shape[1] > 0 else 0
    if hog_feature_dim == 0 and len(all_hog_original_indices) > 0: return None
    for target_idx in target_indices_for_set:
        if target_idx in hog_feature_map:
            aligned_hog_list.append(all_hog_features[hog_feature_map[target_idx]])
        else:
            if hog_feature_dim > 0: aligned_hog_list.append(np.zeros(hog_feature_dim, dtype=all_hog_features.dtype))
            else: aligned_hog_list.append(np.array([], dtype=all_hog_features.dtype))
            missing_count += 1
    if missing_count > 0: print(f"  Warning: {missing_count}/{len(target_indices_for_set)} HOG features for current set not found. Used zero/empty vectors.")
    if all(v.size == 0 for v in aligned_hog_list) and aligned_hog_list: return np.empty((len(aligned_hog_list), 0), dtype=all_hog_features.dtype)
    try:
        aligned_hog_array = np.array(aligned_hog_list) if aligned_hog_list else np.empty((0, hog_feature_dim if hog_feature_dim > 0 else 0))
    except ValueError as e: return None
    print(f"  Aligned global HOG shape: {aligned_hog_array.shape if hasattr(aligned_hog_array, 'shape') else 'N/A'}")
    return aligned_hog_array

# --- Main Execution ---
def run_spm_classification_pipeline():
    print("\n--- Starting SPM Classification Pipeline ---")

    # Load SPM BoVW Features
    print("\n--- Loading SPM BoVW Feature Sets ---")
    X_train_sift_spm = load_spm_features(BOVW_SPM_FEATURES_DIR, "sift", PYRAMID_LEVELS, "train")
    X_test_sift_spm = load_spm_features(BOVW_SPM_FEATURES_DIR, "sift", PYRAMID_LEVELS, "test")
    X_train_orb_spm = load_spm_features(BOVW_SPM_FEATURES_DIR, "orb", PYRAMID_LEVELS, "train")
    X_test_orb_spm = load_spm_features(BOVW_SPM_FEATURES_DIR, "orb", PYRAMID_LEVELS, "test")

    # Load Global HOG Features (reused)
    print("\n--- Loading Global HOG Feature Set (for SPM combinations) ---")
    X_train_hog_global = load_and_align_global_hog(HOG_DATA_FILE, train_indices)
    X_test_hog_global = load_and_align_global_hog(HOG_DATA_FILE, test_indices)

    # Train and Evaluate XGBoost Classifiers for SPM
    print("\n\n" + "="*20 + " XGBoost for SPM BoVW & HOG " + "="*20)

    # Individual SPM Features
    if X_train_sift_spm is not None and X_test_sift_spm is not None:
        train_and_evaluate_xgb(X_train_sift_spm, y_train, X_test_sift_spm, y_test,
                               f"SPM_SIFT_L{PYRAMID_LEVELS-1}", class_names, RESULTS_DIR_XGB_SPM)
    if X_train_orb_spm is not None and X_test_orb_spm is not None:
        train_and_evaluate_xgb(X_train_orb_spm, y_train, X_test_orb_spm, y_test,
                               f"SPM_ORB_L{PYRAMID_LEVELS-1}", class_names, RESULTS_DIR_XGB_SPM)
    
    # Re-evaluate HOG alone if you want its results in the SPM results folder
    # Otherwise, you can refer to its performance from the vanilla run.
    # For completeness, let's include it if you want to run it again here:
    # if X_train_hog_global is not None and X_test_hog_global is not None:
    #     train_and_evaluate_xgb(X_train_hog_global, y_train, X_test_hog_global, y_test,
    #                            "Global_HOG_for_SPM_run", class_names, RESULTS_DIR_XGB_SPM)


    # Combined SPM Features
    # SIFT SPM + HOG
    if X_train_sift_spm is not None and X_train_hog_global is not None:
        if X_train_sift_spm.shape[0] == X_train_hog_global.shape[0]:
            X_train_sift_spm_hog = np.concatenate((X_train_sift_spm, X_train_hog_global), axis=1)
            X_test_sift_spm_hog = np.concatenate((X_test_sift_spm, X_test_hog_global), axis=1)
            train_and_evaluate_xgb(X_train_sift_spm_hog, y_train, X_test_sift_spm_hog, y_test,
                                   f"SPM_SIFT_L{PYRAMID_LEVELS-1}_HOG", class_names, RESULTS_DIR_XGB_SPM)
        else:
            print("Skipping SIFT SPM + HOG: Mismatched sample counts.")

    # ORB SPM + HOG
    if X_train_orb_spm is not None and X_train_hog_global is not None:
        if X_train_orb_spm.shape[0] == X_train_hog_global.shape[0]:
            X_train_orb_spm_hog = np.concatenate((X_train_orb_spm, X_train_hog_global), axis=1)
            X_test_orb_spm_hog = np.concatenate((X_test_orb_spm, X_test_hog_global), axis=1)
            train_and_evaluate_xgb(X_train_orb_spm_hog, y_train, X_test_orb_spm_hog, y_test,
                                   f"SPM_ORB_L{PYRAMID_LEVELS-1}_HOG", class_names, RESULTS_DIR_XGB_SPM)
        else:
            print("Skipping ORB SPM + HOG: Mismatched sample counts.")

    # SIFT SPM + ORB SPM + HOG
    if X_train_sift_spm is not None and X_train_orb_spm is not None and X_train_hog_global is not None:
        if X_train_sift_spm.shape[0] == X_train_orb_spm.shape[0] == X_train_hog_global.shape[0]:
            X_train_all_spm = np.concatenate((X_train_sift_spm, X_train_orb_spm, X_train_hog_global), axis=1)
            X_test_all_spm = np.concatenate((X_test_sift_spm, X_test_orb_spm, X_test_hog_global), axis=1)
            train_and_evaluate_xgb(X_train_all_spm, y_train, X_test_all_spm, y_test,
                                   f"SPM_SIFT_ORB_L{PYRAMID_LEVELS-1}_HOG", class_names, RESULTS_DIR_XGB_SPM)
        else:
            print("Skipping SIFT SPM + ORB SPM + HOG: Mismatched sample counts.")

    print("\n--- SPM Classification Pipeline Complete ---")