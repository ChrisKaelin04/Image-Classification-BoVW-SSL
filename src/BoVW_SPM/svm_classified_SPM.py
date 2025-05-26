# SPM_SIFT_ORB_XGBoost_Classification_balanced.py
import numpy as np
import os
import pickle
import warnings
import joblib
import glob
# h5py is not needed as we only load .npy files now
# import h5py 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight # Correct import
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import traceback
import gc

# --- Configuration for Classification using BALANCED SPM Histograms (SIFT and ORB) ---

# Directory where histogram_creation_SPM_balanced.py saved the final .npy feature/label files
SPM_HISTOGRAMS_DIR = r"E:\CV\features_SPM_balanced\spm_histograms_L1_k1000" # Adjust L and K if different
# Example path: E:\CV_features_SPM_balanced\spm_histograms_L1_k1000 (assuming PYRAMID_LEVELS=2, VOCAB_SIZE=1000)

# Directory containing the label encoder file from the BALANCED splitting script
# This is used to get class names.
BALANCED_SPLITS_INFO_DIR = r"E:\CV\bovw_splits_balanced" # Where NPZ and PKL from balanced split are

# You need to know which N (total images) and S (seed) was used for the balanced split
# to pick the correct label encoder. For simplicity, let's assume a fixed name or find one.
# Example: bovw_label_encoder_N8000_S42.pkl
# If you have multiple, you'll need to specify which one corresponds to the features you're using.
LABEL_ENCODER_FILE_PATTERN = os.path.join(BALANCED_SPLITS_INFO_DIR, "bovw_label_encoder_N*_S*.pkl")

# Results directory for SPM-only results
RESULTS_DIR_XGB_SPM_BALANCED = r"E:\CV\features_SPM_balanced\classification_results_XGB_SPM_SIFT_ORB_balanced"
os.makedirs(RESULTS_DIR_XGB_SPM_BALANCED, exist_ok=True)

# DMatrix cache (can be shared or specific) for SPM features
DMATRIX_CACHE_DIR_BALANCED = os.path.join(r"E:\CV\features_SPM_balanced", "xgb_dmatrix_cache_spm_balanced")
os.makedirs(DMATRIX_CACHE_DIR_BALANCED, exist_ok=True)

# These must match what was used in histogram_creation_SPM_balanced.py to find the .npy files
VOCAB_SIZE_FOR_LOADING = 1000
PYRAMID_LEVELS_FOR_LOADING = 2 # L value used in filenames (e.g., L1 if PYRAMID_LEVELS=2)
MAX_LEVEL_INDEX_FOR_LOADING = PYRAMID_LEVELS_FOR_LOADING - 1


# --- Hyperparameters (can be kept similar) ---
XGB_BASE_PARAMS = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'tree_method': 'hist', # Use 'hist' for CPU/GPU compatibility
    'random_state': 42,
    'use_label_encoder': False # Deprecated, always False
}

PARAM_GRID_XGB = { # Smaller grid for faster runs, expand if needed
    'n_estimators': [200, 300],       # Number of trees
    'learning_rate': [0.05, 0.1],   # Step size shrinkage
    'max_depth': [5, 7],            # Max depth of a tree
    # 'colsample_bytree': [0.8],    # Subsample ratio of columns when constructing each tree
    # 'subsample': [0.8],           # Subsample ratio of the training instances
}
GRIDSEARCH_CV_FOLDS = 3
SAMPLE_FRACTION_FOR_GRIDSEARCH = 1.0 # 1.0 means use full training data for GridSearch
GRIDSEARCH_SCORING = 'f1_macro' # Good for potentially imbalanced classes

warnings.filterwarnings("ignore", message="Parameters: {.*use_label_encoder.*} are not used.", category=UserWarning)
warnings.filterwarnings("ignore", message="omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.", category=UserWarning)


# --- Helper Functions (plot_confusion_matrix remains the same) ---
def plot_confusion_matrix(cm, classes, plot_title='Confusion matrix', cmap=plt.cm.Blues, results_path=None, filename=None):
    plt.figure(figsize=(max(8, len(classes)), max(6, len(classes)*0.8)))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(plot_title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if results_path and filename:
        os.makedirs(results_path, exist_ok=True) # Ensure dir exists
        full_path = os.path.join(results_path, filename)
        try:
            plt.savefig(full_path)
            print(f"Saved confusion matrix to {full_path}")
        except Exception as e:
            print(f"Error saving confusion matrix to {full_path}: {e}")
    plt.close()


# --- Feature Loading Function (only for SPM) ---

def load_balanced_spm_histograms_and_labels(hist_dir, feature_name_in_file, set_type, L_val, K_val):
    """
    Loads pre-computed SPM histograms (X) and their corresponding labels (y) from .npy files.
    Assumes these are the *final*, concatenated SPM histograms per image.
    Args:
        hist_dir (str): Directory containing the .npy files.
        feature_name_in_file (str): 'sift' or 'orb'.
        set_type (str): 'train' or 'test'.
        L_val (int): Max level index used in filename (e.g., 1 for PYRAMID_LEVELS=2).
        K_val (int): Vocabulary size used in filename.
    Returns:
        (np.array, np.array): (X_features, y_labels_numeric) or (None, None)
    """
    hist_filename = f"X_{set_type}_{feature_name_in_file}_spm_L{L_val}_k{K_val}.npy"
    labels_filename = f"y_{set_type}_{feature_name_in_file}_spm_L{L_val}_k{K_val}_labels.npy" # Labels might be common across features, but loaded here for safety

    hist_filepath = os.path.join(hist_dir, hist_filename)
    labels_filepath = os.path.join(hist_dir, labels_filename)

    X_features, y_labels = None, None

    if os.path.exists(hist_filepath):
        try:
            print(f"Loading {set_type} {feature_name_in_file} SPM features from: {hist_filepath}")
            X_features = np.load(hist_filepath)
            print(f"Loaded {set_type} {feature_name_in_file} SPM features. Shape: {X_features.shape}")
        except Exception as e:
            print(f"ERROR loading feature file {hist_filepath}: {e}")
            return None, None
    else:
        print(f"ERROR: Feature file not found: {hist_filepath}")
        return None, None

    # Labels might be stored once per set type in the SPM dir, adjust if needed
    # Here we assume labels are saved alongside features for each feature type
    if os.path.exists(labels_filepath):
        try:
            print(f"Loading {set_type} {feature_name_in_file} SPM labels from: {labels_filepath}")
            y_labels = np.load(labels_filepath)
            print(f"Loaded {set_type} {feature_name_in_file} SPM labels. Shape: {y_labels.shape}")
        except Exception as e:
            print(f"ERROR loading label file {labels_filepath}: {e}")
            return X_features, None # Return features if found, but indicate labels are missing
    else:
        print(f"ERROR: Label file not found: {labels_filepath}")
        # If labels are missing specifically for this feature type, maybe they are general?
        # For this script, let's strictly require the labels to be associated with the feature file pattern.
        # If labels are indeed stored generally (e.g., y_train_spm_labels.npy), this part needs adjustment.
        return X_features, None

    if X_features is not None and y_labels is not None and X_features.shape[0] != y_labels.shape[0]:
        print(f"ERROR: Mismatch between feature count ({X_features.shape[0]}) and label count ({y_labels.shape[0]}) for {set_type} {feature_name_in_file}.")
        return None, None

    return X_features, y_labels


# --- DMatrix Creation (Simplified, as data is already train/test split) ---
# This function remains the same, it works on the concatenated NumPy arrays provided to it.
def create_dmatrix_from_features(X_data, y_data, set_type, feature_desc, output_dir,
                                 perform_scaling=False, scaler_to_use_or_fit=None, sample_weights=None):
    """
    Creates and saves an XGBoost DMatrix from provided feature data and labels.
    Handles scaling and sample weights.
    Args:
        X_data (np.array): Feature matrix.
        y_data (np.array): Label vector.
        set_type (str): 'train' or 'test'.
        feature_desc (str): Description of features for filename.
        output_dir (str): Directory to save DMatrix buffer and scaler.
        perform_scaling (bool): Whether to apply StandardScaler.
        scaler_to_use_or_fit (StandardScaler or None): Pre-fitted scaler for 'test', or None for 'train' to fit a new one.
        sample_weights (np.array or None): Weights for each sample (for training DMatrix).
    Returns:
        (str, StandardScaler or None): Path to saved DMatrix file, and the fitted/used scaler (if scaling).
                                       Returns (None, None) if creation fails.
    """
    print(f"\n--- Creating DMatrix for: {feature_desc} ({set_type}) ---")
    
    filename_base = f"{set_type}_{feature_desc}"
    if set_type == 'train' and sample_weights is not None: filename_base += "_weighted"
    if perform_scaling: filename_base += "_scaled"
    
    dmatrix_filename = os.path.join(output_dir, f"{filename_base}.buffer")
    scaler_filename = os.path.join(output_dir, f"scaler_train_{feature_desc}.joblib") # Scaler always named based on train

    # Skip if DMatrix already exists (and scaler if training and scaling)
    dmatrix_exists = os.path.exists(dmatrix_filename)
    # Need to check if scaler exists ONLY if we are training AND scaling is requested
    scaler_needed_and_missing = (set_type == 'train' and perform_scaling and not os.path.exists(scaler_filename))
    
    loaded_scaler = None
    if perform_scaling:
        if set_type == 'train' and os.path.exists(scaler_filename):
             try:
                 loaded_scaler = joblib.load(scaler_filename)
                 print(f"Loaded existing scaler from {scaler_filename}")
             except Exception as e:
                 print(f"Error loading existing scaler {scaler_filename}: {e}. Will refit/create.")
                 loaded_scaler = None # Force refit
        elif set_type == 'test' and scaler_to_use_or_fit is not None:
             loaded_scaler = scaler_to_use_or_fit # Use the provided scaler for test

    if dmatrix_exists and (set_type == 'test' or (set_type == 'train' and loaded_scaler is not None) or not perform_scaling):
        print(f"DMatrix file {dmatrix_filename} already exists (and scaler loaded/not needed). Skipping creation.")
        return dmatrix_filename, loaded_scaler # Return the loaded scaler if applicable


    X_processed = X_data.copy() # Work on a copy
    fitted_scaler = None

    if perform_scaling:
        if set_type == 'train':
            if loaded_scaler is None: # Only fit if it wasn't loaded (or failed to load)
                print("Fitting StandardScaler on training data...")
                scaler = StandardScaler()
                X_processed = scaler.fit_transform(X_processed)
                try:
                    joblib.dump(scaler, scaler_filename)
                    print(f"Saved fitted scaler to {scaler_filename}")
                except Exception as e:
                     print(f"Warning: Could not save scaler to {scaler_filename}: {e}")
                fitted_scaler = scaler
            else: # Use the loaded scaler if it was found for train
                print("Using existing scaler for training data...")
                X_processed = loaded_scaler.transform(X_processed)
                fitted_scaler = loaded_scaler # Return the loaded scaler
        elif set_type == 'test' and scaler_to_use_or_fit is not None:
            print("Transforming test data using provided scaler...")
            X_processed = scaler_to_use_or_fit.transform(X_processed)
            fitted_scaler = scaler_to_use_or_fit # Return the scaler that was used
        elif set_type == 'test':
            print("WARNING: Scaling requested for test, but no scaler provided. Using unscaled data.")
            # X_processed remains unscaled, fitted_scaler remains None


    print("Creating XGBoost DMatrix...")
    try:
        # Ensure C-contiguous array for DMatrix
        if not X_processed.flags['C_CONTIGUOUS']:
             print("Converting data to C-contiguous array...")
             X_processed = np.ascontiguousarray(X_processed)
        
        dmatrix = xgb.DMatrix(X_processed, label=y_data.astype(np.float32), 
                              weight=sample_weights if set_type == 'train' else None)
        dmatrix.save_binary(dmatrix_filename)
        print(f"Successfully saved DMatrix to {dmatrix_filename}")
        
        # Clean up large arrays immediately
        del X_processed, dmatrix
        gc.collect()
        
        return dmatrix_filename, fitted_scaler
    except Exception as e:
        print(f"ERROR: Failed to create or save DMatrix for {feature_desc} ({set_type}): {e}")
        traceback.print_exc()
        return None, None


# --- Training and Evaluation Functions (reused) ---
def find_best_params_with_gridsearch(
    X_train_for_gs, y_train_for_gs, num_classes,
    base_params, param_grid, cv_folds, feature_type_desc, scoring_metric, perform_scaling=True):

    print(f"\n--- Performing GridSearchCV for {feature_type_desc} ---")
    print(f"Data size for GridSearchCV: {X_train_for_gs.shape[0]} instances")
    
    X_scaled_for_gs = X_train_for_gs # Default to unscaled
    scaler_gs = None
    if perform_scaling:
        print("Scaling data for GridSearchCV...")
        scaler_gs = StandardScaler()
        # Fit and transform on the GS data (which is a sample or full train)
        X_scaled_for_gs = scaler_gs.fit_transform(X_train_for_gs)
        # No need to save this scaler, it's only for the GS step


    sample_weights_gs = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_train_for_gs)
    sample_weights_gs_per_sample = np.array([sample_weights_gs[label] for label in y_train_for_gs])
    
    best_params_found = None
    best_score = -1.0
    
    # Try GPU first
    try:
        print("Attempting GridSearchCV with GPU...")
        # Ensure num_class is in params for estimator init
        gpu_params = {**base_params, 'device': 'cuda', 'num_class': num_classes}
        estimator_gpu = xgb.XGBClassifier(**gpu_params)
        # n_jobs=1 for GPU training within GridSearchCV is typical
        grid_search_gpu = GridSearchCV(estimator_gpu, param_grid, scoring=scoring_metric, cv=cv_folds, verbose=1, n_jobs=1)
        grid_search_gpu.fit(X_scaled_for_gs, y_train_for_gs, sample_weight=sample_weights_gs_per_sample)
        
        best_params_found = grid_search_gpu.best_params_
        best_score = grid_search_gpu.best_score_
        print(f"GridSearchCV with GPU successful. Best {scoring_metric}: {best_score:.4f}")
        
        # Add device to the returned params if GPU worked
        final_params = {**base_params, **best_params_found, '_gridsearch_scoring': scoring_metric, 'num_class': num_classes, 'device': 'cuda'}
        
        del grid_search_gpu, estimator_gpu
        gc.collect()
        return final_params

    except Exception as gpu_err:
        print(f"GridSearchCV with GPU failed: {gpu_err}. Falling back to CPU.")
        traceback.print_exc() # Print GPU error traceback

        try:
            print("Attempting GridSearchCV with CPU...")
            cpu_params = {**base_params, 'num_class': num_classes}
            if 'device' in cpu_params: del cpu_params['device'] # Ensure no device param
            estimator_cpu = xgb.XGBClassifier(**cpu_params)
            # n_jobs=-1 uses all available CPU cores
            grid_search_cpu = GridSearchCV(estimator_cpu, param_grid, scoring=scoring_metric, cv=cv_folds, verbose=1, n_jobs=-1)
            grid_search_cpu.fit(X_scaled_for_gs, y_train_for_gs, sample_weight=sample_weights_gs_per_sample)
            
            best_params_found = grid_search_cpu.best_params_
            best_score = grid_search_cpu.best_score_
            print(f"GridSearchCV with CPU successful. Best {scoring_metric}: {best_score:.4f}")
            
            final_params = {**base_params, **best_params_found, '_gridsearch_scoring': scoring_metric, 'num_class': num_classes}
            # Ensure no device param is included if CPU was used
            if 'device' in final_params: del final_params['device'] 
            
            del grid_search_cpu, estimator_cpu
            gc.collect()
            return final_params

        except Exception as cpu_err:
            print(f"ERROR: GridSearchCV with CPU also failed: {cpu_err}")
            traceback.print_exc() # Print CPU error traceback
            return None
            
    finally:
        # Clean up GS data regardless of success
        del X_train_for_gs, y_train_for_gs, X_scaled_for_gs, sample_weights_gs_per_sample, sample_weights_gs
        if 'scaler_gs' in locals() and scaler_gs is not None: del scaler_gs
        gc.collect()


def train_and_evaluate_xgb_dmatrix(dtrain_path, dtest_path, y_test_actual,
                                  best_params_from_search,
                                  feature_type_desc, target_class_names_list,
                                  output_results_dir_path, num_actual_classes):
    print(f"\n--- Training FINAL XGBoost on FULL data for {feature_type_desc} using DMatrix ---")
    if not (os.path.exists(dtrain_path) and os.path.exists(dtest_path)):
        print("ERROR: DMatrix train or test file not found. Skipping."); return False # Indicate failure

    try:
        dtrain = xgb.DMatrix(dtrain_path)
        dtest = xgb.DMatrix(dtest_path)
        # Ensure test label is correctly set; it might not be saved in the buffer
        dtest.set_label(y_test_actual.astype(np.float32))
        print(f"Loaded DMatrices. Train shape: {dtrain.num_row()}x{dtrain.num_col()}, Test shape: {dtest.num_row()}x{dtest.num_col()}")
    except Exception as e:
        print(f"ERROR loading DMatrix files: {e}"); traceback.print_exc(); return False

    final_params = {**best_params_from_search} # Copy params

    # Remove internal GS scoring param
    if '_gridsearch_scoring' in final_params: del final_params['_gridsearch_scoring']
    
    # Determine final device for training
    actual_device_for_train = 'cpu'
    # Check if 'device' was set to 'cuda' by GridSearchCV
    if final_params.get('device') == 'cuda':
        try:
            # Quick test to see if GPU training works
            temp_d = xgb.DMatrix(np.random.rand(2, dtrain.num_col()))
            # Need a dummy label for the temporary DMatrix to avoid warnings/errors with objective
            dummy_label = np.random.randint(0, num_actual_classes, 2) 
            # Use a temporary dummy eval set as required by device='cuda' in recent versions
            xgb.train(final_params, temp_d, num_boost_round=1, evals=[(temp_d, 'temp_eval')])
            actual_device_for_train = 'cuda'
            final_params['device'] = 'cuda' # Explicitly set for the actual train call
            print("GPU available and selected for final training.")
        except Exception:
            print("GPU check or setting failed for final training. Using CPU.")
            if 'device' in final_params: del final_params['device'] # Remove device param
    else:
        # If 'device' was not 'cuda' or not present in GS results, ensure it's not passed
        if 'device' in final_params: del final_params['device']


    # Pop n_estimators as it's passed to num_boost_round
    num_boost_round_val = final_params.pop('n_estimators', 300)
    
    print(f"Starting final XGBoost training ({actual_device_for_train}) with params: {final_params}, num_boost_round={num_boost_round_val}")
    
    try:
        evals_list = [(dtrain, 'train'), (dtest, 'eval')]
        
        bst_model = xgb.train(
            final_params, dtrain, num_boost_round=num_boost_round_val, evals=evals_list,
            early_stopping_rounds=50, verbose_eval=100
        )

        model_filename = os.path.join(output_results_dir_path, f'xgb_model_{feature_type_desc}.json')
        bst_model.save_model(model_filename)
        print(f"Saved final XGBoost model for {feature_type_desc} to {model_filename}")

        # Predict on test set using the best iteration
        # Check if early stopping occurred and use best_iteration+1
        predict_iteration_range = (0, bst_model.best_iteration + 1 if bst_model.best_iteration is not None else num_boost_round_val)
        print(f"Predicting on test set using iterations {predict_iteration_range}...")
        y_pred_proba_test = bst_model.predict(dtest, iteration_range=predict_iteration_range)
        y_pred_labels_test = np.argmax(y_pred_proba_test, axis=1)

        accuracy_val_test = accuracy_score(y_test_actual, y_pred_labels_test)
        f1_macro_val_test = f1_score(y_test_actual, y_pred_labels_test, average='macro', zero_division=0)
        # Ensure correct labels and target_names are used for classification report
        class_report_str_test = classification_report(y_test_actual, y_pred_labels_test, 
                                                      target_names=target_class_names_list, 
                                                      labels=np.arange(num_actual_classes), # Explicitly list all possible labels
                                                      zero_division=0)
        # Ensure confusion matrix covers all classes
        conf_matrix_test = confusion_matrix(y_test_actual, y_pred_labels_test, 
                                            labels=np.arange(num_actual_classes))

        print(f"Final Model Trained on: {actual_device_for_train}")
        print(f"Test Set Accuracy: {accuracy_val_test:.4f}")
        print(f"Test Set F1-macro: {f1_macro_val_test:.4f}")
        print(f"Classification Report:\n{class_report_str_test}")

        cm_title = f'CM XGB {feature_type_desc} (Acc: {accuracy_val_test:.3f}, F1-M: {f1_macro_val_test:.3f})'
        plot_confusion_matrix(conf_matrix_test, classes=target_class_names_list, plot_title=cm_title, results_path=output_results_dir_path, filename=f'cm_xgb_{feature_type_desc}.png')

        results_text_file = os.path.join(output_results_dir_path, f'results_xgb_{feature_type_desc}.txt')
        with open(results_text_file, 'w') as f:
            f.write(f"--- XGBoost Results for {feature_type_desc} ---\n")
            f.write(f"GridSearchCV Best Params (from sample): {best_params_from_search}\n")
            f.write(f"Final Training Device: {actual_device_for_train}\n")
            f.write(f"Test Accuracy: {accuracy_val_test:.4f}\nTest F1-macro: {f1_macro_val_test:.4f}\n\nReport:\n{class_report_str_test}\n\nCM:\n{np.array2string(conf_matrix_test)}")
        print(f"Saved results to {results_text_file}")
        
        del dtrain, dtest, bst_model, y_pred_proba_test, y_pred_labels_test, conf_matrix_test
        gc.collect()
        return True # Indicate success
    except Exception as e:
        print(f"ERROR during final training or evaluation for {feature_type_desc}: {e}")
        traceback.print_exc()
        del dtrain, dtest # Ensure DMatrices are deleted on error too
        gc.collect()
        return False # Indicate failure


# --- Main Execution Pipeline ---
def run_balanced_spm_classification_pipeline():
    print("\n--- Starting BALANCED SPM (SIFT/ORB) Classification Pipeline ---")

    # --- 1. Load Label Encoder ---
    label_encoder_files = glob.glob(LABEL_ENCODER_FILE_PATTERN)
    if not label_encoder_files:
        print(f"ERROR: No label encoder file found matching pattern: {LABEL_ENCODER_FILE_PATTERN}")
        return
    # Assuming one relevant encoder, or implement logic to choose if multiple
    label_encoder_path = label_encoder_files[0]
    print(f"Loading label encoder from: {label_encoder_path}")
    try:
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        class_names_global = label_encoder.classes_
        num_classes_global = len(class_names_global)
        XGB_BASE_PARAMS['num_class'] = num_classes_global # Set global param
        print(f"Class names: {class_names_global} ({num_classes_global} classes)")
    except Exception as e:
        print(f"ERROR loading label encoder: {e}"); return

    # --- Define Feature Combinations ---
    # L_val should be MAX_LEVEL_INDEX_FOR_LOADING
    sift_spm_name = f"SPM_SIFT_L{MAX_LEVEL_INDEX_FOR_LOADING}"
    orb_spm_name = f"SPM_ORB_L{MAX_LEVEL_INDEX_FOR_LOADING}"
    spm_sift_orb_name = f"SPM_SIFT_ORB_L{MAX_LEVEL_INDEX_FOR_LOADING}"

    feature_sets_to_run = {
        sift_spm_name: {"sift_spm": True, "orb_spm": False},
        orb_spm_name: {"sift_spm": False, "orb_spm": True},
        spm_sift_orb_name: {"sift_spm": True, "orb_spm": True},
    }

    all_best_params_from_gs = {} # Store best params for each feature set

    for feature_desc_key, include_features_dict in feature_sets_to_run.items():
        print(f"\n\n{'='*20} Processing Feature Set: {feature_desc_key} {'='*20}")

        X_train_list, X_test_list = [], []
        y_train_final, y_test_final = None, None # Will hold the labels for the current feature set
        
        # --- Load Training Data ---
        print(f"Loading TRAINING data for {feature_desc_key}...")
        
        load_success_train = True

        if include_features_dict.get("sift_spm"):
            X_sift_tr, y_sift_tr = load_balanced_spm_histograms_and_labels(SPM_HISTOGRAMS_DIR, "sift", "train", MAX_LEVEL_INDEX_FOR_LOADING, VOCAB_SIZE_FOR_LOADING)
            if X_sift_tr is not None: X_train_list.append(X_sift_tr)
            if y_train_final is None and y_sift_tr is not None: y_train_final = y_sift_tr
            # If y_train_final is already set (e.g., by ORB), check consistency
            elif y_sift_tr is not None and y_train_final is not None and not np.array_equal(y_train_final, y_sift_tr): print("WARNING: SIFT train label mismatch! This feature set will be skipped."); load_success_train = False
            if X_sift_tr is None: load_success_train = False # Mark failure if feature loading failed

        if load_success_train and include_features_dict.get("orb_spm"):
            X_orb_tr, y_orb_tr = load_balanced_spm_histograms_and_labels(SPM_HISTOGRAMS_DIR, "orb", "train", MAX_LEVEL_INDEX_FOR_LOADING, VOCAB_SIZE_FOR_LOADING)
            if X_orb_tr is not None: X_train_list.append(X_orb_tr)
            if y_train_final is None and y_orb_tr is not None: y_train_final = y_orb_tr
             # Check labels consistency only if y_train_final was already set by a previous feature
            elif y_orb_tr is not None and y_train_final is not None and not np.array_equal(y_train_final, y_orb_tr): 
                 print("WARNING: ORB train label mismatch! This feature set will be skipped."); load_success_train = False
            if X_orb_tr is None: load_success_train = False # Mark failure if ORB loading failed


        if not load_success_train or not X_train_list or y_train_final is None:
            print(f"Failed to load sufficient training data or labels for {feature_desc_key}. Skipping.");
            # Clean up any partially loaded data
            del X_train_list, y_train_final
            gc.collect()
            continue
        
        # Concatenate training features
        X_train_combined = np.concatenate(X_train_list, axis=1) if len(X_train_list) > 1 else X_train_list[0]
        del X_train_list; gc.collect()
        print(f"Combined training features shape for {feature_desc_key}: {X_train_combined.shape}")

        # --- Load Test Data ---
        print(f"Loading TEST data for {feature_desc_key}...")
        
        load_success_test = True
        # Reset y_test_final as it might have been set by a previous feature set run
        y_test_final = None 

        if include_features_dict.get("sift_spm"):
            X_sift_te, y_sift_te = load_balanced_spm_histograms_and_labels(SPM_HISTOGRAMS_DIR, "sift", "test", MAX_LEVEL_INDEX_FOR_LOADING, VOCAB_SIZE_FOR_LOADING)
            if X_sift_te is not None: X_test_list.append(X_sift_te)
            if y_test_final is None and y_sift_te is not None: y_test_final = y_sift_te
             # If y_test_final is already set (e.g., by ORB), check consistency
            elif y_sift_te is not None and y_test_final is not None and not np.array_equal(y_test_final, y_sift_te): print("WARNING: SIFT test label mismatch! This feature set will be skipped."); load_success_test = False
            if X_sift_te is None: load_success_test = False


        if load_success_test and include_features_dict.get("orb_spm"):
            X_orb_te, y_orb_te = load_balanced_spm_histograms_and_labels(SPM_HISTOGRAMS_DIR, "orb", "test", MAX_LEVEL_INDEX_FOR_LOADING, VOCAB_SIZE_FOR_LOADING)
            if X_orb_te is not None: X_test_list.append(X_orb_te)
            if y_test_final is None and y_orb_te is not None: y_test_final = y_orb_te
            # Check labels consistency only if y_test_final was already set by a previous feature
            elif y_orb_te is not None and y_test_final is not None and not np.array_equal(y_test_final, y_orb_te): 
                 print("WARNING: ORB test label mismatch! This feature set will be skipped."); load_success_test = False
            if X_orb_te is None: load_success_test = False


        if not load_success_test or not X_test_list or y_test_final is None:
            print(f"Failed to load sufficient test data or labels for {feature_desc_key}. Skipping.");
            # Clean up any partially loaded data
            del X_test_list, X_train_combined, y_train_final # Also clean train data if test failed
            gc.collect()
            continue

        # Concatenate test features
        X_test_combined = np.concatenate(X_test_list, axis=1) if len(X_test_list) > 1 else X_test_list[0]
        del X_test_list; gc.collect()
        print(f"Combined test features shape for {feature_desc_key}: {X_test_combined.shape}")

        # --- GridSearchCV on Sample (or full train if fraction is 1.0) ---
        # Create a sample for GridSearchCV to save memory/time if fraction < 1.0
        X_train_gs_sample, y_train_gs_sample = X_train_combined, y_train_final # Default to full train
        
        if SAMPLE_FRACTION_FOR_GRIDSEARCH < 1.0 and len(y_train_final) >= num_classes_global * GRIDSEARCH_CV_FOLDS * 2: # Ensure enough samples for stratified split across folds
            print(f"Creating {SAMPLE_FRACTION_FOR_GRIDSEARCH*100:.1f}% sample for GridSearchCV...")
            try:
                # Use train_test_split to get a *subset* of the training data
                # The 'train' part of this split will be used for GS
                _, X_train_gs_sample, _, y_train_gs_sample = train_test_split(
                    X_train_combined, y_train_final, 
                    test_size=(1.0-SAMPLE_FRACTION_FOR_GRIDSEARCH), # This size will be discarded
                    random_state=42, 
                    stratify=y_train_final
                )
                print(f"Using {len(y_train_gs_sample)} instances for GridSearchCV.")
            except ValueError as e:
                print(f"Warning: Stratified split for GridSearchCV sample failed ({e}). Using full train set for GS.");
                X_train_gs_sample, y_train_gs_sample = X_train_combined, y_train_final
            except Exception as e:
                 print(f"Warning: Error creating GridSearchCV sample ({e}). Using full train set for GS.");
                 traceback.print_exc()
                 X_train_gs_sample, y_train_gs_sample = X_train_combined, y_train_final
        else:
             if SAMPLE_FRACTION_FOR_GRIDSEARCH < 1.0: # Only print this if user requested sampling but we couldn't do it
                  print(f"Warning: Not enough samples ({len(y_train_final)}) for stratified split with sample fraction {SAMPLE_FRACTION_FOR_GRIDSEARCH} and {num_classes_global} classes. Using full train set for GS.")
             print(f"Using full training set ({len(y_train_final)} instances) for GridSearchCV.")


        best_params = find_best_params_with_gridsearch(
            X_train_gs_sample, y_train_gs_sample, num_classes_global,
            XGB_BASE_PARAMS, PARAM_GRID_XGB, GRIDSEARCH_CV_FOLDS, feature_desc_key, GRIDSEARCH_SCORING,
            perform_scaling=True # Scaling done within find_best_params for GS data
        )
        
        # Clean up GS sample data immediately
        del X_train_gs_sample, y_train_gs_sample
        gc.collect()

        if best_params is None: 
            print(f"GridSearchCV failed for {feature_desc_key}. Skipping final train."); 
            # Clean up remaining train/test data for this feature set
            del X_train_combined, X_test_combined, y_train_final, y_test_final
            gc.collect()
            continue

        all_best_params_from_gs[feature_desc_key] = best_params

        # --- Create DMatrices (Scaled & Weighted for Train) ---
        # Compute weights for the *full* training set
        full_train_sample_weights = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes_global), y=y_train_final)
        full_train_sample_weights_per_sample = np.array([full_train_sample_weights[label] for label in y_train_final])
        
        print(f"Creating DMatrices for {feature_desc_key}...")

        # Train DMatrix
        dtrain_path, train_scaler = create_dmatrix_from_features(
            X_train_combined, y_train_final, "train", feature_desc_key, DMATRIX_CACHE_DIR_BALANCED,
            perform_scaling=True, sample_weights=full_train_sample_weights_per_sample # Pass weights only for train
        )
        
        # Clean up full training data after DMatrix creation
        del X_train_combined, y_train_final, full_train_sample_weights_per_sample, full_train_sample_weights
        gc.collect()

        # Test DMatrix
        # Note: y_test_final is only used for the *label* data in DMatrix, not features.
        # The train_scaler MUST be available for the test DMatrix creation if scaling is True.
        dtest_path, _ = create_dmatrix_from_features(
            X_test_combined, y_test_final, "test", feature_desc_key, DMATRIX_CACHE_DIR_BALANCED,
            perform_scaling=True, scaler_to_use_or_fit=train_scaler # Use scaler from training
        )
        
        # Clean up full test features after DMatrix creation
        del X_test_combined, train_scaler # train_scaler might still hold memory
        gc.collect()


        if not dtrain_path or not dtest_path: 
            print(f"DMatrix creation failed for {feature_desc_key}. Skipping final train."); 
             # y_test_final is still needed for evaluation below if DMatrix creation succeeded
            # If DMatrix creation failed, y_test_final should ideally be cleaned here.
            del y_test_final
            gc.collect()
            continue


        # --- Train Final Model on Full DMatrix ---
        print(f"Starting final training for {feature_desc_key}...")
        train_success = train_and_evaluate_xgb_dmatrix(
            dtrain_path, dtest_path, y_test_final, # y_test_actual is needed for evaluation metrics
            best_params, feature_desc_key, class_names_global,
            RESULTS_DIR_XGB_SPM_BALANCED, num_classes_global
        )
        
        # Clean up y_test_final after evaluation
        del y_test_final
        gc.collect()


        if train_success:
            print(f"Successfully processed {feature_desc_key}")
        else:
            print(f"Final training/evaluation failed for {feature_desc_key}")

    print("\n--- BALANCED SPM (SIFT/ORB) Classification Pipeline Complete ---")
    print("Best parameters found from GridSearchCV (on sample/full train):")
    for name, params in all_best_params_from_gs.items(): print(f"  {name}: {params}")
    print(f"Results saved to: {RESULTS_DIR_XGB_SPM_BALANCED}")