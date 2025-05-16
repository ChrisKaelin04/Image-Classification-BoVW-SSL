# Vanilla_BoVW_SOH_XGBoost_Classification_balanced.py
import numpy as np
import os
import pickle
import warnings
import joblib
import h5py # For loading HOG data if still used from an older HDF5 source
import glob # For finding label encoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split # train_test_split for GS sample
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight # Correct import
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import traceback
import gc

# --- Configuration for BALANCED Vanilla BoVW Classification ---

# Directory where histogram_creation_vanilla_balanced.py saved .npy histograms and labels
NORMAL_BOVW_HISTOGRAMS_DIR = r"E:\CV_BoVW_Vanilla_Balanced\raw_features\bovw_histograms_k1000" # Ensure K matches

# Directory where SOH_extract_vanilla_balanced.py saved HOG HDF5 files (if using these HOGs)
# Or, if HOG features are bundled with SPM's balanced HOG, point there.
HOG_FEATURES_BALANCED_DIR = r"E:\CV_BoVW_Vanilla_Balanced\raw_features" # Dir containing hog_data_train_balanced.h5 etc.

# Label encoder file from the BALANCED splitting script (create_balanced_split_for_bovw.py)
BALANCED_SPLITS_INFO_DIR = r"E:\CV_features\bovw_splits_balanced"
LABEL_ENCODER_FILE_PATTERN = os.path.join(BALANCED_SPLITS_INFO_DIR, "bovw_label_encoder_N*_S*.pkl")

# Results directory
RESULTS_DIR_XGB_VANILLA_BALANCED = r"E:\CV_BoVW_Vanilla_Balanced\classification_results_XGB_SOH_balanced"
os.makedirs(RESULTS_DIR_XGB_VANILLA_BALANCED, exist_ok=True)

# DMatrix cache (can be specific to this pipeline)
DMATRIX_CACHE_DIR_VANILLA_BALANCED = os.path.join(r"E:\CV_BoVW_Vanilla_Balanced", "xgb_dmatrix_cache_SOH_balanced")
os.makedirs(DMATRIX_CACHE_DIR_VANILLA_BALANCED, exist_ok=True)

# K value used in histogram filenames
VOCAB_SIZE_FOR_LOADING = 1000

# --- XGBoost Hyperparameters (Keep these moderate for your hardware) ---
XGB_BASE_PARAMS = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'tree_method': 'hist',
    'random_state': 42,
    'use_label_encoder': False
}
PARAM_GRID_XGB = { # Smaller grid for faster runs, expand if needed
    'n_estimators': [200, 300],       # Number of trees
    'learning_rate': [0.05, 0.1],   # Step size shrinkage
    'max_depth': [5, 7],            # Max depth of a tree
    # 'colsample_bytree': [0.8],    # Subsample ratio of columns when constructing each tree
    # 'subsample': [0.8],           # Subsample ratio of the training instances
}
GRIDSEARCH_CV_FOLDS = 3
SAMPLE_FRACTION_FOR_GRIDSEARCH = 1.0 # 1.0 for full train set, or <1.0 for a sample
GRIDSEARCH_SCORING = 'f1_macro'

warnings.filterwarnings("ignore", message="Parameters: {.*use_label_encoder.*} are not used.", category=UserWarning)
warnings.filterwarnings("ignore", message="omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.", category=UserWarning)


# --- Helper Functions (plot_confusion_matrix - can be reused) ---
def plot_confusion_matrix(cm, classes, plot_title='Confusion matrix', cmap=plt.cm.Blues, results_path=None, filename=None):
    # (Same as your previous script)
    plt.figure(figsize=(max(8, len(classes)), max(6, len(classes)*0.8)))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(plot_title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if results_path and filename:
        os.makedirs(results_path, exist_ok=True)
        full_path = os.path.join(results_path, filename)
        plt.savefig(full_path)
        print(f"Saved confusion matrix to {full_path}")
    plt.close()

# --- Feature Loading Functions for BALANCED Vanilla BoVW ---
def load_balanced_vanilla_bovw_histograms_and_labels(hist_dir, feature_name_in_file, set_type, K_val):
    """Loads X (histograms) and y (labels) from .npy files for balanced vanilla BoVW."""
    hist_filename = f"X_{set_type}_{feature_name_in_file}_vanilla_k{K_val}.npy"
    labels_filename = f"y_{set_type}_{feature_name_in_file}_vanilla_labels_k{K_val}.npy"

    hist_filepath = os.path.join(hist_dir, hist_filename)
    labels_filepath = os.path.join(hist_dir, labels_filename)

    X_features, y_labels = None, None
    print(f"Attempting to load: {hist_filepath} and {labels_filepath}")

    if os.path.exists(hist_filepath):
        try: X_features = np.load(hist_filepath)
        except Exception as e: print(f"ERROR loading {hist_filepath}: {e}"); return None, None
    else: print(f"ERROR: Feature file not found: {hist_filepath}"); return None, None

    if os.path.exists(labels_filepath):
        try: y_labels = np.load(labels_filepath)
        except Exception as e: print(f"ERROR loading {labels_filepath}: {e}"); return X_features, None
    else: print(f"ERROR: Label file not found: {labels_filepath}"); return X_features, None

    if X_features is not None and y_labels is not None:
        print(f"  Loaded {set_type} {feature_name_in_file} BoVW features shape: {X_features.shape}, Labels shape: {y_labels.shape}")
        if X_features.shape[0] != y_labels.shape[0]:
            print(f"  ERROR: Mismatch between feature count ({X_features.shape[0]}) and label count ({y_labels.shape[0]}).")
            return None, None
    return X_features, y_labels

def load_balanced_global_hog_data(hog_h5_base_dir, set_type):
    """Loads global HOG X and y from HDF5 created by SOH_extract_vanilla_balanced.py."""
    hog_filepath = os.path.join(hog_h5_base_dir, f'hog_data_{set_type}_balanced.h5')
    
    if not os.path.exists(hog_filepath):
        print(f"ERROR: Global HOG HDF5 file not found: {hog_filepath}")
        return None, None
    
    X_hog, y_hog_labels = None, None
    print(f"Attempting to load HOG data: {hog_filepath}")
    try:
        with h5py.File(hog_filepath, 'r') as hf:
            if 'hog_features' in hf and 'labels_numeric' in hf:
                X_hog = hf['hog_features'][:]
                y_hog_labels = hf['labels_numeric'][:]
                print(f"  Loaded global HOG for {set_type}. Features shape: {X_hog.shape}, Labels shape: {y_hog_labels.shape}")
            else: print(f"  ERROR: 'hog_features' or 'labels_numeric' not found in {hog_filepath}"); return None, None
    except Exception as e: print(f"  ERROR loading HOG from {hog_filepath}: {e}"); return None, None

    if X_hog is not None and y_hog_labels is not None and X_hog.shape[0] != y_hog_labels.shape[0]:
        print(f"  ERROR: Mismatch HOG feature count ({X_hog.shape[0]}) and label count ({y_hog_labels.shape[0]}).")
        return None, None
    return X_hog, y_hog_labels


# --- DMatrix Creation, GridSearchCV, Training/Evaluation Functions ---
# These (create_dmatrix_from_features, find_best_params_with_gridsearch, train_and_evaluate_xgb_dmatrix)
# are copied from the SPM_SOH_XGBoost_Classification_balanced.py script.
# Ensure they are present or imported correctly. For this example, I'll assume they are copied.
def create_dmatrix_from_features(X_data, y_data, set_type, feature_desc, output_dir,
                                 perform_scaling=False, scaler_to_use_or_fit=None, sample_weights=None):
    # (Copied from SPM balanced classification - it's generic)
    print(f"\n--- Creating DMatrix for: {feature_desc} ({set_type}) ---")
    filename_base = f"{set_type}_{feature_desc}"
    if set_type == 'train' and sample_weights is not None: filename_base += "_weighted"
    if perform_scaling: filename_base += "_scaled"
    dmatrix_filename = os.path.join(output_dir, f"{filename_base}.buffer")
    scaler_filename = os.path.join(output_dir, f"scaler_train_{feature_desc}.joblib")
    dmatrix_exists = os.path.exists(dmatrix_filename)
    scaler_needed_and_missing = (set_type == 'train' and perform_scaling and not os.path.exists(scaler_filename))
    if dmatrix_exists and not scaler_needed_and_missing:
        print(f"DMatrix {dmatrix_filename} exists. Skipping.")
        loaded_scaler = None
        if perform_scaling:
            if set_type == 'train' and os.path.exists(scaler_filename): loaded_scaler = joblib.load(scaler_filename)
            elif set_type == 'test' and scaler_to_use_or_fit: loaded_scaler = scaler_to_use_or_fit
        return dmatrix_filename, loaded_scaler
    X_processed = X_data.copy()
    fitted_scaler = None
    if perform_scaling:
        if set_type == 'train':
            scaler = StandardScaler(); X_processed = scaler.fit_transform(X_processed)
            joblib.dump(scaler, scaler_filename); print(f"Saved scaler to {scaler_filename}"); fitted_scaler = scaler
        elif set_type == 'test' and scaler_to_use_or_fit:
            X_processed = scaler_to_use_or_fit.transform(X_processed); fitted_scaler = scaler_to_use_or_fit
        elif set_type == 'test': print("WARNING: Scaling for test, but no scaler. Using unscaled.")
    try:
        if not X_processed.flags['C_CONTIGUOUS']: X_processed = np.ascontiguousarray(X_processed)
        dmatrix = xgb.DMatrix(X_processed, label=y_data.astype(np.float32), weight=sample_weights if set_type == 'train' else None)
        dmatrix.save_binary(dmatrix_filename); print(f"Saved DMatrix to {dmatrix_filename}")
        del X_processed, dmatrix; gc.collect()
        return dmatrix_filename, fitted_scaler
    except Exception as e: print(f"ERROR creating DMatrix for {feature_desc} ({set_type}): {e}"); return None, None

def find_best_params_with_gridsearch(X_train_for_gs, y_train_for_gs, num_classes, base_params, param_grid, cv_folds, feature_type_desc, scoring_metric, perform_scaling=True):
    # (Copied from SPM balanced classification - it's generic)
    print(f"\n--- GridSearchCV for {feature_type_desc} (Data size: {X_train_for_gs.shape[0]}) ---")
    X_scaled_for_gs = X_train_for_gs
    if perform_scaling: scaler_gs = StandardScaler(); X_scaled_for_gs = scaler_gs.fit_transform(X_train_for_gs); del scaler_gs
    weights_dict_gs = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_train_for_gs)
    weights_per_sample_gs = np.array([weights_dict_gs[label] for label in y_train_for_gs])
    best_params, best_score = None, -1.0
    final_xgb_params = {**base_params, 'num_class': num_classes}
    try: # GPU
        gpu_specific_params = {**final_xgb_params, 'device': 'cuda'}
        estimator_gpu = xgb.XGBClassifier(**gpu_specific_params)
        gs_gpu = GridSearchCV(estimator_gpu, param_grid, scoring=scoring_metric, cv=cv_folds, verbose=1, n_jobs=1)
        gs_gpu.fit(X_scaled_for_gs, y_train_for_gs, sample_weight=weights_per_sample_gs)
        best_params, best_score = gs_gpu.best_params_, gs_gpu.best_score_
        print(f"GS GPU: Best {scoring_metric}: {best_score:.4f}")
        final_xgb_params.update(best_params); final_xgb_params['device'] = 'cuda'
    except Exception as gpu_err:
        print(f"GS GPU failed: {gpu_err}. Trying CPU.")
        if 'device' in final_xgb_params: del final_xgb_params['device']
        try: # CPU
            estimator_cpu = xgb.XGBClassifier(**final_xgb_params)
            gs_cpu = GridSearchCV(estimator_cpu, param_grid, scoring=scoring_metric, cv=cv_folds, verbose=1, n_jobs=-1)
            gs_cpu.fit(X_scaled_for_gs, y_train_for_gs, sample_weight=weights_per_sample_gs)
            best_params, best_score = gs_cpu.best_params_, gs_cpu.best_score_
            print(f"GS CPU: Best {scoring_metric}: {best_score:.4f}")
            final_xgb_params.update(best_params)
        except Exception as cpu_err: print(f"GS CPU failed: {cpu_err}"); return None
    del X_scaled_for_gs, weights_per_sample_gs; gc.collect()
    if best_params: final_xgb_params['_gridsearch_scoring'] = scoring_metric; return final_xgb_params
    return None

def train_and_evaluate_xgb_dmatrix(dtrain_path, dtest_path, y_test_actual, best_params_from_search, feature_type_desc, target_class_names_list, output_results_dir_path, num_actual_classes):
    # (Copied from SPM balanced classification - it's generic)
    print(f"\n--- Training FINAL XGBoost for {feature_type_desc} using DMatrix ---")
    if not (os.path.exists(dtrain_path) and os.path.exists(dtest_path)): print("ERROR: DMatrix files missing."); return None
    dtrain = xgb.DMatrix(dtrain_path); dtest = xgb.DMatrix(dtest_path)
    dtest.set_label(y_test_actual.astype(np.float32))
    final_params = {**best_params_from_search}; gs_scoring = final_params.pop('_gridsearch_scoring', 'accuracy')
    actual_device = 'cpu'; final_params.pop('device', None) 
    if best_params_from_search.get('device') == 'cuda':
        try: 
            temp_d = xgb.DMatrix(np.random.rand(2,dtrain.num_col()),label=np.random.randint(0,num_actual_classes,2))
            xgb.train(final_params, temp_d, num_boost_round=1, device='cuda', evals=[(temp_d,'eval')], verbose_eval=False) # Added verbose_eval=False for silent check
            actual_device = 'cuda'; final_params['device'] = 'cuda'
        except: print("GPU final train check failed, using CPU.")
    
    num_boost = final_params.pop('n_estimators', 300)
    print(f"Starting final training ({actual_device}) with params: {final_params}")
    bst = xgb.train(final_params, dtrain, num_boost_round=num_boost, evals=[(dtrain,'train'),(dtest,'eval')], early_stopping_rounds=50, verbose_eval=100)
    bst.save_model(os.path.join(output_results_dir_path, f'xgb_model_{feature_type_desc}.json'))
    y_pred_proba = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1)); y_pred = np.argmax(y_pred_proba, axis=1)
    acc = accuracy_score(y_test_actual, y_pred); f1 = f1_score(y_test_actual, y_pred, average='macro', zero_division=0)
    report = classification_report(y_test_actual, y_pred, target_names=target_class_names_list, labels=np.arange(num_actual_classes), zero_division=0)
    cm = confusion_matrix(y_test_actual, y_pred, labels=np.arange(num_actual_classes))
    print(f"Final Model ({actual_device}) - Test Acc: {acc:.4f}, F1-macro: {f1:.4f}\nReport:\n{report}")
    plot_confusion_matrix(cm, target_class_names_list, f'CM {feature_type_desc} (Acc:{acc:.3f}, F1:{f1:.3f})', results_path=output_results_dir_path, filename=f'cm_xgb_{feature_type_desc}.png')
    with open(os.path.join(output_results_dir_path, f'results_xgb_{feature_type_desc}.txt'), 'w') as f_out:
        f_out.write(f"Results for {feature_type_desc}\nGS Params: {best_params_from_search}\nFinal Train Device: {actual_device}\nTest Acc: {acc:.4f}\nF1-macro: {f1:.4f}\n\nReport:\n{report}\n\nCM:\n{np.array2string(cm)}")
    print(f"Results saved for {feature_type_desc}.")
    del dtrain, dtest, bst; gc.collect()
    return True

# --- Main Execution Pipeline for BALANCED Vanilla BoVW ---
def run_balanced_vanilla_bovw_classification_pipeline():
    print("\n--- Starting BALANCED NORMAL BoVW Classification Pipeline ---")

    # --- 1. Load Label Encoder ---
    label_encoder_files = glob.glob(LABEL_ENCODER_FILE_PATTERN)
    if not label_encoder_files:
        print(f"ERROR: No label encoder file found matching pattern: {LABEL_ENCODER_FILE_PATTERN}")
        print(f"Please ensure 'create_balanced_split_for_bovw.py' has run and created the encoder PKL.")
        return
    label_encoder_path = label_encoder_files[0] # Assume one relevant encoder
    print(f"Loading label encoder from: {label_encoder_path}")
    try:
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        class_names_global = label_encoder.classes_
        num_classes_global = len(class_names_global)
        XGB_BASE_PARAMS['num_class'] = num_classes_global # Set for XGBoost
        print(f"Class names: {class_names_global} ({num_classes_global} classes)")
        if num_classes_global != 4: print(f"Warning: Expected 4 classes, got {num_classes_global}")
    except Exception as e:
        print(f"ERROR loading label encoder: {e}"); return

    # --- Define Feature Combinations ---
    # Filenames will be like X_train_sift_vanilla_k1000.npy
    sift_vanilla_name_key = "sift" # Used in load_balanced_normal_bovw_histograms_and_labels
    orb_vanilla_name_key = "orb"
    hog_global_name_key = "hog" # Used for logic, not directly in filename for HOG loader

    # Keys for feature_sets_to_run become part of filenames/descriptions
    feature_sets_to_run = {
        f"Vanilla_SIFT_k{VOCAB_SIZE_FOR_LOADING}": {sift_vanilla_name_key: True},
        f"Vanilla_ORB_k{VOCAB_SIZE_FOR_LOADING}": {orb_vanilla_name_key: True},
        "Global_HOG": {hog_global_name_key: True}, # If using HOG alone
        f"Vanilla_SIFT_k{VOCAB_SIZE_FOR_LOADING}_HOG": {sift_vanilla_name_key: True, hog_global_name_key: True},
        f"Vanilla_ORB_k{VOCAB_SIZE_FOR_LOADING}_HOG": {orb_vanilla_name_key: True, hog_global_name_key: True},
        f"Vanilla_SIFT_ORB_k{VOCAB_SIZE_FOR_LOADING}_HOG": {sift_vanilla_name_key: True, orb_vanilla_name_key: True, hog_global_name_key: True},
    }
    all_best_params_results = {}

    for feature_desc_for_file, include_features_map in feature_sets_to_run.items():
        print(f"\n\n{'='*20} Processing Feature Set: {feature_desc_for_file} {'='*20}")

        X_train_components, X_test_components = [], []
        y_train_current_set, y_test_current_set = None, None # Labels for the current feature set

        # --- Load Training Data Components ---
        valid_train_data_loaded = True
        if include_features_map.get(sift_vanilla_name_key):
            X_sift_tr, y_sift_tr = load_balanced_vanilla_bovw_histograms_and_labels(NORMAL_BOVW_HISTOGRAMS_DIR, "sift", "train", VOCAB_SIZE_FOR_LOADING)
            if X_sift_tr is None or y_sift_tr is None: valid_train_data_loaded = False; print("! SIFT train load failed")
            else: X_train_components.append(X_sift_tr); y_train_current_set = y_sift_tr if y_train_current_set is None else y_train_current_set
        
        if include_features_map.get(orb_vanilla_name_key) and valid_train_data_loaded:
            X_orb_tr, y_orb_tr = load_balanced_vanilla_bovw_histograms_and_labels(NORMAL_BOVW_HISTOGRAMS_DIR, "orb", "train", VOCAB_SIZE_FOR_LOADING)
            if X_orb_tr is None or y_orb_tr is None: valid_train_data_loaded = False; print("! ORB train load failed")
            else: X_train_components.append(X_orb_tr); y_train_current_set = y_orb_tr if y_train_current_set is None else y_train_current_set
            if y_train_current_set is not None and y_orb_tr is not None and not np.array_equal(y_train_current_set, y_orb_tr): print("! ORB train label mismatch"); valid_train_data_loaded=False

        if include_features_map.get(hog_global_name_key) and valid_train_data_loaded:
            X_hog_tr, y_hog_tr = load_balanced_global_hog_data(HOG_FEATURES_BALANCED_DIR, "train")
            if X_hog_tr is None or y_hog_tr is None: valid_train_data_loaded = False; print("! HOG train load failed")
            else: X_train_components.append(X_hog_tr); y_train_current_set = y_hog_tr if y_train_current_set is None else y_train_current_set
            if y_train_current_set is not None and y_hog_tr is not None and not np.array_equal(y_train_current_set, y_hog_tr): print("! HOG train label mismatch"); valid_train_data_loaded=False

        if not valid_train_data_loaded or not X_train_components or y_train_current_set is None:
            print(f"Skipping {feature_desc_for_file} due to training data load failure."); continue
        
        X_train_combined = np.concatenate(X_train_components, axis=1) if len(X_train_components) > 1 else X_train_components[0]
        del X_train_components; gc.collect()
        print(f"Combined training features for {feature_desc_for_file}: {X_train_combined.shape}")

        # --- Load Test Data Components ---
        valid_test_data_loaded = True
        if include_features_map.get(sift_vanilla_name_key):
            X_sift_te, y_sift_te = load_balanced_vanilla_bovw_histograms_and_labels(NORMAL_BOVW_HISTOGRAMS_DIR, "sift", "test", VOCAB_SIZE_FOR_LOADING)
            if X_sift_te is None or y_sift_te is None: valid_test_data_loaded = False; print("! SIFT test load failed")
            else: X_test_components.append(X_sift_te); y_test_current_set = y_sift_te if y_test_current_set is None else y_test_current_set

        if include_features_map.get(orb_vanilla_name_key) and valid_test_data_loaded:
            X_orb_te, y_orb_te = load_balanced_vanilla_bovw_histograms_and_labels(NORMAL_BOVW_HISTOGRAMS_DIR, "orb", "test", VOCAB_SIZE_FOR_LOADING)
            if X_orb_te is None or y_orb_te is None: valid_test_data_loaded = False; print("! ORB test load failed")
            else: X_test_components.append(X_orb_te); y_test_current_set = y_orb_te if y_test_current_set is None else y_test_current_set
            if y_test_current_set is not None and y_orb_te is not None and not np.array_equal(y_test_current_set, y_orb_te): print("! ORB test label mismatch"); valid_test_data_loaded=False
        
        if include_features_map.get(hog_global_name_key) and valid_test_data_loaded:
            X_hog_te, y_hog_te = load_balanced_global_hog_data(HOG_FEATURES_BALANCED_DIR, "test")
            if X_hog_te is None or y_hog_te is None: valid_test_data_loaded = False; print("! HOG test load failed")
            else: X_test_components.append(X_hog_te); y_test_current_set = y_hog_te if y_test_current_set is None else y_test_current_set
            if y_test_current_set is not None and y_hog_te is not None and not np.array_equal(y_test_current_set, y_hog_te): print("! HOG test label mismatch"); valid_test_data_loaded=False

        if not valid_test_data_loaded or not X_test_components or y_test_current_set is None:
            print(f"Skipping {feature_desc_for_file} due to test data load failure."); continue
        
        X_test_combined = np.concatenate(X_test_components, axis=1) if len(X_test_components) > 1 else X_test_components[0]
        del X_test_components; gc.collect()
        print(f"Combined test features for {feature_desc_for_file}: {X_test_combined.shape}")

        # Ensure train and test labels align if all components loaded successfully
        if not (np.array_equal(y_train_current_set, y_test_current_set) if (y_train_current_set is not None and y_test_current_set is not None and len(y_train_current_set) == len(y_test_current_set) and feature_desc_for_file == hog_global_name_key) # Special case if HOG is the ONLY feature
                else (X_train_combined.shape[0] == y_train_current_set.shape[0] and X_test_combined.shape[0] == y_test_current_set.shape[0])):
             print(f"! CRITICAL: Final label alignment check failed for {feature_desc_for_file}. Train features {X_train_combined.shape[0]} vs labels {y_train_current_set.shape[0]}. Test features {X_test_combined.shape[0]} vs labels {y_test_current_set.shape[0]}. Skipping.")
             del X_train_combined, X_test_combined, y_train_current_set, y_test_current_set; gc.collect()
             continue

        # --- GridSearchCV (on sample or full train) ---
        X_gs_data, y_gs_data = X_train_combined, y_train_current_set
        if SAMPLE_FRACTION_FOR_GRIDSEARCH < 1.0 and len(y_train_current_set) > 10 : # Min samples for stratify
            try:
                # train_test_split returns a list for X if X is a list, we need the features part
                # Here X_train_combined is already a numpy array
                _, X_gs_data, _, y_gs_data = train_test_split(
                    X_train_combined, y_train_current_set, 
                    test_size=(1.0 - SAMPLE_FRACTION_FOR_GRIDSEARCH), # Correct way to get train_size
                    random_state=42, stratify=y_train_current_set
                )
                print(f"Using {X_gs_data.shape[0]} samples for GridSearchCV ({SAMPLE_FRACTION_FOR_GRIDSEARCH*100:.1f}% of train).")
            except ValueError as e_gs_split: # Handle cases like not enough samples in a class
                print(f"Warning: Stratified split for GridSearchCV failed ({e_gs_split}). Using full train set for GS.")
                X_gs_data, y_gs_data = X_train_combined, y_train_current_set
        
        current_best_params = find_best_params_with_gridsearch(
            X_gs_data, y_gs_data, num_classes_global,
            XGB_BASE_PARAMS, PARAM_GRID_XGB, GRIDSEARCH_CV_FOLDS, 
            feature_desc_for_file, GRIDSEARCH_SCORING, perform_scaling=True
        )
        if current_best_params is None: print(f"GridSearchCV failed for {feature_desc_for_file}. Skipping final train."); continue
        all_best_params_results[feature_desc_for_file] = current_best_params
        if X_gs_data is not X_train_combined : del X_gs_data # Clean sample if it was a copy
        if y_gs_data is not y_train_current_set : del y_gs_data
        gc.collect()
        
        # --- DMatrices (Scaled & Weighted for Train) ---
        train_class_weights = compute_class_weight('balanced', classes=np.arange(num_classes_global), y=y_train_current_set)
        train_sample_w = np.array([train_class_weights[lbl] for lbl in y_train_current_set])
        
        dtrain_path, fitted_scaler = create_dmatrix_from_features(
            X_train_combined, y_train_current_set, "train", feature_desc_for_file, DMATRIX_CACHE_DIR_VANILLA_BALANCED,
            perform_scaling=True, sample_weights=train_sample_w
        )
        del X_train_combined; gc.collect()
        
        dtest_path, _ = create_dmatrix_from_features(
            X_test_combined, y_test_current_set, "test", feature_desc_for_file, DMATRIX_CACHE_DIR_VANILLA_BALANCED,
            perform_scaling=True, scaler_to_use_or_fit=fitted_scaler # Use scaler from train
        )
        del X_test_combined, fitted_scaler; gc.collect()

        if not dtrain_path or not dtest_path: print(f"DMatrix creation failed for {feature_desc_for_file}. Skipping."); continue

        # --- Train Final Model ---
        train_and_evaluate_xgb_dmatrix(
            dtrain_path, dtest_path, y_test_current_set,
            current_best_params, feature_desc_for_file, class_names_global,
            RESULTS_DIR_XGB_VANILLA_BALANCED, num_classes_global
        )
        del y_train_current_set, y_test_current_set; gc.collect()

    print("\n--- BALANCED NORMAL BoVW Classification Pipeline Complete ---")
    print("Best parameters from GridSearchCV:")
    for name, params in all_best_params_results.items(): print(f"  {name}: {params}")
    print(f"Results saved to: {RESULTS_DIR_XGB_VANILLA_BALANCED}")
