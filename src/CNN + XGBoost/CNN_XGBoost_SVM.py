import numpy as np
import os
import pickle
import warnings
import joblib
import glob
import h5py
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
# from sklearn.svm import SVC # If you intend to add SVM later

# --- Configuration for BALANCED FC FEATURES ---
# Base directory where the PyTorch BALANCED FC feature extraction script saved its output
CNN_FEATURES_PYTORCH_BASE_DIR_BALANCED = r"E:\CV_Features_CNN_PyTorch_Balanced" # Matches new base dir

# Model name used in the BALANCED FC PyTorch feature extraction script
CNN_MODEL_NAME_BALANCED_FC = "AlexNet_Places365_PyTorch_FC_Features" # Matches new model name

# Directory where the .h5 BALANCED FC feature files are located
CNN_EXTRACTED_FEATURES_DIR_BALANCED = os.path.join(
    CNN_FEATURES_PYTORCH_BASE_DIR_BALANCED,
    "cnn_extracted_features",
    CNN_MODEL_NAME_BALANCED_FC
)

# Shared Splits and Label Info from the BALANCED splitting script
BASE_FEATURES_DIR_FOR_SPLITS_BALANCED = r"E:\CV_features" # Base for splits
SPLITS_SUBDIR_NAME_BALANCED = "train_test_splits_4cat_balanced" # Subdir for balanced splits

# Full path to the directory containing the BALANCED NPZ and PKL files
SPLITS_DIR_COMMON_BALANCED = os.path.join(
    BASE_FEATURES_DIR_FOR_SPLITS_BALANCED,
    SPLITS_SUBDIR_NAME_BALANCED
)

# Label encoder file from the BALANCED splitting script
LABEL_ENCODER_FILE_BALANCED = os.path.join(
    SPLITS_DIR_COMMON_BALANCED,
    "broad_label_encoder_4cat_balanced.pkl" # Matches new PKL name
)

# Results Directory for CNN (Balanced FC) + XGBoost
RESULTS_SUBDIR_NAME_BALANCED = "classification_results_XGB_CNN_4cat_balanced_fc" # New results subdir
RESULTS_DIR_XGB_CNN_BALANCED = os.path.join(
    CNN_FEATURES_PYTORCH_BASE_DIR_BALANCED, # Use balanced base dir
    RESULTS_SUBDIR_NAME_BALANCED,
    CNN_MODEL_NAME_BALANCED_FC # Use balanced model name
)


# --- Helper Functions (plot_confusion_matrix and train_and_evaluate_xgb are mostly the same) ---
def plot_confusion_matrix(cm, classes, plot_title='Confusion matrix', cmap=plt.cm.Blues, results_path=None, filename=None):
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

def train_and_evaluate_xgb(X_train_data, y_train_labels, X_test_data, y_test_labels,
                           feature_type_desc, target_class_names,
                           output_results_dir, perform_scaling=False):
    print(f"\n--- Training XGBoost for {feature_type_desc} ---")

    if X_train_data is None or X_train_data.size == 0 or X_test_data is None or X_test_data.size == 0:
        print(f"Skipping XGBoost for {feature_type_desc}: Missing/empty feature data.")
        return None

    X_train_processed = X_train_data.copy()
    X_test_processed = X_test_data.copy()
    os.makedirs(output_results_dir, exist_ok=True)

    print("Calculating sample weights based on class balance...")
    # Ensure y_train_labels are integers for compute_sample_weight if they are not already
    y_train_labels_int = np.array(y_train_labels, dtype=int)
    sample_weights = compute_sample_weight('balanced', y_train_labels_int)
    print("Sample weights calculated.")

    try:
        base_estimator_xgb = xgb.XGBClassifier(objective='multi:softprob',
                                            num_class=len(target_class_names),
                                            tree_method='hist',
                                            device='cuda',
                                            eval_metric='mlogloss',
                                            random_state=42,
                                            use_label_encoder=False)
        print("XGBoost initialized with CUDA device.")
    except xgb.core.XGBoostError as e:
        if "Cannot find CUDA device" in str(e) or "No GPU found" in str(e) or "XGBoost Library (build = GPU) not found" in str(e):
            print("XGBoost CUDA device not found or GPU support not built. Falling back to CPU for XGBoost.")
            base_estimator_xgb = xgb.XGBClassifier(objective='multi:softprob',
                                                num_class=len(target_class_names),
                                                tree_method='hist',
                                                eval_metric='mlogloss',
                                                random_state=42,
                                                use_label_encoder=False)
        else:
            print(f"An unexpected XGBoost error occurred: {e}")
            return None

    if perform_scaling: # Generally not needed for XGBoost but kept as an option
        print(f"Scaling features for XGBoost: {feature_type_desc}...")
        scaler_xgb = StandardScaler()
        X_train_processed = scaler_xgb.fit_transform(X_train_processed)
        X_test_processed = scaler_xgb.transform(X_test_processed)
        scaler_filename_xgb = os.path.join(output_results_dir, f'scaler_xgb_{feature_type_desc.replace(" ", "_").replace("/", "_")}.joblib')
        joblib.dump(scaler_xgb, scaler_filename_xgb)
        print(f"Saved XGBoost scaler for {feature_type_desc} to {scaler_filename_xgb}")
    '''
    param_grid_xgb = {
        'n_estimators': [200, 300, 400], # Can expand if time permits
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [5, 7, 9],
        'colsample_bytree': [0.7, 0.8, 0.9] # Feature fraction
    }
    '''
    param_grid_xgb = { # Smaller grid for faster runs, expand if needed
        'n_estimators': [200, 300],       # Number of trees
        'learning_rate': [0.05, 0.1],   # Step size shrinkage
        'max_depth': [5, 7],            # Max depth of a tree
        # 'colsample_bytree': [0.8],    # Subsample ratio of columns when constructing each tree
        # 'subsample': [0.8],           # Subsample ratio of the training instances
    }
    print(f"Performing GridSearchCV for XGBoost on {feature_type_desc} (cv=3)...")
    n_jobs_cv = 1
    current_device = base_estimator_xgb.get_params().get('device', 'cpu')
    if 'cuda' not in current_device:
        print(f"XGBoost is configured for CPU (device: {current_device}). Using n_jobs_cv = -1 for GridSearchCV.")
        n_jobs_cv = -1
    else:
        print(f"XGBoost is configured for GPU (device: {current_device}). Using n_jobs_cv = 1 for GridSearchCV.")

    xgb_grid_search = GridSearchCV(estimator=base_estimator_xgb,
                                   param_grid=param_grid_xgb,
                                   scoring='accuracy', # or 'f1_weighted' for imbalanced (though sample_weight helps)
                                   cv=3, verbose=2, n_jobs=n_jobs_cv)

    xgb_grid_search.fit(X_train_processed, y_train_labels_int, sample_weight=sample_weights)
    best_xgb_model = xgb_grid_search.best_estimator_
    print(f"Best XGBoost parameters for {feature_type_desc}: {xgb_grid_search.best_params_}")
    print(f"Best CV score for {feature_type_desc}: {xgb_grid_search.best_score_:.4f}")


    model_filename_xgb = os.path.join(output_results_dir, f'xgb_model_{feature_type_desc.replace(" ", "_").replace("/", "_")}.joblib')
    joblib.dump(best_xgb_model, model_filename_xgb)
    print(f"Saved best XGBoost model for {feature_type_desc} to {model_filename_xgb}")

    print(f"\n--- Evaluating XGBoost for {feature_type_desc} ---")
    y_pred_labels_xgb = best_xgb_model.predict(X_test_processed)
    accuracy_val_xgb = accuracy_score(y_test_labels, y_pred_labels_xgb)
    class_report_str_xgb = classification_report(y_test_labels, y_pred_labels_xgb, target_names=target_class_names, zero_division=0)
    conf_matrix_xgb = confusion_matrix(y_test_labels, y_pred_labels_xgb, labels=np.arange(len(target_class_names)))

    print(f"Accuracy (XGBoost - {feature_type_desc}): {accuracy_val_xgb:.4f}")
    print(f"Classification Report (XGBoost - {feature_type_desc}):\n{class_report_str_xgb}")
    plot_confusion_matrix(conf_matrix_xgb, classes=target_class_names,
                          plot_title=f'CM for XGBoost - {feature_type_desc} (Acc: {accuracy_val_xgb:.3f})',
                          results_path=output_results_dir,
                          filename=f'cm_xgb_{feature_type_desc.replace(" ", "_").replace("/", "_")}.png')

    results_text_file_xgb = os.path.join(output_results_dir, f'results_xgb_{feature_type_desc.replace(" ", "_").replace("/", "_")}.txt')
    with open(results_text_file_xgb, 'w') as f:
        f.write(f"--- XGBoost Results for {feature_type_desc} ---\n")
        f.write(f"Scaling: {perform_scaling}\nParams: {xgb_grid_search.best_params_}\nCV Score (Accuracy): {xgb_grid_search.best_score_:.4f}\nTest Accuracy: {accuracy_val_xgb:.4f}\n\nReport:\n{class_report_str_xgb}\n\nCM:\n{np.array2string(conf_matrix_xgb)}")
    print(f"Saved XGBoost results for {feature_type_desc} to {results_text_file_xgb}")
    return best_xgb_model


# --- Feature Loading Functions (Updated for new HDF5 structure) ---
def load_cnn_features_from_balanced_hdf5(cnn_extracted_features_dir, cnn_model_name, set_type="train"):
    """
    Loads CNN features and numeric labels from an HDF5 file created by the BALANCED pipeline.
    Assumes features are stored under 'features' and labels under 'labels_numeric'.
    Uses glob to find files matching the pattern:
    X_{set_type}_{cnn_model_name_lower}_*balanced*.h5 (or similar based on actual filenames)
    Args:
        cnn_extracted_features_dir (str): The directory containing the HDF5 files (e.g., .../AlexNet_Places365_PyTorch_FC_Features)
        cnn_model_name (str): The name of the CNN model (e.g., "AlexNet_Places365_PyTorch_FC_Features")
        set_type (str): "train" or "test"
    Returns:
        tuple: (features_data, labels_numeric_data) or (None, None) if not found/error.
    """
    cnn_model_name_lower = cnn_model_name.lower()
    # Adjust filename_pattern to match the output of your balanced feature extraction.
    # It might include a suffix like "_balanced_fc_features"
    # Example: X_train_alexnet_places365_pytorch_fc_features_balanced_fc_features.h5
    filename_pattern = f"X_{set_type}_{cnn_model_name_lower}_*balanced*.h5" # More flexible glob pattern
    search_path = os.path.join(cnn_extracted_features_dir, filename_pattern)
    potential_files = glob.glob(search_path)

    if not potential_files:
        print(f"Warning: No BALANCED {cnn_model_name_lower} HDF5 features file found matching pattern: '{filename_pattern}' in '{cnn_extracted_features_dir}'")
        return None, None

    if len(potential_files) > 1:
        print(f"Warning: Multiple files found for '{filename_pattern}' in '{cnn_extracted_features_dir}'. Using the first one: {potential_files[0]}")
        print(f"  All found: {potential_files}")

    filepath = potential_files[0]

    if os.path.exists(filepath):
        print(f"Loading {set_type} {cnn_model_name} BALANCED CNN features and labels from HDF5: {filepath}")
        try:
            with h5py.File(filepath, 'r') as hf:
                # Check for the new label dataset name 'labels_numeric'
                if 'features' in hf and 'labels_numeric' in hf:
                    features_data = hf['features'][:]
                    labels_numeric_data = hf['labels_numeric'][:] # Load numeric labels
                    print(f"  Features shape: {features_data.shape}, Numeric Labels shape: {labels_numeric_data.shape}")
                    return features_data, labels_numeric_data
                else:
                    missing_keys = []
                    if 'features' not in hf: missing_keys.append("'features'")
                    if 'labels_numeric' not in hf: missing_keys.append("'labels_numeric'") # Updated key
                    print(f"Error: Required dataset(s) {', '.join(missing_keys)} not found in HDF5 file {filepath}")
                    return None, None
        except Exception as e:
            print(f"Error loading data from HDF5 file {filepath}: {e}")
            return None, None
    else:
        print(f"Warning: HDF5 features file somehow not found after glob match (this should not happen): {filepath}")
        return None, None


# --- Main Execution Function ---
def run_cnn_xgb_classification_balanced_fc():
    # Ensure the main results directory exists (using new balanced path)
    os.makedirs(RESULTS_DIR_XGB_CNN_BALANCED, exist_ok=True)

    warnings.filterwarnings("ignore", message=".*use_label_encoder.*", category=UserWarning, module="xgboost.core")

    # --- 1. Load Label Encoder (from BALANCED split) ---
    print("--- Loading Common Data (Label Encoder for BALANCED split) ---")
    print(f"Attempting to load label encoder from: {LABEL_ENCODER_FILE_BALANCED}")
    if not os.path.exists(LABEL_ENCODER_FILE_BALANCED):
        print(f"ERROR: Label encoder file not found at {LABEL_ENCODER_FILE_BALANCED}.")
        print("Please ensure the BALANCED data splitting script has run successfully and paths are correct.")
        exit()
    try:
        with open(LABEL_ENCODER_FILE_BALANCED, 'rb') as f:
            label_encoder = pickle.load(f)
        class_names = label_encoder.classes_
        print(f"Class names for classification (from balanced split): {class_names}")
        if len(class_names) != 4: # Assuming 4 broad categories
             print(f"Warning: Expected 4 class names, got {len(class_names)}. This might be okay if intended.")
    except Exception as e:
        print(f"ERROR loading label encoder: {e}")
        exit()

    print(f"\n--- Starting CNN ({CNN_MODEL_NAME_BALANCED_FC}) + XGBoost Classification Pipeline (BALANCED FC Features) ---")
    print(f"Features will be loaded from: {CNN_EXTRACTED_FEATURES_DIR_BALANCED}")
    print(f"Results will be saved to: {RESULTS_DIR_XGB_CNN_BALANCED}")

    # Load CNN Extracted BALANCED FC Features AND THEIR NUMERIC LABELS
    print(f"\n--- Loading {CNN_MODEL_NAME_BALANCED_FC} Extracted Features & Labels ---")
    X_train_cnn, y_train_cnn_numeric = load_cnn_features_from_balanced_hdf5(
        CNN_EXTRACTED_FEATURES_DIR_BALANCED,
        CNN_MODEL_NAME_BALANCED_FC,
        "train"
    )
    X_test_cnn, y_test_cnn_numeric = load_cnn_features_from_balanced_hdf5(
        CNN_EXTRACTED_FEATURES_DIR_BALANCED,
        CNN_MODEL_NAME_BALANCED_FC,
        "test"
    )

    # --- Train and Evaluate XGBoost Classifiers ---
    print(f"\n\n" + "="*20 + f" XGBoost for {CNN_MODEL_NAME_BALANCED_FC} Features " + "="*20)

    if X_train_cnn is not None and y_train_cnn_numeric is not None and \
       X_test_cnn is not None and y_test_cnn_numeric is not None:

        if len(X_train_cnn) != len(y_train_cnn_numeric):
            print(f"CRITICAL ERROR: Mismatch between loaded train features ({len(X_train_cnn)}) and train numeric labels ({len(y_train_cnn_numeric)}) from HDF5.")
            exit()
        if len(X_test_cnn) != len(y_test_cnn_numeric):
            print(f"CRITICAL ERROR: Mismatch between loaded test features ({len(X_test_cnn)}) and test numeric labels ({len(y_test_cnn_numeric)}) from HDF5.")
            exit()

        train_and_evaluate_xgb(
            X_train_cnn, y_train_cnn_numeric, # Use numeric labels directly
            X_test_cnn, y_test_cnn_numeric,   # Use numeric labels directly
            f"{CNN_MODEL_NAME_BALANCED_FC}_Features",
            class_names,
            RESULTS_DIR_XGB_CNN_BALANCED, # Save to new results directory
            perform_scaling=False # XGBoost usually doesn't need scaling
        )
    else:
        print(f"Skipping {CNN_MODEL_NAME_BALANCED_FC} features for XGBoost due to missing feature or label data from HDF5 files.")
        print(f"Please check the output of 'load_cnn_features_from_balanced_hdf5' and ensure that HDF5 files exist in '{CNN_EXTRACTED_FEATURES_DIR_BALANCED}' with the correct datasets ('features', 'labels_numeric').")

    print(f"\n--- CNN ({CNN_MODEL_NAME_BALANCED_FC}) + XGBoost Classification Pipeline Complete ---")
    print(f"All results, models, and plots saved in: {RESULTS_DIR_XGB_CNN_BALANCED}")