import numpy as np
import os
import pickle
import warnings
import joblib
import glob # <<< ADDED for finding feature files
import h5py # <<< ADDED for HDF5 support
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

# --- Helper Functions ---
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
                           output_results_dir, perform_scaling=False):
    print(f"\n--- Training XGBoost for {feature_type_desc} ---")

    if X_train_data is None or X_train_data.size == 0 or X_test_data is None or X_test_data.size == 0:
        print(f"Skipping XGBoost for {feature_type_desc}: Missing/empty feature data.")
        return None

    X_train_processed = X_train_data.copy()
    X_test_processed = X_test_data.copy()

    print("Calculating sample weights based on class balance...")
    sample_weights = compute_sample_weight('balanced', y_train_labels)
    print("Sample weights calculated.")

    try:
        base_estimator_xgb = xgb.XGBClassifier(objective='multi:softprob',
                                            num_class=len(target_class_names),
                                            tree_method='hist', # Use 'hist' for faster training
                                            device='cuda',      # Try to use GPU
                                            eval_metric='mlogloss',
                                            random_state=42,
                                            use_label_encoder=False) # Deprecated, set to False
        print("XGBoost initialized with CUDA device.")
    except xgb.core.XGBoostError as e:
        if "Cannot find CUDA device" in str(e) or "No GPU found" in str(e) or "XGBoost Library (build = GPU) not found" in str(e):
            print("XGBoost CUDA device not found or GPU support not built. Falling back to CPU for XGBoost.")
            base_estimator_xgb = xgb.XGBClassifier(objective='multi:softprob',
                                                num_class=len(target_class_names),
                                                tree_method='hist', # 'hist' is generally good
                                                eval_metric='mlogloss',
                                                random_state=42,
                                                use_label_encoder=False)
        else:
            print(f"An unexpected XGBoost error occurred: {e}")
            return None


    if perform_scaling:
        print(f"Scaling features for XGBoost: {feature_type_desc}...")
        scaler_xgb = StandardScaler()
        X_train_processed = scaler_xgb.fit_transform(X_train_processed)
        X_test_processed = scaler_xgb.transform(X_test_processed)
        scaler_filename_xgb = os.path.join(output_results_dir, f'scaler_xgb_{feature_type_desc.replace(" ", "_").replace("/", "_")}.joblib')
        joblib.dump(scaler_xgb, scaler_filename_xgb)
        print(f"Saved XGBoost scaler for {feature_type_desc} to {scaler_filename_xgb}")

    param_grid_xgb = {
        'n_estimators': [200, 300], # Reduced for faster grid search, adjust as needed
        'learning_rate': [0.05, 0.1],
        'max_depth': [5, 7], # Reduced for faster grid search
        # Add 'colsample_bytree': [0.7, 0.8] if you want to explore more
    }

    print(f"Performing GridSearchCV for XGBoost on {feature_type_desc} (cv=3)...")
    n_jobs_cv = 1 # Keep n_jobs=1 if using GPU with XGBoost, or -1 for CPU
    if 'cuda' not in base_estimator_xgb.get_params().get('device', 'cpu'): # Check if actually on CPU
         n_jobs_cv = -1 # Use all available cores if on CPU

    xgb_grid_search = GridSearchCV(estimator=base_estimator_xgb,
                                   param_grid=param_grid_xgb,
                                   scoring='accuracy',
                                   cv=3, verbose=2, n_jobs=n_jobs_cv)

    xgb_grid_search.fit(X_train_processed, y_train_labels, sample_weight=sample_weights)
    best_xgb_model = xgb_grid_search.best_estimator_
    print(f"Best XGBoost parameters for {feature_type_desc}: {xgb_grid_search.best_params_}")

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
    print(f"Confusion Matrix (XGBoost - {feature_type_desc}):\n{conf_matrix_xgb}")
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

# --- 3. Feature Loading Functions ---
# MODIFIED load_cnn_features function
def load_cnn_features(cnn_features_dir, cnn_model_name_lower, set_type="train"):
    """
    Loads CNN features from an HDF5 file.
    Assumes features are stored under the key 'features' in the HDF5 file.
    Uses glob to find files matching the pattern:
    X_{set_type}_{cnn_model_name_lower}_features*.h5
    """
    # Construct a glob pattern to find the feature file
    # This pattern accounts for potential suffixes like _subsetNUMBER_seedNUMBER
    filename_pattern = f"X_{set_type}_{cnn_model_name_lower}_features*.h5"
    search_path = os.path.join(cnn_features_dir, filename_pattern)
    potential_files = glob.glob(search_path)

    if not potential_files:
        print(f"Warning: No {cnn_model_name_lower} HDF5 features file found matching pattern: {filename_pattern} in {cnn_features_dir}")
        return None

    if len(potential_files) > 1:
        # If multiple files match, you might want to add logic to select the correct one.
        # For now, we'll use the first one found and print a warning.
        print(f"Warning: Multiple files found for {filename_pattern} in {cnn_features_dir}. Using the first one: {potential_files[0]}")

    filepath = potential_files[0] # Use the first matching file

    if os.path.exists(filepath):
        print(f"Loading {set_type} {cnn_model_name_lower} CNN features from HDF5: {filepath}")
        try:
            with h5py.File(filepath, 'r') as hf:
                if 'features' in hf:
                    data = hf['features'][:] # Load the 'features' dataset
                    print(f"  Shape: {data.shape}")
                    return data
                else:
                    print(f"Error: 'features' dataset not found in HDF5 file {filepath}")
                    return None
        except Exception as e:
            print(f"Error loading data from HDF5 file {filepath}: {e}")
            return None
    else:
        # This case should ideally not be reached if glob.glob behaves as expected
        print(f"Warning: {cnn_model_name_lower} HDF5 features file somehow not found after glob match: {filepath}")
        return None


# --- Main Execution Function ---
def run_cnn_xgb_classification():
    # --- Configuration ---
    # Base directory where the PyTorch feature extraction script saved its output folder
    # e.g., if extraction script saves to "E:\CV_Features_CNN_PyTorch\cnn_extracted_features\AlexNet_Places365_PyTorch"
    # then CNN_FEATURES_PYTORCH_BASE_DIR should be "E:\CV_Features_CNN_PyTorch"
    CNN_FEATURES_PYTORCH_BASE_DIR = r"E:\CV_Features_CNN_PyTorch" # Or your equivalent like "E:\CV_Features_CNN" if that's the parent

    # Model name used in the PyTorch feature extraction script
    CNN_MODEL_NAME_FROM_EXTRACTION = "AlexNet_Places365_PyTorch"

    # Directory where the .h5 feature files are located
    CNN_EXTRACTED_FEATURES_DIR = os.path.join(CNN_FEATURES_PYTORCH_BASE_DIR, "cnn_extracted_features", CNN_MODEL_NAME_FROM_EXTRACTION)

    # Shared Splits and Label Info (ensure these paths are correct)
    SPLITS_DIR_COMMON = os.path.join(r"E:\CV_features", "train_test_splits_4cat_revised")
    NPZ_FILE = os.path.join(SPLITS_DIR_COMMON, "train_test_split_data_4cat_revised.npz")
    LABEL_ENCODER_FILE = os.path.join(SPLITS_DIR_COMMON, "broad_label_encoder_4cat_revised.pkl")

    # Results Directory for CNN + XGBoost
    # This will create a folder like: E:\CV_Features_CNN_PyTorch\classification_results_XGB_CNN_4cat\AlexNet_Places365_PyTorch
    RESULTS_DIR_XGB_CNN = os.path.join(CNN_FEATURES_PYTORCH_BASE_DIR, "classification_results_XGB_CNN_4cat", CNN_MODEL_NAME_FROM_EXTRACTION)
    os.makedirs(RESULTS_DIR_XGB_CNN, exist_ok=True)

    # Suppress XGBoost UserWarning about 'use_label_encoder'
    warnings.filterwarnings("ignore", message=".*use_label_encoder.*", category=UserWarning, module="xgboost.core")

    # --- 1. Load Labels, Indices, and Label Encoder ---
    print("--- Loading Common Data (Labels, Splits, Encoder) ---")
    print(f"Loading train/test split data from: {NPZ_FILE}")
    try:
        split_data = np.load(NPZ_FILE)
        # These are original dataset indices, ensure your features correspond to these splits.
        # The PyTorch script used 'actual_train_subset_indices' and 'actual_test_subset_indices'
        # which were derived from 'subset_train_indices_npz' and 'subset_test_indices_npz'.
        # The labels y_train and y_test loaded here should align with the features extracted.
        # The h5 files from the extraction script also save 'subset_indices' and 'labels' which can be
        # used for an extra layer of verification if needed, but typically y_train/y_test from NPZ
        # are the ground truth for training the classifier.
        y_train = split_data['train_labels_numeric']
        y_test = split_data['test_labels_numeric']
        # train_indices_from_npz = split_data['train_indices'] # Original dataset indices for train set
        # test_indices_from_npz = split_data['test_indices']   # Original dataset indices for test set

    except FileNotFoundError:
        print(f"ERROR: NPZ file not found at {NPZ_FILE}. Ensure label splitting script has run.")
        exit()
    except KeyError as e:
        print(f"ERROR: Missing key {e} in NPZ file {NPZ_FILE}. Expected 'train_labels_numeric', 'test_labels_numeric'.")
        exit()

    print(f"Loaded {len(y_train)} train labels.")
    print(f"Loaded {len(y_test)} test labels.")

    print(f"Loading label encoder from: {LABEL_ENCODER_FILE}")
    try:
        with open(LABEL_ENCODER_FILE, 'rb') as f:
            label_encoder = pickle.load(f)
        class_names = label_encoder.classes_
        print(f"Class names for classification: {class_names}")
        if len(class_names) != 4: # Assuming 4 broad categories
            print(f"Warning: Expected 4 class names, got {len(class_names)}.")
    except FileNotFoundError:
        print(f"ERROR: Label encoder file not found at {LABEL_ENCODER_FILE}.")
        exit()

    print(f"\n--- Starting CNN ({CNN_MODEL_NAME_FROM_EXTRACTION}) + XGBoost Classification Pipeline ---")

    # Load CNN Extracted Features
    print(f"\n--- Loading {CNN_MODEL_NAME_FROM_EXTRACTION} Extracted Features ---")
    # The cnn_model_name_lower argument should match the naming from the extraction script
    X_train_cnn = load_cnn_features(CNN_EXTRACTED_FEATURES_DIR, CNN_MODEL_NAME_FROM_EXTRACTION.lower(), "train")
    X_test_cnn = load_cnn_features(CNN_EXTRACTED_FEATURES_DIR, CNN_MODEL_NAME_FROM_EXTRACTION.lower(), "test")

    # --- Train and Evaluate XGBoost Classifiers ---
    print(f"\n\n" + "="*20 + f" XGBoost for {CNN_MODEL_NAME_FROM_EXTRACTION} Features " + "="*20)

    if X_train_cnn is not None and X_test_cnn is not None:
        if len(X_train_cnn) != len(y_train):
            print(f"ERROR: Mismatch between number of loaded train features ({len(X_train_cnn)}) and train labels ({len(y_train)}). Check feature extraction and NPZ file alignment.")
            exit()
        if len(X_test_cnn) != len(y_test):
            print(f"ERROR: Mismatch between number of loaded test features ({len(X_test_cnn)}) and test labels ({len(y_test)}). Check feature extraction and NPZ file alignment.")
            exit()

        train_and_evaluate_xgb(X_train_cnn, y_train, X_test_cnn, y_test,
                               f"{CNN_MODEL_NAME_FROM_EXTRACTION}_Features", class_names, RESULTS_DIR_XGB_CNN,
                               perform_scaling=False) # Scaling is generally not critical for XGBoost
    else:
        print(f"Skipping {CNN_MODEL_NAME_FROM_EXTRACTION} features for XGBoost due to missing data.")


    print(f"\n--- CNN ({CNN_MODEL_NAME_FROM_EXTRACTION}) + XGBoost Classification Pipeline Complete ---")
    print(f"Results saved in: {RESULTS_DIR_XGB_CNN}")