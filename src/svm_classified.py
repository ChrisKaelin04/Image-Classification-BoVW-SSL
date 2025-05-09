import numpy as np
import os
import pickle
import warnings
import joblib
import h5py # For loading HOG data
from sklearn.svm import SVC # Still keeping SVM for comparison or if you want to run both
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler # SVMs often benefit more from scaling than tree models
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb # XGBoost import

# --- Configuration ---
FEATURES_DIR = "E:\CV_features"
SPLITS_DIR = os.path.join(FEATURES_DIR, "train_test_splits_4cat_revised")
NPZ_FILE = os.path.join(SPLITS_DIR, "train_test_split_data_4cat_revised.npz")
LABEL_ENCODER_FILE = os.path.join(SPLITS_DIR, "broad_label_encoder_4cat_revised.pkl")
warnings.filterwarnings("ignore", message="Parameters: {.*use_label_encoder.*} are not used.", category=UserWarning, module="xgboost.core")
BOVW_FEATURES_DIR = os.path.join(FEATURES_DIR, "bovw_features_4cat")
HOG_DATA_FILE = os.path.join(FEATURES_DIR, 'hog_data.h5')

RESULTS_DIR_SVM = os.path.join(FEATURES_DIR, "classification_results_SVM_SOH_4cat")
RESULTS_DIR_XGB = os.path.join(FEATURES_DIR, "classification_results_XGB_SOH_4cat") # Separate results for XGB
os.makedirs(RESULTS_DIR_SVM, exist_ok=True)
os.makedirs(RESULTS_DIR_XGB, exist_ok=True)

VOCAB_SIZE = 1000 # Used for naming SIFT/ORB features if convention includes it
split_data = None

# Determine which SVM implementation to use (CPU sklearn or placeholder for GPU)
# For this version, we'll focus on sklearn SVC (CPU) and add XGBoost (GPU)
# If you had ThunderSVM or cuML successfully installed, you'd manage SVM_IMPLEMENTATION here.
# For now, to avoid errors if ThunderSVM isn't built:
from sklearn.svm import SVC as SklearnSVC
SVM_IMPLEMENTATION = SklearnSVC
print(f"Using SVM implementation: {SVM_IMPLEMENTATION.__name__}")


# --- 1. Load Labels, Indices, and Label Encoder ---
print("--- Loading Data ---")
print(f"Loading train/test split data from: {NPZ_FILE}")
try:
    split_data = np.load(NPZ_FILE)
    train_indices = split_data['train_indices'] # Original dataset indices for train set
    test_indices = split_data['test_indices']   # Original dataset indices for test set
    y_train = split_data['train_labels_numeric'] # Numeric (0-3) broad category labels
    y_test = split_data['test_labels_numeric']
except FileNotFoundError:
    print(f"ERROR: NPZ file not found at {NPZ_FILE}. Ensure label splitting script has run.")
    exit()
except KeyError as e:
    print(f"ERROR: Missing key {e} in NPZ file {NPZ_FILE}. Check keys.")
    print(f"Available keys: {split_data.files if split_data else 'NPZ not loaded'}")
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
    if len(class_names) != 4: # Assuming 4 broad categories
        print(f"Warning: Expected 4 class names, got {len(class_names)}.")
except FileNotFoundError:
    print(f"ERROR: Label encoder file not found at {LABEL_ENCODER_FILE}.")
    exit()

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


# --- XGBoost Training Function ---
def train_and_evaluate_xgb(X_train_data, y_train_labels, X_test_data, y_test_labels,
                           feature_type_desc, target_class_names,
                           output_results_dir, perform_scaling=False): # Scaling often less critical for XGB
    print(f"\n--- Training XGBoost for {feature_type_desc} ---")

    if X_train_data is None or X_train_data.size == 0 or X_test_data is None or X_test_data.size == 0:
        print(f"Skipping XGBoost for {feature_type_desc}: Missing/empty feature data.")
        return None

    X_train_processed = X_train_data.copy()
    X_test_processed = X_test_data.copy()
    
    # Note: XGBoost handles multi-class by default. Labels should be 0 to N-1.
    # GPU usage: tree_method='hist', device='cuda' (for newer XGBoost) or device='gpu'
    # In mid-2025, 'cuda' is standard.
    try:
        base_estimator_xgb = xgb.XGBClassifier(objective='multi:softprob', # Output probabilities
                                            # objective='multi:softmax', # Output direct class predictions
                                            num_class=len(target_class_names), # Explicitly set for softmax
                                            tree_method='hist',      # Essential for GPU & large datasets
                                            device='cuda',           # Use GPU
                                            eval_metric='mlogloss',  # Logarithmic loss for multi-class
                                            random_state=42,
                                            use_label_encoder=False) # Suppress warning for newer XGBoost
    except xgb.core.XGBoostError as e:
        if "Cannot find CUDA device" in str(e) or "No GPU found" in str(e):
            print("XGBoost CUDA device not found. Falling back to CPU for XGBoost.")
            base_estimator_xgb = xgb.XGBClassifier(objective='multi:softprob',
                                                num_class=len(target_class_names),
                                                tree_method='hist', # hist is good for CPU too
                                                eval_metric='mlogloss',
                                                random_state=42,
                                                use_label_encoder=False)
        else:
            print(f"Error initializing XGBoost: {e}")
            return None


    # Scaling for XGBoost (Optional, usually not as impactful as for SVMs)
    if perform_scaling:
        print(f"Scaling features for XGBoost: {feature_type_desc}...")
        scaler_xgb = StandardScaler()
        X_train_processed = scaler_xgb.fit_transform(X_train_processed)
        X_test_processed = scaler_xgb.transform(X_test_processed)
        scaler_filename_xgb = os.path.join(output_results_dir, f'scaler_xgb_{feature_type_desc.replace(" ", "_")}.joblib')
        joblib.dump(scaler_xgb, scaler_filename_xgb)
        print(f"Saved XGBoost scaler for {feature_type_desc} to {scaler_filename_xgb}")

    # Parameters for XGBoost GridSearchCV - adjust as needed
    # This is a smaller grid for faster initial runs
    param_grid_xgb = {
    'n_estimators': [200, 500],        
    'learning_rate': [0.05, 0.1],     
    'max_depth': [5, 8],              
}

    print(f"Performing GridSearchCV for XGBoost on {feature_type_desc} (cv=3)...")
    # n_jobs for GridSearchCV with XGBoost on GPU:
    # - If XGBoost internally uses all GPU resources, n_jobs for GridSearchCV might be best set to 1.
    # - Or, if GridSearchCV parallelizes by running separate XGBoost instances, it might work.
    # - Test with n_jobs=1 first if GPU is the bottleneck.
    # - If you have many CPU cores and GPU is not fully utilized by one XGBoost, n_jobs=-1 might be fine.
    xgb_grid_search = GridSearchCV(estimator=base_estimator_xgb,
                                   param_grid=param_grid_xgb,
                                   scoring='accuracy',
                                   cv=3, verbose=2, n_jobs=1) # Start with n_jobs=1 for GPU
    
    xgb_grid_search.fit(X_train_processed, y_train_labels)

    best_xgb_model = xgb_grid_search.best_estimator_
    print(f"Best XGBoost parameters for {feature_type_desc}: {xgb_grid_search.best_params_}")

    model_filename_xgb = os.path.join(output_results_dir, f'xgb_model_{feature_type_desc.replace(" ", "_")}.joblib')
    joblib.dump(best_xgb_model, model_filename_xgb) # Save using joblib or XGBoost's own save_model
    # best_xgb_model.save_model(model_filename_xgb.replace(".joblib", ".json")) # XGBoost native format
    print(f"Saved best XGBoost model for {feature_type_desc} to {model_filename_xgb}")

    print(f"\n--- Evaluating XGBoost for {feature_type_desc} ---")
    y_pred_labels_xgb = best_xgb_model.predict(X_test_processed)
    # If objective was 'multi:softprob', predict_proba gives probabilities, predict gives class labels
    # y_pred_labels_xgb = np.argmax(best_xgb_model.predict_proba(X_test_processed), axis=1) # If needed

    accuracy_val_xgb = accuracy_score(y_test_labels, y_pred_labels_xgb)
    class_report_str_xgb = classification_report(y_test_labels, y_pred_labels_xgb, target_names=target_class_names, zero_division=0)
    conf_matrix_xgb = confusion_matrix(y_test_labels, y_pred_labels_xgb, labels=np.arange(len(target_class_names)))

    print(f"Accuracy (XGBoost - {feature_type_desc}): {accuracy_val_xgb:.4f}")
    print(f"Classification Report (XGBoost - {feature_type_desc}):\n{class_report_str_xgb}")
    print(f"Confusion Matrix (XGBoost - {feature_type_desc}):\n{conf_matrix_xgb}")
    plot_confusion_matrix(conf_matrix_xgb, classes=target_class_names,
                          plot_title=f'CM for XGBoost - {feature_type_desc} (Acc: {accuracy_val_xgb:.3f})',
                          results_path=output_results_dir,
                          filename=f'cm_xgb_{feature_type_desc.replace(" ", "_")}.png')

    results_text_file_xgb = os.path.join(output_results_dir, f'results_xgb_{feature_type_desc.replace(" ", "_")}.txt')
    with open(results_text_file_xgb, 'w') as f:
        f.write(f"--- XGBoost Results for {feature_type_desc} ---\n")
        f.write(f"Scaling: {perform_scaling}\nParams: {xgb_grid_search.best_params_}\nCV Score: {xgb_grid_search.best_score_:.4f}\nAccuracy: {accuracy_val_xgb:.4f}\n\nReport:\n{class_report_str_xgb}\n\nCM:\n{np.array2string(conf_matrix_xgb)}")
    print(f"Saved XGBoost results for {feature_type_desc} to {results_text_file_xgb}")
    return best_xgb_model

# --- 3. Feature Loading Functions (load_bovw_features, load_and_align_global_hog) ---
# These are the same as in your script. For brevity, I'll assume they are correctly defined above.
def load_bovw_features(bovw_dir, feature_name, set_type="train"):
    filename = f"X_{set_type}_{feature_name}_bovw.npy" # Assuming no k_value in filename
    # If your files are named X_train_sift_k1000_bovw.npy, use:
    # filename = f"X_{set_type}_{feature_name}_k{VOCAB_SIZE}_bovw.npy"
    filepath = os.path.join(bovw_dir, filename)
    if os.path.exists(filepath):
        print(f"Loading {set_type} {feature_name} BoVW features from: {filepath}")
        data = np.load(filepath)
        print(f"  Shape: {data.shape}")
        return data
    else:
        print(f"Warning: {feature_name} BoVW file not found: {filepath}")
        return None

def load_and_align_global_hog(hog_h5_filepath, target_indices_for_set):
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
        else:
             print("Error: Cannot safely reshape 1D HOG features for multiple indices.")
             return None
    if all_hog_features.shape[0] == 0 or all_hog_original_indices.shape[0] == 0:
        print("Warning: No HOG features or indices found in HDF5 file.")
        return None
    hog_feature_map = {original_idx: i for i, original_idx in enumerate(all_hog_original_indices)}
    aligned_hog_list = []
    missing_count = 0
    hog_feature_dim = all_hog_features.shape[1]
    for target_idx in target_indices_for_set:
        if target_idx in hog_feature_map:
            aligned_hog_list.append(all_hog_features[hog_feature_map[target_idx]])
        else:
            aligned_hog_list.append(np.zeros(hog_feature_dim, dtype=all_hog_features.dtype))
            missing_count += 1
    if missing_count > 0:
        print(f"  Warning: {missing_count}/{len(target_indices_for_set)} HOG features for current set not found. Used zero vectors.")
    aligned_hog_array = np.array(aligned_hog_list) if aligned_hog_list else np.empty((0, hog_feature_dim))
    print(f"  Aligned global HOG shape: {aligned_hog_array.shape}")
    return aligned_hog_array

# --- 4. Load Feature Sets ---
print("\n--- Loading Feature Sets ---")
X_train_sift_bovw = load_bovw_features(BOVW_FEATURES_DIR, "sift", "train")
X_test_sift_bovw = load_bovw_features(BOVW_FEATURES_DIR, "sift", "test")
X_train_orb_bovw = load_bovw_features(BOVW_FEATURES_DIR, "orb", "train")
X_test_orb_bovw = load_bovw_features(BOVW_FEATURES_DIR, "orb", "test")
X_train_hog_global = load_and_align_global_hog(HOG_DATA_FILE, train_indices)
X_test_hog_global = load_and_align_global_hog(HOG_DATA_FILE, test_indices)

# --- 5. Train and Evaluate Classifiers ---
# SVMs take too long. Will use XGBoost for GPU usage. You can battle building the thundersvm if you want but fuck that I gave up
# Set to True to run XGBoost, False to skip XGBoost
RUN_XGBOOST_CLASSIFIERS = True

if RUN_XGBOOST_CLASSIFIERS:
    print("\n--- Training and Evaluating XGBoost Classifiers ---")
    # XGBoost Individual Features (Scaling typically not needed, or can be set to False in function call)
    if X_train_sift_bovw is not None and X_test_sift_bovw is not None:
        train_and_evaluate_xgb(X_train_sift_bovw, y_train, X_test_sift_bovw, y_test,
                               f"SIFT_BoVW", class_names, RESULTS_DIR_XGB, perform_scaling=False)
    if X_train_orb_bovw is not None and X_test_orb_bovw is not None:
        train_and_evaluate_xgb(X_train_orb_bovw, y_train, X_test_orb_bovw, y_test,
                               f"ORB_BoVW", class_names, RESULTS_DIR_XGB, perform_scaling=False)
    if X_train_hog_global is not None and X_test_hog_global is not None:
        train_and_evaluate_xgb(X_train_hog_global, y_train, X_test_hog_global, y_test,
                               "HOG_Global", class_names, RESULTS_DIR_XGB, perform_scaling=False)
    # XGBoost Combined Features
    if X_train_sift_bovw is not None and X_train_hog_global is not None:
        if X_train_sift_bovw.shape[0] == X_train_hog_global.shape[0]:
            X_train_sift_hog = np.concatenate((X_train_sift_bovw, X_train_hog_global), axis=1)
            X_test_sift_hog = np.concatenate((X_test_sift_bovw, X_test_hog_global), axis=1)
            train_and_evaluate_xgb(X_train_sift_hog, y_train, X_test_sift_hog, y_test, f"SIFT_HOG_Global", class_names, RESULTS_DIR_XGB, perform_scaling=False)
    if X_train_orb_bovw is not None and X_train_hog_global is not None:
        if X_train_orb_bovw.shape[0] == X_train_hog_global.shape[0]:
            X_train_orb_hog = np.concatenate((X_train_orb_bovw, X_train_hog_global), axis=1)
            X_test_orb_hog = np.concatenate((X_test_orb_bovw, X_test_hog_global), axis=1)
            train_and_evaluate_xgb(X_train_orb_hog, y_train, X_test_orb_hog, y_test, f"ORB_HOG_Global", class_names, RESULTS_DIR_XGB, perform_scaling=False)
    if X_train_sift_bovw is not None and X_train_orb_bovw is not None and X_train_hog_global is not None:
         if X_train_sift_bovw.shape[0] == X_train_orb_bovw.shape[0] == X_train_hog_global.shape[0]:
            X_train_all_xgb = np.concatenate((X_train_sift_bovw, X_train_orb_bovw, X_train_hog_global), axis=1)
            X_test_all_xgb = np.concatenate((X_test_sift_bovw, X_test_orb_bovw, X_test_hog_global), axis=1)
            train_and_evaluate_xgb(X_train_all_xgb, y_train, X_test_all_xgb, y_test, f"SIFT_ORB_HOG_Global", class_names, RESULTS_DIR_XGB, perform_scaling=False)


print("\n--- Classification Pipeline Complete ---")