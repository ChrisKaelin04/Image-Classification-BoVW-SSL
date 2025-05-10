import numpy as np
import os
import pickle
import warnings
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler # Optional for XGBoost
import matplotlib.pyplot as plt
import seaborn as sns
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
    # ... (This function can remain largely the same as you had it) ...
    # ... (Make sure RANDOM_SEED is defined if used by XGBoost, e.g., RANDOM_SEED = 42 at top) ...
    print(f"\n--- Training XGBoost for {feature_type_desc} ---")
    if X_train_data is None or X_train_data.size == 0 or X_test_data is None or X_test_data.size == 0:
        print(f"Skipping XGBoost for {feature_type_desc}: Missing/empty feature data.")
        return None
    X_train_processed = X_train_data.copy()
    X_test_processed = X_test_data.copy()
    try:
        base_estimator_xgb = xgb.XGBClassifier(objective='multi:softprob',
                                            num_class=len(target_class_names),
                                            tree_method='hist', device='cuda',
                                            eval_metric='mlogloss', random_state=42, # Or your global RANDOM_SEED
                                            use_label_encoder=False)
    except xgb.core.XGBoostError as e:
        if "Cannot find CUDA device" in str(e) or "No GPU found" in str(e):
            print("XGBoost CUDA device not found. Falling back to CPU for XGBoost.")
            base_estimator_xgb = xgb.XGBClassifier(objective='multi:softprob',
                                                num_class=len(target_class_names),
                                                tree_method='hist', eval_metric='mlogloss',
                                                random_state=42, use_label_encoder=False) # Or your global RANDOM_SEED
        else:
            print(f"Error initializing XGBoost: {e}")
            return None
    if perform_scaling:
        scaler_xgb = StandardScaler(); X_train_processed = scaler_xgb.fit_transform(X_train_processed)
        X_test_processed = scaler_xgb.transform(X_test_processed)
    
    # Keep your reduced param_grid_xgb or adjust as needed
    param_grid_xgb = {
        'n_estimators': [300],
        'learning_rate': [0.05, 0.1],
        'max_depth': [7],
    }
    print(f"Performing GridSearchCV for XGBoost on {feature_type_desc} (cv=3)...")
    xgb_grid_search = GridSearchCV(estimator=base_estimator_xgb, param_grid=param_grid_xgb,
                                   scoring='accuracy', cv=3, verbose=2, n_jobs=1)
    xgb_grid_search.fit(X_train_processed, y_train_labels)
    best_xgb_model = xgb_grid_search.best_estimator_
    print(f"Best XGBoost parameters for {feature_type_desc}: {xgb_grid_search.best_params_}")
    model_filename_xgb = os.path.join(output_results_dir, f'xgb_model_{feature_type_desc.replace(" ", "_").replace("/", "-")}.joblib')
    joblib.dump(best_xgb_model, model_filename_xgb)
    print(f"Saved best XGBoost model for {feature_type_desc} to {model_filename_xgb}")
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
    results_text_file_xgb = os.path.join(output_results_dir, f'results_xgb_{feature_type_desc.replace(" ", "_").replace("/", "-")}.txt')
    with open(results_text_file_xgb, 'w') as f:
        f.write(f"--- XGBoost Results for {feature_type_desc} ---\n")
        f.write(f"Scaling: {perform_scaling}\nParams: {xgb_grid_search.best_params_}\nCV Score: {xgb_grid_search.best_score_:.4f}\nAccuracy: {accuracy_val_xgb:.4f}\n\nReport:\n{class_report_str_xgb}\n\nCM:\n{np.array2string(conf_matrix_xgb)}")
    print(f"Saved XGBoost results for {feature_type_desc} to {results_text_file_xgb}")
    return best_xgb_model

# --- 3. Feature Loading Functions ---
def load_cnn_features(cnn_features_dir, cnn_model_name_lower, set_type="train"):
    # Filename e.g., X_train_resnet50v2_features.npy
    filename = f"X_{set_type}_{cnn_model_name_lower}_features.npy"
    filepath = os.path.join(cnn_features_dir, filename)
    if os.path.exists(filepath):
        print(f"Loading {set_type} {cnn_model_name_lower} CNN features from: {filepath}")
        data = np.load(filepath)
        print(f"  Shape: {data.shape}")
        return data
    else:
        print(f"Warning: {cnn_model_name_lower} CNN features file not found: {filepath}")
        return None



# --- Main Execution Function ---
def run_cnn_xgb_classification():
    # --- Configuration ---
    # CNN Extracted Features (Update these paths)
    CNN_FEATURES_BASE_DIR = "E:\CV_Features_CNN" # Your main CNN features directory
    CNN_MODEL_NAME = "ResNet50V2"  # Or "MobileNetV2", "EfficientNetB0", etc. - MUST MATCH FOLDER/FILE NAMES
    CNN_EXTRACTED_FEATURES_DIR = os.path.join(CNN_FEATURES_BASE_DIR, "cnn_extracted_features", CNN_MODEL_NAME)

    # HOG Data File (if you want to fuse with HOG)
    FEATURES_DIR_VANILLA_FOR_HOG = "E:\CV_features" # Only if HOG is from vanilla features
    HOG_DATA_FILE = os.path.join(FEATURES_DIR_VANILLA_FOR_HOG, 'hog_data.h5')

    # Shared Splits and Label Info
    SPLITS_DIR_COMMON = os.path.join("E:\CV_features", "train_test_splits_4cat_revised")
    NPZ_FILE = os.path.join(SPLITS_DIR_COMMON, "train_test_split_data_4cat_revised.npz")
    LABEL_ENCODER_FILE = os.path.join(SPLITS_DIR_COMMON, "broad_label_encoder_4cat_revised.pkl")

    # Results Directory for CNN + XGBoost
    RESULTS_DIR_XGB_CNN = os.path.join(CNN_FEATURES_BASE_DIR, "classification_results_XGB_CNN_4cat", CNN_MODEL_NAME)
    os.makedirs(RESULTS_DIR_XGB_CNN, exist_ok=True)

    warnings.filterwarnings("ignore", message="Parameters: {.*use_label_encoder.*} are not used.", category=UserWarning, module="xgboost.core")
    # --- 1. Load Labels, Indices, and Label Encoder ---
    print("--- Loading Common Data (Labels, Splits, Encoder) ---")
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
    print(f"\n--- Starting CNN ({CNN_MODEL_NAME}) + XGBoost Classification Pipeline ---")

    # Load CNN Extracted Features
    print(f"\n--- Loading {CNN_MODEL_NAME} Extracted Features ---")
    X_train_cnn = load_cnn_features(CNN_EXTRACTED_FEATURES_DIR, CNN_MODEL_NAME.lower(), "train")
    X_test_cnn = load_cnn_features(CNN_EXTRACTED_FEATURES_DIR, CNN_MODEL_NAME.lower(), "test")

    # --- Train and Evaluate XGBoost Classifiers ---
    print(f"\n\n" + "="*20 + f" XGBoost for {CNN_MODEL_NAME} Features " + "="*20)

    if X_train_cnn is not None and X_test_cnn is not None:
        train_and_evaluate_xgb(X_train_cnn, y_train, X_test_cnn, y_test,
                               f"{CNN_MODEL_NAME}_Features", class_names, RESULTS_DIR_XGB_CNN)
    else:
        print(f"Skipping {CNN_MODEL_NAME} features alone due to missing data.")


    print(f"\n--- CNN ({CNN_MODEL_NAME}) + XGBoost Classification Pipeline Complete ---")
    print(f"Results saved in: {RESULTS_DIR_XGB_CNN}")