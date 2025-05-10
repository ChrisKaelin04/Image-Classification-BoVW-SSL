'''
Okie so we want to do spectral comparison of the images to see if this might work as basic classification. Its going to suck
'''
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import cv2
import os
import pickle
import h5py
from tqdm import tqdm
import warnings
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# --- Overall Configuration ---
TFDS_DATA_DIR = "E:\CV_imgs"
BASE_OUTPUT_DIR = "E:\CV_Pipeline_Spectral" # Main output for this spectral pipeline
SUBSET_SIZE = 100000 # Or a smaller number for faster testing, e.g., 10000
RANDOM_SEED = 42

# --- Spectral Feature Parameters ---
RESIZE_DIM = (128, 128)
FFT_REGION_SIZE = 32
FEATURE_VECTOR_LENGTH = FFT_REGION_SIZE * FFT_REGION_SIZE

# --- Output Paths ---
SPECTRAL_FEATURES_SUBDIR = os.path.join(BASE_OUTPUT_DIR, "spectral_features_data")
SPECTRAL_H5_FILE = os.path.join(SPECTRAL_FEATURES_SUBDIR, f"spectral_fft_{FFT_REGION_SIZE}x{FFT_REGION_SIZE}_subset{SUBSET_SIZE}.h5")

SPLITS_SUBDIR = os.path.join(BASE_OUTPUT_DIR, "train_test_splits_spectral_4cat")
SPECTRAL_NPZ_FILE = os.path.join(SPLITS_SUBDIR, "train_test_split_data_spectral_4cat.npz")
SPECTRAL_LABEL_ENCODER_FILE = os.path.join(SPLITS_SUBDIR, "broad_label_encoder_spectral_4cat.pkl")

RESULTS_DIR_XGB_SPECTRAL = os.path.join(BASE_OUTPUT_DIR, "classification_results_XGB_Spectral_4cat")

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(SPECTRAL_FEATURES_SUBDIR, exist_ok=True)
os.makedirs(SPLITS_SUBDIR, exist_ok=True)
os.makedirs(RESULTS_DIR_XGB_SPECTRAL, exist_ok=True)

warnings.filterwarnings("ignore", message="Parameters: {.*use_label_encoder.*} are not used.", category=UserWarning, module="xgboost.core")

# --- Broad Category Definitions (from your split_data.py) ---
broad_categories_list = [
    "Indoor Residential", "Indoor Public/Commercial",
    "Outdoor Natural", "Outdoor Urban"
]
broad_category_definitions = {
    # ... (PASTE YOUR FULL broad_category_definitions DICTIONARY HERE) ...
    'Indoor Public/Commercial': [   'airplane_cabin', 'airport_terminal', # ... and so on for all 365 fine classes
                                    # ... make sure this is complete ...
                                    'youth_hostel'],
    'Indoor Residential': [   'alcove', 'attic', # ... and so on ...
                              'garage/indoor'],
    'Outdoor Natural': [   'badlands', 'bamboo_forest', # ... and so on ...
                           'wave'],
    'Outdoor Urban': [   'airfield', 'alley', # ... and so on ...
                         'zen_garden']
}
if not all(len(v) > 0 for v in broad_category_definitions.values()):
    print("ERROR: broad_category_definitions is incomplete. Please paste the full dictionary.")
    exit()


# === PART 1: SPECTRAL FEATURE EXTRACTION ===
def extract_spectral_fft_features(img_np):
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    elif img_np.ndim == 2:
        gray = img_np
    else:
        return np.array([], dtype=np.float32)
    resized_gray = cv2.resize(gray, RESIZE_DIM, interpolation=cv2.INTER_AREA)
    f_transform = np.fft.fft2(resized_gray)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_transform_shifted)
    rows, cols = magnitude_spectrum.shape
    crow, ccol = rows // 2, cols // 2
    r_start, r_end = crow - FFT_REGION_SIZE // 2, crow + FFT_REGION_SIZE // 2
    c_start, c_end = ccol - FFT_REGION_SIZE // 2, ccol + FFT_REGION_SIZE // 2
    if not (0 <= r_start < r_end <= rows and 0 <= c_start < c_end <= cols):
        return np.array([], dtype=np.float32) # Invalid region
    central_region = magnitude_spectrum[r_start:r_end, c_start:c_end]
    feature_vector = central_region.flatten().astype(np.float32)
    if feature_vector.shape[0] != FEATURE_VECTOR_LENGTH:
        return np.array([], dtype=np.float32)
    return feature_vector

def extract_spectral_features_tf_element(index_tensor, img_tensor, label_tensor):
    try:
        img_np = img_tensor.numpy()
        label_np = label_tensor.numpy()
        idx_np = index_tensor.numpy()
        spectral_features = extract_spectral_fft_features(img_np)
        if spectral_features.size != FEATURE_VECTOR_LENGTH:
            spectral_features = np.zeros(FEATURE_VECTOR_LENGTH, dtype=np.float32)
            label_np = np.int64(-1) # Mark as problematic
        return idx_np, label_np, spectral_features
    except Exception:
        idx_np_err = index_tensor.numpy() if hasattr(index_tensor, 'numpy') else -1
        return idx_np_err, np.int64(-1), np.zeros(FEATURE_VECTOR_LENGTH, dtype=np.float32)

def run_spectral_feature_extraction_if_needed():
    if os.path.exists(SPECTRAL_H5_FILE):
        print(f"Spectral features file already exists: {SPECTRAL_H5_FILE}. Skipping extraction.")
        return

    print("--- Starting Spectral Feature Extraction (FFT-based) ---")
    print(f"Output file: {SPECTRAL_H5_FILE}")

    ds_train = tfds.load('places365_small', split='train', data_dir=TFDS_DATA_DIR, shuffle_files=True)
    ds_subset_indexed = ds_train.take(SUBSET_SIZE).enumerate()

    all_spectral_features, all_labels, all_indices = [], [], []
    tout_spectral = [tf.int64, tf.int64, tf.float32]

    ds_processed_spectral = ds_subset_indexed.map(
        lambda i, x: tf.py_function(func=extract_spectral_features_tf_element, inp=[i, x['image'], x['label']], Tout=tout_spectral),
        num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)

    successful_extractions = 0
    for idx, label, spectral_feat in tqdm(ds_processed_spectral.as_numpy_iterator(), total=SUBSET_SIZE, desc="Extracting Spectral Features"):
        if label != -1 and spectral_feat.shape[0] == FEATURE_VECTOR_LENGTH:
            all_spectral_features.append(spectral_feat)
            all_labels.append(label)
            all_indices.append(idx)
            successful_extractions += 1

    print(f"Successfully extracted spectral features for {successful_extractions}/{SUBSET_SIZE} images.")
    if not all_spectral_features:
        print("No spectral features were successfully extracted. Exiting.")
        exit()

    X_spectral = np.array(all_spectral_features, dtype=np.float32)
    y_original_labels = np.array(all_labels, dtype=np.int64)
    original_indices = np.array(all_indices, dtype=np.int64)

    with h5py.File(SPECTRAL_H5_FILE, 'w') as hf:
        hf.create_dataset('features', data=X_spectral)
        hf.create_dataset('labels', data=y_original_labels)
        hf.create_dataset('indices', data=original_indices)
    print(f"Saved spectral features to: {SPECTRAL_H5_FILE}")
    print("--- Spectral Feature Extraction Complete ---")

# === PART 2: DATA SPLITTING AND LABEL MAPPING (for Spectral Features) ===
def create_spectral_train_test_split_if_needed():
    if os.path.exists(SPECTRAL_NPZ_FILE) and os.path.exists(SPECTRAL_LABEL_ENCODER_FILE):
        print(f"Spectral train/test split files already exist ({SPECTRAL_NPZ_FILE}, {SPECTRAL_LABEL_ENCODER_FILE}). Skipping split creation.")
        return

    print("\n--- Creating Train/Test Split for Spectral Features ---")
    if not os.path.exists(SPECTRAL_H5_FILE):
        print(f"Error: Spectral features file {SPECTRAL_H5_FILE} not found. Run feature extraction first.")
        exit()

    print("Loading dataset info for fine-grained label names...")
    ds_info = tfds.load('places365_small', data_dir=TFDS_DATA_DIR, with_info=True, download=False)[1]
    fine_label_names = ds_info.features['label'].names

    category_mapping_fine_to_broad = {}
    for broad_cat, fine_list in broad_category_definitions.items():
        for fine_name in fine_list: category_mapping_fine_to_broad[fine_name] = broad_cat
    # Basic check (more robust checks were in your original split_data.py)
    if len(set(fine_label_names) - set(category_mapping_fine_to_broad.keys())) > 0:
        print("Warning: Some fine-grained labels from TFDS might be missing in your mapping.")


    print("Loading indices and original labels from spectral H5 file...")
    with h5py.File(SPECTRAL_H5_FILE, 'r') as hf:
        processed_image_indices = hf['indices'][:]
        original_fine_grained_numeric_labels = hf['labels'][:]

    mapped_broad_labels_str_list, valid_indices_for_split_list = [], []
    for i in range(len(original_fine_grained_numeric_labels)):
        fine_numeric_label = original_fine_grained_numeric_labels[i]
        current_image_original_tfds_idx = processed_image_indices[i]
        if 0 <= fine_numeric_label < len(fine_label_names):
            fine_label_name = fine_label_names[fine_numeric_label]
            if fine_label_name in category_mapping_fine_to_broad:
                broad_label_name = category_mapping_fine_to_broad[fine_label_name]
                mapped_broad_labels_str_list.append(broad_label_name)
                valid_indices_for_split_list.append(current_image_original_tfds_idx)

    if not mapped_broad_labels_str_list:
        print("Error: No labels mapped. Check mapping or H5 file. Exiting.")
        exit()

    label_encoder = LabelEncoder()
    label_encoder.fit(broad_categories_list) # Fit on the predefined list to ensure consistent encoding
    numeric_broad_labels = label_encoder.transform(mapped_broad_labels_str_list)

    train_indices, test_indices, \
    train_broad_labels_numeric, test_broad_labels_numeric = train_test_split(
        valid_indices_for_split_list, numeric_broad_labels,
        test_size=0.2, random_state=RANDOM_SEED, stratify=numeric_broad_labels)

    np.savez(SPECTRAL_NPZ_FILE,
             train_indices=np.array(train_indices), test_indices=np.array(test_indices),
             train_labels_numeric=train_broad_labels_numeric, test_labels_numeric=test_broad_labels_numeric)
    print(f"Saved spectral train/test indices and labels to: {SPECTRAL_NPZ_FILE}")
    with open(SPECTRAL_LABEL_ENCODER_FILE, 'wb') as f: pickle.dump(label_encoder, f)
    print(f"Saved spectral label encoder to: {SPECTRAL_LABEL_ENCODER_FILE}")
    print("--- Spectral Train/Test Split Creation Complete ---")


# === PART 3: XGBOOST CLASSIFICATION (using Spectral Features) ===
# plot_confusion_matrix and train_and_evaluate_xgb are the same as in your classifier_spm_xgb.py
# (Assuming they are defined above or imported)
# For brevity, I'll assume they are available. Ensure they are pasted here.
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
        # print(f"Saved confusion matrix to {full_path}") # Reduce verbosity
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
    try:
        base_estimator_xgb = xgb.XGBClassifier(objective='multi:softprob',
                                            num_class=len(target_class_names),
                                            tree_method='hist', device='cuda',
                                            eval_metric='mlogloss', random_state=RANDOM_SEED, # Use global RANDOM_SEED
                                            use_label_encoder=False)
    except xgb.core.XGBoostError as e:
        if "Cannot find CUDA device" in str(e) or "No GPU found" in str(e):
            print("XGBoost CUDA device not found. Falling back to CPU for XGBoost.")
            base_estimator_xgb = xgb.XGBClassifier(objective='multi:softprob',
                                                num_class=len(target_class_names),
                                                tree_method='hist', eval_metric='mlogloss',
                                                random_state=RANDOM_SEED, use_label_encoder=False)
        else: print(f"Error initializing XGBoost: {e}"); return None
    if perform_scaling:
        scaler_xgb = StandardScaler(); X_train_processed = scaler_xgb.fit_transform(X_train_processed)
        X_test_processed = scaler_xgb.transform(X_test_processed)
    param_grid_xgb = {'n_estimators': [100, 300], 'learning_rate': [0.05, 0.1], 'max_depth': [4, 6]} # Further reduced grid
    
    print(f"Performing GridSearchCV for XGBoost on {feature_type_desc} (cv=2)...") # cv=2 for speed
    xgb_grid_search = GridSearchCV(estimator=base_estimator_xgb, param_grid=param_grid_xgb,
                                   scoring='accuracy', cv=2, verbose=1, n_jobs=1) # cv=2, verbose=1
    
    xgb_grid_search.fit(X_train_processed, y_train_labels)
    best_xgb_model = xgb_grid_search.best_estimator_
    print(f"Best XGBoost parameters for {feature_type_desc}: {xgb_grid_search.best_params_}")
    model_filename_xgb = os.path.join(output_results_dir, f'xgb_model_{feature_type_desc.replace(" ", "_").replace("/", "-")}.joblib')
    joblib.dump(best_xgb_model, model_filename_xgb)
    y_pred_labels_xgb = best_xgb_model.predict(X_test_processed)
    accuracy_val_xgb = accuracy_score(y_test_labels, y_pred_labels_xgb)
    class_report_str_xgb = classification_report(y_test_labels, y_pred_labels_xgb, target_names=target_class_names, zero_division=0)
    conf_matrix_xgb = confusion_matrix(y_test_labels, y_pred_labels_xgb, labels=np.arange(len(target_class_names)))
    print(f"Accuracy (XGBoost - {feature_type_desc}): {accuracy_val_xgb:.4f}")
    # print(f"Classification Report (XGBoost - {feature_type_desc}):\n{class_report_str_xgb}") # Reduce verbosity
    # print(f"Confusion Matrix (XGBoost - {feature_type_desc}):\n{conf_matrix_xgb}")
    plot_confusion_matrix(conf_matrix_xgb, classes=target_class_names,
                          plot_title=f'CM for XGBoost - {feature_type_desc} (Acc: {accuracy_val_xgb:.3f})',
                          results_path=output_results_dir,
                          filename=f'cm_xgb_{feature_type_desc.replace(" ", "_").replace("/", "-")}.png')
    results_text_file_xgb = os.path.join(output_results_dir, f'results_xgb_{feature_type_desc.replace(" ", "_").replace("/", "-")}.txt')
    with open(results_text_file_xgb, 'w') as f:
        f.write(f"--- XGBoost Results for {feature_type_desc} ---\nParams: {xgb_grid_search.best_params_}\nCV Score: {xgb_grid_search.best_score_:.4f}\nAccuracy: {accuracy_val_xgb:.4f}\n\nReport:\n{class_report_str_xgb}\n\nCM:\n{np.array2string(conf_matrix_xgb)}")
    print(f"Saved XGBoost results for {feature_type_desc} to {results_text_file_xgb}")
    return best_xgb_model


def run_spectral_classification():
    print("\n--- Running XGBoost Classification on Spectral Features ---")

    # Load the split data specific to spectral features
    print(f"Loading spectral train/test split data from: {SPECTRAL_NPZ_FILE}")
    try:
        spectral_split_data = np.load(SPECTRAL_NPZ_FILE)
        train_indices_spectral = spectral_split_data['train_indices']
        test_indices_spectral = spectral_split_data['test_indices']
        y_train_spectral = spectral_split_data['train_labels_numeric']
        y_test_spectral = spectral_split_data['test_labels_numeric']
    except FileNotFoundError:
        print(f"ERROR: Spectral NPZ file not found at {SPECTRAL_NPZ_FILE}. Run split creation first.")
        return
    except KeyError as e:
        print(f"ERROR: Missing key {e} in spectral NPZ file {SPECTRAL_NPZ_FILE}.")
        return

    # Load the label encoder
    try:
        with open(SPECTRAL_LABEL_ENCODER_FILE, 'rb') as f:
            label_encoder_spectral = pickle.load(f)
        class_names_spectral = label_encoder_spectral.classes_
    except FileNotFoundError:
        print(f"ERROR: Spectral Label encoder file not found at {SPECTRAL_LABEL_ENCODER_FILE}.")
        return

    # Load all extracted spectral features and their original indices from the H5 file
    if not os.path.exists(SPECTRAL_H5_FILE):
        print(f"ERROR: Spectral features H5 file {SPECTRAL_H5_FILE} not found.")
        return
    
    print(f"Loading all spectral features from {SPECTRAL_H5_FILE} for alignment...")
    with h5py.File(SPECTRAL_H5_FILE, 'r') as hf:
        all_X_spectral = hf['features'][:]
        all_original_indices_spectral = hf['indices'][:]

    # Create a map for quick lookup: original_tfds_index -> row_in_all_X_spectral
    feature_map_spectral = {idx: i for i, idx in enumerate(all_original_indices_spectral)}

    # Align X_train_spectral
    X_train_spectral_list = []
    for idx in train_indices_spectral:
        if idx in feature_map_spectral:
            X_train_spectral_list.append(all_X_spectral[feature_map_spectral[idx]])
        else:
            # This should not happen if split was made from the H5 file's indices
            print(f"Warning: Index {idx} from train_indices_spectral not found in H5 features. Appending zeros.")
            X_train_spectral_list.append(np.zeros(FEATURE_VECTOR_LENGTH, dtype=np.float32))
    X_train_spectral = np.array(X_train_spectral_list)

    # Align X_test_spectral
    X_test_spectral_list = []
    for idx in test_indices_spectral:
        if idx in feature_map_spectral:
            X_test_spectral_list.append(all_X_spectral[feature_map_spectral[idx]])
        else:
            print(f"Warning: Index {idx} from test_indices_spectral not found in H5 features. Appending zeros.")
            X_test_spectral_list.append(np.zeros(FEATURE_VECTOR_LENGTH, dtype=np.float32))
    X_test_spectral = np.array(X_test_spectral_list)

    print(f"Aligned X_train_spectral shape: {X_train_spectral.shape}, y_train_spectral shape: {y_train_spectral.shape}")
    print(f"Aligned X_test_spectral shape: {X_test_spectral.shape}, y_test_spectral shape: {y_test_spectral.shape}")

    if X_train_spectral.shape[0] != y_train_spectral.shape[0] or X_test_spectral.shape[0] != y_test_spectral.shape[0]:
        print("Error: Mismatch after aligning spectral features with labels. Halting classification.")
        return

    # Train XGBoost on spectral features
    if X_train_spectral.size > 0 and X_test_spectral.size > 0:
        train_and_evaluate_xgb(X_train_spectral, y_train_spectral, X_test_spectral, y_test_spectral,
                               f"Spectral_FFT_{FFT_REGION_SIZE}x{FFT_REGION_SIZE}",
                               class_names_spectral, RESULTS_DIR_XGB_SPECTRAL)
    else:
        print("Skipping XGBoost training for spectral features as data is empty after alignment.")

    print("--- Spectral Classification Complete ---")


if __name__ == '__main__':
    # Step 1: Extract spectral features (if they don't exist)
    run_spectral_feature_extraction_if_needed()

    # Step 2: Create train/test splits for these spectral features (if they don't exist)
    create_spectral_train_test_split_if_needed()

    # Step 3: Run classification on the spectral features
    run_spectral_classification()

    print("\n=== Full Spectral Pipeline Finished ===")