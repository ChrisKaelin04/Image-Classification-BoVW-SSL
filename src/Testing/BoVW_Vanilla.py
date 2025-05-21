import cv2
import numpy as np
import os
import pickle
import xgboost as xgb
from sklearn.neighbors import NearestNeighbors # For efficient nearest neighbor search

# --- Configuration (ADJUST THESE PATHS) ---
# Path to a sample image you want to classify
SAMPLE_IMAGE_PATH = r"E:\CV_BoVW_Balanced_Raw_Images\Indoor Public\Commercial\idx1_office.jpg" # <--- IMPORTANT: Change this

# Paths to your trained Visual Word Codebooks (from your BoVW training)
SIFT_CODEBOOK_PATH = r"E:\CV_BoVW_Vanilla_Balanced\raw_features\sift_vanilla_balanced_k1000_train_vocabulary.pkl" # <--- IMPORTANT: Change this
ORB_CODEBOOK_PATH = r"E:\CV_BoVW_Vanilla_Balanced\raw_features\orb_vanilla_balanced_k1000_train_vocabulary.pkl"   # <--- IMPORTANT: Change this

# Paths to your trained XGBoost models
SIFT_XGB_MODEL_PATH = r"E:\CV_BoVW_Vanilla_Balanced\classification_results_XGB_SOH_balanced\xgb_model_Vanilla_SIFT_k1000.json" # <--- IMPORTANT: Change this
ORB_XGB_MODEL_PATH = r"E:\CV_BoVW_Vanilla_Balanced\classification_results_XGB_SOH_balanced\xgb_model_Vanilla_ORB_k1000.json"  # <--- IMPORTANT: Change this


# --- Helper Function (Modified from your SOH_extract_vanilla_balanced.py) ---
def extract_raw_features(image_path):
    """
    Extracts SIFT descriptors and ORB descriptors from an image file.
    Returns (descriptors_sift, descriptors_orb)
    """
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Warning: Could not read image at {image_path}. Skipping.")
            return None, None

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        orb = cv2.ORB_create(nfeatures=1000) # Ensure nfeatures matches what you used during training

        # SIFT descriptors
        _, descriptors_sift = sift.detectAndCompute(gray, None)
        if descriptors_sift is None: descriptors_sift = np.array([], dtype=np.float32).reshape(0, 128)

        # ORB descriptors
        _, descriptors_orb = orb.detectAndCompute(gray, None)
        if descriptors_orb is None: descriptors_orb = np.array([], dtype=np.uint8).reshape(0, 32)
        
        return descriptors_sift, descriptors_orb

    except Exception as e:
        print(f"Error processing image {image_path} for raw features: {e}")
        return None, None

def quantize_descriptors_to_histogram(descriptors, codebook):
    """
    Quantizes local descriptors to a Bag of Visual Words histogram using a codebook.
    Args:
        descriptors (np.array): N x D array of local descriptors (e.g., SIFT or ORB).
        codebook (np.array): K x D array of visual words (cluster centers).
    Returns:
        np.array: 1D array representing the BoVW histogram.
    """
    if descriptors.shape[0] == 0:
        return np.zeros(codebook.shape[0], dtype=np.float32) # Return empty histogram if no descriptors

    # Use NearestNeighbors for efficient assignment to visual words
    neigh = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', n_jobs=-1)
    neigh.fit(codebook)

    distances, indices = neigh.kneighbors(descriptors)
    
    # Create the histogram
    histogram = np.zeros(codebook.shape[0], dtype=np.float32)
    for i in indices.flatten():
        histogram[i] += 1

    # Normalize the histogram (e.g., L1 normalization)
    histogram /= histogram.sum() + 1e-6 # Add small epsilon to avoid division by zero

    return histogram

def load_pkl_codebook(filepath):
    """Loads a visual word codebook from a .pkl file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Codebook file not found: {filepath}")
    with open(filepath, 'rb') as f:
        codebook = pickle.load(f)
    print(f"Loaded codebook from: {filepath} with shape {codebook.shape}")
    return codebook

def load_xgb_model(filepath):
    """Loads an XGBoost model from a .json file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"XGBoost model file not found: {filepath}")
    model = xgb.Booster()
    model.load_model(filepath)
    print(f"Loaded XGBoost model from: {filepath}")
    return model

# --- Main Inference Script ---
print("--- Starting Single Image BoVW Classification ---")

# 1. Load trained models and codebooks
try:
    sift_codebook = load_pkl_codebook(SIFT_CODEBOOK_PATH)
    orb_codebook = load_pkl_codebook(ORB_CODEBOOK_PATH)
    
    sift_xgb_model = load_xgb_model(SIFT_XGB_MODEL_PATH)
    orb_xgb_model = load_xgb_model(ORB_XGB_MODEL_PATH)
except FileNotFoundError as e:
    print(f"Error loading required files: {e}")
    print("Please ensure all path configurations at the top of the script are correct.")
    exit()

# 2. Extract raw features from the single image
print(f"\nProcessing image: {SAMPLE_IMAGE_PATH}")
sift_descriptors, orb_descriptors = extract_raw_features(SAMPLE_IMAGE_PATH)

if sift_descriptors is None: # Image reading/processing failed
    print(f"Failed to extract features from {SAMPLE_IMAGE_PATH}. Exiting.")
    exit()

# 3. Generate BoVW histograms for SIFT and ORB
print("Generating BoVW histograms...")
sift_bovw_histogram = quantize_descriptors_to_histogram(sift_descriptors, sift_codebook)
orb_bovw_histogram = quantize_descriptors_to_histogram(orb_descriptors, orb_codebook)

print(f"  SIFT BoVW Histogram shape: {sift_bovw_histogram.shape}")
print(f"  ORB BoVW Histogram shape: {orb_bovw_histogram.shape}")

# 4. Make predictions using the trained XGBoost models
print("\nMaking predictions...")

# For XGBoost, the input needs to be a DMatrix (or a numpy array if it's a single sample)
# Ensure it's reshaped to (1, feature_dim) for a single sample prediction

# SIFT Prediction
sift_dmatrix = xgb.DMatrix(sift_bovw_histogram.reshape(1, -1))
sift_prediction_raw = sift_xgb_model.predict(sift_dmatrix)
print(f"  SIFT BoVW Prediction Raw Output: {sift_prediction_raw}")
sift_predicted_class = np.argmax(sift_prediction_raw)
sift_confidence = np.max(sift_prediction_raw) if sift_prediction_raw.size > 1 else sift_prediction_raw[0]

# ORB Prediction
orb_dmatrix = xgb.DMatrix(orb_bovw_histogram.reshape(1, -1))
orb_prediction_raw = orb_xgb_model.predict(orb_dmatrix)
print(f"  ORB BoVW Prediction Raw Output: {orb_prediction_raw}")
orb_predicted_class = np.argmax(orb_prediction_raw)
orb_confidence = np.max(orb_prediction_raw) if orb_prediction_raw.size > 1 else orb_prediction_raw[0]

# 5. Display Results
print("\n--- Classification Results ---")
print(f"Image: {os.path.basename(SAMPLE_IMAGE_PATH)}")
print(f"  SIFT BoVW: Predicted Class = {sift_predicted_class}, Confidence/Score = {sift_confidence:.4f}")
print(f"  ORB BoVW:  Predicted Class = {orb_predicted_class}, Confidence/Score = {orb_confidence:.4f}")

print("\n--- Single Image Classification Complete ---")