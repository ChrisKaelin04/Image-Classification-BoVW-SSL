# This file is the third part of the BoVWs pipeline. We already extracted the features and created the vocabulary. Now we will create the histograms for each image using the vocabulary we just created.
import numpy as np
import os
import glob
import pickle
import joblib # For loading KMeans models and for Parallel
from tqdm import tqdm
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed

FEATURES_DIR = "E:\CV_features"
SPLITS_DIR = os.path.join(FEATURES_DIR, "train_test_splits_4cat_revised")
NPZ_FILE = os.path.join(SPLITS_DIR, "train_test_split_data_4cat_revised.npz")
LABEL_ENCODER_FILE = os.path.join(SPLITS_DIR, "broad_label_encoder_4cat_revised.pkl")

VOCAB_SIZE = 1000 # Must match our K for Kmeans

BOVW_FEATURES_DIR = os.path.join(FEATURES_DIR, "bovw_features_4cat")
os.makedirs(BOVW_FEATURES_DIR, exist_ok=True)

def generate_bovw_histogram(image_descriptors, kmeans_model, vocab_size):
    if image_descriptors is None or image_descriptors.shape[0] == 0:
        return np.zeros(vocab_size, dtype=np.float32)
    
    if image_descriptors.dtype == np.uint8: # ORB descriptors are uint8
        image_descriptors = image_descriptors.astype(np.float32)
    
    visual_words = kmeans_model.predict(image_descriptors)
    histogram = np.bincount(visual_words, minlength=vocab_size).astype(np.float32)
    
    if np.sum(histogram) > 0:
        histogram = normalize(histogram.reshape(1, -1), norm='l2')[0]
    
    return histogram

# Helper function for parallel processing of a single image's BoVW
def _generate_single_bovw_for_parallel(image_idx, processed_indices_map, kmeans_model, vocab_size):
    """
    Generates BoVW for a single image_idx.
    This function will be called by each parallel worker.
    It handles its own batch loading for the given image_idx.
    """
    descriptors = None
    if image_idx in processed_indices_map:
        target_batch_file = processed_indices_map[image_idx]
        try:
            with open(target_batch_file, 'rb') as f:
                batch_descriptors_data = pickle.load(f)
            descriptors = batch_descriptors_data.get(image_idx)
        except Exception as e:
            print(f"Error loading batch file {target_batch_file} for index {image_idx} in worker: {e}")
            # Fall through, descriptors will be None, and a zero histogram will be returned

    hist = generate_bovw_histogram(descriptors, kmeans_model, vocab_size)
    return hist

def process_indices_bovw_parallel(indices, descriptor_batches_dir, feature_type, kmeans_model, vocab_size, desc="Processing Images", n_jobs=-1):
    processed_indices_map = {} # This map is built once and passed to workers
    batch_files = sorted(glob.glob(os.path.join(FEATURES_DIR, descriptor_batches_dir, f'{feature_type}_batch_*.pkl')))
    
    if not batch_files:
        print(f"Error: No batch files found for {feature_type} in {os.path.join(FEATURES_DIR, descriptor_batches_dir)}")
        return np.array([])
    
    print(f"Mapping indices from {len(batch_files)} batch files for {feature_type}...")
    # Build the map of image_idx to its batch file path
    for batch_file_path in tqdm(batch_files, desc=f"Scanning {feature_type} batches for mapping"):
        try:
            with open(batch_file_path, 'rb') as f:
                batch_data = pickle.load(f)
            for idx_in_batch in batch_data.keys():
                processed_indices_map[idx_in_batch] = batch_file_path
        except Exception as e:
            print(f"Warning: Could not load or process {batch_file_path} during mapping: {e}")
            continue
    print(f"Mapped {len(processed_indices_map)} unique descriptor sets for {feature_type}.")

    if not processed_indices_map:
        print(f"No descriptors found for {feature_type}. Returning empty array.")
        return np.array([])
        
    print(f"\nGenerating BoVW histograms for {len(indices)} images ({feature_type}) using {n_jobs if n_jobs != -1 else os.cpu_count()} workers...")
    
    # Use joblib.Parallel to process images
    # `prefer="threads"` might be good if I/O is the bottleneck and GIL is released by pickle/numpy
    # `prefer="processes"` (default) is better if CPU-bound (like predict) and avoids GIL issues
    # For predict, processes is generally better.
    histograms_list = Parallel(n_jobs=n_jobs)(
        delayed(_generate_single_bovw_for_parallel)(
            image_idx, processed_indices_map, kmeans_model, vocab_size
        ) for image_idx in tqdm(indices, desc=desc)
    )
    
    return np.array(histograms_list)

def histogram_creation():
    print("--- Starting BoVW Histogram Generation ---")

    print(f"Loading train/test split data from: {NPZ_FILE}")
    split_data = np.load(NPZ_FILE)
    train_indices = split_data['train_indices']
    test_indices = split_data['test_indices']

    print(f"Loaded {len(train_indices)} training and {len(test_indices)} testing indices.")

    N_JOBS = os.cpu_count() - 4

    print("\n--- Processing SIFT Features ---")
    sift_kmeans_model_file = os.path.join(FEATURES_DIR, 'sift_kmeans_model_k1000_partial_fit.joblib')
    sift_batches_subdir = 'sift_batches'

    if os.path.exists(sift_kmeans_model_file):
        print(f"Loading SIFT KMeans model from: {sift_kmeans_model_file}")
        sift_kmeans = joblib.load(sift_kmeans_model_file)

        X_train_sift_bovw = process_indices_bovw_parallel(
            train_indices, sift_batches_subdir, 'sift', sift_kmeans, VOCAB_SIZE, desc="SIFT Train BoVW", n_jobs=N_JOBS
        )
        print(f"SIFT Training BoVW histograms shape: {X_train_sift_bovw.shape}")
        if X_train_sift_bovw.size > 0:
            np.save(os.path.join(BOVW_FEATURES_DIR, 'X_train_sift_bovw.npy'), X_train_sift_bovw)
            print(f"Saved SIFT training BoVW features to {BOVW_FEATURES_DIR}")

        X_test_sift_bovw = process_indices_bovw_parallel(
            test_indices, sift_batches_subdir, 'sift', sift_kmeans, VOCAB_SIZE, desc="SIFT Test BoVW", n_jobs=N_JOBS
        )
        print(f"SIFT Test BoVW histograms shape: {X_test_sift_bovw.shape}")
        if X_test_sift_bovw.size > 0:
            np.save(os.path.join(BOVW_FEATURES_DIR, 'X_test_sift_bovw.npy'), X_test_sift_bovw)
            print(f"Saved SIFT test BoVW features to {BOVW_FEATURES_DIR}")
    else:
        print(f"SIFT KMeans model not found at {sift_kmeans_model_file}. Skipping SIFT BoVW generation.")

    print("\n--- Processing ORB Features ---")
    orb_kmeans_model_file = os.path.join(FEATURES_DIR, 'orb_kmeans_model_k1000_partial_fit.joblib')
    orb_batches_subdir = 'orb_batches'

    if os.path.exists(orb_kmeans_model_file):
        print(f"Loading ORB KMeans model from: {orb_kmeans_model_file}")
        orb_kmeans = joblib.load(orb_kmeans_model_file)

        X_train_orb_bovw = process_indices_bovw_parallel(
            train_indices, orb_batches_subdir, 'orb', orb_kmeans, VOCAB_SIZE, desc="ORB Train BoVW", n_jobs=N_JOBS
        )
        print(f"ORB Training BoVW histograms shape: {X_train_orb_bovw.shape}")
        if X_train_orb_bovw.size > 0:
            np.save(os.path.join(BOVW_FEATURES_DIR, 'X_train_orb_bovw.npy'), X_train_orb_bovw)
            print(f"Saved ORB training BoVW features to {BOVW_FEATURES_DIR}")

        X_test_orb_bovw = process_indices_bovw_parallel(
            test_indices, orb_batches_subdir, 'orb', orb_kmeans, VOCAB_SIZE, desc="ORB Test BoVW", n_jobs=N_JOBS
        )
        print(f"ORB Test BoVW histograms shape: {X_test_orb_bovw.shape}")
        if X_test_orb_bovw.size > 0:
            np.save(os.path.join(BOVW_FEATURES_DIR, 'X_test_orb_bovw.npy'), X_test_orb_bovw)
            print(f"Saved ORB test BoVW features to {BOVW_FEATURES_DIR}")
    else:
        print(f"ORB KMeans model not found at {orb_kmeans_model_file}. Skipping ORB BoVW generation.")

    print("\n--- Phase 2: BoVW Histogram Generation Complete ---")
    print(f"BoVW features saved in: {BOVW_FEATURES_DIR}")