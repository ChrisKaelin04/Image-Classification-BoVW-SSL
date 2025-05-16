# build_vocabulary_vanilla_balanced.py
import os
import pickle
import numpy as np
import glob
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import joblib
import gc

# --- Configuration (can be arguments to the main function) ---
DEFAULT_VANILLA_FEATURES_RAW_DIR = r"E:\CV_BoVW_Vanilla_Balanced\raw_features"
DEFAULT_VOCABULARY_SIZE = 1000
DEFAULT_MINIBATCH_KMEANS_INTERNAL_BATCH_SIZE = 1024 * 4
DEFAULT_RANDOM_SEED = 42
DEFAULT_KMEANS_N_INIT = 1

def _build_single_feature_type_vocabulary(
    feature_type_to_process,
    vanilla_features_raw_dir,
    vocabulary_size,
    minibatch_kmeans_internal_batch_size,
    random_seed,
    kmeans_n_init
):
    """
    Internal helper function to build vocabulary for a single feature type.
    Returns True on success, False on failure.
    """
    print(f"\n--- Building Vocabulary for {feature_type_to_process.upper()} (Vanilla BoVW - Balanced Training Data) ---")

    descriptor_batches_subdir_name = f'{feature_type_to_process}_descriptors_batches'
    input_descriptor_batches_path = os.path.join(vanilla_features_raw_dir, descriptor_batches_subdir_name)

    output_base_filename_part = f"{feature_type_to_process}_vanilla_balanced_k{vocabulary_size}_train"
    output_vocab_file = os.path.join(vanilla_features_raw_dir, f"{output_base_filename_part}_vocabulary.pkl")
    output_kmeans_model_file = os.path.join(vanilla_features_raw_dir, f"{output_base_filename_part}_kmeans_model.joblib")

    print(f"Source: {input_descriptor_batches_path}")
    print(f"Vocab K: {vocabulary_size}")
    print(f"Output Vocab PKL: {output_vocab_file}")
    print(f"Output KMeans Joblib: {output_kmeans_model_file}")

    np.random.seed(random_seed)

    batch_files_pattern_regular = os.path.join(input_descriptor_batches_path, f'{feature_type_to_process}_descriptors_train_batch_*.pkl')
    train_batch_files = sorted(glob.glob(batch_files_pattern_regular))

    if not train_batch_files:
        print(f"Error: No TRAINING SET descriptor batch files found for {feature_type_to_process.upper()} in {input_descriptor_batches_path}")
        print(f"Searched for pattern: '{batch_files_pattern_regular}'")
        return False

    print(f"Found {len(train_batch_files)} TRAINING batch files for {feature_type_to_process.upper()}.")

    kmeans_model = MiniBatchKMeans(
        n_clusters=vocabulary_size,
        random_state=random_seed,
        batch_size=minibatch_kmeans_internal_batch_size,
        n_init=kmeans_n_init,
        max_iter=100,
        verbose=0, # Changed to 0 for cleaner loop output, tqdm handles progress
        compute_labels=False
    )

    total_descriptors_processed = 0
    print(f"Starting iterative K-Means training (partial_fit)...")
    for batch_file_path in tqdm(train_batch_files, desc=f"Fitting {feature_type_to_process.upper()}"):
        try:
            with open(batch_file_path, 'rb') as f:
                list_of_descriptor_arrays_in_batch = pickle.load(f)
            if not list_of_descriptor_arrays_in_batch: continue

            valid_descriptors = [d for (d, x) in list_of_descriptor_arrays_in_batch if d is not None and d.shape[0] > 0]
            if not valid_descriptors: continue
            
            current_batch_np = np.vstack(valid_descriptors)
            
            if current_batch_np.dtype != np.float32:
                current_batch_np = current_batch_np.astype(np.float32)
            
            if current_batch_np.size == 0: continue

            total_descriptors_processed += current_batch_np.shape[0]
            kmeans_model.partial_fit(current_batch_np)

            del list_of_descriptor_arrays_in_batch, current_batch_np, valid_descriptors; gc.collect()
        except Exception as e:
            tqdm.write(f"Warning processing batch {batch_file_path}: {e}. Skipping.")
            continue

    print(f"K-Means training complete. Processed {total_descriptors_processed} {feature_type_to_process.upper()} descriptors.")

    if hasattr(kmeans_model, 'cluster_centers_') and kmeans_model.cluster_centers_ is not None and \
       kmeans_model.cluster_centers_.shape[0] == vocabulary_size:
        vocabulary = kmeans_model.cluster_centers_
        print(f"Vocabulary shape: {vocabulary.shape}")
        os.makedirs(os.path.dirname(output_vocab_file), exist_ok=True)
        try:
            with open(output_vocab_file, 'wb') as f: pickle.dump(vocabulary, f)
            print(f"Saved vocabulary: {output_vocab_file}")
        except Exception as e: print(f"Error saving vocab PKL: {e}")
        try:
            joblib.dump(kmeans_model, output_kmeans_model_file)
            print(f"Saved KMeans model: {output_kmeans_model_file}")
        except Exception as e: print(f"Error saving KMeans Joblib: {e}")
        print(f"--- {feature_type_to_process.upper()} Vocabulary built successfully. ---")
        return True
    else:
        shape_info = kmeans_model.cluster_centers_.shape if hasattr(kmeans_model, 'cluster_centers_') and kmeans_model.cluster_centers_ is not None else 'None'
        print(f"Error: {feature_type_to_process.upper()} KMeans model cluster_centers_ issue. Expected {vocabulary_size}, got shape: {shape_info}.")
        return False

def build_all_vanilla_bovw_vocabularies(
    vanilla_features_raw_dir=DEFAULT_VANILLA_FEATURES_RAW_DIR,
    vocabulary_size=DEFAULT_VOCABULARY_SIZE,
    minibatch_kmeans_internal_batch_size=DEFAULT_MINIBATCH_KMEANS_INTERNAL_BATCH_SIZE,
    random_seed=DEFAULT_RANDOM_SEED,
    kmeans_n_init=DEFAULT_KMEANS_N_INIT,
    feature_types_to_process=['sift', 'orb']
):
    """
    Builds and saves vocabularies for specified feature types (e.g., SIFT, ORB)
    for the vanilla Bag of Visual Words model using balanced training data.

    Args:
        vanilla_features_raw_dir (str): Path to the directory containing descriptor batch subdirectories
                                       (e.g., 'sift_descriptors_batches', 'orb_descriptors_batches').
        vocabulary_size (int): The number of visual words (k) for KMeans.
        minibatch_kmeans_internal_batch_size (int): Internal batch size for MiniBatchKMeans.
        random_seed (int): Seed for reproducibility.
        kmeans_n_init (int): n_init parameter for MiniBatchKMeans (typically 1 for partial_fit).
        feature_types_to_process (list): List of strings, e.g., ['sift', 'orb'].
    """
    print("=" * 70)
    print("Starting All Vanilla BoVW Vocabulary Building (Balanced Data)")
    print(f"Reading raw descriptor batches from: {vanilla_features_raw_dir}")
    print("=" * 70)

    overall_success = True
    for ft_type in feature_types_to_process:
        success = _build_single_feature_type_vocabulary(
            feature_type_to_process=ft_type,
            vanilla_features_raw_dir=vanilla_features_raw_dir,
            vocabulary_size=vocabulary_size,
            minibatch_kmeans_internal_batch_size=minibatch_kmeans_internal_batch_size,
            random_seed=random_seed,
            kmeans_n_init=kmeans_n_init
        )
        if not success:
            print(f"!!! Vocabulary building FAILED for {ft_type.upper()} !!!")
            overall_success = False
        print("-" * 70)
    
    if overall_success:
        print("All specified Vanilla BoVW vocabularies built successfully.")
    else:
        print("One or more Vanilla BoVW vocabulary building processes failed. Please check logs.")
    print("=" * 70)
    return overall_success
