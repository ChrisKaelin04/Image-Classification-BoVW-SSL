# build_vocabulary_spm_balanced.py
import os
import pickle
import numpy as np
import glob
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import joblib
import gc

# --- Configuration for SPM Vocabulary (from BALANCED features) ---
# This should be the output directory of SOH_extract_SPM_balanced.py
FEATURES_SPM_BALANCED_DIR = r"E:\CV_features_SPM_balanced"

# --- K-Means Parameters ---
VOCABULARY_SIZE = 1000  # (k) Number of visual words
MINIBATCH_KMEANS_BATCH_SIZE = 1024 * 4 # Internal batch size for MiniBatchKMeans updates
RANDOM_SEED = 42
KMEANS_N_INIT = 10 # Number of initializations for MiniBatchKMeans

def build_single_vocab_for_feature_type_balanced(feature_type_to_process):
    """
    Builds the K-Means vocabulary for a single specified feature type (e.g., 'sift' or 'orb')
    using BALANCED SPM-structured batch files FROM THE TRAINING SET ONLY.
    """
    print(f"\n--- Starting K-Means Vocabulary Creation for {feature_type_to_process.upper()} (Balanced SPM - TRAINING SET ONLY) ---")

    batches_subdir = f'{feature_type_to_process}_batches_spm' # e.g., 'sift_batches_spm'
    # IMPORTANT: Only use TRAINING set batches for vocabulary building
    input_batches_path = os.path.join(FEATURES_SPM_BALANCED_DIR, batches_subdir)

    # Output files will reflect that they are from balanced data and training set
    output_vocab_file = os.path.join(FEATURES_SPM_BALANCED_DIR, f'{feature_type_to_process}_vocabulary_spm_balanced_k{VOCABULARY_SIZE}_train.pkl')
    output_kmeans_model_file = os.path.join(FEATURES_SPM_BALANCED_DIR, f'{feature_type_to_process}_kmeans_model_spm_balanced_k{VOCABULARY_SIZE}_train.joblib')

    print(f"Loading TRAINING descriptors iteratively from: {input_batches_path}")
    print(f"Target vocabulary size (k): {VOCABULARY_SIZE}")
    print(f"MiniBatchKMeans n_init: {KMEANS_N_INIT}")

    np.random.seed(RANDOM_SEED)

    # Glob pattern to find ONLY TRAINING set batch files
    # Names from SOH_extract_SPM_balanced.py:
    # e.g., sift_spm_train_batch_0.pkl, sift_spm_train_final_batch_processedXXXX.pkl
    batch_files_pattern_regular = os.path.join(input_batches_path, f'{feature_type_to_process}_spm_train_batch_*.pkl')
    batch_files_pattern_final = os.path.join(input_batches_path, f'{feature_type_to_process}_spm_train_final_batch_processed*.pkl')
    
    train_batch_files_spm = sorted(glob.glob(batch_files_pattern_regular) + glob.glob(batch_files_pattern_final))

    if not train_batch_files_spm:
        print(f"Error: No TRAINING SET SPM batch files found for {feature_type_to_process.upper()} in {input_batches_path}")
        print(f"Searched for patterns: '{batch_files_pattern_regular}' and '{batch_files_pattern_final}'")
        return False

    print(f"Found {len(train_batch_files_spm)} TRAINING SET SPM batch files to process for {feature_type_to_process.upper()}.")

    print(f"Initializing MiniBatchKMeans model with k={VOCABULARY_SIZE}...")
    kmeans_model = MiniBatchKMeans(
        n_clusters=VOCABULARY_SIZE,
        random_state=RANDOM_SEED,
        batch_size=MINIBATCH_KMEANS_BATCH_SIZE,
        n_init=KMEANS_N_INIT,
        max_iter=100, # Iterations within each partial_fit call
        verbose=1,
        compute_labels=False # More efficient for partial_fit if labels aren't needed during training
    )

    print(f"Starting iterative training using partial_fit over {len(train_batch_files_spm)} TRAINING {feature_type_to_process.upper()} SPM batches...")
    total_descriptors_processed = 0

    for i, batch_file in enumerate(tqdm(train_batch_files_spm, desc=f"Processing TRAINING {feature_type_to_process.upper()} SPM batches")):
        try:
            with open(batch_file, 'rb') as f:
                # batch_data_spm is a dict: {image_path: {'descriptors': ..., 'coordinates': ..., ...}}
                batch_data_spm = pickle.load(f)

            current_batch_descriptors_list = []
            for image_path, image_info_dict in batch_data_spm.items(): # Iterate items to get path if needed for debug
                descriptors = image_info_dict.get('descriptors')
                if descriptors is not None and descriptors.shape[0] > 0:
                    current_batch_descriptors_list.append(descriptors)

            if not current_batch_descriptors_list:
                tqdm.write(f"Warning: No descriptors found in training batch file: {batch_file}. Skipping.")
                continue

            batch_np_descriptors = None
            try:
                if feature_type_to_process == 'orb': # ORB descriptors are uint8
                    stacked_descriptors = np.vstack(current_batch_descriptors_list)
                    batch_np_descriptors = stacked_descriptors.astype(np.float32) # KMeans expects float
                else: # SIFT descriptors should already be float32
                    stacked_descriptors = np.vstack(current_batch_descriptors_list)
                    if stacked_descriptors.dtype != np.float32:
                        tqdm.write(f"Warning: Expected {feature_type_to_process.upper()} descriptors to be float32, "
                                   f"found {stacked_descriptors.dtype} in {batch_file}. Converting.")
                        batch_np_descriptors = stacked_descriptors.astype(np.float32)
                    else:
                        batch_np_descriptors = stacked_descriptors
            except MemoryError:
                tqdm.write(f"CRITICAL Error: Ran out of memory while stacking descriptors from batch: {batch_file}.")
                tqdm.write("Try reducing BATCH_SAVE_SIZE in SOH_extract_SPM_balanced.py and re-extract features.")
                return False
            except Exception as e:
                tqdm.write(f"Warning: An error occurred during stacking/dtype conversion for batch {batch_file}: {e}. Skipping batch.")
                continue

            if batch_np_descriptors is None or batch_np_descriptors.size == 0:
                tqdm.write(f"Warning: Stacked batch data is empty or None after processing {batch_file}. Skipping.")
                continue

            total_descriptors_processed += batch_np_descriptors.shape[0]
            kmeans_model.partial_fit(batch_np_descriptors)

            del batch_data_spm, current_batch_descriptors_list, batch_np_descriptors
            if 'stacked_descriptors' in locals(): del stacked_descriptors
            gc.collect()

        except FileNotFoundError:
            tqdm.write(f"Warning: SPM Batch file not found: {batch_file}. Skipping.")
        except pickle.UnpicklingError:
            tqdm.write(f"Warning: Could not unpickle SPM file: {batch_file}. Skipping.")
        except Exception as e:
            tqdm.write(f"Warning: An unexpected error occurred processing batch {batch_file}: {e}. Skipping batch.")

    print(f"\nK-Means partial_fit training for {feature_type_to_process.upper()} (Balanced SPM - Training Set) complete.")
    print(f"Processed {total_descriptors_processed} descriptors in total.")

    if hasattr(kmeans_model, 'cluster_centers_') and kmeans_model.cluster_centers_ is not None and \
       kmeans_model.cluster_centers_.shape[0] == VOCABULARY_SIZE:
        vocabulary = kmeans_model.cluster_centers_
        print(f"SPM Vocabulary shape for {feature_type_to_process.upper()}: {vocabulary.shape}")

        os.makedirs(os.path.dirname(output_vocab_file), exist_ok=True) # Ensure dir exists
        print(f"Saving {feature_type_to_process.upper()} SPM vocabulary to: {output_vocab_file}")
        try:
            with open(output_vocab_file, 'wb') as f: pickle.dump(vocabulary, f)
        except Exception as e: print(f"Error saving {feature_type_to_process.upper()} SPM vocabulary file: {e}")

        print(f"Saving {feature_type_to_process.upper()} SPM KMeans model object to: {output_kmeans_model_file}")
        try:
            joblib.dump(kmeans_model, output_kmeans_model_file)
        except Exception as e: print(f"Error saving {feature_type_to_process.upper()} SPM KMeans model object: {e}")

        print(f"--- {feature_type_to_process.upper()} SPM Vocabulary (Balanced - Training Set) creation finished successfully! ---")
        return True
    else:
        print(f"Error: {feature_type_to_process.upper()} SPM KMeans model does not have correctly formed cluster_centers_ after training.")
        cluster_shape_info = kmeans_model.cluster_centers_.shape if hasattr(kmeans_model, 'cluster_centers_') and kmeans_model.cluster_centers_ is not None else 'None or not initialized'
        print(f"Expected {VOCABULARY_SIZE} clusters, got shape: {cluster_shape_info}")
        print("Vocabulary creation failed.")
        return False

def build_all_spm_vocabularies_balanced():
    """
    Main function to build SPM vocabularies for all specified feature types using balanced training data.
    """
    print("--- Building All SPM Vocabularies from Balanced Training Data ---")
    feature_types_to_build = ['sift', 'orb'] # List of feature types you want to process

    for ft_type in feature_types_to_build:
        success = build_single_vocab_for_feature_type_balanced(ft_type)
        if not success:
            print(f"IMPORTANT: Vocabulary building failed for {ft_type.upper()} (Balanced SPM). Please check errors.")
        print("-" * 50)

    print("\nAll specified SPM vocabulary building attempts (Balanced) finished.")
