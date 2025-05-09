# build_vocabulary_spm_refactored.py
import os
import pickle
import numpy as np
import glob
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import joblib

# --- Configuration for SPM Vocabulary ---
FEATURES_SPM_DIR = "E:\CV_features_SPM" # Main directory for SPM features

# --- K-Means Parameters (should be consistent) ---
VOCABULARY_SIZE = 1000  # (k) Number of visual words
MINIBATCH_SIZE = 1024 * 4 # Internal batch size for MiniBatchKMeans
RANDOM_SEED = 42

def build_single_vocab_for_feature_type(feature_type_to_process):
    """
    Builds the K-Means vocabulary for a single specified feature type (e.g., 'sift' or 'orb')
    using SPM-structured batch files.
    """
    print(f"\n--- Starting K-Means Vocabulary Creation for {feature_type_to_process.upper()} (SPM Batches) ---")

    # Derive paths based on feature_type_to_process
    batches_subdir = f'{feature_type_to_process}_batches_spm'
    input_batches_path = os.path.join(FEATURES_SPM_DIR, batches_subdir)
    output_vocab_file = os.path.join(FEATURES_SPM_DIR, f'{feature_type_to_process}_vocabulary_spm_k{VOCABULARY_SIZE}_partial_fit.pkl')
    output_kmeans_model_file = os.path.join(FEATURES_SPM_DIR, f'{feature_type_to_process}_kmeans_model_spm_k{VOCABULARY_SIZE}_partial_fit.joblib')

    print(f"Loading descriptors iteratively from: {input_batches_path}")
    print(f"Target vocabulary size (k): {VOCABULARY_SIZE}")

    np.random.seed(RANDOM_SEED) # Seed for MiniBatchKMeans reproducibility

    batch_files_spm = sorted(glob.glob(os.path.join(input_batches_path, f'{feature_type_to_process}_spm_batch_*.pkl')))

    if not batch_files_spm:
        print(f"Error: No SPM batch files found for {feature_type_to_process.upper()} in {input_batches_path}")
        return False

    print(f"Found {len(batch_files_spm)} SPM batch files to process for {feature_type_to_process.upper()}.")

    print(f"Initializing MiniBatchKMeans model with k={VOCABULARY_SIZE}...")
    kmeans_model = MiniBatchKMeans(n_clusters=VOCABULARY_SIZE,
                                   random_state=RANDOM_SEED,
                                   batch_size=MINIBATCH_SIZE,
                                   n_init=1,
                                   max_iter=100,
                                   verbose=1, # Set to 0 for less output during training
                                   compute_labels=False)

    print(f"Starting iterative training using partial_fit over {len(batch_files_spm)} SPM batches for {feature_type_to_process.upper()}...")
    total_descriptors_processed = 0

    for batch_file in tqdm(batch_files_spm, desc=f"Processing {feature_type_to_process.upper()} SPM batches"):
        try:
            with open(batch_file, 'rb') as f:
                batch_data_spm = pickle.load(f)

            current_batch_descriptors = []
            for image_info_dict in batch_data_spm.values():
                descriptors = image_info_dict.get('descriptors')
                if descriptors is not None and descriptors.shape[0] > 0:
                    current_batch_descriptors.append(descriptors)

            if not current_batch_descriptors:
                continue

            # Handle dtype conversion
            if feature_type_to_process == 'orb':
                batch_np = np.vstack(current_batch_descriptors).astype(np.float32)
            else: # Assuming SIFT or other float descriptors
                batch_np = np.vstack(current_batch_descriptors)
                if batch_np.dtype != np.float32:
                    batch_np = batch_np.astype(np.float32)

            total_descriptors_processed += batch_np.shape[0]
            kmeans_model.partial_fit(batch_np)

            del batch_data_spm, current_batch_descriptors, batch_np # Explicit cleanup

        except FileNotFoundError:
            print(f"\nWarning: SPM Batch file not found: {batch_file}")
        except pickle.UnpicklingError:
            print(f"\nWarning: Could not unpickle SPM file: {batch_file}")
        except MemoryError:
            print(f"\nError: Ran out of memory while processing SPM batch: {batch_file}.")
            return False # Critical error, stop for this feature type
        except Exception as e:
            print(f"\nWarning: An error occurred processing SPM batch {batch_file}: {e}")

    print(f"\nK-Means partial_fit training for {feature_type_to_process.upper()} (SPM) complete. Processed {total_descriptors_processed} descriptors.")

    if hasattr(kmeans_model, 'cluster_centers_') and kmeans_model.cluster_centers_.shape[0] == VOCABULARY_SIZE:
        vocabulary = kmeans_model.cluster_centers_
        print(f"SPM Vocabulary shape for {feature_type_to_process.upper()}: {vocabulary.shape}")

        print(f"Saving {feature_type_to_process.upper()} SPM vocabulary to: {output_vocab_file}")
        try:
            with open(output_vocab_file, 'wb') as f: pickle.dump(vocabulary, f)
        except Exception as e: print(f"Error saving {feature_type_to_process.upper()} SPM vocabulary file: {e}")

        print(f"Saving {feature_type_to_process.upper()} SPM KMeans model object to: {output_kmeans_model_file}")
        try:
            joblib.dump(kmeans_model, output_kmeans_model_file)
            print(f"Successfully saved {feature_type_to_process.upper()} SPM KMeans model to {output_kmeans_model_file}")
        except Exception as e: print(f"Error saving {feature_type_to_process.upper()} SPM KMeans model object: {e}")

        print(f"--- {feature_type_to_process.upper()} SPM Vocabulary creation finished successfully! ---")
        return True
    else:
        print(f"Error: {feature_type_to_process.upper()} SPM KMeans model does not have correctly formed cluster_centers_. Expected {VOCABULARY_SIZE}, got {kmeans_model.cluster_centers_.shape[0] if hasattr(kmeans_model, 'cluster_centers_') else 'None'}")
        return False


def build_all_spm_vocabularies():
    """
    Main function to build SPM vocabularies for all specified feature types.
    """
    feature_types_to_build = ['sift', 'orb'] # List of feature types you want to process

    for ft_type in feature_types_to_build:
        success = build_single_vocab_for_feature_type(ft_type)
        if not success:
            print(f"IMPORTANT: Vocabulary building failed for {ft_type.upper()} (SPM). Please check errors.")
        print("-" * 50) # Separator

    print("\nAll specified SPM vocabulary building attempts finished.")