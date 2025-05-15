# build_vocabulary_spm_refactored.py
import os
import pickle
import numpy as np
import glob
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import joblib
import gc # Added for explicit garbage collection

# --- Configuration for SPM Vocabulary ---
FEATURES_SPM_DIR = "E:\CV_features_SPM" # Main directory for SPM features

# --- K-Means Parameters (should be consistent) ---
VOCABULARY_SIZE = 1000  # (k) Number of visual words
MINIBATCH_SIZE = 1024 * 4 # Internal batch size for MiniBatchKMeans
RANDOM_SEED = 42

# --- K-Means Training Parameters ---
KMEANS_N_INIT = 10 # IMPORTANT: Number of times MiniBatchKMeans is run with different centroid seeds.
                   # Increase for potentially better clustering results, but increases computation time.
                   # Must be > 0. For partial_fit, n_init=1 is often used if initial centroids are fixed,
                   # but for random seeds, >1 is better practice for quality.


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
    print(f"MiniBatchKMeans n_init: {KMEANS_N_INIT}")


    np.random.seed(RANDOM_SEED) # Seed for MiniBatchKMeans reproducibility *across runs*

    # Use glob to find batch files, ensure sorted order for consistent processing
    batch_files_spm = sorted(glob.glob(os.path.join(input_batches_path, f'{feature_type_to_process}_spm_batch_*.pkl')) +
                             glob.glob(os.path.join(input_batches_path, f'{feature_type_to_process}_spm_final_batch_processed*.pkl'))) # Include final batch

    if not batch_files_spm:
        print(f"Error: No SPM batch files found for {feature_type_to_process.upper()} in {input_batches_path}")
        return False

    print(f"Found {len(batch_files_spm)} SPM batch files to process for {feature_type_to_process.upper()}.")

    print(f"Initializing MiniBatchKMeans model with k={VOCABULARY_SIZE}...")
    # NOTE: With partial_fit, n_init>1 runs n_init separate initializations.
    # Subsequent calls to partial_fit continue training the best initialization found so far.
    kmeans_model = MiniBatchKMeans(n_clusters=VOCABULARY_SIZE,
                                   random_state=RANDOM_SEED, # Seed applies to centroid initialization
                                   batch_size=MINIBATCH_SIZE,
                                   n_init=KMEANS_N_INIT,     # Changed from 1 to KMEANS_N_INIT
                                   max_iter=100, # Iterations within each partial_fit call
                                   verbose=1, # Set to 0 for less output during training
                                   compute_labels=False) # Don't compute labels in partial_fit for efficiency


    print(f"Starting iterative training using partial_fit over {len(batch_files_spm)} SPM batches for {feature_type_to_process.upper()}...")
    total_descriptors_processed = 0

    for i, batch_file in enumerate(tqdm(batch_files_spm, desc=f"Processing {feature_type_to_process.upper()} SPM batches")):
        try:
            with open(batch_file, 'rb') as f:
                batch_data_spm = pickle.load(f)

            current_batch_descriptors = []
            # Iterate through values (the dictionaries per image) in the batch data
            for image_info_dict in batch_data_spm.values():
                descriptors = image_info_dict.get('descriptors')
                if descriptors is not None and descriptors.shape[0] > 0:
                    current_batch_descriptors.append(descriptors)

            if not current_batch_descriptors:
                print(f"\nWarning: No descriptors found in batch file: {batch_file}. Skipping.")
                continue

            # Stack descriptors from all images in the current batch into a single numpy array
            # NOTE: This np.vstack can cause MemoryError if a single batch is too large.
            # If OOM occurs here, reduce BATCH_SAVE_SIZE in SOH_extract_SPM and re-extract features.
            batch_np = None
            try:
                # Handle dtype conversion before stacking
                if feature_type_to_process == 'orb':
                    # Stack as uint8 first, then convert to float32 for KMeans
                    stacked_descriptors = np.vstack(current_batch_descriptors)
                    batch_np = stacked_descriptors.astype(np.float32)
                else: # Assuming SIFT or other float descriptors (should be float32 from extraction)
                    stacked_descriptors = np.vstack(current_batch_descriptors)
                    if stacked_descriptors.dtype != np.float32:
                         print(f"\nWarning: Expected {feature_type_to_process.upper()} descriptors to be float32, found {stacked_descriptors.dtype} in {batch_file}. Converting.")
                         batch_np = stacked_descriptors.astype(np.float32)
                    else:
                         batch_np = stacked_descriptors

            except MemoryError:
                 print(f"\nCritical Error: Ran out of memory while stacking descriptors from batch: {batch_file}.")
                 print("Try reducing BATCH_SAVE_SIZE in the feature extraction script.")
                 return False # Critical error, stop for this feature type
            except Exception as e:
                 print(f"\nWarning: An error occurred during stacking/dtype conversion for batch {batch_file}: {e}. Skipping batch.")
                 # print(traceback.format_exc()) # Uncomment for detailed error
                 continue # Skip this batch but continue with others

            if batch_np is None or batch_np.size == 0:
                 print(f"\nWarning: Stacked batch data is empty or None after processing {batch_file}. Skipping.")
                 continue


            total_descriptors_processed += batch_np.shape[0]
            # Perform partial_fit on the stacked descriptors from the current batch
            kmeans_model.partial_fit(batch_np)


            # Explicit cleanup after processing each batch
            del batch_data_spm, current_batch_descriptors, batch_np
            if 'stacked_descriptors' in locals(): del stacked_descriptors
            gc.collect() # Request garbage collection

        except FileNotFoundError:
            print(f"\nWarning: SPM Batch file not found: {batch_file}. Skipping.")
        except pickle.UnpicklingError:
            print(f"\nWarning: Could not unpickle SPM file: {batch_file}. Skipping.")
        except Exception as e:
            print(f"\nWarning: An unexpected error occurred processing batch {batch_file}: {e}. Skipping batch.")
            # print(traceback.format_exc()) # Uncomment for detailed error


    print(f"\nK-Means partial_fit training for {feature_type_to_process.upper()} (SPM) complete. Processed {total_descriptors_processed} descriptors in total.")

    # Check if clustering was successful
    if hasattr(kmeans_model, 'cluster_centers_') and kmeans_model.cluster_centers_.shape[0] == VOCABULARY_SIZE:
        vocabulary = kmeans_model.cluster_centers_
        print(f"SPM Vocabulary shape for {feature_type_to_process.upper()}: {vocabulary.shape}")

        # Save the vocabulary (cluster centers)
        print(f"Saving {feature_type_to_process.upper()} SPM vocabulary to: {output_vocab_file}")
        try:
            with open(output_vocab_file, 'wb') as f: pickle.dump(vocabulary, f)
            print(f"Successfully saved {feature_type_to_process.upper()} SPM vocabulary.")
        except Exception as e: print(f"Error saving {feature_type_to_process.upper()} SPM vocabulary file: {e}")

        # Save the full KMeans model object
        print(f"Saving {feature_type_to_process.upper()} SPM KMeans model object to: {output_kmeans_model_file}")
        try:
            joblib.dump(kmeans_model, output_kmeans_model_file)
            print(f"Successfully saved {feature_type_to_process.upper()} SPM KMeans model to {output_kmeans_model_file}")
        except Exception as e: print(f"Error saving {feature_type_to_process.upper()} SPM KMeans model object: {e}")

        print(f"--- {feature_type_to_process.upper()} SPM Vocabulary creation finished successfully! ---")
        return True
    else:
        print(f"Error: {feature_type_to_process.upper()} SPM KMeans model does not have correctly formed cluster_centers_ after training.")
        print(f"Expected {VOCABULARY_SIZE}, got {kmeans_model.cluster_centers_.shape[0] if hasattr(kmeans_model, 'cluster_centers_') else 'None'}")
        print("Vocabulary creation failed.")
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