# histogram_creation_vanilla_balanced.py
import numpy as np
import os
import glob
import pickle
import joblib
from tqdm import tqdm
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed
import gc

# --- Configuration for Vanilla BoVW Histograms from BALANCED Data ---

# Input: Directory where SOH_extract_vanilla_balanced.py saved SIFT/ORB descriptor batches
# This is the parent directory of 'sift_descriptors_batches' and 'orb_descriptors_batches'
VANILLA_FEATURES_RAW_DIR = r"E:\CV\features_BoVW_Vanilla_balanced"

# Directory where build_vocabulary_vanilla_balanced.py saved KMeans models
# (Often the same as VANILLA_FEATURES_RAW_DIR)
VOCAB_MODELS_DIR = VANILLA_FEATURES_RAW_DIR

VOCAB_SIZE = 1000  # Must match K used for vocabulary building

# Output directory for the final Vanilla BoVW histograms and their labels
HISTOGRAMS_OUTPUT_DIR = os.path.join(VANILLA_FEATURES_RAW_DIR, f"bovw_histograms_k{VOCAB_SIZE}")
os.makedirs(HISTOGRAMS_OUTPUT_DIR, exist_ok=True)

N_JOBS_PARALLEL = os.cpu_count() - 2 if os.cpu_count() > 2 else 1 # For parallel processing


def generate_bovw_histogram(image_descriptors_single_image, kmeans_model, vocab_size):
    """
    Generates a BoVW histogram for a single image's descriptors.
    """
    if image_descriptors_single_image is None or image_descriptors_single_image.shape[0] == 0:
        return np.zeros(vocab_size, dtype=np.float32)
    
    # Ensure descriptors are float32 for KMeans prediction
    descriptors_float = image_descriptors_single_image.astype(np.float32) if image_descriptors_single_image.dtype != np.float32 else image_descriptors_single_image
    
    try:
        visual_words = kmeans_model.predict(descriptors_float)
        histogram = np.bincount(visual_words, minlength=vocab_size).astype(np.float32)
    except Exception as e:
        # This might happen if a descriptor array is malformed or has unexpected dimensions
        # For safety, return a zero histogram. Log the error.
        # tqdm.write(f"Error during kmeans.predict or bincount: {e}. Desc shape: {descriptors_float.shape}. Returning zeros.")
        return np.zeros(vocab_size, dtype=np.float32)

    # L2 Normalize the histogram (common practice for BoVW)
    if np.sum(histogram) > 0:
        histogram = normalize(histogram.reshape(1, -1), norm='l2')[0]
    
    return histogram

def _process_descriptor_label_tuple_for_hist(descriptor_label_tuple, kmeans_model, vocab_size):
    """Helper for joblib.Parallel. Processes one (descriptors, label) tuple."""
    descriptors_for_image, label_for_image = descriptor_label_tuple
    hist = generate_bovw_histogram(descriptors_for_image, kmeans_model, vocab_size)
    return hist, label_for_image


def create_histograms_for_set_and_feature_type(
    feature_type,
    set_name, # 'train' or 'test'
    kmeans_model_loaded,
    vocab_size_param
):
    """
    Loads descriptor batches for a given feature type and set,
    generates BoVW histograms in parallel, and returns them along with labels.
    """
    print(f"\n--- Creating Vanilla BoVW Histograms for {feature_type.upper()} - {set_name.upper()} SET ---")

    descriptor_batches_subdir_name = f'{feature_type}_descriptors_batches'
    input_descriptor_batches_path = os.path.join(VANILLA_FEATURES_RAW_DIR, descriptor_batches_subdir_name)

    # Glob pattern for the descriptor batch files for this feature_type and set_name
    batch_files_pattern = os.path.join(input_descriptor_batches_path, f'{feature_type}_descriptors_{set_name}_batch_*.pkl')
    descriptor_batch_files = sorted(glob.glob(batch_files_pattern))

    if not descriptor_batch_files:
        print(f"Error: No {set_name.upper()} descriptor batch files found for {feature_type.upper()} in {input_descriptor_batches_path}")
        print(f"Searched for pattern: '{batch_files_pattern}'")
        return None, None # Return None for both histograms and labels

    print(f"Found {len(descriptor_batch_files)} {set_name.upper()} descriptor batch files for {feature_type.upper()}.")

    # Collect all (descriptors_for_one_image, label_for_that_image) tuples from all batch files
    all_descriptor_label_tuples_for_set = []
    for batch_file_path in tqdm(descriptor_batch_files, desc=f"Loading {set_name} {feature_type} descriptor batches"):
        try:
            with open(batch_file_path, 'rb') as f:
                # Each PKL file now contains a LIST of (descriptor_array, label) tuples
                list_of_desc_label_tuples = pickle.load(f)
                all_descriptor_label_tuples_for_set.extend(list_of_desc_label_tuples)
        except Exception as e:
            tqdm.write(f"Warning: Could not load or process batch file {batch_file_path}: {e}. Skipping.")
            continue
    
    if not all_descriptor_label_tuples_for_set:
        print(f"No descriptor-label tuples loaded from batches for {feature_type.upper()} {set_name.upper()} set.")
        return None, None

    print(f"Collected {len(all_descriptor_label_tuples_for_set)} descriptor-label items for {feature_type.upper()} {set_name.upper()} set.")
    
    # Generate histograms in parallel
    # Input to delayed: (descriptor_label_tuple, kmeans_model, vocab_size)
    # Output from parallel call: list of (histogram, label) tuples
    results_hist_label_list = Parallel(n_jobs=N_JOBS_PARALLEL)(
        delayed(_process_descriptor_label_tuple_for_hist)(
            desc_label_tuple, kmeans_model_loaded, vocab_size_param
        ) for desc_label_tuple in tqdm(all_descriptor_label_tuples_for_set, desc=f"Building {set_name} {feature_type} BoVW hists")
    )
    
    if not results_hist_label_list:
        print(f"No histograms generated for {feature_type.upper()} {set_name.upper()} set.")
        return None, None

    # Separate histograms and labels
    # The order is preserved from all_descriptor_label_tuples_for_set by joblib.Parallel by default
    histograms_array = np.array([item[0] for item in results_hist_label_list if item is not None])
    labels_array = np.array([item[1] for item in results_hist_label_list if item is not None], dtype=np.int8)

    if histograms_array.size == 0 or labels_array.size == 0 or histograms_array.shape[0] != labels_array.shape[0]:
        print(f"Error: Mismatch or empty results after parallel processing for {feature_type.upper()} {set_name.upper()} set.")
        return None, None
        
    print(f"Generated BoVW histograms for {feature_type.upper()} {set_name.upper()} set. Shape: {histograms_array.shape}")
    print(f"Corresponding labels shape: {labels_array.shape}")
    
    return histograms_array, labels_array


def main_histogram_creation_vanilla_balanced():
    print("--- Starting Vanilla BoVW Histogram Generation (from Balanced Data) ---")

    feature_types_to_process = ['sift', 'orb']
    data_sets_to_process = ['train', 'test']

    for ft_type in feature_types_to_process:
        # Load the corresponding KMeans model (trained only on training data)
        kmeans_model_filename = f'{ft_type}_vanilla_balanced_k{VOCAB_SIZE}_train_kmeans_model.joblib'
        kmeans_model_path = os.path.join(VOCAB_MODELS_DIR, kmeans_model_filename)

        if not os.path.exists(kmeans_model_path):
            print(f"ERROR: KMeans model for {ft_type.upper()} not found at {kmeans_model_path}.")
            print(f"Please run 'build_vocabulary_vanilla_balanced.py' first for {ft_type}.")
            continue # Skip this feature type
        
        print(f"\nLoading {ft_type.upper()} KMeans model from: {kmeans_model_path}")
        try:
            kmeans_model = joblib.load(kmeans_model_path)
        except Exception as e:
            print(f"ERROR loading {ft_type.upper()} KMeans model: {e}. Skipping this feature type.")
            continue

        for set_name in data_sets_to_process:
            X_bovw_histograms, y_bovw_labels = create_histograms_for_set_and_feature_type(
                feature_type=ft_type,
                set_name=set_name,
                kmeans_model_loaded=kmeans_model,
                vocab_size_param=VOCAB_SIZE
            )

            if X_bovw_histograms is not None and y_bovw_labels is not None:
                # Save the histograms and labels
                hist_output_filename = f"X_{set_name}_{ft_type}_vanilla_k{VOCAB_SIZE}.npy"
                labels_output_filename = f"y_{set_name}_{ft_type}_vanilla_labels_k{VOCAB_SIZE}.npy" # Changed to match SPM

                np.save(os.path.join(HISTOGRAMS_OUTPUT_DIR, hist_output_filename), X_bovw_histograms)
                np.save(os.path.join(HISTOGRAMS_OUTPUT_DIR, labels_output_filename), y_bovw_labels)
                
                print(f"Saved {set_name} {ft_type.upper()} Vanilla BoVW histograms to: {hist_output_filename}")
                print(f"Saved {set_name} {ft_type.upper()} Vanilla BoVW labels to: {labels_output_filename}")
            else:
                print(f"Failed to generate histograms or labels for {ft_type.upper()} {set_name.upper()} set.")
            
            del X_bovw_histograms, y_bovw_labels # Explicit cleanup
            gc.collect()
        del kmeans_model # Cleanup model after processing train/test for this feature type
        gc.collect()

    print("\n--- All Vanilla BoVW Histogram Generation (Balanced Data) Complete ---")
    print(f"Final BoVW histograms and labels saved in: {HISTOGRAMS_OUTPUT_DIR}")
