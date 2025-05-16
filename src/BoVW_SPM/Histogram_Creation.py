# histogram_creation_SPM_balanced.py
import numpy as np
import os
import glob
import pickle
import joblib
from tqdm import tqdm
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed
import gc

# --- Configuration for SPM Histograms from BALANCED Features ---
# Directory where SOH_extract_SPM_from_balanced_split.py saved its output
FEATURES_SPM_BALANCED_DIR = r"E:\CV_features_SPM_balanced"

# Directory where build_vocabulary_spm_balanced.py saved KMeans models/vocabularies
# (This is often the same as FEATURES_SPM_BALANCED_DIR)
VOCAB_DIR_SPM_BALANCED = FEATURES_SPM_BALANCED_DIR

VOCAB_SIZE = 1000  # Must match K used for vocabulary building
PYRAMID_LEVELS = 3 # L=2 gives levels 0 and 1 (1x1, 2x2 grids) -> 1+4=5 regions.

# Output directory for the final SPM histograms
SPM_HISTOGRAMS_OUTPUT_DIR = os.path.join(FEATURES_SPM_BALANCED_DIR, f"spm_histograms_L{PYRAMID_LEVELS-1}_k{VOCAB_SIZE}")
os.makedirs(SPM_HISTOGRAMS_OUTPUT_DIR, exist_ok=True)


def generate_spm_histogram_for_image_data(image_data_dict, kmeans_model, vocab_size, num_pyramid_levels):
    """
    Generates an SPM histogram for a single image's data.
    image_data_dict: dict containing {'descriptors', 'coordinates', 'width', 'height', 'label'}
    """
    total_regions_in_pyramid = sum([(2**l)**2 for l in range(num_pyramid_levels)])
    expected_hist_shape = total_regions_in_pyramid * vocab_size

    if image_data_dict is None:
        return np.zeros(expected_hist_shape, dtype=np.float32)

    descriptors = image_data_dict.get('descriptors')
    coordinates = image_data_dict.get('coordinates')
    img_width = image_data_dict.get('width')
    img_height = image_data_dict.get('height')
    # Label is also in image_data_dict but not directly used for histogram math here,
    # it's used later for creating y_train/y_test.

    if descriptors is None or descriptors.shape[0] == 0 or \
       coordinates is None or coordinates.shape[0] != descriptors.shape[0] or \
       img_width is None or img_height is None or img_width == 0 or img_height == 0:
        # tqdm.write(f"Warning: Missing data for SPM hist. Desc: {descriptors.shape if descriptors is not None else 'None'}") # Can be too verbose
        return np.zeros(expected_hist_shape, dtype=np.float32)

    descriptors_float = descriptors.astype(np.float32) if descriptors.dtype != np.float32 else descriptors

    try:
        visual_words_for_image = kmeans_model.predict(descriptors_float)
    except Exception as e:
        # tqdm.write(f"Error during kmeans.predict: {e}. Desc shape: {descriptors_float.shape}. Returning zeros.")
        return np.zeros(expected_hist_shape, dtype=np.float32)

    all_histograms_weighted = []
    for l_idx in range(num_pyramid_levels):
        num_splits_per_dim = 2**l_idx
        weight = 1.0 / (2.0 ** l_idx) # Correct SPM weighting from literature (or 2^(l-L) if L is max level)
                                     # Using 1/(2^l) for l=0,1... L-1 implies more weight to fine levels.
                                     # If L is max level (e.g. L=2 for levels 0,1), weight = 2^(l_idx - (num_pyramid_levels -1))
                                     # Let's stick to simpler 1/(2^l_idx) giving more weight to coarse level 0 if l_idx=0 is coarsest.
                                     # Actually, the common one is 2^(l-L_max) for l=0..L_max-1 where L_max is the finest level.
                                     # So for levels 0, 1 (PYRAMID_LEVELS=2), max_level_idx = 1.
                                     # Level 0 (1x1): weight = 2^(0-1) = 0.5
                                     # Level 1 (2x2): weight = 2^(1-1) = 1.0
                                     # This gives more weight to finer details. Let's use this.
        max_level_index = num_pyramid_levels -1
        weight = 2.0**(l_idx - max_level_index)


        region_width_float = img_width / num_splits_per_dim
        region_height_float = img_height / num_splits_per_dim

        for i_col in range(num_splits_per_dim):
            for j_row in range(num_splits_per_dim):
                x_min, x_max = i_col * region_width_float, (i_col + 1) * region_width_float
                y_min, y_max = j_row * region_height_float, (j_row + 1) * region_height_float
                if i_col == num_splits_per_dim - 1: x_max = img_width + 1e-6
                if j_row == num_splits_per_dim - 1: y_max = img_height + 1e-6

                region_visual_words = [
                    visual_words_for_image[kp_idx]
                    for kp_idx, (coord_x, coord_y) in enumerate(coordinates)
                    if x_min <= coord_x < x_max and y_min <= coord_y < y_max
                ]

                histogram_region = np.bincount(region_visual_words, minlength=vocab_size).astype(np.float32)
                if np.sum(histogram_region) > 0: # L1 norm for region
                    histogram_region /= np.sum(histogram_region)
                
                all_histograms_weighted.append(histogram_region * weight)

    if not all_histograms_weighted:
        return np.zeros(expected_hist_shape, dtype=np.float32)
    
    final_spm_histogram = np.concatenate(all_histograms_weighted)
    
    # Global L2 normalization (optional but common)
    # final_spm_histogram = normalize(final_spm_histogram.reshape(1, -1), norm='l2')[0]
    # For BoVW, sum of histograms usually not L2 normalized globally, each region might be L1.
    # Let's keep regional L1 and skip global L2 for now unless results are poor. Summing weighted L1 hists.

    return final_spm_histogram

def _process_image_for_spm_hist_parallel(image_path_key, image_data_dict_value, kmeans_model, vocab_size, num_pyramid_levels):
    """Helper for joblib.Parallel, processes a single image_data_dict."""
    hist = generate_spm_histogram_for_image_data(image_data_dict_value, kmeans_model, vocab_size, num_pyramid_levels)
    # Return the image_path (key) and its numeric label (from image_data_dict) along with the histogram
    return image_path_key, image_data_dict_value.get('label'), hist

def generate_histograms_for_set(feature_type, data_set_name, kmeans_model, vocab_size, num_pyramid_levels, n_jobs=-1):
    """
    Generates SPM histograms for all images in a given set (train or test) for a feature type.
    """
    print(f"\nGenerating {feature_type.upper()} SPM histograms for {data_set_name} set...")
    
    batches_subdir = f'{feature_type}_batches_spm'
    input_batches_path = os.path.join(FEATURES_SPM_BALANCED_DIR, batches_subdir)

    batch_files_pattern_regular = os.path.join(input_batches_path, f'{feature_type}_spm_{data_set_name}_batch_*.pkl')
    batch_files_pattern_final = os.path.join(input_batches_path, f'{feature_type}_spm_{data_set_name}_final_batch_processed*.pkl')
    
    batch_files_for_set = sorted(glob.glob(batch_files_pattern_regular) + glob.glob(batch_files_pattern_final))

    if not batch_files_for_set:
        print(f"Error: No {feature_type.upper()} SPM batch files found for {data_set_name} set in {input_batches_path}")
        return None, None

    print(f"Found {len(batch_files_for_set)} {feature_type.upper()} SPM batch files for {data_set_name} set.")

    # Collect all image data items (image_path -> image_data_dict) from all batch files for this set
    all_image_data_items_for_set = [] # List of (image_path, image_data_dict) tuples
    for batch_file in tqdm(batch_files_for_set, desc=f"Loading {data_set_name} {feature_type} batches"):
        try:
            with open(batch_file, 'rb') as f:
                batch_content = pickle.load(f) # dict: {image_path: image_data_dict}
                all_image_data_items_for_set.extend(batch_content.items())
        except Exception as e:
            tqdm.write(f"Warning: Could not load or process batch file {batch_file}: {e}. Skipping.")
            continue
    
    if not all_image_data_items_for_set:
        print(f"No image data loaded from batches for {feature_type}, {data_set_name} set.")
        return None, None

    print(f"Collected data for {len(all_image_data_items_for_set)} images for {feature_type}, {data_set_name} set.")
    
    # Use joblib.Parallel to generate histograms
    # The input to delayed is (image_path_key, image_data_dict_value, ...)
    results_with_labels = Parallel(n_jobs=n_jobs)(
        delayed(_process_image_for_spm_hist_parallel)(
            img_path, img_data_dict, kmeans_model, vocab_size, num_pyramid_levels
        ) for img_path, img_data_dict in tqdm(all_image_data_items_for_set, desc=f"Building {data_set_name} {feature_type} SPM hists")
    )
    
    # Results are: [(image_path, label, histogram), ...]
    # Filter out any None results if a worker failed catastrophically (though _process_image... should return zeros)
    valid_results = [res for res in results_with_labels if res is not None and res[2] is not None]

    if not valid_results:
        print(f"No {feature_type} SPM histograms successfully generated for {data_set_name} set.")
        return None, None

    # Separate paths, labels, and histograms
    # It's good practice to sort by image_path if a consistent order is absolutely needed,
    # though for training, the order of X and y just needs to match.
    # For now, assume joblib output order is sufficient if input list was fixed.
    # To be safe, let's re-sort based on the image_path to ensure determinism if needed later.
    # However, since we iterate `all_image_data_items_for_set` which is built from sorted glob,
    # and joblib preserves input order for the results list *if the tasks are independent and finish in order*,
    # it might be okay. Let's sort to be absolutely sure.
    
    sorted_valid_results = sorted(valid_results, key=lambda x: x[0]) # Sort by image_path

    image_paths_processed = [res[0] for res in sorted_valid_results]
    labels_numeric = np.array([res[1] for res in sorted_valid_results], dtype=np.int8)
    histograms = np.array([res[2] for res in sorted_valid_results], dtype=np.float32)

    if histograms.ndim == 1 and histograms.size == 0: # Handle case of single empty hist
        print(f"Warning: Only empty histogram data returned for {feature_type} {data_set_name}.")
        return None, None
    elif histograms.ndim == 1 and histograms.size > 0 : # Single histogram returned
        histograms = histograms.reshape(1, -1)


    print(f"Generated {histograms.shape[0]} {feature_type.upper()} SPM histograms for {data_set_name} set. Shape: {histograms.shape}")
    return histograms, labels_numeric, image_paths_processed


def main_histogram_creation_spm_balanced():
    print("--- Starting SPM Histogram Generation (from Balanced Data) ---")

    feature_types = ['sift', 'orb']
    data_sets = ['train', 'test']

    for ft_type in feature_types:
        kmeans_model_file = os.path.join(VOCAB_DIR_SPM_BALANCED, f'{ft_type}_kmeans_model_spm_balanced_k{VOCAB_SIZE}_train.joblib')
        if not os.path.exists(kmeans_model_file):
            print(f"ERROR: KMeans model for {ft_type.upper()} not found at {kmeans_model_file}. Skipping this feature type.")
            print(f"Please run 'build_vocabulary_spm_balanced.py' first.")
            continue
        
        print(f"\nLoading {ft_type.upper()} KMeans model from: {kmeans_model_file}")
        try:
            kmeans_model = joblib.load(kmeans_model_file)
        except Exception as e:
            print(f"ERROR loading {ft_type.upper()} KMeans model: {e}. Skipping this feature type.")
            continue

        for set_name in data_sets:
            X_spm_hist, y_labels_numeric, processed_paths = generate_histograms_for_set(
                ft_type, set_name, kmeans_model, VOCAB_SIZE, PYRAMID_LEVELS,
                n_jobs=os.cpu_count() - 2 if os.cpu_count() > 2 else 1 # Adjust n_jobs
            )

            if X_spm_hist is not None and X_spm_hist.size > 0:
                output_hist_filename = f"X_{set_name}_{ft_type}_spm_L{PYRAMID_LEVELS-1}_k{VOCAB_SIZE}.npy"
                output_labels_filename = f"y_{set_name}_{ft_type}_spm_L{PYRAMID_LEVELS-1}_k{VOCAB_SIZE}_labels.npy"
                output_paths_filename = f"paths_{set_name}_{ft_type}_spm_L{PYRAMID_LEVELS-1}_k{VOCAB_SIZE}.pkl" # Save paths for reference

                np.save(os.path.join(SPM_HISTOGRAMS_OUTPUT_DIR, output_hist_filename), X_spm_hist)
                np.save(os.path.join(SPM_HISTOGRAMS_OUTPUT_DIR, output_labels_filename), y_labels_numeric)
                with open(os.path.join(SPM_HISTOGRAMS_OUTPUT_DIR, output_paths_filename), 'wb') as f_paths:
                    pickle.dump(processed_paths, f_paths)

                print(f"Saved {set_name} {ft_type.upper()} SPM features to: {output_hist_filename}")
                print(f"Saved {set_name} {ft_type.upper()} SPM labels to: {output_labels_filename}")
                print(f"Saved {set_name} {ft_type.upper()} SPM processed paths to: {output_paths_filename}")
            else:
                print(f"No SPM histograms generated for {ft_type.upper()} {set_name} set.")
            
            # Explicit GC
            del X_spm_hist, y_labels_numeric, processed_paths
            gc.collect()
        del kmeans_model
        gc.collect()


    print("\n--- All SPM Histogram Generation (Balanced) Complete ---")
    print(f"Final SPM histograms saved in: {SPM_HISTOGRAMS_OUTPUT_DIR}")
