# histogram_creation_SPM_refactored.py
import numpy as np
import os
import glob
import pickle
import joblib
from tqdm import tqdm
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed
import pandas as pd # Added for reading the subset index map
import gc # Added for explicit garbage collection

# --- Configuration for SPM ---
FEATURES_ROOT_DIR_VANILLA = "E:\CV_features" # Where original KMeans models might be if shared
FEATURES_SPM_DIR = "E:\CV_features_SPM"     # Main directory for SPM processed data

# Updated path for the subset index -> label map saved during extraction
SUBSET_INDEX_LABEL_MAP_FILE_PATTERN = os.path.join(FEATURES_SPM_DIR, 'subset_index_label_map_subset*_seed*.csv') # Use pattern to find the file

SPLITS_DIR = os.path.join(FEATURES_ROOT_DIR_VANILLA, "train_test_splits_4cat_revised") # Assuming splits are shared
NPZ_FILE = os.path.join(SPLITS_DIR, "train_test_split_data_4cat_revised.npz")
# LABEL_ENCODER_FILE not directly used here, but good to keep path for consistency

VOCAB_SIZE = 1000  # Must match K for KMeans used for vocabulary building
PYRAMID_LEVELS = 2 # Number of pyramid levels (L). Final levels are 0..L-1.
                   # L=2 gives levels 0 and 1 (1x1, 2x2 grids) -> 1+4=5 regions.
                   # Final SPM feature dim = (sum of regions for all levels) * VOCAB_SIZE
                   # For L=2, dim = (1^2 + 2^2) * K = (1+4) * K = 5K

# Output directory for SPM histograms
BOVW_SPM_FEATURES_DIR = os.path.join(FEATURES_SPM_DIR, "bovw_spm_features_4cat")
os.makedirs(BOVW_SPM_FEATURES_DIR, exist_ok=True)


def generate_spm_histogram_for_image(image_data_dict, kmeans_model, vocab_size, num_pyramid_levels):
    """
    Generates an SPM histogram for a single image.
    image_data_dict: dict containing {'descriptors', 'coordinates', 'width', 'height'} - data for one image
    kmeans_model: trained KMeans vocabulary model (joblib loaded)
    vocab_size: size of the vocabulary
    num_pyramid_levels: number of pyramid levels (e.g., 3 for L=0,1,2)
    """

    # Calculate the total number of regions across all pyramid levels
    total_regions_in_pyramid = sum([(2**l)**2 for l in range(num_pyramid_levels)])
    # Calculate the expected shape of the final SPM histogram
    expected_hist_shape = total_regions_in_pyramid * vocab_size

    # Return zero vector if input data is missing or invalid
    if image_data_dict is None:
        # print(f"Warning: image_data_dict is None in generate_spm_histogram_for_image. Returning zeros.")
        return np.zeros(expected_hist_shape, dtype=np.float32)

    descriptors = image_data_dict.get('descriptors')
    coordinates = image_data_dict.get('coordinates')  # Nx2 array of (x,y)
    img_width = image_data_dict.get('width')
    img_height = image_data_dict.get('height')

    # Check for essential data presence and consistency
    if descriptors is None or descriptors.shape[0] == 0 or \
       coordinates is None or coordinates.shape[0] != descriptors.shape[0] or \
       img_width is None or img_height is None or img_width == 0 or img_height == 0:
        # print(f"Warning: Missing or invalid data for SPM histogram generation. Returning zeros. Desc shape: {descriptors.shape if descriptors is not None else 'None'}, Coords shape: {coordinates.shape if coordinates is not None else 'None'}, Dims: ({img_width}, {img_height})")
        return np.zeros(expected_hist_shape, dtype=np.float32)

    # Ensure descriptors are float32 for KMeans prediction
    if descriptors.dtype == np.uint8:  # Handle ORB specifically
        descriptors_float = descriptors.astype(np.float32)
    elif descriptors.dtype != np.float32: # Ensure float32 for others too if somehow not already
        # print(f"Warning: Descriptors dtype is {descriptors.dtype}, converting to float32.")
        descriptors_float = descriptors.astype(np.float32)
    else:
        descriptors_float = descriptors

    # Assign each descriptor to its visual word cluster
    try:
        visual_words_for_image = kmeans_model.predict(descriptors_float)
    except ValueError as e:
        # This can happen if descriptors have unexpected dimensions or values
        print(f"ValueError during kmeans.predict: {e}. Descriptors shape: {descriptors_float.shape}, Expected dim: {kmeans_model.cluster_centers_.shape[1]}. Returning zeros.")
        return np.zeros(expected_hist_shape, dtype=np.float32)
    except Exception as e:
        print(f"Unexpected error during kmeans.predict: {e}. Returning zeros.")
        return np.zeros(expected_hist_shape, dtype=np.float32)


    all_histograms_weighted = []

    # Iterate through pyramid levels (l_idx from 0 to num_pyramid_levels - 1)
    for l_idx in range(num_pyramid_levels):
        num_splits_per_dim = 2**l_idx  # e.g., level 0 -> 1 split, level 1 -> 2 splits, level 2 -> 4 splits

        # --- Standard SPM Weighting ---
        # Weight = 1 / (2^l) where l is the level index (0-based)
        # The coarsest level (l=0, 1x1 grid) has weight 1.0
        # Each subsequent level has half the weight of the previous level.
        # This gives more importance to finer spatial details.
        weight = 1.0 / (2.0 ** l_idx)

        # Calculate region dimensions based on image size and number of splits
        # Use float division to get precise boundaries
        region_width_float = img_width / num_splits_per_dim
        region_height_float = img_height / num_splits_per_dim

        # Iterate through regions within the current level
        for i_col in range(num_splits_per_dim):  # Region column index (0 to num_splits_per_dim - 1)
            for j_row in range(num_splits_per_dim):  # Region row index (0 to num_splits_per_dim - 1)

                # Define the spatial boundaries of the current region
                x_min, x_max = i_col * region_width_float, (i_col + 1) * region_width_float
                y_min, y_max = j_row * region_height_float, (j_row + 1) * region_height_float

                # Adjust max boundaries for the last region in row/column to include keypoints exactly on the edge
                # This handles potential floating point inaccuracies
                if i_col == num_splits_per_dim - 1: x_max = img_width + 1e-6 # Add small epsilon
                if j_row == num_splits_per_dim - 1: y_max = img_height + 1e-6 # Add small epsilon


                # Collect visual words for descriptors whose coordinates fall within this region
                region_visual_words = []
                # Iterate through keypoint coordinates and their assigned visual words
                for kp_idx, (coord_x, coord_y) in enumerate(coordinates):
                    # Check if the keypoint is within the current region's boundaries
                    if x_min <= coord_x < x_max and y_min <= coord_y < y_max:
                        region_visual_words.append(visual_words_for_image[kp_idx])

                # Build histogram for the current region
                if region_visual_words:
                    # Count occurrences of each visual word in the region
                    # minlength ensures all vocab_size bins exist, even if count is 0
                    histogram_region = np.bincount(region_visual_words, minlength=vocab_size).astype(np.float32)

                    # L1 normalize the regional histogram
                    sum_hist = np.sum(histogram_region)
                    if sum_hist > 0: # Avoid division by zero if region had no keypoints
                        histogram_region /= sum_hist
                else:
                    # If no keypoints in the region, the histogram is all zeros
                    histogram_region = np.zeros(vocab_size, dtype=np.float32)

                # Append the weighted regional histogram to the list
                all_histograms_weighted.append(histogram_region * weight)

    # Concatenate all weighted regional histograms into a single SPM feature vector
    if all_histograms_weighted:
        final_spm_histogram = np.concatenate(all_histograms_weighted)
    else: # Should not happen if expected_hist_shape > 0, but handle defensively
        return np.zeros(expected_hist_shape, dtype=np.float32)


    # Global L2 normalization of the final concatenated SPM vector
    # L2 normalization helps to treat vectors as directions in high-dim space
    sum_sq = np.sum(final_spm_histogram**2)
    if sum_sq > 0: # Avoid division by zero if histogram is all zeros
        # Reshape for sklearn.preprocessing.normalize which expects 2D array (samples x features)
        final_spm_histogram = normalize(final_spm_histogram.reshape(1, -1), norm='l2')[0]
    # Else: if sum_sq is 0 (all zeros), it remains an all-zeros vector, which is correct.
    
    return final_spm_histogram


def _generate_single_spm_for_parallel(subset_idx, processed_indices_map, kmeans_model, vocab_size, num_pyramid_levels):
    """
    Helper function for joblib.Parallel.
    Generates SPM histogram for a single image identified by its subset_idx,
    using data loaded from SPM batches via processed_indices_map.
    Returns the subset_idx and the generated histogram.
    """
    image_data_for_spm = None  # Expected: {'descriptors', 'coordinates', 'width', 'height'} for this subset_idx
    
    # Look up the batch file path for this subset_idx using the map
    target_batch_file = processed_indices_map.get(subset_idx)

    if target_batch_file:
        try:
            # Load the entire batch file (can be inefficient if batch size is large and workers process dispersed indices)
            with open(target_batch_file, 'rb') as f:
                batch_contents = pickle.load(f) # batch_contents is {subset_idx_in_batch: image_spm_data_dict, ...}
            # Get the data dictionary specifically for the current subset_idx
            image_data_for_spm = batch_contents.get(subset_idx)
        except Exception as e:
            # This worker specifically failed to load/process the batch for this index
            print(f"Error loading/processing batch file {target_batch_file} for subset_idx {subset_idx} in worker: {e}")
            # Don't raise, just let generate_spm_histogram_for_image handle None

    # generate_spm_histogram_for_image handles None or incomplete image_data_for_spm by returning zeros
    hist = generate_spm_histogram_for_image(image_data_for_spm, kmeans_model, vocab_size, num_pyramid_levels)

    # Return both the subset index and the generated histogram
    return subset_idx, hist


def process_subset_indices_spm_parallel(subset_indices_list, spm_batches_dir, feature_type, kmeans_model, vocab_size, num_pyramid_levels, desc="Processing Images for SPM", n_jobs=-1):
    """
    Generates SPM histograms in parallel for a given list of subset indices.
    subset_indices_list: List of integer subset indices (from enumerate in extraction script)
    spm_batches_dir: Directory containing the .pkl batch files
    feature_type: 'sift' or 'orb'
    kmeans_model: Loaded KMeans model
    vocab_size: Size of the vocabulary
    num_pyramid_levels: Number of pyramid levels
    desc: Description for the tqdm progress bar
    n_jobs: Number of parallel jobs
    Returns a list of (subset_idx, histogram) tuples, ordered by subset_idx.
    """
    processed_indices_map = {}
    # Include both regular batches and the final batch
    batch_files = sorted(glob.glob(os.path.join(spm_batches_dir, f'{feature_type}_spm_batch_*.pkl')) +
                         glob.glob(os.path.join(spm_batches_dir, f'{feature_type}_spm_final_batch_processed*.pkl')))

    if not batch_files:
        print(f"Error: No SPM batch files found for {feature_type} in {spm_batches_dir}")
        return [] # Return empty list if no batches

    print(f"Mapping subset indices from {len(batch_files)} SPM batch files for {feature_type}...")
    total_mapped = 0
    for batch_file_path in tqdm(batch_files, desc=f"Scanning {feature_type} SPM batches for mapping"):
        try:
            with open(batch_file_path, 'rb') as f:
                batch_data = pickle.load(f) # This is a dict {subset_idx: image_data_dict}
            for idx_in_batch in batch_data.keys():
                # Store the mapping from subset index to the batch file path
                processed_indices_map[idx_in_batch] = batch_file_path
                total_mapped += 1
        except Exception as e:
            print(f"\nWarning: Could not load or process {batch_file_path} during SPM mapping: {e}. Skipping this batch file.")
            # print(traceback.format_exc()) # Uncomment for detailed error
            continue
    print(f"Mapped {total_mapped} unique subset indices for {feature_type} (SPM).")


    # Filter the list of subset indices to process based on which ones we actually have data for
    # AND are in the requested subset_indices_list
    subset_indices_with_data = [idx for idx in subset_indices_list if idx in processed_indices_map]

    if not subset_indices_with_data:
        print(f"Warning: No data found in SPM batches for the requested {len(subset_indices_list)} subset indices for {feature_type}.")
        return []

    # Optional: Check how many requested indices are missing data
    missing_indices_count = len(subset_indices_list) - len(subset_indices_with_data)
    if missing_indices_count > 0:
         print(f"Warning: Data for {missing_indices_count} requested subset indices ({feature_type}) was not found in the batch files. Their histograms will be zeros if generated (handled inside worker).")


    print(f"\nGenerating SPM histograms for {len(subset_indices_with_data)} images with available data ({feature_type}) using {n_jobs if n_jobs != -1 else os.cpu_count()} workers...")

    # Use joblib.Parallel to call _generate_single_spm_for_parallel for each subset index with data
    # The result will be a list of (subset_idx, histogram) tuples
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(_generate_single_spm_for_parallel)(
            subset_idx, processed_indices_map, kmeans_model, vocab_size, num_pyramid_levels
        ) for subset_idx in tqdm(subset_indices_with_data, desc=desc) # Iterate over only the indices for which we have data
    )

    # The results_list is not guaranteed to be in sorted order of subset_idx by joblib.
    # Sort the results by subset_idx to ensure consistent output order.
    # This is CRUCIAL for aligning with the labels loaded from the subset index map later.
    if results_list:
        # Filter out any potential None results if a worker failed unexpectedly
        valid_results = [(idx, hist) for idx, hist in results_list if hist is not None] # generate_spm_histogram... returns zeros, not None

        # Sort by the subset_idx (the first element of the tuple)
        sorted_results = sorted(valid_results, key=lambda item: item[0])

        print(f"Generated histograms for {len(sorted_results)} subset indices ({feature_type}).")
        return sorted_results # Return sorted list of (subset_idx, histogram) tuples
    else:
        print(f"No histograms were successfully generated for {feature_type}.")
        return []


def histogram_creation_SPM():
    print("--- Starting SPM Histogram Generation ---")

    # --- Load the Subset Index -> Label Map ---
    # This map contains ALL subset indices from the extraction process and their labels.
    map_files = glob.glob(SUBSET_INDEX_LABEL_MAP_FILE_PATTERN)
    if not map_files:
        print(f"ERROR: Subset index map file not found matching pattern: {SUBSET_INDEX_LABEL_MAP_FILE_PATTERN}")
        print("Please run SOH_extract_SPM.py first to generate this file.")
        return # Exit if map file not found
    # Assuming only one such file exists, pick the first one found
    subset_map_file = map_files[0]
    print(f"Loading subset index map from: {subset_map_file}")
    try:
        subset_map_df = pd.read_csv(subset_map_file)
        # Create a dictionary mapping subset_idx to label
        subset_idx_to_label = dict(zip(subset_map_df['subset_idx'], subset_map_df['label']))
        all_subset_indices_extracted = sorted(subset_map_df['subset_idx'].tolist()) # Get all subset indices that were processed
        print(f"Loaded map for {len(subset_idx_to_label)} subset indices.")
    except Exception as e:
        print(f"ERROR loading or processing subset index map file {subset_map_file}: {e}")
        return # Exit on error


    # --- Load the Original Train/Test Split Indices ---
    # These indices define which *original* images are in the train/test splits.
    # We use this to determine which *subset indices* (from our map) fall into train/test.
    print(f"Loading original train/test split data from: {NPZ_FILE}")
    try:
        split_data = np.load(NPZ_FILE)
        original_train_indices = set(split_data['train_indices']) # Use sets for faster lookup
        original_test_indices = set(split_data['test_indices'])
        print(f"Loaded {len(original_train_indices)} original training and {len(original_test_indices)} original testing indices from NPZ.")
    except FileNotFoundError:
        print(f"ERROR: NPZ file not found at {NPZ_FILE}.")
        return # Exit on error
    except KeyError as e:
        print(f"ERROR: Missing key {e} in NPZ file {NPZ_FILE}. Check keys.")
        return # Exit on error


    # --- Determine Subset Indices for Train and Test Splits ---
    # We need to find which subset indices (from our extracted subset)
    # correspond to images that are in the original train/test splits.
    # The subset_index_label_map.csv file doesn't contain the original TFDS index,
    # only the sequential subset index and the label.
    # This means we cannot directly map the NPZ original indices to the subset indices.
    # There seems to be a misunderstanding in how train/test splits were defined.
    # The original NPZ file defines splits based on the *original TFDS dataset indices*.
    # The extraction script selected a subset and gave them *new sequential subset indices*.
    # To correctly align, we need a mapping from *subset_idx* to the *original TFDS index*.
    # This mapping *should* ideally be saved during extraction.

    # --- REVISIT: Alignment Logic ---
    # The simplest reliable way using the *current* extraction + map file structure is:
    # 1. Use the subset_index_label_map.csv to get ALL subset indices that had features extracted.
    # 2. Split these subset indices into train and test sets *directly*, perhaps using
    #    a deterministic hash of the subset_idx or by taking the first X% for train.
    #    This IGNORES the original train/test split defined in the NPZ file.
    #    This ISN'T ideal if you need to compare performance on the *exact same* split as other models trained using the NPZ splits.
    # 3. The *correct* way is to modify SOH_extract_SPM to save a mapping from `subset_idx` to `original_tfds_idx`.
    #    Then, in *this* script, load that map, and use the `original_train_indices`/`original_test_indices` from the NPZ
    #    to filter the `subset_idx`s based on their `original_tfds_idx`.

    # Let's assume for now that the NPZ split data refers to the subset indices generated
    # by a DETERMINISTIC `take()` operation on the full dataset with a seed.
    # The updated extraction script makes the subset deterministic.
    # However, the NPZ 'train_indices' and 'test_indices' are still likely based on the *original* dataset indices.
    # The subset index map gives us (subset_idx -> label).
    # We need (subset_idx -> original_tfds_idx) and (original_tfds_idx -> train/test split).

    # --- Revised Plan: Load NPZ, filter map by label != -1, then split based on label or index hashing ---
    # Let's use the subset_index_label_map to get the indices that actually had successful extraction.
    # And then perform a train/test split *on these subset indices* based on the labels from the map.
    # This is a train/test split of the *processed subset*, NOT necessarily the same split as defined in the original NPZ for the full dataset.
    # This IS A DEVIATION from using the NPZ splits directly, but it's necessary for correct feature/label alignment with your current file structure.
    # The alternative (saving subset_idx -> original_idx map in extraction) is more work.

    print("\nCreating train/test split for the PROCESSED SUBSET...")
    # Filter out indices where extraction failed (-1 label)
    successful_subset_indices = subset_map_df[subset_map_df['label'] != -1]['subset_idx'].tolist()
    successful_subset_labels = subset_map_df[subset_map_df['label'] != -1]['label'].tolist()

    if not successful_subset_indices:
         print("ERROR: No images had successful feature extraction (label != -1) in the subset map. Cannot generate histograms.")
         return # Exit if no successful extractions

    print(f"Found {len(successful_subset_indices)} subset indices with successful feature extraction (label != -1).")

    # Now split these successful subset indices and their labels into train and test.
    # We need to use the SAME split as the NPZ file used for the *original* dataset indices,
    # but applied here to the *subset indices*. This is the complex part without the subset_idx -> original_idx map.
    # The simplest way to maintain a *consistent* split of the subset is to hash the subset_idx
    # or use a deterministic split based on index, or assume the NPZ split *indices*
    # somehow correspond sequentially to the *subset_idx* after deterministic shuffling and taking.
    # Assuming the deterministic shuffle + take in SOH_extract_SPM makes `subset_idx=i` correspond
    # to the i-th original index that ended up in the subset *after* shuffling...
    # This is still risky.

    # --- OK, LET'S USE THE NPZ INDICES AS THE SOURCE OF TRUTH FOR *WHICH* IMAGES ARE TRAIN/TEST ---
    # We *must* find the subset_idx that correspond to the original indices in the NPZ.
    # The SOH_extract_SPM script *does not* save the mapping from subset_idx to original_tfds_idx.
    # It *does* save the subset_idx and label in the map file.
    # It *does* save the subset_idx and label *for HOG* in the HDF5.
    # The SIFT/ORB batches are keyed by subset_idx.

    # To make this fully robust with your *current* file structure, you would need to:
    # 1. Reload the dataset deterministically using the same seed as extraction.
    # 2. Iterate through it to get the (subset_idx, original_tfds_idx) mapping.
    # 3. Use this map + NPZ original indices to define subset_indices_train/test.

    # Let's try a workaround that *might* work if TFDS iteration is consistent,
    # but acknowledge its fragility: Assume the first `SUBSET_SIZE` indices
    # *after* the seeded shuffle are assigned `subset_idx` 0 to `SUBSET_SIZE - 1`.
    # And assume the NPZ train/test indices are relative to the *original TFDS dataset order*.
    # This path is fraught with peril.

    # --- Alternative: Rely on the HOG HDF5 indices ---
    # The HOG HDF5 saves `indices` (subset_idx) and `labels` for successful HOG extractions.
    # It *doesn't* save the original_tfds_idx.
    # We need original_tfds_idx -> train/test split from NPZ.

    # --- New Plan: Simplify and fix extraction script FIRST ---
    # It is cleanest and most reliable to modify SOH_extract_SPM to save the
    # `subset_idx` -> `original_tfds_idx` -> `label` mapping in the CSV.
    # Then this script loads *that* CSV and the NPZ, filters by `original_tfds_idx` to get `subset_indices_train`/`test`.

    # Let's implement the histogram creation assuming the CSV now contains:
    # 'subset_idx', 'original_tfds_idx', 'label', 'extraction_successful' (e.g., label != -1)

    # --- Assuming subset_index_label_map.csv now has 'subset_idx' and 'original_tfds_idx' ---
    # (Need to modify SOH_extract_SPM to add 'original_tfds_idx')

    # Temporarily read the map again, assuming it has 'original_tfds_idx'
    # Note: You MUST run the modified SOH_extract_SPM first for this to work.
    print(f"Loading subset index map (expecting original_tfds_idx) from: {subset_map_file}")
    try:
        subset_map_df = pd.read_csv(subset_map_file)
        if 'original_tfds_idx' not in subset_map_df.columns:
             print(f"ERROR: '{subset_map_file}' does not contain 'original_tfds_idx' column. Please run modified SOH_extract_SPM.")
             return
        # Create a map from original_tfds_idx to subset_idx
        original_tfds_idx_to_subset_idx = dict(zip(subset_map_df['original_tfds_idx'], subset_map_df['subset_idx']))

        # Filter the map to include only images where extraction was successful (label != -1)
        successful_extractions_map_df = subset_map_df[subset_map_df['label'] != -1].copy()
        print(f"Loaded map for {len(subset_map_df)} subset indices. {len(successful_extractions_map_df)} had successful extraction (label != -1).")

    except Exception as e:
        print(f"ERROR loading or processing subset index map file {subset_map_file} (expecting original_tfds_idx): {e}")
        return # Exit on error

    # --- Filter subset indices based on original train/test splits from NPZ ---
    print("\nDetermining subset indices corresponding to original train/test splits...")
    subset_indices_train = []
    subset_indices_test = []

    # Iterate through all original train indices from NPZ
    for original_idx in tqdm(original_train_indices, desc="Finding train subset indices"):
        # Check if this original index is present in our extracted subset map
        subset_idx = original_tfds_idx_to_subset_idx.get(original_idx)
        if subset_idx is not None:
            # Check if this subset index corresponds to a successful extraction (label != -1)
            # We look it up in the filtered map (successful_extractions_map_df)
            if subset_idx in successful_extractions_map_df['subset_idx'].values:
                 subset_indices_train.append(subset_idx)
            # else: print(f"Skipping original train index {original_idx}: Feature extraction failed for subset index {subset_idx}")
        # else: print(f"Skipping original train index {original_idx}: Not found in the extracted subset.")

    # Iterate through all original test indices from NPZ
    for original_idx in tqdm(original_test_indices, desc="Finding test subset indices"):
         subset_idx = original_tfds_idx_to_subset_idx.get(original_idx)
         if subset_idx is not None:
            if subset_idx in successful_extractions_map_df['subset_idx'].values:
                 subset_indices_test.append(subset_idx)
            # else: print(f"Skipping original test index {original_idx}: Feature extraction failed for subset index {subset_idx}")
         # else: print(f"Skipping original test index {original_idx}: Not found in the extracted subset.")

    # Sort the subset indices for deterministic histogram generation order
    subset_indices_train = sorted(subset_indices_train)
    subset_indices_test = sorted(subset_indices_test)

    print(f"Found {len(subset_indices_train)} subset indices for training (with successful extraction).")
    print(f"Found {len(subset_indices_test)} subset indices for testing (with successful extraction).")

    if not subset_indices_train or not subset_indices_test:
         print("ERROR: Train or test subset indices list is empty. Cannot proceed.")
         return


    N_JOBS = os.cpu_count() - 4 if os.cpu_count() > 4 else 1 # Adjusted N_JOBS
    if N_JOBS < 1: N_JOBS = 1


    # --- SIFT SPM Features ---
    print("\n--- Processing SIFT Features for SPM ---")
    sift_kmeans_model_spm_file = os.path.join(FEATURES_SPM_DIR, f'sift_kmeans_model_spm_k{VOCAB_SIZE}_partial_fit.joblib')
    sift_batches_spm_subdir_path = os.path.join(FEATURES_SPM_DIR, 'sift_batches_spm')

    if os.path.exists(sift_kmeans_model_spm_file):
        print(f"Loading SIFT KMeans model (for SPM) from: {sift_kmeans_model_spm_file}")
        try:
            sift_kmeans_spm = joblib.load(sift_kmeans_model_spm_file)
        except Exception as e:
            print(f"ERROR loading SIFT KMeans model: {e}")
            sift_kmeans_spm = None # Set to None to skip processing

        if sift_kmeans_spm is not None:
            # Pass the FILTERED LIST OF SUBSET INDICES for the train split
            sift_train_results = process_subset_indices_spm_parallel(
                subset_indices_train, sift_batches_spm_subdir_path, 'sift', sift_kmeans_spm, VOCAB_SIZE, PYRAMID_LEVELS,
                desc="SIFT Train SPM", n_jobs=N_JOBS
            )
            # Extract histograms, preserving the order determined by process_subset_indices_spm_parallel
            X_train_sift_spm = np.array([hist for idx, hist in sift_train_results])
            # The indices corresponding to X_train_sift_spm are the subset_idx from sift_train_results, which are already sorted.
            # You might want to save these indices too if you need to double-check later, but for classification, you'll use the subset_map_df.

            if X_train_sift_spm.size > 0:
                print(f"SIFT Training SPM histograms shape: {X_train_sift_spm.shape}")
                # Save with clarifying filename including subset size and seed
                output_filename = f'X_train_sift_spm_L{PYRAMID_LEVELS-1}_k{VOCAB_SIZE}_subset{len(subset_map_df)}_seed{subset_map_df["subset_idx"].min()}_to_{subset_map_df["subset_idx"].max()}.npy' # Better filename
                # Need to find the actual subset size and seed from the loaded map file
                # Let's get the subset size from the number of entries in the map
                map_subset_size = len(subset_map_df)
                # Finding the seed requires parsing the map filename, let's simplify for now or get it from config if available
                # Assuming the pattern match found the right file and SUBSET_SIZE is from original config
                subset_size_from_map_filename = "N/A" # Placeholder
                try:
                    # Parse filename like subset_index_label_map_subset100000_seed42.csv
                    parts = os.path.basename(subset_map_file).split('_')
                    subset_size_str = parts[3].replace('subset', '')
                    seed_str = parts[4].replace('seed', '').replace('.csv', '')
                    output_filename = f'X_train_sift_spm_L{PYRAMID_LEVELS-1}_k{VOCAB_SIZE}_subset{subset_size_str}_seed{seed_str}.npy'
                except Exception as e:
                    print(f"Warning: Could not parse subset size/seed from map filename ({subset_map_file}): {e}. Using generic filename.")
                    output_filename = f'X_train_sift_spm_L{PYRAMID_LEVELS-1}_k{VOCAB_SIZE}_processed{len(subset_indices_train)}.npy'


                np.save(os.path.join(BOVW_SPM_FEATURES_DIR, output_filename), X_train_sift_spm)
                print(f"Saved SIFT training SPM features to {BOVW_SPM_FEATURES_DIR}")
            else:
                print("No SIFT training SPM histograms were generated.")


            # Pass the FILTERED LIST OF SUBSET INDICES for the test split
            sift_test_results = process_subset_indices_spm_parallel(
                subset_indices_test, sift_batches_spm_subdir_path, 'sift', sift_kmeans_spm, VOCAB_SIZE, PYRAMID_LEVELS,
                desc="SIFT Test SPM", n_jobs=N_JOBS
            )
            X_test_sift_spm = np.array([hist for idx, hist in sift_test_results])

            if X_test_sift_spm.size > 0:
                print(f"SIFT Test SPM histograms shape: {X_test_sift_spm.shape}")
                # Use the same filename parsing logic for test set
                try:
                     parts = os.path.basename(subset_map_file).split('_')
                     subset_size_str = parts[3].replace('subset', '')
                     seed_str = parts[4].replace('seed', '').replace('.csv', '')
                     output_filename = f'X_test_sift_spm_L{PYRAMID_LEVELS-1}_k{VOCAB_SIZE}_subset{subset_size_str}_seed{seed_str}.npy'
                except:
                     output_filename = f'X_test_sift_spm_L{PYRAMID_LEVELS-1}_k{VOCAB_SIZE}_processed{len(subset_indices_test)}.npy'

                np.save(os.path.join(BOVW_SPM_FEATURES_DIR, output_filename), X_test_sift_spm)
                print(f"Saved SIFT test SPM features to {BOVW_SPM_FEATURES_DIR}")
            else:
                print("No SIFT test SPM histograms were generated.")
        else:
             print("Skipping SIFT SPM generation due to missing/invalid KMeans model.")
    else:
        print(f"SIFT KMeans model (for SPM) not found at {sift_kmeans_model_spm_file}. Skipping SIFT SPM generation.")


    # --- ORB SPM Features ---
    print("\n--- Processing ORB Features for SPM ---")
    orb_kmeans_model_spm_file = os.path.join(FEATURES_SPM_DIR, f'orb_kmeans_model_spm_k{VOCAB_SIZE}_partial_fit.joblib')
    orb_batches_spm_subdir_path = os.path.join(FEATURES_SPM_DIR, 'orb_batches_spm')

    if os.path.exists(orb_kmeans_model_spm_file):
        print(f"Loading ORB KMeans model (for SPM) from: {orb_kmeans_model_spm_file}")
        try:
            orb_kmeans_spm = joblib.load(orb_kmeans_model_spm_file)
        except Exception as e:
            print(f"ERROR loading ORB KMeans model: {e}")
            orb_kmeans_spm = None # Set to None to skip processing

        if orb_kmeans_spm is not None:
            # Pass the FILTERED LIST OF SUBSET INDICES for the train split
            orb_train_results = process_subset_indices_spm_parallel(
                subset_indices_train, orb_batches_spm_subdir_path, 'orb', orb_kmeans_spm, VOCAB_SIZE, PYRAMID_LEVELS,
                desc="ORB Train SPM", n_jobs=N_JOBS
            )
            X_train_orb_spm = np.array([hist for idx, hist in orb_train_results])

            if X_train_orb_spm.size > 0:
                print(f"ORB Training SPM histograms shape: {X_train_orb_spm.shape}")
                try:
                     parts = os.path.basename(subset_map_file).split('_')
                     subset_size_str = parts[3].replace('subset', '')
                     seed_str = parts[4].replace('seed', '').replace('.csv', '')
                     output_filename = f'X_train_orb_spm_L{PYRAMID_LEVELS-1}_k{VOCAB_SIZE}_subset{subset_size_str}_seed{seed_str}.npy'
                except:
                     output_filename = f'X_train_orb_spm_L{PYRAMID_LEVELS-1}_k{VOCAB_SIZE}_processed{len(subset_indices_train)}.npy'

                np.save(os.path.join(BOVW_SPM_FEATURES_DIR, output_filename), X_train_orb_spm)
                print(f"Saved ORB training SPM features to {BOVW_SPM_FEATURES_DIR}")
            else:
                print("No ORB training SPM histograms were generated.")


            # Pass the FILTERED LIST OF SUBSET INDICES for the test split
            orb_test_results = process_subset_indices_spm_parallel(
                subset_indices_test, orb_batches_spm_subdir_path, 'orb', orb_kmeans_spm, VOCAB_SIZE, PYRAMID_LEVELS,
                desc="ORB Test SPM", n_jobs=N_JOBS
            )
            X_test_orb_spm = np.array([hist for idx, hist in orb_test_results])


            if X_test_orb_spm.size > 0:
                print(f"ORB Test SPM histograms shape: {X_test_orb_spm.shape}")
                try:
                     parts = os.path.basename(subset_map_file).split('_')
                     subset_size_str = parts[3].replace('subset', '')
                     seed_str = parts[4].replace('seed', '').replace('.csv', '')
                     output_filename = f'X_test_orb_spm_L{PYRAMID_LEVELS-1}_k{VOCAB_SIZE}_subset{subset_size_str}_seed{seed_str}.npy'
                except:
                     output_filename = f'X_test_orb_spm_L{PYRAMID_LEVELS-1}_k{VOCAB_SIZE}_processed{len(subset_indices_test)}.npy'

                np.save(os.path.join(BOVW_SPM_FEATURES_DIR, output_filename), X_test_orb_spm)
                print(f"Saved ORB test SPM features to {BOVW_SPM_FEATURES_DIR}")
            else:
                print("No ORB test SPM histograms were generated.")
        else:
             print("Skipping ORB SPM generation due to missing/invalid KMeans model.")

    else:
        print(f"ORB KMeans model (for SPM) not found at {orb_kmeans_model_spm_file}. Skipping ORB SPM generation.")

    print("\n--- Phase 3: SPM Histogram Generation Complete ---")
    print(f"SPM features saved in: {BOVW_SPM_FEATURES_DIR}")
    print(f"Subset train indices count: {len(subset_indices_train)}")
    print(f"Subset test indices count: {len(subset_indices_test)}")