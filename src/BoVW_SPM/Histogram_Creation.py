import numpy as np
import os
import glob
import pickle
import joblib
from tqdm import tqdm
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed

# --- Configuration for SPM ---
FEATURES_ROOT_DIR_VANILLA = "E:\CV_features" # Where original KMeans models might be if shared
FEATURES_SPM_DIR = "E:\CV_features_SPM"     # Main directory for SPM processed data

SPLITS_DIR = os.path.join(FEATURES_ROOT_DIR_VANILLA, "train_test_splits_4cat_revised") # Assuming splits are shared
NPZ_FILE = os.path.join(SPLITS_DIR, "train_test_split_data_4cat_revised.npz")
# LABEL_ENCODER_FILE not directly used here, but good to keep path for consistency

VOCAB_SIZE = 1000  # Must match K for KMeans
PYRAMID_LEVELS = 3 # Number of pyramid levels (e.g., 3 for L=0, L=1, L=2 giving 1+4+16=21 regions)
                   # Final SPM feature dim = (sum of regions for all levels) * VOCAB_SIZE

# Output directory for SPM histograms
BOVW_SPM_FEATURES_DIR = os.path.join(FEATURES_SPM_DIR, "bovw_spm_features_4cat")
os.makedirs(BOVW_SPM_FEATURES_DIR, exist_ok=True)


def generate_spm_histogram_for_image(image_data_dict, kmeans_model, vocab_size, num_pyramid_levels):
    """
    Generates an SPM histogram for a single image.
    image_data_dict: dict containing {'descriptors', 'coordinates', 'width', 'height'}
    kmeans_model: trained KMeans vocabulary
    vocab_size: size of the vocabulary
    num_pyramid_levels: number of pyramid levels (e.g., 3 for L=0,1,2)
    """
    descriptors = image_data_dict.get('descriptors')
    coordinates = image_data_dict.get('coordinates')  # Nx2 array of (x,y)
    img_width = image_data_dict.get('width')
    img_height = image_data_dict.get('height')

    total_regions_in_pyramid = sum([(2**l)**2 for l in range(num_pyramid_levels)])
    empty_hist_shape = total_regions_in_pyramid * vocab_size

    if descriptors is None or descriptors.shape[0] == 0 or \
       coordinates is None or coordinates.shape[0] != descriptors.shape[0] or \
       img_width is None or img_height is None or img_width == 0 or img_height == 0:
        # print(f"Warning: Missing data for SPM histogram generation. Returning zeros. Desc shape: {descriptors.shape if descriptors is not None else 'None'}")
        return np.zeros(empty_hist_shape, dtype=np.float32)

    if descriptors.dtype == np.uint8:  # ORB descriptors are uint8
        descriptors_float = descriptors.astype(np.float32)
    elif descriptors.dtype != np.float32: # Ensure float32 for others too
        descriptors_float = descriptors.astype(np.float32)
    else:
        descriptors_float = descriptors
    
    try:
        visual_words_for_image = kmeans_model.predict(descriptors_float)
    except ValueError as e:
        # print(f"ValueError during kmeans.predict: {e}. Descriptors shape: {descriptors_float.shape}. Returning zeros.")
        return np.zeros(empty_hist_shape, dtype=np.float32)

    all_histograms_weighted = []

    for l_idx in range(num_pyramid_levels):  # Iterate through levels (0, 1, 2 for num_pyramid_levels=3)
        num_splits_per_dim = 2**l_idx  # e.g., l_idx=0 -> 1 split, l_idx=1 -> 2 splits, l_idx=2 -> 4 splits

        # Weight for this level (higher weight for finer levels)
        if l_idx == 0 and num_pyramid_levels > 1: # Coarsest level (if not the only level)
            weight = 1 / (2**(num_pyramid_levels - 1))
        elif num_pyramid_levels == 1: # Only one level (global BoVW)
             weight = 1.0
        else: # Finer levels
            weight = 1 / (2**(num_pyramid_levels - 1 - l_idx))


        region_width_float = img_width / num_splits_per_dim
        region_height_float = img_height / num_splits_per_dim

        for i_col in range(num_splits_per_dim):  # Region column index
            for j_row in range(num_splits_per_dim):  # Region row index
                x_min, x_max = i_col * region_width_float, (i_col + 1) * region_width_float
                y_min, y_max = j_row * region_height_float, (j_row + 1) * region_height_float

                # Ensure x_max and y_max don't accidentally exclude edge keypoints due to float precision
                if i_col == num_splits_per_dim - 1: x_max = img_width + 1
                if j_row == num_splits_per_dim - 1: y_max = img_height + 1


                region_visual_words = []
                for kp_idx, (coord_x, coord_y) in enumerate(coordinates):
                    if x_min <= coord_x < x_max and y_min <= coord_y < y_max:
                        region_visual_words.append(visual_words_for_image[kp_idx])
                
                if region_visual_words:
                    histogram_region = np.bincount(region_visual_words, minlength=vocab_size).astype(np.float32)
                    # L1 normalize each regional histogram
                    sum_hist = np.sum(histogram_region)
                    if sum_hist > 0:
                        histogram_region /= sum_hist
                else:
                    histogram_region = np.zeros(vocab_size, dtype=np.float32)
                
                all_histograms_weighted.append(histogram_region * weight)

    final_spm_histogram = np.concatenate(all_histograms_weighted)

    # Global L2 normalization of the concatenated weighted SPM vector
    if np.sum(final_spm_histogram**2) > 0:
        final_spm_histogram = normalize(final_spm_histogram.reshape(1, -1), norm='l2')[0]
    
    return final_spm_histogram


def _generate_single_spm_for_parallel(image_idx, processed_indices_map, kmeans_model, vocab_size, num_pyramid_levels):
    """
    Generates SPM histogram for a single image_idx using data from SPM batches.
    """
    image_data_for_spm = None  # Expected: {'descriptors': ..., 'coordinates': ..., 'width': ..., 'height': ...}
    if image_idx in processed_indices_map:
        target_batch_file = processed_indices_map[image_idx]
        try:
            with open(target_batch_file, 'rb') as f:
                batch_contents = pickle.load(f) # batch_contents is {img_idx_in_batch: image_spm_data_dict, ...}
            image_data_for_spm = batch_contents.get(image_idx) # Get the dict for this specific image_idx
        except Exception as e:
            print(f"Error loading/processing batch file {target_batch_file} for index {image_idx} in worker: {e}")

    # generate_spm_histogram_for_image handles None image_data_for_spm
    hist = generate_spm_histogram_for_image(image_data_for_spm, kmeans_model, vocab_size, num_pyramid_levels)
    return hist


def process_indices_spm_parallel(indices, spm_batches_dir, feature_type, kmeans_model, vocab_size, num_pyramid_levels, desc="Processing Images for SPM", n_jobs=-1):
    processed_indices_map = {}
    batch_files = sorted(glob.glob(os.path.join(spm_batches_dir, f'{feature_type}_spm_batch_*.pkl')))
    
    if not batch_files:
        print(f"Error: No SPM batch files found for {feature_type} in {spm_batches_dir}")
        return np.array([])
    
    print(f"Mapping indices from {len(batch_files)} SPM batch files for {feature_type}...")
    for batch_file_path in tqdm(batch_files, desc=f"Scanning {feature_type} SPM batches for mapping"):
        try:
            with open(batch_file_path, 'rb') as f:
                batch_data = pickle.load(f)
            for idx_in_batch in batch_data.keys():
                processed_indices_map[idx_in_batch] = batch_file_path
        except Exception as e:
            print(f"Warning: Could not load or process {batch_file_path} during SPM mapping: {e}")
            continue
    print(f"Mapped {len(processed_indices_map)} unique descriptor sets for {feature_type} (SPM).")

    # --- MODIFIED CHECKS HERE ---
    # Check if indices array has any elements (i.e., is not empty)
    if not processed_indices_map and indices.size > 0: 
        print(f"Warning: No descriptors mapped for {feature_type} (SPM), but indices were provided (count: {indices.size}). Histograms might be zeros for many.")
    # Check if indices array is empty
    elif indices.size == 0: 
        print(f"No indices provided for {feature_type} (SPM). Returning empty array.")
        return np.array([])
    # --- END OF MODIFIED CHECKS ---

    # Additional check: if processed_indices_map is empty but we have indices, 
    # all histograms will be zero, which is fine, but good to be aware.
    if not processed_indices_map and indices.size > 0:
        print(f"Note: processed_indices_map is empty for {feature_type}, all SPM histograms will be zero vectors.")


    print(f"\nGenerating SPM histograms for {len(indices)} images ({feature_type}) using {n_jobs if n_jobs != -1 else os.cpu_count()} workers...")
    
    histograms_list = Parallel(n_jobs=n_jobs)(
        delayed(_generate_single_spm_for_parallel)(
            image_idx, processed_indices_map, kmeans_model, vocab_size, num_pyramid_levels
        ) for image_idx in tqdm(indices, desc=desc)
    )
    
    if not histograms_list:
        return np.array([])
        
    return np.array(histograms_list)


def histogram_creation_SPM():
    print("--- Starting SPM Histogram Generation ---")

    print(f"Loading train/test split data from: {NPZ_FILE}")
    split_data = np.load(NPZ_FILE)
    train_indices = split_data['train_indices']
    test_indices = split_data['test_indices']
    print(f"Loaded {len(train_indices)} training and {len(test_indices)} testing indices.")

    N_JOBS = os.cpu_count() - 2 if os.cpu_count() > 2 else 1 # Adjusted N_JOBS

    # --- SIFT SPM Features ---
    print("\n--- Processing SIFT Features for SPM ---")
    # KMEANS MODEL: Use the vocabulary created by build_vocabulary_spm.py
    # This model should be in FEATURES_SPM_DIR or a shared vocab directory
    sift_kmeans_model_spm_file = os.path.join(FEATURES_SPM_DIR, f'sift_kmeans_model_spm_k{VOCAB_SIZE}_partial_fit.joblib')
    sift_batches_spm_subdir_path = os.path.join(FEATURES_SPM_DIR, 'sift_batches_spm') # Full path

    if os.path.exists(sift_kmeans_model_spm_file):
        print(f"Loading SIFT KMeans model (for SPM) from: {sift_kmeans_model_spm_file}")
        sift_kmeans_spm = joblib.load(sift_kmeans_model_spm_file)

        X_train_sift_spm = process_indices_spm_parallel(
            train_indices, sift_batches_spm_subdir_path, 'sift', sift_kmeans_spm, VOCAB_SIZE, PYRAMID_LEVELS,
            desc="SIFT Train SPM", n_jobs=N_JOBS
        )
        if X_train_sift_spm.size > 0:
            print(f"SIFT Training SPM histograms shape: {X_train_sift_spm.shape}")
            np.save(os.path.join(BOVW_SPM_FEATURES_DIR, f'X_train_sift_spm_L{PYRAMID_LEVELS-1}.npy'), X_train_sift_spm)
            print(f"Saved SIFT training SPM features to {BOVW_SPM_FEATURES_DIR}")
        else:
            print("No SIFT training SPM histograms were generated.")


        X_test_sift_spm = process_indices_spm_parallel(
            test_indices, sift_batches_spm_subdir_path, 'sift', sift_kmeans_spm, VOCAB_SIZE, PYRAMID_LEVELS,
            desc="SIFT Test SPM", n_jobs=N_JOBS
        )
        if X_test_sift_spm.size > 0:
            print(f"SIFT Test SPM histograms shape: {X_test_sift_spm.shape}")
            np.save(os.path.join(BOVW_SPM_FEATURES_DIR, f'X_test_sift_spm_L{PYRAMID_LEVELS-1}.npy'), X_test_sift_spm)
            print(f"Saved SIFT test SPM features to {BOVW_SPM_FEATURES_DIR}")
        else:
            print("No SIFT test SPM histograms were generated.")
    else:
        print(f"SIFT KMeans model (for SPM) not found at {sift_kmeans_model_spm_file}. Skipping SIFT SPM generation.")

    # --- ORB SPM Features ---
    print("\n--- Processing ORB Features for SPM ---")
    orb_kmeans_model_spm_file = os.path.join(FEATURES_SPM_DIR, f'orb_kmeans_model_spm_k{VOCAB_SIZE}_partial_fit.joblib')
    orb_batches_spm_subdir_path = os.path.join(FEATURES_SPM_DIR, 'orb_batches_spm') # Full path

    if os.path.exists(orb_kmeans_model_spm_file):
        print(f"Loading ORB KMeans model (for SPM) from: {orb_kmeans_model_spm_file}")
        orb_kmeans_spm = joblib.load(orb_kmeans_model_spm_file)

        X_train_orb_spm = process_indices_spm_parallel(
            train_indices, orb_batches_spm_subdir_path, 'orb', orb_kmeans_spm, VOCAB_SIZE, PYRAMID_LEVELS,
            desc="ORB Train SPM", n_jobs=N_JOBS
        )
        if X_train_orb_spm.size > 0:
            print(f"ORB Training SPM histograms shape: {X_train_orb_spm.shape}")
            np.save(os.path.join(BOVW_SPM_FEATURES_DIR, f'X_train_orb_spm_L{PYRAMID_LEVELS-1}.npy'), X_train_orb_spm)
            print(f"Saved ORB training SPM features to {BOVW_SPM_FEATURES_DIR}")
        else:
            print("No ORB training SPM histograms were generated.")


        X_test_orb_spm = process_indices_spm_parallel(
            test_indices, orb_batches_spm_subdir_path, 'orb', orb_kmeans_spm, VOCAB_SIZE, PYRAMID_LEVELS,
            desc="ORB Test SPM", n_jobs=N_JOBS
        )
        if X_test_orb_spm.size > 0:
            print(f"ORB Test SPM histograms shape: {X_test_orb_spm.shape}")
            np.save(os.path.join(BOVW_SPM_FEATURES_DIR, f'X_test_orb_spm_L{PYRAMID_LEVELS-1}.npy'), X_test_orb_spm)
            print(f"Saved ORB test SPM features to {BOVW_SPM_FEATURES_DIR}")
        else:
            print("No ORB test SPM histograms were generated.")

    else:
        print(f"ORB KMeans model (for SPM) not found at {orb_kmeans_model_spm_file}. Skipping ORB SPM generation.")

    print("\n--- Phase 3: SPM Histogram Generation Complete ---")
    print(f"SPM features saved in: {BOVW_SPM_FEATURES_DIR}")