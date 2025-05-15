'''
SOH_SPM stands for "SIFT, ORB, HOG" - with SIFT and ORB modified for Spatial Pyramid Matching.
Extracts features, keypoint coordinates, and image dimensions from Places365.
Features will be saved in SPM-specific output directories.
This uses tfds fully for more efficient data handling.
'''

import cv2
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
from tqdm import tqdm
import pickle
import h5py
import pandas as pd # Added for saving subset index -> label map

# --- Configuration for SPM ---
TFDS_DATA_DIR = "E:\CV_imgs"
OUTPUT_FEATURES_SPM_DIR = "E:\CV_features_SPM"  # New main output directory for SPM related features
SUBSET_SIZE = 100000  # Should match vanilla if you want to compare on same subset
BATCH_SAVE_SIZE = 5000
RANDOM_SEED_FOR_SUBSET = 42 # Optional: for reproducibility of subset if shuffle_files=False for ds_train

# --- SPM Specific Output Directories ---
SIFT_BATCHES_SPM_DIR = os.path.join(OUTPUT_FEATURES_SPM_DIR, 'sift_batches_spm')
ORB_BATCHES_SPM_DIR = os.path.join(OUTPUT_FEATURES_SPM_DIR, 'orb_batches_spm')
HOG_DATA_SPM_FILE = os.path.join(OUTPUT_FEATURES_SPM_DIR, 'hog_data_spm.h5') # HOG data (if kept separate)

# Add a file to save the subset index -> label mapping
SUBSET_INDEX_LABEL_MAP_FILE = os.path.join(OUTPUT_FEATURES_SPM_DIR, f'subset_index_label_map_subset{SUBSET_SIZE}_seed{RANDOM_SEED_FOR_SUBSET}.csv') # Using CSV, include size and seed in name


os.makedirs(OUTPUT_FEATURES_SPM_DIR, exist_ok=True)
os.makedirs(SIFT_BATCHES_SPM_DIR, exist_ok=True)
os.makedirs(ORB_BATCHES_SPM_DIR, exist_ok=True)

# Function to extract features for SPM
def extract_features_tf_element_spm(index_tensor, img_tensor, label_tensor):
    '''Extracts SIFT (desc+coords), ORB (desc+coords), HOG features, and image dimensions.'''
    # Note: index_tensor here is the SUBSET index from .enumerate()
    # label_tensor here is the original label from TFDS
    try:
        img_np = img_tensor.numpy()
        label_np = label_tensor.numpy() # Keep original label for the map
        idx_np = index_tensor.numpy() # Keep the subset index

        # Ensure image is in the correct format (e.g., uint8)
        if img_np.dtype != np.uint8:
             img_np = img_np.astype(np.uint8)

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_height, img_width = gray.shape[:2]

        sift = cv2.SIFT_create()
        # Keep nfeatures consistent if comparing - tune this based on desired density/speed
        orb = cv2.ORB_create(nfeatures=1000)
        hog_win_size = (128, 128) # Assuming global HOG, same as before
        # Review HOG parameters if needed - default is good start
        hog = cv2.HOGDescriptor(_winSize=hog_win_size, _blockSize=(16,16), _blockStride=(8,8), _cellSize=(8,8), _nbins=9)

        # SIFT features and coordinates
        keypoints_sift_cv, descriptors_sift = sift.detectAndCompute(gray, None)
        sift_coords_np = np.array([kp.pt for kp in keypoints_sift_cv], dtype=np.float32) if keypoints_sift_cv else np.empty((0, 2), dtype=np.float32)
        if descriptors_sift is None: descriptors_sift = np.empty((0, 128), dtype=np.float32)

        # ORB features and coordinates
        keypoints_orb_cv, descriptors_orb = orb.detectAndCompute(gray, None)
        orb_coords_np = np.array([kp.pt for kp in keypoints_orb_cv], dtype=np.float32) if keypoints_orb_cv else np.empty((0, 2), dtype=np.float32)
        if descriptors_orb is None: descriptors_orb = np.empty((0, 32), dtype=np.uint8)

        # HOG feature (global)
        # Resize to match winSize is common when computing HOG over the whole image
        if gray.shape[:2] != hog_win_size:
             resized_for_hog = cv2.resize(gray, hog_win_size)
        else:
             resized_for_hog = gray # Avoid resizing if already correct size

        descriptor_hog_np = hog.compute(resized_for_hog)
        # Flatten the descriptor if it's not None. Handle size 0 case.
        descriptor_hog_np = descriptor_hog_np.flatten() if descriptor_hog_np is not None else np.array([], dtype=np.float32)
        if descriptor_hog_np.size == 0 : descriptor_hog_np = np.array([], dtype=np.float32) # Ensure it's an empty 1D array

        # Return subset index, original label, features, and dimensions
        return (idx_np, label_np,
                descriptors_sift, sift_coords_np,
                descriptors_orb, orb_coords_np,
                descriptor_hog_np,
                np.int32(img_width), np.int32(img_height))

    except Exception as e:
        # Get subset index and original label even on error if possible
        idx_np_err = index_tensor.numpy() if hasattr(index_tensor, 'numpy') else -1
        label_np_err = label_tensor.numpy() if hasattr(label_tensor, 'numpy') else -1
        print(f"Error processing subset index {idx_np_err} (Original Label: {label_np_err}) for SPM: {e}")
        # Return empty arrays with correct dtypes and expected number of items for Tout,
        # But return the subset index and original label captured before or after the error.
        # Returning -1 for label signifies extraction failure for downstream processes.
        return (idx_np_err, np.int64(-1),             # Return subset index, and -1 for label to signal error
                np.empty((0, 128), dtype=np.float32), # sift_desc
                np.empty((0, 2), dtype=np.float32),   # sift_coords
                np.empty((0, 32), dtype=np.uint8),    # orb_desc
                np.empty((0, 2), dtype=np.float32),   # orb_coords
                np.array([], dtype=np.float32),       # hog_desc
                np.int32(0), np.int32(0))             # img_width, img_height


def SOH_extract_SPM():
    print(f"--- Starting SIFT/ORB/HOG Feature Extraction for SPM ---")
    print(f"Output directory for SPM features: {OUTPUT_FEATURES_SPM_DIR}")
    print(f"Subset size: {SUBSET_SIZE}")
    print(f"Loading dataset from: {TFDS_DATA_DIR}")

    ds_train = tfds.load(
        'places365_small',
        split='train',
        data_dir=TFDS_DATA_DIR,
        # Crucial: Ensure deterministic subset by disabling file shuffling and using dataset shuffle with seed
        shuffle_files=False # Disable file shuffling to maintain consistent base order
        # download_and_prepare_kwargs={'download_dir': TFDS_DOWNLOAD_DIR} # If needed
    )
    # Apply deterministic shuffle WITH A SEED and then take the subset
    # buffer_size should be large enough, at least SUBSET_SIZE, preferably more.
    ds_subset_deterministic = ds_train.shuffle(buffer_size=SUBSET_SIZE*2, seed=RANDOM_SEED_FOR_SUBSET)
    ds_subset_taken_indexed = ds_subset_deterministic.take(SUBSET_SIZE).enumerate()


    print(f"Selected deterministic subset of {SUBSET_SIZE} images for SPM processing; adding indices.")
    print(f"Using subset random seed: {RANDOM_SEED_FOR_SUBSET}")
    print(f"Subset index to label map will be saved to: {SUBSET_INDEX_LABEL_MAP_FILE}")


    sift_batch_data_spm = {} # Keyed by subset index (idx_np)
    orb_batch_data_spm = {} # Keyed by subset index (idx_np)

    # Lists for HOG data and their corresponding subset indices and labels
    hog_features_list_spm = []
    labels_list_spm = []       # Corresponding original labels (from TFDS)
    indices_list_spm = []      # Corresponding subset indices (idx_np)

    # List to build the final subset index -> original label mapping
    subset_index_label_map_list = []

    processed_count = 0 # Counts images where extraction did NOT return label = -1

    print(f"Beginning feature extraction for SPM with tf.data parallelism...")

    # Define output types for tf.py_function - match the return signature
    tout_types = [
        tf.int64,    # idx (subset index)
        tf.int64,    # label (original or -1 on error)
        tf.float32,  # sift_desc
        tf.float32,  # sift_coords
        tf.uint8,    # orb_desc
        tf.float32,  # orb_coords
        tf.float32,  # hog_desc (1D)
        tf.int32,    # img_width
        tf.int32     # img_height
    ]

    ds_processed_spm = ds_subset_taken_indexed.map(
        lambda i, x: tf.py_function(
            func=extract_features_tf_element_spm,
            inp=[i, x['image'], x['label']], # Pass subset index, image tensor, original label tensor
            Tout=tout_types
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    print("Processing images for SPM features...")
    # Unpack results from the tf.py_function call
    for (idx, label, # These are the returned subset index and label (-1 on error)
         sift_desc, sift_coords,
         orb_desc, orb_coords,
         hog_desc,
         img_w, img_h) in tqdm(ds_processed_spm.as_numpy_iterator(), total=SUBSET_SIZE):

        # Always store the mapping from subset index to the label returned by the extraction function.
        # This includes indices where extraction might have failed (label=-1).
        # This list will be exactly SUBSET_SIZE long (unless iteration somehow breaks).
        subset_index_label_map_list.append({'subset_idx': idx, 'label': label})


        if label != -1: # Only store features if extraction was successful for this image
            # Store SIFT data for SPM - keyed by subset index
            if sift_desc.shape[0] > 0:
                sift_batch_data_spm[idx] = {
                    'descriptors': sift_desc,
                    'coordinates': sift_coords,
                    'width': img_w,
                    'height': img_h
                }

            # Store ORB data for SPM - keyed by subset index
            if orb_desc.shape[0] > 0:
                orb_batch_data_spm[idx] = {
                    'descriptors': orb_desc,
                    'coordinates': orb_coords,
                    'width': img_w,
                    'height': img_h
                }

            # Store Global HOG - in lists
            if hog_desc.size > 0 :
                hog_features_list_spm.append(hog_desc)
                labels_list_spm.append(label) # Store the label associated with this successful HOG extraction
                indices_list_spm.append(idx) # Store the subset index associated with this successful HOG extraction

            processed_count += 1 # Count images where extraction was successful

            # Save batches periodically
            if processed_count > 0 and processed_count % BATCH_SAVE_SIZE == 0:
                batch_num_actual = processed_count // BATCH_SAVE_SIZE
                print(f"\nSaving SPM feature batches up to {processed_count} processed images...")

                if sift_batch_data_spm:
                    # Save with a name reflecting the batch number based on *processed* images
                    sift_file_spm = os.path.join(SIFT_BATCHES_SPM_DIR, f'sift_spm_batch_{batch_num_actual-1}.pkl')
                    with open(sift_file_spm, 'wb') as f: pickle.dump(sift_batch_data_spm, f)
                    sift_batch_data_spm = {} # Reset for next batch
                    #print(f"Saved batch to {sift_file_spm}")

                if orb_batch_data_spm:
                    orb_file_spm = os.path.join(ORB_BATCHES_SPM_DIR, f'orb_spm_batch_{batch_num_actual-1}.pkl')
                    with open(orb_file_spm, 'wb') as f: pickle.dump(orb_batch_data_spm, f)
                    orb_batch_data_spm = {} # Reset for next batch
                    #print(f"Saved batch to {orb_file_spm}")
        # Note: Error messages for label == -1 are printed inside extract_features_tf_element_spm

    # --- Save any remaining data in the last batch ---
    print(f"\nFeature extraction for SPM complete. Processed {processed_count} successful images.")
    # Calculate the batch number for the final batch based on total processed count
    final_batch_num_processed = (processed_count + BATCH_SAVE_SIZE -1) // BATCH_SAVE_SIZE if processed_count > 0 else 0
    # Use a distinct index for the final batch if it contains data less than BATCH_SAVE_SIZE
    # Or simply use the actual count or a fixed "final" name if processed_count > 0
    # Let's use the processed_count in the filename for uniqueness of the final batch
    if sift_batch_data_spm:
        print(f"Saving final SIFT SPM batch ({len(sift_batch_data_spm)} items)...")
        sift_file_spm = os.path.join(SIFT_BATCHES_SPM_DIR, f'sift_spm_final_batch_processed{processed_count}.pkl')
        with open(sift_file_spm, 'wb') as f: pickle.dump(sift_batch_data_spm, f)
        print(f"Saved final batch to {sift_file_spm}")

    if orb_batch_data_spm:
        print(f"Saving final ORB SPM batch ({len(orb_batch_data_spm)} items)...")
        orb_file_spm = os.path.join(ORB_BATCHES_SPM_DIR, f'orb_spm_final_batch_processed{processed_count}.pkl')
        with open(orb_file_spm, 'wb') as f: pickle.dump(orb_batch_data_spm, f)
        print(f"Saved final batch to {orb_file_spm}")


    # --- Save HOG data ---
    # The lists already contain data ONLY for images where extraction was successful (label != -1)
    if hog_features_list_spm:
        # Robust stacking for potentially mixed empty/non-empty HOG descriptors
        if any(h.size == 0 for h in hog_features_list_spm):
            max_len = 0
            non_empty_hogs = [h for h in hog_features_list_spm if h.size > 0]
            if non_empty_hogs: max_len = non_empty_hogs[0].shape[0]
            padded_hogs = [h if h.size > 0 else np.zeros(max_len, dtype=np.float32) for h in hog_features_list_spm]
            hog_array_spm = np.vstack(padded_hogs) if padded_hogs else np.empty((0, max_len if max_len > 0 else 0), dtype=np.float32) # Handle case where all are empty
        else:
             hog_array_spm = np.vstack(hog_features_list_spm)

        labels_array_spm = np.array(labels_list_spm)     # Labels for successful HOG extractions
        indices_array_spm = np.array(indices_list_spm) # Subset indices for successful HOG extractions

        if hog_array_spm.size > 0 :
            try:
                with h5py.File(HOG_DATA_SPM_FILE, 'w') as hf:
                    hf.create_dataset('hog_features', data=hog_array_spm)
                    hf.create_dataset('labels', data=labels_array_spm) # These labels align with the HOG features and indices in this H5
                    hf.create_dataset('indices', data=indices_array_spm) # These are the subset indices
                print(f"Saved HOG SPM data (aligned by subset index) to: {HOG_DATA_SPM_FILE}")
                print(f"  HOG shape: {hog_array_spm.shape}")
                print(f"  Labels shape: {labels_array_spm.shape}")
                print(f"  Indices shape: {indices_array_spm.shape}")
            except Exception as e:
                 print(f"ERROR saving HOG SPM data to HDF5: {e}")
        else:
            print("No valid HOG features collected to save for SPM.")
    else:
        print("No HOG features were collected for SPM.")


    # --- Save the subset index to original label mapping ---
    # This map contains an entry for *every* image in the subset (SUBSET_SIZE total),
    # mapping its subset index to the label returned by the extraction function (-1 if failed).
    print(f"Saving subset index to label mapping to: {SUBSET_INDEX_LABEL_MAP_FILE}")
    try:
        map_df = pd.DataFrame(subset_index_label_map_list)
        # Sort by subset_idx for consistency, although not strictly necessary for DataFrame lookup
        map_df = map_df.sort_values(by='subset_idx')
        map_df.to_csv(SUBSET_INDEX_LABEL_MAP_FILE, index=False)
        print(f"Subset index to label mapping saved successfully. Total entries: {len(map_df)}")
        if len(map_df) != SUBSET_SIZE:
             print(f"WARNING: Mismatch in map entries ({len(map_df)}) and expected subset size ({SUBSET_SIZE}).")
    except Exception as e:
        print(f"ERROR saving subset index to label mapping: {e}")


    print(f"\nSPM feature data (descriptors, coords, dims) and subset index map saved in: {OUTPUT_FEATURES_SPM_DIR}")