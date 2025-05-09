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

os.makedirs(OUTPUT_FEATURES_SPM_DIR, exist_ok=True)
os.makedirs(SIFT_BATCHES_SPM_DIR, exist_ok=True)
os.makedirs(ORB_BATCHES_SPM_DIR, exist_ok=True)

# Function to extract features for SPM
def extract_features_tf_element_spm(index_tensor, img_tensor, label_tensor):
    '''Extracts SIFT (desc+coords), ORB (desc+coords), HOG features, and image dimensions.'''
    try:
        img_np = img_tensor.numpy()
        label_np = label_tensor.numpy()
        idx_np = index_tensor.numpy()

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_height, img_width = gray.shape[:2]

        sift = cv2.SIFT_create()
        orb = cv2.ORB_create(nfeatures=1000) # Keep nfeatures consistent if comparing
        hog_win_size = (128, 128) # Assuming global HOG, same as before
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
        resized_for_hog = cv2.resize(gray, hog_win_size) # Assuming HOG descriptor window size matches resized image
        descriptor_hog_np = hog.compute(resized_for_hog)
        descriptor_hog_np = descriptor_hog_np.flatten() if descriptor_hog_np is not None else np.array([], dtype=np.float32)
        if descriptor_hog_np.size == 0 : descriptor_hog_np = np.array([], dtype=np.float32) # Ensure it's an empty 1D array not None

        return (idx_np, label_np,
                descriptors_sift, sift_coords_np,
                descriptors_orb, orb_coords_np,
                descriptor_hog_np,
                np.int32(img_width), np.int32(img_height))

    except Exception as e:
        idx_np_err = index_tensor.numpy() if hasattr(index_tensor, 'numpy') else -1 # Fallback if tensor conversion fails early
        print(f"Error processing index {idx_np_err} for SPM: {e}")
        # Return empty arrays with correct dtypes and expected number of items for Tout
        return (idx_np_err, np.int64(-1),             # idx, label
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

    # ds_train, ds_info = tfds.load( # Load with_info if you need original class names from here
    ds_train = tfds.load(
        'places365_small',
        split='train',
        data_dir=TFDS_DATA_DIR,
        shuffle_files=True, # Shuffle before take for a more random subset
        # download_and_prepare_kwargs={'download_dir': TFDS_DOWNLOAD_DIR} # If needed
    )

    # Take a subset and enumerate to get indices
    # If shuffle_files=True, take() gives a random subset.
    # If shuffle_files=False, add .shuffle(buffer_size, seed=RANDOM_SEED_FOR_SUBSET) before .take() for reproducible subset.
    ds_subset_indexed = ds_train.take(SUBSET_SIZE).enumerate()
    print(f"Selected subset of {SUBSET_SIZE} images for SPM processing; adding indices.")

    sift_batch_data_spm = {}
    orb_batch_data_spm = {}
    hog_features_list_spm = [] # For global HOG
    labels_list_spm = []       # Corresponding labels for HOG
    indices_list_spm = []      # Corresponding original indices for HOG

    processed_count = 0

    print(f"Beginning feature extraction for SPM with tf.data parallelism...")

    # Define output types and shapes for tf.py_function
    # For variable length outputs (like descriptors and coords), shape is None for that dimension
    tout_types = [
        tf.int64, tf.int64,      # idx, label
        tf.float32, tf.float32,  # sift_desc, sift_coords
        tf.uint8, tf.float32,    # orb_desc, orb_coords
        tf.float32,              # hog_desc (1D)
        tf.int32, tf.int32       # img_width, img_height
    ]
    # For tf.TensorSpec, if using:
    # tout_specs = [
    #     tf.TensorSpec(shape=(), dtype=tf.int64),
    #     tf.TensorSpec(shape=(), dtype=tf.int64),
    #     tf.TensorSpec(shape=(None, 128), dtype=tf.float32), # SIFT descriptors
    #     tf.TensorSpec(shape=(None, 2), dtype=tf.float32),   # SIFT coordinates
    #     tf.TensorSpec(shape=(None, 32), dtype=tf.uint8),    # ORB descriptors
    #     tf.TensorSpec(shape=(None, 2), dtype=tf.float32),   # ORB coordinates
    #     tf.TensorSpec(shape=(None,), dtype=tf.float32),     # HOG descriptor (flattened)
    #     tf.TensorSpec(shape=(), dtype=tf.int32),
    #     tf.TensorSpec(shape=(), dtype=tf.int32),
    # ]


    ds_processed_spm = ds_subset_indexed.map(
        lambda i, x: tf.py_function(
            func=extract_features_tf_element_spm,
            inp=[i, x['image'], x['label']],
            Tout=tout_types # Use the list of tf dtypes
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_processed_spm = ds_processed_spm.prefetch(buffer_size=tf.data.AUTOTUNE)

    print("Processing images for SPM features...")
    for (idx, label,
         sift_desc, sift_coords,
         orb_desc, orb_coords,
         hog_desc,
         img_w, img_h) in tqdm(ds_processed_spm.as_numpy_iterator(), total=SUBSET_SIZE):

        if label != -1: # Process only if feature extraction was successful (label != -1)
            # Store SIFT data for SPM
            if sift_desc.shape[0] > 0: # Only if descriptors were found
                sift_batch_data_spm[idx] = {
                    'descriptors': sift_desc,
                    'coordinates': sift_coords,
                    'width': img_w,
                    'height': img_h
                }

            # Store ORB data for SPM
            if orb_desc.shape[0] > 0: # Only if descriptors were found
                orb_batch_data_spm[idx] = {
                    'descriptors': orb_desc,
                    'coordinates': orb_coords,
                    'width': img_w,
                    'height': img_h
                }
            
            # Store Global HOG (same as before, but use SPM lists)
            if hog_desc.size > 0 : # Check if HOG descriptor is not empty
                hog_features_list_spm.append(hog_desc)
                labels_list_spm.append(label)
                indices_list_spm.append(idx)

            processed_count += 1

            if processed_count > 0 and processed_count % BATCH_SAVE_SIZE == 0:
                batch_num_actual = processed_count // BATCH_SAVE_SIZE # More robust batch numbering
                print(f"\nSaving SPM batch {batch_num_actual}...")
                if sift_batch_data_spm:
                    sift_file_spm = os.path.join(SIFT_BATCHES_SPM_DIR, f'sift_spm_batch_{batch_num_actual-1}.pkl')
                    with open(sift_file_spm, 'wb') as f: pickle.dump(sift_batch_data_spm, f)
                    sift_batch_data_spm = {}
                
                if orb_batch_data_spm:
                    orb_file_spm = os.path.join(ORB_BATCHES_SPM_DIR, f'orb_spm_batch_{batch_num_actual-1}.pkl')
                    with open(orb_file_spm, 'wb') as f: pickle.dump(orb_batch_data_spm, f)
                    orb_batch_data_spm = {}
                # batch_num += 1 # Not needed if using batch_num_actual
        else:
            print(f"Skipping index {idx} due to feature extraction error (label -1).")


    # Save any remaining data in the last batch
    final_batch_num = (processed_count + BATCH_SAVE_SIZE -1) // BATCH_SAVE_SIZE if processed_count > 0 else 0
    print(f"\nFeature extraction for SPM complete. Processed {processed_count} images.")
    if sift_batch_data_spm:
        print(f"Saving final SIFT SPM batch (approx {final_batch_num})...")
        sift_file_spm = os.path.join(SIFT_BATCHES_SPM_DIR, f'sift_spm_batch_{final_batch_num-1 if final_batch_num > 0 else 0}.pkl')
        with open(sift_file_spm, 'wb') as f: pickle.dump(sift_batch_data_spm, f)

    if orb_batch_data_spm:
        print(f"Saving final ORB SPM batch (approx {final_batch_num})...")
        orb_file_spm = os.path.join(ORB_BATCHES_SPM_DIR, f'orb_spm_batch_{final_batch_num-1 if final_batch_num > 0 else 0}.pkl')
        with open(orb_file_spm, 'wb') as f: pickle.dump(orb_batch_data_spm, f)

    # Save HOG data (same structure as before, but to SPM directory)
    if hog_features_list_spm:
        hog_array_spm = np.array(hog_features_list_spm) # Use vstack if hog_desc is guaranteed to be 1D and non-empty
        # Robust stacking for potentially mixed empty/non-empty HOG descriptors
        if any(h.size == 0 for h in hog_features_list_spm):
            max_len = 0
            if hog_features_list_spm: # Check if list is not empty
                non_empty_hogs = [h for h in hog_features_list_spm if h.size > 0]
                if non_empty_hogs: # Check if there's at least one non-empty HOG
                     max_len = non_empty_hogs[0].shape[0] # Assume all non-empty HOGs have same length
            
            padded_hogs = []
            for h_desc in hog_features_list_spm:
                if h_desc.size > 0:
                    padded_hogs.append(h_desc)
                else: # Append zeros if HOG descriptor was empty
                    padded_hogs.append(np.zeros(max_len, dtype=np.float32) if max_len > 0 else np.array([], dtype=np.float32))
            if padded_hogs:
                 hog_array_spm = np.vstack(padded_hogs)
            else: # All HOGs were empty
                 hog_array_spm = np.empty((0,0), dtype=np.float32) # Adjust based on expected HOG dim if all are empty
        else:
             hog_array_spm = np.vstack(hog_features_list_spm)


        labels_array_spm = np.array(labels_list_spm)
        indices_array_spm = np.array(indices_list_spm)

        if hog_array_spm.size > 0 : # Only save if there's actual HOG data
            with h5py.File(HOG_DATA_SPM_FILE, 'w') as hf:
                hf.create_dataset('hog_features', data=hog_array_spm)
                hf.create_dataset('labels', data=labels_array_spm)
                hf.create_dataset('indices', data=indices_array_spm)
            print(f"Saved HOG SPM data to: {HOG_DATA_SPM_FILE}")
            print(f"  HOG shape: {hog_array_spm.shape}")
            print(f"  Labels shape: {labels_array_spm.shape}")
            print(f"  Indices shape: {indices_array_spm.shape}")
        else:
            print("No valid HOG features collected to save for SPM.")
    else:
        print("No HOG features were collected for SPM.")

    print(f"\nSPM feature data (descriptors, coords, dims) saved in: {OUTPUT_FEATURES_SPM_DIR}")
