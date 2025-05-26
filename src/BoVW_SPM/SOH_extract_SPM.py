import cv2
import numpy as np
import os
from tqdm import tqdm # Ensure tqdm is imported
import pickle
import h5py

def extract_features_from_image_path_spm(image_path, sift_detector, orb_detector, hog_detector, hog_win_size):
    '''Extracts SIFT (desc+coords), ORB (desc+coords), HOG features, and image dimensions from an image file.'''
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Error: Could not read image at {image_path}")
            return None

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_height, img_width = gray.shape[:2]

        keypoints_sift_cv, descriptors_sift = sift_detector.detectAndCompute(gray, None)
        sift_coords_np = np.array([kp.pt for kp in keypoints_sift_cv], dtype=np.float32) if keypoints_sift_cv else np.empty((0, 2), dtype=np.float32)
        if descriptors_sift is None: descriptors_sift = np.empty((0, 128), dtype=np.float32)

        keypoints_orb_cv, descriptors_orb = orb_detector.detectAndCompute(gray, None)
        orb_coords_np = np.array([kp.pt for kp in keypoints_orb_cv], dtype=np.float32) if keypoints_orb_cv else np.empty((0, 2), dtype=np.float32)
        if descriptors_orb is None: descriptors_orb = np.empty((0, 32), dtype=np.uint8)

        if gray.shape[:2] != hog_win_size:
            resized_for_hog = cv2.resize(gray, hog_win_size)
        else:
            resized_for_hog = gray
        descriptor_hog_np = hog_detector.compute(resized_for_hog)
        descriptor_hog_np = descriptor_hog_np.flatten() if descriptor_hog_np is not None else np.array([], dtype=np.float32)
        if descriptor_hog_np.size == 0 : descriptor_hog_np = np.array([], dtype=np.float32)


        return (image_path,
                descriptors_sift, sift_coords_np,
                descriptors_orb, orb_coords_np,
                descriptor_hog_np,
                np.int32(img_width), np.int32(img_height))

    except Exception as e:
        print(f"Error processing image {image_path} for SPM: {e}")
        return None

def save_hog_batch_to_hdf5(hog_features_list, hog_paths_list, hog_labels_list, set_name, batch_num_str, base_dir):
    """Saves a batch of HOG data to an HDF5 file."""
    if not hog_features_list:
        print(f"No HOG features to save for {set_name} batch {batch_num_str}.")
        return

    # HOG descriptor length for (128,128) window, (16,16) block, (8,8) stride, (8,8) cell, 9 bins
    # ( (128-16)/8 + 1 ) * ( (128-16)/8 + 1 ) * (16/8) * (16/8) * 9 = 15 * 15 * 4 * 9 = 8100
    # However, OpenCV's HOGDescriptor.compute() for a single (128,128) image returns 3780 features when flattened.
    # This depends on default block normalization scheme. Let's use the observed value.
    # If you change HOG parameters, this MUST be updated.
    # A more robust way is hog_detector.getDescriptorSize() if it's available and gives the correct flattened size.
    EXPECTED_HOG_LEN = 8100 # Update if your HOG parameters yield a different size.

    max_len_hog = 0
    non_empty_hogs = [h for h in hog_features_list if h.size > 0]

    if non_empty_hogs:
        max_len_hog = non_empty_hogs[0].shape[0]
        if max_len_hog != EXPECTED_HOG_LEN: # Check consistency
             print(f"Warning: Observed HOG length {max_len_hog} differs from expected {EXPECTED_HOG_LEN} for batch {batch_num_str}. Using observed length for this batch.")
    else: # All HOGs in this batch were empty or had inconsistent non-zero size
        print(f"Warning: All HOG descriptors in {set_name} batch {batch_num_str} are empty or inconsistent. Using expected HOG length {EXPECTED_HOG_LEN} for padding.")
        max_len_hog = EXPECTED_HOG_LEN


    padded_hogs = []
    for h_idx, h in enumerate(hog_features_list):
        if h.size == max_len_hog:
            padded_hogs.append(h)
        elif h.size == 0:
            padded_hogs.append(np.zeros(max_len_hog, dtype=np.float32))
        else:
            print(f"Warning: Inconsistent HOG descriptor length ({h.shape[0] if h.size > 0 else 0} vs {max_len_hog}) for path {hog_paths_list[h_idx]} in batch {batch_num_str}. Padding/truncating to {max_len_hog}.")
            temp_h = np.zeros(max_len_hog, dtype=np.float32)
            copy_len = min(h.shape[0] if h.size > 0 else 0, max_len_hog)
            if copy_len > 0: temp_h[:copy_len] = h[:copy_len]
            padded_hogs.append(temp_h)

    if not padded_hogs: # Should not happen if hog_features_list was not empty
        print(f"No HOG features (after attempting padding) to save for {set_name} batch {batch_num_str}.")
        return

    hog_array_batch = np.vstack(padded_hogs)
    hog_labels_array_batch = np.array(hog_labels_list, dtype=np.int8)

    if hog_array_batch.size > 0:
        hog_batch_file = os.path.join(base_dir, f'hog_spm_{set_name}_batch_{batch_num_str}.h5')
        try:
            with h5py.File(hog_batch_file, 'w') as hf:
                hf.create_dataset('hog_features', data=hog_array_batch, compression="gzip")
                hf.create_dataset('labels_numeric', data=hog_labels_array_batch)
                dt = h5py.string_dtype(encoding='utf-8')
                hf.create_dataset('image_paths', data=np.array(hog_paths_list, dtype=dt))
            print(f"Saved HOG SPM batch to: {hog_batch_file} (Features: {hog_array_batch.shape})")
        except Exception as e:
            print(f"ERROR saving HOG SPM batch {batch_num_str} for {set_name} to HDF5: {e}")
    else:
        print(f"No HOG features with positive size collected to save for {set_name} batch {batch_num_str}.")


def SOH_extract_SPM_from_balanced_split(image_paths_to_process, corresponding_labels, BATCH_SAVE_SIZE, ORB_BATCHES_SPM_DIR=None, SIFT_BATCHES_SPM_DIR=None, HOG_BATCHES_SPM_DIR=None, set_name="train"):
    print(f"\n--- Starting SIFT/ORB/HOG Feature Extraction for SPM ({set_name} set) ---")
    print(f"Processing {len(image_paths_to_process)} images.")
    # ... (print output directories)

    sift_detector = cv2.SIFT_create()
    orb_detector = cv2.ORB_create(nfeatures=1000)
    hog_win_size_config = (128, 128)
    hog_detector = cv2.HOGDescriptor(_winSize=hog_win_size_config,
                                     _blockSize=(16,16),
                                     _blockStride=(8,8),
                                     _cellSize=(8,8),
                                     _nbins=9)
    # You can get the expected descriptor size once here:
    # global EXPECTED_HOG_LEN_FROM_DETECTOR # if you want to make it global for save_hog_batch_to_hdf5
    # EXPECTED_HOG_LEN_FROM_DETECTOR = hog_detector.getDescriptorSize()
    # print(f"Expected HOG descriptor size: {EXPECTED_HOG_LEN_FROM_DETECTOR}") # This should be 3780

    sift_batch_data = {}
    orb_batch_data = {}
    hog_batch_features = []
    hog_batch_paths = []
    hog_batch_labels = []

    processed_in_set_count = 0
    processed_in_batch_count = 0
    current_batch_num = 1

    # Assign tqdm iterator to a variable
    progress_bar = tqdm(iterable=image_paths_to_process,
                        desc=f"Processing {set_name} images (Batch {current_batch_num})",
                        total=len(image_paths_to_process), # Explicitly set total for better display
                        unit="img")

    for i, image_path in enumerate(progress_bar): # Iterate over the progress_bar instance
        label_for_image = corresponding_labels[i]
        extraction_result = extract_features_from_image_path_spm(image_path, sift_detector, orb_detector, hog_detector, hog_win_size_config)

        if extraction_result is not None:
            (img_path_ret, sift_desc, sift_coords, orb_desc, orb_coords, hog_desc, img_w, img_h) = extraction_result

            if sift_desc.shape[0] > 0:
                sift_batch_data[image_path] = {
                    'descriptors': sift_desc, 'coordinates': sift_coords,
                    'width': img_w, 'height': img_h, 'label': label_for_image
                }
            if orb_desc.shape[0] > 0:
                orb_batch_data[image_path] = {
                    'descriptors': orb_desc, 'coordinates': orb_coords,
                    'width': img_w, 'height': img_h, 'label': label_for_image
                }
            hog_batch_features.append(hog_desc)
            hog_batch_paths.append(image_path)
            hog_batch_labels.append(label_for_image)

            processed_in_set_count += 1
            processed_in_batch_count += 1

            if processed_in_batch_count >= BATCH_SAVE_SIZE:
                # Temporarily clear description for saving messages
                original_desc = progress_bar.desc
                progress_bar.set_description("Saving batch...")
                progress_bar.refresh() # Force update

                print(f"\nSaving batch {current_batch_num} for {set_name} set ({processed_in_batch_count} images in this batch)...")
                if sift_batch_data:
                    sift_file = os.path.join(SIFT_BATCHES_SPM_DIR, f'sift_spm_{set_name}_batch_{current_batch_num}.pkl')
                    with open(sift_file, 'wb') as f: pickle.dump(sift_batch_data, f)
                    print(f"Saved SIFT batch to {sift_file}")
                    sift_batch_data = {}
                if orb_batch_data:
                    orb_file = os.path.join(ORB_BATCHES_SPM_DIR, f'orb_spm_{set_name}_batch_{current_batch_num}.pkl')
                    with open(orb_file, 'wb') as f: pickle.dump(orb_batch_data, f)
                    print(f"Saved ORB batch to {orb_file}")
                    orb_batch_data = {}

                save_hog_batch_to_hdf5(hog_batch_features, hog_batch_paths, hog_batch_labels, set_name, str(current_batch_num), HOG_BATCHES_SPM_DIR)
                hog_batch_features, hog_batch_paths, hog_batch_labels = [], [], []

                current_batch_num += 1
                processed_in_batch_count = 0

                # Update tqdm description for the new batch, only if not the end of processing
                if i + 1 < len(image_paths_to_process):
                    progress_bar.set_description(f"Processing {set_name} images (Batch {current_batch_num})", refresh=True)
                else: # Restore original or set to complete if it was the last item
                    progress_bar.set_description(original_desc if original_desc else "Processing complete", refresh=True)


    # --- Save any remaining data in the final (potentially partial) batch ---
    if processed_in_batch_count > 0:
        progress_bar.set_description("Saving final batch...", refresh=True)
        print(f"\nSaving final (partial) batch {current_batch_num} for {set_name} set ({processed_in_batch_count} images)...")
        if sift_batch_data:
            sift_file = os.path.join(SIFT_BATCHES_SPM_DIR, f'sift_spm_{set_name}_batch_{current_batch_num}_final.pkl')
            with open(sift_file, 'wb') as f: pickle.dump(sift_batch_data, f)
            print(f"Saved final SIFT batch to {sift_file}")
        if orb_batch_data:
            orb_file = os.path.join(ORB_BATCHES_SPM_DIR, f'orb_spm_{set_name}_batch_{current_batch_num}_final.pkl')
            with open(orb_file, 'wb') as f: pickle.dump(orb_batch_data, f)
            print(f"Saved final ORB batch to {orb_file}")

        save_hog_batch_to_hdf5(hog_batch_features, hog_batch_paths, hog_batch_labels, set_name, f"{current_batch_num}_final", HOG_BATCHES_SPM_DIR)
    elif processed_in_set_count == 0:
        print(f"No images were successfully processed for the {set_name} set.")

    progress_bar.close() # Important to close the tqdm bar

    print(f"\n--- Feature extraction for {set_name} set complete. Processed {processed_in_set_count} successful images. ---")
    # ... (print final messages)

def main_spm_feature_extraction(BALANCED_SPLIT_NPZ_FILE, BATCH_SAVE_SIZE, SIFT_BATCHES_SPM_DIR, ORB_BATCHES_SPM_DIR, HOG_BATCHES_SPM_DIR):
    print("--- Main SPM Feature Extraction from Balanced Split ---")

    if not os.path.exists(BALANCED_SPLIT_NPZ_FILE):
        print(f"ERROR: Balanced split NPZ file not found at {BALANCED_SPLIT_NPZ_FILE}")
        return

    print(f"Loading image paths and labels from: {BALANCED_SPLIT_NPZ_FILE}")
    try:
        data_npz = np.load(BALANCED_SPLIT_NPZ_FILE, allow_pickle=True)
        train_image_paths = data_npz['train_image_paths']
        train_labels_numeric = data_npz['train_labels_numeric']
        test_image_paths = data_npz['test_image_paths']
        test_labels_numeric = data_npz['test_labels_numeric']
        print(f"Loaded {len(train_image_paths)} train paths and {len(test_image_paths)} test paths.")
    except Exception as e:
        print(f"Error loading NPZ file: {e}")
        return
    SOH_extract_SPM_from_balanced_split(train_image_paths, train_labels_numeric, BATCH_SAVE_SIZE, ORB_BATCHES_SPM_DIR, SIFT_BATCHES_SPM_DIR, HOG_BATCHES_SPM_DIR, set_name="train")
    SOH_extract_SPM_from_balanced_split(test_image_paths, test_labels_numeric, BATCH_SAVE_SIZE, ORB_BATCHES_SPM_DIR, SIFT_BATCHES_SPM_DIR, HOG_BATCHES_SPM_DIR, set_name="test")

    print("\n--- All SPM Feature Extraction from Balanced Split Complete ---")
