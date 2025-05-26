# SOH_extract_vanilla_balanced.py
import cv2
import numpy as np
import os
from tqdm import tqdm
import pickle
import h5py
import gc

# --- Configuration for Vanilla BoVW Feature Extraction from BALANCED DATA ---

# Input: NPZ file from your balanced BoVW data preparation script (create_balanced_split_for_bovw.py)
# ADJUST THE FILENAME to match the output of your create_balanced_split_for_bovw.py
# Example: bovw_train_test_paths_N8000_S42.npz (N=total images, S=seed)
BALANCED_SPLIT_NPZ_FILE = r"E:\CV\bovw_splits_balanced\bovw_train_test_paths_N100000_S42.npz"

# Output directory for these "vanilla" BoVW features (descriptors and HOG)
OUTPUT_VANILLA_FEATURES_DIR = r"E:\CV\features_BoVW_Vanilla_balanced" # New specific dir

BATCH_SAVE_SIZE = 1000  # Number of images to process before saving a SIFT/ORB descriptor batch

# --- Output Subdirectories ---
SIFT_DESCRIPTORS_BATCHES_DIR = os.path.join(OUTPUT_VANILLA_FEATURES_DIR, 'sift_descriptors_batches')
ORB_DESCRIPTORS_BATCHES_DIR = os.path.join(OUTPUT_VANILLA_FEATURES_DIR, 'orb_descriptors_batches')
# HOG will be saved as HDF5 directly in OUTPUT_VANILLA_FEATURES_DIR

def ensure_directories():
    os.makedirs(OUTPUT_VANILLA_FEATURES_DIR, exist_ok=True)
    os.makedirs(SIFT_DESCRIPTORS_BATCHES_DIR, exist_ok=True)
    os.makedirs(ORB_DESCRIPTORS_BATCHES_DIR, exist_ok=True)

# Function to extract features from a single image file path for Vanilla BoVW
def extract_vanilla_features_from_image_path(image_path):
    """
    Extracts SIFT descriptors, ORB descriptors, and global HOG features from an image file.
    For vanilla BoVW, keypoint coordinates for SIFT/ORB are not typically stored.
    """
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Warning: Could not read image at {image_path}. Skipping.")
            return None # Signal error

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        orb = cv2.ORB_create(nfeatures=1000) # Number of features for ORB
        hog_win_size = (128, 128) # Standard HOG window size
        hog = cv2.HOGDescriptor(_winSize=hog_win_size, _blockSize=(16,16), _blockStride=(8,8), _cellSize=(8,8), _nbins=9)

        # SIFT descriptors (keypoints not needed for basic BoVW histogram)
        _, descriptors_sift = sift.detectAndCompute(gray, None)
        if descriptors_sift is None: descriptors_sift = np.array([], dtype=np.float32).reshape(0, 128)

        # ORB descriptors
        _, descriptors_orb = orb.detectAndCompute(gray, None)
        if descriptors_orb is None: descriptors_orb = np.array([], dtype=np.uint8).reshape(0, 32)

        # Global HOG feature
        # Resize to match HOG window size before computing
        if gray.shape[:2] != hog_win_size:
            resized_for_hog = cv2.resize(gray, hog_win_size)
        else:
            resized_for_hog = gray
        descriptor_hog = hog.compute(resized_for_hog)
        descriptor_hog = descriptor_hog.flatten() if descriptor_hog is not None else np.array([], dtype=np.float32)
        if descriptor_hog.size == 0: descriptor_hog = np.array([], dtype=np.float32) # Ensure 1D empty

        return (image_path, descriptors_sift, descriptors_orb, descriptor_hog)

    except Exception as e:
        print(f"Error processing image {image_path} for vanilla BoVW features: {e}")
        return None # Signal error

def process_image_set_for_vanilla_bovw(image_paths_for_set, corresponding_labels_for_set, set_name):
    """
    Processes a list of image paths (train or test) to extract and save SOH features for vanilla BoVW.
    Args:
        image_paths_for_set (list): List of full paths to images.
        corresponding_labels_for_set (list): List of numeric broad category labels.
        set_name (str): "train" or "test", used for naming output files.
    """
    print(f"\n--- Starting Vanilla BoVW Feature Extraction for {set_name.upper()} SET ---")
    print(f"Processing {len(image_paths_for_set)} images.")

    # For SIFT/ORB, we'll save batches of descriptors. Each batch file will be a list of descriptor arrays.
    sift_descriptors_current_batch_list = []
    orb_descriptors_current_batch_list = []
    # We also need to save corresponding labels and image_paths for these batched descriptors
    # if we want to easily recombine them later or for traceability.
    # For simplicity in BoVW vocab building, often just the descriptors are collected.
    # Let's store descriptors directly in lists for batching.

    # For HOG, collect all features, labels, and paths for this set
    hog_features_list_all = []
    hog_labels_list_all = []
    hog_image_paths_list_all = [] # Store original image paths for HOG

    processed_count_for_batching = 0
    batch_num_sift = 0
    batch_num_orb = 0

    for i, image_path in enumerate(tqdm(image_paths_for_set, desc=f"Extracting Vanilla BoVW features ({set_name})")):
        label_for_image = corresponding_labels_for_set[i]
        extraction_result = extract_vanilla_features_from_image_path(image_path)

        if extraction_result is not None:
            (_img_path_ret, sift_desc, orb_desc, hog_desc) = extraction_result

            # Add SIFT descriptors AND LABEL to current batch list
            if sift_desc.shape[0] > 0:
                sift_descriptors_current_batch_list.append((sift_desc, label_for_image)) # Store as tuple

            # Add ORB descriptors AND LABEL to current batch list
            if orb_desc.shape[0] > 0:
                orb_descriptors_current_batch_list.append((orb_desc, label_for_image)) # Store as tuple

            # Store HOG features directly
            if hog_desc.size > 0:
                hog_features_list_all.append(hog_desc)
                hog_labels_list_all.append(label_for_image)
                hog_image_paths_list_all.append(image_path)
            # Else: if HOG is empty, we might still want a placeholder or to note it.
            # For now, only non-empty HOGs are added. This means HOG array might be shorter
            # than total images if some images yield no HOG. This needs care in alignment.
            # Better: Always add HOG, even if it's zeros of expected length.
            # Assuming hog_desc is always 1D (flattened or empty 1D)
            # For consistency, let's assume if hog_desc.size == 0, it means an empty feature vector
            # that we should still record if we want HOG array to match input image count.
            # However, extract_vanilla_features_from_image_path returns empty np.array for empty HOG.
            # If a fixed-length HOG is desired for failed extractions, it should be handled there.
            # The current code only appends non-empty HOG. This is fine if downstream handles it.


            processed_count_for_batching += 1

            # Save SIFT descriptor batch if BATCH_SAVE_SIZE is reached
            if len(sift_descriptors_current_batch_list) >= BATCH_SAVE_SIZE:
                sift_batch_filename = os.path.join(SIFT_DESCRIPTORS_BATCHES_DIR, f'sift_descriptors_{set_name}_batch_{batch_num_sift}.pkl')
                with open(sift_batch_filename, 'wb') as f_sift:
                    pickle.dump(sift_descriptors_current_batch_list, f_sift)
                tqdm.write(f"Saved SIFT descriptor batch: {sift_batch_filename}")
                sift_descriptors_current_batch_list = [] # Reset
                batch_num_sift += 1

            # Save ORB descriptor batch
            if len(orb_descriptors_current_batch_list) >= BATCH_SAVE_SIZE:
                orb_batch_filename = os.path.join(ORB_DESCRIPTORS_BATCHES_DIR, f'orb_descriptors_{set_name}_batch_{batch_num_orb}.pkl')
                with open(orb_batch_filename, 'wb') as f_orb:
                    pickle.dump(orb_descriptors_current_batch_list, f_orb)
                tqdm.write(f"Saved ORB descriptor batch: {orb_batch_filename}")
                orb_descriptors_current_batch_list = [] # Reset
                batch_num_orb += 1
        # else: extraction_result was None, error already printed

    # Save any remaining SIFT descriptors
    if sift_descriptors_current_batch_list:
        sift_batch_filename = os.path.join(SIFT_DESCRIPTORS_BATCHES_DIR, f'sift_descriptors_{set_name}_batch_{batch_num_sift}_final.pkl')
        with open(sift_batch_filename, 'wb') as f_sift:
            pickle.dump(sift_descriptors_current_batch_list, f_sift)
        print(f"Saved final SIFT descriptor batch: {sift_batch_filename}")

    # Save any remaining ORB descriptors
    if orb_descriptors_current_batch_list:
        orb_batch_filename = os.path.join(ORB_DESCRIPTORS_BATCHES_DIR, f'orb_descriptors_{set_name}_batch_{batch_num_orb}_final.pkl')
        with open(orb_batch_filename, 'wb') as f_orb:
            pickle.dump(orb_descriptors_current_batch_list, f_orb)
        print(f"Saved final ORB descriptor batch: {orb_batch_filename}")

    # --- Save HOG data for this set ---
    if hog_features_list_all:
        # Ensure all HOG descriptors have the same length if stacking, or handle variable length
        # Assuming all valid HOG descriptors from hog.compute().flatten() have the same length
        try:
            hog_array_all = np.vstack(hog_features_list_all)
            hog_labels_array_all = np.array(hog_labels_list_all, dtype=np.int8)
            hog_paths_array_all = np.array(hog_image_paths_list_all, dtype=object) # For traceability

            hog_output_file = os.path.join(OUTPUT_VANILLA_FEATURES_DIR, f'hog_data_{set_name}_balanced.h5')
            with h5py.File(hog_output_file, 'w') as hf:
                hf.create_dataset('hog_features', data=hog_array_all, compression="gzip")
                hf.create_dataset('labels_numeric', data=hog_labels_array_all)
                hf.create_dataset('image_paths', data=hog_paths_array_all)
            print(f"Saved HOG data for {set_name} set to: {hog_output_file}")
            print(f"  HOG features shape: {hog_array_all.shape}")
            print(f"  Labels shape: {hog_labels_array_all.shape}")
            print(f"  Paths shape: {hog_paths_array_all.shape}")
        except ValueError as ve:
            print(f"Error stacking HOG features for {set_name} (likely due to inconsistent shapes for empty HOGs): {ve}")
            print("Consider ensuring extract_vanilla_features_from_image_path returns a fixed-size zero vector for empty HOGs if this is an issue.")
        except Exception as e:
            print(f"ERROR saving HOG data for {set_name} to HDF5: {e}")
    else:
        print(f"No HOG features were collected for {set_name} set.")

    print(f"--- Vanilla BoVW Feature Extraction for {set_name.upper()} SET Complete ---")
    print(f"Descriptor batches saved in: {SIFT_DESCRIPTORS_BATCHES_DIR} and {ORB_DESCRIPTORS_BATCHES_DIR}")
    if hog_features_list_all: print(f"HOG data saved in: {OUTPUT_VANILLA_FEATURES_DIR}")


def main_vanilla_bovw_feature_extraction():
    print("--- Main Vanilla BoVW Feature Extraction from Balanced Split ---")
    ensure_directories()

    if not os.path.exists(BALANCED_SPLIT_NPZ_FILE):
        print(f"ERROR: Balanced split NPZ file not found at {BALANCED_SPLIT_NPZ_FILE}")
        print(f"Please run 'create_balanced_split_for_bovw.py' first.")
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

    # Process Training Set
    process_image_set_for_vanilla_bovw(train_image_paths, train_labels_numeric, set_name="train")

    # Process Test Set
    process_image_set_for_vanilla_bovw(test_image_paths, test_labels_numeric, set_name="test")

    print("\n--- All Vanilla BoVW Feature Extraction from Balanced Split Complete ---")
    print(f"Raw features saved in subdirectories of: {OUTPUT_VANILLA_FEATURES_DIR}")
