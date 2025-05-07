'''
Code by Christopher Kaelin, 2025

SOH stands for "SIFT, ORB, HOG" - these are the three feature extraction methods used in this script.
All three methods are used to extract features from images in the Places365 dataset.
Features will be saved in the specified output directory in a pickle file.
'''

import cv2
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import numpy as np
import os
from tqdm import tqdm
import pickle
import h5py


# Config
TDFS_DATA_DIR = "E:\CV_imgs"
OUTPUT_FEATURES_DIR = "E:\CV_features"
SUBSET_SIZE = 100000
BATCH_SAVE_SIZE = 5000

os.makedirs(OUTPUT_FEATURES_DIR, exist_ok=True) # Create the output directory if it doesn't exist
os.makedirs(os.path.join(OUTPUT_FEATURES_DIR, 'sift_batches'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FEATURES_DIR, 'orb_batches'), exist_ok=True)
print(f"Output directory for features: {OUTPUT_FEATURES_DIR}")
print(f"Subset size: {SUBSET_SIZE}")
print(f"Loading dataset from: {TDFS_DATA_DIR}")
print(f"--- Starting Places365 Download and Preparation ---")

ds_train = tfds.load(
    'places365_small',          # Use the 256x256 version
    split='train', # Specify the splits needed
    data_dir=TDFS_DATA_DIR,     # Tell TFDS where to store everything
    shuffle_files=True
)

# Note: places365_small is still far too large for SIFT to run on; instead we will be taking a subset
ds_subset_indexed = ds_train.take(SUBSET_SIZE).enumerate()
print(f"Selected subset of {SUBSET_SIZE} images from the dataset; adding indices")
# Output of ds_subset_indexed will be (index, tf_example) tuples

# Function to extract features
def extract_features_tf_element(index_tensor, img_tensor, label_tensor): # Takes tensors directly
    '''Extracts SIFT, ORB, and HOG features from indexed tensors.'''
    try:
        # Convert input tensors to numpy
        img_np = img_tensor.numpy()
        label = label_tensor.numpy()
        idx = index_tensor.numpy()

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Initialize Feature Extractors
        sift = cv2.SIFT_create()
        orb = cv2.ORB_create(nfeatures=1000)
        hog_win_size = (128, 128)
        hog = cv2.HOGDescriptor(_winSize=hog_win_size, _blockSize=(16,16), _blockStride=(8,8), _cellSize=(8,8), _nbins=9)

        # Begin Feature Extraction - keypoints not needed
        keypoints_sift, descriptors_sift = sift.detectAndCompute(gray, None)
        keypoints_orb, descriptors_orb = orb.detectAndCompute(gray, None)
        resized_for_hog = cv2.resize(gray, hog_win_size)
        descriptor_hog = hog.compute(resized_for_hog)
        descriptor_hog = descriptor_hog.flatten() if descriptor_hog is not None else None

        # Handle None cases
        if descriptors_sift is None: descriptors_sift = np.array([], dtype=np.float32).reshape(0, 128)
        if descriptors_orb is None: descriptors_orb = np.array([], dtype=np.uint8).reshape(0, 32)
        if descriptor_hog is None: descriptor_hog = np.array([], dtype=np.float32)

        return (idx, label, descriptors_sift, descriptors_orb, descriptor_hog)
    except Exception as e:
        # Need index even on error
        idx = index_tensor.numpy() # Try converting index even if others fail
        return (idx, -1,
                np.array([], dtype=np.float32).reshape(0, 128),
                np.array([], dtype=np.uint8).reshape(0, 32),
                np.array([], dtype=np.float32))
        
# Multiprocessing using tf.data
hog_list = []
label_list = []
index_list = []
sift_batch_data = {} # Temp dict for batch: {idx: sift_desc}
orb_batch_data = {}  # Temp dict for batch: {idx: orb_desc}
processed_count = 0
batch_num = 0

print(f"Beginning feature extraction with tf.data paralellism")

# Indexed function mapping
ds_processed = ds_subset_indexed.map(lambda i, x: tf.py_function(
                func=extract_features_tf_element,
                inp=[i, x['image'], x['label']],  
                Tout=[tf.int64, tf.int64, tf.float32, tf.uint8, tf.float32] 
            ),
        num_parallel_calls=tf.data.AUTOTUNE)

ds_processed = ds_processed.prefetch(buffer_size=tf.data.AUTOTUNE)

for idx, label, sift_desc, orb_desc, hog_desc in tqdm(ds_processed.as_numpy_iterator(), total=SUBSET_SIZE):
    # Store results in the dictionary
    if label != -1:  # Only store valid labels
        # HOG can fit in memory
        if hog_desc.shape[0] > 0:
            hog_list.append(hog_desc)
            label_list.append(label)
            index_list.append(idx)
        # SIFT and ORB are too large to fit in memory, so we save them in batches
        if sift_desc.shape[0] > 0:
            sift_batch_data[idx] = sift_desc
        if orb_desc.shape[0] > 0:
            orb_batch_data[idx] = orb_desc
        processed_count += 1
        
        if processed_count % BATCH_SAVE_SIZE == 0:
            # Save the current batch of SIFT and ORB features
            sift_batch_file = os.path.join(OUTPUT_FEATURES_DIR, 'sift_batches', f'sift_batch_{batch_num}.pkl')
            orb_batch_file = os.path.join(OUTPUT_FEATURES_DIR, 'orb_batches', f'orb_batch_{batch_num}.pkl')
            
            with open(sift_batch_file, 'wb') as f_sift:
                pickle.dump(sift_batch_data, f_sift)
            sift_batch_data = {}
            
            with open(orb_batch_file, 'wb') as f_orb:
                pickle.dump(orb_batch_data, f_orb)
            orb_batch_data = {} 
            
            batch_num += 1
    else:
        print(f"Error processing index {idx}: Invalid label or feature extraction failed.")
        
print("Feature extraction complete. Saving results...")

if sift_batch_data:
    batch_num += 1
    print(f"\nSaving final SIFT batch {batch_num}...")
    sift_batch_file = os.path.join(OUTPUT_FEATURES_DIR, 'sift_batches', f'sift_batch_{batch_num}.pkl')
    with open(sift_batch_file, 'wb') as f_sift:
        pickle.dump(sift_batch_data, f_sift)

if orb_batch_data:
     # Use same batch_num if SIFT wasn't saved, or increment if it was
    if not sift_batch_data: batch_num += 1
    print(f"\nSaving final ORB batch {batch_num}...")
    orb_batch_file = os.path.join(OUTPUT_FEATURES_DIR, 'orb_batches', f'orb_batch_{batch_num}.pkl')
    with open(orb_batch_file, 'wb') as f_orb:
        pickle.dump(orb_batch_data, f_orb)

if hog_list:
    hog_array = np.vstack(hog_list)
    labels_array = np.array(label_list)
    indices_array = np.array(index_list)

    # Use HDF5 for potentially large arrays - more robust than npy
    hog_output_file = os.path.join(OUTPUT_FEATURES_DIR, 'hog_data.h5')
    with h5py.File(hog_output_file, 'w') as hf:
        hf.create_dataset('hog_features', data=hog_array)
        hf.create_dataset('labels', data=labels_array)
        hf.create_dataset('indices', data=indices_array)
    print(f"Saved HOG data to: {hog_output_file}")
    print(f"  HOG shape: {hog_array.shape}")
    print(f"  Labels shape: {labels_array.shape}")
    print(f"  Indices shape: {indices_array.shape}")
else:
    print("No HOG descriptors were collected.")

print(f"\nProcessing finished. Batched SIFT/ORB and consolidated HOG saved in {OUTPUT_FEATURES_DIR}")
