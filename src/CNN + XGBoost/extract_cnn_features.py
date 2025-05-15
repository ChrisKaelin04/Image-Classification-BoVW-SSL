import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os
# import pandas as pd
# import glob
from tqdm import tqdm
import h5py
from PIL import Image

import tensorflow as tf
import tensorflow_datasets as tfds
import gc

# --- Configuration ---
CNN_FEATURES_BASE_DIR = r"E:\CV_Features_CNN_PyTorch"
CNN_MODEL_NAME = "AlexNet_Places365_PyTorch"
CNN_EXTRACTED_FEATURES_DIR = os.path.join(CNN_FEATURES_BASE_DIR, "cnn_extracted_features", CNN_MODEL_NAME)
NPZ_FILE_SUBSET_SPLIT = os.path.join(r"E:\CV_features", "train_test_splits_4cat_revised", "train_test_split_data_4cat_revised.npz")
TFDS_DATA_DIR = r"E:\CV_imgs"
TFDS_SUBSET_SIZE = 100000
TFDS_RANDOM_SEED = 42

IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE_PYTORCH_CNN = 32
PLACES365_WEIGHTS_PATH = r"E:\CV_Features_CNN_PyTorch\alexnet_places365.pth.tar"

# Move device definition outside functions so it's accessible globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- Custom PyTorch Dataset (Keep this outside the pipeline function) ---
class TFDSSubsetFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, subset_indices_list, image_numpy_list, broad_labels_list, transform=None):
        self.subset_indices = subset_indices_list
        self.images_numpy = image_numpy_list
        self.labels = broad_labels_list
        self.transform = transform
        if not (len(self.subset_indices) == len(self.images_numpy) == len(self.labels)):
            raise ValueError("Indices, images, and labels lists must have the same length.")
    def __len__(self):
        return len(self.subset_indices)
    def __getitem__(self, list_idx):
        subset_idx_val = self.subset_indices[list_idx]
        img_np = self.images_numpy[list_idx]
        label_val = self.labels[list_idx]
        try:
            image = Image.fromarray(img_np).convert('RGB')
        except Exception as e:
             print(f"\nERROR: Convert NumPy to PIL for subset_idx {subset_idx_val} (list_idx {list_idx}): {e}. Returning dummy image/data.")
             dummy_img = torch.zeros(3, IMG_HEIGHT, IMG_WIDTH, dtype=torch.float32)
             return subset_idx_val, dummy_img, label_val
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                 print(f"\nWARN: Transform error for subset_idx {subset_idx_val} (list_idx {list_idx}): {e}. Returning dummy image/data.")
                 dummy_img = torch.zeros(3, IMG_HEIGHT, IMG_WIDTH, dtype=torch.float32)
                 return subset_idx_val, dummy_img, label_val
        return subset_idx_val, image, label_val

# --- Load Places365 Pretrained AlexNet Model (Keep this outside the pipeline function) ---
def load_alexnet_places365_model(weights_path):
    print(f"\nLoading AlexNet model architecture (PyTorch)...")
    try: model = models.alexnet(weights=None)
    except TypeError: model = models.alexnet(pretrained=False)

    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 365)
    print("Adjusted AlexNet classifier for 365 classes.")

    print(f"Loading Places365 weights from: {weights_path}")
    try:
        checkpoint = torch.load(weights_path, map_location=device)

        state_dict_from_checkpoint = None
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict_from_checkpoint = checkpoint['state_dict']
                # print("Found 'state_dict' key in checkpoint.") # Keep prints minimal after debugging load logic
            else:
                 state_dict_from_checkpoint = checkpoint
                 # print("No common state_dict key found, assuming checkpoint dictionary is the state_dict.")
        else:
            state_dict_from_checkpoint = checkpoint
            # print("Loaded object is not a dictionary, assuming it's the state_dict directly.")


        if state_dict_from_checkpoint is None:
             raise ValueError(f"Could not find state_dict in the loaded checkpoint from {weights_path}.")

        # --- FIX: Create a new state_dict with 'module.' prefix removed ---
        new_state_dict = {}
        for k, v in state_dict_from_checkpoint.items():
            if 'features.module.' in k:
                name = k.replace('features.module.', 'features.')
            # elif 'classifier.module.' in k: # Uncomment if classifier keys also have module.
            #     name = k.replace('classifier.module.', 'classifier.')
            else:
                name = k
            new_state_dict[name] = v
        # print("Corrected keys in state_dict (replaced 'features.module.' where found).") # Keep prints minimal

        # --- Debug Prints (Optional, remove if you want clean output now) ---
        # print("\n--- Debug: Inspecting new_state_dict keys BEFORE loading ---")
        # print(f"Number of keys in new_state_dict: {len(new_state_dict.keys())}")
        # print("First 20 keys in new_state_dict:")
        # sorted_keys = sorted(new_state_dict.keys())
        # for i, key in enumerate(sorted_keys):
        #     if i < 20:
        #         print(key)
        #     else:
        #         break
        # print("--------------------------------------------------------\n")
        # --- End Debug Prints ---

        model.load_state_dict(new_state_dict, strict=True)
        print("Successfully loaded Places365 weights into AlexNet.")

    except FileNotFoundError:
        print(f"ERROR: Places365 weights file not found at {weights_path}. Please check the path.")
        exit()
    except Exception as e:
        print(f"ERROR loading Places365 weights (after 'features.module.' fix attempt): {e}")
        # print("\n--- Debug: Model state_dict keys expected ---") # Optional print
        # try:
        #     model_keys = sorted(model.state_dict().keys())
        #     for i, key in enumerate(model_keys):
        #          if i < 20: print(key)
        #          else: break
        # except Exception as model_e:
        #      print(f"Could not print model keys: {model_e}")
        # print("--------------------------------------\n")
        exit()

    feature_extractor = model.features
    for param in feature_extractor.parameters():
        param.requires_grad = False

    print("AlexNet 'features' block ready for extraction, weights frozen.")
    return feature_extractor.to(device)

# --- Define Preprocessing Transform (Keep this outside the pipeline function) ---
def get_alexnet_preprocessing_transform():
    print(f"\nDefining preprocessing transform for AlexNet ({IMG_HEIGHT}x{IMG_WIDTH} input)...")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_HEIGHT),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("AlexNet preprocessing transform defined.")
    return transform

# --- Feature Extraction Loop and Saving (Keep this outside the pipeline function) ---
def extract_and_save_features(
    subset_indices_for_dataset, images_numpy_for_dataset, broad_labels_for_dataset,
    model_obj, transform_obj, output_filepath):
    print(f"\nPreparing dataset for {os.path.basename(output_filepath)} with {len(images_numpy_for_dataset)} images.")
    dataset = TFDSSubsetFeatureDataset(subset_indices_for_dataset, images_numpy_for_dataset, broad_labels_for_dataset, transform=transform_obj)


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE_PYTORCH_CNN, shuffle=False, num_workers=0, pin_memory=False)

    features_list, actual_indices_processed, actual_labels_processed = [], [], []
    print(f"Extracting features for {len(dataset)} images...")
    model_obj.eval()
    dataloader_tqdm = tqdm(dataloader, desc=f"Extracting {CNN_MODEL_NAME} Features", leave=True)
    with torch.no_grad():
        for subset_indices_batch, images_batch, labels_batch in dataloader_tqdm:
            if images_batch.ndim != 4 or images_batch.shape[1] != 3 or (torch.sum(images_batch) == 0 and images_batch.shape[0] > 0): # Check for zero sum on non-empty batch
                 dataloader_tqdm.write(f"Skipping a potentially malformed or dummy batch. Subset indices: {subset_indices_batch.tolist()}")
                 continue

            images_batch = images_batch.to(device)

            try:
                features_batch = model_obj(images_batch)
                features_batch_flattened = features_batch.view(features_batch.size(0), -1)

                features_list.extend(features_batch_flattened.cpu().numpy())
                actual_indices_processed.extend(subset_indices_batch.cpu().tolist())
                actual_labels_processed.extend(labels_batch.cpu().tolist())

            except Exception as e:
                dataloader_tqdm.write(f"ERROR processing batch (indices {subset_indices_batch.tolist()}): {e}. Skipping batch.")
                continue


    if features_list:
        features_array = np.array(features_list)
        indices_array = np.array(actual_indices_processed, dtype=np.int32)
        labels_array = np.array(actual_labels_processed, dtype=np.int8)

        print(f"\nSaving extracted data to {output_filepath}...")
        try:
            with h5py.File(output_filepath, 'w') as hf:
                hf.create_dataset('features', data=features_array)
                hf.create_dataset('subset_indices', data=indices_array)
                hf.create_dataset('labels', data=labels_array)
            print(f"Saved: {output_filepath}\n  Features shape: {features_array.shape}, Indices shape: {indices_array.shape}, Labels shape: {labels_array.shape}")

            if features_array.shape[0] != len(images_numpy_for_dataset):
                 print(f"WARN: Number of saved features ({features_array.shape[0]}) is less than the number of images provided to the dataset ({len(images_numpy_for_dataset)}). This may be due to errors or malformed batches.")

        except Exception as e:
            print(f"ERROR saving HDF5 file {output_filepath}: {e}")
    else:
        print(f"No features were successfully extracted for {os.path.basename(output_filepath)}. No file saved.")


# --- Main Execution Pipeline (All data loading/prep moves HERE) ---
def extract_alexnet_places365_features_pipeline():
    if torch.cuda.is_available():
        print(f"PyTorch using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch using CPU.")

    os.makedirs(CNN_EXTRACTED_FEATURES_DIR, exist_ok=True)
    print(f"Saving extracted features to: {CNN_EXTRACTED_FEATURES_DIR}")

    # --- 1. Load TFDS Dataset Definition (MOVED HERE) ---
    print("--- Defining TFDS Data Source ---")
    tf.config.set_visible_devices([], 'GPU')

    ds_train_tfds = tfds.load(
        'places365_small',
        split='train',
        data_dir=TFDS_DATA_DIR,
        shuffle_files=False,
    )

    ds_subset_tfds = ds_train_tfds.shuffle(buffer_size=max(TFDS_SUBSET_SIZE * 2, 2048), seed=TFDS_RANDOM_SEED, reshuffle_each_iteration=False)
    ds_subset_indexed_tfds = ds_subset_tfds.take(TFDS_SUBSET_SIZE).enumerate()

    # --- 2. Load Train/Test Split Indices and Broad Category Labels from NPZ (MOVED HERE) ---
    print(f"Loading train/test split data (subset indices and broad labels) from: {NPZ_FILE_SUBSET_SPLIT}")
    try:
        split_data = np.load(NPZ_FILE_SUBSET_SPLIT)
        subset_train_indices_npz = split_data['train_indices'].tolist()
        subset_test_indices_npz = split_data['test_indices'].tolist()
        y_train_broad_npz = split_data['train_labels_numeric'].tolist()
        y_test_broad_npz = split_data['test_labels_numeric'].tolist()

        if len(subset_train_indices_npz) != len(y_train_broad_npz) or \
           len(subset_test_indices_npz) != len(y_test_broad_npz):
            print("ERROR: Mismatch between number of indices and labels in NPZ file. Halting.")
            exit()
        print(f"Loaded {len(subset_train_indices_npz)} train indices/labels and {len(subset_test_indices_npz)} test indices/labels from NPZ.")
    except FileNotFoundError:
        print(f"ERROR: NPZ file not found at {NPZ_FILE_SUBSET_SPLIT}.")
        exit()
    except KeyError as e:
        print(f"ERROR: Missing key {e} in NPZ file. Ensure 'train_indices', 'test_indices', 'train_labels_numeric', 'test_labels_numeric' exist.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred loading NPZ: {e}")
        exit()

    # --- 3. Cache ONLY the REQUIRED Images from TFDS (MOVED HERE) ---
    all_required_subset_indices_set = set(subset_train_indices_npz + subset_test_indices_npz)
    print(f"Identified {len(all_required_subset_indices_set)} unique subset indices required from NPZ splits.")

    print(f"Caching ONLY the {len(all_required_subset_indices_set)} required images from TFDS...")
    required_idx_to_image_map = {}
    num_found_in_tfds = 0
    for i_tensor, item in tqdm(ds_subset_indexed_tfds.as_numpy_iterator(), total=TFDS_SUBSET_SIZE, desc="Filtering/Caching TFDS images"):
        current_subset_idx = int(i_tensor)
        if current_subset_idx in all_required_subset_indices_set:
            required_idx_to_image_map[current_subset_idx] = item['image']
            num_found_in_tfds += 1
            if num_found_in_tfds == len(all_required_subset_indices_set):
                print(f"All {num_found_in_tfds} required images have been found and cached. Breaking TFDS iteration.")
                break
    print(f"Cached {len(required_idx_to_image_map)} images from TFDS that are present in train/test splits.")

    if num_found_in_tfds < len(all_required_subset_indices_set):
        print(f"WARNING: Only found {num_found_in_tfds} out of {len(all_required_subset_indices_set)} required images in the TFDS subset defined by TFDS_SUBSET_SIZE and TFDS_RANDOM_SEED. Some indices from your NPZ may be out of bounds or not in that specific subset sequence.")

    # --- 4. Prepare Data Lists for PyTorch Dataset (MOVED HERE) ---
    # For training set
    images_train_actual_numpy = []
    y_train_actual_broad = []
    actual_train_subset_indices = []

    for i in range(len(subset_train_indices_npz)):
        idx = subset_train_indices_npz[i]
        if idx in required_idx_to_image_map:
            images_train_actual_numpy.append(required_idx_to_image_map[idx])
            y_train_actual_broad.append(y_train_broad_npz[i])
            actual_train_subset_indices.append(idx)
    y_train_actual_broad = np.array(y_train_actual_broad)

    # For testing set
    images_test_actual_numpy = []
    y_test_actual_broad = []
    actual_test_subset_indices = []

    for i in range(len(subset_test_indices_npz)):
        idx = subset_test_indices_npz[i]
        if idx in required_idx_to_image_map:
            images_test_actual_numpy.append(required_idx_to_image_map[idx])
            y_test_actual_broad.append(y_test_broad_npz[i])
            actual_test_subset_indices.append(idx)
    y_test_actual_broad = np.array(y_test_actual_broad)

    print(f"Prepared {len(images_train_actual_numpy)} images for training.")
    print(f"Prepared {len(images_test_actual_numpy)} images for testing.")

    if not images_train_actual_numpy or not images_test_actual_numpy:
        print("ERROR: No valid images found for train or test after filtering based on NPZ indices. Cannot proceed.")
        exit()

    # Clear the large map after preparing split-specific lists
    del required_idx_to_image_map
    gc.collect()
    print("Cleaned up intermediate image cache map.")
    # --- End of data loading/prep moved here ---


    # Load the model (already defined outside, call it here)
    feature_extractor_model = load_alexnet_places365_model(PLACES365_WEIGHTS_PATH)
    # Define preprocessing transform (already defined outside, call it here)
    preprocessing_transform = get_alexnet_preprocessing_transform()

    output_file_suffix = f"subset{TFDS_SUBSET_SIZE}_seed{TFDS_RANDOM_SEED}"

    # --- Process Training Set ---
    train_output_filepath = os.path.join(CNN_EXTRACTED_FEATURES_DIR, f'X_train_{CNN_MODEL_NAME.lower()}_features_{output_file_suffix}.h5')
    # These variables are now defined in the scope of this function
    extract_and_save_features(actual_train_subset_indices, images_train_actual_numpy, y_train_actual_broad,
                              feature_extractor_model, preprocessing_transform, train_output_filepath)

    # Explicitly delete train data to free memory before processing test data
    del images_train_actual_numpy, y_train_actual_broad, actual_train_subset_indices
    gc.collect()
    print("Cleaned up training data from RAM.")

    # --- Process Testing Set ---
    test_output_filepath = os.path.join(CNN_EXTRACTED_FEATURES_DIR, f'X_test_{CNN_MODEL_NAME.lower()}_features_{output_file_suffix}.h5')
     # These variables are now defined in the scope of this function
    extract_and_save_features(actual_test_subset_indices, images_test_actual_numpy, y_test_actual_broad,
                              feature_extractor_model, preprocessing_transform, test_output_filepath)

    # Clean up test data too
    del images_test_actual_numpy, y_test_actual_broad, actual_test_subset_indices
    gc.collect()
    print("Cleaned up testing data from RAM.")


    print(f"\n--- PyTorch CNN Feature Extraction Pipeline Complete ({CNN_MODEL_NAME}) ---")

