import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import pickle # For potentially saving lists if not using h5py immediately
from tqdm import tqdm

# --- Configuration ---
# Common paths (for loading splits)
FEATURES_DIR_VANILLA = "E:\CV_features" # Where your original splits are
SPLITS_DIR_COMMON = os.path.join(FEATURES_DIR_VANILLA, "train_test_splits_4cat_revised")
NPZ_FILE = os.path.join(SPLITS_DIR_COMMON, "train_test_split_data_4cat_revised.npz")

# CNN Feature Output Paths
CNN_FEATURES_BASE_DIR = "E:\CV_Features_CNN"
CNN_MODEL_NAME = "ResNet50V2" # Or "MobileNetV2", "EfficientNetB0" etc.
CNN_EXTRACTED_FEATURES_DIR = os.path.join(CNN_FEATURES_BASE_DIR, "cnn_extracted_features", CNN_MODEL_NAME)

os.makedirs(CNN_EXTRACTED_FEATURES_DIR, exist_ok=True)

# TFDS Configuration
TFDS_DATA_DIR = "E:\CV_imgs"

# CNN Model Specifics (for ResNet50V2)
IMG_WIDTH, IMG_HEIGHT = 224, 224 # Input size for ResNet50V2
# Preprocessing function for the chosen model
PREPROCESS_INPUT_FUNC = tf.keras.applications.resnet_v2.preprocess_input

# --- Helper function to build the feature extraction model ---
def get_feature_extractor_model(model_name="ResNet50V2"):
    if model_name == "ResNet50V2":
        base_model = tf.keras.applications.ResNet50V2(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
            include_top=False,  # Exclude the ImageNet classifier
            weights='imagenet',
            pooling='avg'       # Adds a GlobalAveragePooling2D layer
        )
    elif model_name == "MobileNetV2":
        # IMG_WIDTH, IMG_HEIGHT would be 224, 224 for MobileNetV2 too
        # PREPROCESS_INPUT_FUNC would be tf.keras.applications.mobilenet_v2.preprocess_input
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
            include_top=False, weights='imagenet', pooling='avg'
        )
    # Add other models like EfficientNet here if needed
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    # The base_model with pooling='avg' is already our feature extractor
    # No need to create a new tf.keras.Model if pooling='avg' is used
    # If pooling=None, you'd do:
    # inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    # x = base_model(inputs, training=False)
    # outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
    # feature_extractor = tf.keras.Model(inputs, outputs)
    # return feature_extractor
    
    base_model.trainable = False # Freeze the weights
    return base_model

# --- Function to process a batch of images ---
@tf.function # For potential graph mode optimization
def extract_features_batch(images, model):
    images_resized = tf.image.resize(images, [IMG_HEIGHT, IMG_WIDTH])
    images_preprocessed = PREPROCESS_INPUT_FUNC(images_resized)
    features = model(images_preprocessed, training=False) # training=False is important
    return features

def run_cnn_feature_extraction():
    print(f"--- Starting CNN Feature Extraction using {CNN_MODEL_NAME} ---")

    # 1. Load train/test indices from your common NPZ file
    print(f"Loading train/test indices from: {NPZ_FILE}")
    split_data = np.load(NPZ_FILE)
    train_indices_original = split_data['train_indices']
    test_indices_original = split_data['test_indices']
    print(f"Loaded {len(train_indices_original)} train and {len(test_indices_original)} test original indices.")

    # Convert to sets for faster lookup
    train_indices_set = set(train_indices_original)
    test_indices_set = set(test_indices_original)

    # 2. Load TFDS dataset
    # We need to iterate through it to find our specific images by index
    print(f"Loading Places365 dataset from: {TFDS_DATA_DIR}...")
    # We load the 'train' split because your indices likely come from an enumeration of this split
    ds_full_train = tfds.load(
        'places365_small',
        split='train', # Assuming your indices are from the 'train' split
        data_dir=TFDS_DATA_DIR,
        shuffle_files=False # IMPORTANT: Keep order for index matching
    ).enumerate() # Enumerate to get (index, data_element)

    # 3. Load the pretrained CNN feature extractor model
    print(f"Loading pretrained {CNN_MODEL_NAME} model...")
    feature_extractor = get_feature_extractor_model(model_name=CNN_MODEL_NAME)
    feature_extractor.summary() # Print model summary

    # 4. Prepare lists to store features and map original indices to new sequential indices
    # We need to re-map because np.save will save sequentially
    
    X_train_features_list = []
    y_train_labels_list = [] # We also need to save corresponding labels if we build a new NPZ
    train_original_idx_map = {} # Maps original TFDS index to new sequential index in our saved array

    X_test_features_list = []
    y_test_labels_list = []
    test_original_idx_map = {}

    # We can process in batches for GPU efficiency
    BATCH_SIZE_CNN = 32 # Adjust based on GPU memory
    
    # Temporary lists for current batch
    current_batch_images = []
    current_batch_indices_labels = [] # Store (original_idx, original_label, 'train'/'test')

    print("Iterating through dataset to find and process selected train/test images...")
    # Iterate through the *entire* TFDS train split to find our selected indices
    # This can be slow if SUBSET_SIZE is much smaller than the full dataset.
    # A more efficient way if SUBSET_SIZE was from a .take() on a shuffled dataset is harder to map back.
    # Assuming your original SUBSET_SIZE was a .take() on the ordered 'train' split for this to work easily.
    
    # Create dictionaries to hold data for each split to ensure order later
    # Key: original_tfds_index, Value: {'image': img_tensor, 'label': label_tensor}
    train_data_to_process = {}
    test_data_to_process = {}

    print("Scanning dataset to collect images for selected train/test indices...")
    for original_idx, data_element in tqdm(ds_full_train, desc="Scanning TFDS"):
        original_idx = original_idx.numpy() # Convert EagerTensor to numpy
        if original_idx in train_indices_set:
            train_data_to_process[original_idx] = {'image': data_element['image'], 'label': data_element['label']}
        elif original_idx in test_indices_set:
            test_data_to_process[original_idx] = {'image': data_element['image'], 'label': data_element['label']}

    # Process Training Images (in the order of train_indices_original)
    print(f"\nExtracting features for {len(train_indices_original)} training images...")
    img_buffer = []
    for i, original_idx in enumerate(tqdm(train_indices_original, desc="Train Features")):
        if original_idx in train_data_to_process:
            data = train_data_to_process[original_idx]
            img_buffer.append(data['image'])
            y_train_labels_list.append(data['label'].numpy()) # Save original label

            if len(img_buffer) == BATCH_SIZE_CNN or i == len(train_indices_original) - 1:
                if img_buffer: # Ensure buffer is not empty
                    img_batch_tf = tf.stack(img_buffer)
                    features_batch = extract_features_batch(img_batch_tf, feature_extractor)
                    X_train_features_list.extend(features_batch.numpy())
                    img_buffer = [] # Clear buffer
        else:
            print(f"Warning: Original index {original_idx} from train_indices not found in scanned TFDS data.")
            # Add a placeholder or handle error - for now, this might lead to misaligned features/labels
            # It's better if all train_indices_original are found.

    # Process Test Images (in the order of test_indices_original)
    print(f"\nExtracting features for {len(test_indices_original)} test images...")
    img_buffer = []
    for i, original_idx in enumerate(tqdm(test_indices_original, desc="Test Features")):
        if original_idx in test_data_to_process:
            data = test_data_to_process[original_idx]
            img_buffer.append(data['image'])
            y_test_labels_list.append(data['label'].numpy()) # Save original label

            if len(img_buffer) == BATCH_SIZE_CNN or i == len(test_indices_original) - 1:
                if img_buffer:
                    img_batch_tf = tf.stack(img_buffer)
                    features_batch = extract_features_batch(img_batch_tf, feature_extractor)
                    X_test_features_list.extend(features_batch.numpy())
                    img_buffer = []
        else:
            print(f"Warning: Original index {original_idx} from test_indices not found in scanned TFDS data.")


    # 5. Save features
    if X_train_features_list:
        X_train_cnn_features = np.array(X_train_features_list)
        train_output_file = os.path.join(CNN_EXTRACTED_FEATURES_DIR, f'X_train_{CNN_MODEL_NAME.lower()}_features.npy')
        np.save(train_output_file, X_train_cnn_features)
        print(f"\nSaved training CNN features to: {train_output_file}, Shape: {X_train_cnn_features.shape}")
        
        # Optionally save the corresponding original labels if you want to make a new NPZ specific to these features
        # For now, we assume you'll use the original y_train from your common NPZ, aligned by order.
        if len(y_train_labels_list) != X_train_cnn_features.shape[0]:
            print(f"WARNING: Mismatch in number of extracted train features ({X_train_cnn_features.shape[0]}) and collected labels ({len(y_train_labels_list)})")

    if X_test_features_list:
        X_test_cnn_features = np.array(X_test_features_list)
        test_output_file = os.path.join(CNN_EXTRACTED_FEATURES_DIR, f'X_test_{CNN_MODEL_NAME.lower()}_features.npy')
        np.save(test_output_file, X_test_cnn_features)
        print(f"Saved test CNN features to: {test_output_file}, Shape: {X_test_cnn_features.shape}")
        if len(y_test_labels_list) != X_test_cnn_features.shape[0]:
            print(f"WARNING: Mismatch in number of extracted test features ({X_test_cnn_features.shape[0]}) and collected labels ({len(y_test_labels_list)})")


    print(f"--- CNN Feature Extraction using {CNN_MODEL_NAME} Complete ---")
    
def extract_cnn_features():
    """
    Main function to run the CNN feature extraction pipeline.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPU found by TensorFlow. Running on CPU.")

    run_cnn_feature_extraction()