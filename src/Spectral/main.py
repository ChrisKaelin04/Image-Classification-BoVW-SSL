'''
Okie so we want to do spectral comparison of the images to see if this might work as basic classification. Its going to suck
'''
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import cv2
import os
import pickle
import h5py
from tqdm import tqdm
import warnings
import joblib # For saving/loading models and potentially scalers
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import collections # For Counter

# --- Overall Configuration ---
TFDS_DATA_DIR = "E:\CV_imgs"
BASE_OUTPUT_DIR = "E:\CV_Pipeline_Spectral_Balanced" # Main output for this spectral pipeline, indicating balanced data
RANDOM_SEED = 42

# --- Balancing Parameters ---
TARGET_IMAGES_PER_BROAD_CATEGORY = 25000 # Desired number of images per broad category
MAX_TFDS_IMAGES_TO_SCAN = 500000 # How many images to scan from TFDS to find your samples.
                                  # Increase if you don't find enough for all categories.

# --- Spectral Feature Parameters ---
RESIZE_DIM = (128, 128)
FFT_REGION_SIZE = 32
FEATURE_VECTOR_LENGTH = FFT_REGION_SIZE * FFT_REGION_SIZE # 32*32 = 1024

# --- Output Paths ---
BALANCED_SPLITS_INFO_SUBDIR = os.path.join(BASE_OUTPUT_DIR, "balanced_split_info_4cat")
SELECTED_INDICES_NPY = os.path.join(BALANCED_SPLITS_INFO_SUBDIR, "selected_original_tfds_indices.npy")
SELECTED_BROAD_LABELS_NPY = os.path.join(BALANCED_SPLITS_INFO_SUBDIR, "selected_broad_numeric_labels.npy")
BROAD_LABEL_ENCODER_FILE = os.path.join(BALANCED_SPLITS_INFO_SUBDIR, "broad_label_encoder_spectral_4cat.pkl")

SPECTRAL_FEATURES_SUBDIR = os.path.join(BASE_OUTPUT_DIR, "spectral_features_data")
SPECTRAL_H5_FILE = os.path.join(SPECTRAL_FEATURES_SUBDIR, f"spectral_fft_{FFT_REGION_SIZE}x{FFT_REGION_SIZE}_balanced.h5")

SPLITS_SUBDIR = os.path.join(BASE_OUTPUT_DIR, "train_test_splits_spectral_4cat_balanced")
SPECTRAL_NPZ_FILE = os.path.join(SPLITS_SUBDIR, "train_test_split_data_spectral_4cat_balanced.npz")

RESULTS_DIR_XGB_SPECTRAL = os.path.join(BASE_OUTPUT_DIR, "classification_results_XGB_Spectral_4cat_Balanced")

# Create all necessary directories
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(BALANCED_SPLITS_INFO_SUBDIR, exist_ok=True)
os.makedirs(SPECTRAL_FEATURES_SUBDIR, exist_ok=True)
os.makedirs(SPLITS_SUBDIR, exist_ok=True)
os.makedirs(RESULTS_DIR_XGB_SPECTRAL, exist_ok=True)

warnings.filterwarnings("ignore", message="Parameters: {.*use_label_encoder.*} are not used.", category=UserWarning, module="xgboost.core")

# --- Broad Category Definitions (from your split_data.py) ---
broad_categories_list = [
    "Indoor Residential", "Indoor Public/Commercial",
    "Outdoor Natural", "Outdoor Urban"
]
broad_category_definitions = {
    'Indoor Public/Commercial': [   'airplane_cabin',
                                    'airport_terminal',
                                    'amusement_arcade',
                                    'arcade',
                                    'aquarium',
                                    'archive',
                                    'arena/hockey',
                                    'arena/performance',
                                    'art_gallery',
                                    'art_school',
                                    'art_studio',
                                    'assembly_line',
                                    'auditorium',
                                    'auto_factory',
                                    'auto_showroom',
                                    'atrium/public',
                                    'bakery/shop',
                                    'ball_pit',
                                    'banquet_hall',
                                    'ballroom',
                                    'bank_vault',
                                    'bar',
                                    'beauty_salon',
                                    'bedchamber',
                                    'beer_hall',
                                    'biology_laboratory',
                                    'basketball_court/indoor',
                                    'bazaar/indoor',
                                    'bookstore',
                                    'booth/indoor',
                                    'bowling_alley',
                                    'boxing_ring',
                                    'burial_chamber',
                                    'bus_interior',
                                    'bus_station/indoor',
                                    'butchers_shop',
                                    'cafeteria',
                                    'candy_store',
                                    'car_interior',
                                    'catacomb',
                                    'chemistry_lab',
                                    'church/indoor',
                                    'classroom',
                                    'clean_room',
                                    'clothing_store',
                                    'coffee_shop',
                                    'cockpit',
                                    'computer_room',
                                    'conference_center',
                                    'conference_room',
                                    'corridor',
                                    'courthouse',
                                    'delicatessen',
                                    'department_store',
                                    'dining_hall',
                                    'discotheque',
                                    'drugstore',
                                    'elevator/door',
                                    'elevator_lobby',
                                    'elevator_shaft',
                                    'engine_room',
                                    'entrance_hall',
                                    'escalator/indoor',
                                    'fabric_store',
                                    'fastfood_restaurant',
                                    'fire_station',
                                    'flea_market/indoor',
                                    'florist_shop/indoor',
                                    'food_court',
                                    'galley',
                                    'general_store/indoor',
                                    'gift_shop',
                                    'greenhouse/indoor',
                                    'gymnasium/indoor',
                                    'hangar/indoor',
                                    'hardware_store',
                                    'hospital_room',
                                    'hotel_room',
                                    'ice_cream_parlor',
                                    'ice_skating_rink/indoor',
                                    'jail_cell',
                                    'jewelry_shop',
                                    'kindergarden_classroom',
                                    'laundromat',
                                    'lecture_room',
                                    'legislative_chamber',
                                    'library/indoor',
                                    'lobby',
                                    'locker_room',
                                    'market/indoor',
                                    'martial_arts_gym',
                                    'mezzanine',
                                    'movie_theater/indoor',
                                    'museum/indoor',
                                    'music_studio',
                                    'natural_history_museum',
                                    'nursing_home',
                                    'office',
                                    'office_building',
                                    'office_cubicles',
                                    'operating_room',
                                    'orchestra_pit',
                                    'parking_garage/indoor',
                                    'pet_shop',
                                    'pharmacy',
                                    'physics_laboratory',
                                    'pizzeria',
                                    'pub/indoor',
                                    'reception',
                                    'recreation_room',
                                    'repair_shop',
                                    'restaurant',
                                    'restaurant_kitchen',
                                    'sauna',
                                    'science_museum',
                                    'server_room',
                                    'shoe_shop',
                                    'shopping_mall/indoor',
                                    'stage/indoor',
                                    'subway_station/platform',
                                    'supermarket',
                                    'sushi_bar',
                                    'swimming_pool/indoor',
                                    'television_studio',
                                    'throne_room',
                                    'ticket_booth',
                                    'toyshop',
                                    'train_interior',
                                    'veterinarians_office',
                                    'waiting_room',
                                    'wet_bar',
                                    'youth_hostel'],
    'Indoor Residential': [   'alcove',
                              'attic',
                              'basement',
                              'bathroom',
                              'bedroom',
                              'childs_room',
                              'closet',
                              'dining_room',
                              'dorm_room',
                              'dressing_room',
                              'home_office',
                              'home_theater',
                              'jacuzzi/indoor',
                              'kitchen',
                              'living_room',
                              'nursery',
                              'pantry',
                              'playroom',
                              'shower',
                              'staircase',
                              'storage_room',
                              'television_room',
                              'utility_room',
                              'artists_loft',
                              'balcony/interior',
                              'bow_window/indoor',
                              'garage/indoor'],
    'Outdoor Natural': [   'badlands',
                           'bamboo_forest',
                           'beach',
                           'butte',
                           'canyon',
                           'canal/natural',
                           'campsite',
                           'cliff',
                           'coast',
                           'creek',
                           'crevasse',
                           'desert/sand',
                           'desert/vegetation',
                           'field/wild',
                           'forest/broadleaf',
                           'forest_path',
                           'forest_road',
                           'glacier',
                           'grotto',
                           'hot_spring',
                           'ice_floe',
                           'ice_shelf',
                           'iceberg',
                           'igloo',
                           'islet',
                           'lagoon',
                           'lake/natural',
                           'marsh',
                           'mountain',
                           'mountain_path',
                           'mountain_snowy',
                           'ocean',
                           'rainforest',
                           'river',
                           'rock_arch',
                           'sky',
                           'ski_slope',
                           'snowfield',
                           'swamp',
                           'swimming_hole',
                           'tree_farm',
                           'tundra',
                           'underwater/ocean_deep',
                           'valley',
                           'volcano',
                           'waterfall',
                           'watering_hole',
                           'wave'],
    'Outdoor Urban': [   'airfield',
                         'alley',
                         'amphitheater',
                         'amusement_park',
                         'apartment_building/outdoor',
                         'aqueduct',
                         'arch',
                         'archaelogical_excavation',
                         'arena/rodeo',
                         'army_base',
                         'athletic_field/outdoor',
                         'balcony/exterior',
                         'barn',
                         'barndoor',
                         'baseball_field',
                         'bazaar/outdoor',
                         'beach_house',
                         'beer_garden',
                         'berth',
                         'boardwalk',
                         'boathouse',
                         'boat_deck',
                         'botanical_garden',
                         'bridge',
                         'building_facade',
                         'bullring',
                         'cabin/outdoor',
                         'campus',
                         'canal/urban',
                         'carrousel',
                         'castle',
                         'cemetery',
                         'chalet',
                         'church/outdoor',
                         'construction_site',
                         'corn_field',
                         'corral',
                         'cottage',
                         'courtyard',
                         'crosswalk',
                         'dam',
                         'desert_road',
                         'diner/outdoor',
                         'doorway/outdoor',
                         'downtown',
                         'driveway',
                         'embassy',
                         'excavation',
                         'farm',
                         'field/cultivated',
                         'field_road',
                         'fire_escape',
                         'fishpond',
                         'football_field',
                         'formal_garden',
                         'fountain',
                         'garage/outdoor',
                         'gas_station',
                         'gazebo/exterior',
                         'general_store/outdoor',
                         'golf_course',
                         'greenhouse/outdoor',
                         'hangar/outdoor',
                         'harbor',
                         'hayfield',
                         'heliport',
                         'highway',
                         'hospital',
                         'hotel/outdoor',
                         'house',
                         'hunting_lodge/outdoor',
                         'ice_skating_rink/outdoor',
                         'industrial_area',
                         'inn/outdoor',
                         'japanese_garden',
                         'junkyard',
                         'kasbah',
                         'kennel/outdoor',
                         'landing_deck',
                         'landfill',
                         'lawn',
                         'library/outdoor',
                         'lighthouse',
                         'loading_dock',
                         'lock_chamber',
                         'mansion',
                         'manufactured_home',
                         'market/outdoor',
                         'mausoleum',
                         'medina',
                         'moat/water',
                         'mosque/outdoor',
                         'motel',
                         'museum/outdoor',
                         'oast_house',
                         'oilrig',
                         'orchard',
                         'pagoda',
                         'palace',
                         'park',
                         'parking_garage/outdoor',
                         'parking_lot',
                         'pasture',
                         'patio',
                         'pavilion',
                         'phone_booth',
                         'picnic_area',
                         'pier',
                         'playground',
                         'plaza',
                         'pond',
                         'porch',
                         'promenade',
                         'raceway',
                         'racecourse',
                         'raft',
                         'railroad_track',
                         'residential_neighborhood',
                         'restaurant_patio',
                         'rice_paddy',
                         'roof_garden',
                         'rope_bridge',
                         'ruin',
                         'runway',
                         'sandbox',
                         'schoolhouse',
                         'shed',
                         'shopfront',
                         'ski_resort',
                         'skyscraper',
                         'slum',
                         'soccer_field',
                         'stable',
                         'stadium/baseball',
                         'stadium/football',
                         'stadium/soccer',
                         'stage/outdoor',
                         'street',
                         'swimming_pool/outdoor',
                         'synagogue/outdoor',
                         'temple/asia',
                         'topiary_garden',
                         'tower',
                         'train_station/platform',
                         'tree_house',
                         'trench',
                         'vegetable_garden',
                         'viaduct',
                         'village',
                         'vineyard',
                         'volleyball_court/outdoor',
                         'water_park',
                         'water_tower',
                         'wheat_field',
                         'wind_farm',
                         'windmill',
                         'yard',
                         'zen_garden']}
if not all(len(v) > 0 for v in broad_category_definitions.values()):
    print("ERROR: broad_category_definitions is incomplete. Please paste the full dictionary.")
    exit()

# === NEW PART: BALANCED DATA SELECTION ===
def prepare_balanced_dataset_info():
    """
    Scans the TFDS 'places365_small' dataset to select a balanced number of images
    per broad category and saves their original TFDS indices and broad numeric labels.
    """
    print(f"--- Starting Balanced Data Preparation for {len(broad_categories_list)} Broad Categories ---")
    print(f"Targeting {TARGET_IMAGES_PER_BROAD_CATEGORY} images per broad category.")
    print(f"Scanning up to {MAX_TFDS_IMAGES_TO_SCAN} images from TFDS.")

    if os.path.exists(SELECTED_INDICES_NPY) and os.path.exists(SELECTED_BROAD_LABELS_NPY) and \
       os.path.exists(BROAD_LABEL_ENCODER_FILE):
        print("Balanced dataset info already exists. Skipping balanced data preparation.")
        return

    print("\nLoading dataset info for 'places365_small'...")
    try:
        full_ds_train_tfds, ds_info = tfds.load('places365_small',
                                                split='train',
                                                data_dir=TFDS_DATA_DIR,
                                                with_info=True,
                                                download=False,
                                                shuffle_files=False)
    except Exception as e:
        print(f"Error loading dataset info or base dataset: {e}")
        exit()

    fine_label_names = ds_info.features['label'].names
    num_fine_classes = len(fine_label_names)
    print(f"Total fine-grained classes in Places365: {num_fine_classes}")

    print("\nSetting up broad category mapping...")
    fine_to_broad_mapping = {}
    for broad_cat, fine_list in broad_category_definitions.items():
        for fine_name in fine_list:
            if fine_name not in fine_label_names:
                # print(f"Warning: Fine label '{fine_name}' in your definition is not in TFDS fine_label_names. Skipping.")
                continue
            fine_to_broad_mapping[fine_name] = broad_cat

    # Initialize structures to collect selected data
    selected_original_tfds_indices = []
    selected_broad_labels_str = []
    images_found_count = {broad_cat: 0 for broad_cat in broad_categories_list}
    total_selected_count = 0
    target_total_images = len(broad_categories_list) * TARGET_IMAGES_PER_BROAD_CATEGORY

    ds_enumerated = full_ds_train_tfds.enumerate()

    scanned_count = 0
    with tqdm(total=min(ds_info.splits['train'].num_examples, MAX_TFDS_IMAGES_TO_SCAN), desc="Scanning TFDS for balance") as pbar:
        for original_tfds_idx_tensor, item in ds_enumerated:
            scanned_count += 1
            pbar.update(1)

            if scanned_count > MAX_TFDS_IMAGES_TO_SCAN:
                print(f"\nReached MAX_TFDS_IMAGES_TO_SCAN ({MAX_TFDS_IMAGES_TO_SCAN}). Stopping scan.")
                break
            if total_selected_count >= target_total_images:
                print(f"\nReached target total images ({target_total_images}). Stopping scan.")
                break

            fine_numeric_label = item['label'].numpy()
            original_tfds_idx = original_tfds_idx_tensor.numpy()

            if 0 <= fine_numeric_label < num_fine_classes:
                fine_label_name = fine_label_names[fine_numeric_label]
                if fine_label_name in fine_to_broad_mapping:
                    broad_cat_name = fine_to_broad_mapping[fine_label_name]
                    if images_found_count[broad_cat_name] < TARGET_IMAGES_PER_BROAD_CATEGORY:
                        selected_original_tfds_indices.append(original_tfds_idx)
                        selected_broad_labels_str.append(broad_cat_name)
                        images_found_count[broad_cat_name] += 1
                        total_selected_count += 1
                        pbar.set_postfix({cat: count for cat, count in images_found_count.items()})

    print("\n--- Image Selection Summary ---")
    all_categories_filled = True
    for broad_cat in broad_categories_list:
        count = images_found_count[broad_cat]
        print(f"Category '{broad_cat}': Selected {count} images (Target: {TARGET_IMAGES_PER_BROAD_CATEGORY})")
        if count < TARGET_IMAGES_PER_BROAD_CATEGORY:
            all_categories_filled = False
            print(f"  WARNING: Category '{broad_cat}' did not reach the target. Consider increasing MAX_TFDS_IMAGES_TO_SCAN or checking category definitions.")

    if total_selected_count == 0:
        print("ERROR: No images were selected. Check mappings and TFDS scan logic. Exiting.")
        exit()

    # Encode broad category string labels to numeric labels
    print("\nEncoding broad category string labels to numeric labels...")
    label_encoder = LabelEncoder()
    label_encoder.fit(broad_categories_list) # Fit on predefined list for consistency
    numeric_broad_labels = label_encoder.transform(selected_broad_labels_str)

    print("Broad Category String to Numeric Mapping (based on LabelEncoder):")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  '{class_name}': {i}")

    # Save the selected indices and labels
    np.save(SELECTED_INDICES_NPY, np.array(selected_original_tfds_indices, dtype=np.int64))
    np.save(SELECTED_BROAD_LABELS_NPY, np.array(numeric_broad_labels, dtype=np.int64))
    with open(BROAD_LABEL_ENCODER_FILE, 'wb') as f:
        pickle.dump(label_encoder, f)

    print(f"\nSaved selected TFDS indices to: {SELECTED_INDICES_NPY}")
    print(f"Saved selected broad numeric labels to: {SELECTED_BROAD_LABELS_NPY}")
    print(f"Saved broad label encoder to: {BROAD_LABEL_ENCODER_FILE}")
    print("--- Balanced Data Preparation Complete ---")


# === PART 1: SPECTRAL FEATURE EXTRACTION ===
def extract_spectral_fft_features(img_np):
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    elif img_np.ndim == 2:
        gray = img_np
    else:
        return np.array([], dtype=np.float32)
    resized_gray = cv2.resize(gray, RESIZE_DIM, interpolation=cv2.INTER_AREA)
    f_transform = np.fft.fft2(resized_gray)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_transform_shifted)
    rows, cols = magnitude_spectrum.shape
    crow, ccol = rows // 2, cols // 2
    r_start, r_end = crow - FFT_REGION_SIZE // 2, crow + FFT_REGION_SIZE // 2
    c_start, c_end = ccol - FFT_REGION_SIZE // 2, ccol + FFT_REGION_SIZE // 2
    if not (0 <= r_start < r_end <= rows and 0 <= c_start < c_end <= cols):
        return np.array([], dtype=np.float32) # Invalid region
    central_region = magnitude_spectrum[r_start:r_end, c_start:c_end]
    feature_vector = central_region.flatten().astype(np.float32)
    if feature_vector.shape[0] != FEATURE_VECTOR_LENGTH:
        return np.array([], dtype=np.float32)
    return feature_vector

def extract_spectral_features_tf_element(idx_tensor, img_tensor, label_tensor):
    """Wrapper for py_function to handle numpy operations for feature extraction."""
    # We pass the original TFDS label_tensor and idx_tensor, but these are from the *full* dataset.
    # The actual broad label we care about for this specific image needs to be looked up
    # based on idx_tensor (original TFDS index) from the pre-selected balanced data.
    try:
        img_np = img_tensor.numpy()
        # The label_tensor here is the original fine-grained TFDS label, which we don't directly use.
        # We'll use the pre-calculated broad_numeric_label based on original_tfds_idx.
        idx_np = idx_tensor.numpy()

        spectral_features = extract_spectral_fft_features(img_np)
        
        # Check if feature extraction was successful. If not, mark as problematic.
        if spectral_features.size != FEATURE_VECTOR_LENGTH:
            return idx_np, np.int64(-1), np.zeros(FEATURE_VECTOR_LENGTH, dtype=np.float32) # Broad label -1 for problematic
        
        # We can't pass the actual broad_numeric_label directly here via py_function
        # because the dataset being iterated is the full TFDS dataset, not the filtered one.
        # So, we return a placeholder (the original TFDS label) and filter/map later.
        # This function only needs to return the original_tfds_idx and the extracted feature.
        # The true broad label mapping happens outside, after we collect all features.
        return idx_np, spectral_features # Return original_tfds_idx and feature
    except Exception as e:
        print(f"Error processing image {idx_tensor.numpy()}: {e}")
        return idx_tensor.numpy(), np.zeros(FEATURE_VECTOR_LENGTH, dtype=np.float32) # Return dummy feature for error

def run_spectral_feature_extraction_if_needed():
    if os.path.exists(SPECTRAL_H5_FILE):
        print(f"Spectral features H5 file already exists: {SPECTRAL_H5_FILE}. Skipping extraction.")
        return

    print("--- Starting Spectral Feature Extraction (FFT-based) for Balanced Subset ---")
    print(f"Output file: {SPECTRAL_H5_FILE}")

    # Load pre-selected indices and labels
    try:
        selected_original_tfds_indices = np.load(SELECTED_INDICES_NPY)
        selected_broad_numeric_labels = np.load(SELECTED_BROAD_LABELS_NPY)
        total_images_to_extract = len(selected_original_tfds_indices)
        
        # Create a dictionary for quick lookup: original_tfds_idx -> broad_numeric_label
        # This maps the original TFDS index (which will be returned by enumerate)
        # to the broad numeric label for that specific image.
        idx_to_broad_label_map = {idx: label for idx, label in 
                                  zip(selected_original_tfds_indices, selected_broad_numeric_labels)}
        
        selected_indices_set = set(selected_original_tfds_indices) # For efficient `in` checks

    except FileNotFoundError:
        print(f"ERROR: Balanced dataset info files not found. Run `prepare_balanced_dataset_info()` first.")
        exit()

    # Load the full TFDS training dataset, ensuring no shuffling to match original indices
    ds_train = tfds.load('places365_small', split='train', data_dir=TFDS_DATA_DIR, shuffle_files=False)
    ds_train_enumerated = ds_train.enumerate() # Get (original_tfds_idx, features_dict)

    all_spectral_features_extracted = []
    all_original_tfds_indices_extracted = []
    all_broad_numeric_labels_extracted = [] # This will hold the true broad labels for extracted features

    # Define the output types for tf.py_function
    # We return the original TFDS index and the feature vector.
    tout_spectral_extraction = [tf.int64, tf.float32] 

    # Process the dataset to extract features only for selected images
    processed_count = 0
    with tqdm(total=total_images_to_extract, desc="Extracting Spectral Features") as pbar:
        for original_tfds_idx_tensor, item in ds_train_enumerated:
            original_tfds_idx = original_tfds_idx_tensor.numpy()

            if original_tfds_idx in selected_indices_set:
                # Call py_function on the image. The label_tensor passed to py_function is dummy here.
                # The actual broad label comes from our pre-calculated map.
                idx_tensor, spectral_feat_tensor = tf.py_function(func=extract_spectral_features_tf_element,
                                                  inp=[original_tfds_idx_tensor, item['image'], item['label']],
                                                  Tout=tout_spectral_extraction)

                # Now convert each Tensor to a NumPy array individually
                idx = idx_tensor.numpy()
                spectral_feat = spectral_feat_tensor.numpy()
                
                if spectral_feat.size == FEATURE_VECTOR_LENGTH:
                    all_spectral_features_extracted.append(spectral_feat)
                    all_original_tfds_indices_extracted.append(idx)
                    all_broad_numeric_labels_extracted.append(idx_to_broad_label_map[idx]) # Add the correct broad label
                    processed_count += 1
                    pbar.update(1)
                else:
                    print(f"Warning: Failed to extract features for original TFDS index {idx}. Skipping.")

    print(f"Successfully extracted spectral features for {processed_count}/{total_images_to_extract} selected images.")
    if not all_spectral_features_extracted:
        print("No spectral features were successfully extracted. Exiting.")
        exit()

    X_spectral = np.array(all_spectral_features_extracted, dtype=np.float32)
    y_broad_labels = np.array(all_broad_numeric_labels_extracted, dtype=np.int64)
    extracted_original_indices = np.array(all_original_tfds_indices_extracted, dtype=np.int64)

    with h5py.File(SPECTRAL_H5_FILE, 'w') as hf:
        hf.create_dataset('features', data=X_spectral)
        hf.create_dataset('broad_labels', data=y_broad_labels) # Store broad labels
        hf.create_dataset('original_tfds_indices', data=extracted_original_indices) # Store original TFDS indices
    print(f"Saved spectral features, broad labels, and original TFDS indices to: {SPECTRAL_H5_FILE}")
    print("--- Spectral Feature Extraction Complete ---")

# === PART 2: DATA SPLITTING (for Spectral Features) ===
def create_spectral_train_test_split_if_needed():
    if os.path.exists(SPECTRAL_NPZ_FILE):
        print(f"Train/test split NPZ file already exists: {SPECTRAL_NPZ_FILE}. Skipping split creation.")
        return

    print("\n--- Creating Train/Test Split for Spectral Features (Balanced Data) ---")
    if not os.path.exists(SPECTRAL_H5_FILE):
        print(f"Error: Spectral features H5 file {SPECTRAL_H5_FILE} not found. Run feature extraction first.")
        exit()

    print("Loading extracted data from spectral H5 file...")
    with h5py.File(SPECTRAL_H5_FILE, 'r') as hf:
        # Load the original TFDS indices and their corresponding broad numeric labels
        # These are already for the balanced subset
        original_tfds_indices_from_h5 = hf['original_tfds_indices'][:]
        broad_numeric_labels_from_h5 = hf['broad_labels'][:]

    if len(original_tfds_indices_from_h5) == 0:
        print("Error: No data loaded from H5 file for splitting. Exiting.")
        exit()

    print(f"Total {len(original_tfds_indices_from_h5)} images with extracted features found.")
    
    # Perform the train/test split on the original TFDS indices and their broad labels
    # The data is already balanced at this point.
    train_indices, test_indices, \
    train_broad_labels_numeric, test_broad_labels_numeric = train_test_split(
        original_tfds_indices_from_h5, broad_numeric_labels_from_h5,
        test_size=0.2, random_state=RANDOM_SEED, stratify=broad_numeric_labels_from_h5)

    print(f"Training set size: {len(train_indices)} indices")
    print(f"Test set size: {len(test_indices)} indices")

    # Verify distribution (optional, but good for balanced splits)
    train_counts = collections.Counter(train_broad_labels_numeric)
    test_counts = collections.Counter(test_broad_labels_numeric)
    print("\nTrain broad label distribution:")
    for label_id in sorted(train_counts.keys()):
        print(f"  Label {label_id}: {train_counts[label_id]} samples")
    print("Test broad label distribution:")
    for label_id in sorted(test_counts.keys()):
        print(f"  Label {label_id}: {test_counts[label_id]} samples")

    np.savez(SPECTRAL_NPZ_FILE,
             train_indices=train_indices, test_indices=test_indices,
             train_labels_numeric=train_broad_labels_numeric, test_labels_numeric=test_broad_labels_numeric)
    print(f"Saved spectral train/test original TFDS indices and broad labels to: {SPECTRAL_NPZ_FILE}")
    print("--- Spectral Train/Test Split Creation Complete ---")


# === PART 3: XGBOOST CLASSIFICATION (using Spectral Features) ===
def plot_confusion_matrix(cm, classes, plot_title='Confusion matrix', cmap=plt.cm.Blues, results_path=None, filename=None):
    plt.figure(figsize=(max(8, len(classes)), max(6, len(classes)*0.8)))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(plot_title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if results_path and filename:
        full_path = os.path.join(results_path, filename)
        plt.savefig(full_path)
    plt.close()

def train_and_evaluate_xgb(X_train_data, y_train_labels, X_test_data, y_test_labels,
                           feature_type_desc, target_class_names,
                           output_results_dir, perform_scaling=True): # Changed default to True
    print(f"\n--- Training XGBoost for {feature_type_desc} ---")
    if X_train_data is None or X_train_data.size == 0 or X_test_data is None or X_test_data.size == 0:
        print(f"Skipping XGBoost for {feature_type_desc}: Missing/empty feature data.")
        return None
    X_train_processed = X_train_data.copy()
    X_test_processed = X_test_data.copy()
    try:
        base_estimator_xgb = xgb.XGBClassifier(objective='multi:softprob',
                                            num_class=len(target_class_names),
                                            tree_method='hist', device='cuda', # Prefer CUDA if available
                                            eval_metric='mlogloss', random_state=RANDOM_SEED,
                                            use_label_encoder=False)
    except xgb.core.XGBoostError as e:
        if "Cannot find CUDA device" in str(e) or "No GPU found" in str(e):
            print("XGBoost CUDA device not found. Falling back to CPU.")
            base_estimator_xgb = xgb.XGBClassifier(objective='multi:softprob',
                                                num_class=len(target_class_names),
                                                tree_method='hist', eval_metric='mlogloss',
                                                random_state=RANDOM_SEED, use_label_encoder=False)
        else: print(f"Error initializing XGBoost: {e}"); return None
    
    scaler_xgb = None
    if perform_scaling:
        scaler_xgb = StandardScaler()
        X_train_processed = scaler_xgb.fit_transform(X_train_processed)
        X_test_processed = scaler_xgb.transform(X_test_processed)
        # Consider saving the scaler if you need to use it later for new data
        # joblib.dump(scaler_xgb, os.path.join(output_results_dir, f'scaler_{feature_type_desc.replace(" ", "_").replace("/", "-")}.joblib'))

    param_grid_xgb = {'n_estimators': [100, 300], 'learning_rate': [0.05, 0.1], 'max_depth': [4, 6]}
    
    print(f"Performing GridSearchCV for XGBoost on {feature_type_desc} (cv=2)...")
    xgb_grid_search = GridSearchCV(estimator=base_estimator_xgb, param_grid=param_grid_xgb,
                                   scoring='accuracy', cv=2, verbose=1, n_jobs=1)
    
    xgb_grid_search.fit(X_train_processed, y_train_labels)
    best_xgb_model = xgb_grid_search.best_estimator_
    print(f"Best XGBoost parameters for {feature_type_desc}: {xgb_grid_search.best_params_}")
    
    model_filename_xgb = os.path.join(output_results_dir, f'xgb_model_{feature_type_desc.replace(" ", "_").replace("/", "-")}.joblib')
    joblib.dump(best_xgb_model, model_filename_xgb)
    
    y_pred_labels_xgb = best_xgb_model.predict(X_test_processed)
    accuracy_val_xgb = accuracy_score(y_test_labels, y_pred_labels_xgb)
    class_report_str_xgb = classification_report(y_test_labels, y_pred_labels_xgb, target_names=target_class_names, zero_division=0)
    conf_matrix_xgb = confusion_matrix(y_test_labels, y_pred_labels_xgb, labels=np.arange(len(target_class_names)))
    
    print(f"Accuracy (XGBoost - {feature_type_desc}): {accuracy_val_xgb:.4f}")
    
    plot_confusion_matrix(conf_matrix_xgb, classes=target_class_names,
                          plot_title=f'CM for XGBoost - {feature_type_desc} (Acc: {accuracy_val_xgb:.3f})',
                          results_path=output_results_dir,
                          filename=f'cm_xgb_{feature_type_desc.replace(" ", "_").replace("/", "-")}.png')
    
    results_text_file_xgb = os.path.join(output_results_dir, f'results_xgb_{feature_type_desc.replace(" ", "_").replace("/", "-")}.txt')
    with open(results_text_file_xgb, 'w') as f:
        f.write(f"--- XGBoost Results for {feature_type_desc} ---\nParams: {xgb_grid_search.best_params_}\nCV Score: {xgb_grid_search.best_score_:.4f}\nAccuracy: {accuracy_val_xgb:.4f}\n\nReport:\n{class_report_str_xgb}\n\nCM:\n{np.array2string(conf_matrix_xgb)}")
    print(f"Saved XGBoost results for {feature_type_desc} to {results_text_file_xgb}")
    return best_xgb_model

def run_spectral_classification():
    print("\n--- Running XGBoost Classification on Spectral Features (Balanced Data) ---")

    # Load the split data (original TFDS indices and broad numeric labels)
    print(f"Loading spectral train/test split data from: {SPECTRAL_NPZ_FILE}")
    try:
        spectral_split_data = np.load(SPECTRAL_NPZ_FILE)
        train_indices_tfds = spectral_split_data['train_indices']
        test_indices_tfds = spectral_split_data['test_indices']
        y_train_spectral = spectral_split_data['train_labels_numeric']
        y_test_spectral = spectral_split_data['test_labels_numeric']
    except FileNotFoundError:
        print(f"ERROR: Spectral NPZ file not found at {SPECTRAL_NPZ_FILE}. Run split creation first.")
        return
    except KeyError as e:
        print(f"ERROR: Missing key {e} in spectral NPZ file {SPECTRAL_NPZ_FILE}.")
        return

    # Load the broad label encoder
    try:
        with open(BROAD_LABEL_ENCODER_FILE, 'rb') as f:
            label_encoder_spectral = pickle.load(f)
        class_names_spectral = label_encoder_spectral.classes_
    except FileNotFoundError:
        print(f"ERROR: Broad Label encoder file not found at {BROAD_LABEL_ENCODER_FILE}.")
        return

    # Load all extracted spectral features and their original TFDS indices from the H5 file
    if not os.path.exists(SPECTRAL_H5_FILE):
        print(f"ERROR: Spectral features H5 file {SPECTRAL_H5_FILE} not found.")
        return
    
    print(f"Loading all spectral features from {SPECTRAL_H5_FILE} for alignment...")
    with h5py.File(SPECTRAL_H5_FILE, 'r') as hf:
        all_X_spectral = hf['features'][:]
        all_original_indices_spectral_h5 = hf['original_tfds_indices'][:]

    # Create a map for quick lookup: original_tfds_index -> row_in_all_X_spectral
    feature_map_spectral = {idx: i for i, idx in enumerate(all_original_indices_spectral_h5)}

    # Align X_train_spectral features using the train_indices_tfds
    X_train_spectral_list = []
    for idx_tfds in train_indices_tfds:
        if idx_tfds in feature_map_spectral:
            X_train_spectral_list.append(all_X_spectral[feature_map_spectral[idx_tfds]])
        else:
            print(f"Warning: Original TFDS index {idx_tfds} from train_indices_tfds not found in H5 features. Data inconsistency.")
            # Depending on strictness, you might want to raise an error or fill with zeros.
            # For now, append zeros, which will likely lead to poor performance for that sample.
            X_train_spectral_list.append(np.zeros(FEATURE_VECTOR_LENGTH, dtype=np.float32))

    # Align X_test_spectral features using the test_indices_tfds
    X_test_spectral_list = []
    for idx_tfds in test_indices_tfds:
        if idx_tfds in feature_map_spectral:
            X_test_spectral_list.append(all_X_spectral[feature_map_spectral[idx_tfds]])
        else:
            print(f"Warning: Original TFDS index {idx_tfds} from test_indices_tfds not found in H5 features. Data inconsistency.")
            X_test_spectral_list.append(np.zeros(FEATURE_VECTOR_LENGTH, dtype=np.float32))

    X_train_spectral = np.array(X_train_spectral_list)
    X_test_spectral = np.array(X_test_spectral_list)

    print(f"Aligned X_train_spectral shape: {X_train_spectral.shape}, y_train_spectral shape: {y_train_spectral.shape}")
    print(f"Aligned X_test_spectral shape: {X_test_spectral.shape}, y_test_spectral shape: {y_test_spectral.shape}")

    if X_train_spectral.shape[0] != y_train_spectral.shape[0] or X_test_spectral.shape[0] != y_test_spectral.shape[0]:
        print("Error: Mismatch after aligning spectral features with labels. Halting classification.")
        return

    # Train XGBoost on spectral features
    if X_train_spectral.size > 0 and X_test_spectral.size > 0:
        train_and_evaluate_xgb(X_train_spectral, y_train_spectral, X_test_spectral, y_test_spectral,
                               f"Spectral_FFT_{FFT_REGION_SIZE}x{FFT_REGION_SIZE}_Balanced",
                               class_names_spectral, RESULTS_DIR_XGB_SPECTRAL,
                               perform_scaling=True) # Always scale for spectral features
    else:
        print("Skipping XGBoost training for spectral features as data is empty after alignment.")

    print("--- Spectral Classification Complete ---")



# Step 0: Prepare balanced dataset info (original TFDS indices and broad labels)
prepare_balanced_dataset_info()

# Step 1: Extract spectral features for the balanced subset (if they don't exist)
run_spectral_feature_extraction_if_needed()

# Step 2: Create train/test splits based on the extracted and balanced data
create_spectral_train_test_split_if_needed()

# Step 3: Run classification on the spectral features using the prepared splits
run_spectral_classification()

print("\n=== Full Spectral Pipeline Finished (Balanced Classes) ===")