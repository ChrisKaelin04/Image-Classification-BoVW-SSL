# prepare_balanced_4cat_places_hdf5.py # New suggested filename

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import collections # For Counter
import random # For shuffling within categories
import h5py # Need this!
import gc # For garbage collection

# --- Configuration (ensure these are consistent with other scripts) ---
TFDS_DATA_DIR = r"E:\CV_imgs"
OUTPUT_FEATURES_DIR = r"E:\CV_features" # Base directory for outputs
RANDOM_SEED = 42

# --- Define counts for Train, Validation, and Final Test per broad category ---
TARGET_TRAIN_IMAGES_PER_CATEGORY = 25000
TARGET_VAL_IMAGES_PER_CATEGORY = 3000
TARGET_TEST_IMAGES_PER_CATEGORY = 5000

# Calculate total images needed per category from TFDS scan
# This determines how many we try to collect *per category* during the initial TFDS scan
MAX_IMAGES_NEEDED_PER_CATEGORY = (TARGET_TRAIN_IMAGES_PER_CATEGORY +
                                  TARGET_VAL_IMAGES_PER_CATEGORY +
                                  TARGET_TEST_IMAGES_PER_CATEGORY)

# How many images to scan from the *entire* TFDS 'train' split.
# This should be large enough to find the MAX_IMAGES_NEEDED_PER_CATEGORY for *each* broad category,
# especially important for less common categories.
# (25k+3k+5k) * 4 cats = 132k total needed. Scanning 700k or 1M should be safe.
MAX_TFDS_IMAGES_TO_SCAN = 1000000 # Increased scan limit for safety

# --- Output File Paths ---
OUTPUT_DATA_DIR = os.path.join(OUTPUT_FEATURES_DIR, "balanced_4cat_hdf5") # Subdirectory for HDF5 and encoder
HDF5_OUTPUT_PATH = os.path.join(OUTPUT_DATA_DIR, "balanced_4cat_data.h5") # The HDF5 file path
# We will no longer save the NPZ file with just indices/labels - this is correct
# SPLIT_DATA_FILE = os.path.join(OUTPUT_SPLITS_DIR, "train_test_split_data_4cat_balanced.npz")
LABEL_ENCODER_FILE = os.path.join(OUTPUT_DATA_DIR, "broad_label_encoder_4cat.pkl") # The PKL file


# --- Broad Category Definitions (Same as before) ---
broad_categories_list = [
    "Indoor Residential",
    "Indoor Public/Commercial",
    "Outdoor Natural",
    "Outdoor Urban"
]
broad_category_definitions = {
    'Indoor Public/Commercial': [   'airplane_cabin', 'airport_terminal', 'amusement_arcade', 'arcade', 'aquarium', 'archive', 'arena/hockey', 'arena/performance', 'art_gallery', 'art_school', 'art_studio', 'assembly_line', 'auditorium', 'auto_factory', 'auto_showroom', 'atrium/public', 'bakery/shop', 'ball_pit', 'banquet_hall', 'ballroom', 'bank_vault', 'bar', 'beauty_salon', 'bedchamber', 'beer_hall', 'biology_laboratory', 'basketball_court/indoor', 'bazaar/indoor', 'bookstore', 'booth/indoor', 'bowling_alley', 'boxing_ring', 'burial_chamber', 'bus_interior', 'bus_station/indoor', 'butchers_shop', 'cafeteria', 'candy_store', 'car_interior', 'catacomb', 'chemistry_lab', 'church/indoor', 'classroom', 'clean_room', 'clothing_store', 'coffee_shop', 'cockpit', 'computer_room', 'conference_center', 'conference_room', 'corridor', 'courthouse', 'delicatessen', 'department_store', 'dining_hall', 'discotheque', 'drugstore', 'elevator/door', 'elevator_lobby', 'elevator_shaft', 'engine_room', 'entrance_hall', 'escalator/indoor', 'fabric_store', 'fastfood_restaurant', 'fire_station', 'flea_market/indoor', 'florist_shop/indoor', 'food_court', 'galley', 'general_store/indoor', 'gift_shop', 'greenhouse/indoor', 'gymnasium/indoor', 'hangar/indoor', 'hardware_store', 'hospital_room', 'hotel_room', 'ice_cream_parlor', 'ice_skating_rink/indoor', 'jail_cell', 'jewelry_shop', 'kindergarden_classroom', 'laundromat', 'lecture_room', 'legislative_chamber', 'library/indoor', 'lobby', 'locker_room', 'market/indoor', 'martial_arts_gym', 'mezzanine', 'movie_theater/indoor', 'museum/indoor', 'music_studio', 'natural_history_museum', 'nursing_home', 'office', 'office_building', 'office_cubicles', 'operating_room', 'orchestra_pit', 'parking_garage/indoor', 'pet_shop', 'pharmacy', 'physics_laboratory', 'pizzeria', 'pub/indoor', 'reception', 'recreation_room', 'repair_shop', 'restaurant', 'restaurant_kitchen', 'sauna', 'science_museum', 'server_room', 'shoe_shop', 'shopping_mall/indoor', 'stage/indoor', 'subway_station/platform', 'supermarket', 'sushi_bar', 'swimming_pool/indoor', 'television_studio', 'throne_room', 'ticket_booth', 'toyshop', 'train_interior', 'veterinarians_office', 'waiting_room', 'wet_bar', 'youth_hostel'],
    'Indoor Residential': [   'alcove', 'attic', 'basement', 'bathroom', 'bedroom', 'childs_room', 'closet', 'dining_room', 'dorm_room', 'dressing_room', 'home_office', 'home_theater', 'jacuzzi/indoor', 'kitchen', 'living_room', 'nursery', 'pantry', 'playroom', 'shower', 'staircase', 'storage_room', 'television_room', 'utility_room', 'artists_loft', 'balcony/interior', 'bow_window/indoor', 'garage/indoor'],
    'Outdoor Natural': [   'badlands', 'bamboo_forest', 'beach', 'butte', 'canyon', 'canal/natural', 'campsite', 'cliff', 'coast', 'creek', 'crevasse', 'desert/sand', 'desert/vegetation', 'field/wild', 'forest/broadleaf', 'forest_path', 'forest_road', 'glacier', 'grotto', 'hot_spring', 'ice_floe', 'ice_shelf', 'iceberg', 'igloo', 'islet', 'lagoon', 'lake/natural', 'marsh', 'mountain', 'mountain_path', 'mountain_snowy', 'ocean', 'rainforest', 'river', 'rock_arch', 'sky', 'ski_slope', 'snowfield', 'swamp', 'swimming_hole', 'tree_farm', 'tundra', 'underwater/ocean_deep', 'valley', 'volcano', 'waterfall', 'watering_hole', 'wave'],
    'Outdoor Urban': [   'airfield', 'alley', 'amphitheater', 'amusement_park', 'apartment_building/outdoor', 'aqueduct', 'arch', 'archaelogical_excavation', 'arena/rodeo', 'army_base', 'athletic_field/outdoor', 'balcony/exterior', 'barn', 'barndoor', 'baseball_field', 'bazaar/outdoor', 'beach_house', 'beer_garden', 'berth', 'boardwalk', 'boathouse', 'boat_deck', 'botanical_garden', 'bridge', 'building_facade', 'bullring', 'cabin/outdoor', 'campus', 'canal/urban', 'carrousel', 'castle', 'cemetery', 'chalet', 'church/outdoor', 'construction_site', 'corn_field', 'corral', 'cottage', 'courtyard', 'crosswalk', 'dam', 'desert_road', 'diner/outdoor', 'doorway/outdoor', 'downtown', 'driveway', 'embassy', 'excavation', 'farm', 'field/cultivated', 'field_road', 'fire_escape', 'fishpond', 'football_field', 'formal_garden', 'fountain', 'garage/outdoor', 'gas_station', 'gazebo/exterior', 'general_store/outdoor', 'golf_course', 'greenhouse/outdoor', 'hangar/outdoor', 'harbor', 'hayfield', 'heliport', 'highway', 'hospital', 'hotel/outdoor', 'house', 'hunting_lodge/outdoor', 'ice_skating_rink/outdoor', 'industrial_area', 'inn/outdoor', 'japanese_garden', 'junkyard', 'kasbah', 'kennel/outdoor', 'landing_deck', 'landfill', 'lawn', 'library/outdoor', 'lighthouse', 'loading_dock', 'lock_chamber', 'mansion', 'manufactured_home', 'market/outdoor', 'mausoleum', 'medina', 'moat/water', 'mosque/outdoor', 'motel', 'museum/outdoor', 'oast_house', 'oilrig', 'orchard', 'pagoda', 'palace', 'park', 'parking_garage/outdoor', 'parking_lot', 'pasture', 'patio', 'pavilion', 'phone_booth', 'picnic_area', 'pier', 'playground', 'plaza', 'pond', 'porch', 'promenade', 'raceway', 'racecourse', 'raft', 'railroad_track', 'residential_neighborhood', 'restaurant_patio', 'rice_paddy', 'roof_garden', 'rope_bridge', 'ruin', 'runway', 'sandbox', 'schoolhouse', 'shed', 'shopfront', 'ski_resort', 'skyscraper', 'slum', 'soccer_field', 'stable', 'stadium/baseball', 'stadium/football', 'stadium/soccer', 'stage/outdoor', 'street', 'swimming_pool/outdoor', 'synagogue/outdoor', 'temple/asia', 'topiary_garden', 'tower', 'train_station/platform', 'tree_house', 'trench', 'vegetable_garden', 'viaduct', 'village', 'vineyard', 'volleyball_court/outdoor', 'water_park', 'water_tower', 'wheat_field', 'wind_farm', 'windmill', 'yard', 'zen_garden']}

NUM_BROAD_CATEGORIES = len(broad_categories_list)


def create_balanced_train_val_test_hdf5():
    print(f"--- Starting Data Preparation for {NUM_BROAD_CATEGORIES} Broad Categories (Balanced Train/Val/Test HDF5) ---")
    print(f"Targeting per category: Train={TARGET_TRAIN_IMAGES_PER_CATEGORY}, Val={TARGET_VAL_IMAGES_PER_CATEGORY}, Test={TARGET_TEST_IMAGES_PER_CATEGORY}")
    print(f"Scanning up to {MAX_TFDS_IMAGES_TO_SCAN} images from TFDS 'train' split.")
    print(f"Output HDF5 file: {HDF5_OUTPUT_PATH}")
    print(f"Output Label Encoder file: {LABEL_ENCODER_FILE}")

    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

    # --- 1. Load Dataset Info and Setup Broad Category Mapping ---
    print("\nLoading dataset info for 'places365_small'...")
    try:
        # Load the full dataset, but keep it as TF tensors/objects initially
        # Do NOT use as_numpy() or as_numpy_iterator() on the whole dataset if possible
        # until filtering the list, to potentially save initial memory.
        # However, TFDS load might still use memory. The enumerate().as_numpy_iterator() is necessary
        # for accessing original indices and image data efficiently in a loop.
        full_ds_train_tfds, ds_info = tfds.load('places365_small',
                                                split='train',
                                                data_dir=TFDS_DATA_DIR,
                                                with_info=True,
                                                download=False, # Set to True if you need to download
                                                shuffle_files=False, # Keep False for consistent indexing during scan
                                                as_supervised=False # Get dictionary structure
                                                )
    except Exception as e:
        print(f"Error loading dataset info or base dataset: {e}")
        exit()

    fine_label_names = ds_info.features['label'].names
    num_fine_classes = len(fine_label_names)
    print(f"Total fine-grained classes in Places365 'train' split: {num_fine_classes}")

    print("\nSetting up broad category mapping...")
    fine_to_broad_mapping = {}
    for broad_cat, fine_list in broad_category_definitions.items():
        if broad_cat not in broad_categories_list:
             print(f"ERROR: Broad category '{broad_cat}' defined in broad_category_definitions but not in broad_categories_list. Exiting.")
             exit()
        for fine_name in fine_list:
            if fine_name not in fine_label_names:
                # It's possible some fine labels in the definition aren't in the actual TFDS dataset
                # (e.g., if the list was compiled from a full Places365 list). Print a warning.
                print(f"Warning: Fine label '{fine_name}' in definition for '{broad_cat}' is not in TFDS fine_label_names. Skipping.")
                continue
            # Map fine label string to broad category string
            fine_to_broad_mapping[fine_name] = broad_cat

    print("Fine-grained labels mapped to broad categories.")
    # print("Mapping preview:", list(fine_to_broad_mapping.items())[:10], "...") # Optional preview


    # --- 2. Scan TFDS and Collect Images + Labels for All Splits ---
    print(f"\nScanning TFDS 'train' split and collecting image data for allocation...")

    # Dictionary to hold collected items (including images) per broad category string name
    # Store {'original_tfds_idx', 'image_np', 'broad_category'}
    collected_items_by_broad_cat = {broad_cat: [] for broad_cat in broad_categories_list}
    images_found_count = {broad_cat: 0 for broad_cat in broad_categories_list}

    ds_enumerated = full_ds_train_tfds.enumerate()
    scanned_count = 0
    total_tfds_train_examples = ds_info.splits['train'].num_examples

    # Target scan count per category to collect enough for TRAIN+VAL+TEST
    target_per_category_scan = MAX_IMAGES_NEEDED_PER_CATEGORY
    total_scan_limit = min(total_tfds_train_examples, MAX_TFDS_IMAGES_TO_SCAN)

    # Configure TF to use CPU only during this loop to avoid potential GPU memory conflicts
    # if PyTorch might be running or you have tight memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # We want to ensure no GPU memory is used by TF during this potentially long scan
            # if we anticipate PyTorch needing it immediately after.
            # Setting visible devices to empty works well.
            tf.config.set_visible_devices([], 'GPU')
            print("Configured TensorFlow to use CPU for data scanning.")
        except RuntimeError as e:
            print(f"Could not set TensorFlow to CPU: {e}. Proceeding, but watch GPU memory.")


    with tqdm(total=total_scan_limit, desc="Scanning TFDS & Collecting Images", unit="images") as pbar:
        # Use .as_numpy_iterator() to get data out as NumPy arrays
        for original_tfds_idx_tensor, item in ds_enumerated.as_numpy_iterator():
            scanned_count += 1
            pbar.update(1)

            if scanned_count > MAX_TFDS_IMAGES_TO_SCAN:
                print(f"\nReached MAX_TFDS_IMAGES_TO_SCAN ({MAX_TFDS_IMAGES_TO_SCAN}). Stopping scan.")
                break

            fine_numeric_label = item['label'] # Already numpy from iterator
            original_tfds_idx = int(original_tfds_idx_tensor) # Ensure int

            # Check if fine label is valid and maps to a broad category
            if 0 <= fine_numeric_label < num_fine_classes:
                fine_label_name = fine_label_names[fine_numeric_label]
                if fine_label_name in fine_to_broad_mapping:
                    broad_cat_name = fine_to_broad_mapping[fine_label_name]

                    # Only add if we haven't reached the scan target for this specific broad category
                    if broad_cat_name in collected_items_by_broad_cat and \
                       images_found_count[broad_cat_name] < target_per_category_scan:

                        collected_items_by_broad_cat[broad_cat_name].append(
                            {'original_tfds_idx': original_tfds_idx,
                             'image_np': item['image'], # Store the image NumPy array!
                             'broad_category': broad_cat_name}
                        )
                        images_found_count[broad_cat_name] += 1

                        # Update progress bar postfix with current counts
                        pbar.set_postfix({cat: count for cat, count in images_found_count.items()})

            # Check if all categories have collected enough for all splits
            # Only stop if *all* are >= target_per_category_scan
            all_scan_targets_met = all(images_found_count[cat] >= target_per_category_scan for cat in broad_categories_list)
            if all_scan_targets_met:
                print(f"\nAll categories have collected at least {target_per_category_scan} images. Stopping scan.")
                break

    # Restore TF GPU visibility if it was disabled
    if gpus:
         try:
             tf.config.set_visible_devices(gpus, 'GPU')
             print("Restored TensorFlow GPU visibility.")
         except RuntimeError as e:
             print(f"Could not restore TensorFlow GPU visibility: {e}.")


    print("\n--- Image Collection Summary (before allocating to splits) ---")
    all_categories_sufficiently_filled_for_scan = True
    total_collected_count = 0
    for broad_cat in broad_categories_list:
        # Use the actual collected count for the summary
        count = len(collected_items_by_broad_cat[broad_cat])
        total_collected_count += count
        print(f"Category '{broad_cat}': Collected {count} images (Scan Target: {target_per_category_scan})")
        if count < target_per_category_scan:
            all_categories_sufficiently_filled_for_scan = False
            print(f"  WARNING: Category '{broad_cat}' did not reach the scan target needed for all splits. "
                  f"Needed {target_per_category_scan}, found {count}. "
                  "Splits for this category might be smaller than targeted.")

    if not all_categories_sufficiently_filled_for_scan:
        print("\nWARNING: Not all categories have enough images for the planned train/val/test splits. "
              "The script will proceed but splits might be smaller than targeted.")
    print(f"Total images collected from scan: {total_collected_count}")

    if total_collected_count == 0:
        print("ERROR: No images were collected. Check broad category mappings, TFDS data_dir, and MAX_TFDS_IMAGES_TO_SCAN.")
        exit()

    # Release the full TFDS dataset object to free memory
    del full_ds_train_tfds
    gc.collect() # Encourage garbage collection
    print("Released full TFDS dataset object.")


    # --- 3. Allocate to Train, Validation, and Test sets (Ensuring Disjoint Sets) ---
    print("\nAllocating collected images to Train, Validation, and Test splits...")
    train_items, val_items, test_items = [], [], []

    for broad_cat_name in broad_categories_list:
        available_items = collected_items_by_broad_cat[broad_cat_name]
        num_available_for_cat = len(available_items)

        # Shuffle images *within* this category before assigning to splits
        # Using a random seed for reproducibility
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED) # Use Python's random for list shuffling
        random.shuffle(available_items)

        current_pos = 0

        # Allocate to Training set
        num_to_take_train = min(TARGET_TRAIN_IMAGES_PER_CATEGORY, num_available_for_cat)
        train_items.extend(available_items[current_pos : current_pos + num_to_take_train])
        current_pos += num_to_take_train

        # Allocate to Validation set from remaining
        num_remaining_after_train = num_available_for_cat - current_pos
        num_to_take_val = min(TARGET_VAL_IMAGES_PER_CATEGORY, num_remaining_after_train)
        val_items.extend(available_items[current_pos : current_pos + num_to_take_val])
        current_pos += num_to_take_val

        # Allocate to Final Test set from remaining
        num_remaining_after_val = num_available_for_cat - current_pos
        num_to_take_test = min(TARGET_TEST_IMAGES_PER_CATEGORY, num_remaining_after_val)
        test_items.extend(available_items[current_pos : current_pos + num_to_take_test])
        # current_pos += num_to_take_test # Not strictly needed after this


        print(f"  Category '{broad_cat_name}': "
              f"Allocated Train={num_to_take_train}, Val={num_to_take_val}, Test={num_to_take_test} "
              f"(from {num_available_for_cat} collected)")
        if num_to_take_train < TARGET_TRAIN_IMAGES_PER_CATEGORY or \
           num_to_take_val < TARGET_VAL_IMAGES_PER_CATEGORY or \
           num_to_take_test < TARGET_TEST_IMAGES_PER_CATEGORY:
            print(f"    NOTE: Category '{broad_cat_name}' did not meet all targets due to insufficient collected images.")


    # Release the category-grouped data to potentially free memory before stacking
    del collected_items_by_broad_cat
    gc.collect()
    print("Released category-grouped collected items.")


    # Shuffle each final list globally now that they are combined from all categories
    # This mixes images from different categories within each split
    # Using different seeds for each split shuffle for better mixing
    np.random.seed(RANDOM_SEED + 1)
    random.seed(RANDOM_SEED + 1)
    random.shuffle(train_items)

    np.random.seed(RANDOM_SEED + 2)
    random.seed(RANDOM_SEED + 2)
    random.shuffle(val_items)

    np.random.seed(RANDOM_SEED + 3)
    random.seed(RANDOM_SEED + 3)
    random.shuffle(test_items)

    print("\n--- Final Split Summary ---")
    print(f"Total Training set size: {len(train_items)}")
    print(f"Total Validation set size: {len(val_items)}")
    print(f"Total Final Test set size: {len(test_items)}")

    if len(train_items) == 0 or len(val_items) == 0 or len(test_items) == 0:
         print("ERROR: One or more splits are empty. Cannot proceed with HDF5 creation. Check your targets and data.")
         exit()


    # --- 4. Prepare Data and Encode Labels for HDF5 ---
    print("\nPreparing data and encoding labels for HDF5...")

    # Extract lists of images and string labels per split
    train_images_np_list = [item['image_np'] for item in train_items]
    train_broad_labels_str = [item['broad_category'] for item in train_items]

    val_images_np_list = [item['image_np'] for item in val_items]
    val_broad_labels_str = [item['broad_category'] for item in val_items]

    test_images_np_list_final = [item['image_np'] for item in test_items]
    test_broad_labels_str_final = [item['broad_category'] for item in test_items]

    # Release item lists to free memory
    del train_items, val_items, test_items
    gc.collect()
    print("Released split item lists.")

    # Encode Broad Category String Labels to Numeric Labels
    label_encoder = LabelEncoder()
    # Fit on the predefined list of broad categories to ensure consistent mapping
    label_encoder.fit(broad_categories_list)

    train_broad_labels_numeric = label_encoder.transform(train_broad_labels_str)
    val_broad_labels_numeric = label_encoder.transform(val_broad_labels_str)
    test_broad_labels_numeric_final = label_encoder.transform(test_broad_labels_str_final)

    print("Broad Category String to Numeric Mapping (based on LabelEncoder):")
    # Print the mapping based on the encoder's learned classes
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  '{class_name}': {i}")

    # Release string label lists
    del train_broad_labels_str, val_broad_labels_str, test_broad_labels_str_final
    gc.collect()
    print("Released string label lists.")

    # Stack image lists into single NumPy arrays
    # THIS IS WHERE THE LARGEST TEMPORARY MEMORY ALLOCATION WILL OCCUR
    # Ensure you have enough RAM (estimated ~20GB for your targets + overhead)
    print("\nStacking image lists into large NumPy arrays for HDF5 writing...")
    try:
        # Only stack if there's data to stack
        train_images_np_array = np.stack(train_images_np_list) if train_images_np_list else np.array([], dtype=np.uint8) # Assuming image dtype is uint8
        val_images_np_array = np.stack(val_images_np_list) if val_images_np_list else np.array([], dtype=np.uint8)
        test_images_np_array_final = np.stack(test_images_np_list_final) if test_images_np_list_final else np.array([], dtype=np.uint8)
        print("Image arrays stacked successfully.")
    except Exception as e:
        print(f"ERROR stacking image arrays. Ran out of memory? {e}. Exiting.")
        exit()

    # Release image lists to free memory after stacking
    del train_images_np_list, val_images_np_list, test_images_np_list_final
    gc.collect()
    print("Released image lists after stacking.")


    # --- 5. Write Data to HDF5 File ---
    print(f"\nWriting data to HDF5 file: {HDF5_OUTPUT_PATH}")

    # Get image shape for chunking from non-empty array if possible
    img_shape = None
    if train_images_np_array.shape[0] > 0:
        img_shape = train_images_np_array.shape[1:] # (H, W, C)
    elif val_images_np_array.shape[0] > 0:
         img_shape = val_images_np_array.shape[1:]
    elif test_images_np_array_final.shape[0] > 0:
         img_shape = test_images_np_array_final.shape[1:]

    hdf5_chunk_shape = None
    if img_shape:
         # Chunk size: Process one image at a time in HDF5 reads
         # Or use a multiple of 1 for better compression/writing performance, e.g., 32 or 64
         # Chunking along the first dimension (number of images)
         hdf5_chunk_shape = (32,) + img_shape # Example chunking
         print(f"Using HDF5 chunk shape: {hdf5_chunk_shape}")
    else:
         print("Warning: All image arrays are empty, cannot determine chunk shape. H5py will choose.")


    try:
        with h5py.File(HDF5_OUTPUT_PATH, 'w') as f:
            # Train Split
            print(f"Writing train data ({train_images_np_array.shape[0]} images)...")
            # Only create datasets if there is data
            if train_images_np_array.shape[0] > 0:
                f.create_group('train')
                f.create_dataset('train/images', data=train_images_np_array, chunks=hdf5_chunk_shape, compression="gzip", dtype=train_images_np_array.dtype) # Preserve original dtype
                f.create_dataset('train/labels', data=train_broad_labels_numeric.astype(np.int8), compression="gzip")
            else:
                 print("  Train dataset is empty, skipping train group/datasets.")


            # Validation Split
            print(f"Writing validation data ({val_images_np_array.shape[0]} images)...")
            if val_images_np_array.shape[0] > 0:
                f.create_group('val')
                f.create_dataset('val/images', data=val_images_np_array, chunks=hdf5_chunk_shape, compression="gzip", dtype=val_images_np_array.dtype) # Preserve original dtype
                f.create_dataset('val/labels', data=val_broad_labels_numeric.astype(np.int8), compression="gzip")
            else:
                print("  Validation dataset is empty, skipping val group/datasets.")

            # Test Split (Final Evaluation)
            print(f"Writing final test data ({test_images_np_array_final.shape[0]} images)...")
            if test_images_np_array_final.shape[0] > 0:
                f.create_group('test') # Use 'test' group name
                f.create_dataset('test/images', data=test_images_np_array_final, chunks=hdf5_chunk_shape, compression="gzip", dtype=test_images_np_array_final.dtype) # Preserve original dtype
                f.create_dataset('test/labels', data=test_broad_labels_numeric_final.astype(np.int8), compression="gzip")
            else:
                 print("  Test dataset is empty, skipping test group/datasets.")

        print("Successfully wrote data to HDF5 file.")

    except Exception as e:
        print(f"ERROR writing to HDF5 file: {e}. Exiting.")
        # Clean up potentially incomplete file
        if os.path.exists(HDF5_OUTPUT_PATH):
             print(f"Removing incomplete HDF5 file: {HDF5_OUTPUT_PATH}")
             try:
                 os.remove(HDF5_OUTPUT_PATH)
             except OSError as remove_e:
                 print(f"Error removing file {HDF5_OUTPUT_PATH}: {remove_e}")
        exit()

    # Release the large NumPy arrays after writing to HDF5
    del train_images_np_array, val_images_np_array, test_images_np_array_final
    gc.collect()
    print("Released image arrays after writing to HDF5.")


    # --- 6. Save the Label Encoder ---
    print(f"\nSaving label encoder to: {LABEL_ENCODER_FILE}")
    try:
        with open(LABEL_ENCODER_FILE, 'wb') as f:
            pickle.dump(label_encoder, f)
        print("Successfully saved label encoder.")
    except Exception as e:
        print(f"ERROR saving label encoder file: {e}")


    print("\n--- Data Preparation (Balanced Train/Val/Test HDF5) Complete ---")
    print(f"HDF5 file created at: {HDF5_OUTPUT_PATH}")
    print(f"Label encoder saved at: {LABEL_ENCODER_FILE}")


# --- Execute Main Pipeline ---
# You could call create_balanced_train_val_test_hdf5() directly,
# but keeping the run() function provides a single entry point if preferred.
def run():
    # You might need to add TensorFlow GPU configuration here if TFDS load requires it
    # even just for info loading, depending on your TF installation.
    # For data loading in the loop, we explicitly set it to CPU.
    # Let's remove the redundant TF GPU config from here, it's already handled inside the main function
    # if gpus:
    #     try:
    #         tf.config.experimental.set_memory_growth(gpus[0], True)
    #         print("Configured TensorFlow GPU memory growth for initial load.")
    #     except RuntimeError as e:
    #         print(f"Error setting TF GPU options for initial load: {e}. TF might still grab GPU memory.")
    # else:
    #     print("No GPUs detected by TensorFlow.")
    print("Note: TensorFlow GPU configuration is handled within create_balanced_train_val_test_hdf5().")

    create_balanced_train_val_test_hdf5()

    # Ensure TensorFlow releases GPU memory if it acquired any before exiting
    # (might not be fully effective depending on context)
    # This is also handled inside the main function now.
    # if gpus:
    #     try:
    #         tf.config.set_visible_devices([], 'GPU')
    #         print("Attempted to release TensorFlow GPU memory.")
    #     except RuntimeError as e:
    #          print(f"Could not release TensorFlow GPU memory: {e}.")

