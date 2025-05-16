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

# --- Configuration (ensure these are consistent with other scripts) ---
TFDS_DATA_DIR = r"E:\CV_imgs"
OUTPUT_FEATURES_DIR = r"E:\CV_features"
RANDOM_SEED = 42

# --- NEW: Define counts for Train, Validation, and Final Test per broad category ---
# Total Train Images = 100k (25k * 4)
TARGET_TRAIN_IMAGES_PER_CATEGORY = 25000
# Total Validation Images for fine-tuning = 12k (3k * 4) - adjust as needed
TARGET_VAL_IMAGES_PER_CATEGORY = 3000
# Total Final Test Images = 20k (5k * 4)
TARGET_TEST_IMAGES_PER_CATEGORY = 5000

# Calculate total images needed per category from TFDS scan
MAX_IMAGES_NEEDED_PER_CATEGORY = (TARGET_TRAIN_IMAGES_PER_CATEGORY +
                                  TARGET_VAL_IMAGES_PER_CATEGORY +
                                  TARGET_TEST_IMAGES_PER_CATEGORY)

MAX_TFDS_IMAGES_TO_SCAN = 700000 # Increased to ensure enough images are found for all 3 splits.
                                  # Adjust based on rarity of your broad categories.
                                  # (25k+3k+5k) * 4 cats = 132k. Scanning 700k should be plenty.



# Define the broad categories (as in your original script)
broad_categories_list = [
    "Indoor Residential",
    "Indoor Public/Commercial",
    "Outdoor Natural",
    "Outdoor Urban"
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

def create_all_splits_for_finetuning(): # Renamed function
    print(f"--- Starting Data Preparation for {len(broad_categories_list)} Broad Categories (Train/Val/Test) ---")
    print(f"Targeting per category: Train={TARGET_TRAIN_IMAGES_PER_CATEGORY}, Val={TARGET_VAL_IMAGES_PER_CATEGORY}, Test={TARGET_TEST_IMAGES_PER_CATEGORY}")
    print(f"Scanning up to {MAX_TFDS_IMAGES_TO_SCAN} images from TFDS.")

    # --- 1. Load Dataset Info for Labels ---
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

    # --- 2. Setup Broad Category Mapping ---
    print("\nSetting up broad category mapping...")
    fine_to_broad_mapping = {}
    for broad_cat, fine_list in broad_category_definitions.items():
        for fine_name in fine_list:
            if fine_name not in fine_label_names:
                print(f"Warning: Fine label '{fine_name}' in your definition is not in TFDS fine_label_names. Skipping.")
                continue
            fine_to_broad_mapping[fine_name] = broad_cat

    # --- 3. Select Images for Balanced Broad Categories ---
    print(f"\nScanning TFDS 'train' split to select images for all splits...")

    selected_images_data = {broad_cat: [] for broad_cat in broad_categories_list}
    images_found_count = {broad_cat: 0 for broad_cat in broad_categories_list}
    total_selected_count = 0
    # Target for the initial scan is to get enough for ALL splits for each category
    target_per_category_scan = MAX_IMAGES_NEEDED_PER_CATEGORY
    target_total_images_scan = len(broad_categories_list) * target_per_category_scan

    ds_enumerated = full_ds_train_tfds.enumerate()
    scanned_count = 0

    with tqdm(total=min(ds_info.splits['train'].num_examples, MAX_TFDS_IMAGES_TO_SCAN), desc="Scanning TFDS") as pbar:
        for original_tfds_idx_tensor, item in ds_enumerated:
            scanned_count += 1
            pbar.update(1)

            if scanned_count > MAX_TFDS_IMAGES_TO_SCAN:
                print(f"\nReached MAX_TFDS_IMAGES_TO_SCAN ({MAX_TFDS_IMAGES_TO_SCAN}). Stopping scan.")
                break
            # No need to check total_selected_count here yet, we want to fill each category up to MAX_IMAGES_NEEDED_PER_CATEGORY

            fine_numeric_label = item['label'].numpy()
            original_tfds_idx = original_tfds_idx_tensor.numpy()

            if 0 <= fine_numeric_label < num_fine_classes:
                fine_label_name = fine_label_names[fine_numeric_label]
                if fine_label_name in fine_to_broad_mapping:
                    broad_cat_name = fine_to_broad_mapping[fine_label_name]
                    if broad_cat_name in images_found_count and \
                       images_found_count[broad_cat_name] < target_per_category_scan: # Use target_per_category_scan
                        selected_images_data[broad_cat_name].append(
                            {'original_tfds_idx': original_tfds_idx, 'broad_category': broad_cat_name}
                        )
                        images_found_count[broad_cat_name] += 1
                        total_selected_count += 1 # This tracks total images added to selected_images_data
                        pbar.set_postfix({cat: count for cat, count in images_found_count.items()})
            # Check if all categories have reached their scan target
            all_scan_targets_met = all(images_found_count[cat] >= target_per_category_scan for cat in broad_categories_list)
            if all_scan_targets_met:
                print(f"\nAll categories have collected at least {target_per_category_scan} images. Stopping scan.")
                break


    print("\n--- Image Collection Summary (before allocating to splits) ---")
    all_categories_sufficiently_filled_for_scan = True
    for broad_cat in broad_categories_list:
        count = images_found_count[broad_cat]
        print(f"Category '{broad_cat}': Collected {count} images (Scan Target: {target_per_category_scan})")
        if count < target_per_category_scan: # Check against scan target
            all_categories_sufficiently_filled_for_scan = False
            print(f"  WARNING: Category '{broad_cat}' did not reach the scan target needed for all splits. "
                  f"Needed {target_per_category_scan}, found {count}. "
                  "Consider increasing MAX_TFDS_IMAGES_TO_SCAN or check category definition.")

    if not all_categories_sufficiently_filled_for_scan:
        print("WARNING: Not all categories have enough images for the planned train/val/test splits. "
              "The script will proceed but splits might be smaller than targeted.")
    print(f"Total images collected from scan: {total_selected_count}")

    if total_selected_count == 0:
        print("ERROR: No images were collected. Check mappings and TFDS scan logic.")
        exit()

    # --- 4. Prepare for Splitting ---
    # Consolidate all selected image data and shuffle images *within each category* before taking subsets
    all_collected_items_for_splits = {broad_cat: [] for broad_cat in broad_categories_list}
    for broad_cat in broad_categories_list:
        # Shuffle the collected images for this category before assigning
        np.random.seed(RANDOM_SEED) # Ensure consistent shuffling
        shuffled_category_images = list(selected_images_data[broad_cat]) # Make a mutable copy
        random.shuffle(shuffled_category_images)
        all_collected_items_for_splits[broad_cat] = shuffled_category_images


    # --- 5. Allocate to Train, Validation, and Test sets (Ensuring Disjoint Sets) ---
    print("\nAllocating images to Train, Validation, and Final Test splits...")
    train_items, val_items, test_items = [], [], []

    for broad_cat_name in broad_categories_list:
        available_items = all_collected_items_for_splits[broad_cat_name]
        num_available_for_cat = len(available_items)
        current_pos = 0

        # Allocate to Training set
        num_to_take_train = min(TARGET_TRAIN_IMAGES_PER_CATEGORY, num_available_for_cat - current_pos)
        train_items.extend(available_items[current_pos : current_pos + num_to_take_train])
        current_pos += num_to_take_train

        # Allocate to Validation set
        num_to_take_val = 0
        if current_pos < num_available_for_cat:
            num_to_take_val = min(TARGET_VAL_IMAGES_PER_CATEGORY, num_available_for_cat - current_pos)
            val_items.extend(available_items[current_pos : current_pos + num_to_take_val])
            current_pos += num_to_take_val

        # Allocate to Final Test set
        num_to_take_test = 0
        if current_pos < num_available_for_cat:
            num_to_take_test = min(TARGET_TEST_IMAGES_PER_CATEGORY, num_available_for_cat - current_pos)
            test_items.extend(available_items[current_pos : current_pos + num_to_take_test])
            current_pos += num_to_take_test

        print(f"  Category '{broad_cat_name}': "
              f"Train={num_to_take_train}, Val={num_to_take_val}, Test={num_to_take_test} "
              f"(from {num_available_for_cat} available)")
        if num_to_take_train < TARGET_TRAIN_IMAGES_PER_CATEGORY or \
           num_to_take_val < TARGET_VAL_IMAGES_PER_CATEGORY or \
           num_to_take_test < TARGET_TEST_IMAGES_PER_CATEGORY:
            print(f"    WARNING: Category '{broad_cat_name}' did not meet all targets for splits.")


    # Shuffle each list globally now that they are balanced per category
    np.random.seed(RANDOM_SEED)
    random.shuffle(train_items)
    random.shuffle(val_items)
    random.shuffle(test_items)

    # Extract original TFDS indices and string labels for each split
    train_original_tfds_indices = [item['original_tfds_idx'] for item in train_items]
    train_broad_labels_str = [item['broad_category'] for item in train_items]

    val_original_tfds_indices = [item['original_tfds_idx'] for item in val_items]
    val_broad_labels_str = [item['broad_category'] for item in val_items]

    test_original_tfds_indices_final = [item['original_tfds_idx'] for item in test_items]
    test_broad_labels_str_final = [item['broad_category'] for item in test_items]


    # --- 6. Encode Broad Category String Labels to Numeric Labels ---
    print("\nEncoding broad category string labels to numeric labels...")
    label_encoder = LabelEncoder()
    # Fit on the predefined list to ensure consistent mapping even if some categories are missing in a small sample
    label_encoder.fit(broad_categories_list)

    train_broad_labels_numeric = label_encoder.transform(train_broad_labels_str)
    val_broad_labels_numeric = label_encoder.transform(val_broad_labels_str)
    test_broad_labels_numeric_final = label_encoder.transform(test_broad_labels_str_final)

    print("Broad Category String to Numeric Mapping (based on LabelEncoder):")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  '{class_name}': {i}")


    # --- 7. Print Split Summaries ---
    print("\n--- Final Split Summary ---")
    print(f"Training set size: {len(train_original_tfds_indices)} (Indices), {len(train_broad_labels_numeric)} (Labels)")
    print(f"Validation set size: {len(val_original_tfds_indices)}, {len(val_broad_labels_numeric)}")
    print(f"Final Test set size: {len(test_original_tfds_indices_final)}, {len(test_broad_labels_numeric_final)}")

    print("\nTrain broad label distribution (numeric):", np.bincount(train_broad_labels_numeric, minlength=len(label_encoder.classes_)))
    print("Validation broad label distribution (numeric):", np.bincount(val_broad_labels_numeric, minlength=len(label_encoder.classes_)))
    print("Final Test broad label distribution (numeric):", np.bincount(test_broad_labels_numeric_final, minlength=len(label_encoder.classes_)))
    for i, class_name in enumerate(label_encoder.classes_):
        tr_c = np.sum(train_broad_labels_numeric == i)
        v_c = np.sum(val_broad_labels_numeric == i)
        te_c = np.sum(test_broad_labels_numeric_final == i)
        print(f"  Category '{class_name}' (ID {i}): Train={tr_c}, Val={v_c}, Test={te_c}")


    # --- 8. Save the Splits and Label Encoder ---
    output_splits_dir = os.path.join(OUTPUT_FEATURES_DIR, "all_splits_data_4cat") # New subdir name
    os.makedirs(output_splits_dir, exist_ok=True)
    split_data_file = os.path.join(output_splits_dir, "all_splits_data_4cat.npz") # New NPZ file name
    label_encoder_file = os.path.join(output_splits_dir, "broad_label_encoder_4cat.pkl") # New PKL file name

    np.savez_compressed( # Use savez_compressed for smaller file size
        split_data_file,
        train_indices=np.array(train_original_tfds_indices, dtype=np.int32),
        train_labels_numeric=train_broad_labels_numeric.astype(np.int8),
        # train_labels_str=np.array(train_broad_labels_str), # Optional to save string labels

        val_indices=np.array(val_original_tfds_indices, dtype=np.int32),
        val_labels_numeric=val_broad_labels_numeric.astype(np.int8),
        # val_labels_str=np.array(val_broad_labels_str), # Optional

        test_indices_final=np.array(test_original_tfds_indices_final, dtype=np.int32),
        test_labels_numeric_final=test_broad_labels_numeric_final.astype(np.int8)
        # test_labels_str_final=np.array(test_broad_labels_str_final) # Optional
    )
    print(f"\nSaved train/validation/test ORIGINAL TFDS indices and labels to: {split_data_file}")

    with open(label_encoder_file, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Saved label encoder to: {label_encoder_file}")

    print("\n--- Data Preparation for All Splits Complete ---")
    print(f"The indices in '{split_data_file}' are ORIGINAL 0-based indices from 'places365_small/train'.enumerate().")