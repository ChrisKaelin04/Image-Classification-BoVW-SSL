import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import collections # For Counter

# --- Configuration (ensure these are consistent with other scripts) ---
TFDS_DATA_DIR = r"E:\CV_imgs" # As used in feature extraction
OUTPUT_FEATURES_DIR = r"E:\CV_features" # Where the NPZ/PKL will be saved
RANDOM_SEED = 42
TARGET_IMAGES_PER_BROAD_CATEGORY = 25000 # Desired number of images per broad category
# Adjust TARGET_IMAGES_PER_BROAD_CATEGORY based on total TFDS_SUBSET_SIZE
# For example, if you want 100k total images, and 4 categories, this is 25k per category.
# You might need to scan more than TARGET_IMAGES_PER_BROAD_CATEGORY * num_categories
# from TFDS to find enough samples, especially for rarer broad categories.
MAX_TFDS_IMAGES_TO_SCAN = 500000 # How many images to scan from TFDS to find your samples.
                                  # Increase if you don't find enough for all categories.


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

def create_balanced_split_and_labels():
    print(f"--- Starting Balanced Data Preparation for {len(broad_categories_list)} Broad Categories ---")
    print(f"Targeting {TARGET_IMAGES_PER_BROAD_CATEGORY} images per broad category.")
    print(f"Scanning up to {MAX_TFDS_IMAGES_TO_SCAN} images from TFDS.")

    # --- 1. Load Dataset Info for Labels ---
    print("\nLoading dataset info for 'places365_small'...")
    try:
        # Load the 'train' split. We'll iterate through it.
        # No need to shuffle here yet, as we'll be picking based on category.
        full_ds_train_tfds, ds_info = tfds.load('places365_small',
                                                split='train',
                                                data_dir=TFDS_DATA_DIR,
                                                with_info=True,
                                                download=False,
                                                shuffle_files=False) # Shuffle_files=False for deterministic scan
    except Exception as e:
        print(f"Error loading dataset info or base dataset: {e}")
        exit()

    fine_label_names = ds_info.features['label'].names
    num_fine_classes = len(fine_label_names)
    print(f"Total fine-grained classes in Places365: {num_fine_classes}")

    # --- 2. Setup Broad Category Mapping (Reverse mapping is useful too) ---
    print("\nSetting up broad category mapping...")
    fine_to_broad_mapping = {}
    for broad_cat, fine_list in broad_category_definitions.items():
        for fine_name in fine_list:
            if fine_name not in fine_label_names:
                print(f"Warning: Fine label '{fine_name}' in your definition is not in TFDS fine_label_names. Skipping.")
                continue
            fine_to_broad_mapping[fine_name] = broad_cat

    # Check coverage (optional but good)
    # ... (your existing checks for mapping completeness can go here) ...

    # --- 3. Select Images for Balanced Broad Categories ---
    print(f"\nScanning TFDS 'train' split to select images for balanced broad categories...")

    # We need to store the TFDS *original index* or some unique identifier if we are not
    # loading images into memory here. TFDS doesn't easily provide a persistent unique ID
    # other than iterating in a fixed order.
    # For simplicity, we'll collect (enumerated_original_tfds_index, broad_category_name)
    # The enumerated_original_tfds_index will be the 0-based index IF TFDS was read sequentially without shuffle.
    # This is CRITICAL for the feature extractor to find the *exact same* images later.

    selected_images_data = {broad_cat: [] for broad_cat in broad_categories_list}
    images_found_count = {broad_cat: 0 for broad_cat in broad_categories_list}
    total_selected_count = 0
    target_total_images = len(broad_categories_list) * TARGET_IMAGES_PER_BROAD_CATEGORY

    # Enumerate the dataset to get a consistent index for each item
    # This ds_enumerated will give (0, item_0), (1, item_1), ... from the *original* 'train' split order
    ds_enumerated = full_ds_train_tfds.enumerate()

    scanned_count = 0
    with tqdm(total=min(ds_info.splits['train'].num_examples, MAX_TFDS_IMAGES_TO_SCAN), desc="Scanning TFDS") as pbar:
        for original_tfds_idx_tensor, item in ds_enumerated:
            scanned_count += 1
            pbar.update(1)

            if scanned_count > MAX_TFDS_IMAGES_TO_SCAN:
                print(f"\nReached MAX_TFDS_IMAGES_TO_SCAN ({MAX_TFDS_IMAGES_TO_SCAN}). Stopping scan.")
                break
            if total_selected_count >= target_total_images:
                print(f"\nReached target total images ({target_total_images}). Stopping scan.")
                break # All categories filled

            fine_numeric_label = item['label'].numpy()
            original_tfds_idx = original_tfds_idx_tensor.numpy() # This is the 0, 1, 2... index from the *original* train split

            if 0 <= fine_numeric_label < num_fine_classes:
                fine_label_name = fine_label_names[fine_numeric_label]
                if fine_label_name in fine_to_broad_mapping:
                    broad_cat_name = fine_to_broad_mapping[fine_label_name]
                    if broad_cat_name in images_found_count and \
                       images_found_count[broad_cat_name] < TARGET_IMAGES_PER_BROAD_CATEGORY:
                        # Store the original_tfds_idx. The feature extractor will need to iterate
                        # through the *same un-shuffled, enumerated* TFDS 'train' split
                        # and pick out images whose original_tfds_idx matches these.
                        selected_images_data[broad_cat_name].append(
                            {'original_tfds_idx': original_tfds_idx, 'broad_category': broad_cat_name}
                        )
                        images_found_count[broad_cat_name] += 1
                        total_selected_count += 1
                        pbar.set_postfix({cat: count for cat, count in images_found_count.items()})
                # else: fine label not in our broad mapping (should be caught by earlier checks)
            # else: invalid fine label (should be rare)

    print("\n--- Image Selection Summary ---")
    all_categories_filled = True
    for broad_cat in broad_categories_list:
        count = images_found_count[broad_cat]
        print(f"Category '{broad_cat}': Selected {count} images (Target: {TARGET_IMAGES_PER_BROAD_CATEGORY})")
        if count < TARGET_IMAGES_PER_BROAD_CATEGORY:
            all_categories_filled = False
            print(f"  WARNING: Category '{broad_cat}' did not reach the target. Consider increasing MAX_TFDS_IMAGES_TO_SCAN or check category definition.")

    if not all_categories_filled:
        print("WARNING: Not all categories reached their target counts.")
    print(f"Total images selected: {total_selected_count}")

    if total_selected_count == 0:
        print("ERROR: No images were selected. Check mappings and TFDS scan logic.")
        exit()

    # --- 4. Prepare for Train/Test Split ---
    # Consolidate all selected image data: list of {'original_tfds_idx': X, 'broad_category': Y}
    all_selected_items_for_split = []
    for broad_cat in broad_categories_list:
        all_selected_items_for_split.extend(selected_images_data[broad_cat])

    # Shuffle this consolidated list before splitting to ensure randomness in train/test
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(all_selected_items_for_split)

    # Extract the original TFDS indices and broad category labels for splitting
    # These original_tfds_indices are what we save.
    # The feature extraction script will iterate TFDS train.enumerate() and pick images
    # whose enumerated index matches one of these saved indices.
    split_indices = [item['original_tfds_idx'] for item in all_selected_items_for_split]
    split_labels_str = [item['broad_category'] for item in all_selected_items_for_split]

    # --- 5. Encode Broad Category String Labels to Numeric Labels ---
    print("\nEncoding broad category string labels to numeric labels...")
    label_encoder = LabelEncoder()
    label_encoder.fit(broad_categories_list) # Fit on predefined list for consistency
    numeric_broad_labels_for_split = label_encoder.transform(split_labels_str)

    print("Broad Category String to Numeric Mapping (based on LabelEncoder):")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  '{class_name}': {i}")

    # --- 6. Create Train/Test Split ---
    print("\nCreating train/test split (stratified by numeric broad labels)...")
    # We are splitting the `split_indices` (which are original_tfds_idx values)
    train_original_tfds_indices, test_original_tfds_indices, \
    train_broad_labels_numeric, test_broad_labels_numeric, \
    train_broad_labels_str, test_broad_labels_str = train_test_split(
        split_indices,
        numeric_broad_labels_for_split,
        split_labels_str,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=numeric_broad_labels_for_split
    )
    print(f"Training set size: {len(train_original_tfds_indices)} original TFDS indices")
    print(f"Test set size: {len(test_original_tfds_indices)} original TFDS indices")

    print("\nTrain broad label distribution (numeric):", np.bincount(train_broad_labels_numeric, minlength=len(label_encoder.classes_)))
    print("Test broad label distribution (numeric):", np.bincount(test_broad_labels_numeric, minlength=len(label_encoder.classes_)))
    for i, class_name in enumerate(label_encoder.classes_):
        train_count = np.sum(train_broad_labels_numeric == i)
        test_count = np.sum(test_broad_labels_numeric == i)
        print(f"  Category '{class_name}' (ID {i}): Train={train_count}, Test={test_count}")

    # --- 7. Save the Splits and Label Encoder ---
    output_splits_dir = os.path.join(OUTPUT_FEATURES_DIR, "train_test_splits_4cat_balanced") # New subdir
    os.makedirs(output_splits_dir, exist_ok=True)
    split_data_file = os.path.join(output_splits_dir, "train_test_split_data_4cat_balanced.npz") # New name
    label_encoder_file = os.path.join(output_splits_dir, "broad_label_encoder_4cat_balanced.pkl") # New name

    np.savez(
        split_data_file,
        train_indices=np.array(train_original_tfds_indices, dtype=np.int32), # These are ORIGINAL TFDS indices
        test_indices=np.array(test_original_tfds_indices, dtype=np.int32),   # These are ORIGINAL TFDS indices
        train_labels_numeric=train_broad_labels_numeric.astype(np.int8),
        test_labels_numeric=test_broad_labels_numeric.astype(np.int8),
        train_labels_str=np.array(train_broad_labels_str),
        test_labels_str=np.array(test_broad_labels_str)
    )
    print(f"\nSaved train/test ORIGINAL TFDS indices and labels to: {split_data_file}")

    with open(label_encoder_file, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Saved label encoder to: {label_encoder_file}")

    print("\n--- Balanced Data Preparation Complete ---")
    print(f"The 'train_indices' and 'test_indices' in '{split_data_file}' are ORIGINAL 0-based indices")
    print(f"referring to the order of items in the TFDS 'places365_small/train' split")
    print(f"as obtained by 'full_ds_train_tfds.enumerate()'.")
    print(f"Your feature extraction script MUST iterate this original TFDS train split and pick items by these indices.")