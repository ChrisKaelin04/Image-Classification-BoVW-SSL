# We need to get 4 broad categories of images from the dataset. We will use the following categories:
# 1. Indoor residential
# 2. Indoor Public/Commercial
# 3. Outdoor Natural
# 4. Outdoor Urban
import tensorflow_datasets as tfds
import numpy as np
import h5py
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

TFDS_DATA_DIR = "E:\CV_imgs"
OUTPUT_FEATURES_DIR = "E:\CV_features"
SUBSET_SIZE = 100000 # Should match what you used in feature extraction
RANDOM_SEED = 30 # For reproducibility

# Define the broad categories, AI generated (hand checked)

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

def split_data():
    # Load Dataset Info for Labels
    print("Loading dataset info...")
    try:
        ds_info = tfds.load('places365_small',
                            data_dir=TFDS_DATA_DIR,
                            with_info=True,
                            download=False)[1]
    except Exception as e:
        print(f"Error loading dataset info: {e}")
        exit()

    fine_label_names = ds_info.features['label'].names
    print(f"Total fine-grained classes: {len(fine_label_names)}")
    
    category_mapping_fine_to_broad = {}
    all_fine_labels_in_mapping = set()
    for broad_cat, fine_list in broad_category_definitions.items():
        for fine_name in fine_list:
            if fine_name in category_mapping_fine_to_broad:
                print(f"Warning: Fine label '{fine_name}' mapped to '{category_mapping_fine_to_broad[fine_name]}' is being re-mapped to '{broad_cat}'. Ensure this is intended (e.g. resolving initial misclassification).")
            category_mapping_fine_to_broad[fine_name] = broad_cat # Allow re-mapping to fix initial errors
            all_fine_labels_in_mapping.add(fine_name)

    # Check for missing fine-grained labels from TFDS that are not in your mapping
    missing_fine_labels_from_tfds = []
    tfds_fine_label_set = set(fine_label_names)

    for fine_label_name_from_tfds in tfds_fine_label_set:
        if fine_label_name_from_tfds not in all_fine_labels_in_mapping:
            missing_fine_labels_from_tfds.append(fine_label_name_from_tfds)

    if missing_fine_labels_from_tfds:
        print(f"\nERROR: {len(missing_fine_labels_from_tfds)} fine-grained labels from TFDS are STILL NOT in your mapping dictionary:")
        for mfl in sorted(missing_fine_labels_from_tfds):
            print(f"  - {mfl}")
        print("Please complete the 'broad_category_definitions' dictionary by adding these missing labels.")
        exit() # Exit if still missing
    else:
        print("\nSuccess: All fine-grained labels from TFDS appear to be covered in your mapping definitions.")

    # Check for labels in your mapping that are not in TFDS (typos, etc.)
    extra_fine_labels_in_mapping = all_fine_labels_in_mapping - tfds_fine_label_set
    if extra_fine_labels_in_mapping:
        print(f"\nWARNING: {len(extra_fine_labels_in_mapping)} fine-grained labels are in your mapping dictionary but NOT in TFDS fine_grained_labels_names (potential typos or outdated labels):")
        for efl in sorted(list(extra_fine_labels_in_mapping)):
            print(f"  - {efl}")
        print("You might want to remove or correct these in 'broad_category_definitions'.")

    final_broad_categories = sorted(list(set(category_mapping_fine_to_broad.values())))
    print(f"\nUnique Broad Categories Defined ({len(final_broad_categories)}): {final_broad_categories}")
    if set(final_broad_categories) != set(broad_categories_list):
        print("WARNING: The unique broad categories derived from your mapping do not exactly match your intended 'new_broad_categories_list'.")
        print(f"  Intended: {sorted(broad_categories_list)}")
        print(f"  Derived:  {final_broad_categories}")

    # --- 3. Load Processed Image Indices and Original Fine-Grained Labels ---
    print("\nLoading indices and original labels from HOG data file...")
    hog_data_file = os.path.join(OUTPUT_FEATURES_DIR, 'hog_data.h5')
    try:
        with h5py.File(hog_data_file, 'r') as hf:
            processed_image_indices = hf['indices'][:]
            original_fine_grained_numeric_labels = hf['labels'][:]
        print(f"Loaded {len(processed_image_indices)} indices and labels.")
        if len(processed_image_indices) == 0 :
            print("Error: No indices found in HOG data. Did feature extraction complete correctly?")
            exit()
    except Exception as e:
        print(f"Error loading HOG data file {hog_data_file}: {e}")
        exit()

    # --- 4. Map Original Fine-Grained Labels to Broad Category Labels ---
    print("\nMapping fine-grained labels to broad categories...")
    mapped_broad_labels_str_list = []
    valid_indices_for_split_list = []

    for i in range(len(original_fine_grained_numeric_labels)):
        fine_numeric_label = original_fine_grained_numeric_labels[i]
        current_image_original_tfds_idx = processed_image_indices[i]

        if 0 <= fine_numeric_label < len(fine_label_names):
            fine_label_name = fine_label_names[fine_numeric_label]
            if fine_label_name in category_mapping_fine_to_broad:
                broad_label_name = category_mapping_fine_to_broad[fine_label_name]
                mapped_broad_labels_str_list.append(broad_label_name)
                valid_indices_for_split_list.append(current_image_original_tfds_idx)
            else:
                print(f"Critical Error: Fine label name '{fine_label_name}' (from TFDS) was not found in the final mapping dictionary, even after checks. This should not happen. Halting.")
                exit()
        else:
            print(f"Warning: Invalid fine_numeric_label {fine_numeric_label} for original TFDS index {current_image_original_tfds_idx}. Skipping this image.")

    if not mapped_broad_labels_str_list:
        print("Error: No labels were successfully mapped to broad categories. Check your mapping dictionary and data.")
        exit()
    print(f"Successfully mapped {len(mapped_broad_labels_str_list)} images to broad categories.")

    # --- 5. Encode Broad Category String Labels to Numeric Labels ---
    print("\nEncoding broad category string labels to numeric labels...")
    label_encoder = LabelEncoder()
    label_encoder.fit(broad_categories_list)
    numeric_broad_labels = label_encoder.transform(mapped_broad_labels_str_list)
    print("Broad Category String to Numeric Mapping (based on LabelEncoder):")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  {class_name}: {i}")

    # --- 6. Create Train/Test Split ---
    print("\nCreating train/test split...")
    train_indices, test_indices, \
    train_broad_labels_numeric, test_broad_labels_numeric, \
    train_broad_labels_str, test_broad_labels_str = train_test_split(
        valid_indices_for_split_list,
        numeric_broad_labels,
        mapped_broad_labels_str_list,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=numeric_broad_labels
    )
    print(f"Training set size: {len(train_indices)} images")
    print(f"Test set size: {len(test_indices)} images")
    print("\nTrain broad label distribution (numeric):", np.bincount(train_broad_labels_numeric, minlength=len(label_encoder.classes_)))
    print("Test broad label distribution (numeric):", np.bincount(test_broad_labels_numeric, minlength=len(label_encoder.classes_)))
    for i, class_name in enumerate(label_encoder.classes_):
        train_count = np.sum(train_broad_labels_numeric == i)
        test_count = np.sum(test_broad_labels_numeric == i)
        print(f"  Category '{class_name}' (ID {i}): Train={train_count}, Test={test_count}")

    # --- 7. Save the Splits and Label Encoder ---
    output_splits_dir = os.path.join(OUTPUT_FEATURES_DIR, "train_test_splits_4cat_revised")
    os.makedirs(output_splits_dir, exist_ok=True)
    split_data_file = os.path.join(output_splits_dir, "train_test_split_data_4cat_revised.npz")
    label_encoder_file = os.path.join(output_splits_dir, "broad_label_encoder_4cat_revised.pkl")
    np.savez(
        split_data_file,
        train_indices=np.array(train_indices), test_indices=np.array(test_indices),
        train_labels_numeric=train_broad_labels_numeric, test_labels_numeric=test_broad_labels_numeric,
        train_labels_str=np.array(train_broad_labels_str), test_labels_str=np.array(test_broad_labels_str)
    )
    print(f"\nSaved train/test indices and labels to: {split_data_file}")
    with open(label_encoder_file, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Saved label encoder to: {label_encoder_file}")
    print("\n--- Data Preparation (Labels & Splits for 4 Categories) Complete ---")