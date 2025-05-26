# We need to get 4 broad categories of images from the dataset. We will use the following categories:
# 1. Indoor residential
# 2. Indoor Public/Commercial
# 3. Outdoor Natural
# 4. Outdoor Urban
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from PIL import Image # For saving images


RANDOM_SEED = 42
TARGET_IMAGES_PER_BROAD_CATEGORY = 25000
MAX_TFDS_IMAGES_TO_SCAN = 350000

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
                           'corn_field',
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
                           'hayfield',
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
                           'wheat_field',
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
                         'wind_farm',
                         'windmill',
                         'yard',
                         'zen_garden']}

def create_balanced_split_for_bovw(TFDS_DATA_DIR, BOVW_RAW_IMAGE_DATA_DIR, OUTPUT_SPLITS_INFO_DIR_BOVW):
    print(f"--- Starting Balanced Data Preparation for BoVW ({len(broad_categories_list)} Broad Categories) ---")
    print(f"Targeting {TARGET_IMAGES_PER_BROAD_CATEGORY} images per broad category.")
    print(f"Raw images for BoVW will be saved to: {BOVW_RAW_IMAGE_DATA_DIR}")
    print(f"Split information (NPZ/PKL) will be saved to: {OUTPUT_SPLITS_INFO_DIR_BOVW}")

    # Create output directories
    os.makedirs(BOVW_RAW_IMAGE_DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_SPLITS_INFO_DIR_BOVW, exist_ok=True)
    for cat_name in broad_categories_list:
        os.makedirs(os.path.join(BOVW_RAW_IMAGE_DATA_DIR, cat_name), exist_ok=True)

    # --- 1. Load TFDS Dataset Info ---
    print("\nLoading dataset info for 'places365_small'...")
    try:
        # Load the 'train' split. We'll iterate through it without shuffling at this stage.
        full_ds_train_tfds, ds_info = tfds.load(
            'places365_small',
            split='train',
            data_dir=TFDS_DATA_DIR,
            with_info=True,
            download=False, # Assumes data is already downloaded by your download_dataset()
            shuffle_files=False # CRITICAL for consistent iteration if script is rerun partway
        )
    except Exception as e:
        print(f"Error loading TFDS dataset info or base dataset: {e}")
        print("Please ensure Places365 data is downloaded and TFDS_DATA_DIR is correct.")
        exit()

    fine_label_names = ds_info.features['label'].names
    num_fine_classes = len(fine_label_names)
    total_tfds_train_images = ds_info.splits['train'].num_examples
    print(f"Total fine-grained classes in Places365: {num_fine_classes}")
    print(f"Total images in TFDS 'train' split: {total_tfds_train_images}")

    # --- 2. Setup Broad Category Mapping ---
    print("\nSetting up and validating broad category mapping...")
    fine_to_broad_mapping = {}
    # Inverse mapping for easier lookup: broad_category -> list of fine_grained_indices
    broad_to_fine_indices_mapping = {broad_cat: [] for broad_cat in broad_categories_list}

    # Populate fine_to_broad_mapping
    unmapped_tfds_labels = set(fine_label_names)
    for broad_cat, fine_list_for_cat in broad_category_definitions.items():
        if broad_cat not in broad_categories_list:
            print(f"Warning: Broad category '{broad_cat}' in definitions is not in broad_categories_list. Skipping.")
            continue
        for fine_name in fine_list_for_cat:
            if fine_name in fine_label_names:
                fine_to_broad_mapping[fine_name] = broad_cat
                try: # Keep track of unmapped TFDS labels
                    unmapped_tfds_labels.remove(fine_name)
                except KeyError:
                    print(f"Warning: Fine label '{fine_name}' in your definition for '{broad_cat}' was already mapped or not in TFDS originally.")
            else:
                print(f"Warning: Fine label '{fine_name}' defined for '{broad_cat}' is not a valid Places365 fine_label_name. Skipping.")
    
    if unmapped_tfds_labels:
        print(f"\nERROR: {len(unmapped_tfds_labels)} fine-grained labels from TFDS are NOT in your 'broad_category_definitions':")
        for unmapped_label in sorted(list(unmapped_tfds_labels))[:20]: # Print a sample
            print(f"  - {unmapped_label}")
        if len(unmapped_tfds_labels) > 20:
            print(f"  ... and {len(unmapped_tfds_labels) - 20} more.")
        print("Please complete your 'broad_category_definitions' to include all 365 fine-grained Places365 labels.")
        # exit() # You might want to exit here in a real run

    print(f"Successfully processed {len(fine_to_broad_mapping)} fine-to-broad mappings.")

    # --- 3. Select and Save/Copy Images for Balanced Broad Categories ---
    print(f"\nScanning TFDS 'train' split to select and save images for BoVW...")
    # This will store {'image_path': path_to_saved_img, 'broad_category': category_name}
    # Grouped by broad category initially for easy counting
    collected_items_by_category = {broad_cat: [] for broad_cat in broad_categories_list}
    images_saved_count = {broad_cat: 0 for broad_cat in broad_categories_list}
    total_images_to_select = len(broad_categories_list) * TARGET_IMAGES_PER_BROAD_CATEGORY
    
    # Enumerate the dataset to get a consistent index for each item if needed for filenames
    # The original_tfds_idx can help make filenames unique if fine_label_name is not enough
    ds_enumerated = full_ds_train_tfds.enumerate()

    scanned_tfds_count = 0
    total_images_saved_so_far = 0

    with tqdm(total=min(total_tfds_train_images, MAX_TFDS_IMAGES_TO_SCAN), desc="Scanning & Saving for BoVW") as pbar:
        for original_tfds_idx_tensor, item_tfds in ds_enumerated:
            scanned_tfds_count += 1
            pbar.update(1)

            if scanned_tfds_count > MAX_TFDS_IMAGES_TO_SCAN:
                print(f"\nReached MAX_TFDS_IMAGES_TO_SCAN ({MAX_TFDS_IMAGES_TO_SCAN}). Stopping scan.")
                break
            if total_images_saved_so_far >= total_images_to_select:
                # This condition means all categories have met their target
                all_filled = all(images_saved_count[cat] >= TARGET_IMAGES_PER_BROAD_CATEGORY for cat in broad_categories_list)
                if all_filled:
                    print(f"\nAll broad categories have reached their target of {TARGET_IMAGES_PER_BROAD_CATEGORY} images.")
                    break

            fine_numeric_label = item_tfds['label'].numpy()
            original_tfds_idx = original_tfds_idx_tensor.numpy() # For unique filenames

            if 0 <= fine_numeric_label < num_fine_classes:
                fine_label_name = fine_label_names[fine_numeric_label]
                if fine_label_name in fine_to_broad_mapping:
                    broad_cat_name = fine_to_broad_mapping[fine_label_name]
                    
                    # Check if this broad category still needs more images
                    if images_saved_count.get(broad_cat_name, 0) < TARGET_IMAGES_PER_BROAD_CATEGORY:
                        image_numpy = item_tfds['image'].numpy()
                        pil_image = Image.fromarray(image_numpy).convert('RGB')
                        
                        # Create a unique filename
                        # Sanitize fine_label_name for use in filename
                        sanitized_fine_label = fine_label_name.replace('/', '_').replace(' ', '_')
                        image_filename = f"idx{original_tfds_idx}_{sanitized_fine_label}.jpg"
                        
                        category_image_dir = os.path.join(BOVW_RAW_IMAGE_DATA_DIR, broad_cat_name)
                        image_save_path = os.path.join(category_image_dir, image_filename)

                        try:
                            pil_image.save(image_save_path)
                            collected_items_by_category[broad_cat_name].append({
                                'image_path': image_save_path,
                                'broad_category_name': broad_cat_name # Store name for stratification
                            })
                            images_saved_count[broad_cat_name] += 1
                            total_images_saved_so_far += 1
                            pbar.set_postfix({cat: count for cat, count in images_saved_count.items()})
                        except Exception as e:
                            pbar.write(f"Error saving image {image_filename} (idx {original_tfds_idx}): {e}")
                            continue # Skip this image if saving fails
                # else: fine_label_name was not mapped (e.g., if definition is incomplete)
            # else: invalid fine_numeric_label (should be rare)

    print("\n--- Image Selection and Saving Summary ---")
    all_categories_filled = True
    for broad_cat in broad_categories_list:
        count = images_saved_count[broad_cat]
        print(f"Category '{broad_cat}': Saved {count} images (Target: {TARGET_IMAGES_PER_BROAD_CATEGORY})")
        if count < TARGET_IMAGES_PER_BROAD_CATEGORY:
            all_categories_filled = False
            print(f"  WARNING: Category '{broad_cat}' did not reach the target. Consider increasing MAX_TFDS_IMAGES_TO_SCAN or check category definition abundance in Places365.")

    if not all_categories_filled:
        print("WARNING: Not all categories reached their target counts.")
    print(f"Total images saved across all categories: {total_images_saved_so_far}")

    if total_images_saved_so_far == 0:
        print("ERROR: No images were saved. Check mappings, paths, and TFDS scan logic.")
        exit()

    # --- 4. Prepare for Train/Test Split using Image Paths ---
    # Consolidate all collected items (dictionaries of 'image_path' and 'broad_category_name')
    all_selected_items_for_split = []
    for broad_cat_name in broad_categories_list:
        all_selected_items_for_split.extend(collected_items_by_category[broad_cat_name])

    # Shuffle this consolidated list before splitting to ensure randomness in train/test
    # (and that images from different categories are mixed)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(all_selected_items_for_split)

    # Extract the image paths and string labels for splitting
    split_image_paths = [item['image_path'] for item in all_selected_items_for_split]
    split_labels_str = [item['broad_category_name'] for item in all_selected_items_for_split]

    # --- 5. Encode Broad Category String Labels to Numeric Labels ---
    print("\nEncoding broad category string labels to numeric labels...")
    label_encoder = LabelEncoder()
    # Fit the encoder on the predefined 'broad_categories_list' to ensure consistent ordering
    label_encoder.fit(broad_categories_list)
    numeric_broad_labels_for_split = label_encoder.transform(split_labels_str)

    print("Broad Category String to Numeric Mapping (based on LabelEncoder):")
    for i, class_name_le in enumerate(label_encoder.classes_):
        print(f"  '{class_name_le}': {i}")

    # --- 6. Create Train/Test Split ---
    print("\nCreating train/test split (stratified by numeric broad labels)...")
    # We are splitting the `split_image_paths`
    train_image_paths, test_image_paths, \
    train_broad_labels_numeric, test_broad_labels_numeric, \
    train_broad_labels_str, test_broad_labels_str = train_test_split(
        split_image_paths,                # These are the paths to images saved in BOVW_RAW_IMAGE_DATA_DIR
        numeric_broad_labels_for_split,   # Numeric labels for stratification
        split_labels_str,                 # String labels (for reference in NPZ)
        test_size=0.2,                    # Standard 80/20 split
        random_state=RANDOM_SEED,         # Use the same seed for reproducible splits
        stratify=numeric_broad_labels_for_split  # Ensure class balance based on numeric labels
    )
    print(f"Training set size: {len(train_image_paths)} image paths")
    print(f"Test set size: {len(test_image_paths)} image paths")

    print("\nTrain broad label distribution (numeric):")
    train_counts = np.bincount(train_broad_labels_numeric, minlength=len(label_encoder.classes_))
    for i, class_name_le in enumerate(label_encoder.classes_): print(f"  '{class_name_le}' (ID {i}): {train_counts[i]}")
    
    print("Test broad label distribution (numeric):")
    test_counts = np.bincount(test_broad_labels_numeric, minlength=len(label_encoder.classes_))
    for i, class_name_le in enumerate(label_encoder.classes_): print(f"  '{class_name_le}' (ID {i}): {test_counts[i]}")


    # --- 7. Save the Splits (Image Paths and Labels) and Label Encoder ---
    split_data_file_bovw = os.path.join(OUTPUT_SPLITS_INFO_DIR_BOVW, f"bovw_train_test_paths_N{total_images_saved_so_far}_S{RANDOM_SEED}.npz")
    label_encoder_file_bovw = os.path.join(OUTPUT_SPLITS_INFO_DIR_BOVW, f"bovw_label_encoder_N{total_images_saved_so_far}_S{RANDOM_SEED}.pkl")

    np.savez_compressed( # Use compressed to save space if many paths
        split_data_file_bovw,
        train_image_paths=np.array(train_image_paths, dtype=object), # Save image paths
        test_image_paths=np.array(test_image_paths, dtype=object),   # Save image paths
        train_labels_numeric=train_broad_labels_numeric.astype(np.int8),
        test_labels_numeric=test_broad_labels_numeric.astype(np.int8),
        train_labels_str=np.array(train_broad_labels_str, dtype=object), # For inspection
        test_labels_str=np.array(test_broad_labels_str, dtype=object)    # For inspection
    )
    print(f"\nSaved BoVW train/test image paths and labels to: {split_data_file_bovw}")

    with open(label_encoder_file_bovw, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Saved BoVW label encoder to: {label_encoder_file_bovw}")

    print("\n--- Balanced Data Preparation for BoVW Complete ---")
    print(f"Selected raw images for BoVW are saved in subdirectories within: {BOVW_RAW_IMAGE_DATA_DIR}")
    print(f"The NPZ file '{split_data_file_bovw}' contains lists of these image paths for your BoVW train/test sets.")
    print("Your BoVW pipeline should now load these image paths to process the images.")
