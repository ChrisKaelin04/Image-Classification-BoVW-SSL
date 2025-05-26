import sys
from download_places import download_dataset
from SOH_extract_SPM import main_spm_feature_extraction as SOH_extract_SPM
from Vocabulary_SPM import build_all_spm_vocabularies_balanced
from Label_Split import create_balanced_split_for_bovw
from Histogram_Creation import main_histogram_creation_spm_balanced
from svm_classified_SPM import run_balanced_spm_classification_pipeline as run_spm_classification_pipeline
import os
'''
Note: This script is written with GPU usage in mind! If you are using a CPU switch to a different usage for XGBoost or use SVMs.
Side Note: Computer Vision is a rat bastard of a field what the hell is going on
'''
def main():
    """
    Main function to orchestrate the workflow.
    Alter file paths and parameters as needed.
    """
    print("Starting the Image Classification Pipeline...")

    # Step 1: Get the data, process it
    TFDS_DATA_DIR = "E:\CV\Images_Downloaded"
    RAW_IMAGE_DATA_DIR = "E:\CV\Raw_Images"
    OUTPUT_SPLITS_INFO_DIR = "E:\CV\Splits_Info"
    download_dataset(TFDS_DATA_DIR)
    create_balanced_split_for_bovw(TFDS_DATA_DIR, RAW_IMAGE_DATA_DIR, OUTPUT_SPLITS_INFO_DIR)
    # Step 2: Extract features
    BALANCED_SPLIT_NPZ_FILE = r"E:\CV\bovw_SPM_splits_balanced\bovw_train_test_paths_N100000_S42.npz"
    OUTPUT_FEATURES_SPM_DIR = r"E:\CV\features_SPM_balanced"
    BATCH_SAVE_SIZE = 5000
    SIFT_BATCHES_SPM_DIR = os.path.join(OUTPUT_FEATURES_SPM_DIR, 'sift_batches_spm')
    ORB_BATCHES_SPM_DIR = os.path.join(OUTPUT_FEATURES_SPM_DIR, 'orb_batches_spm')
    HOG_BATCHES_SPM_DIR = os.path.join(OUTPUT_FEATURES_SPM_DIR, 'hog_batches_spm')
    SOH_extract_SPM(BALANCED_SPLIT_NPZ_FILE, BATCH_SAVE_SIZE, SIFT_BATCHES_SPM_DIR, ORB_BATCHES_SPM_DIR, HOG_BATCHES_SPM_DIR)
    
    # Step 3: Build Vocabulary with KMeans
    # --- Configuration for SPM Vocabulary (from BALANCED features) ---
    build_all_spm_vocabularies_balanced(OUTPUT_FEATURES_SPM_DIR)
    
    # Step 4: Build Histograms for each image
    main_histogram_creation_spm_balanced(OUTPUT_FEATURES_SPM_DIR)
    
    # Step 5: Train and evaluate the model. If its better than 0.25 accuracy great success
    run_spm_classification_pipeline()

    print("Pipeline completed successfully!")
    
if __name__ == "__main__":
    # Ensures the script runs only when executed directly
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)