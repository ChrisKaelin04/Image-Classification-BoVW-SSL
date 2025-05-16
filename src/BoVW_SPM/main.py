import sys
from download_places import download_dataset
from SOH_extract_SPM import main_spm_feature_extraction as SOH_extract_SPM
from Vocabulary_SPM import build_all_spm_vocabularies_balanced
from Label_Split import create_balanced_split_for_bovw
from Histogram_Creation import main_histogram_creation_spm_balanced
from svm_classified_SPM import run_balanced_spm_classification_pipeline as run_spm_classification_pipeline
'''
Note: This script is written with GPU usage in mind! If you are using a CPU switch to a different usage for XGBoost or use SVMs.
Side Note: Computer Vision is a rat bastard of a field what the hell is going on
'''
def main():
    """
    Main function to orchestrate the workflow.
    Uncomment the steps you want to run.
    """
    print("Starting the Image Classification Pipeline...")

    # Step 1: Get the data, process it
    #download_dataset()
    #create_balanced_split_for_bovw()
    # Step 2: Extract features
    #SOH_extract_SPM()
    
    # Step 3: Build Vocabulary with KMeans
    #build_all_spm_vocabularies_balanced()
    
    # Step 4: Build Histograms for each image
    #main_histogram_creation_spm_balanced()
    
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