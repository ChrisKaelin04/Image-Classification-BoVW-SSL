import sys
from download_places import download_dataset
from SOH_extract import main_vanilla_bovw_feature_extraction as SOH_extract
from Vocabulary import build_all_vanilla_bovw_vocabularies
from Label_Split import create_balanced_split_for_bovw
from Histogram_creation import main_histogram_creation_vanilla_balanced
from svm_classified import run_balanced_vanilla_bovw_classification_pipeline as test

'''
Note: This script is written with GPU usage in mind! If you are using a CPU switch to a different usage for XGBoost or use SVMs.
Side Note: Computer Vision is a rat bastard of a field what the hell is going on
'''
def main():
    """
    Main function to orchestrate the workflow.
    """
    print("Starting the Image Classification Pipeline...")

    # Step 1: Get the data
    #download_dataset()
    #create_balanced_split_for_bovw()

    # Step 2: Extract features
    #SOH_extract()
    
    # Step 3: Build Vocabulary with KMeans
    #build_all_vanilla_bovw_vocabularies()
    
    # Step 5: Build Histograms for each image
    #main_histogram_creation_vanilla_balanced()
    
    # Step 6: Train and evaluate the model. If its better than 0.25 accuracy great success
    test()

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    # Ensures the script runs only when executed directly
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)