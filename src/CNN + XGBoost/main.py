import sys
from extract_cnn_features import extract_alexnet_places365_features_pipeline
from Label_Split import create_balanced_split_and_labels
from CNN_XGBoost_SVM import run_cnn_xgb_classification_balanced_fc
from download_places import download_dataset

'''
Note 1: This script assumes you already have the data downloaded. Running the main for BoVW_Vanilla will set it up for you. Then come back here.
Note 2: This script is written with GPU usage in mind! If you are using a CPU switch to a different usage for XGBoost or use SVMs.
Side Note: Computer Vision is a rat bastard of a field what the hell is going on
'''
def main():
    """
    Main function to orchestrate the workflow.
    """
    print("Starting the Image Classification Pipeline...")
    # Step 1: Download and prepare the dataset
    #download_dataset()
    #create_balanced_split_and_labels()
    # Step 2: Extract features using CNN
    #extract_alexnet_places365_features_pipeline()
    
    # Step 3: Train the model using XGBoost, then test it
    run_cnn_xgb_classification_balanced_fc()
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    # Ensures the script runs only when executed directly
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)