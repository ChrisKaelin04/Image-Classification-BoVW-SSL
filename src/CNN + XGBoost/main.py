import sys
from extract_cnn_features import extract_cnn_features
from Label_Split import split_data
from CNN_XGBoost_SVM import run_cnn_xgb_classification

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

    # Step 1: Extract features using CNN (we already have the data downloaded)
    extract_cnn_features()
    
    # Step 2: Split the data into the 4 categories
    split_data()
    
    # Step 3: Train the model using XGBoost, then test it
    run_cnn_xgb_classification()
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    # Ensures the script runs only when executed directly
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)