import sys
from download_places import download_dataset
from SOH_extract import SOH_extract
from Vocabulary import build_vocab_KMeans
from Label_Split import split_data
from Histogram_creation import histogram_creation

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
    download_dataset()

    # Step 2: Extract features
    SOH_extract()
    
    # Step 3: Build Vocabulary with KMeans
    build_vocab_KMeans()
    
    # Step 4: Split data into the 4 categories
    split_data()
    
    # Step 5: Build Histograms for each image
    histogram_creation()
    
    # Step 6: Train and evaluate the model. If its better than 0.25 accuracy great success

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    # Ensures the script runs only when executed directly
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)