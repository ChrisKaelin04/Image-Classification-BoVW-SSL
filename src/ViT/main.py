import sys
from fine_tune import main_finetuning_pipeline
from Label_Split import run as create_all_splits_for_finetuning
from download_places import download_dataset
from eval_model import evaluate

def main():
    """
    Main function to orchestrate the workflow.
    """
    print("Starting the Image Classification Pipeline...")
    # Step 1: Download and prepare the dataset
    #download_dataset()
    #create_all_splits_for_finetuning()
    # Step 2: Extract features using CNN
    main_finetuning_pipeline()
    
    # Step 3: Evaluate the model
    evaluate()
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    # Ensures the script runs only when executed directly
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)