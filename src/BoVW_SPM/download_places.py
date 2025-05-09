import tensorflow_datasets as tfds
import os
import time
TFDS_DATA_DIR = "E:\CV_imgs"
#Setup for downloading and preparing the Places365 dataset using TensorFlow Datasets (TFDS). This was not written by me, used online references.
def setup_download():
    print(f"--- Starting Places365 Download and Preparation ---")
    print(f"Target directory for TFDS data: {TFDS_DATA_DIR}")
    print("WARNING: This process will download ~60GB+ and requires significant additional space for preparation.")
    print("Ensure the selected drive has sufficient free space (100-150GB free recommended).")

    # Create the target directory if it doesn't exist
    try:
        os.makedirs(TFDS_DATA_DIR, exist_ok=True)
        print(f"Directory '{TFDS_DATA_DIR}' ensured.")
    except OSError as e:
        print(f"Error creating directory {TFDS_DATA_DIR}: {e}")
        print("Please ensure you have permissions and the path is valid before proceeding.")
        exit() # Exit if directory creation fails

def begin_download():
    # Record start time
    start_time = time.time()

    try:
        print("\nCalling tfds.load() for 'places365_small' [train, validation] splits...")
        print("This will now download the raw data and then prepare it.")
        print("This process can take a VERY long time (potentially several hours) depending on your internet speed and CPU.")
        print("Please leave this script running until it prints 'Download and Preparation Complete!' or an error message.")

        # This command triggers the download and preparation process.
        # We load both 'train' and 'validation' so they are both ready for tomorrow.
        # We capture the ds_info to print some stats at the end.
        _, ds_info = tfds.load(
            'places365_small',          # Use the 256x256 version
            split=['train', 'validation'], # Specify the splits needed
            data_dir=TFDS_DATA_DIR,     # Tell TFDS where to store everything
            download=True,              # Explicitly enable download
            with_info=True              # Get dataset metadata
            # We are NOT iterating through the dataset here, just loading/preparing.
        )

        # If the load command finishes without error, the data is ready on disk.
        print("\n--- Download and Preparation Complete! ---")
        print(f"Places365 'train' and 'validation' splits are now prepared and stored in:")
        print(f"{TFDS_DATA_DIR}")
        print(f"\nDataset Info:")
        print(f"Number of training examples: {ds_info.splits['train'].num_examples}")
        print(f"Number of validation examples: {ds_info.splits['validation'].num_examples}")
        print(f"Number of classes: {ds_info.features['label'].num_classes}")

    except Exception as e:
        # Catch potential errors during download/prep (e.g., disk space, network)
        print(f"\n--- An Error Occurred During Download/Preparation ---")
        print(e)
        print("\nPlease check:")
        print(f"  - Internet connection")
        print(f"  - Available disk space on the drive containing '{TFDS_DATA_DIR}'")
        print(f"  - Path validity and permissions for '{TFDS_DATA_DIR}'")

    # Record end time and duration
    end_time = time.time()
    duration_seconds = end_time - start_time
    print(f"\nTotal time elapsed: {duration_seconds / 60:.2f} minutes ({duration_seconds / 3600:.2f} hours)")

    print("\nYou can now close this script.")
    print("Tomorrow, in your main code, use tfds.load('places365_small', data_dir='{}', ...) ".format(TFDS_DATA_DIR))
    print("and it will load the prepared data instantly from disk.")

def download_dataset():
    setup_download()
    begin_download()
