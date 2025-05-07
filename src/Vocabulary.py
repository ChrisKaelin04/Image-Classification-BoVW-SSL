import os
import pickle
import numpy as np
import glob
import random
from tqdm import tqdm
# *** Import MiniBatchKMeans ***
from sklearn.cluster import MiniBatchKMeans
import joblib # Keep for potential model saving later

# --- Configuration ---
FEATURES_DIR = "E:\CV_features"
FEATURE_TYPE = 'orb'
BATCHES_SUBDIR = 'orb_batches'
OUTPUT_VOCAB_FILE = os.path.join(FEATURES_DIR, f'{FEATURE_TYPE}_vocabulary_k1000_partial_fit.pkl')
OUTPUT_KMEANS_MODEL_FILE = os.path.join(FEATURES_DIR, f'{FEATURE_TYPE}_kmeans_model_k1000_partial_fit.joblib') # Optional

# --- K-Means Parameters ---
VOCABULARY_SIZE = 1000 # (k) Number of visual words
# MiniBatchKMeans specific parameter used by partial_fit internally if data is large,
# but we control the batch size by how much we feed to partial_fit.
# Set a reasonable batch_size for internal calculations if needed.
MINIBATCH_SIZE = 1024 * 4 # Example: 4096 (Less critical now, as we feed external batches)
RANDOM_SEED = 42

# --- Script ---
print(f"--- Starting K-Means Vocabulary Creation for {FEATURE_TYPE} using MiniBatchKMeans.partial_fit ---")
print(f"Loading descriptors iteratively from: {os.path.join(FEATURES_DIR, BATCHES_SUBDIR)}")
print(f"Target vocabulary size (k): {VOCABULARY_SIZE}")

# Set seed for reproducibility if MiniBatchKMeans uses it for internal shuffling/sampling
np.random.seed(RANDOM_SEED)
# random.seed(RANDOM_SEED) # Less critical now we aren't doing the big sampling step

batch_files = sorted(glob.glob(os.path.join(FEATURES_DIR, BATCHES_SUBDIR, f'{FEATURE_TYPE}_batch_*.pkl')))

if not batch_files:
    print(f"Error: No batch files found in {os.path.join(FEATURES_DIR, BATCHES_SUBDIR)}")
    exit()

print(f"Found {len(batch_files)} batch files to process.")

# --- Initialize MiniBatchKMeans ---
# Initialize the model *before* the loop.
# Use n_init=1 because partial_fit updates a single model incrementally.
# It doesn't run multiple initializations like fit() does with n_init='auto' or > 1.
print(f"Initializing MiniBatchKMeans model with k={VOCABULARY_SIZE}...")
kmeans = MiniBatchKMeans(n_clusters=VOCABULARY_SIZE,
                         random_state=RANDOM_SEED,
                         batch_size=MINIBATCH_SIZE, # Used for internal processing if needed
                         n_init=1, # CRITICAL: Must be 1 for partial_fit
                         max_iter=100, # Max iterations *per partial_fit call* (usually low is fine)
                         # tol=1e-4, # Tolerance check might be less useful with partial_fit
                         verbose=1, # Print progress updates from MiniBatchKMeans
                         compute_labels=False # Don't need labels during training, saves memory/time
                        )

# --- Iteratively Train with partial_fit ---
print(f"Starting iterative training using partial_fit over {len(batch_files)} batches...")
total_descriptors_processed = 0

# Loop through each saved batch file
# Wrap the loop with tqdm for progress bar over the files
for batch_file in tqdm(batch_files, desc="Processing batches"):
    try:
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)

        # Collect descriptors ONLY from the current batch
        current_batch_descriptors = []
        for idx, descriptors in batch_data.items():
            if descriptors is not None and descriptors.shape[0] > 0:
                current_batch_descriptors.append(descriptors)

        if not current_batch_descriptors:
            # print(f"No descriptors found in {batch_file}, skipping.") # Optional: reduce noise
            continue # Skip empty batches

        # Concatenate descriptors *from this batch only*
        # Memory usage is now limited to the size of one batch + model
        batch_np = np.vstack(current_batch_descriptors).astype(np.float32) # Ensure float32
        total_descriptors_processed += batch_np.shape[0]

        # Train the model on this batch
        kmeans.partial_fit(batch_np)

        # Optional: Clear memory explicitly (might help on some systems)
        del batch_data
        del current_batch_descriptors
        del batch_np

    except FileNotFoundError:
        print(f"\nWarning: Batch file not found: {batch_file}")
    except pickle.UnpicklingError:
        print(f"\nWarning: Could not unpickle file: {batch_file}")
    except MemoryError:
        print(f"\nError: Ran out of memory while processing batch: {batch_file}.")
        print("This single batch might be too large for available RAM.")
        # Consider re-running feature extraction with smaller BATCH_SAVE_SIZE if this occurs
        exit()
    except Exception as e:
        print(f"\nWarning: An error occurred processing {batch_file}: {e}")


print(f"\nK-Means partial_fit training complete. Processed {total_descriptors_processed} descriptors.")

# --- Get Vocabulary (Cluster Centers) ---
# The centers are now trained based on all the data fed via partial_fit
print("Extracting final cluster centers (vocabulary)...")
if hasattr(kmeans, 'cluster_centers_'):
    vocabulary = kmeans.cluster_centers_
    print(f"Vocabulary shape: {vocabulary.shape}") # Should be (VOCABULARY_SIZE, 128)

    # --- Save Vocabulary ---
    print(f"Saving vocabulary to: {OUTPUT_VOCAB_FILE}")
    try:
        with open(OUTPUT_VOCAB_FILE, 'wb') as f:
            pickle.dump(vocabulary, f)
    except Exception as e:
        print(f"Error saving vocabulary file: {e}")


    # Optional: Save the entire fitted KMeans model
    # Useful if you want to predict cluster indices later without retraining
    print(f"Saving KMeans model object to: {OUTPUT_KMEANS_MODEL_FILE}")
    try:
        joblib.dump(kmeans, OUTPUT_KMEANS_MODEL_FILE)
        print(f"Successfully saved KMeans model to {OUTPUT_KMEANS_MODEL_FILE}")
    except Exception as e:
        print(f"Error saving KMeans model object: {e}")


    print("--- Vocabulary creation finished successfully! ---")
else:
    print("Error: KMeans model does not have cluster_centers_ attribute. Training might have failed.")