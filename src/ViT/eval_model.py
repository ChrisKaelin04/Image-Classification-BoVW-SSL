# evaluate_vit_model.py

import torch
import torch.nn as nn
# Import both AlexNet and ViT models, though we'll only use ViT in the load function
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights # Needed for potentially retrieving transform details

import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import os
from tqdm import tqdm
import tensorflow_datasets as tfds
import tensorflow as tf
from PIL import Image
import gc
import pickle # To load the label encoder for class names
import random # Needed if using num_workers > 0 in DataLoader
import time

# --- Configuration ---
# Path to your saved fine-tuned ViT model
# --- CHANGE THIS PATH ---
SAVED_MODEL_PATH = r"E:\CV_Models_PyTorch_Balanced\finetuned_vit_b_16_imagenet_4cat_best_balanced.pth" # Path to your SAVED ViT model file

# Path to your NPZ file containing ALL splits (train/val/test_final)
NPZ_ALL_SPLITS_FILE = r"E:\CV_features\all_splits_data_4cat\all_splits_data_4cat.npz"
# Path to the saved label encoder (created by create_all_splits_for_finetuning.py)
LABEL_ENCODER_PATH = r"E:\CV_features\all_splits_data_4cat\broad_label_encoder_4cat.pkl"

TFDS_DATA_DIR = r"E:\CV_imgs"
# --- Ensure these match the input size the ViT was trained with (should be 224) ---
IMG_WIDTH, IMG_HEIGHT = 224, 224
IMG_RESIZE_SIZE = 256 # Size for evaluation transforms before center crop

BATCH_SIZE = 64 # Can potentially increase for evaluation as gradients are off
NUM_BROAD_CATEGORIES = 4 # Must match your model's output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure directory for confusion matrix plot exists (same as model save dir)
PLOT_SAVE_DIR = os.path.dirname(SAVED_MODEL_PATH)
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

# --- TF GPU Configuration (kept for robustness with tfds.load) ---
def configure_tf_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # print(f"Configured TensorFlow GPU memory growth for {len(gpus)} GPU(s).") # Commented out to reduce clutter
        except RuntimeError as e: print(f"Error setting TF GPU memory growth: {e}.")
    # else: print("No GPUs detected by TensorFlow. TF will use CPU.") # Commented out

configure_tf_gpu()

# --- TOP-LEVEL FUNCTION FOR DATALOADER WORKER INITIALIZATION ---
# This is generic and works for any model, keep as is.
def worker_init_fn(worker_id):
    """Initializes worker processes for the DataLoader."""
    worker_info = torch.utils.data.get_worker_info()
    dataset_obj = worker_info.dataset
    worker_seed = torch.initial_seed() % (2**32 - 1)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    # No explicit file opening/closing needed here for HDF5/TFDS cached data if dataset handles it.
    pass


# --- Reusable Dataset Class (same as in training/finetuning) ---
# This works because it operates on NumPy images and labels, independent of the model architecture.
class TFDSSubsetFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, original_tfds_indices_list, image_numpy_list, broad_numeric_labels_list, transform=None):
        self.original_tfds_indices = original_tfds_indices_list
        self.images_numpy = image_numpy_list
        self.labels = broad_numeric_labels_list
        self.transform = transform
        if not (len(self.original_tfds_indices) == len(self.images_numpy) == len(self.labels)):
            raise ValueError("Indices, images, and labels lists must have the same length.")
    def __len__(self): return len(self.images_numpy)
    def __getitem__(self, list_idx):
        # original_tfds_idx_val = self.original_tfds_indices[list_idx] # Not strictly needed for eval output
        img_np = self.images_numpy[list_idx]
        label_val = self.labels[list_idx]
        try: image = Image.fromarray(img_np).convert('RGB')
        except Exception as e:
            print(f"\nERROR: Convert NumPy to PIL for list_idx {list_idx}: {e}. Returning dummy.")
            # Return dummy tensor matching expected output shape (C, H, W)
            return torch.zeros(3, IMG_HEIGHT, IMG_WIDTH, dtype=torch.float32), torch.tensor(label_val, dtype=torch.long)
        if self.transform:
            try: image = self.transform(image)
            except Exception as e:
                print(f"\nWARN: Transform error for list_idx {list_idx}: {e}. Returning dummy.")
                 # Return dummy tensor matching expected output shape (C, H, W)
                return torch.zeros(3, IMG_HEIGHT, IMG_WIDTH, dtype=torch.float32), torch.tensor(label_val, dtype=torch.long)
        return image, torch.tensor(label_val, dtype=torch.long)


# --- Preprocessing (ADAPTED for ViT/Standard ImageNet eval) ---
# Use the 'val' / non-augmented version for evaluation. This should match the transform
# used for the validation set during ViT fine-tuning.
def get_vit_preprocessing_transform_eval():
    print(f"\nDefining preprocessing transform for ViT evaluation ({IMG_HEIGHT}x{IMG_WIDTH} input)...")
    # Standard ImageNet normalization values
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize(IMG_RESIZE_SIZE),     # Resize smallest edge to 256
        transforms.CenterCrop(IMG_HEIGHT),      # Take center 224x224 crop
        transforms.ToTensor(),
        normalize
    ])
    print("Defined evaluation transform (Resize, CenterCrop, ToTensor, Normalize).")
    return transform


# --- Load ViT Model Function (NEW/ADAPTED) ---
# --- Load ViT Model Function (NEW/ADAPTED - Correcting .head to .heads) ---
def load_finetuned_vit_model(model_path, num_classes):
    print(f"\n--- Loading fine-tuned ViT model from: {model_path} ---")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        exit()

    try:
        # 1. Load the ViT architecture (vit_b_16).
        model = models.vit_b_16(weights=None) # Load architecture only

        # 2. Modify the classifier head (.heads) to match the number of classes it was fine-tuned on.
        # We need to access the linear layer within the 'heads' Sequential
        if not hasattr(model, 'heads') or not isinstance(model.heads, nn.Sequential) or len(model.heads) == 0:
             print("ERROR: Model structure unexpected. Does not have a 'heads' Sequential attribute with modules.")
             # Consider adding dir(model) print here too if this error occurs during eval
             exit()

        # Access the linear layer inside the original 'heads' sequential (usually at index 0)
        original_linear_head = model.heads[0]
        if not isinstance(original_linear_head, nn.Linear):
             print("ERROR: The first module in model.heads is not an nn.Linear layer as expected.")
             print(f"model.heads: {model.heads}")
             exit()

        num_ftrs = original_linear_head.in_features # Get the dimension of features *before* the head

        # Replace the existing 'heads' sequential with a new Sequential containing your linear layer
        model.heads = nn.Sequential(nn.Linear(num_ftrs, num_classes)) # Create a new Sequential with your new linear layer
        print(f"Adjusted ViT classifier head (was Sequential -> {original_linear_head.__class__.__name__} -> {original_linear_head.out_features} classes) to a new Sequential containing Linear layer for {num_classes} outputs.")


        # 3. Load the fine-tuned state_dict from your saved file
        state_dict = torch.load(model_path, map_location=device)

        # Load the state dictionary into the model architecture
        # strict=True is important here to ensure the state_dict keys match the modified model structure
        # (i.e., it expects a key like 'heads.0.weight' and 'heads.0.bias' for the new linear layer)
        model.load_state_dict(state_dict, strict=True)
        print(f"Successfully loaded fine-tuned ViT state_dict from {model_path}.")

        # 4. Move the model to the specified device (redundant but safe)
        model.to(device)

        # 5. Set the model to evaluation mode!
        model.eval()
        print("Model set to evaluation mode.")

        return model

    except Exception as e:
        print(f"ERROR loading or setting up ViT model for evaluation: {e}")
        # Consider adding state_dict.keys() and model.state_dict().keys() print here if strict=True fails
        exit()

    try:
        # 1. Load the ViT architecture (vit_b_16). We just need the structure, not pre-trained weights initially.
        model = models.vit_b_16(weights=None) # Use weights=None for current torchvision versions
        # Fallback for older versions if needed:
        # try: model = models.vit_b_16(pretrained=False)
        # except Exception: print("Error loading ViT architecture.") ; exit()

        # 2. Modify the classifier head to match the number of classes it was fine-tuned on (your 4 categories).
        # The ViT head is accessed via `model.head`
        num_ftrs = model.head.in_features # Get the dimension of features *before* the head
        model.head = nn.Linear(num_ftrs, num_classes) # Replace the head with a new linear layer for num_classes outputs
        print(f"Adjusted ViT classifier head to {num_classes} outputs.")

        # 3. Load the fine-tuned state_dict from your saved file
        # map_location=device ensures the weights are loaded onto the correct device
        state_dict = torch.load(model_path, map_location=device)

        # Load the state dictionary into the model architecture
        # strict=True means all keys in the state_dict must match keys in the model's state_dict
        # This is a good check that you're loading weights into the correct model architecture.
        model.load_state_dict(state_dict, strict=True)
        print(f"Successfully loaded fine-tuned ViT state_dict from {model_path}.")

        # 4. Move the model to the specified device (redundant if map_location is used, but harmless)
        model.to(device)

        # 5. Set the model to evaluation mode! This is crucial for inference.
        # It disables dropout, batch normalization tracking, etc.
        model.eval()
        print("Model set to evaluation mode.")

        return model

    except Exception as e:
        print(f"ERROR loading or setting up ViT model for evaluation: {e}")
        # Consider printing the keys in the loaded state_dict vs model state_dict if strict=True fails
        # print("Keys in loaded state_dict:", state_dict.keys())
        # print("Keys in model state_dict:", model.state_dict().keys())
        exit()


# --- Data Loading for Final Evaluation (Same as before) ---
# This function is generic as it just loads data into NumPy arrays and lists.
def load_final_test_data(npz_all_splits_path, tfds_name='places365_small', tfds_split_name='train'):
    print(f"\n--- Loading FINAL TEST Data and Caching Images ---")
    if not os.path.exists(npz_all_splits_path):
        print(f"ERROR: NPZ file for all splits not found at {npz_all_splits_path}.")
        exit()

    print(f"Loading FINAL TEST split data from: {npz_all_splits_path}")
    try:
        all_splits_data_npz = np.load(npz_all_splits_path)
    except Exception as e:
        print(f"ERROR loading NPZ file {npz_all_splits_path}: {e}. Exiting.")
        exit()


    # Load the FINAL TEST data using the correct keys
    # Ensure keys match what was saved in create_all_splits_for_finetuning.py
    try:
        original_tfds_test_indices_final = all_splits_data_npz['test_indices_final'].tolist()
        y_test_numeric_broad_final = all_splits_data_npz['test_labels_numeric_final'].tolist()
    except KeyError as e:
        print(f"ERROR: Missing expected key in NPZ file {npz_all_splits_path}: {e}. Ensure it contains 'test_indices_final' and 'test_labels_numeric_final'. Available keys: {list(all_splits_data_npz.keys())}. Exiting.")
        exit()

    print(f"Loaded {len(original_tfds_test_indices_final)} final test original TFDS indices/labels from NPZ.")

    if not original_tfds_test_indices_final:
        print("ERROR: No test indices found in the NPZ file. Cannot proceed.")
        # Ensure the NPZ file wasn't saved empty for the test split.
        # Check the script that created it.
        return [], [], [] # Return empty lists to indicate failure/empty data


    all_required_original_tfds_indices_set = set(original_tfds_test_indices_final)

    print(f"Loading TFDS dataset: {tfds_name}, split: {tfds_split_name} for image data")
    try:
        # Use the standard TFDS load approach. Configure TF GPU memory growth beforehand.
        ds_info_obj = tfds.builder(tfds_name, data_dir=TFDS_DATA_DIR).info
        num_total_tfds_images_in_split = ds_info_obj.splits[tfds_split_name].num_examples

        full_ds_tfds_enumerated = tfds.load(
            tfds_name, split=tfds_split_name, data_dir=TFDS_DATA_DIR, shuffle_files=False, as_supervised=False # as_supervised=False to get dict with 'image' key
        ).enumerate() # Enumerate to get original TFDS indices

    except Exception as e:
        print(f"ERROR loading TFDS dataset '{tfds_name}': {e}. Please check TFDS_DATA_DIR and dataset name. Exiting.")
        exit()


    required_images_map = {} # Stores original_tfds_idx -> image_numpy
    num_found_in_tfds = 0
    print(f"Iterating through TFDS split '{tfds_split_name}' to find {len(all_required_original_tfds_indices_set)} test images...")

    # Convert TF dataset to numpy iterator *after* setting GPU config
    try:
        tf_dataset_iterator = full_ds_tfds_enumerated.as_numpy_iterator()
    except Exception as e:
        print(f"ERROR creating NumPy iterator from TFDS dataset: {e}. Check TF installation or GPU configuration.")
        exit()


    for original_tfds_idx, item_tfds in tqdm(tf_dataset_iterator,
                                             total=num_total_tfds_images_in_split,
                                             desc="Caching FINAL TEST TFDS images"):
        current_original_tfds_idx = int(original_tfds_idx) # Get the enumerated index
        if current_original_tfds_idx in all_required_original_tfds_indices_set:
            # Ensure 'image' key exists in the returned item
            if 'image' in item_tfds:
                 required_images_map[current_original_tfds_idx] = item_tfds['image']
                 num_found_in_tfds += 1
                 if num_found_in_tfds == len(all_required_original_tfds_indices_set):
                     print(f"\nAll {num_found_in_tfds} required FINAL TEST images found and cached.")
                     break # Stop iterating once all needed images are found
            else:
                 print(f"\nWarning: TFDS item at index {current_original_tfds_idx} does not contain 'image' key. Skipping.")

    print(f"Cached {len(required_images_map)} images for FINAL TEST set from TFDS.")

    if num_found_in_tfds < len(all_required_original_tfds_indices_set):
        print(f"WARNING: Could not find all required test images in TFDS. "
              f"Found {num_found_in_tfds} out of {len(all_required_original_tfds_indices_set)} indexed in NPZ.")
        print("The final test set size will be smaller than expected.")


    # Prepare images_test_numpy and corresponding labels/indices
    # Filter and order based on the original test indices from the NPZ file
    images_test_numpy = []
    labels_test_numeric_ordered = []
    indices_test_ordered = []

    for original_idx, label in zip(original_tfds_test_indices_final, y_test_numeric_broad_final):
        if original_idx in required_images_map:
            images_test_numpy.append(required_images_map[original_idx])
            labels_test_numeric_ordered.append(label)
            indices_test_ordered.append(original_idx)
        else:
            # This warning is already given above, but repeating confirms which specific index was missed.
            # print(f"Warning: Test original_tfds_idx {original_idx} from NPZ not found in cached TFDS data. Skipping this image.")
            pass # Silence this specific warning inside the loop if the summary is sufficient

    print(f"Prepared {len(images_test_numpy)} images and labels for final evaluation after filtering.")

    del required_images_map # Free memory
    gc.collect()
    return indices_test_ordered, images_test_numpy, labels_test_numeric_ordered


# --- Main Evaluation ---
def evaluate():
    print("===== ViT Model Evaluation on Final Unseen Test Set =====")

    # 1. Load Data
    # `original_indices` here are the TFDS indices for the test set, `true_labels_numeric` are their labels.
    original_indices, images_np, true_labels_numeric = load_final_test_data(NPZ_ALL_SPLITS_FILE)

    # --- Use the ViT-specific evaluation transform ---
    transform = get_vit_preprocessing_transform_eval()

    if not images_np:
        print("ERROR: No images loaded for evaluation. Cannot proceed.")
        return

    # 2. Create PyTorch Dataset and DataLoader
    # The TFDSSubsetFeatureDataset class is generic
    eval_dataset = TFDSSubsetFeatureDataset(
        original_indices, # Pass the original indices
        images_np,
        true_labels_numeric,
        transform=transform
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, # Adjust workers based on your system
        pin_memory=True if device.type == 'cuda' else False,
        worker_init_fn=worker_init_fn # Use the top-level worker init function
    )
    print(f"Final Test dataset size for evaluation: {len(eval_dataset)}")

    del images_np # Free memory after dataset creation, DataLoader will handle batches from dataset
    gc.collect()

    # 3. Load Fine-tuned ViT Model
    # --- Call the new ViT loading function ---
    model = load_finetuned_vit_model(SAVED_MODEL_PATH, NUM_BROAD_CATEGORIES)


    # 4. Load Label Encoder to get class names
    print("\n--- Loading Label Encoder ---")
    if not os.path.exists(LABEL_ENCODER_PATH):
        print(f"Warning: Label encoder not found at {LABEL_ENCODER_PATH}. Metrics will use numeric labels.")
        broad_category_names = [f"Class {i}" for i in range(NUM_BROAD_CATEGORIES)]
    else:
        try:
            with open(LABEL_ENCODER_PATH, 'rb') as f:
                label_encoder = pickle.load(f)
            broad_category_names = label_encoder.classes_
            print(f"Loaded broad category names from encoder: {list(broad_category_names)}") # Print as list for readability
        except Exception as e:
            print(f"Error loading label encoder from {LABEL_ENCODER_PATH}: {e}. Metrics will use numeric labels.")
            broad_category_names = [f"Class {i}" for i in range(NUM_BROAD_CATEGORIES)]


    # 5. Perform Inference and Collect Predictions
    all_preds_numeric = []
    # We'll use the `true_labels_numeric` list loaded earlier as the source of truth
    # rather than collecting from the DataLoader batches, which can be affected by skipped samples.

    print("\n--- Running Inference on Final Test Set ---")
    # model is already in model.eval() from load function
    with torch.no_grad(): # IMPORTANT: Disable gradient calculations
        # Added check for empty loader if dataset somehow ended up empty despite warnings
        if len(eval_loader) == 0:
             print("Evaluation DataLoader is empty. Skipping inference.")
        else:
            for inputs, _ in tqdm(eval_loader, desc="Evaluating"): # We get labels in batch but use the pre-loaded list as source of truth
                if inputs is None or inputs.shape[0] == 0:
                    # This check is also in Dataset, but robust to double check here for batch issues
                    tqdm.write("Skipping an empty or None batch during evaluation.")
                    continue

                inputs = inputs.to(device)

                outputs = model(inputs)

                # Get predictions (numeric index)
                _, preds_batch_numeric = torch.max(outputs, 1)

                all_preds_numeric.extend(preds_batch_numeric.cpu().numpy())

    # 6. Align Predictions with True Labels
    # It's possible (though hopefully rare with error handling) that some batches/samples were skipped
    # during data loading or getitem errors. The length of all_preds_numeric might be less than
    # the original `len(true_labels_numeric)`. We should only compare predictions to the
    # true labels for which we successfully got an image and prediction.
    # Since we used the `indices_test_ordered` to build the dataset and the dataset
    # should maintain order if no errors occur within __getitem__, the order should match.
    # If dummy data is returned by __getitem__, the DataLoader batch size might be smaller
    # than expected, but the number of *batches* processed should still correspond to the dataset size.
    # A robust way is to collect *both* predictions and labels from the successful batches.
    # Let's stick to collecting labels from the loader batches for robustness against skipped items.

    # Re-running inference loop to collect *both* predictions and labels from successful batches
    # This ensures the collected predictions and true labels are perfectly aligned.
    print("\n--- Re-running Inference to Collect Aligned Predictions and Labels ---")
    aligned_preds_numeric = []
    aligned_true_labels_numeric = []

    model.eval() # Ensure evaluation mode
    with torch.no_grad():
        if len(eval_loader) == 0:
            print("Evaluation DataLoader is empty. Skipping inference for aligned collection.")
        else:
            for inputs, labels_batch in tqdm(eval_loader, desc="Collecting Results"):
                 if inputs is None or inputs.shape[0] == 0:
                     tqdm.write("Skipping an empty or None batch during collection.")
                     continue

                 inputs = inputs.to(device)
                 labels_batch = labels_batch.to(device) # Labels need to be on device for comparison later

                 outputs = model(inputs)
                 _, preds_batch_numeric = torch.max(outputs, 1)

                 aligned_preds_numeric.extend(preds_batch_numeric.cpu().numpy())
                 aligned_true_labels_numeric.extend(labels_batch.cpu().numpy())

    # Use the collected aligned lists for metrics
    final_true_labels = np.array(aligned_true_labels_numeric)
    final_preds = np.array(aligned_preds_numeric)

    print(f"Collected {len(final_preds)} predictions and {len(final_true_labels)} true labels.")


    # 7. Calculate and Print Metrics
    print("\n--- Evaluation Results ---")
    if len(final_preds) == 0:
        print("No predictions were successfully collected. Cannot calculate metrics.")
        return

    accuracy = accuracy_score(final_true_labels, final_preds)
    print(f"Overall Accuracy on Final Test Set: {accuracy:.4f}")

    print("\nClassification Report (Final Test Set):")
    # Handle cases where some classes might be missing in the collected samples
    try:
        report = classification_report(final_true_labels, final_preds, target_names=broad_category_names, digits=4)
        print(report)
    except ValueError as e:
         print(f"Error generating classification report: {e}. This might happen if samples from some classes are missing.")
         print("Using numeric labels for report:")
         print(classification_report(final_true_labels, final_preds, digits=4))


    print("\nConfusion Matrix (Final Test Set):")
    # Ensure all possible labels (0 to NUM_BROAD_CATEGORIES-1) are included in the confusion matrix
    try:
        cm = confusion_matrix(final_true_labels, final_preds, labels=list(range(NUM_BROAD_CATEGORIES)))
        print(cm)
    except Exception as e:
        print(f"Error generating confusion matrix: {e}. This might happen if labels/predictions are unexpected.")
        # Fallback without specifying all labels, might miss rows/cols for unseen classes
        print("Attempting to generate confusion matrix without specifying all labels:")
        cm = confusion_matrix(final_true_labels, final_preds)
        print(cm)


    # Optional: Save metrics to a file
    metrics_file_path = os.path.join(PLOT_SAVE_DIR, "final_evaluation_metrics_vit.txt") # Added _vit for clarity
    try:
        with open(metrics_file_path, "w") as f:
            f.write(f"Model: ViT\n")
            f.write(f"Saved Model Path: {SAVED_MODEL_PATH}\n")
            f.write(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Evaluated Samples: {len(final_true_labels)}\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            # Try saving with target names, fallback if needed
            try:
                f.write(classification_report(final_true_labels, final_preds, target_names=broad_category_names, digits=4))
            except ValueError:
                 f.write(classification_report(final_true_labels, final_preds, digits=4))
            f.write("\n\nConfusion Matrix:\n")
            f.write(np.array2string(cm))
        print(f"Evaluation metrics saved to {metrics_file_path}")
    except Exception as e:
        print(f"Error saving evaluation metrics to file: {e}")


    # Optional: Visualize the confusion matrix
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        # Ensure labels cover the full range 0..NUM_BROAD_CATEGORIES-1 even if some weren't predicted/present
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=broad_category_names[:cm.shape[1]], # Use names up to matrix size
                    yticklabels=broad_category_names[:cm.shape[0]], # Use names up to matrix size
                    cbar=False)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - Final Test Set (ViT)')
        # plt.show() # Uncomment to display plot immediately
        plot_save_path = os.path.join(PLOT_SAVE_DIR, "confusion_matrix_final_test_vit.png") # Added _vit
        plt.savefig(plot_save_path)
        print(f"Confusion matrix plot saved to {plot_save_path}")
    except ImportError:
        print("Seaborn or Matplotlib not found. Cannot generate confusion matrix plot.")
    except Exception as e:
         print(f"Error generating or saving confusion matrix plot: {e}")

