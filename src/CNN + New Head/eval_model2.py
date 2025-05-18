# evaluate_model.py
import torch
import torch.nn as nn
import torchvision.models as models
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

# --- Configuration ---
# Path to your saved fine-tuned model
SAVED_MODEL_PATH = r"E:\CV_Features_CNN_PyTorch_Balanced\alexnet_places365.pth\alexnet_places365.pth" # From the factory
# Path to your NPZ file containing ALL splits (train/val/test_final)
NPZ_ALL_SPLITS_FILE = r"E:\CV_features\all_splits_data_4cat\all_splits_data_4cat.npz"
# Path to the saved label encoder (created by create_all_splits_for_finetuning.py)
LABEL_ENCODER_PATH = r"E:\CV_features\all_splits_data_4cat\broad_label_encoder_4cat.pkl"

TFDS_DATA_DIR = r"E:\CV_imgs"
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32 # Can be larger for inference if memory allows
NUM_BROAD_CATEGORIES = 4 # Must match your model's output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- TF GPU Configuration (same as in training) ---
def configure_tf_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Configured TensorFlow GPU memory growth for {len(gpus)} GPU(s).")
        except RuntimeError as e: print(f"Error setting TF GPU memory growth: {e}.")
    else: print("No GPUs detected by TensorFlow. TF will use CPU.")
configure_tf_gpu()


# --- Reusable Dataset Class (same as in training/finetuning) ---
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
            # print(f"\nERROR: Convert NumPy to PIL for list_idx {list_idx}: {e}. Returning dummy.")
            return torch.zeros(3, IMG_HEIGHT, IMG_WIDTH, dtype=torch.float32), torch.tensor(label_val, dtype=torch.long)
        if self.transform:
            try: image = self.transform(image)
            except Exception as e:
                # print(f"\nWARN: Transform error for list_idx {list_idx}: {e}. Returning dummy.")
                return torch.zeros(3, IMG_HEIGHT, IMG_WIDTH, dtype=torch.float32), torch.tensor(label_val, dtype=torch.long)
        return image, torch.tensor(label_val, dtype=torch.long)


# --- Reusable Preprocessing (use the 'val' / non-augmented version for evaluation) ---
def get_alexnet_preprocessing_transform_eval():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_HEIGHT),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- Load Model Function ---
def load_finetuned_model(model_path, num_classes):
    print(f"Loading fine-tuned model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        exit()
    try:
        model = models.alexnet(weights=None) # Load architecture
    except TypeError:
        model = models.alexnet(pretrained=False)

    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes) # Adjust to your num_classes

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode!
    print("Model loaded and set to evaluation mode.")
    return model

# --- Data Loading for Final Evaluation ---
def load_final_test_data(npz_all_splits_path, tfds_name='places365_small', tfds_split_name='train'):
    print(f"\n--- Loading FINAL TEST Data and Caching Images ---")
    if not os.path.exists(npz_all_splits_path):
        print(f"ERROR: NPZ file for all splits not found at {npz_all_splits_path}.")
        exit()

    print(f"Loading FINAL TEST split data from: {npz_all_splits_path}")
    all_splits_data_npz = np.load(npz_all_splits_path)

    # Load the FINAL TEST data using the correct keys
    original_tfds_test_indices_final = all_splits_data_npz['test_indices_final'].tolist()
    y_test_numeric_broad_final = all_splits_data_npz['test_labels_numeric_final'].tolist()
    print(f"Loaded {len(original_tfds_test_indices_final)} final test original TFDS indices/labels from NPZ.")

    if not original_tfds_test_indices_final:
        print("ERROR: No test indices found in the NPZ file. Cannot proceed.")
        exit()

    all_required_original_tfds_indices_set = set(original_tfds_test_indices_final)

    print(f"Loading TFDS dataset: {tfds_name}, split: {tfds_split_name} for image data")
    ds_info_obj = tfds.builder(tfds_name, data_dir=TFDS_DATA_DIR).info
    num_total_tfds_images_in_split = ds_info_obj.splits[tfds_split_name].num_examples

    full_ds_tfds_enumerated = tfds.load(
        tfds_name, split=tfds_split_name, data_dir=TFDS_DATA_DIR, shuffle_files=False,
    ).enumerate() # Enumerate to get original TFDS indices

    required_images_map = {} # Stores original_tfds_idx -> image_numpy
    num_found_in_tfds = 0
    for original_tfds_idx_tensor, item_tfds in tqdm(full_ds_tfds_enumerated.as_numpy_iterator(),
                                             total=num_total_tfds_images_in_split,
                                             desc="Caching FINAL TEST TFDS images"):
        current_original_tfds_idx = int(original_tfds_idx_tensor) # Get the enumerated index
        if current_original_tfds_idx in all_required_original_tfds_indices_set:
            required_images_map[current_original_tfds_idx] = item_tfds['image']
            num_found_in_tfds += 1
            if num_found_in_tfds == len(all_required_original_tfds_indices_set):
                print(f"\nAll {num_found_in_tfds} required FINAL TEST images found and cached.")
                break # Stop iterating once all needed images are found
    print(f"Cached {len(required_images_map)} images for FINAL TEST set from TFDS.")

    if num_found_in_tfds < len(all_required_original_tfds_indices_set):
        print(f"WARNING: Could not find all required test images. "
              f"Found {num_found_in_tfds} out of {len(all_required_original_tfds_indices_set)}.")

    # Prepare images_test_numpy in the order of original_tfds_test_indices_final
    images_test_numpy = []
    labels_test_numeric_ordered = [] # To ensure labels match the order of images if some are missing
    indices_test_ordered = []

    for original_idx, label in zip(original_tfds_test_indices_final, y_test_numeric_broad_final):
        if original_idx in required_images_map:
            images_test_numpy.append(required_images_map[original_idx])
            labels_test_numeric_ordered.append(label)
            indices_test_ordered.append(original_idx)
        else:
            print(f"Warning: Test original_tfds_idx {original_idx} from NPZ not found in cached TFDS data. Skipping this image.")
            # If this happens, your evaluation will be on fewer images than intended from the NPZ.
            # This should be rare if the splitting script and this loading logic are correct.

    print(f"Prepared {len(images_test_numpy)} images for final evaluation.")

    del required_images_map, full_ds_tfds_enumerated
    gc.collect()
    return indices_test_ordered, images_test_numpy, labels_test_numeric_ordered


# --- Main Evaluation ---
def evaluate():
    print("===== Model Evaluation on Final Unseen Test Set =====")

    # 1. Load Data
    # `original_indices` here are the TFDS indices for the test set, `true_labels_numeric` are their labels.
    original_indices, images_np, true_labels_numeric = load_final_test_data(NPZ_ALL_SPLITS_FILE)
    transform = get_alexnet_preprocessing_transform_eval()

    if not images_np:
        print("ERROR: No images loaded for evaluation. Exiting.")
        return

    eval_dataset = TFDSSubsetFeatureDataset(
        original_indices, # Pass the original indices
        images_np,
        true_labels_numeric,
        transform=transform
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True if device.type == 'cuda' else False
    )
    print(f"Final Test dataset size for evaluation: {len(eval_dataset)}")

    del images_np # Free memory after dataset creation
    gc.collect()

    # 2. Load Fine-tuned Model
    model = load_finetuned_model(SAVED_MODEL_PATH, NUM_BROAD_CATEGORIES)

    # 3. Load Label Encoder to get class names
    if not os.path.exists(LABEL_ENCODER_PATH):
        print(f"Warning: Label encoder not found at {LABEL_ENCODER_PATH}. Metrics will use numeric labels.")
        broad_category_names = [f"Class {i}" for i in range(NUM_BROAD_CATEGORIES)]
    else:
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        broad_category_names = label_encoder.classes_
        print(f"Loaded broad category names from encoder: {broad_category_names}")


    # 4. Perform Inference and Collect Predictions
    all_preds_numeric = []
    all_true_labels_numeric_from_loader = [] # To verify consistency with `true_labels_numeric`

    print("\n--- Running Inference on Final Test Set ---")
    model.eval() # Ensure model is in evaluation mode
    with torch.no_grad(): # IMPORTANT: Disable gradient calculations
        for inputs, labels_batch in tqdm(eval_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            # labels_batch are already numeric tensors from dataset

            outputs = model(inputs)
            _, preds_batch_numeric = torch.max(outputs, 1)

            all_preds_numeric.extend(preds_batch_numeric.cpu().numpy())
            all_true_labels_numeric_from_loader.extend(labels_batch.cpu().numpy())

    # Ensure the labels from loader match the initially loaded true_labels_numeric
    # This is a sanity check. For evaluation, use the `true_labels_numeric` that corresponds to `images_np`
    if not np.array_equal(true_labels_numeric, all_true_labels_numeric_from_loader):
        print("WARNING: Mismatch between initially loaded true labels and labels from DataLoader. Using initially loaded for metrics.")
        # This might happen if some images were skipped during TFDSSubsetFeatureDataset __getitem__ errors.
        # It's safer to use `true_labels_numeric` which was filtered along with `images_np`.

    # 5. Calculate and Print Metrics
    print("\n--- Evaluation Results ---")
    if not all_preds_numeric:
        print("No predictions were made. Cannot calculate metrics.")
        return

    accuracy = accuracy_score(true_labels_numeric, all_preds_numeric)
    print(f"Overall Accuracy on Final Test Set: {accuracy:.4f}")

    print("\nClassification Report (Final Test Set):")
    print(classification_report(true_labels_numeric, all_preds_numeric, target_names=broad_category_names, digits=4))

    print("\nConfusion Matrix (Final Test Set):")
    cm = confusion_matrix(true_labels_numeric, all_preds_numeric, labels=list(range(NUM_BROAD_CATEGORIES)))
    print(cm)

    # Optional: Save metrics to a file
    with open(os.path.join(os.path.dirname(SAVED_MODEL_PATH), "final_evaluation_metrics.txt"), "w") as f:
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(true_labels_numeric, all_preds_numeric, target_names=broad_category_names, digits=4))
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
    print("Evaluation metrics saved to file.")

    # You can visualize the confusion matrix nicely with seaborn/matplotlib
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                 xticklabels=broad_category_names, yticklabels=broad_category_names,
                 cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Final Test Set')
    plt.show()
    plt.savefig(os.path.join(os.path.dirname(SAVED_MODEL_PATH),"confusion_matrix_final_test.png"))
    print("Confusion matrix plot saved.")

