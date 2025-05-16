# finetune_alexnet_places365.py

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR # Optional: Learning rate scheduler
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import tensorflow_datasets as tfds
import tensorflow as tf # Used for TFDS loading
from tqdm import tqdm
import gc # Garbage collector

# --- Configuration ---
# Data Paths
NPZ_FILE_ALL_SPLITS = r"E:\CV_features\all_splits_data_4cat\all_splits_data_4cat.npz" # NEW PATH
TFDS_DATA_DIR = r"E:\CV_imgs"
PLACES365_WEIGHTS_PATH = r"E:\CV_Features_CNN_PyTorch_Balanced\alexnet_places365.pth.tar"

# Model & Training Hyperparameters
NUM_BROAD_CATEGORIES = 4  # Number of output classes for your specific problem
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32           # Adjust based on GPU memory
LEARNING_RATE = 0.0001    # Initial learning rate (tune this!)
NUM_EPOCHS = 25           # Number of training epochs (tune this!)
FREEZE_CONV_LAYERS = True # If True, freeze model.features; otherwise, fine-tune all layers
# Optional LR Scheduler params
LR_STEP_SIZE = 7          # Scheduler: decay LR every N epochs
LR_GAMMA = 0.1            # Scheduler: LR decay factor

# Output
MODEL_SAVE_DIR = r"E:\CV_Models_PyTorch_Balanced"
MODEL_SAVE_FILENAME = f"finetuned_alexnet_places365_{NUM_BROAD_CATEGORIES}cat_best_v2.pth"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_FILENAME)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure output directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- TensorFlow GPU Configuration (to play nice with PyTorch) ---
def configure_tf_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Allow memory growth to avoid TensorFlow pre-allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Configured TensorFlow GPU memory growth for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(f"Error setting TF GPU memory growth: {e}. TF might still grab GPU memory.")
            # Fallback: Try to make TF use CPU only if PyTorch is primary
            # try:
            #     tf.config.set_visible_devices([], 'GPU')
            #     print("Set TensorFlow to use CPU only.")
            # except RuntimeError as e_cpu:
            #     print(f"Error setting TF to CPU only: {e_cpu}")
    else:
        print("No GPUs detected by TensorFlow. TF will use CPU.")

configure_tf_gpu()

# --- Custom PyTorch Dataset (from your original script) ---
class TFDSSubsetFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, original_tfds_indices_list, image_numpy_list, broad_numeric_labels_list, transform=None):
        self.original_tfds_indices = original_tfds_indices_list # For reference/debugging, not directly used by model
        self.images_numpy = image_numpy_list
        self.labels = broad_numeric_labels_list
        self.transform = transform
        if not (len(self.original_tfds_indices) == len(self.images_numpy) == len(self.labels)):
            raise ValueError("Indices, images, and labels lists must have the same length.")

    def __len__(self):
        return len(self.images_numpy)

    def __getitem__(self, list_idx):
        original_tfds_idx_val = self.original_tfds_indices[list_idx]
        img_np = self.images_numpy[list_idx]
        label_val = self.labels[list_idx]
        try:
            image = Image.fromarray(img_np).convert('RGB')
        except Exception as e:
            print(f"\nERROR: Convert NumPy to PIL for original_tfds_idx {original_tfds_idx_val} (list_idx {list_idx}): {e}. Returning dummy image/data.")
            dummy_img = torch.zeros(3, IMG_HEIGHT, IMG_WIDTH, dtype=torch.float32) # Match transform output
            return dummy_img, torch.tensor(label_val, dtype=torch.long)
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"\nWARN: Transform error for original_tfds_idx {original_tfds_idx_val} (list_idx {list_idx}): {e}. Returning dummy image/data.")
                dummy_img = torch.zeros(3, IMG_HEIGHT, IMG_WIDTH, dtype=torch.float32)
                return dummy_img, torch.tensor(label_val, dtype=torch.long)
        return image, torch.tensor(label_val, dtype=torch.long)


# --- Preprocessing Transform (from your original script) ---
def get_alexnet_preprocessing_transform(is_train=True):
    print(f"\nDefining preprocessing transform for AlexNet ({IMG_HEIGHT}x{IMG_WIDTH} input)...")
    # Augmentations for training, simpler for validation/test
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(IMG_HEIGHT),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("Defined training transform with augmentations.")
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_HEIGHT),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("Defined validation/test transform.")
    return transform


# --- Data Loading and Preparation Function ---
def load_and_prepare_data_splits(npz_path, tfds_name='places365_small', tfds_split_name='train'): # Renamed for clarity
    print(f"\n--- Loading Data Splits and Caching Images (Train/Validation) ---")
    if not os.path.exists(npz_path):
        print(f"ERROR: NPZ file for splits not found at {npz_path}.")
        exit()

    print(f"Loading train/validation split data from: {npz_path}")
    split_data_npz = np.load(npz_path)

    # Load TRAIN data from NPZ
    original_tfds_train_indices_npz = split_data_npz['train_indices'].tolist()
    y_train_numeric_broad_npz = split_data_npz['train_labels_numeric'].tolist()

    # Load VALIDATION data from NPZ (using 'val_indices' and 'val_labels_numeric' keys)
    original_tfds_val_indices_npz = split_data_npz['val_indices'].tolist()
    y_val_numeric_broad_npz = split_data_npz['val_labels_numeric'].tolist()

    print(f"Loaded {len(original_tfds_train_indices_npz)} train indices/labels from NPZ.")
    print(f"Loaded {len(original_tfds_val_indices_npz)} validation indices/labels from NPZ.")

    # Combine indices from both train and val splits to cache all necessary images
    all_required_original_tfds_indices_set = set(original_tfds_train_indices_npz + original_tfds_val_indices_npz)
    print(f"Identified {len(all_required_original_tfds_indices_set)} unique ORIGINAL TFDS indices required for train & val.")

    print(f"Loading TFDS dataset: {tfds_name}, split: {tfds_split_name} for image data")
    ds_info_obj = tfds.builder(tfds_name, data_dir=TFDS_DATA_DIR).info
    num_total_tfds_images = ds_info_obj.splits[tfds_split_name].num_examples

    full_ds_tfds = tfds.load(
        tfds_name,
        split=tfds_split_name,
        data_dir=TFDS_DATA_DIR,
        shuffle_files=False,
    ).enumerate()

    # Create a map for labels for quick lookup if needed, though not strictly necessary here
    # as we are directly using the labels from the NPZ.
    # However, it's useful if we were to verify consistency.
    npz_idx_to_train_label_map = {idx: lbl for idx, lbl in zip(original_tfds_train_indices_npz, y_train_numeric_broad_npz)}
    npz_idx_to_val_label_map = {idx: lbl for idx, lbl in zip(original_tfds_val_indices_npz, y_val_numeric_broad_npz)}


    required_data_map = {} # Stores original_tfds_idx -> {'image': image_numpy}
    num_found_in_tfds = 0
    for original_tfds_idx_tensor, item_tfds in tqdm(full_ds_tfds.as_numpy_iterator(),
                                             total=num_total_tfds_images,
                                             desc="Caching TFDS images for Train/Val"):
        current_original_tfds_idx = int(original_tfds_idx_tensor)
        if current_original_tfds_idx in all_required_original_tfds_indices_set:
            required_data_map[current_original_tfds_idx] = {'image': item_tfds['image']}
            num_found_in_tfds += 1
            if num_found_in_tfds == len(all_required_original_tfds_indices_set):
                print(f"\nAll {num_found_in_tfds} required images for train/val found and cached. Breaking TFDS iteration.")
                break
    print(f"Cached {len(required_data_map)} images from TFDS for train/val.")

    if num_found_in_tfds < len(all_required_original_tfds_indices_set):
        missing_count = len(all_required_original_tfds_indices_set) - num_found_in_tfds
        print(f"WARNING: Only found {num_found_in_tfds} of {len(all_required_original_tfds_indices_set)} required images ({missing_count} missing for train/val).")

    # Prepare final lists for datasets_prepared
    datasets_prepared = {'train': {}, 'val': {}}

    # Training set
    train_images_numpy = []
    # train_labels_numeric = y_train_numeric_broad_npz # Direct use
    # train_actual_original_indices = original_tfds_train_indices_npz # Direct use
    train_actual_original_indices_filtered = []
    train_labels_numeric_filtered = []
    for original_idx, label in zip(original_tfds_train_indices_npz, y_train_numeric_broad_npz):
        if original_idx in required_data_map:
            train_images_numpy.append(required_data_map[original_idx]['image'])
            train_actual_original_indices_filtered.append(original_idx)
            train_labels_numeric_filtered.append(label)
        else:
            print(f"Warning: Train original_tfds_idx {original_idx} from NPZ not found in cached TFDS data. Skipping.")
    datasets_prepared['train']['images_numpy'] = train_images_numpy
    datasets_prepared['train']['labels_numeric'] = train_labels_numeric_filtered
    datasets_prepared['train']['original_tfds_indices'] = train_actual_original_indices_filtered
    print(f"Prepared {len(train_images_numpy)} images for TRAIN split.")


    # Validation set
    val_images_numpy = []
    # val_labels_numeric = y_val_numeric_broad_npz # Direct use
    # val_actual_original_indices = original_tfds_val_indices_npz # Direct use
    val_actual_original_indices_filtered = []
    val_labels_numeric_filtered = []
    for original_idx, label in zip(original_tfds_val_indices_npz, y_val_numeric_broad_npz):
        if original_idx in required_data_map:
            val_images_numpy.append(required_data_map[original_idx]['image'])
            val_actual_original_indices_filtered.append(original_idx)
            val_labels_numeric_filtered.append(label)
        else:
            print(f"Warning: Val original_tfds_idx {original_idx} from NPZ not found in cached TFDS data. Skipping.")

    datasets_prepared['val']['images_numpy'] = val_images_numpy
    datasets_prepared['val']['labels_numeric'] = val_labels_numeric_filtered
    datasets_prepared['val']['original_tfds_indices'] = val_actual_original_indices_filtered
    print(f"Prepared {len(val_images_numpy)} images for VALIDATION split.")


    del required_data_map, full_ds_tfds # Clean up
    gc.collect()
    print("Cleaned up intermediate TFDS objects and image cache map for train/val.")
    return datasets_prepared

# --- Model Loading and Modification ---
def load_and_modify_alexnet_for_finetuning(weights_path, num_classes, freeze_conv_layers=True):
    print(f"\n--- Loading and Modifying Pre-trained AlexNet ---")
    try:
        model = models.alexnet(weights=None) # Load architecture
    except TypeError:
        model = models.alexnet(pretrained=False) # Fallback for older torchvision
    print("AlexNet architecture loaded.")

    # Modify the final classifier layer for Places365 (365 classes) *before* loading weights
    # This ensures the state_dict keys match if the checkpoint is for a 365-class model
    num_ftrs_original_classifier = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs_original_classifier, 365)
    print("Temporarily adjusted AlexNet final classifier for 365 Places365 classes to match checkpoint.")

    print(f"Loading Places365 weights from: {weights_path}")
    if not os.path.exists(weights_path):
        print(f"ERROR: Places365 weights file not found at {weights_path}. Exiting.")
        exit()
    try:
        checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage) # Load to CPU first
        state_dict_from_checkpoint = None
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict_from_checkpoint = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict_from_checkpoint = checkpoint['model']
            else:
                state_dict_from_checkpoint = checkpoint # Assume it's the state_dict itself
        else:
            state_dict_from_checkpoint = checkpoint # Assume it's the state_dict itself

        if state_dict_from_checkpoint is None:
            raise ValueError(f"Could not find state_dict in the loaded checkpoint from {weights_path}.")

        # Fix keys if they have 'module.' prefix (from DataParallel saving)
        new_state_dict = {}
        for k, v in state_dict_from_checkpoint.items():
            name = k
            if k.startswith('module.'):
                name = k[7:]
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=True)
        print("Successfully loaded Places365 pre-trained weights into AlexNet.")

    except Exception as e:
        print(f"ERROR loading Places365 weights: {e}")
        print("Ensure the weights file corresponds to an AlexNet model with a final layer for 365 classes.")
        exit()

    # Now, replace the final classifier for *our* number of classes
    num_ftrs_final_fc = model.classifier[6].in_features # Should be 4096 if Places365 weights loaded correctly
    model.classifier[6] = nn.Linear(num_ftrs_final_fc, num_classes)
    print(f"Replaced final classifier layer with a new one for {num_classes} target categories.")

    # Freeze layers strategy
    if freeze_conv_layers:
        print("Freezing convolutional layers (model.features)...")
        for param in model.features.parameters():
            param.requires_grad = False
        # Optionally, you could also freeze earlier FC layers if desired:
        # for i in range(4): # Freeze classifier layers 0, 1, 2, 3
        #    for param in model.classifier[i].parameters():
        #        param.requires_grad = False
        # print("Kept model.classifier layers 4, 5, and the new 6 trainable.")
    else:
        print("All model parameters will be trainable (fine-tuning entire network).")

    # Ensure the newly added classifier layer is trainable
    for param in model.classifier[6].parameters():
        param.requires_grad = True

    model = model.to(device)
    return model


# --- Training and Validation Function ---
def train_val_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, model_save_path=None):
    since = time.time()
    best_model_wts = model.state_dict() # Initialize with current model weights
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\n--- Starting Training for {num_epochs} Epochs ---")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 15)
        epoch_start_time = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            num_samples_in_phase = len(dataloaders[phase].dataset)

            # Iterate over data.
            loader_tqdm = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}", leave=False)
            for inputs, labels in loader_tqdm:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                loader_tqdm.set_postfix(loss=loss.item(), acc=torch.sum(preds == labels.data).item() / inputs.size(0))


            epoch_loss = running_loss / num_samples_in_phase
            epoch_acc = running_corrects.double() / num_samples_in_phase

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
                if scheduler:
                    scheduler.step() # Step the LR scheduler
                print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} (LR: {optimizer.param_groups[0]['lr']:.2e})")
            else: # phase == 'val'
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                print(f"Val Loss:   {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                # Deep copy the model if it's the best so far
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict() # Save the weights
                    if model_save_path:
                        print(f"New best validation accuracy: {best_acc:.4f}. Saving model to {model_save_path}")
                        torch.save(best_model_wts, model_save_path) # Save the best weights

        epoch_time_elapsed = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s")


    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    # Load best model weights
    if model_save_path and os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print("Loaded best model weights for final model state.")
    else: # If no save path or if saving failed, use the weights from the epoch that had best_acc
        model.load_state_dict(best_model_wts)
        print("Loaded model weights from the best epoch during training.")

    return model, history


# --- Main Execution Pipeline ---
def main_finetuning_pipeline():
    print("===== AlexNet Places365 Fine-tuning Pipeline (v2 with Train/Val/Test structure) =====")
    configure_tf_gpu() # Call TF GPU config

    # 1. Load and Prepare Data Splits (MODIFIED TO USE NEW NPZ and 'val' set)
    data_splits = load_and_prepare_data_splits(NPZ_FILE_ALL_SPLITS) # Use the new NPZ file

    # 2. Create Datasets and DataLoaders
    print("\n--- Preparing PyTorch Datasets and DataLoaders ---")
    image_transforms = {
        'train': get_alexnet_preprocessing_transform(is_train=True),
        'val': get_alexnet_preprocessing_transform(is_train=False) # 'val' uses non-augmented transform
    }

    train_dataset = TFDSSubsetFeatureDataset(
        data_splits['train']['original_tfds_indices'], # Use data from 'train' key
        data_splits['train']['images_numpy'],
        data_splits['train']['labels_numeric'],
        transform=image_transforms['train']
    )
    # The validation set for the training loop now comes from the 'val' split in the NPZ
    val_dataset = TFDSSubsetFeatureDataset(
        data_splits['val']['original_tfds_indices'],   # Use data from 'val' key
        data_splits['val']['images_numpy'],
        data_splits['val']['labels_numeric'],
        transform=image_transforms['val']
    )

    dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=4, pin_memory=True if device.type == 'cuda' else False),
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, # Use val_dataset
                                           shuffle=False, num_workers=4, pin_memory=True if device.type == 'cuda' else False)
    }
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    del data_splits # Clean up
    gc.collect()
    print("Cleaned up initial image/label data arrays from RAM.")

    # 3. Load and Modify Model
    model_ft = load_and_modify_alexnet_for_finetuning(
        PLACES365_WEIGHTS_PATH,
        num_classes=NUM_BROAD_CATEGORIES,
        freeze_conv_layers=FREEZE_CONV_LAYERS
    )
    total_params = sum(p.numel() for p in model_ft.parameters())
    trainable_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable model parameters: {trainable_params:,}")

    # 4. Define Loss, Optimizer, and (Optional) Scheduler
    criterion = nn.CrossEntropyLoss()
    parameters_to_optimize = filter(lambda p: p.requires_grad, model_ft.parameters())
    optimizer_ft = torch.optim.Adam(parameters_to_optimize, lr=LEARNING_RATE) # Corrected optim.Adam
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=LR_STEP_SIZE, gamma=LR_GAMMA) # Corrected optim.lr_scheduler
    print(f"Optimizer: Adam, LR: {LEARNING_RATE}. LR Scheduler: StepLR (step={LR_STEP_SIZE}, gamma={LR_GAMMA})")

    # 5. Train and Validate Model
    model_ft_trained, history = train_val_model(
        model_ft,
        dataloaders, # This now correctly uses the 'val' split for validation
        criterion,
        optimizer_ft,
        lr_scheduler,
        num_epochs=NUM_EPOCHS,
        model_save_path=MODEL_SAVE_PATH
    )

    print("\n--- Fine-tuning Pipeline Complete ---")
    best_model_saved_path = MODEL_SAVE_PATH if os.path.exists(MODEL_SAVE_PATH) else "Save path not specified or saving failed."
    print(f"The best fine-tuned model state has been saved to: {best_model_saved_path}")

    # --- Optional: Immediate Post-Training Analysis (using history) ---
    if history:
        print("\n--- Training History (Last Epoch) ---") # Note: history contains per-epoch values
        # Accessing the last element for final values
        if history['train_loss']: print(f"Final Training Loss: {history['train_loss'][-1]:.4f}, Accuracy: {history['train_acc'][-1]:.4f}")
        if history['val_loss']: print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}, Accuracy: {history['val_acc'][-1]:.4f}")