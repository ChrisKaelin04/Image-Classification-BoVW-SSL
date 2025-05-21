# finetune_vit_4cat_places_hdf5.py

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
# Import the ViT model from torchvision
import torchvision.models as models
# Import the specific weights class for the ViT model
from torchvision.models import ViT_B_16_Weights

import torchvision.transforms as transforms
from PIL import Image
import h5py
from tqdm import tqdm
import gc
import random

# --- Configuration ---
# Data Path
HDF5_IMAGES_PATH = r"E:\CV_features\balanced_4cat_hdf5\balanced_4cat_data.h5"

# Model & Training Hyperparameters
# --- CHANGED: Specify ViT model details ---
VIT_MODEL_NAME = 'vit_b_16' # Vision Transformer Base, patch size 16x16
# Note: This ViT model is typically trained on ImageNet-1k and expects 224x224 input.
# The preprocessing will handle this.
NUM_BROAD_CATEGORIES = 4
# IMG_WIDTH, IMG_HEIGHT = 224, 224 # Kept the same as AlexNet, aligns with ViT_B_16_Weights
IMG_WIDTH, IMG_HEIGHT = ViT_B_16_Weights.IMAGENET1K_V1.transforms().resize_size, ViT_B_16_Weights.IMAGENET1K_V1.transforms().crop_size
# Let's double check these - often resize is 256, crop 224. Let's set manually for clarity if needed.
# Check ViT_B_16_Weights.IMAGENET1K_V1.transforms() output
# Typical standard is Resize(256), CenterCrop(224), so let's stick to that input processing.
IMG_INPUT_SIZE = 224 # The size the model expects (crop size)
IMG_RESIZE_SIZE = 256 # The size to resize the smallest edge to before cropping

BATCH_SIZE = 32
LEARNING_RATE = 0.0001 # You might need to tune this for ViT (might need smaller or different schedule)
NUM_EPOCHS = 25       # Tune this for ViT - might converge faster or slower than AlexNet head
# --- CHANGED: ViT fine-tuning strategy ---
# For ViT, we typically freeze the entire transformer block and only train the head.
# The FREEZE_CONV_LAYERS concept doesn't map directly. We will freeze everything except the head.
# We keep a variable to explicitly state the strategy, but the code will enforce freezing all but the head.
FINE_TUNE_HEAD_ONLY = True # True: Freeze transformer encoder, train only the final head.

# Optional LR Scheduler params
LR_STEP_SIZE = 7
LR_GAMMA = 0.1

# Output
MODEL_SAVE_DIR = r"E:\CV_Models_PyTorch_Balanced"
# --- CHANGED: Filename for ViT ---
MODEL_SAVE_FILENAME = f"finetuned_{VIT_MODEL_NAME}_imagenet_{NUM_BROAD_CATEGORIES}cat_best_balanced.pth"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_FILENAME)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure output directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- TOP-LEVEL FUNCTION FOR DATALOADER WORKER INITIALIZATION ---
# This is generic and works for any model, keep as is.
def worker_init_fn(worker_id):
    """Initializes worker processes for the DataLoader."""
    worker_info = torch.utils.data.get_worker_info()
    dataset_obj = worker_info.dataset
    worker_seed = torch.initial_seed() % (2**32 - 1)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    # HDF5 handling is within __getitem__
    pass

# --- Custom PyTorch Dataset (MODIFIED for HDF5) ---
# This is generic and works for any model, keep as is.
class HDF5SubsetImageDataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_path, split_name, transform=None):
        self.hdf5_path = hdf5_path
        self.split_name = split_name
        self.transform = transform
        self.h5_file = None # Will be opened in __getitem__

        print(f"Initializing HDF5 dataset for split '{split_name}' from {hdf5_path}...")
        try:
            with h5py.File(self.hdf5_path, 'r', libver='latest') as f:
                images_dset_path = f'{split_name}/images'
                labels_dset_path = f'{split_name}/labels'

                if images_dset_path not in f:
                     raise ValueError(f"Images dataset '{images_dset_path}' not found in HDF5 file at {hdf5_path}. Available keys at root: {list(f.keys())}")
                if labels_dset_path not in f:
                     raise ValueError(f"Labels dataset '{labels_dset_path}' not found in HDF5 file at {hdf5_path}. Available keys at root: {list(f.keys())}")

                self.dataset_size = f[images_dset_path].shape[0]
                self.labels = f[labels_dset_path][:]

        except FileNotFoundError:
            print(f"ERROR: HDF5 file not found at {self.hdf5_path}")
            raise
        except Exception as e:
            print(f"ERROR accessing HDF5 file {self.hdf5_path} or required datasets for split '{split_name}': {e}")
            raise

        if self.dataset_size == 0:
             print(f"WARNING: HDF5 dataset for split '{split_name}' is empty.")
             self.labels = np.array([], dtype=self.labels.dtype)

        print(f"Initialized HDF5 dataset for split '{split_name}' with {self.dataset_size} samples.")

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.dataset_size == 0:
             raise StopIteration

        if self.h5_file is None:
            try:
                self.h5_file = h5py.File(self.hdf5_path, 'r', libver='latest')
                self.h5_images_dataset = self.h5_file[f'{self.split_name}/images']
            except Exception as e:
                 print(f"ERROR: Worker failed to open HDF5 file {self.hdf5_path} or access dataset: {e}. Cannot load data for index {idx}.")
                 dummy_img = torch.zeros(3, IMG_INPUT_SIZE, IMG_INPUT_SIZE, dtype=torch.float32)
                 dummy_label = torch.tensor(0, dtype=torch.long)
                 return dummy_img, dummy_label

        try:
            img_np = self.h5_images_dataset[idx]
            label_val = self.labels[idx]
            image = Image.fromarray(img_np).convert('RGB')

        except Exception as e:
            print(f"\nERROR: Worker failed to read HDF5 for {self.split_name} split, index {idx}: {e}. Returning dummy image/data.")
            dummy_img = torch.zeros(3, IMG_INPUT_SIZE, IMG_INPUT_SIZE, dtype=torch.float32)
            dummy_label = torch.tensor(0, dtype=torch.long)
            return dummy_img, dummy_label

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"\nWARN: Transform error for {self.split_name} split, index {idx}: {e}. Returning zero tensor image.")
                dummy_img = torch.zeros(3, IMG_INPUT_SIZE, IMG_INPUT_SIZE, dtype=torch.float32)
                return dummy_img, torch.tensor(label_val, dtype=torch.long) # Return valid label if read successfully


        return image, torch.tensor(label_val, dtype=torch.long)


# --- Preprocessing Transform (ADAPTED for standard ImageNet/ViT input) ---
# We will use the standard transforms recommended for ViT_B_16_Weights.IMAGENET1K_V1
# This is similar to the AlexNet transform, ensuring consistency.
def get_standard_preprocessing_transform(is_train=True):
    print(f"\nDefining preprocessing transform for ViT ({IMG_INPUT_SIZE}x{IMG_INPUT_SIZE} input)...")
    # Get the recommended transforms from the weights object
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    # This provides the standard normalization mean/std and required resize/crop sizes
    # transforms = weights.transforms() # This gives the Eval transforms by default

    # Let's define them manually for clarity and to include train augmentations
    # Standard ImageNet normalization values
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(IMG_INPUT_SIZE), # Use IMG_INPUT_SIZE (224)
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        print("Defined training transform with augmentations (RandomResizedCrop, RandomHorizontalFlip).")
    else:
        transform = transforms.Compose([
            transforms.Resize(IMG_RESIZE_SIZE),     # Resize smallest edge to 256
            transforms.CenterCrop(IMG_INPUT_SIZE),  # Take center 224x224 crop
            transforms.ToTensor(),
            normalize
        ])
        print("Defined validation/test transform (Resize, CenterCrop).")
    return transform


# --- Data Loading and Preparation Function (Same as before) ---
# This function only checks and returns the HDF5 path, the Dataset handles loading.
def load_and_prepare_data_splits_hdf5(hdf5_path):
    """
    Prepares PyTorch Datasets using data directly from the HDF5 file.
    Args:
        hdf5_path (str): Path to the HDF5 file containing the splits.
    Returns:
        str: The verified HDF5 path.
    """
    print(f"\n--- Preparing PyTorch Datasets from HDF5 ---")
    if not os.path.exists(hdf5_path):
        print(f"ERROR: HDF5 file not found at {hdf5_path}. Exiting.")
        exit()
    try:
        with h5py.File(hdf5_path, 'r') as f:
            required_groups = ['train', 'val']
            if not all(s in f for s in required_groups):
                 print(f"ERROR: HDF5 file {hdf5_path} does not contain expected groups for splits: {required_groups}. Found: {list(f.keys())}. Exiting.")
                 exit()
            print(f"HDF5 file {hdf5_path} appears valid and contains groups: {required_groups}.")
    except Exception as e:
        print(f"Error checking HDF5 file {hdf5_path}: {e}. Exiting.")
        exit()

    return hdf5_path


# --- Model Loading and Modification (ADAPTED for ViT) ---
# This function now loads a pre-trained ViT and modifies its head.
# --- Model Loading and Modification (ADAPTED for ViT - Correcting .head to .heads) ---
def load_and_modify_vit_for_finetuning(num_classes, fine_tune_head_only=True):
     print(f"\n--- Loading and Modifying Pre-trained ViT ({VIT_MODEL_NAME}) ---")

     try:
         # Load the ViT model with ImageNet-1k weights
         weights = ViT_B_16_Weights.IMAGENET1K_V1
         model = models.vit_b_16(weights=weights)
         print(f"Loaded pre-trained {VIT_MODEL_NAME} with {weights.name} weights.")

     except Exception as e:
         print(f"ERROR loading pre-trained {VIT_MODEL_NAME} model: {e}")
         print("Ensure torchvision is installed correctly and can download weights.")
         exit()

     # --- CORRECTING: Access the classifier via .heads ---
     # The classifier head of the ViT is stored in the 'heads' attribute
     # It's usually an nn.Sequential containing the final linear layer at index 0
     if not hasattr(model, 'heads') or not isinstance(model.heads, nn.Sequential) or len(model.heads) == 0:
          print("ERROR: Model structure unexpected. Does not have a 'heads' Sequential attribute with modules.")
          print(f"Model attributes: {dir(model)}") # Print dir again if this error occurs
          exit()

     # Get the number of input features for the *last* layer in the original head
     # Access the linear layer inside the 'heads' sequential (usually at index 0)
     original_linear_head = model.heads[0]
     if not isinstance(original_linear_head, nn.Linear):
          print("ERROR: The first module in model.heads is not an nn.Linear layer as expected.")
          print(f"model.heads: {model.heads}")
          exit()

     num_ftrs = original_linear_head.in_features # Get the dimension of the features before the head

     # Replace the existing 'heads' sequential with a new Sequential containing your linear layer
     # We replace the entire 'heads' attribute
     # Keep it as a Sequential for consistency, though it only has one layer
     model.heads = nn.Sequential(nn.Linear(num_ftrs, num_classes)) # Create a new Sequential with your new linear layer
     print(f"Replaced original classifier head (was Sequential -> {original_linear_head.__class__.__name__} -> {original_linear_head.out_features} classes) with a new Sequential containing Linear layer for {num_classes} target categories.")


     # Configure trainable parameters
     if fine_tune_head_only:
         print("Freezing all model parameters except the classifier head (model.heads)...")
         # Freeze all parameters first
         for param in model.parameters():
             param.requires_grad = False

         # Then, specifically unfreeze the parameters in the new 'heads' sequential
         # Since model.heads is now a Sequential with our Linear layer, this works correctly
         for param in model.heads.parameters():
             param.requires_grad = True
         print("Ensured only the new classifier head is trainable.")
     else:
         print("All model parameters will be trainable (fine-tuning entire network - WARNING: computationally expensive!).")
         for param in model.parameters(): # Ensure all are trainable if not freezing
              param.requires_grad = True


     # Move model to the specified device
     model = model.to(device)
     print(f"Model moved to {device}.")
     return model


# --- Training and Validation Function (Same as before) ---
# This function is generic and works for any PyTorch model, criterion, optimizer, etc.
def train_val_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, model_save_path=None):
    since = time.time()
    # State dict saving might be slightly different depending on model structure, but .state_dict() is standard.
    # For simplicity and robustness, we save the entire state_dict.
    best_model_wts = model.state_dict()
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\n--- Starting Training for {num_epochs} Epochs ---")

    if len(dataloaders['train'].dataset) == 0:
         print("ERROR: Train dataset is empty. Cannot perform training. Exiting.")
         return model, history
    if len(dataloaders['val'].dataset) == 0:
         print("WARNING: Validation dataset is empty. Training will proceed, but validation will be skipped.")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 15)
        epoch_start_time = time.time()

        for phase in ['train', 'val']:
            if phase == 'val' and len(dataloaders['val'].dataset) == 0:
                 continue

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            num_samples_in_phase = len(dataloaders[phase].dataset)
            if num_samples_in_phase == 0:
                print(f"Warning: {phase} dataset is empty during epoch loop. Skipping phase.")
                if phase == 'train': history['train_loss'].append(0); history['train_acc'].append(0)
                else: history['val_loss'].append(0); history['val_acc'].append(0)
                continue

            loader_tqdm = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}", leave=False)
            for inputs, labels in loader_tqdm:
                if inputs is None or labels is None or inputs.shape[0] == 0:
                    loader_tqdm.write("Skipping an empty or None batch.")
                    continue
                batch_size = inputs.size(0)
                if batch_size == 0:
                    loader_tqdm.write("Skipping batch with size 0.")
                    continue

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)

                current_loss_item = loss.item()
                current_batch_acc = torch.sum(preds == labels.data).item() / batch_size if batch_size > 0 else 0
                loader_tqdm.set_postfix(loss=f"{current_loss_item:.4f}", acc=f"{current_batch_acc:.4f}")


            epoch_loss = running_loss / num_samples_in_phase
            epoch_acc = running_corrects.double() / num_samples_in_phase

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
                # Scheduler steps per epoch *after* the training phase
                if scheduler:
                    scheduler.step()
                # Retrieve current LR after scheduler step
                current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else LEARNING_RATE # Fallback if needed
                print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc.item():.4f} (LR: {current_lr:.2e})")
            else: # phase == 'val'
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                print(f"Val Loss:   {epoch_loss:.4f} Acc: {epoch_acc.item():.4f}")

                # Deep copy the model weights if it's the best validation accuracy seen so far
                # Ensure we compare accuracy using .item()
                if epoch_acc.item() > best_acc:
                    best_acc = epoch_acc.item() # Store scalar value
                    # It's safer to deepcopy if the model structure is complex,
                    # but state_dict copy is standard for models.
                    best_model_wts = model.state_dict()
                    if model_save_path:
                        print(f"New best validation accuracy: {best_acc:.4f}. Saving model state_dict to {model_save_path}")
                        try:
                            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                            torch.save(best_model_wts, model_save_path)
                        except Exception as save_e:
                             print(f"Error saving model state_dict to {model_save_path}: {save_e}")


        epoch_time_elapsed = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s")

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    # Load best model weights for final return value
    if model_save_path and os.path.exists(model_save_path):
        try:
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            print(f"Loaded best model state_dict from {model_save_path} for final model state.")
        except Exception as load_e:
            print(f"Error loading best model state_dict from {model_save_path}: {load_e}")
            print("Returning model with weights from the end of the last epoch instead.")
    else:
         print("No best model state_dict file found at save path to reload. Returning model with weights from the end of the last epoch.")


    return model, history


# --- Main Execution Pipeline (MODIFIED to call ViT loading) ---
def main_finetuning_pipeline():
    print("===== Vision Transformer ImageNet Fine-tuning Pipeline (HDF5 Data) =====")

    # 1. Prepare Data (using HDF5 path) - Same as AlexNet
    hdf5_data_path = load_and_prepare_data_splits_hdf5(HDF5_IMAGES_PATH)

    # 2. Create Datasets and DataLoaders
    # Use the ViT/Standard preprocessing transform
    print("\n--- Preparing PyTorch Datasets and DataLoaders from HDF5 ---")
    image_transforms = {
        'train': get_standard_preprocessing_transform(is_train=True),
        'val': get_standard_preprocessing_transform(is_train=False)
    }

    try:
        train_dataset = HDF5SubsetImageDataset(
            hdf5_path=hdf5_data_path,
            split_name='train',
            transform=image_transforms['train']
        )
        val_dataset = HDF5SubsetImageDataset(
            hdf5_path=hdf5_data_path,
            split_name='val',
            transform=image_transforms['val']
        )
    except Exception as e:
         print(f"ERROR initializing one of the HDF5 datasets: {e}. Exiting.")
         exit()

    dataloader_num_workers = 4 # Adjust based on your system

    dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=dataloader_num_workers, pin_memory=True if device.type == 'cuda' else False,
                                             worker_init_fn=worker_init_fn),
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                           shuffle=False, num_workers=dataloader_num_workers, pin_memory=True if device.type == 'cuda' else False,
                                           worker_init_fn=worker_init_fn)
    }
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    gc.collect()
    print("Cleaned up after dataset initialization.")


    # 3. Load and Modify Model - Call the ViT specific function
    model_ft = load_and_modify_vit_for_finetuning(
        num_classes=NUM_BROAD_CATEGORIES,
        fine_tune_head_only=FINE_TUNE_HEAD_ONLY # Use the new flag
    )
    total_params = sum(p.numel() for p in model_ft.parameters())
    trainable_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable model parameters: {trainable_params:,}") # This should be only the head's params if FINE_TUNE_HEAD_ONLY is True


    # 4. Define Loss, Optimizer, and (Optional) Scheduler
    criterion = nn.CrossEntropyLoss()
    # Ensure optimizer only sees the trainable parameters (the head)
    parameters_to_optimize = filter(lambda p: p.requires_grad, model_ft.parameters())
    # ViTs might benefit from different optimizers like AdamW, but Adam is a fine start
    optimizer_ft = optim.Adam(parameters_to_optimize, lr=LEARNING_RATE)
    lr_scheduler = StepLR(optimizer_ft, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    print(f"Optimizer: Adam, Initial LR: {LEARNING_RATE}. LR Scheduler: StepLR (step={LR_STEP_SIZE}, gamma={LR_GAMMA})")


    # 5. Train and Validate Model - Same function, works generically
    model_ft_trained, history = train_val_model(
        model_ft,
        dataloaders,
        criterion,
        optimizer_ft,
        lr_scheduler,
        num_epochs=NUM_EPOCHS,
        model_save_path=MODEL_SAVE_PATH
    )

    print("\n--- Vision Transformer Fine-tuning Pipeline Complete ---")
    best_model_saved_path = MODEL_SAVE_PATH if os.path.exists(MODEL_SAVE_PATH) else "No best model file was saved."
    print(f"The best fine-tuned model state has been saved to: {best_model_saved_path}")

    if history:
        print("\n--- Training History ---")
        if history['train_loss']:
            print(f"Final Training Loss: {history['train_loss'][-1]:.4f}, Accuracy: {history['train_acc'][-1]:.4f}")
        if history['val_loss'] and len(dataloaders['val'].dataset) > 0:
            print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}, Accuracy: {history['val_acc'][-1]:.4f}")
        elif history['val_loss']: # Case where val_loss list exists but validation was skipped
             print("Validation skipped due to empty dataset.")

