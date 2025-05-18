# finetune_alexnet_4cat_places_hdf5.py

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
# import tensorflow_datasets as tfds # No longer needed for image loading
# import tensorflow as tf # No longer needed for image loading
import h5py # Need this!
from tqdm import tqdm
import gc # Garbage collector
import random # Needed for worker_init_fn seeding

# --- Configuration ---
# Data Paths
# NPZ_FILE_ALL_SPLITS = r"E:\CV_features\train_test_splits_4cat_balanced\train_test_split_data_4cat_balanced.npz" # NO LONGER NEEDED FOR PYTORCH SCRIPT - HDF5 HAS EVERYTHING
HDF5_IMAGES_PATH = r"E:\CV_features\balanced_4cat_hdf5\balanced_4cat_data.h5" # Path to the HDF5 file containing images and labels
PLACES365_WEIGHTS_PATH = r"E:\CV_Features_CNN_PyTorch_Balanced\alexnet_places365.pth.tar" # Make sure this path is correct

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
MODEL_SAVE_FILENAME = f"finetuned_alexnet_places365_{NUM_BROAD_CATEGORIES}cat_best_balanced.pth" # Updated filename
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_FILENAME)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure output directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- TOP-LEVEL FUNCTION FOR DATALOADER WORKER INITIALIZATION ---
# THIS FUNCTION MUST BE DEFINED AT THE TOP LEVEL TO BE PICKLABLE
def worker_init_fn(worker_id):
    """Initializes worker processes for the DataLoader."""
    # Get worker info
    worker_info = torch.utils.data.get_worker_info()
    # Access the dataset object created in the main process
    dataset_obj = worker_info.dataset

    # Seeding: Ensure different data augmentations across workers
    # Use torch.initial_seed() which is unique per worker
    worker_seed = torch.initial_seed() % (2**32 - 1)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

    # HDF5 file handling:
    # The HDF5 file handle is designed to be opened *within* __getitem__
    # if it's None (meaning it's not open in this worker).
    # This is the recommended way for HDF5 with num_workers > 0.
    # No explicit file opening is needed here, the dataset handles it.
    # The print statements from the dataset's __getitem__ will confirm this logic.
    # worker_info.dataset.h5_file = h5py.File(worker_info.dataset.hdf5_path, 'r', libver='latest') # Alternative: Explicitly open here if __getitem__ didn't handle it


# --- Custom PyTorch Dataset (MODIFIED for HDF5) ---
class HDF5SubsetImageDataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_path, split_name, transform=None):
        """
        Args:
            hdf5_path (str): Path to the HDF5 file containing images and labels.
            split_name (str): 'train', 'val', or 'test', corresponding to HDF5 group names.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.hdf5_path = hdf5_path
        self.split_name = split_name
        self.transform = transform
        self.h5_file = None # Will be opened in __getitem__ (or worker init)

        # Temporarily open to get dataset size and load labels (assuming labels are small)
        print(f"Initializing HDF5 dataset for split '{split_name}' from {hdf5_path}...")
        try:
            with h5py.File(self.hdf5_path, 'r', libver='latest') as f:
                # --- START OF FIX ---
                # Check if the dataset paths exist directly in the file object 'f'
                images_dset_path = f'{split_name}/images'
                labels_dset_path = f'{split_name}/labels'

                if images_dset_path not in f:
                     raise ValueError(f"Images dataset '{images_dset_path}' not found in HDF5 file at {hdf5_path}. Available keys at root: {list(f.keys())}")
                if labels_dset_path not in f:
                     raise ValueError(f"Labels dataset '{labels_dset_path}' not found in HDF5 file at {hdf5_path}. Available keys at root: {list(f.keys())}")
                # --- END OF FIX ---

                # Get dataset size using the correct path
                self.dataset_size = f[images_dset_path].shape[0]
                # Read all labels for this split into memory using the correct path
                # It's generally safe to load labels into memory if the number of samples is reasonable
                self.labels = f[labels_dset_path][:]

        except FileNotFoundError:
            print(f"ERROR: HDF5 file not found at {self.hdf5_path}")
            raise
        except Exception as e:
            # This error message is now more specific after the fix
            print(f"ERROR accessing HDF5 file {self.hdf5_path} or required datasets for split '{split_name}': {e}")
            raise

        if self.dataset_size == 0:
             print(f"WARNING: HDF5 dataset for split '{split_name}' is empty.")
             self.labels = np.array([], dtype=self.labels.dtype) # Ensure labels array is also empty if size is 0


        print(f"Initialized HDF5 dataset for split '{split_name}' with {self.dataset_size} samples.")


    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Handle case where dataset might be empty
        if self.dataset_size == 0:
             # Return dummy data or raise StopIteration (DataLoader handles StopIteration)
             # Raising StopIteration might be cleaner for empty datasets
             raise StopIteration # Or return dummy data if preferred: return torch.zeros(3, IMG_HEIGHT, IMG_WIDTH, dtype=torch.float32), torch.tensor(0, dtype=torch.long)

        # Open the HDF5 file handle in the worker process on first access
        # This is necessary because file handles cannot be pickled and passed between processes
        if self.h5_file is None:
            try:
                self.h5_file = h5py.File(self.hdf5_path, 'r', libver='latest')
                # Access the images dataset using the full path
                self.h5_images_dataset = self.h5_file[f'{self.split_name}/images']
                 # Labels are already loaded into self.labels in __init__
                # print(f"HDF5 file opened in worker process for split {self.split_name}.") # Optional: Debug print
            except Exception as e:
                 print(f"ERROR: Worker failed to open HDF5 file {self.hdf5_path} or access dataset: {e}. Cannot load data for index {idx}.")
                 # Return dummy data to prevent crash, DataLoader should handle this
                 # The shape should match the output of the transform (C, H, W)
                 dummy_img = torch.zeros(3, IMG_HEIGHT, IMG_WIDTH, dtype=torch.float32)
                 dummy_label = torch.tensor(0, dtype=torch.long) # Placeholder label
                 return dummy_img, dummy_label


        # Read image and label from HDF5 using the index 'idx'
        try:
            # HDF5 dataset slicing returns a numpy array
            img_np = self.h5_images_dataset[idx]
            label_val = self.labels[idx] # Get label from in-memory list

            # Convert NumPy array to PIL Image (TFDS images are usually uint8, so PIL works)
            image = Image.fromarray(img_np).convert('RGB')

        except Exception as e:
            print(f"\nERROR: Worker failed to read HDF5 for {self.split_name} split, index {idx}: {e}. Returning dummy image/data.")
            # Return dummy data to prevent crash
            dummy_img = torch.zeros(3, IMG_HEIGHT, IMG_WIDTH, dtype=torch.float32) # Match transform output shape
            # Cannot return a meaningful label without a valid index if reading failed, use a placeholder
            dummy_label = torch.tensor(0, dtype=torch.long) # Placeholder label
            return dummy_img, dummy_label

        # Apply transform
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"\nWARN: Transform error for {self.split_name} split, index {idx}: {e}. Returning dummy image (pre-transform format).")
                # If transform fails, maybe return the image before transform? Or a zero tensor?
                # Let's return a zero tensor to match the expected output shape
                dummy_img = torch.zeros(3, IMG_HEIGHT, IMG_WIDTH, dtype=torch.float32)
                # Return the label we successfully read
                return dummy_img, torch.tensor(label_val, dtype=torch.long)


        return image, torch.tensor(label_val, dtype=torch.long)

    # __del__ is not guaranteed to be called, rely on worker process exit for file closing.
    # def __del__(self):
    #     if self.h5_file is not None:
    #         self.h5_file.close()
    #         self.h5_file = None
    #         print(f"HDF5 file explicitly closed for {self.split_name} split.")


# --- Preprocessing Transform (Same as before) ---
def get_alexnet_preprocessing_transform(is_train=True):
    print(f"\nDefining preprocessing transform for AlexNet ({IMG_HEIGHT}x{IMG_WIDTH} input)...")
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


# --- Data Loading and Preparation Function (Simplified for HDF5) ---
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

    # Basic check to see if the HDF5 file contains the expected splits ('train', 'val') groups
    # The dataset __init__ now checks for the datasets *within* these groups more robustly.
    try:
        with h5py.File(hdf5_path, 'r') as f:
            required_groups = ['train', 'val']
            # We are checking for groups, the dataset __init__ will check for datasets within
            if not all(s in f for s in required_groups):
                 print(f"ERROR: HDF5 file {hdf5_path} does not contain expected groups for splits: {required_groups}. Found: {list(f.keys())}. Exiting.")
                 exit()
            print(f"HDF5 file {hdf5_path} appears valid and contains groups: {required_groups}.")
    except Exception as e:
        print(f"Error checking HDF5 file {hdf5_path}: {e}. Exiting.")
        exit()

    return hdf5_path # Return the path, the dataset will handle opening in workers


# --- Model Loading and Modification (Same as before) ---
def load_and_modify_alexnet_for_finetuning(weights_path, num_classes, freeze_conv_layers=True):
     # ... (this function was already correct for its purpose, leaving as is) ...
     print(f"\n--- Loading and Modifying Pre-trained AlexNet ---")
     try:
         # Use weights=None then load state_dict
         # Try with weights=AlexNet_Weights.IMAGENET1K_V1 if you want ImageNet weights first
         # but the prompt specifically mentions Places365 weights loading logic
         model = models.alexnet(weights=None) # Load architecture
     except TypeError:
         # Fallback for older torchvision versions that used 'pretrained' arg
         try:
             model = models.alexnet(pretrained=False) # Fallback for older torchvision
         except Exception:
             # If both fail, maybe torchvision version is too old or broken
             print("ERROR: Could not load AlexNet architecture. Ensure torchvision is installed correctly.")
             exit()

     print("AlexNet architecture loaded.")

     # Modify the classifier to accept the number of features from the original AlexNet classifier's last layer
     # This is crucial BEFORE loading the Places365 weights if the checkpoint expects 365 outputs
     num_ftrs_original_classifier = model.classifier[6].in_features
     model.classifier[6] = nn.Linear(num_ftrs_original_classifier, 365)
     print("Temporarily adjusted AlexNet final classifier for 365 Places365 classes to match checkpoint structure.")


     print(f"Loading Places365 weights from: {weights_path}")
     if not os.path.exists(weights_path):
         print(f"ERROR: Places365 weights file not found at {weights_path}. Exiting.")
         exit()
     try:
         # Load the checkpoint, mapping to CPU first to save GPU memory during loading
         checkpoint = torch.load(weights_path, map_location='cpu') # map_location=lambda storage, loc: storage is also okay

         # Handle common checkpoint structures (e.g., just state_dict, or dict with 'state_dict')
         state_dict_from_checkpoint = None
         if isinstance(checkpoint, dict):
             # Look for common keys like 'state_dict' or 'model'
             if 'state_dict' in checkpoint:
                 state_dict_from_checkpoint = checkpoint['state_dict']
             elif 'model' in checkpoint: # Sometimes used in tutorials
                 state_dict_from_checkpoint = checkpoint['model']
             else:
                 # Assume the dictionary itself IS the state_dict if no key found
                 state_dict_from_checkpoint = checkpoint
         else:
             # Assume the loaded object IS the state_dict if it's not a dict
             state_dict_from_checkpoint = checkpoint

         if state_dict_from_checkpoint is None:
             raise ValueError(f"Could not find a usable state_dict in the loaded checkpoint from {weights_path}.")

         # Fix potential key mismatches from DataParallel or other wrappers
         new_state_dict = {}
         # Check if keys start with 'module.' (often from DataParallel)
         has_module_prefix = any(k.startswith('module.') for k in state_dict_from_checkpoint.keys())
         # Check for specific "features.module." prefix sometimes seen
         has_features_module_prefix = any('features.module.' in k for k in state_dict_from_checkpoint.keys())

         for k, v in state_dict_from_checkpoint.items():
             name = k
             if has_module_prefix and name.startswith('module.'):
                 name = name[7:] # remove 'module.' prefix
             if has_features_module_prefix:
                 name = name.replace('features.module.', 'features.') # remove features.module. prefix

             new_state_dict[name] = v

         # Load the state dict, strict=True means all keys must match exactly
         # This ensures we loaded the correct model weights
         model.load_state_dict(new_state_dict, strict=True)
         print("Successfully loaded Places365 pre-trained weights into AlexNet (common key prefixes handled).")

     except FileNotFoundError:
         print(f"ERROR: Places365 weights file not found at {weights_path}. Please check the path.")
         exit()
     except RuntimeError as e:
          print(f"ERROR loading state_dict with strict=True: {e}")
          print("Key mismatch likely occurred. Double-check the checkpoint file structure or model definition.")
          print("Consider trying strict=False, but be aware it means some weights weren't loaded.")
          # You could potentially try with strict=False here as a fallback, but strict=True is safer
          # model.load_state_dict(new_state_dict, strict=False)
          exit()
     except Exception as e:
         print(f"An unexpected ERROR occurred loading Places365 weights: {e}")
         print("Ensure the weights file corresponds to an AlexNet model and is not corrupted.")
         exit()

     # Now replace the final classifier layer with one for your number of classes
     # Get the number of input features for the *new* final layer (which was the 365-output layer)
     num_ftrs_final_fc = model.classifier[6].in_features
     model.classifier[6] = nn.Linear(num_ftrs_final_fc, num_classes)
     print(f"Replaced final classifier layer with a new one for {num_classes} target categories.")

     # Configure trainable parameters
     if freeze_conv_layers:
         print("Freezing convolutional layers (model.features)...")
         for param in model.features.parameters():
             param.requires_grad = False
     else:
         print("All model parameters will be trainable (fine-tuning entire network).")

     # Ensure the new final classifier layer is trainable regardless of freeze_conv_layers setting
     for param in model.classifier[6].parameters():
         param.requires_grad = True
     print("Ensured the new final classifier layer is trainable.")

     # Move model to the specified device
     model = model.to(device)
     print(f"Model moved to {device}.")
     return model


# --- Training and Validation Function (Same as before) ---
def train_val_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, model_save_path=None):
    # ... (this function was already mostly correct, leaving as is) ...
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\n--- Starting Training for {num_epochs} Epochs ---")

    # Check if dataloaders are empty before starting epochs
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
                 continue # Skip validation if empty

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            num_samples_in_phase = len(dataloaders[phase].dataset)
            # We already checked for empty datasets before the epoch loop,
            # but this handles cases where num_samples_in_phase might somehow still be zero.
            if num_samples_in_phase == 0:
                print(f"Warning: {phase} dataset is empty during epoch loop. Skipping phase.")
                if phase == 'train': history['train_loss'].append(0); history['train_acc'].append(0)
                else: history['val_loss'].append(0); history['val_acc'].append(0)
                continue


            loader_tqdm = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}", leave=False)
            for inputs, labels in loader_tqdm:
                # Check for empty or None batch data explicitly if your dataset might return it on error
                # The HDF5 dataset's dummy data return logic should provide tensors, but check shape
                if inputs is None or labels is None or inputs.shape[0] == 0:
                    loader_tqdm.write("Skipping an empty or None batch.")
                    continue # Skip this batch entirely

                # Check if batch size is inconsistent (can happen near the end of dataset)
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

                running_loss += loss.item() * batch_size # Use actual batch_size
                running_corrects += torch.sum(preds == labels.data)

                current_loss_item = loss.item()
                current_batch_acc = torch.sum(preds == labels.data).item() / batch_size if batch_size > 0 else 0
                loader_tqdm.set_postfix(loss=f"{current_loss_item:.4f}", acc=f"{current_batch_acc:.4f}")


            epoch_loss = running_loss / num_samples_in_phase
            epoch_acc = running_corrects.double() / num_samples_in_phase # Use .double() for more precise division

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
                if scheduler:
                    scheduler.step()
                print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc.item():.4f} (LR: {optimizer.param_groups[0]['lr']:.2e})")
            else: # phase == 'val'
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                print(f"Val Loss:   {epoch_loss:.4f} Acc: {epoch_acc.item():.4f}")

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    if model_save_path:
                        print(f"New best validation accuracy: {best_acc:.4f}. Saving model state_dict to {model_save_path}")
                        try:
                            # Ensure save directory exists - although main_finetuning_pipeline does this too
                            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                            torch.save(best_model_wts, model_save_path)
                        except Exception as save_e:
                             print(f"Error saving model state_dict to {model_save_path}: {save_e}")

        epoch_time_elapsed = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s")

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    # Load best model weights if saved
    if model_save_path and os.path.exists(model_save_path):
        try:
            # Load map_location=device to ensure it's on the correct device after loading
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            print(f"Loaded best model state_dict from {model_save_path} for final model state.")
        except Exception as load_e:
            print(f"Error loading best model state_dict from {model_save_path}: {load_e}")
            print("Returning model with weights from the end of the last epoch instead.")
    else:
         print("No best model state_dict file found at save path to reload. Returning model with weights from the end of the last epoch.")


    return model, history


# --- Main Execution Pipeline (MODIFIED for HDF5) ---
def main_finetuning_pipeline():
    print("===== AlexNet Places365 Fine-tuning Pipeline (HDF5 Data) =====")

    # 1. Prepare Data (using HDF5 path)
    # load_and_prepare_data_splits_hdf5 now only checks the HDF5 file
    hdf5_data_path = load_and_prepare_data_splits_hdf5(HDF5_IMAGES_PATH)

    # 2. Create Datasets and DataLoaders
    print("\n--- Preparing PyTorch Datasets and DataLoaders from HDF5 ---")
    image_transforms = {
        'train': get_alexnet_preprocessing_transform(is_train=True),
        'val': get_alexnet_preprocessing_transform(is_train=False)
        # Note: Test split dataset can be created similarly using 'test' split_name if needed later
    }

    # Use the HDF5Dataset for train and val splits
    # If a split is empty (e.g., val=0 images), the dataset should still be created but len() will be 0
    try:
        train_dataset = HDF5SubsetImageDataset(
            hdf5_path=hdf5_data_path,
            split_name='train', # Assumes HDF5 has a 'train' group/datasets
            transform=image_transforms['train']
        )
        val_dataset = HDF5SubsetImageDataset(
            hdf5_path=hdf5_data_path,
            split_name='val', # Assumes HDF5 has a 'val' group/datasets
            transform=image_transforms['val']
        )
    except Exception as e:
         print(f"ERROR initializing one of the HDF5 datasets: {e}. Exiting.")
         exit()


    # Increase num_workers if you have more CPU cores and are bottle-necked by data loading
    # Make sure num_workers doesn't exceed your CPU core count to avoid overhead
    # Setting it too high can also cause memory issues or slower performance
    # Start with 0 or 2 and increase if needed. 4 might be okay depending on system.
    # Check system monitors (CPU usage, I/O, RAM) while training to identify bottlenecks.
    dataloader_num_workers = 4 # Set based on your CPU capabilities and system load

    dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=dataloader_num_workers, pin_memory=True if device.type == 'cuda' else False,
                                             worker_init_fn=worker_init_fn), # Use the TOP-LEVEL worker_init_fn
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                           shuffle=False, num_workers=dataloader_num_workers, pin_memory=True if device.type == 'cuda' else False,
                                           worker_init_fn=worker_init_fn) # Use the TOP-LEVEL worker_init_fn
    }
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    # Clean up any potential remaining large objects from initial load if necessary
    gc.collect()
    print("Cleaned up after dataset initialization.")


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
    # Only optimize parameters that are trainable
    parameters_to_optimize = filter(lambda p: p.requires_grad, model_ft.parameters())
    optimizer_ft = optim.Adam(parameters_to_optimize, lr=LEARNING_RATE)
    lr_scheduler = StepLR(optimizer_ft, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    print(f"Optimizer: Adam, Initial LR: {LEARNING_RATE}. LR Scheduler: StepLR (step={LR_STEP_SIZE}, gamma={LR_GAMMA})")


    # 5. Train and Validate Model
    # The train_val_model function now includes checks for empty dataloaders
    model_ft_trained, history = train_val_model(
        model_ft,
        dataloaders,
        criterion,
        optimizer_ft,
        lr_scheduler,
        num_epochs=NUM_EPOCHS,
        model_save_path=MODEL_SAVE_PATH
    )

    print("\n--- Fine-tuning Pipeline Complete ---")
    # Check if a best model was actually saved
    best_model_saved_path = MODEL_SAVE_PATH if os.path.exists(MODEL_SAVE_PATH) else "No best model file was saved."
    print(f"The best fine-tuned model state has been saved to: {best_model_saved_path}")

    # Note: HDF5 file handles are managed by workers opening them in __getitem__
    # and implicitly closed when workers exit. No explicit close needed here for worker handles.
    # The dataset object in the main process might have a handle open if __init__
    # didn't close it immediately, but the current __init__ uses a 'with' statement,
    # so that temporary handle is closed.

    if history:
        print("\n--- Training History ---")
        if history['train_loss']:
            print(f"Final Training Loss: {history['train_loss'][-1]:.4f}, Accuracy: {history['train_acc'][-1]:.4f}")
        if history['val_loss']:
            # Only print final validation stats if validation actually ran
            if len(dataloaders['val'].dataset) > 0:
                 print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}, Accuracy: {history['val_acc'][-1]:.4f}")
            else:
                 print("Validation skipped due to empty dataset.")
