import numpy as np
import os
import matplotlib.pyplot as plt
# import seaborn as sns # We will NOT use seaborn for this function
import warnings # Keep warnings for general use
from sklearn.metrics import confusion_matrix # Just for example usage below


# --- Modified Confusion Matrix Plotting Function ---
def plot_confusion_matrix_matplotlib_only(cm, classes, plot_title='Confusion matrix', cmap=plt.cm.Blues, results_path=None, filename=None):
    """
    This function plots the confusion matrix using only Matplotlib.
    It takes a pre-computed confusion matrix as a NumPy array.

    Args:
        cm (np.ndarray): The confusion matrix as a NumPy array.
                         Rows are true labels, columns are predicted labels.
        classes (list): A list of class names (strings) corresponding to the labels.
        plot_title (str): Title for the plot.
        cmap (matplotlib.colors.Colormap): Colormap for the heatmap. Defaults to plt.cm.Blues.
        results_path (str, optional): Directory to save the plot. If None, plot is shown but not saved.
        filename (str, optional): Filename for the saved plot. Requires results_path.
    """
    if not isinstance(cm, np.ndarray):
        raise TypeError("Confusion matrix (cm) must be a NumPy array.")
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("Confusion matrix (cm) must be a 2D square array.")
    if len(classes) != cm.shape[0]:
        raise ValueError("Number of classes must match the dimensions of the confusion matrix.")

    plt.figure(figsize=(max(8, len(classes)), max(6, len(classes)*0.8)))

    # Use imshow to display the matrix
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(plot_title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Set up ticks and labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right') # Rotate x-labels for readability
    plt.yticks(tick_marks, classes)

    plt.tight_layout()

    # Add text annotations
    # Determine text color based on background color for readability
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Use a slightly larger font for numbers in cells
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=10) # Adjust fontsize as needed

    if results_path and filename:
        os.makedirs(results_path, exist_ok=True)
        full_path = os.path.join(results_path, filename)
        plt.savefig(full_path, bbox_inches='tight') # bbox_inches='tight' helps prevent labels from being cut off
        print(f"Saved confusion matrix to {full_path}")
    else:
        plt.show() # If no path/filename provided, just show the plot

    plt.close()


# --- Example Usage (How you would call it) ---
if __name__ == '__main__':
    # Your class names
    class_names = [
        "Indoor Residential",
        "Indoor Public/Commercial",
        "Outdoor Natural",
        "Outdoor Urban"
    ]

    # --- EXAMPLE: How to get your 'text version' into a NumPy array ---
    # You would typically have this from sklearn.metrics.confusion_matrix(y_true, y_pred)
    # For demonstration, let's create a sample confusion matrix as a NumPy array.
    # This is what you would paste in as 'my_text_cm'.

    # Sample "text version" confusion matrix (replace with your actual data)
    # This matrix should be 4x4 for your 4 classes.
    # Example:
    # True Positives on diagonal
    # Off-diagonal are misclassifications
    my_text_cm = np.array([
        [3569,  941,   94,  396],  # True: Indoor Residential
        [657, 4131,   39 , 173],  # True: Indoor Public/Commercial
        [34,   39, 4487,  440],  # True: Outdoor Natural
        [347,  167,  610, 3876]   # True: Outdoor Urban
    ])

    print("--- Simulating your text CM input ---")
    print("Example Confusion Matrix (NumPy array):")
    print(my_text_cm)
    print("\n")

    # Define a dummy output directory for the example
    dummy_results_dir = "temp_cm_plots"
    os.makedirs(dummy_results_dir, exist_ok=True)


    # Call the new plotting function
    print("Calling plot_confusion_matrix_matplotlib_only...")
    plot_confusion_matrix_matplotlib_only(
        cm=my_text_cm,
        classes=class_names,
        plot_title='ViT Fine-Tuning (Acc: 80.32%, F1: 80.21%)',
        cmap=plt.cm.Blues, # Or plt.cm.Blues, etc.
        results_path=dummy_results_dir,
        filename='example_text_cm_plot.png'
    )

    print("\n--- Example of integration into your existing train_and_evaluate_xgb function ---")
    print("Inside train_and_evaluate_xgb, you would change:")
    print("plot_confusion_matrix(conf_matrix_xgb, classes=target_class_names, ...)")
    print("to:")
    print("plot_confusion_matrix_matplotlib_only(conf_matrix_xgb, classes=target_class_names, ...)")

    # Clean up dummy dir (optional)
    # import shutil
    # shutil.rmtree(dummy_results_dir)