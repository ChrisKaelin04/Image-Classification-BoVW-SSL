# This file is the third part of the BoVWs pipeline. We already extracted the features and created the vocabulary. Now we will create the histograms for each image using the vocabulary we just created.
import numpy as np
import os
import pickle
import joblib # For loading KMeans models
from tqdm import tqdm
from sklearn.preprocessing import normalize

FEATURES_DIR = "E:\CV_features"
SPLITS_DIR = os.path.join(FEATURES_DIR, "train_test_splits_4cat_revised")
NPZ_FILE = os.path.join(SPLITS_DIR, "train_test_split_data_4cat_revised.npz")
LABEL_ENCODER_FILE = os.path.join(SPLITS_DIR, "broad_label_encoder_4cat_revised.pkl")