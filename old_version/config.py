import os

# Project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Annotation files
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")

TRAIN_ANNOTATIONS = os.path.join(ANNOTATIONS_DIR, "train.csv")
VAL_ANNOTATIONS = os.path.join(ANNOTATIONS_DIR, "val.csv")
TEST_ANNOTATIONS = os.path.join(ANNOTATIONS_DIR, "test.csv")

# Model directory
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "efficientnetv2b0_tampering.keras")

# Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# Training settings
EPOCHS = 15

# Classes
CLASS_NAMES = ["Genuine", "Tampered"]
NUM_CLASSES = 2