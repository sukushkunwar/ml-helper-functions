
from helpers.data_utils import load_hf_dataset

DATASET_NAME = "train"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"
VALIDATION_SPLIT = "validation"
RANDOM_SEED = 42
MODEL_NAME = ["bert-base-uncased", "finBERT"] #list of models


#load datasets
train_ds = load_hf_dataset(DATASET_NAME, TRAIN_SPLIT )
breakpoint()
test_ds = load_hf_dataset(DATASET_NAME, TEST_SPLIT )
val_ds = load_hf_dataset(DATASET_NAME, VALIDATION_SPLIT )