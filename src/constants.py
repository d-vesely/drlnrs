# CONSTANTS

import os

_THIS = os.path.abspath(__file__)

# Dataset type is "small" or "large"
DATASET_TYPE = "large"

# Paths to dataset slices
DATASET_PATH = os.path.abspath(
    os.path.join(_THIS, "../../dataset_MIND")
)

TRAIN_PATH = os.path.join(
    DATASET_PATH,
    f"MIND{DATASET_TYPE}_train"
)
DEV_PATH = os.path.join(
    DATASET_PATH,
    f"MIND{DATASET_TYPE}_dev"
)
# Only large version exists for test set
TEST_PATH = os.path.join(
    DATASET_PATH,
    "MINDlarge_test"
)

# Paths to concatenated data
CONCAT_TRAINFULL_PATH = os.path.join(
    DATASET_PATH,
    f"MIND{DATASET_TYPE}_trainfull"
)
CONCAT_ALL_PATH = os.path.join(
    DATASET_PATH,
    f"MINDlarge_all"
)

MODELS_PATH = os.path.abspath(
    os.path.join(_THIS, "../../models")
)

BASE_EMB_PATH = os.path.join(CONCAT_ALL_PATH, "embeddings")
