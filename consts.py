import os


RESOURCES_DIR = 'resources'

SNAPSHOT_DIR = os.path.join(RESOURCES_DIR, 'snapshot')
SNAPSHOT_BASE_NAME = "vira_logs_2022_05"
SNAPSHOT_FILE = os.path.join(SNAPSHOT_DIR, f"{SNAPSHOT_BASE_NAME}.csv")
SNAPSHOT_SPLIT_FILE = os.path.join(SNAPSHOT_DIR, f"{SNAPSHOT_BASE_NAME}_split.csv")
SNAPSHOT_TRAIN_FILE = os.path.join(SNAPSHOT_DIR, f"{SNAPSHOT_BASE_NAME}_train.csv")
SNAPSHOT_TEST_FILE = os.path.join(SNAPSHOT_DIR, f"{SNAPSHOT_BASE_NAME}_test.csv")

INTENT_DATASET_DIR = os.path.join(RESOURCES_DIR, 'intent_expressions')
INTENT_MODEL_DIR = os.path.join('.', 'intent_model')

SILVERDATA_DIR = os.path.join(RESOURCES_DIR, 'silverdata')
PREDICTIONS_DIR = os.path.join(RESOURCES_DIR, 'predictions')

VECTORS_DIR = os.path.join(RESOURCES_DIR, 'vectors')
