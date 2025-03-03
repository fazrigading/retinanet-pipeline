import torch

BATCH_SIZE = 8 # Increase / decrease according to GPU memeory.
RESIZE_TO = 640 # Base image resolution transforms.
NUM_EPOCHS = 100 # Number of epochs to train for.
NUM_WORKERS = 4 # Number of parallel workers for data loading.
LR = 0.001667 # Initial learning rate. 
LRF = 0.1 # Learning rate factor by which it will be reduced.
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Keep `resolutions=None` for not using multi-resolution training,
# else it will be 50% lower than base `RESIZE_TO`, then base `RESIZE_TO`, 
# and 50% higher than base `RESIZE_TO`
RESOLUTIONS = [
    (int(RESIZE_TO/2), int(RESIZE_TO/2)), 
    (int(RESIZE_TO/1.777), int(RESIZE_TO/1.777)), 
    (int(RESIZE_TO/1.5), int(RESIZE_TO/1.5)), 
    (int(RESIZE_TO/1.333), int(RESIZE_TO/1.333)), 
    (RESIZE_TO, RESIZE_TO), 
    (int(RESIZE_TO*1.333), int(RESIZE_TO*1.333)), 
    (int(RESIZE_TO*1.5), int(RESIZE_TO*1.5)), 
    (int(RESIZE_TO*1.777), int(RESIZE_TO*1.777)), 
    (int(RESIZE_TO*2), int(RESIZE_TO*2))
]
# RESOLUTIONS = None

# Training images and XML files directory.
TRAIN_IMG = 'data/ganoderma-3-8/train'
TRAIN_ANNOT = 'data/ganoderma-3-8/train'
# Validation images and XML files directory.
VALID_IMG = 'data/ganoderma-3-8/valid'
VALID_ANNOT = 'data/ganoderma-3-8/valid'
# Test images and XML files directory.
TEST_IMG = 'data/ganoderma-3-8/test'
TEST_ANNOT = 'data/ganoderma-3-8/test'

# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__',
    'ganoderma', 
    'primordium'
]

NUM_CLASSES = len(CLASSES)

# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Automatic Mixed Preicision?
AMP = True

# If kept None, it will be incremental as exp1, exp2,
# else it will be name provided.
PROJECT_NAME = 'ganoderma-retinanet-3-8' 
