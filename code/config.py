# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN


_C = CN()

# Base config files
_C.BASE = ['']

# Toolbox mode
_C.TOOLBOX_MODE = ""


### -----------------------------------------------------------------------------
### -----------------------------------------------------------------------------
# DATA SETTINGS (START)
### -----------------------------------------------------------------------------
### -----------------------------------------------------------------------------

_C.DATA = CN()

### -----------------------------------------------------------------------------
# Global Preprocess Settings 
# Train/Val/Test Preprocessing settings are the same if DATA.PREPROCESS.GLOBAL_PREPROCESS_SETTINGS = True
### -----------------------------------------------------------------------------
_C.DATA.PREPROCESS = CN()
_C.DATA.PREPROCESS.GLOBAL_PREPROCESS_SETTINGS = True # Apply Preprocess Data Settings Globally? (To Train/Val/Test Data)
_C.DATA.PREPROCESS.DO_CHUNK = True
_C.DATA.PREPROCESS.CHUNK_LENGTH = 180
_C.DATA.PREPROCESS.DYNAMIC_DETECTION = False
_C.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY = 180
_C.DATA.PREPROCESS.CROP_FACE = True
_C.DATA.PREPROCESS.LARGE_FACE_BOX = True
_C.DATA.PREPROCESS.LARGE_BOX_COEF = 1.5
_C.DATA.PREPROCESS.W = 128
_C.DATA.PREPROCESS.H = 128
_C.DATA.PREPROCESS.DATA_TYPE = ['']
_C.DATA.PREPROCESS.LABEL_TYPE = ''


### -----------------------------------------------------------------------------
# Global Label settings
### -----------------------------------------------------------------------------
_C.DATA.LABELS = CN()
_C.DATA.LABELS.GLOBAL_LABELS_SETTINGS = True # Global Data Labels
_C.DATA.LABELS.LABEL_LIST = ['first_label']
_C.DATA.LABELS.USED_LABELS = ['first_label']


### -----------------------------------------------------------------------------
# Training Data Settings
### -----------------------------------------------------------------------------
_C.DATA.TRAIN = CN()
_C.DATA.TRAIN.DATASET = '' # name of datasets (eg. 'PURE', 'UBFC')
_C.DATA.TRAIN.FS = 0 # video data frame sampling rate
_C.DATA.TRAIN.DATA_PATH = '' # base path to raw data directory
_C.DATA.TRAIN.CACHED_PATH = 'PreprocessedData' # base path to cached preprocessed datasets
_C.DATA.TRAIN.EXP_DATA_NAME = '' # Name of preprocessed dataset directory (if not specified will be auto generated)
_C.DATA.TRAIN.ADDITIONAL_EXP_IDENTIFIER = '' # Additional identified that will be added to the end of EXP_DATA_NAME
_C.DATA.TRAIN.FILE_LIST_PATH = os.path.join(_C.DATA.TRAIN.CACHED_PATH, 'DataFileLists') # path to directory containing filelist
# Explanation of the File Lists: A csv containing filenames of input files used in the specific data split 
# (in this case a csv file containing the list of files used for training)
_C.DATA.TRAIN.DO_PREPROCESS = False
_C.DATA.TRAIN.DATA_FORMAT = 'NDCHW' # data format in model: N (batches), D (samples), C (channels), H (height), W (width) 
_C.DATA.TRAIN.BEGIN = 0.0 # start of used data in dataset
_C.DATA.TRAIN.END = 1.0 # end of used data in dataset 

# Train Data labels
_C.DATA.TRAIN.LABELS = CN()
_C.DATA.TRAIN.LABELS.LABEL_LIST = ['first_label'] # list of labels in the labels processed file
_C.DATA.TRAIN.LABELS.USED_LABELS = ['first_label'] # list of labels used out of all available labels

# Train Data preprocessing
_C.DATA.TRAIN.PREPROCESS = CN()
_C.DATA.TRAIN.PREPROCESS.DO_CHUNK = True # chunk the video 
_C.DATA.TRAIN.PREPROCESS.CHUNK_LENGTH = 180 # length to chunk the video (in frames)
_C.DATA.TRAIN.PREPROCESS.DYNAMIC_DETECTION = True # detection of face over the course of the video
_C.DATA.TRAIN.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY  = 180 # how often to re-detect the face
_C.DATA.TRAIN.PREPROCESS.CROP_FACE = True # crop around the face
_C.DATA.TRAIN.PREPROCESS.LARGE_FACE_BOX = True # TODO 
_C.DATA.TRAIN.PREPROCESS.LARGE_BOX_COEF = 1.5 # TODO
_C.DATA.TRAIN.PREPROCESS.W = 128 # Width to sample the frame to
_C.DATA.TRAIN.PREPROCESS.H = 128 # Height to sample the frame to
_C.DATA.TRAIN.PREPROCESS.DATA_TYPE = [''] # How to transform input frames (eg. 'Normalized', 'Standardized')
_C.DATA.TRAIN.PREPROCESS.LABEL_TYPE = '' # How to transform label (eg. 'Normalized', 'Standardized')


# -----------------------------------------------------------------------------
# Validation Data Settings
# -----------------------------------------------------------------------------
_C.DATA.VALID = CN()
_C.DATA.VALID.DATASET = ''
_C.DATA.VALID.FS = 0
_C.DATA.VALID.DATA_PATH = ''
_C.DATA.VALID.CACHED_PATH = 'PreprocessedData'
_C.DATA.VALID.EXP_DATA_NAME = ''
_C.DATA.VALID.ADDITIONAL_EXP_IDENTIFIER = ''
_C.DATA.VALID.FILE_LIST_PATH = os.path.join(_C.DATA.VALID.CACHED_PATH, 'DataFileLists')
_C.DATA.VALID.DO_PREPROCESS = False
_C.DATA.VALID.DATA_FORMAT = 'NDCHW'
_C.DATA.VALID.BEGIN = 0.0
_C.DATA.VALID.END = 1.0

# Valid Data labels
_C.DATA.VALID.LABELS = CN()
_C.DATA.VALID.LABELS.LABEL_LIST = ['first_label']
_C.DATA.VALID.LABELS.USED_LABELS = ['first_label']

# Valid Data preprocessing
_C.DATA.VALID.PREPROCESS = CN()
_C.DATA.VALID.PREPROCESS.DO_CHUNK = True
_C.DATA.VALID.PREPROCESS.CHUNK_LENGTH = 180
_C.DATA.VALID.PREPROCESS.DYNAMIC_DETECTION = True
_C.DATA.VALID.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY  = 180
_C.DATA.VALID.PREPROCESS.CROP_FACE = True
_C.DATA.VALID.PREPROCESS.LARGE_FACE_BOX = True
_C.DATA.VALID.PREPROCESS.LARGE_BOX_COEF = 1.5
_C.DATA.VALID.PREPROCESS.W = 128
_C.DATA.VALID.PREPROCESS.H = 128
_C.DATA.VALID.PREPROCESS.DATA_TYPE = ['']
_C.DATA.VALID.PREPROCESS.LABEL_TYPE = ''


# -----------------------------------------------------------------------------
# Test Data Settings
# -----------------------------------------------------------------------------\
_C.DATA.TEST = CN()
_C.DATA.TEST.DATASET = ''
_C.DATA.TEST.FS = 0
_C.DATA.TEST.DATA_PATH = ''
_C.DATA.TEST.CACHED_PATH = 'PreprocessedData'
_C.DATA.TEST.EXP_DATA_NAME = ''
_C.DATA.TEST.ADDITIONAL_EXP_IDENTIFIER = ''
_C.DATA.TEST.FILE_LIST_PATH = os.path.join(_C.DATA.TEST.CACHED_PATH, 'DataFileLists')
_C.DATA.TEST.DO_PREPROCESS = False
_C.DATA.TEST.DATA_FORMAT = 'NDCHW'
_C.DATA.TEST.BEGIN = 0.0
_C.DATA.TEST.END = 1.0

# Test Data labels
_C.DATA.TEST.LABELS = CN()
_C.DATA.TEST.LABELS.LABEL_LIST = ['first_label']
_C.DATA.TEST.LABELS.USED_LABELS = ['first_label']

# Test Data preprocessing
_C.DATA.TEST.PREPROCESS = CN()
_C.DATA.TEST.PREPROCESS.DO_CHUNK = True
_C.DATA.TEST.PREPROCESS.CHUNK_LENGTH = 180
_C.DATA.TEST.PREPROCESS.DYNAMIC_DETECTION = True
_C.DATA.TEST.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY  = 180
_C.DATA.TEST.PREPROCESS.CROP_FACE = True
_C.DATA.TEST.PREPROCESS.LARGE_FACE_BOX = True
_C.DATA.TEST.PREPROCESS.LARGE_BOX_COEF = 1.5
_C.DATA.TEST.PREPROCESS.W = 128
_C.DATA.TEST.PREPROCESS.H = 128
_C.DATA.TEST.PREPROCESS.DATA_TYPE = ['']
_C.DATA.TEST.PREPROCESS.LABEL_TYPE = ''



# -----------------------------------------------------------------------------
# Pretrain Data Settings
# -----------------------------------------------------------------------------
_C.DATA.PRETRAIN = CN()
_C.DATA.PRETRAIN.DATASET = ''
_C.DATA.PRETRAIN.FS = 0
_C.DATA.PRETRAIN.DATA_PATH = ''
_C.DATA.PRETRAIN.CACHED_PATH = 'PreprocessedData'
_C.DATA.PRETRAIN.EXP_DATA_NAME = ''
_C.DATA.PRETRAIN.ADDITIONAL_EXP_IDENTIFIER = ''
_C.DATA.PRETRAIN.FILE_LIST_PATH = os.path.join(_C.DATA.VALID.CACHED_PATH, 'DataFileLists')
_C.DATA.PRETRAIN.DO_PREPROCESS = False
_C.DATA.PRETRAIN.DATA_FORMAT = 'NDCHW'
_C.DATA.PRETRAIN.BEGIN = 0.0
_C.DATA.PRETRAIN.END = 1.0

# Pretrain Data labels
_C.DATA.PRETRAIN.LABELS = CN()
_C.DATA.PRETRAIN.LABELS.LABEL_LIST = ['first_label']
_C.DATA.PRETRAIN.LABELS.USED_LABELS = ['first_label']

# Pretrain Data preprocessing
_C.DATA.PRETRAIN.PREPROCESS = CN()
_C.DATA.PRETRAIN.PREPROCESS.DO_CHUNK = True
_C.DATA.PRETRAIN.PREPROCESS.CHUNK_LENGTH = 180
_C.DATA.PRETRAIN.PREPROCESS.DYNAMIC_DETECTION = True
_C.DATA.PRETRAIN.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY  = 180
_C.DATA.PRETRAIN.PREPROCESS.CROP_FACE = True
_C.DATA.PRETRAIN.PREPROCESS.LARGE_FACE_BOX = True
_C.DATA.PRETRAIN.PREPROCESS.LARGE_BOX_COEF = 1.5
_C.DATA.PRETRAIN.PREPROCESS.W = 128
_C.DATA.PRETRAIN.PREPROCESS.H = 128
_C.DATA.PRETRAIN.PREPROCESS.DATA_TYPE = ['']
_C.DATA.PRETRAIN.PREPROCESS.LABEL_TYPE = ''



# -----------------------------------------------------------------------------
# Signal Method Data Settings
# -----------------------------------------------------------------------------\
_C.DATA.SIGNAL = CN()
_C.DATA.SIGNAL.FS = 0
_C.DATA.SIGNAL.DATA_PATH = ''
_C.DATA.SIGNAL.EXP_DATA_NAME = ''
_C.DATA.SIGNAL.ADDITIONAL_EXP_IDENTIFIER = ''
_C.DATA.SIGNAL.CACHED_PATH = 'PreprocessedData'
_C.DATA.SIGNAL.FILE_LIST_PATH = os.path.join(_C.DATA.SIGNAL.CACHED_PATH, 'DataFileLists')
_C.DATA.SIGNAL.DATASET = ''
_C.DATA.SIGNAL.DO_PREPROCESS = False
_C.DATA.SIGNAL.DATA_FORMAT = 'NDCHW'
_C.DATA.SIGNAL.BEGIN = 0.0
_C.DATA.SIGNAL.END = 1.0

# Signal Data labels
_C.DATA.SIGNAL.LABELS = CN()
_C.DATA.SIGNAL.LABELS.LABEL_LIST = ['first_label']
_C.DATA.SIGNAL.LABELS.USED_LABELS = ['first_label']

# Signal Data preprocessing
_C.DATA.SIGNAL.PREPROCESS = CN()
_C.DATA.SIGNAL.PREPROCESS.DO_CHUNK = True
_C.DATA.SIGNAL.PREPROCESS.CHUNK_LENGTH = 180
_C.DATA.SIGNAL.PREPROCESS.DYNAMIC_DETECTION = True
_C.DATA.SIGNAL.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY  = 180
_C.DATA.SIGNAL.PREPROCESS.CROP_FACE = True
_C.DATA.SIGNAL.PREPROCESS.LARGE_FACE_BOX = True
_C.DATA.SIGNAL.PREPROCESS.LARGE_BOX_COEF = 1.5
_C.DATA.SIGNAL.PREPROCESS.W = 128
_C.DATA.SIGNAL.PREPROCESS.H = 128
_C.DATA.SIGNAL.PREPROCESS.DATA_TYPE = ['']
_C.DATA.SIGNAL.PREPROCESS.LABEL_TYPE = ''

### -----------------------------------------------------------------------------
### -----------------------------------------------------------------------------
# DATA SETTINGS (END)
### -----------------------------------------------------------------------------
### -----------------------------------------------------------------------------



### -----------------------------------------------------------------------------
### -----------------------------------------------------------------------------
# MODEL ARCHITECTURE/TRAIN/TEST SETTINGS (START)
### -----------------------------------------------------------------------------
### -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Model/Training/Inference Settings
# -----------------------------------------------------------------------------
_C.MODEL_SPECS = CN()

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL_SPECS.MODEL = CN()
_C.MODEL_SPECS.MODEL.NAME = '' # Model name
_C.MODEL_SPECS.MODEL.RESUME = '' # Checkpoint to resume, could be overwritten by command line argument
_C.MODEL_SPECS.MODEL.MODEL_DIR = 'PreTrainedModels' # Pretrained model base directory

# Model Settings for PhysNet
_C.MODEL_SPECS.MODEL.PHYSNET = CN()
_C.MODEL_SPECS.MODEL.PHYSNET.FRAME_NUM = 64
# Model Settings for TS-CAN
_C.MODEL_SPECS.MODEL.TSCAN = CN()
_C.MODEL_SPECS.MODEL.TSCAN.FRAME_DEPTH = 10
# Model Settings for EfficientPhys
_C.MODEL_SPECS.MODEL.EFFICIENTPHYS = CN()
_C.MODEL_SPECS.MODEL.EFFICIENTPHYS.FRAME_DEPTH = 10

# -----------------------------------------------------------------------------
# Model Training Settings
# -----------------------------------------------------------------------------
_C.MODEL_SPECS.TRAIN = CN()
_C.MODEL_SPECS.TRAIN.EPOCHS = 10 # number of epochs
_C.MODEL_SPECS.TRAIN.BATCH_SIZE = 4 # batch size
_C.MODEL_SPECS.TRAIN.LR = 1e-3 # learning rate

# Data Augmentation For Training Batches
_C.MODEL_SPECS.TRAIN.DATA_AUG = False

# Learning Schedulers / Tools
_C.MODEL_SPECS.TRAIN.GRAD_SURGERY = False
_C.MODEL_SPECS.TRAIN.OCLR_SCHEDULER = False

# Drop Rate #TODO This is unused and not passed to the models
_C.MODEL_SPECS.TRAIN.DROP_RATE = 0.0 # Dropout rate

# Optimizer # TODO: These Dont Do Anything (Optimized hardcoded to Adam in basetrainer)
_C.MODEL_SPECS.TRAIN.OPTIMIZER = CN()
_C.MODEL_SPECS.TRAIN.OPTIMIZER.EPS = 1e-4 # Optimizer Epsilon
_C.MODEL_SPECS.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999) # Optimizer Betas
_C.MODEL_SPECS.TRAIN.OPTIMIZER.MOMENTUM = 0.9 # SGD momentum
_C.MODEL_SPECS.TRAIN.MODEL_FILE_NAME = ''

# Loss 
_C.MODEL_SPECS.TRAIN.LOSS_NAME = 'MSE'

# FINE Tuning model settings: 
_C.MODEL_SPECS.TRAIN.FINE_TUNE_MODEL = False
_C.MODEL_SPECS.TRAIN.FREEZE_LAYERS = False
_C.MODEL_SPECS.TRAIN.RESET_DENSE_LAYERS = False

# -----------------------------------------------------------------------------
# Model Validation Settings
# -----------------------------------------------------------------------------
_C.MODEL_SPECS.VALID = CN()
_C.MODEL_SPECS.VALID.RUN_VALIDATION = False

# -----------------------------------------------------------------------------
# Model Test/Inference Settings
# -----------------------------------------------------------------------------
_C.MODEL_SPECS.TEST = CN()
_C.MODEL_SPECS.TEST.BVP_METRICS = []
_C.MODEL_SPECS.TEST.RESP_METRICS = []
_C.MODEL_SPECS.TEST.AU_METRICS = []
_C.MODEL_SPECS.TEST.EVALUATION_METHOD = 'FFT' # either 'FFT' or 'peak detection'
_C.MODEL_SPECS.TEST.BATCH_SIZE = _C.MODEL_SPECS.TRAIN.BATCH_SIZE
_C.MODEL_SPECS.TEST.MODEL_TO_USE = 'last_epoch' # can be 'best_epoch' or 'last_epoch'
_C.MODEL_SPECS.TEST.MODEL_PATH = ''

# -----------------------------------------------------------------------------
# Model Pretrain Settings
# -----------------------------------------------------------------------------
_C.MODEL_SPECS.PRETRAIN = CN()
_C.MODEL_SPECS.PRETRAIN.BATCH_SIZE = 4

### -----------------------------------------------------------------------------
### -----------------------------------------------------------------------------
# MODEL ARCHITECTURE/TRAIN/TEST SETTINGS (END)
### -----------------------------------------------------------------------------
### -----------------------------------------------------------------------------



### -----------------------------------------------------------------------------
### -----------------------------------------------------------------------------
# SIGNAL PROCESSING ANALYSIS SETTINGS (START)
### -----------------------------------------------------------------------------
### -----------------------------------------------------------------------------

_C.SIGNAL_SPECS = CN()
_C.SIGNAL_SPECS.METHOD = []
_C.SIGNAL_SPECS.EVALUATION_METHOD = 'FFT' # either 'FFT' or 'peak detection'
_C.SIGNAL_SPECS.METRICS = []

### -----------------------------------------------------------------------------
### -----------------------------------------------------------------------------
# SIGNAL PROCESSING ANALYSIS SETTINGS (END)
### -----------------------------------------------------------------------------
### -----------------------------------------------------------------------------


### -----------------------------------------------------------------------------
### -----------------------------------------------------------------------------
# OTHER SETTINGS (START)
### -----------------------------------------------------------------------------
### -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Device settings
# -----------------------------------------------------------------------------
_C.DEVICE = "cuda:0"
_C.NUM_OF_GPU_TRAIN = 1

# -----------------------------------------------------------------------------
# Log settings
# -----------------------------------------------------------------------------
_C.LOG = CN()
_C.LOG.PATH = "runs/exp"


# TODO Move this somewhere in the modeling settings???
_C.SAVE_DATA = CN()
_C.SAVE_DATA.SAVE_DATA = False
_C.SAVE_DATA.SAVE_TEST = False
_C.SAVE_DATA.SAVE_TRAIN = False
_C.SAVE_DATA.SAVE_METRICS = False
_C.SAVE_DATA.PATH = ""

### -----------------------------------------------------------------------------
### -----------------------------------------------------------------------------
# OTHER SETTINGS (END)
### -----------------------------------------------------------------------------
### -----------------------------------------------------------------------------




def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> Merging a config file from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()



def update_config(config, args):

    # save default file list path to check against later (below in this script)
    # currently default is CACHED_PATH/DataFileLists
    default_TRAIN_FILE_LIST_PATH = config.DATA.TRAIN.FILE_LIST_PATH
    default_VALID_FILE_LIST_PATH = config.DATA.VALID.FILE_LIST_PATH
    default_TEST_FILE_LIST_PATH = config.DATA.TEST.FILE_LIST_PATH
    default_PRETRAIN_FILE_LIST_PATH = config.DATA.TEST.FILE_LIST_PATH
    default_SIGNAL_FILE_LIST_PATH = config.DATA.SIGNAL.FILE_LIST_PATH


    # update flag from config file
    _update_config_from_file(config, args.config_file)
    config.defrost()

    # TODO: weird issue here when batch size is small for test loaders...
    # config.MODEL_SPECS.TEST.BATCH_SIZE = config.MODEL_SPECS.TRAIN.BATCH_SIZE


    # UPDATE PREPROCESS METADATA TO REPRESENT GLOBAL PREPROCESS DATA (IF GLOBAL_PREPROCESS_SETTINGS == True)
    if config.DATA.PREPROCESS.GLOBAL_PREPROCESS_SETTINGS:
        # Train Preprocess Data
        config.DATA.TRAIN.PREPROCESS.DO_CHUNK = config.DATA.PREPROCESS.DO_CHUNK
        config.DATA.TRAIN.PREPROCESS.CHUNK_LENGTH = config.DATA.PREPROCESS.CHUNK_LENGTH
        config.DATA.TRAIN.PREPROCESS.DYNAMIC_DETECTION = config.DATA.PREPROCESS.DYNAMIC_DETECTION
        config.DATA.TRAIN.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY  = config.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY
        config.DATA.TRAIN.PREPROCESS.CROP_FACE = config.DATA.PREPROCESS.CROP_FACE
        config.DATA.TRAIN.PREPROCESS.LARGE_FACE_BOX = config.DATA.PREPROCESS.LARGE_FACE_BOX
        config.DATA.TRAIN.PREPROCESS.LARGE_BOX_COEF = config.DATA.PREPROCESS.LARGE_BOX_COEF
        config.DATA.TRAIN.PREPROCESS.W = config.DATA.PREPROCESS.W
        config.DATA.TRAIN.PREPROCESS.H = config.DATA.PREPROCESS.H
        config.DATA.TRAIN.PREPROCESS.DATA_TYPE = config.DATA.PREPROCESS.DATA_TYPE
        config.DATA.TRAIN.PREPROCESS.LABEL_TYPE = config.DATA.PREPROCESS.LABEL_TYPE

        # Valid Preprocess Data
        config.DATA.VALID.PREPROCESS.DO_CHUNK = config.DATA.PREPROCESS.DO_CHUNK
        config.DATA.VALID.PREPROCESS.CHUNK_LENGTH = config.DATA.PREPROCESS.CHUNK_LENGTH
        config.DATA.VALID.PREPROCESS.DYNAMIC_DETECTION = config.DATA.PREPROCESS.DYNAMIC_DETECTION
        config.DATA.VALID.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY = config.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY
        config.DATA.VALID.PREPROCESS.CROP_FACE = config.DATA.PREPROCESS.CROP_FACE
        config.DATA.VALID.PREPROCESS.LARGE_FACE_BOX = config.DATA.PREPROCESS.LARGE_FACE_BOX
        config.DATA.VALID.PREPROCESS.LARGE_BOX_COEF = config.DATA.PREPROCESS.LARGE_BOX_COEF
        config.DATA.VALID.PREPROCESS.W = config.DATA.PREPROCESS.W
        config.DATA.VALID.PREPROCESS.H = config.DATA.PREPROCESS.H
        config.DATA.VALID.PREPROCESS.DATA_TYPE = config.DATA.PREPROCESS.DATA_TYPE
        config.DATA.VALID.PREPROCESS.LABEL_TYPE = config.DATA.PREPROCESS.LABEL_TYPE

        # Test Preprocess Data
        config.DATA.TEST.PREPROCESS.DO_CHUNK = config.DATA.PREPROCESS.DO_CHUNK
        config.DATA.TEST.PREPROCESS.CHUNK_LENGTH = config.DATA.PREPROCESS.CHUNK_LENGTH
        config.DATA.TEST.PREPROCESS.DYNAMIC_DETECTION = config.DATA.PREPROCESS.DYNAMIC_DETECTION
        config.DATA.TEST.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY = config.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY
        config.DATA.TEST.PREPROCESS.CROP_FACE = config.DATA.PREPROCESS.CROP_FACE
        config.DATA.TEST.PREPROCESS.LARGE_FACE_BOX = config.DATA.PREPROCESS.LARGE_FACE_BOX
        config.DATA.TEST.PREPROCESS.LARGE_BOX_COEF = config.DATA.PREPROCESS.LARGE_BOX_COEF
        config.DATA.TEST.PREPROCESS.W = config.DATA.PREPROCESS.W
        config.DATA.TEST.PREPROCESS.H = config.DATA.PREPROCESS.H
        config.DATA.TEST.PREPROCESS.DATA_TYPE = config.DATA.PREPROCESS.DATA_TYPE
        config.DATA.TEST.PREPROCESS.LABEL_TYPE = config.DATA.PREPROCESS.LABEL_TYPE

        # Pretrain Preprocess Data
        config.DATA.PRETRAIN.PREPROCESS.DO_CHUNK = config.DATA.PREPROCESS.DO_CHUNK
        config.DATA.PRETRAIN.PREPROCESS.CHUNK_LENGTH = config.DATA.PREPROCESS.CHUNK_LENGTH
        config.DATA.PRETRAIN.PREPROCESS.DYNAMIC_DETECTION = config.DATA.PREPROCESS.DYNAMIC_DETECTION
        config.DATA.PRETRAIN.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY = config.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY
        config.DATA.PRETRAIN.PREPROCESS.CROP_FACE = config.DATA.PREPROCESS.CROP_FACE
        config.DATA.PRETRAIN.PREPROCESS.LARGE_FACE_BOX = config.DATA.PREPROCESS.LARGE_FACE_BOX
        config.DATA.PRETRAIN.PREPROCESS.LARGE_BOX_COEF = config.DATA.PREPROCESS.LARGE_BOX_COEF
        config.DATA.PRETRAIN.PREPROCESS.W = config.DATA.PREPROCESS.W
        config.DATA.PRETRAIN.PREPROCESS.H = config.DATA.PREPROCESS.H
        config.DATA.PRETRAIN.PREPROCESS.DATA_TYPE = config.DATA.PREPROCESS.DATA_TYPE
        config.DATA.PRETRAIN.PREPROCESS.LABEL_TYPE = config.DATA.PREPROCESS.LABEL_TYPE

        # Signal Preprocess Data
        config.DATA.SIGNAL.PREPROCESS.DO_CHUNK = config.DATA.PREPROCESS.DO_CHUNK
        config.DATA.SIGNAL.PREPROCESS.CHUNK_LENGTH = config.DATA.PREPROCESS.CHUNK_LENGTH
        config.DATA.SIGNAL.PREPROCESS.DYNAMIC_DETECTION = config.DATA.PREPROCESS.DYNAMIC_DETECTION
        config.DATA.SIGNAL.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY = config.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY
        config.DATA.SIGNAL.PREPROCESS.CROP_FACE = config.DATA.PREPROCESS.CROP_FACE
        config.DATA.SIGNAL.PREPROCESS.LARGE_FACE_BOX = config.DATA.PREPROCESS.LARGE_FACE_BOX
        config.DATA.SIGNAL.PREPROCESS.LARGE_BOX_COEF = config.DATA.PREPROCESS.LARGE_BOX_COEF
        config.DATA.SIGNAL.PREPROCESS.W = config.DATA.PREPROCESS.W
        config.DATA.SIGNAL.PREPROCESS.H = config.DATA.PREPROCESS.H
        config.DATA.SIGNAL.PREPROCESS.DATA_TYPE = config.DATA.PREPROCESS.DATA_TYPE
        config.DATA.SIGNAL.PREPROCESS.LABEL_TYPE = config.DATA.PREPROCESS.LABEL_TYPE


    # UPDATE LABELS USED FOR EACH SPLIT (TRAIN/VAL/TEST)
    if config.DATA.LABELS.GLOBAL_LABELS_SETTINGS:
        # Train Labels
        config.DATA.TRAIN.LABELS.LABEL_LIST = config.DATA.LABELS.LABEL_LIST
        config.DATA.TRAIN.LABELS.USED_LABELS = config.DATA.LABELS.USED_LABELS

        # Valid Labels
        config.DATA.VALID.LABELS.LABEL_LIST = config.DATA.LABELS.LABEL_LIST
        config.DATA.VALID.LABELS.USED_LABELS = config.DATA.LABELS.USED_LABELS

        # Test Labels
        config.DATA.TEST.LABELS.LABEL_LIST = config.DATA.LABELS.LABEL_LIST
        config.DATA.TEST.LABELS.USED_LABELS = config.DATA.LABELS.USED_LABELS

        # Pretrain Labels
        config.DATA.PRETRAIN.LABELS.LABEL_LIST = config.DATA.LABELS.LABEL_LIST
        config.DATA.PRETRAIN.LABELS.USED_LABELS = config.DATA.LABELS.USED_LABELS

        # Signal Labels
        config.DATA.SIGNAL.LABELS.LABEL_LIST = config.DATA.LABELS.LABEL_LIST
        config.DATA.SIGNAL.LABELS.USED_LABELS = config.DATA.LABELS.USED_LABELS



    # UPDATE TRAIN PATHS
    if config.DATA.TRAIN.FILE_LIST_PATH == default_TRAIN_FILE_LIST_PATH: # Update File List Path
        config.DATA.TRAIN.FILE_LIST_PATH = os.path.join(config.DATA.TRAIN.CACHED_PATH, 'DataFileLists')

    if config.DATA.TRAIN.EXP_DATA_NAME == '': # Update the dataset name (experiment data name) if not provided
        config.DATA.TRAIN.EXP_DATA_NAME = "_".join([config.DATA.TRAIN.DATASET, "SizeW{0}".format(
            str(config.DATA.TRAIN.PREPROCESS.W)), "SizeH{0}".format(str(config.DATA.TRAIN.PREPROCESS.W)), "ClipLength{0}".format(
            str(config.DATA.TRAIN.PREPROCESS.CHUNK_LENGTH)), "DataType{0}".format("_".join(config.DATA.TRAIN.PREPROCESS.DATA_TYPE)),
                                      "LabelType{0}".format(config.DATA.TRAIN.PREPROCESS.LABEL_TYPE),
                                      "Large_box{0}".format(config.DATA.TRAIN.PREPROCESS.LARGE_FACE_BOX),
                                      "Large_size{0}".format(config.DATA.TRAIN.PREPROCESS.LARGE_BOX_COEF),
                                      "Dyamic_Det{0}".format(config.DATA.TRAIN.PREPROCESS.DYNAMIC_DETECTION),
                                      "det_len{0}".format(config.DATA.TRAIN.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY)
                                              ])
    if config.DATA.TRAIN.ADDITIONAL_EXP_IDENTIFIER: # add additional exp identifier to experiment dataset name
        config.DATA.TRAIN.EXP_DATA_NAME = config.DATA.TRAIN.EXP_DATA_NAME + '_' + config.DATA.TRAIN.ADDITIONAL_EXP_IDENTIFIER
    config.DATA.TRAIN.CACHED_PATH = os.path.join(config.DATA.TRAIN.CACHED_PATH, config.DATA.TRAIN.EXP_DATA_NAME)

    name, ext = os.path.splitext(config.DATA.TRAIN.FILE_LIST_PATH)
    if not ext: # no file extension (file list directory specified but not csv file)
        config.DATA.TRAIN.FILE_LIST_PATH = os.path.join(config.DATA.TRAIN.FILE_LIST_PATH, \
                                                        config.DATA.TRAIN.EXP_DATA_NAME + '_' + \
                                                        str(config.DATA.TRAIN.BEGIN) + '_' + \
                                                        str(config.DATA.TRAIN.END) + '.csv')
    elif ext != '.csv':
        raise ValueError(self.name, 'FILE_LIST_PATH must either be a directory path or a .csv file name')
    
    if ext == '.csv' and config.DATA.TRAIN.DO_PREPROCESS:
        raise ValueError(self.name, 'User specified FILE_LIST_PATH .csv file already exists. \
                         Please turn DO_PREPROCESS to False or delete existing FILE_LIST_PATH .csv file.')



    # UPDATE VALID PATHS
    if config.DATA.VALID.FILE_LIST_PATH == default_VALID_FILE_LIST_PATH: # Update File List Path
        config.DATA.VALID.FILE_LIST_PATH = os.path.join(config.DATA.VALID.CACHED_PATH, 'DataFileLists')

    if config.DATA.VALID.EXP_DATA_NAME == '': # Update the dataset name (experiment data name) if not provided
        config.DATA.VALID.EXP_DATA_NAME = "_".join([config.DATA.VALID.DATASET, "SizeW{0}".format(
            str(config.DATA.VALID.PREPROCESS.W)), "SizeH{0}".format(str(config.DATA.VALID.PREPROCESS.W)), "ClipLength{0}".format(
            str(config.DATA.VALID.PREPROCESS.CHUNK_LENGTH)), "DataType{0}".format("_".join(config.DATA.VALID.PREPROCESS.DATA_TYPE)),
                                      "LabelType{0}".format(config.DATA.VALID.PREPROCESS.LABEL_TYPE),
                                      "Large_box{0}".format(config.DATA.VALID.PREPROCESS.LARGE_FACE_BOX),
                                      "Large_size{0}".format(config.DATA.VALID.PREPROCESS.LARGE_BOX_COEF),
                                      "Dyamic_Det{0}".format(config.DATA.VALID.PREPROCESS.DYNAMIC_DETECTION),
                                      "det_len{0}".format(config.DATA.VALID.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY)
                                              ])
    if config.DATA.VALID.ADDITIONAL_EXP_IDENTIFIER: # add additional exp identifier to experiment dataset name
        config.DATA.VALID.EXP_DATA_NAME = config.DATA.VALID.EXP_DATA_NAME + '_' + config.DATA.VALID.ADDITIONAL_EXP_IDENTIFIER
    config.DATA.VALID.CACHED_PATH = os.path.join(config.DATA.VALID.CACHED_PATH, config.DATA.VALID.EXP_DATA_NAME)

    name, ext = os.path.splitext(config.DATA.VALID.FILE_LIST_PATH)
    if not ext: # no file extension (file list directory specified but not csv file)
        config.DATA.VALID.FILE_LIST_PATH = os.path.join(config.DATA.VALID.FILE_LIST_PATH, \
                                                        config.DATA.VALID.EXP_DATA_NAME + '_' + \
                                                        str(config.DATA.VALID.BEGIN) + '_' + \
                                                        str(config.DATA.VALID.END) + '.csv')
    elif ext != '.csv':
        raise ValueError(self.name, 'FILE_LIST_PATH must either be a directory path or a .csv file name')

    if ext == '.csv' and config.DATA.VALID.DO_PREPROCESS:
        raise ValueError(self.name, 'User specified FILE_LIST_PATH .csv file already exists. \
                         Please turn DO_PREPROCESS to False or delete existing FILE_LIST_PATH .csv file.')



    # UPDATE TEST PATHS
    if config.DATA.TEST.FILE_LIST_PATH == default_TEST_FILE_LIST_PATH: # Update File List Path
        config.DATA.TEST.FILE_LIST_PATH = os.path.join(config.DATA.TEST.CACHED_PATH, 'DataFileLists')

    if config.DATA.TEST.EXP_DATA_NAME == '': # Update the dataset name (experiment data name) if not provided
        config.DATA.TEST.EXP_DATA_NAME = "_".join([config.DATA.TEST.DATASET, "SizeW{0}".format(
            str(config.DATA.TEST.PREPROCESS.W)), "SizeH{0}".format(str(config.DATA.TEST.PREPROCESS.W)), "ClipLength{0}".format(
            str(config.DATA.TEST.PREPROCESS.CHUNK_LENGTH)), "DataType{0}".format("_".join(config.DATA.TEST.PREPROCESS.DATA_TYPE)),
                                      "LabelType{0}".format(config.DATA.TEST.PREPROCESS.LABEL_TYPE),
                                      "Large_box{0}".format(config.DATA.TEST.PREPROCESS.LARGE_FACE_BOX),
                                      "Large_size{0}".format(config.DATA.TEST.PREPROCESS.LARGE_BOX_COEF),
                                      "Dyamic_Det{0}".format(config.DATA.TEST.PREPROCESS.DYNAMIC_DETECTION),
                                      "det_len{0}".format(config.DATA.TEST.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY)
                                              ])
    if config.DATA.TEST.ADDITIONAL_EXP_IDENTIFIER: # add additional exp identifier to experiment dataset name
        config.DATA.TEST.EXP_DATA_NAME = config.DATA.TEST.EXP_DATA_NAME + '_' + config.DATA.TEST.ADDITIONAL_EXP_IDENTIFIER
    config.DATA.TEST.CACHED_PATH = os.path.join(config.DATA.TEST.CACHED_PATH, config.DATA.TEST.EXP_DATA_NAME)

    name, ext = os.path.splitext(config.DATA.TEST.FILE_LIST_PATH)
    if not ext: # no file extension (file list directory specified but not csv file)
        config.DATA.TEST.FILE_LIST_PATH = os.path.join(config.DATA.TEST.FILE_LIST_PATH, \
                                                       config.DATA.TEST.EXP_DATA_NAME + '_' + \
                                                       str(config.DATA.TEST.BEGIN) + '_' + \
                                                       str(config.DATA.TEST.END) + '.csv')
    elif ext != '.csv':
        raise ValueError(self.name, 'FILE_LIST_PATH must either be a directory path or a .csv file name')

    if ext == '.csv' and config.DATA.TEST.DO_PREPROCESS:
        raise ValueError(self.name, 'User specified FILE_LIST_PATH .csv file already exists. \
                         Please turn DO_PREPROCESS to False or delete existing FILE_LIST_PATH .csv file.')



    # UPDATE PRETRAIN PATHS
    if config.DATA.PRETRAIN.FILE_LIST_PATH == default_PRETRAIN_FILE_LIST_PATH: # Update File List Path
        config.DATA.PRETRAIN.FILE_LIST_PATH = os.path.join(config.DATA.PRETRAIN.CACHED_PATH, 'DataFileLists')

    if config.DATA.PRETRAIN.EXP_DATA_NAME == '': # Update the dataset name (experiment data name) if not provided
        config.DATA.PRETRAIN.EXP_DATA_NAME = "_".join([config.DATA.PRETRAIN.DATASET, "SizeW{0}".format(
            str(config.DATA.PRETRAIN.PREPROCESS.W)), "SizeH{0}".format(str(config.DATA.PRETRAIN.PREPROCESS.W)), "ClipLength{0}".format(
            str(config.DATA.PRETRAIN.PREPROCESS.CHUNK_LENGTH)), "DataType{0}".format("_".join(config.DATA.PRETRAIN.PREPROCESS.DATA_TYPE)),
                                      "LabelType{0}".format(config.DATA.PRETRAIN.PREPROCESS.LABEL_TYPE),
                                      "Large_box{0}".format(config.DATA.PRETRAIN.PREPROCESS.LARGE_FACE_BOX),
                                      "Large_size{0}".format(config.DATA.PRETRAIN.PREPROCESS.LARGE_BOX_COEF),
                                      "Dyamic_Det{0}".format(config.DATA.PRETRAIN.PREPROCESS.DYNAMIC_DETECTION),
                                      "det_len{0}".format(config.DATA.PRETRAIN.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY)
                                              ])
    if config.DATA.PRETRAIN.ADDITIONAL_EXP_IDENTIFIER: # add additional exp identifier to experiment dataset name
        config.DATA.PRETRAIN.EXP_DATA_NAME = config.DATA.PRETRAIN.EXP_DATA_NAME + '_' + config.DATA.PRETRAIN.ADDITIONAL_EXP_IDENTIFIER
    config.DATA.PRETRAIN.CACHED_PATH = os.path.join(config.DATA.PRETRAIN.CACHED_PATH, config.DATA.PRETRAIN.EXP_DATA_NAME)

    name, ext = os.path.splitext(config.DATA.PRETRAIN.FILE_LIST_PATH)
    if not ext: # no file extension (file list directory specified but not csv file)
        config.DATA.PRETRAIN.FILE_LIST_PATH = os.path.join(config.DATA.PRETRAIN.FILE_LIST_PATH, \
                                                       config.DATA.PRETRAIN.EXP_DATA_NAME + '_' + \
                                                       str(config.DATA.PRETRAIN.BEGIN) + '_' + \
                                                       str(config.DATA.PRETRAIN.END) + '.csv')
    elif ext != '.csv':
        raise ValueError(self.name, 'FILE_LIST_PATH must either be a directory path or a .csv file name')

    if ext == '.csv' and config.DATA.PRETRAIN.DO_PREPROCESS:
        raise ValueError(self.name, 'User specified FILE_LIST_PATH .csv file already exists. \
                         Please turn DO_PREPROCESS to False or delete existing FILE_LIST_PATH .csv file.')
    


    # UPDATE SIGNAL PATHS
    if config.DATA.SIGNAL.FILE_LIST_PATH == default_SIGNAL_FILE_LIST_PATH: # Update File List Path
        config.DATA.SIGNAL.FILE_LIST_PATH = os.path.join(config.DATA.SIGNAL.CACHED_PATH, 'DataFileLists')

    if config.DATA.SIGNAL.EXP_DATA_NAME == '': # Update the dataset name (experiment data name) if not provided
        config.DATA.SIGNAL.EXP_DATA_NAME = "_".join([config.DATA.SIGNAL.DATASET, "SizeW{0}".format(
            str(config.DATA.SIGNAL.PREPROCESS.W)), "SizeH{0}".format(str(config.DATA.SIGNAL.PREPROCESS.W)), "ClipLength{0}".format(
            str(config.DATA.SIGNAL.PREPROCESS.CHUNK_LENGTH)), "DataType{0}".format("_".join(config.DATA.SIGNAL.PREPROCESS.DATA_TYPE)),
                                      "LabelType{0}".format(config.DATA.SIGNAL.PREPROCESS.LABEL_TYPE),
                                      "Large_box{0}".format(config.DATA.SIGNAL.PREPROCESS.LARGE_FACE_BOX),
                                      "Large_size{0}".format(config.DATA.SIGNAL.PREPROCESS.LARGE_BOX_COEF),
                                      "Dyamic_Det{0}".format(config.DATA.SIGNAL.PREPROCESS.DYNAMIC_DETECTION),
                                        "det_len{0}".format(config.DATA.SIGNAL.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY),
                                        "signal"
                                              ])
    if config.DATA.SIGNAL.ADDITIONAL_EXP_IDENTIFIER: # add additional exp identifier to experiment dataset name
        config.DATA.SIGNAL.EXP_DATA_NAME = config.DATA.SIGNAL.EXP_DATA_NAME + '_' + config.DATA.SIGNAL.ADDITIONAL_EXP_IDENTIFIER
    config.DATA.SIGNAL.CACHED_PATH = os.path.join(config.DATA.SIGNAL.CACHED_PATH, config.DATA.SIGNAL.EXP_DATA_NAME)

    name, ext = os.path.splitext(config.DATA.SIGNAL.FILE_LIST_PATH)
    if not ext: # no file extension (file list directory specified but not csv file)
        config.DATA.SIGNAL.FILE_LIST_PATH = os.path.join(config.DATA.SIGNAL.FILE_LIST_PATH, \
                                                         config.DATA.SIGNAL.EXP_DATA_NAME + '_' + \
                                                         str(config.DATA.SIGNAL.BEGIN) + '_' + \
                                                         str(config.DATA.SIGNAL.END) + '.csv')
    elif ext != '.csv':
        raise ValueError(self.name, 'FILE_LIST_PATH must either be a directory path or a .csv file name')

    if ext == '.csv' and config.DATA.SIGNAL.DO_PREPROCESS:
        raise ValueError(self.name, 'User specified FILE_LIST_PATH .csv file already exists. \
                         Please turn DO_PREPROCESS to False or delete existing FILE_LIST_PATH .csv file.')
    

    #UPDATE LOG PATHS #TODO check what this is
    config.LOG.PATH = os.path.join(config.LOG.PATH, config.DATA.TRAIN.EXP_DATA_NAME)


    # UPDATE MODEL PATHS #TODO check what this is
    config.MODEL_SPECS.MODEL.MODEL_DIR = os.path.join(config.MODEL_SPECS.MODEL.MODEL_DIR, config.DATA.TRAIN.EXP_DATA_NAME)
    config.freeze()
    return



def get_config(args):
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config


