# BigSmall

The pipelines and infrastructure for this work is extended from rPPG-Toolbox:
Liu, Xin, Xiaoyu Zhang, Girish Narayanswamy, Yuzhe Zhang, Yuntao Wang, Shwetak Patel, and Daniel McDuff. 
"Deep physiological sensing toolbox." arXiv preprint arXiv:2210.00716 (2022).



# Setup

STEP0: ensure you first cd into the `./code` directory to run all the following code steps.

STEP1: `bash setup.sh` 
STEP2: `conda activate bigsmall_multitask` 
STEP3: `pip install -r requirements.txt` 



# BigSmall Datasets

BigSmall datasets are stored as input/label file pairs. The input file is of type `.pickle` and contains Big high-resolution raw-standardized frames under key `1`, and Small low-resolution normalized-difference frames under key `2`. These frame arrays are of shape NxHxWxC (number of frames, height, width, color channels). The default Big size is 144x144 and the default Small size is 9x9.

These input files are paired with a label file of type `.npy`. This array is of shape NxY, where Y is the number of signals labeled. For BP4D+ this value is 49. 



# Preprocessing BigSmall Datasets

We provide preprocessing code for the BP4D+ dataset. This code can be adapated for other datasets (PURE/UBFC/DISFA) though this code is not provided. To adapt this code for other datasets please refer to the preprocessing script `TODO.py` and re-write the functions marked with `# Change For Other Datasets`.



# Preprocessing BP4D+

Note, that due to the size of BP4D+ (~6TB), the data was pre-preprocessed to better run on our server. The preprocessing entailes: 

0. Download the BP4D+ raw data by asking the paper authors: 
https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html

1. Pushing all frame and label information, per-video trial, into a `.mat` file format (this can be read as a dictionary like structure in python). 

2. The frames of `X` are square center cropped

3. Ensuring that all labels matched the length of the video data (this ay require some downsampling of certain label signals).

4. Further crop the `X` and label signals to only include the AU Subset (most expressive portion) of the BP4D+ dataset, that contains the AU labels. This is the portion of the dataset used to train the BigSmall model. 

5. The keys for values pushed to the `.mat` file are as follows: 

video frames: `X`, blood pressure waveform signal: `bp_wave`, heart rate beats per min: `HR_bpm` , systolic blood pressure: `systolic_bp`, diastolic blood pressure: `diastolic_bp`, sys/dia mean blood pressure: `mean_bp`, respiration waveform signal: `resp_wave`, resp rate beats per min: `resp_bpm`, electrodermal activity: `eda`,action units (int relays intestity encoded AUs): `AU01`, `AU02`, `AU04`, `AU05`, `AU06`, `AU06int`, `AU07`, `AU09`, `AU10`, `AU10int`, `AU11`, `AU12`, `AU12int`, `AU13`, `AU14`, `AU14int`, `AU15`, `AU16`, `AU17`, `AU17int`, `AU18`, `AU19`, `AU20`, `AU22`, `AU23`, `AU24`, `AU27`, `AU28`, `AU29`, `AU30`, `AU31`, `AU32`, `AU33`, `AU34`, `AU35`, `AU36`, `AU37`, `AU38`, `AU39`

6. As BP Wave labels are not conducive to PPG training. Thus derive a set of PPG psuedo labels using POS, and save them in 2 additional keys in the `.mat` files. This is done by configuring the data path to the raw `.mat` data directory and running: `python ./dataset_builder_utils/bp4d_bigsmall/psuedoPPG_bp4d.py`. This will add the `pos_bvp`,`pos_env_norm_bvp` keys to the `.mat` data files.

7. Run `python python ./dataset_builder_utils/bp4d_bigsmall/preprocess_bp4d_bigsmall.py` to build the BP4D+ BigSmall dataset from the .mat files. 

8. Build the file lists corresponding to the data splits of the processed data. First set data and file paths in both `./dataset_builder_utils/bp4d_bigsmall/bp4d_subject_datasplit/bp4d_datasplit_test.py` and 
`./dataset_builder_utils/bp4d_bigsmall/bp4d_subject_datasplit/bp4d_datasplit_train.py`. 
And then in order run:
a. `python ./dataset_builder_utils/bp4d_bigsmall/bp4d_subject_datasplit/bp4d_datasplit_test.py`
b. `python ./dataset_builder_utils/bp4d_bigsmall/bp4d_subject_datasplit/bp4d_datasplit_train.py`



# Training/Testing on BP4D+ BigSmall

0. Pretrained models can be found in `./pretrained_models`

1. Modify a config file simillar to  `./configs/BP4D_BIGSMALL_MULTITASK_SPLIT1.yaml` 
This can be modified to either train and test, or test. Preprocessing is not supported. 

2. `python bigsmall_main.py --config_file ./configs/BP4D_BIGSMALL_MULTITASK_SPLIT1.yaml` 


# Yaml File Setting

BigSmall uses yaml file to control all parameters for training and evaluation. 
These have been adapted from the rPPG-Toolbox system. 
You can modify the existing yaml files to meet your own training and testing requirements.


Here are some explanation of parameters (as described by the rPPG-Toolbox Documentation):
* #### TOOLBOX_MODE: 

  * `train_and_test`: train on the dataset and use the newly trained model to test.
  * `only_test`: you need to set INFERENCE-MODEL_PATH, and it will use pre-trained model initialized with the MODEL_PATH to test.
  * `signal method`: use signal methods to predict rppg BVP signal and calculate heart rate.
* #### TRAIN / VALID / TEST / SIGNAL DATA: 
  * `DATA_PATH`: The input path of raw data
  * `CACHED_PATH`: The output path to preprocessed data. This path also houses a directory of .csv files containing data paths to files loaded by the dataloader. This filelist (found in default at CACHED_PATH/DataFileLists). These can be viewed for users to understand which files are used in each data split (train/val/test)
  * `EXP_DATA_NAME` If it is "", the toolbox generates a EXP_DATA_NAME based on other defined parameters. Otherwise, it uses the user-defined EXP_DATA_NAME.  
  * `BEGIN" & "END`: The portion of the dataset used for training/validation/testing. For example, if the `DATASET` is PURE, `BEGIN` is 0.0 and `END` is 0.8 under the TRAIN, the first 80% PURE is used for training the network. If the `DATASET` is PURE, `BEGIN` is 0.8 and `END` is 1.0 under the VALID, the last 20% PURE is used as the validation set. It is worth noting that validation and training sets don't have overlapping subjects.  
  * `DATA_TYPE`: How to preprocess the video data
  * `LABEL_TYPE`: How to preprocess the label data
  * `DO_CHUNK`: Whether to split the raw data into smaller chunks
  * `CHUNK_LENGTH`: The length of each chunk (number of frames)
  * `CROP_FACE`: Whether to perform face detection
  * `DYNAMIC_DETECTION`: If False, face detection is only performed at the first frame and the detected box is used to crop the video for all of the subsequent frames. If True, face detection is performed at a specific frequency which is defined by `DYNAMIC_DETECTION_FREQUENCY`. 
  * `DYNAMIC_DETECTION_FREQUENCY`: The frequency of face detection (number of frames) if DYNAMIC_DETECTION is True
  * `LARGE_FACE_BOX`: Whether to enlarge the rectangle of the detected face region in case the detected box is not large enough for some special cases (e.g., motion videos)
  * `LARGE_BOX_COEF`: The coefficient of enlarging. See more details at `https://github.com/ubicomplab/rPPG-Toolbox/blob/main/dataset/data_loader/BaseLoader.py#L162-L165`. 

  


