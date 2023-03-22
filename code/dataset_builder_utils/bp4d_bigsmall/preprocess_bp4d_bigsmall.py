from multiprocessing import Pool, Process, Value, Array, Manager
import glob
import os
from tqdm import tqdm
from dataset.data_loader.BaseLoader import BaseLoader
import mat73
import numpy as np
import cv2
import pickle




# TODO SWITCH FOR OTHER DATASETS
def read_video(file_path):
    """ Reads a video file, returns frames (N,H,W,3) """
    f = mat73.loadmat(file_path)
    frames = f['X']
    return np.asarray(frames)



# TODO SWITCH FOR OTHER DATASETS
def read_labels(file_path):
    """Reads labels corresponding to video file."""
    f = mat73.loadmat(file_path)
    keys = list(f.keys())
    data_len = f['X'].shape[0]
    keys.remove('X')

    labels = np.ones((data_len, 49)) # 47 tasks from original dataset, and added psuedo labels: 'pos_bvp','pos_env_norm_bvp'
    labels = -1*labels # make all values -1 originally

    # Labels BY INDEX IN OUTPUT NPY ARRAY
    # 0: bp_wave, 1: hr_bpm, 2: systolic_bp, 3: diastolic_bp, 4: mean_bp,
    # 5: resp_wave, 6: resp_bpm, 7: eda, [8,47]: AUs, 'pos_bvp', 'pos_env_norm_bvp'
    labels_order_list = ['bp_wave', 'HR_bpm', 'systolic_bp', 'diastolic_bp', 'mean_bp', 'resp_wave', 'resp_bpm', 'eda', 
                            'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU06int', 'AU07', 'AU09', 'AU10', 'AU10int', 'AU11', 'AU12', 'AU12int', 
                            'AU13', 'AU14', 'AU14int', 'AU15', 'AU16', 'AU17', 'AU17int', 'AU18', 'AU19', 'AU20', 'AU22', 'AU23', 'AU24', 
                            'AU27', 'AU28', 'AU29', 'AU30', 'AU31', 'AU32', 'AU33', 'AU34', 'AU35', 'AU36', 'AU37', 'AU38', 'AU39', 
                            'pos_bvp','pos_env_norm_bvp']

    # Adding All Labels To Array Dataset
    # If Label DNE Then Array Is -1 Filled For That Label
    for i in range(len(labels_order_list)):
        if labels_order_list[i] in keys:
            labels[:, i] = f[labels_order_list[i]]

    return np.asarray(labels) # Return labels as 



def resize(frames, dynamic_det, det_length,
            w, h, larger_box, crop_face, larger_box_size):
    """

    :param dynamic_det: If False, it will use the only first frame to do facial detection and
                        the detected result will be used for all frames to do cropping and resizing.
                        If True, it will implement facial detection every "det_length" frames,
                        [i*det_length, (i+1)*det_length] of frames will use the i-th detected region to do cropping.
    :param det_length: the interval of dynamic detection
    :param larger_box: whether enlarge the detected region.
    :param crop_face:  whether crop the frames.
    :param larger_box_size: the coefficient of the larger region(height and weight),
                        the middle point of the detected region will stay still during the process of enlarging.
    """
    if dynamic_det:
        det_num = ceil(frames.shape[0] / det_length)
    else:
        det_num = 1
    face_region = list()

    # obtain detection region. it will do facial detection every "det_length" frames, totally "det_num" times.
    for idx in range(det_num):
        if crop_face:
            pass
        else:
            # if crop_face:False, the face_region will be the whole frame, namely cropping nothing.
            face_region.append([0, 0, frames.shape[1], frames.shape[2]])
    face_region_all = np.asarray(face_region, dtype='int')
    resize_frames = np.zeros((frames.shape[0], h, w, 3))

    # if dynamic_det: True, the frame under processing will use the (i // det_length)-th facial region.
    # if dynamic_det: False, the frame will only use the first region obtrained from the first frame.
    for i in range(0, frames.shape[0]):
        frame = frames[i]
        if dynamic_det:
            reference_index = i // det_length
        else:
            reference_index = 0
        if crop_face:
            face_region = face_region_all[reference_index]
            frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                    max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
        resize_frames[i] = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    return resize_frames


# TODO: This function DOES NOT support a number of edge cases but is okay for now...
def chunk(big_frames, small_frames, labels, config_preprocess):
    """Chunks the data into clips."""

    # get the max length of the time window
    big_chunklen = config_preprocess['BIG_CHUNKLEN']
    small_chunklen = config_preprocess['SMALL_CHUNKLEN']
    chunk_len = max(big_chunklen, small_chunklen)
    stride = config_preprocess['CHUNK_STRIDE'] 

    # If Sliding Window
    if config_preprocess['USE_CHUNK_SLIDING_WINDOW']:
        clip_num = (labels.shape[0] - chunk_len) // stride + 1
        big_clips = [big_frames[i*stride : i*stride + 1] for i in range(clip_num)] # TODO Hardcoded Chunklen of 1
        small_clips = [small_frames[i*stride : i*stride + chunk_len] for i in range(clip_num)]
        labels_clips = [labels[i*stride : i*stride + chunk_len] for i in range(clip_num)]

    # If Straight Chunk (No Sliding Window)
    else: 
        clip_num = labels.shape[0] // chunk_len
        big_clips = [big_frames[i * chunk_len:(i + 1) * chunk_len] for i in range(clip_num)]
        small_clips = [small_frames[i * chunk_len:(i + 1) * chunk_len] for i in range(clip_num)]
        labels_clips = [labels[i * chunk_len:(i + 1) * chunk_len] for i in range(clip_num)]

    # TODO - Add Downsampleing If Need Be: Replace chunk lens with downsample and other chunklens

    return np.array(big_clips), np.array(small_clips), np.array(labels_clips)



def preprocess(frames, labels, config_preprocess):
    
    ######################################
    ########## PROCESSED FRAMES ##########
    ######################################

    # Resize Frames To The Size Of Big
    frames = resize(
            frames,
            config_preprocess['DYNAMIC_DETECTION'],
            config_preprocess['DYNAMIC_DETECTION_FREQUENCY'],
            config_preprocess['BIG_W'],
            config_preprocess['BIG_H'],
            config_preprocess['LARGE_FACE_BOX'],
            config_preprocess['CROP_FACE'],
            config_preprocess['LARGE_BOX_COEF']) 

    # Big Data Frames
    bigsmall_data = dict()
    big_data = list()
    for data_type in config_preprocess['BIG_DATA_TYPE']:
        f_c = frames.copy()
        if data_type == "Raw": # Raw Frames
            bigsmall_data[0] = f_c[:-1, :, :, :]
        elif data_type == "Normalized": # Normalized Difference Frames
            data.append(BaseLoader.diff_normalize_data(f_c))
        elif data_type == "Standardized": # Raw Standardized Frames
            data.append(BaseLoader.standardized_data(f_c)[:-1, :, :, :])
        else:
            raise ValueError("Unsupported data type!")
    data = np.concatenate(data, axis=3)

    # Small Data Frames
    small_data = list()
    for data_type in config_preprocess['SMALL_DATA_TYPE']:
        f_c = frames.copy()
        if data_type == "Raw": # Raw Frames
            small_data.append(f_c[:-1, :, :, :])
        elif data_type == "Normalized": # Normalized Difference Frames
            small_data.append(BaseLoader.diff_normalize_data(f_c))
        elif data_type == "Standardized": # Raw Standardized Frames
            small_data.append(BaseLoader.standardized_data(f_c)[:-1, :, :, :])
        else:
            raise ValueError("Unsupported data type!")
    small_data = np.concatenate(small_data, axis=3)

    # Resize Small
    small_data = resize(
            small_data,
            False,
            False,
            config_preprocess['SMALL_W'],
            config_preprocess['SMALL_H'],
            False,
            False,
            False)  

    ######################################
    ########## PROCESSED LABELS ##########
    ######################################

    # Pull signals from .mat array
    bp_wave = labels[:, 0]
    hr = labels[:, 1]
    bp_sys = labels[:, 2]
    bp_dia = labels [:, 3]
    bp_mean = labels [:, 4]
    resp_wave = labels[:, 5]
    rr = labels[:, 6]
    eda = labels[:, 7]
    au = labels[:, 8:47]
    pos_bvp = labels[:, 47]
    pos_env_norm_bvp = labels[:, 48]

    # Remove BP Outliers
    bp_sys[bp_sys < 5] = 5
    bp_sys[bp_sys > 250] = 250
    bp_dia[bp_dia < 5] = 5
    bp_dia[bp_dia > 200] = 200

    # Remove EDA Outliers
    eda[eda < 1] = 1
    eda[eda > 40] = 40

    # If there are au labels for this data file
    # TODO: Find a more elegant way to do this...
    if np.average(au) != -1:
        au[np.where(au != 0) and np.where(au != 1)] = 0 # remove unknown values (-1) from au: can do this better later: TODO GIRISH
        labels[:, 8:47] = au

    if config_preprocess['LABEL_TYPE'] == "Raw":
        labels = labels[:-1] # adjust size to match normalized size

    elif config_preprocess['LABEL_TYPE'] == "Normalized":
        labels = labels[:-1] # adjust size to match normalized size

        bp_wave = BaseLoader.diff_normalize_label(bp_wave)
        labels[:, 0] = bp_wave

        resp_wave = BaseLoader.diff_normalize_label(resp_wave)
        labels[:, 5] = resp_wave

        pos_bvp = BaseLoader.diff_normalize_label(pos_bvp)
        labels[:, 47] = pos_bvp

        pos_env_norm_bvp = BaseLoader.diff_normalize_label(pos_env_norm_bvp)
        labels[:, 48] = pos_env_norm_bvp

    elif config_preprocess['LABEL_TYPE'] == "Standardized":
        labels = labels[:-1] # adjust size to match normalized size

        bp_wave = BaseLoader.standardized_label(bp_wave)[:-1]
        labels[:, 0] = bp_wave

        resp_wave = BaseLoader.standardized_label(resp_wave)[:-1]
        labels[:, 5] = resp_wave

        pos_bvp = BaseLoader.standardized_label(pos_bvp)[:-1]
        labels[:, 47] = pos_bvp

        pos_env_norm_bvp = BaseLoader.standardized_label(pos_env_norm_bvp)[:-1]
        labels[:, 48] = pos_env_norm_bvp       

    ######################################
    ######## CHUNK DATA / LABELS #########
    ######################################
    
    # Chunk clips and labels
    if config_preprocess['DO_CHUNK']:
        big_clips, small_clips, labels_clips = chunk(big_data, small_data, labels, config_preprocess)
    else:
        big_clips = np.array([big_data])
        small_clips = np.array([small_data])
        labels_clips = np.array([labels])

    ######################################
    ########### RETURN CHUNKS ############
    ######################################
    return big_clips, small_clips, labels_clips



def save_multi_process(big_clips, small_clips, label_clips, filename, config_preprocess):
    """Saves the preprocessing data."""
    cached_path = config_preprocess['CACHED_PATH']
    if not os.path.exists(cached_path):
        os.makedirs(cached_path, exist_ok=True)
    count = 0
    input_path_name_list = []
    label_path_name_list = []
    for i in range(len(label_clips)):
        assert (len(big_clips) == len(label_clips) and len(small_clips) == len(label_clips))
        
        input_path_name = cached_path + os.sep + \
                            "{0}_input{1}.pickle".format(filename, str(count))

        label_path_name = cached_path + os.sep + \
                            "{0}_label{1}.npy".format(filename, str(count))

        frames_dict = dict()
        frames_dict[0] = big_clips[i]
        frames_dict[1] = small_clips[i]

        input_path_name_list.append(input_path_name)
        label_path_name_list.append(label_path_name)

        np.save(label_path_name, label_clips[i]) # save out labels npy file
        with open(input_path_name, 'wb') as handle: # save out frame dict pickle file
            pickle.dump(frames_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        count += 1 # count of processed clips

    return count, input_path_name_list, label_path_name_list



def preprocess_dataset_subprocess(data_dirs, config_preprocess, i, file_list_dict):
    """ invoked by preprocess_dataset for multi_process """
    filename = data_dirs[i]['path'] # get data file name
    saved_filename = data_dirs[i]['index'] # get file name w/out .mat extension
    
    frames = read_video(filename) # read in the video frames
    labels = read_labels(os.path.join(filename))

    if frames.shape[0] != labels.shape[0]:  # CHECK IF ALL DATA THE SAME LENGTH
        raise ValueError(' Preprocessing dataset subprocess: frame and label time axis not the same')

    # Preprocess the read in data files and labels
    big_clips, small_clips, labels_clips = preprocess(frames, labels, config_preprocess)

    # Save the preprocessed file chunks 
    count, input_name_list, label_name_list = save_multi_process(big_clips, small_clips, labels_clips, saved_filename, config_preprocess)
    file_list_dict[i] = input_name_list



def multi_process_manager(data_dirs, config_preprocess):
    file_num = len(data_dirs) # number of files in the dataset
    choose_range = choose_range = range(0, file_num)
    pbar = tqdm(list(choose_range))

    MAX_THREADS = 8 # max number of threads allowed for dataset multithread preprocessing

    # shared data resource
    manager = Manager()
    file_list_dict = manager.dict()
    p_list = []
    running_num = 0

    for i in choose_range:
        process_flag = True
        while process_flag:  # ensure that every i creates a process

            if running_num < MAX_THREADS:  # in case of too many processes

                # preprocess ith file from data_dirs
                p = Process(target=preprocess_dataset_subprocess, \
                            args=(data_dirs, config_preprocess, i, file_list_dict))
                p.start()
                p_list.append(p)
                running_num += 1
                process_flag = False

            # remove killed threads
            for p_ in p_list:
                if not p_.is_alive():
                    p_list.remove(p_)
                    p_.join()
                    running_num -= 1
                    pbar.update(1)

    # join all processes
    for p_ in p_list:
        p_.join()
        pbar.update(1)
    pbar.close() # finish multiprocessing

    return file_list_dict



# TODO SWITCH FOR OTHER DATASETS
# Get all data files and some meta data
def get_data(data_path):
    """Returns data directories under the path(For PURE dataset)."""

    # all file lists are of example naming format: F001T01.mat
    data_dirs = glob.glob(data_path + os.sep + "*.mat") # read in data mat files 

    if not data_dirs:
        raise ValueError(" dataset get data error!")
    dirs = list()
    for data_dir in data_dirs: # build a dict list of input files
        subject_data = os.path.split(data_dir)[-1].replace('.mat', '')
        subj_sex = subject_data[0] # subject biological sex
        subject = int(subject_data[1:4]) # subject number (by sex)
        index = subject_data # data name w/out extension
        dirs.append({"index": subject_data, "path": data_dir, "subject": subject, "sex": subj_sex})
    return dirs



def bigsmall_preprocessing(raw_data_path, config_preprocess):
    data_dirs = get_data(raw_data_path) 
    file_list_dict = multi_process_manager(data_dirs, config_preprocess)



##########################################################
#################### CONFIGS TO CHANGE ###################
##########################################################

# CONFIGURATION DICTIONARY
config_preprocess = dict()

# RAW DATA PATHS: TODO SET THIS
RAW_DATA_PATH = '/gscratch/ubicomp/girishvn/datasets/BP4D_plus/BP4DPlus_AUSubset'

# WHERE TO STORE THE PREPROCESSED DATA # TODO SET THESE
EXP_NAME = 'BP4DPlus_Big144RawStd_Small9DiffNorm_ClipLen3_AUSubset'
config_preprocess['CACHED_PATH'] = os.path.join('/gscratch/ubicomp/girishvn/rppg/rppg_datasets/PreprocessedData/', EXP_NAME)



##########################################################
###################### OTHER CONFIGS #####################
##########################################################

# Label Processing
config_preprocess['LABEL_TYPE'] = "Normalized" # Default: "Normalized"

# Data / Frame Processing
config_preprocess['BIG_DATA_TYPE'] = ["Standardized"] # Default: ["Standardized"]
config_preprocess['BIG_W'] = 144 # Default: 144
config_preprocess['BIG_H'] = 144 # Default: 144
config_preprocess['SMALL_DATA_TYPE'] = ["Normalized"] # Default: ["Normalized"]
config_preprocess['SMALL_W'] = 9 # Default: 9
config_preprocess['SMALL_H'] = 9 # Default: 9

# Number Of Consecutive Frames In Data Sample:
config_preprocess['DO_CHUNK'] = True # Default: True
config_preprocess['BIG_CHUNKLEN'] = 3 # Default: 3
config_preprocess['SMALL_CHUNKLEN'] = 3 # Default: 3
config_preprocess['CHUNK_STRIDE'] = 3 # Default: 3

# Sliding Window For Data Chunks
config_preprocess['USE_CHUNK_SLIDING_WINDOW'] = False # Default: False

# Resize Parameters
config_preprocess['DYNAMIC_DETECTION'] = False # Default: False
config_preprocess['DYNAMIC_DETECTION_FREQUENCY'] = False # Default: False
config_preprocess['LARGE_FACE_BOX'] = False # Default: False
config_preprocess['CROP_FACE'] = False # Default: False
config_preprocess['LARGE_BOX_COEF'] = False # Default: False

##########################################################
########################## MAIN ##########################
##########################################################
bigsmall_preprocessing(RAW_DATA_PATH, config_preprocess)

print('')
print('DONE DONE DONE')
print('')
