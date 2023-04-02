from multiprocessing import Pool, Process, Value, Array, Manager
import glob
import os

from tqdm import tqdm
from dataset.data_loader.BaseLoader import BaseLoader
import cv2

import numpy as np
import pickle

import hdf5storage as hdf5



############################################
####### Generating Psuedo POS Labels #######
############################################

# CALCULATE AVG R/G/B VALUE FOR EACH FRAME
def rgb_process_video(frames):
    """Calculates the average value of each frame."""
    RGB = []
    for frame in frames:
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(summation / (frame.shape[0] * frame.shape[1]))
    return np.asarray(RGB)



# REMMOVE SIGNAL LINEAR TRENDS (SOMEWHAT SIMILLAR TO LOW-PASS FILTERING
def detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal



# PLANE ORTHOGONAL TO SKIN RPPG METHOD
def POS_WANG(frames, fs):
    WinSec = 1.6
    RGB = rgb_process_video(frames)
    N = RGB.shape[0]
    H = np.zeros((1, N))
    l = math.ceil(WinSec * fs)

    for n in range(N):
        m = n - l
        if m >= 0:
            Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
            Cn = np.mat(Cn).H
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0, temp] = h[0, temp] - mean_h
            H[0, m:n] = H[0, m:n] + (h[0])

    BVP = H
    BVP = detrend(np.mat(BVP).H, 100)
    BVP = np.asarray(np.transpose(BVP))[0]
    # b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    # BVP = signal.filtfilt(b, a, BVP.astype(np.double))
    return BVP



# ITERATE THROUGH DATASET AND GENERATE BVP PSUEDO LABELS
def generate_psuedo_labels(data_dict):

    print('Generating POS Psuedo BVP Labels...')

    # READ VIDEO FRAMES
    x = data_dict['X'] # read in video frames

    # GENERATE POS PPG SIGNAL
    fs = 25 # bp4d sampling rate: 25hz
    bvp = POS_WANG(x, fs) # generate POS PPG signal
    bvp = np.array(bvp)

    # AGGRESSIVELY FILTER PPG SIGNAL
    hr_arr = data_dict['HR_bpm'] # get hr freq from GT label
    avg_hr_bpm = np.sum(hr_arr)/len(hr_arr) # calculate avg hr for the entire trial
    hr_freq = avg_hr_bpm / 60 # divide beats per min by 60, to get beats pers secone
    halfband = 20 / fs # half bandwith to account for HR variation (accounts for +/- 20 bpm variation from mean HR)

    # MAX BANDWIDTH [0.70, 3]Hz = [42, 180]BPM (BANDWIDTH MAY BE SMALLER)
    min_freq = hr_freq - halfband # calculate min cutoff frequency
    if min_freq < 0.70:
        min_freq = 0.70
    max_freq = hr_freq + halfband # calculate max cutoff frequency
    if max_freq > 3:
        max_freq = 3

    # FILTER POS PPG W/ 2nd ORDER BUTTERWORTH FILTER
    b, a = signal.butter(2, [(min_freq) / fs * 2, (max_freq) / fs * 2], btype='bandpass')
    pos_bvp = signal.filtfilt(b, a, bvp.astype(np.double))

    # APPLY HILBERT NORMALIZATION TO NORMALIZE PPG AMPLITUDE
    analytic_signal = signal.hilbert(pos_bvp)
    amplitude_envelope = np.abs(analytic_signal)
    env_norm_bvp = pos_bvp/amplitude_envelope

    # Add New Fields to Data Mat File
    data_dict['pos_bvp'] = pos_bvp
    data_dict['pos_env_norm_bvp'] = env_norm_bvp

    return data_dict # return data dict w/ POS psuedo labels



################################################################
####### CONSTRUCTING DATA DICTIONARY FROM RAW DATA FILES #######
################################################################


def read_raw_vid_frames(data_dir_info)

    data_path = data_dir_info['path']
    subject = data_dir_info['subject']
    trial = data_dir_info['trial']

    # GRAB EACH FRAME FROM ZIP FILE
    imgzip = open(os.path.join(data_path, '2D+3D', subject+'.zip'))
    zipfile_path = os.path.join(data_path, '2D+3D', subject+'.zip')
    print(zipfile_path)
    print('Current time after upzipping the file: ', datetime.now())
    cnt = 0
    with zipfile.ZipFile(zipfile_path, "r") as zippedImgs:
        for ele in zippedImgs.namelist():
            ext = os.path.splitext(ele)[-1]
            ele_task = str(ele).split('/')[1]
            if ext == '.jpg' and ele_task == trial:
                data = zippedImgs.read(ele)
                vid_frame = cv2.imdecode(np.fromstring(data, np.uint8), cv2.IMREAD_COLOR)
                vid_LxL = downsample_frame(vid_frame, dim)
                # clip image values to range (1/255, 1)
                vid_LxL[vid_LxL > 1] = 1
                vid_LxL[vid_LxL < 1./255] = 1./255
                vid_LxL = np.expand_dims(vid_LxL, axis=0)
                if cnt == 0:
                    Xsub = vid_LxL
                else:
                    Xsub = np.concatenate((Xsub, vid_LxL), axis=0)
                cnt += 1
    
    if cnt == 0:
        return


def read_raw_phys_labels(data_dir_info, len_Xsub):

    data_path = data_dir_info['path']
    subject = data_dir_info['subject']
    trial = data_dir_info['trial']
    base_path = os.path.join(data_path, "Physiology", subject, trial)

    # READ IN PHYSIOLOGICAL LABELS TXT FILE DATA
    try:
        bp_wave = pd.read_csv(os.path.join(base_path, "BP_mmHg.txt")).to_numpy().flatten()
        HR_bpm = pd.read_csv(os.path.join(base_path, "Pulse Rate_BPM.txt")).to_numpy().flatten()
        resp_wave = pd.read_csv(os.path.join(base_path, "Resp_Volts.txt")).to_numpy().flatten()
        resp_bpm = pd.read_csv(os.path.join(base_path, "Respiration Rate_BPM.txt")).to_numpy().flatten()
        mean_BP = pd.read_csv(os.path.join(base_path, "LA Mean BP_mmHg.txt")).to_numpy().flatten()
        sys_BP = pd.read_csv(os.path.join(base_path, "LA Systolic BP_mmHg.txt")).to_numpy().flatten()
        dia_BP = pd.read_csv(os.path.join(base_path, "BP Dia_mmHg.txt")).to_numpy().flatten()
        eda = pd.read_csv(os.path.join(base_path, "EDA_microsiemens.txt")).to_numpy().flatten()
    except FileNotFoundError:
        print('Label File Not Found At Basepath', base_path)
        return

    # RESIZE SIGNALS TO LENGTH OF X (FRAMES) AND CONVERT TO NPY ARRAY
    bp_wave = np.interp(np.linspace(0, len(bp_wave), len_Xsub), np.arange(0, len(bp_wave)), bp_wave)
    HR_bpm = np.interp(np.linspace(0, len(HR_bpm), len_Xsub), np.arange(0, len(HR_bpm)), HR_bpm)
    resp_wave = np.interp(np.linspace(0, len(resp_wave), len_Xsub), np.arange(0, len(resp_wave)), resp_wave)
    resp_bpm = np.interp(np.linspace(0, len(resp_bpm), len_Xsub), np.arange(0, len(resp_bpm)), resp_bpm)
    mean_BP = np.interp(np.linspace(0, len(mean_BP), len_Xsub), np.arange(0, len(mean_BP)), mean_BP)
    sys_BP = np.interp(np.linspace(0, len(sys_BP), len_Xsub), np.arange(0, len(sys_BP)), sys_BP)
    dia_BP = np.interp(np.linspace(0, len(dia_BP), len_Xsub), np.arange(0, len(dia_BP)), dia_BP)
    eda = np.interp(np.linspace(0, len(eda), len_Xsub), np.arange(0, len(eda)), eda)

    
    return bp_wave, HR_bpm, resp_wave, resp_bpm, mean_BP, sys_BP, dia_BP, eda  


def construct_data_dict(data_dir_info):

    # BUILD DICTIONARY TO STORE FRAMES AND LABELS
    data_dict = dict()

    # READ IN RAW VIDEO FRAMES
    X = read_raw_vid_frames(data_dir_info)
    lenX = X.shape[0]

    # READ IN RAW PHYSIOLOGICAL SIGNAL LABELS 
    bp_wave, HR_bpm, resp_wave, resp_bpm, mean_BP, sys_BP, dia_BP, eda = read_raw_phys_labels(data_dir_info, lenX)

    # READ IN ACTION UNIT (AU) LABELS
    if trial in [1, 6, 7, 8]: # trials w/ AU labels
     # TODO

    # CROP FRAMES AND LABELS TO AU SUBSET (IF TRUE)
    if config_preprocess.USE_AU_SUBSET:
        au_subset_idx = # TODO - write function to get AU subset dataset


    # SAVE LABELS AND DATA TO DICTIONARY
    data_dict['X'] = X

    data_dict['bp_wave'] = bp_wave
    data_dict['HR_bpm'] = HR_bpm
    data_dict['mean_bp'] = mean_BP
    data_dict['systolic_bp'] = sys_BP
    data_dict['diastolic_bp'] = dia_BP
    data_dict['resp_wave'] = resp_wave
    data_dict['resp_bpm'] = resp_bpm
    data_dict['eda'] = eda



# GET VIDEO FRAMES FROM DATA DICTIONARY
def read_video(data_dict):
    """ Reads a video file, returns frames (N,H,W,3) """
    frames = data_dict['X']
    return np.asarray(frames)



# GET VIDEO LABELS FROM DATA DICTIONARY AND FORMAT AS ARRAY
def read_labels(data_dict):
    """Reads labels corresponding to video file."""
    f = data_dict
    keys = list(f.keys())
    data_len = f['X'].shape[0] # get the video data length
    keys.remove('X') # remove X from the processed keys (not a label)

    # Init labels array
    labels = np.ones((data_len, 49)) # 47 tasks from original dataset, and added psuedo labels: 'pos_bvp','pos_env_norm_bvp'
    labels = -1*labels # make all values -1 originally

    # LABELS BY INDEX IN OUTPUT LABELS NPY ARRAY
    # 0: bp_wave, 1: hr_bpm, 2: systolic_bp, 3: diastolic_bp, 4: mean_bp,
    # 5: resp_wave, 6: resp_bpm, 7: eda, [8,47]: AUs, 'pos_bvp', 'pos_env_norm_bvp'
    labels_order_list = ['bp_wave', 'HR_bpm', 'systolic_bp', 'diastolic_bp', 'mean_bp', 'resp_wave', 'resp_bpm', 'eda', 
                            'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU06int', 'AU07', 'AU09', 'AU10', 'AU10int', 'AU11', 'AU12', 'AU12int', 
                            'AU13', 'AU14', 'AU14int', 'AU15', 'AU16', 'AU17', 'AU17int', 'AU18', 'AU19', 'AU20', 'AU22', 'AU23', 'AU24', 
                            'AU27', 'AU28', 'AU29', 'AU30', 'AU31', 'AU32', 'AU33', 'AU34', 'AU35', 'AU36', 'AU37', 'AU38', 'AU39', 
                            'pos_bvp','pos_env_norm_bvp']

    # ADDING LABELS TO DATA ARRAY
    # If Label DNE Then Array Is -1 Filled For That Label
    # Note: BP4D does not have AU labels for all trials: These fields are thus COMPLETELY -1 filled for these trials
    for i in range(len(labels_order_list)):
        if labels_order_list[i] in keys:
            labels[:, i] = f[labels_order_list[i]]

    return np.asarray(labels) # Return labels as np array



###############################################
####### PREPROCESS VIDEO AND LABEL DATA #######
###############################################

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



def chunk(big_frames, small_frames, labels, config_preprocess):
    """Chunks the data into clips."""

    chunk_len = max(big_chunklen, small_chunklen)
    clip_num = labels.shape[0] // chunk_len
    big_clips = [big_frames[i * chunk_len:(i + 1) * chunk_len] for i in range(clip_num)]
    small_clips = [small_frames[i * chunk_len:(i + 1) * chunk_len] for i in range(clip_num)]
    labels_clips = [labels[i * chunk_len:(i + 1) * chunk_len] for i in range(clip_num)]

    return np.array(big_clips), np.array(small_clips), np.array(labels_clips)



def preprocess(frames, labels, config_preprocess):
    
    #######################################
    ########## PROCESSING FRAMES ##########
    #######################################

    # RESIZE FRAMES TO BIG SIZE  (144x144 DEFAULT)
    frames = resize(
            frames,
            config_preprocess['DYNAMIC_DETECTION'], # dynamic face detection
            config_preprocess['DYNAMIC_DETECTION_FREQUENCY'], # how often to use face detection
            config_preprocess['BIG_W'], # Big width
            config_preprocess['BIG_H'], # Big height
            config_preprocess['LARGE_FACE_BOX'], # larger-than-face bounding box coefficient
            config_preprocess['CROP_FACE'], # use face cropping
            config_preprocess['LARGE_BOX_COEF']) # use larger-than-face bounding box


    # PROCESS BIG FRAMES
    big_data = list()
    for data_type in config_preprocess['BIG_DATA_TYPE']:
        f_c = frames.copy()
        if data_type == "Raw": # Raw Frames
            big_data.append(f_c[:-1, :, :, :])
        elif data_type == "Normalized": # Normalized Difference Frames
            big_data.append(BaseLoader.diff_normalize_data(f_c))
        elif data_type == "Standardized": # Raw Standardized Frames
            big_data.append(BaseLoader.standardized_data(f_c)[:-1, :, :, :])
        else:
            raise ValueError("Unsupported data type!")
    big_data = np.concatenate(big_data, axis=3)

    # PROCESS SMALL FRAMES
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

    # RESIZE SMALL FRAMES TO LOWER RESOLUTION (9x9 DEFAULT)
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

    # EXTRACT LABELS FROM ARRAY
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

    # REMOVE BP OUTLIERS
    bp_sys[bp_sys < 5] = 5
    bp_sys[bp_sys > 250] = 250
    bp_dia[bp_dia < 5] = 5
    bp_dia[bp_dia > 200] = 200

    # REMOVE EDA OUTLIERS
    eda[eda < 1] = 1
    eda[eda > 40] = 40

    # REMOVE AU -1 LABELS IN AU SUBSET
    if np.average(au) != -1:
        au[np.where(au != 0) and np.where(au != 1)] = 0
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

    data_dir_info = data_dirs[i]['path'] # get data raw data file path 
    saved_filename = data_dirs[i]['index'] # get file name w/out .mat extension # TODO CHANGE THIS COMMENT

    # TODO (IF TRUE) AND NOT AN AU TRIAL: SKIP
    if config_preprocess.USE_AU_SUBSET:
        au_subset_idx = # TODO - write function to get AU subset dataset

    # CONSTRUCT DATA DICTIONARY FOR VIDEO TRIAL
    data_dict = construct_data_dict(data_dir_info) # TODO construct a dictionary of ALL labels and video frames (of equal length)
    data_dict = generate_psuedo_labels(data_dict) # adds POS psuedo BVP labels to dataset
    
    # SEPERATE DATA INTO VIDEO FRAMES AND LABELS ARRAY
    frames = read_video(data_dict) # read in the video frames
    labels = read_labels(data_dict) # read in video labels 
    if frames.shape[0] != labels.shape[0]: # check if data and labels are the same length
        raise ValueError(' Preprocessing dataset subprocess: frame and label time axis not the same')

    # PREPROCESS VIDEO FRAMES AND LABELS (eg. DIFF-NORM, RAW_STD)
    big_clips, small_clips, labels_clips = preprocess(frames, labels, config_preprocess)

    # SAVE PREPROCESSED FILE CHUNKS
    count, input_name_list, label_name_list = save_multi_process(big_clips, small_clips, labels_clips, saved_filename, config_preprocess)
    file_list_dict[i] = input_name_list



def multi_process_manager(data_dirs, config_preprocess):
    
    file_num = len(data_dirs) # number of files in the dataset
    choose_range = range(0, file_num)
    pbar = tqdm(list(choose_range))

    MAX_THREADS = 8 # max number of threads allowed for dataset multithread preprocessing

    # SHARED DATA RESOURCES FOR CROSS-PROCESS DATA SHARING
    manager = Manager() # multiprocess manager 
    file_list_dict = manager.dict() # dictionary to store file list of each processed file (thread safe)
    p_list = [] # list of active threads
    running_num = 0 # number of active threads

    for i in choose_range:
        process_flag = True
        while process_flag:  # ensure that every i creates a process

            if running_num < MAX_THREADS:  # in case of too many processes

                # PREPROCESS i-TH FILE FROM data_dirs
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

    # all subject trials # F008T8.mat

    # GET ALL SUBJECT TRIALS IN DATASET
    subj_trials = glob.glob(os.path.join(data_path, "Physiology", "F*", "T*"))

    # SPLIT PATH UP INTO INFORMATION (SUBJECT, TRIAL, ETC.)
    data_dirs = list()
    for trial_path in subj_trials:
        trial_data = trial_path.split(os.sep)
        index = trial_data[-1] + trial_data[-2] # should be of format: F008T8 (TODO verify this)
        trial = trial_data[-1]
        subj_sex = index[0] # subject biological sex
        subject = int(index[1:4]) # subject number (by sex)
        data_dirs.append({"index": index, "path": data_path, "subject": subject, "trial": trial, ,"sex": subj_sex})

    # RETURN DATA DIRS 
    return data_dirs



        
        

    # TODO FILTER OUT TRIALS W/OUT AU LABELS  
    if config_preprocess.USE_AU_SUBSET:
        au_subset_idx = # TODO - write function to get AU subset dataset

    if not subj_trials:
        raise ValueError(" dataset get data error!")

    data_dirs = list()
    for data_dir in subj_trials: # build a dict list of input files
        subject_data = os.path.split(data_dir)[-1].replace('.mat', '')
        subj_sex = subject_data[0] # subject biological sex
        subject = int(subject_data[1:4]) # subject number (by sex)
        index = subject_data # data name w/out extension
        data_dirs.append({"index": subject_data, "path": data_dir, "subject": subject, "sex": subj_sex})
    return data_dirs



def bigsmall_preprocessing(raw_data_path, config_preprocess):

    # GET DATASET INFORMATION (PATHS AND OTHER META DATA REGARDING ALL VIDEO TRIALS)
    data_dirs = get_data(raw_data_path) # TODO read in raw data and generate data dictionary

    # READ RAW DATA, PREPROCESS, AND SAVE PROCESSED DATA FILES
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
