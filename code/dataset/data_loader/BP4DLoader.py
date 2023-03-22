"""The dataloader for PURE datasets.

Details for the PURE Dataset see https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure
If you use this dataset, please cite the following publication:
Stricker, R., Müller, S., Gross, H.-M.
Non-contact Video-based Pulse Rate Measurement on a Mobile Service Robot
in: Proc. 23st IEEE Int. Symposium on Robot and Human Interactive Communication (Ro-Man 2014), Edinburgh, Scotland, UK, pp. 1056 - 1062, IEEE 2014
"""
import glob
import glob
import json
import os
import re
import mat73

import cv2
import numpy as np
import scipy.signal as signal 
from utils.utils import sample
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm

class BP4DLoader(BaseLoader):
    """The data loader for the PURE dataset."""

    def __init__(self, name, data_path, config_data, total_config):
        """Initializes an PURE dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:

        """
        super().__init__(name, data_path, config_data, total_config)

    def get_data(self, data_path):
        """Returns data directories under the path(For PURE dataset)."""

        data_dirs = glob.glob(data_path + os.sep + "*.mat")
        if not data_dirs:
            raise ValueError(self.name + " dataset get data error!")
        dirs = list()
        for data_dir in data_dirs:
            subject_data = os.path.split(data_dir)[-1].replace('.mat', '')
            subj_sex = subject_data[0]
            subject = int(subject_data[1:4])
            index = subject_data
            dirs.append({"index": subject_data, "path": data_dir, "subject": subject, "sex": subj_sex})
        return dirs

    def get_data_subset(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits"""

        # get info about the dataset: subject list and num vids per subject
        m_data_info = dict() # data dict for Male subjects
        f_data_info = dict() # data dict for Female subjects
        for data in data_dirs:

            subject = data['subject']
            data_dir = data['path']
            index = data['index']
            sex = data['sex']

            trials_to_skip = ['F041T7', 'F054T9'] # TODO GIRISH, Talk to BP4D ppl about this
            if index in trials_to_skip:
                continue

            if sex == 'M':
                # creates a dictionary of data_dirs indexed by subject number
                if subject not in m_data_info:  # if subject not in the data info dictionary
                    m_data_info[subject] = []  # make an emplty list for that subject
                # append a tuple of the filename, subject num, trial num, and chunk num
                m_data_info[subject].append({"index": index, "path": data_dir, "subject": subject, "sex": sex})


            elif sex == 'F':
                # creates a dictionary of data_dirs indexed by subject number
                if subject not in f_data_info:  # if subject not in the data info dictionary
                    f_data_info[subject] = []  # make an emplty list for that subject
                # append a tuple of the filename, subject num, trial num, and chunk num
                f_data_info[subject].append({"index": index, "path": data_dir, "subject": subject, "sex": sex})

        # List of Male subjects
        m_subj_list = list(m_data_info.keys())  # all subjects by number ID
        m_subj_list.sort()
        m_num_subjs = len(m_subj_list)  # number of unique subjects

        # get male split of data set (depending on start / end)
        m_subj_range = list(range(0, m_num_subjs))
        if begin != 0 or end != 1:
            m_subj_range = list(range(int(begin * m_num_subjs), int(end * m_num_subjs)))
        print('Used Male subject ids for split:', [m_subj_list[i] for i in m_subj_range])

        # List of Female subjects
        f_subj_list = list(f_data_info.keys())  # all subjects by number ID 
        f_subj_list.sort()
        f_num_subjs = len(f_subj_list)  # number of unique subjects

        # get female split of data set (depending on start / end)
        f_subj_range = list(range(0, f_num_subjs))
        if begin != 0 or end != 1:
            f_subj_range = list(range(int(begin * f_num_subjs), int(end * f_num_subjs)))
        print('Used Female subject ids for split:', [f_subj_list[i] for i in f_subj_range])

        # compile file list
        file_info_list = []

        # add male subjects to file list
        for i in m_subj_range:
            subj_num = m_subj_list[i]
            subj_files = m_data_info[subj_num]
            file_info_list += subj_files  # add file info to file_list (tuple of fname, subj ID, trial num, # chunk num)

        # add female subjects to file list
        for i in f_subj_range:
            subj_num = f_subj_list[i]
            subj_files = f_data_info[subj_num]
            file_info_list += subj_files  # add file info to file_list (tuple of fname, subj ID, trial num, # chunk num)

        return file_info_list

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """   invoked by preprocess_dataset for multi_process.   """
        filename = data_dirs[i]['path']
        saved_filename = data_dirs[i]['index']
        
        frames = self.read_video(filename)
        labels = self.read_labels(os.path.join(filename))

        if frames.shape[0] != labels.shape[0]:  # CHECK IF ALL DATA THE SAME LENGTH
            raise ValueError(self.name, 'frame and label time axis not the same')

        frames_clips, labels_clips = self.preprocess(frames, labels, config_preprocess, config_preprocess.LARGE_FACE_BOX)
        count, input_name_list, label_name_list = self.save_multi_process(frames_clips, labels_clips, saved_filename)
        file_list_dict[i] = input_name_list

    def preprocess(self, frames, labels, config_preprocess, large_box=False):
        """Preprocesses a pair of data.

        Args:
            frames(np.array): Frames in a video.
            bvps(np.array): Bvp signal labels for a video.
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).
            large_box(bool): Whether to use a large bounding box in face cropping, e.g. in moving situations.
        """

        # Resize Frames
        frames = self.resize(
            frames,
            config_preprocess.DYNAMIC_DETECTION,
            config_preprocess.DYNAMIC_DETECTION_FREQUENCY,
            config_preprocess.W,
            config_preprocess.H,
            config_preprocess.LARGE_FACE_BOX,
            config_preprocess.CROP_FACE,
            config_preprocess.LARGE_BOX_COEF)     
        
        # data_type
        data = list()
        for data_type in config_preprocess.DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                data.append(f_c[:-1, :, :, :])
            elif data_type == "Normalized":
                data.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(BaseLoader.standardized_data(f_c)[:-1, :, :, :])
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data, axis=3)

        # Get signals from .mat file
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

        # Remove Outlier Data
        bp_sys[bp_sys < 5] = 5
        bp_sys[bp_sys > 250] = 250

        bp_dia[bp_dia < 5] = 5
        bp_dia[bp_dia > 200] = 200

        eda[eda < 1] = 1
        eda[eda > 40] = 40

        # TODO GIRISH: Do i need to replace the -1s in the dataset (for unknown points mid AU task)??
        # TODO GIRISH: Do i need to normalize the 0-5 ranked AUs?? (indexes: 13, 17, 20, 23, 27)
        au[np.where(au != 0) and np.where(au != 1)] = 0 # remove unknown values (-1) from au: can do this better later: TODO GIRISH
        labels[:, 8:47] = au

        if config_preprocess.LABEL_TYPE == "Raw":
            labels = labels[:-1] # adjust size to match normalized size

        elif config_preprocess.LABEL_TYPE == "Normalized":
            labels = labels[:-1] # adjust size to match normalized size

            bp_wave = BaseLoader.diff_normalize_label(bp_wave)
            labels[:, 0] = bp_wave

            # TODO Does this need to be diffed?
            bp_sys = BaseLoader.diff_max_min_normalize_label(bp_sys, 180, 50)
            labels[:, 2] = bp_sys

            # TODO Does this need to be diffed?
            bp_dia = BaseLoader.diff_max_min_normalize_label(bp_dia, 125, 25)
            labels[:, 3] = bp_dia

            resp_wave = BaseLoader.diff_normalize_label(resp_wave)
            labels[:, 5] = resp_wave

            # TODO Does this need to be diffed?
            eda = BaseLoader.diff_max_min_normalize_label(eda, 50, 0)
            labels[:, 7] = eda

            pos_bvp = BaseLoader.diff_normalize_label(pos_bvp)
            labels[:, 47] = pos_bvp

            pos_env_norm_bvp = BaseLoader.diff_normalize_label(pos_env_norm_bvp)
            labels[:, 48] = pos_env_norm_bvp

        elif config_preprocess.LABEL_TYPE == "Standardized":
            labels = labels[:-1] # adjust size to match normalized size
            bp_wave = BaseLoader.standardized_label(bp_wave)[:-1]
            labels[:, 0] = bp_wave

            resp_wave = BaseLoader.standardized_label(resp_wave)[:-1]
            labels[:, 5] = resp_wave

            eda = BaseLoader.max_min_normalize_label(eda, 50, 0)[:-1]
            labels[:, 7] = eda
            # TODO GIRISH: Add other labels (BP Sys / Dia) 

            pos_bvp = BaseLoader.standardized_label(pos_bvp)[:-1]
            labels[:, 47] = pos_bvp

            pos_env_norm_bvp = BaseLoader.standardized_label(pos_env_norm_bvp)[:-1]
            labels[:, 48] = pos_env_norm_bvp       
        
        # Chunk clips and labels
        if config_preprocess.DO_CHUNK:
            frames_clips, labels_clips = self.chunk(data, labels, config_preprocess.CHUNK_LENGTH)
        else:
            frames_clips = np.array([data])
            labels_clips = np.array([labels])

        return frames_clips, labels_clips

    def chunk(self, frames, labels, chunk_length):
        """Chunks the data into clips."""
        clip_num = frames.shape[0] // chunk_length
        frames_clips = [frames[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        labels_clips = [labels[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        return np.array(frames_clips), np.array(labels_clips)

    @staticmethod
    def read_video(file_path):
        """Reads a video file, returns frames(T,H,W,3) """
        f = mat73.loadmat(file_path)
        frames = f['X']
        return np.asarray(frames)

    @staticmethod
    def read_labels(file_path):
        """Reads a bvp signal file."""
        f = mat73.loadmat(file_path)
        keys = list(f.keys())
        data_len = f['X'].shape[0]
        keys.remove('X')

        labels = np.zeros((data_len, 49)) # TODO - is there a way to not hardcode this value?
        # labels by index in array 0: bp_wave, 1: hr_bpm, 2: systolic_bp, 3: diastolic_bp, 4: mean_bp, 5: resp_wave, 6: resp_bpm, 7: eda, [8,47]: AUs, 'pos_bvp', 'pos_env_norm_bvp'
        labels_order_list = ['bp_wave', 'HR_bpm', 'systolic_bp', 'diastolic_bp', 'mean_bp', 'resp_wave', 'resp_bpm', 'eda', 
                             'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU06int', 'AU07', 'AU09', 'AU10', 'AU10int', 'AU11', 'AU12', 'AU12int', 
                             'AU13', 'AU14', 'AU14int', 'AU15', 'AU16', 'AU17', 'AU17int', 'AU18', 'AU19', 'AU20', 'AU22', 'AU23', 'AU24', 
                             'AU27', 'AU28', 'AU29', 'AU30', 'AU31', 'AU32', 'AU33', 'AU34', 'AU35', 'AU36', 'AU37', 'AU38', 'AU39', 
                             'pos_bvp','pos_env_norm_bvp']

        # Adding Non-AU labels to np array
        for i in range(len(labels_order_list)):
            if labels_order_list[i] in keys:
                labels[:, i] = f[labels_order_list[i]]

        return np.asarray(labels)

        