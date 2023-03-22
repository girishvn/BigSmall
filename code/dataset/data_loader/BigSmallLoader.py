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
import pickle

import cv2
import numpy as np
import pandas as pd
import scipy.signal as signal 
from utils.utils import sample
from dataset.data_loader.BaseLoader import BaseLoader
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from tqdm import tqdm


######################################
################ BigSmall Loader ################
######################################
class BP4DBigSmallLoader(BaseLoader):
    """The data loader for the BP4D dataset."""



    def __init__(self, name, data_path, config_data, total_config):
        """Initializes an PURE dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:

        """
        super().__init__(name, data_path, config_data, total_config)



    # Editted for BP4D BIG SMALL LOADER - Use pickle format input files instead numpy
    def build_file_list_retroactive(self, data_dirs, begin, end):

        # get data split
        data_dirs = self.get_data_subset(data_dirs, begin, end)

        # generate a list of unique raw-data file names 
        filename_list = []
        for i in range(len(data_dirs)):
            filename_list.append(data_dirs[i]['index'])
        filename_list = list(set(filename_list)) # ensure all indexes are unique

        # generate a list of all preprocessed / chunked data files 
        file_list = []
        for fname in filename_list:
            processed_file_data = list(glob.glob(self.cached_path + os.sep + "{0}_input*.pickle".format(fname)))
            file_list += processed_file_data

        if not file_list:
            raise ValueError(self.name, 'File list empty. Check preprocessed data folder exists and is not empty.')

        file_list_df = pd.DataFrame(file_list, columns = ['input_files'])   
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)



    def load(self):
        """Loads the preprocessing data listed in the file list"""
        file_list_path = self.file_list_path # get list of files in 
        file_list_df = pd.read_csv(file_list_path) 
        inputs = file_list_df['input_files'].tolist()
        if inputs == []:
            raise ValueError(self.name + ' dataset loading data error!')
        inputs = sorted(inputs)  # sort input file name list
        labels = [input_file.replace("input", "label") for input_file in inputs]
        labels = [label_file.replace(".pickle", ".npy") for label_file in labels]
        self.inputs = inputs
        self.labels = labels
        self.len = len(inputs)
        print("Loaded data len:", self.len)


    # Data Augmentations
    def bigsmall_randImgTransforms(self, data):

        # transform npy image to tensor - changes shape to [C,H,W]
        toTensorTransform = transforms.ToTensor()

        # big_data = torch.from_numpy(data[0])
        # big = torch.moveaxis(big, -1, 1)
        # small = torch.from_numpy(data[1])
        # small = torch.moveaxis(small, -1, 1)

        chunk_len = data[0].shape[0]

        for n in range(chunk_len):

            big = toTensorTransform(data[0][n,:,:,:])
            small = toTensorTransform(data[1][n,:,:,:])
            big_size = big.shape[-1]
            small_size = small.shape[-1]

            # random horizontal flip
            if np.random.rand() >= 0.5 and False:
                big = TF.hflip(big)
                small = TF.hflip(small)

            # random crop of 15%
            if np.random.rand() >= 0.5:
                crop_size = 0.85
                new_big_size = int(big_size*crop_size)
                new_small_size = int(small_size*crop_size)

                # crop params
                i, j, h, w = transforms.RandomCrop.get_params(big, output_size=(new_big_size, new_big_size))

                # crop big
                for n in range(chunk_len):
                    big_crop = TF.crop(big, i, j, h, w)
                    big[:,i:i+new_big_size, j:j+new_big_size] = big_crop

            # reshape
            big = big.numpy()
            big = np.moveaxis(big, 0, 2)

            small = small.numpy()
            small = np.moveaxis(small, 0, 2)

            data[0][n,:,:,:] = big
            data[1][n,:,:,:] = small

        # adjust image in batch to reflect transformations
        return data



    def __getitem__(self, index):
        """Returns a clip of video(3,T,W,H) and it's corresponding signals(T)."""

        with open(self.inputs[index], 'rb') as handle:
            data = pickle.load(handle)

        # Applying random image transforms to training data (if enabled)
        if self.name == 'train' and self.data_augmentation:
            data = self.bigsmall_randImgTransforms(data)

        # format data shapes
        if self.data_format == 'NDCHW':
            data[0] = np.float32(np.transpose(data[0], (0, 3, 1, 2)))
            data[1] = np.float32(np.transpose(data[1], (0, 3, 1, 2)))
        elif self.data_format == 'NCDHW':
            data[0] = np.float32(np.transpose(data[0], (3, 0, 1, 2)))
            data[1] = np.float32(np.transpose(data[1], (3, 0, 1, 2)))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')

        label = np.load(self.labels[index])
        label = np.float32(label)
        
        item_path = self.inputs[index]
        item_path_filename = item_path.split(os.sep)[-1]
        split_idx = item_path_filename.index('_')
        filename = item_path_filename[:split_idx]
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]
        return data, label, filename, chunk_id
    


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






######################################
################ PURE ################
######################################
class PUREBigSmallLoader(BaseLoader):
    """The data loader for the PURE dataset."""

    def __init__(self, name, data_path, config_data, total_config):

        super().__init__(name, data_path, config_data, total_config)


    # Editted for BP4D BIG SMALL LOADER - Use pickle format input files instead numpy
    def build_file_list_retroactive(self, data_dirs, begin, end):

        # get data split
        data_dirs = self.get_data_subset(data_dirs, begin, end)

        # generate a list of unique raw-data file names 
        filename_list = []
        for i in range(len(data_dirs)):
            filename_list.append(data_dirs[i]['index'])
        filename_list = list(set(filename_list)) # ensure all indexes are unique

        # generate a list of all preprocessed / chunked data files 
        file_list = []
        for fname in filename_list:
            processed_file_data = list(glob.glob(self.cached_path + os.sep + "{0}_input*.pickle".format(fname)))
            file_list += processed_file_data

        if not file_list:
            raise ValueError(self.name, 'File list empty. Check preprocessed data folder exists and is not empty.')

        file_list_df = pd.DataFrame(file_list, columns = ['input_files'])   
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)


    def load(self):
        """Loads the preprocessing data listed in the file list"""
        file_list_path = self.file_list_path # get list of files in 
        file_list_df = pd.read_csv(file_list_path) 
        inputs = file_list_df['input_files'].tolist()
        if inputs == []:
            raise ValueError(self.name + ' dataset loading data error!')
        inputs = sorted(inputs)  # sort input file name list
        labels = [input_file.replace("input", "label") for input_file in inputs]
        labels = [label_file.replace(".pickle", ".npy") for label_file in labels]
        self.inputs = inputs
        self.labels = labels
        self.len = len(inputs)
        print("Loaded data len:", self.len)


    def __getitem__(self, index):
        """Returns a clip of video(3,T,W,H) and it's corresponding signals(T)."""

        with open(self.inputs[index], 'rb') as handle:
            data = pickle.load(handle)

        if self.data_format == 'NDCHW':
            data[0] = np.float32(np.transpose(data[0], (0, 3, 1, 2)))
            data[1] = np.float32(np.transpose(data[1], (0, 3, 1, 2)))
        elif self.data_format == 'NCDHW':
            data[0] = np.float32(np.transpose(data[0], (3, 0, 1, 2)))
            data[1] = np.float32(np.transpose(data[1], (3, 0, 1, 2)))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')

        label = np.load(self.labels[index])
        label = np.float32(label)
        
        item_path = self.inputs[index]
        item_path_filename = item_path.split(os.sep)[-1]
        split_idx = item_path_filename.index('_')
        filename = item_path_filename[:split_idx]
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]
        return data, label, filename, chunk_id
    

    def get_data(self, data_path):
        """Returns data directories under the path(For PURE dataset)."""

        data_dirs = glob.glob(data_path + os.sep + "*-*")
        if not data_dirs:
            raise ValueError(self.name + " dataset get data error!")
        dirs = list()
        for data_dir in data_dirs:
            subject_trail_val = os.path.split(data_dir)[-1].replace('-', '')
            index = int(subject_trail_val)
            subject = int(subject_trail_val[0:2])
            dirs.append({"index": index, "path": data_dir, "subject": subject})
        return dirs

    def get_data_subset(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits"""

        # return the full directory
        if begin == 0 and end == 1:
            return data_dirs

        # get info about the dataset: subject list and num vids per subject
        data_info = dict()
        for data in data_dirs:

            subject = data['subject']
            data_dir = data['path']
            index = data['index']

            # creates a dictionary of data_dirs indexed by subject number
            if subject not in data_info:  # if subject not in the data info dictionary
                data_info[subject] = []  # make an emplty list for that subject
            # append a tuple of the filename, subject num, trial num, and chunk num
            data_info[subject].append({"index": index, "path": data_dir, "subject": subject})

        subj_list = list(data_info.keys())  # all subjects by number ID (1-27)
        subj_list = sorted(subj_list)
        num_subjs = len(subj_list)  # number of unique subjects

        # get split of data set (depending on start / end)
        subj_range = list(range(0, num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))
        print('used subject ids for split:', [subj_list[i] for i in subj_range])

        # compile file list
        data_dirs_new = []
        for i in subj_range:
            subj_num = subj_list[i]
            subj_files = data_info[subj_num]
            data_dirs_new += subj_files  # add file information to file_list (tuple of fname, subj ID, trial num,
            # chunk num)

        return data_dirs_new



######################################
################ UBFC ################
######################################
class UBFCBigSmallLoader(BaseLoader):
    """The data loader for the UBFC dataset."""

    def __init__(self, name, data_path, config_data, total_config):
        super().__init__(name, data_path, config_data, total_config)


     # Editted for BP4D BIG SMALL LOADER - Use pickle format input files instead numpy
    def build_file_list_retroactive(self, data_dirs, begin, end):

        # get data split
        data_dirs = self.get_data_subset(data_dirs, begin, end)

        # generate a list of unique raw-data file names 
        filename_list = []
        for i in range(len(data_dirs)):
            filename_list.append(data_dirs[i]['index'])
        filename_list = list(set(filename_list)) # ensure all indexes are unique

        # generate a list of all preprocessed / chunked data files 
        file_list = []
        for fname in filename_list:
            processed_file_data = list(glob.glob(self.cached_path + os.sep + "{0}_input*.pickle".format(fname)))
            file_list += processed_file_data

        if not file_list:
            raise ValueError(self.name, 'File list empty. Check preprocessed data folder exists and is not empty.')

        file_list_df = pd.DataFrame(file_list, columns = ['input_files'])   
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)


    def load(self):
        """Loads the preprocessing data listed in the file list"""
        file_list_path = self.file_list_path # get list of files in 
        file_list_df = pd.read_csv(file_list_path) 
        inputs = file_list_df['input_files'].tolist()
        if inputs == []:
            raise ValueError(self.name + ' dataset loading data error!')
        inputs = sorted(inputs)  # sort input file name list
        labels = [input_file.replace("input", "label") for input_file in inputs]
        labels = [label_file.replace(".pickle", ".npy") for label_file in labels]
        self.inputs = inputs
        self.labels = labels
        self.len = len(inputs)
        print("Loaded data len:", self.len)


    def __getitem__(self, index):
        """Returns a clip of video(3,T,W,H) and it's corresponding signals(T)."""

        with open(self.inputs[index], 'rb') as handle:
            data = pickle.load(handle)

        if self.data_format == 'NDCHW':
            data[0] = np.float32(np.transpose(data[0], (0, 3, 1, 2)))
            data[1] = np.float32(np.transpose(data[1], (0, 3, 1, 2)))
        elif self.data_format == 'NCDHW':
            data[0] = np.float32(np.transpose(data[0], (3, 0, 1, 2)))
            data[1] = np.float32(np.transpose(data[1], (3, 0, 1, 2)))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')

        label = np.load(self.labels[index])
        label = np.float32(label)
        
        item_path = self.inputs[index]
        item_path_filename = item_path.split(os.sep)[-1]
        split_idx = item_path_filename.index('_')
        filename = item_path_filename[:split_idx]
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]
        return data, label, filename, chunk_id


    def get_data(self, data_path):
        """Returns data directories under the path(For UBFC dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "subject*")
        if not data_dirs:
            raise ValueError(self.name + " dataset get data error!")
        dirs = [{"index": re.search(
            'subject(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
        return dirs

    def get_data_subset(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values"""
        if begin == 0 and end == 1: # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new








    


        