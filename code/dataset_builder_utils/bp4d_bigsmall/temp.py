import os
import glob
import numpy as np
from numpy.core.defchararray import index
import pandas as pd
import argparse
import scipy
from hdf5storage import savemat
from hdf5storage import loadmat
import matplotlib.pyplot as plt
import skimage
from skimage.transform import resize
import zipfile
import shutil
from datetime import datetime
import zipfile
from io import StringIO
from PIL import Image
import imghdr
import cv2
from skimage.util import img_as_float




def downsample_frame(frame, dim):
    # if 3D
    if len(frame.shape) == 3:
        # return skimage.transform.resize(frame[int(frame.shape[0]-frame.shape[1]):,:,:], output_shape=(dim, dim))
        vidLxL = cv2.resize(img_as_float(frame[int(frame.shape[0]-frame.shape[1]):,:,:]), (dim, dim), interpolation = cv2.INTER_AREA)
        return cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
        
    elif len(frame.shape) == 2:
        return skimage.transform.resize(frame[:,
            int(frame.shape[1]/2-frame.shape[0]/2):int(frame.shape[1]/2+frame.shape[0]/2)],
            output_shape=(dim, dim))
    else:
        raise IndexError
 
​
def read_labels(phys_url, len_Xsub):
    try:
        ppg_wave = pd.read_csv(os.path.join(phys_url, "BP_mmHg.txt")).to_numpy().flatten()
        HR_bpm = pd.read_csv(os.path.join(phys_url, "Pulse Rate_BPM.txt")).to_numpy().flatten()
        resp_wave = pd.read_csv(os.path.join(phys_url, "Resp_Volts.txt")).to_numpy().flatten()
        resp_bpm = pd.read_csv(os.path.join(phys_url, "Respiration Rate_BPM.txt")).to_numpy().flatten()
        bp_dia = pd.read_csv(os.path.join(phys_url, "BP Dia_mmHg.txt")).to_numpy().flatten()
        mean_BP = pd.read_csv(os.path.join(phys_url, "LA Mean BP_mmHg.txt")).to_numpy().flatten()
        systolic_BP = pd.read_csv(os.path.join(phys_url, "LA Systolic BP_mmHg.txt")).to_numpy().flatten()
        eda = pd.read_csv(os.path.join(phys_url, "EDA_microsiemens.txt")).to_numpy().flatten()
    except FileNotFoundError:
        return
​
    ppg_wave = np.interp(np.linspace(0, len(ppg_wave), len_Xsub), np.arange(0, len(ppg_wave)), ppg_wave)
    HR_bpm = np.interp(np.linspace(0, len(HR_bpm), len_Xsub), np.arange(0, len(HR_bpm)), HR_bpm)
    resp_wave = np.interp(np.linspace(0, len(resp_wave), len_Xsub), np.arange(0, len(resp_wave)), resp_wave)
    resp_bpm = np.interp(np.linspace(0, len(resp_bpm), len_Xsub), np.arange(0, len(resp_bpm)), resp_bpm)
    bp_dia = np.interp(np.linspace(0, len(bp_dia), len_Xsub), np.arange(0, len(bp_dia)), bp_dia)
    mean_BP = np.interp(np.linspace(0, len(mean_BP), len_Xsub), np.arange(0, len(mean_BP)), mean_BP)
    systolic_BP = np.interp(np.linspace(0, len(systolic_BP), len_Xsub), np.arange(0, len(systolic_BP)), systolic_BP)
    eda = np.interp(np.linspace(0, len(eda), len_Xsub), np.arange(0, len(eda)), eda)
​
    return ppg_wave, resp_wave, HR_bpm, resp_bpm, bp_dia, mean_BP, systolic_BP, eda  
​
def read_AU(output_folder, subject, task, AU_url, AU_num, frame_shape):
    AU_int_num = [6, 10, 12, 14, 17]
    
    AU_OCC_url = os.path.join(AU_url, "AU_OCC", subject + '_' + task + '.csv')
    
    file_name = os.path.join(output_folder, "{0:s}{1:s}.mat".format(subject, task))
    
    # read in each csv file
    AUs = pd.read_csv(AU_OCC_url, header = 0).to_numpy()
​
    start_frame = AUs[0,0]
    end_frame = AUs[AUs.shape[0] - 1, 0]
    print("AU length: ", AUs.shape[0])
    print("start_frame: ", start_frame)
    print("end_frame: ", end_frame)
 
    for au_idx, au in enumerate(AU_num):
        if au < 10:
            AU_key = 'AU' + '0' + str(au)
        else:
            AU_key = 'AU' + str(au)
        print(AU_key)
        aucoding = AUs[:, au_idx + 1]
        print(AU_key +': ' + str(aucoding.shape))
        print("nonzero index: ", np.nonzero(aucoding))
        if start_frame > 1:
            # pad the previous frame with -1
            aucoding = np.pad(aucoding, (start_frame - 1, 0), 'constant', constant_values = (-1, -1))
        if end_frame < frame_shape:
            # pad the following frame with -1 as well
            aucoding = np.pad(aucoding, (0, frame_shape - end_frame), 'constant', constant_values = (-1, -1))
        print("aucoding after padding: ", aucoding.shape)
        current_file = loadmat(file_name)
        current_file[str(AU_key)] = aucoding
        savemat(file_name, current_file, format='7.3', oned_as='row', truncate_existing=True)
        # if AU is in AU_int_num, get the AU_INT
        if au in AU_int_num:
            AU_INT_url = os.path.join(AU_url, "AU_INT", AU_key, subject + '_' + task + '_' + AU_key + '.csv')
            # read in each csv file
            AUs_int = pd.read_csv(AU_INT_url, header = None).to_numpy()
            print("AUs_int.shape: ", AUs_int.shape)
            assert (AUs_int.shape[0] == AUs.shape[0])
            print("AUs_int start frame: ", AUs_int[0,0])
            aucoding_int = AUs_int[:, 1]
            if start_frame > 1:
                # pad the previous frame with -1
                aucoding_int = np.pad(aucoding_int, (start_frame - 1, 0), 'constant', constant_values = (-1, -1))
            if end_frame < frame_shape:
                # pad the following frame with -1
                aucoding_int = np.pad(aucoding_int, (0, frame_shape - end_frame), 'constant', constant_values = (-1, -1))
            print("aucoding_int after padding: ", aucoding_int.shape)
            AU_int_key = AU_key + 'int'
            current_file = loadmat(file_name)
            current_file[str(AU_int_key)] = aucoding_int
            savemat(file_name, current_file, format = '7.3', oned_as = 'row', truncate_existing=True)
    # return start_frame, end_frame
​
​
def process_images(input_folder, input_subfolder, output_folder, subject, task, dim,  
    targets_to_scale=["ppg", "resp", "ecg", "abp"], plot=False, skip_existing=False, normalize_frames=True):
​
    # get list of frame files
    phys_url = os.path.join(input_folder, "Physiology", subject, task)
    file_name = os.path.join(output_folder, "{0:s}{1:s}.mat".format(subject, task))
    # if the file exists and the skip_existing flag is set, skip file
    if os.path.exists(file_name) and skip_existing:
        print("{} exists and skip_existing flag is set. Skipping...".format(file_name))
        return
​
    # for each frame
    imgzip = open(os.path.join(input_folder, '2D+3D', subject+'.zip'))
    zipfile_path = os.path.join(input_folder, '2D+3D', subject+'.zip')
    print(zipfile_path)
    print('Current time after upzipping the file: ', datetime.now())
    cnt = 0
    with zipfile.ZipFile(zipfile_path, "r") as zippedImgs:
        for ele in zippedImgs.namelist():
            ext = os.path.splitext(ele)[-1]
            ele_task = str(ele).split('/')[1]
            if ext == '.jpg' and ele_task == task:
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
    # read in ground truth waveforms
    ppg_wave, resp_wave, HR_bpm, resp_bpm, bp_dia, mean_BP, systolic_BP, eda = read_labels(phys_url, len(Xsub))
​
    # calculate differences between consecutive frames
    # dXsub = np.diff(Xsub, axis=0) / (Xsub[:-1]+Xsub[1:])
​
    # if normalize_frames:
    #     # # normalize frame differences
    #     dXsub = dXsub / np.std(dXsub)
​
    #     # # normalize appearance branch
    #     Xsub = (Xsub - Xsub.mean()) / Xsub.std()
​
    # # concatenate frames at each time step as additional channels
    # dXsub = np.concatenate((dXsub, Xsub[:-1]), axis=-1)
    
    # get the final shape
    final_shape = Xsub.shape[0]
    print("Final shape:", Xsub.shape)
​
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].imshow(Xsub[0, :, :, 0:3])
        ax[1].imshow(Xsub[:, :, :, 3:].mean(axis=0))
        plt.show()
​
    # return final_shape, Xsub, ppg_wave, resp_wave, HR_bpm, resp_bpm, bp_dia, mean_BP, systolic_BP, eda
    Xsub, ppg_wave, resp_wave, HR_bpm, resp_bpm, bp_dia, mean_BP, systolic_BP, eda
    savemat(file_name, {'X': Xsub, 'bp_wave': ppg_wave, 'resp_wave': resp_wave, 'HR_bpm': HR_bpm, 'resp_bpm': resp_bpm, 
                        'diastolic_bp': bp_dia, 'mean_bp': mean_BP, 'systolic_bp': systolic_BP, 'eda': eda}, format='7.3', oned_as='row', truncate_existing=True)    
    return final_shape
    
​
​
def main():
    print("Starting")
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str,
                    default='C:\Data\ippg/', help='Location of the raw input data')
    parser.add_argument('-g', '--gender', type=str,
                    default='female', help='Location of the raw input data')
    parser.add_argument('-o', '--output_dir', type=str,
                    default='C:\Data\ippg/', help='Location for saving the preprocessed data')
    parser.add_argument('-s', '--sets', nargs='+',
                    default=[None], help='Sets to use (eg. 320x240_V006')
    parser.add_argument('-img', '--img_size', type=int, default=256, help='img_size')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--normalize_frames', action='store_true')
    parser.add_argument('--participants_div', type=int, default = 0, help ='divide the participants into differnet groups')
    args = parser.parse_args()
​
    input_folder = args.input_dir
    output_folder = args.output_dir
    sets = args.sets
    dim = args.img_size
    skip_existing = args.skip_existing
    normalize_frames = args.normalize_frames
    participants_div = args.participants_div
    os.makedirs(output_folder, exist_ok=True)
​
    # bp4d = np.load('bp4d.npy')
    # print(bp4d)
    
    AU_url = os.path.join(input_folder, "AUCoding")
    # the tasks that contains AU data
    AU_task = ['T1', 'T6', 'T7', 'T8']
    # Action Units
    AU_num = [1, 2, 4, 5, 6, 7, 9, 10, 11, 
              12, 13, 14, 15, 16, 17, 18, 19, 20,
              22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    
​
    for s in sets:
        if s:
            glob_string = os.path.join(input_folder, s, "Physiology/*")
        else:
            glob_string = os.path.join(input_folder, "Physiology/*")
        
        orig_subjects = [os.path.basename(x) for x in glob.glob(glob_string)]
        # if flag set, shuffle order of subjects (useful for parallel processing)
        if args.shuffle:
            np.random.shuffle(orig_subjects)
        print("Subjects:", orig_subjects)
​
​
        # divide the participants for multi-preprocessing:
        if participants_div == 1:
            subjects = orig_subjects[0:46]
        elif participants_div == 2:
            subjects = orig_subjects[46:92]
        elif participants_div == 3:
            subjects = orig_subjects[92:]
        else:
            subjects = orig_subjects
        print(subjects)
​
        count = 0
​
        for subject in subjects:
            # participant that missing data
            if subject == 'F082':
                continue
            
            # if count < 1:
            #     count = count + 1
            print('Subject', subject)
            print('===========================')
            # processing time
            start_time = datetime.now()
            
            
            ​
            tasks = ['T1', 'T2', 'T3', 'T4', 'T5','T6', 'T7', 'T8', 'T9', 'T10']
​
​
            for task in tasks:
                subj_task_id = subject + task
                print(subj_task_id)
​
                #process the image data
                final_shape = process_images(input_folder, s, output_folder, subject, task, dim, skip_existing=skip_existing, normalize_frames=normalize_frames)
 
                #participant that missing data
                if subj_task_id == 'F041T7':
                    continue
​
                # if the task is T1, T6, T7, T8, read in AU_OCC
                if task in AU_task:
                    # process the AU data
                    read_AU(output_folder, subject, task, AU_url, AU_num, final_shape)
            
            end_time = datetime.now()
            print('Processing Time: {}'.format(end_time - start_time))
​
​
if __name__ == "__main__":
    main()