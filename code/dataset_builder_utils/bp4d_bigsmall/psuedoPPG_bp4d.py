import os 
import glob
import mat73
import hdf5storage as hdf5
import numpy as np
import pickle

import math

import numpy as np
from scipy import signal
from scipy import sparse

from tqdm import tqdm

# Signal Prococessing Methods
def process_video(frames):
    """Calculates the average value of each frame."""
    RGB = []
    for frame in frames:
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(summation / (frame.shape[0] * frame.shape[1]))
    return np.asarray(RGB)


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


def POS_WANG(frames, fs):
    WinSec = 1.6
    RGB = process_video(frames)
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



#####################################
########### THINGS TO SET ###########
#####################################

# TODO SET THIS RAW DATASET PATH:
data_path = '/gscratch/ubicomp/girishvn/datasets/BP4D_plus/BP4DPlus_FullRawDataset_144x144'



############################
########### MAIN ###########
############################
print(data_path)
print()

data_list = glob.glob(os.path.join(data_path, '*.mat'))
print('Num of files to process', len(data_list))
print('')

hr_avg_mae = 0
count = 0
fs = 25 # bp4d sampling rate: 25hz

for fpath in tqdm(data_list):

    data = hdf5.loadmat(fpath)
    x = data['X'] # read in video frames

    bvp = POS_WANG(x, fs)
    bvp = np.array(bvp)

    # get avg h freq from GT label
    hr_arr = data['HR_bpm']
    avg_hr_bpm = np.sum(hr_arr)/len(hr_arr)
    hr_freq = avg_hr_bpm / 60 # divide beats per min by 60, to get beats pers
    halfband = 20 / fs # bandwith to account for HR variation (accounts for 40 bpm window)

    # Bound filter
    min_freq = hr_freq - halfband
    if min_freq < 0.70:
        min_freq = 0.70
    max_freq = hr_freq + halfband
    if max_freq > 3:
        max_freq = 3

    # HR filt - more aggressive filtering
    b, a = signal.butter(2, [(min_freq) / fs * 2, (max_freq) / fs * 2], btype='bandpass')
    pos_bvp = signal.filtfilt(b, a, bvp.astype(np.double))

    # envelope normalization
    analytic_signal = signal.hilbert(pos_bvp)
    amplitude_envelope = np.abs(analytic_signal)
    env_norm_bvp = pos_bvp/amplitude_envelope

    # Add New Fields to Data Mat File
    data['pos_bvp'] = pos_bvp
    data['pos_env_norm_bvp'] = env_norm_bvp
    hdf5.savemat(fpath, data)

    count += 1

print('DONE')
print('Num processed files:', count)

