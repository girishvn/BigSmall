"""The post processing files for caluclating heart rate using FFT or peak detection.
The file also  includes helper funcs such as detrend, mag2db etc.
"""

import numpy as np
import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags


# %% Helper Function
def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal


def mag2db(mag):
    return 20. * np.log10(mag)


def calculate_HR(pxx_pred, frange_pred, fmask_pred, pxx_label, frange_label, fmask_label):
    pred_HR = np.take(frange_pred, np.argmax(
        np.take(pxx_pred, fmask_pred), 0))[0] * 60
    ground_truth_HR = np.take(frange_label, np.argmax(
        np.take(pxx_label, fmask_label), 0))[0] * 60
    return pred_HR, ground_truth_HR


def calculate_SNR(pxx_pred, f_pred, currHR, signal):
    currHR = currHR / 60
    f = f_pred
    pxx = pxx_pred
    gtmask1 = (f >= currHR - 0.1) & (f <= currHR + 0.1)
    gtmask2 = (f >= currHR * 2 - 0.1) & (f <= currHR * 2 + 0.1)
    sPower = np.sum(np.take(pxx, np.where(gtmask1 | gtmask2)))
    if signal == 'pulse':
        fmask2 = (f >= 0.75) & (f <= 4)
    else:
        fmask2 = (f >= 0.08) & (f <= 0.5)
    allPower = np.sum(np.take(pxx, np.where(fmask2 == True)))
    SNR_temp = mag2db(sPower / (allPower - sPower))
    return SNR_temp


# %%  Processing


def calculate_metric_peak_per_video(predictions, labels, diff_flag=True, signal='pulse', window_size=360, fs=30, bpFlag=True):

    if signal == 'pulse':
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')  # 2.5 -> 1.7
    elif signal == 'resp':
        [b, a] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
    else:
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')

    data_len = len(predictions)
    HR_pred = []
    HR0_pred = []
    all_peaks = []
    all_peaks0 = []
    pred_signal = []
    label_signal = []
    window_size = data_len
    for j in range(0, data_len, window_size):
        if j == 0 and (j + window_size) > data_len:
            pred_window = predictions
            label_window = labels
        elif (j + window_size) > data_len:
            break
        else:
            pred_window = predictions[j:j + window_size]
            label_window = labels[j:j + window_size]
        if diff_flag:
            if signal == 'pulse' or 'resp':
                pred_window = detrend(np.cumsum(pred_window), 100)
                label_window = detrend(np.cumsum(label_window), 100)
            else:
                pred_window = np.cumsum(pred_window)
        else:
            if signal == 'pulse' or 'resp':
                pred_window = detrend(pred_window, 100)
                label_window = detrend(label_window, 100)
            else:
                pred_window = pred_window

        if bpFlag:
            pred_window = scipy.signal.filtfilt(b, a, pred_window)
            label_window = scipy.signal.filtfilt(b, a, label_window)

        # Peak detection
        labels_peaks, _ = scipy.signal.find_peaks(label_window)
        preds_peaks, _ = scipy.signal.find_peaks(pred_window)

        temp_HR_0 = 60 / (np.mean(np.diff(labels_peaks)) / fs)
        temp_HR = 60 / (np.mean(np.diff(preds_peaks)) / fs)

        HR_pred.append(temp_HR)
        HR0_pred.append(temp_HR_0)
        all_peaks.extend(preds_peaks + j)
        all_peaks0.extend(labels_peaks + j)
        pred_signal.extend(pred_window.tolist())
        label_signal.extend(label_window.tolist())

    HR = np.mean(np.array(HR_pred))
    HR0 = np.mean(np.array(HR0_pred))

    return HR0, HR


def calculate_metric_per_video(predictions, labels, diff_flag=True, signal='pulse', fs=30, bpFlag=True):

    if signal == 'pulse':
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')  # 2.5 -> 1.7
    elif signal == 'resp':
        [b, a] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
    else:
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')

    if diff_flag:
        if signal == 'pulse' or 'resp':
            pred_window = detrend(np.cumsum(predictions), 100)
            label_window = detrend(np.cumsum(labels), 100)
        else:
            pred_window = np.cumsum(predictions)
    else:
        if signal == 'pulse' or 'resp':
            pred_window = detrend(predictions, 100)
            label_window = detrend(labels, 100)
        else:
            pred_window = predictions

    if bpFlag:
        pred_window = scipy.signal.filtfilt(b, a, np.double(pred_window))
        label_window = scipy.signal.filtfilt(b, a, np.double(label_window))

    pred_window = np.expand_dims(pred_window, 0)
    label_window = np.expand_dims(label_window, 0)
    # Predictions FFT
    N = next_power_of_2(pred_window.shape[1])
    f_prd, pxx_pred = scipy.signal.periodogram(
        pred_window, fs=fs, nfft=N, detrend=False)
    if signal == 'pulse' or 'resp':
        # regular Heart beat are 0.75*60 and 2.5*60
        fmask_pred = np.argwhere((f_prd >= 0.75) & (f_prd <= 2.5))
    else:
        # regular Heart beat are 0.75*60 and 2.5*60
        fmask_pred = np.argwhere((f_prd >= 0.08) & (f_prd <= 0.5))
    pred_window = np.take(f_prd, fmask_pred)
    # Labels FFT
    f_label, pxx_label = scipy.signal.periodogram(
        label_window, fs=fs, nfft=N, detrend=False)
    if signal == 'pulse' or 'resp':
        # regular Heart beat are 0.75*60 and 2.5*60
        fmask_label = np.argwhere((f_label >= 0.75) & (f_label <= 2.5))
    else:
        # regular Heart beat are 0.75*60 and 2.5*60
        fmask_label = np.argwhere((f_label >= 0.08) & (f_label <= 0.5))
    label_window = np.take(f_label, fmask_label)

    # MAE
    temp_HR, temp_HR_0 = calculate_HR(
        pxx_pred, pred_window, fmask_pred, pxx_label, label_window, fmask_label)
    return temp_HR_0, temp_HR


def calculate_metric(predictions, labels, signal='pulse', window_size=360, fs=30, bpFlag=True):
# def calculate_metric(predictions, labels, signal='resp', window_size=360, fs=30, bpFlag=True):

    if signal == 'pulse':
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')  # 2.5 -> 1.7
    elif signal == 'resp':
        [b, a] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
    else:
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')

    data_len = len(predictions)
    HR_pred = []
    HR0_pred = []
    mySNR = []
    for j in range(0, data_len, window_size):
        if j == 0 and (j + window_size) > data_len:
            pred_window = predictions
            label_window = labels
        elif (j + window_size) > data_len:
            break
        else:
            pred_window = predictions[j:j + window_size]
            label_window = labels[j:j + window_size]
        if signal == 'pulse' or 'resp':
            pred_window = detrend(np.cumsum(pred_window), 100)
        else:
            pred_window = np.cumsum(pred_window)

        label_window = np.squeeze(label_window)
        if bpFlag:
            pred_window = scipy.signal.filtfilt(b, a, np.double(pred_window))
            label_window = scipy.signal.filtfilt(b, a, np.double(label_window))

        pred_window = np.expand_dims(pred_window, 0)
        label_window = np.expand_dims(label_window, 0)
        # Predictions FFT
        f_prd, pxx_pred = scipy.signal.periodogram(
            pred_window, fs=fs, nfft=4 * window_size, detrend=False)
        if signal == 'pulse' or 'resp':
            # regular Heart beat are 0.75*60 and 2.5*60
            fmask_pred = np.argwhere((f_prd >= 0.75) & (f_prd <= 2.5))
        else:
            # regular Heart beat are 0.75*60 and 2.5*60
            fmask_pred = np.argwhere((f_prd >= 0.08) & (f_prd <= 0.5))
        pred_window = np.take(f_prd, fmask_pred)
        # Labels FFT
        f_label, pxx_label = scipy.signal.periodogram(
            label_window, fs=fs, nfft=4 * window_size, detrend=False)
        if signal == 'pulse' or 'resp':
            # regular Heart beat are 0.75*60 and 2.5*60
            fmask_label = np.argwhere((f_label >= 0.75) & (f_label <= 2.5))
        else:
            # regular Heart beat are 0.75*60 and 2.5*60
            fmask_label = np.argwhere((f_label >= 0.08) & (f_label <= 0.5))
        label_window = np.take(f_label, fmask_label)

        # MAE
        temp_HR, temp_HR_0 = calculate_HR(
            pxx_pred, pred_window, fmask_pred, pxx_label, label_window, fmask_label)
        temp_SNR = calculate_SNR(pxx_pred, f_prd, temp_HR_0, signal)
        HR_pred.append(temp_HR)
        HR0_pred.append(temp_HR_0)
        mySNR.append(temp_SNR)

    HR = np.array(HR_pred)
    HR0 = np.array(HR0_pred)
    mySNR = np.array(mySNR)

    MAE = np.mean(np.abs(HR - HR0))
    RMSE = np.sqrt(np.mean(np.square(HR - HR0)))
    meanSNR = np.nanmean(mySNR)
    return MAE, RMSE, meanSNR, HR0, HR