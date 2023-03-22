"""Unsupervised learning methods including POS, GREEN, CHROME, ICA, LGI and PBV."""

# TODO This was adapted for BP4D w/ diffnormed frames - need to change this back eventually

import logging
import os
from collections import OrderedDict

import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader
from metrics.metrics import calculate_metrics
from signal_methods.methods.CHROME_DEHAAN import *
from signal_methods.methods.GREEN import *
from signal_methods.methods.ICA_POH import *
from signal_methods.methods.LGI import *
from signal_methods.methods.PBV import *
from signal_methods.methods.POS_WANG import *
from tqdm import tqdm
from utils.utils import *


def signal_predict(config, data_loader, method_name):
    """ Model evaluation on the testing dataset."""
    if data_loader["signal"] is None:
        raise ValueError("No data for signal method predicting")
    print("===Signal Method ( " + method_name + " ) Predicting ===")
    predict_hr_peak_all = []
    gt_hr_peak_all = []
    predict_hr_fft_all = []
    gt_hr_fft_all = []
    sbar = tqdm(data_loader["signal"], ncols=80)
    for _, test_batch in enumerate(sbar):
        batch_size = test_batch[0].shape[0]
        for idx in range(batch_size):
            data_input, labels_input = test_batch[0][idx].cpu().numpy(), test_batch[1][idx].cpu().numpy()

            data_input = data_input[:,:,:,3:6] # TODO - only raw frames
            # labels_input = np.cumsum(labels_input)

            if method_name == "POS":
                BVP = POS_WANG(data_input, config.DATA.SIGNAL.FS)
            elif method_name == "CHROM":
                BVP = CHROME_DEHAAN(data_input, config.DATA.SIGNAL.FS)
            elif method_name == "ICS":
                BVP = ICA_POH(data_input, config.DATA.SIGNAL.FS)
            elif method_name == "GREEN":
                BVP = GREEN(data_input)
            elif method_name == "LGI":
                BVP = LGI(data_input)
            elif (method_name == "PBV"):
                BVP = PBV(data_input)
            else:
                raise ValueError("signal method name wrong!")

            # TODO Girish
            
            labels_input = labels_input[:,0]
            labels_input = labels_input[0:-1]
            BVP = np.diff(BVP, axis=0)

            # plt.figure()
            # plt.plot(BVP)
            # plt.savefig('/gscratch/ubicomp/girishvn/rppg/multimodal_physiological_sensing/BVPTEST.png')

            # BVP = np.diff(BVP, axis=0)

            # plt.figure()
            # plt.plot(BVP)
            # plt.savefig('/gscratch/ubicomp/girishvn/rppg/multimodal_physiological_sensing/BVPTEST2.png')

            # plt.figure()
            # plt.plot(labels_input)
            # plt.savefig('/gscratch/ubicomp/girishvn/rppg/multimodal_physiological_sensing/labels_input.png')

            # print('')
            # print(len(BVP))
            # print('')

            # raise ValueError('KILL')


            if config.SIGNAL_SPECS.EVALUATION_METHOD == "peak detection":
                gt_hr, pre_hr = calculate_metric_per_video(BVP, labels_input, diff_flag=True,fs=config.DATA.SIGNAL.FS)
                predict_hr_peak_all.append(pre_hr)
                gt_hr_peak_all.append(gt_hr)
            if config.SIGNAL_SPECS.EVALUATION_METHOD == "FFT":
                gt_fft_hr, pre_fft_hr = calculate_metric_per_video(BVP, labels_input, diff_flag=True,fs=config.DATA.SIGNAL.FS)
                predict_hr_fft_all.append(pre_fft_hr)
                gt_hr_fft_all.append(gt_fft_hr)

    print("Used Signal Method: " + method_name)
    if config.SIGNAL_SPECS.EVALUATION_METHOD == "peak detection":
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        for metric in config.SIGNAL_SPECS.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                print("Peak MAE (Peak Label):{0}".format(MAE_PEAK))
            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(
                    np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                print("PEAK RMSE (Peak Label):{0}".format(RMSE_PEAK))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(
                    np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                print("PEAK MAPE (Peak Label):{0}".format(MAPE_PEAK))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                print("PEAK Pearson  (Peak Label):{0}".format(Pearson_PEAK[0][1]))
            else:
                raise ValueError("Wrong Test Metric Type")
    if config.SIGNAL_SPECS.EVALUATION_METHOD == "FFT":
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        for metric in config.SIGNAL_SPECS.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                print("FFT MAE (FFT Label):{0}".format(MAE_PEAK))
            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(
                    np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                print("FFT RMSE (FFT Label):{0}".format(RMSE_PEAK))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(
                    np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                print("FFT MAPE (FFT Label):{0}".format(MAPE_PEAK))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                print("FFT Pearson  (FFT Label):{0}".format(Pearson_PEAK[0][1]))
            else:
                raise ValueError("Wrong Test Metric Type")


def process_video(frames):
    # Standard:
    RGB = []
    for frame in frames:
        sum = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(sum / (frame.shape[0] * frame.shape[1]))
    RGB = np.asarray(RGB)
    RGB = RGB.transpose(1, 0).reshape(1, 3, -1)
    return np.asarray(RGB)
