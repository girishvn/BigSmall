import torch
from neural_methods.model.BigSmall import BigSmallV1, BigSmallV2, BigSmallV3
from neural_methods.model.BigSmall import BigSmallV2AttnV1, BigSmallV2AttnV2, BigSmallV3AttnV1
from neural_methods.model.Final_Models import BigSmallSlowFastWTSM
from neural_methods.trainer.BaseTrainer import BaseTrainer

import logging
import os
from collections import OrderedDict

import torch
import torch.optim as optim

import numpy as np
import pandas as pd
import csv
import pickle



import matplotlib.pyplot as plt

def format_data_shape(data):
    # reshape big data
    data_big = data[0]
    print(data_big.shape)

    D, C, H, W = data_big.shape
    data_big = data_big.view(1 * D, C, H, W)

    # reshape small data
    data_small = data[1]
    D, C, H, W = data_small.shape
    data_small = data_small.view(1 * D, C, H, W)

    data[0] = data_big
    data[1] = data_small

    return data


def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    return new_state_dict


# Plot Attn Mask 
def plot_attn_and_input(input_img, attn_mask_big, attn_mask_small):

    # print('INPUT SHAPE') # shape should be [N, 3, H, W]
    # print(input_img.shape)
    # print('')
    # print('MASK SHAPE')
    # print(attn_mask_out.shape) # shape should be [N, 1, H, W]
    # print('')

    in2plot = input_img[0,:,:,:]
    in2plot = in2plot.to('cpu')
    in2plot = in2plot.detach().numpy()
    in2plot = np.moveaxis(in2plot, [0], [2])
    
    attn2plot = attn_mask_big[0,:,:,:]
    attn2plot = attn2plot.to('cpu')
    attn2plot = attn2plot.detach().numpy()
    attn2plot = np.squeeze(attn2plot)
    attn2plot = np.sum(attn2plot, axis=0)

    attn3plot = attn_mask_small[0,:,:,:]
    attn3plot = attn3plot.to('cpu')
    attn3plot = attn3plot.detach().numpy()
    attn3plot = np.squeeze(attn3plot)
    attn3plot = np.nan_to_num(attn3plot)
    attn3plot = np.sum(attn3plot, axis=0)

    plt.figure()
    plt.imshow(in2plot)
    # plt.title('Big RawStd Input')
    plt.savefig('/gscratch/ubicomp/girishvn/rppg/multimodal_physiological_sensing/input.png')

    plt.figure()
    plt.imshow(attn2plot)
    # plt.colorbar()
    plt.grid(False)
    plt.axis('off')
    plt.savefig('/gscratch/ubicomp/girishvn/rppg/multimodal_physiological_sensing/big_attnmask.png')
    
    plt.figure()
    plt.imshow(attn3plot)
    # plt.title('Small Attention Mask Output')
    # plt.colorbar()
    plt.grid(False)
    plt.axis('off')
    plt.savefig('/gscratch/ubicomp/girishvn/rppg/multimodal_physiological_sensing/small_attnmask.png')

    print('PLOTTED ATTENTION MASKS')


# Dataset to Get Datafrom
file_list_path = '/gscratch/ubicomp/girishvn/rppg/rppg_datasets/PreprocessedData/DataFileLists/BP4D_Clip1_AUSubset_3Fold/BP4D_Big144RawStd_Small18DiffNorm_LabelDiffNorm_ClipLength1_AU_Subset_Split1_Test.csv'
# model_path = './PreTrainedModels/BP4D_3FoldCrossVal_LastEpoch/BP4D_BigSmallV2AttnV1Multitask_Split1_Epoch4.pth'

# file_list_path = '/gscratch/ubicomp/girishvn/rppg/rppg_datasets/PreprocessedData/DataFileLists/BP4D_Clip1_AUSubset_3Fold/BP4D_Big144RawStd_Small9DiffNorm_LabelDiffNorm_ClipLength1_AU_Subset_Split1_Test.csv'
# model_path = './PreTrainedModels/BP4D_3FoldCrossVal_LastEpoch/BP4D_BigSmallV3AttnV1Multitask_Split1_Epoch4.pth'

# BP4D Full AU Subset. PURE/UBFC TEST (used 0/2500 for UBFC, 0/20000 for PURE)
#file_list_path = '/gscratch/ubicomp/girishvn/rppg/rppg_datasets/PreprocessedData/DataFileLists/PURE_Big144RawStd_Small9DiffNorm_LabelDiffNorm_ClipLength1_0.0_1.0.csv'
# file_list_path = '/gscratch/ubicomp/girishvn/rppg/rppg_datasets/PreprocessedData/DataFileLists/UBFC_Big144RawStd_Small9DiffNorm_LabelDiffNorm_ClipLength1_0.0_1.0.csv'
#model_path = "./PreTrainedModels/BP4D_Big144RawStd_Small9DiffNorm_LabelDiffNorm_ClipLength1_AU_Subset/BP4DFullAUSubset_BigSmallV3AttnV1Multitask_Epoch4.pth"




# previously plotted (0, 2500, 5000, 7500)
# F001T6_input95, 
# F002T6_input295
# F004T8_input164
# F006T6_input265
# M005T8_input57


#file2plot = 50000
# file2plot = 50000
# # Get Data File
# df = pd.read_csv(file_list_path)
# fname = df['input_files'][file2plot]

model_path = './PreTrainedModels/FinalModels/BP4D_BSSF_WTSM_SummedFeatsSinglePool_Split1_Epoch4.pth'
file2plot = 100 #6000 #5155 #400 #1000
trial_arr = ['F001T6_input31.pickle', 'F002T6_input98.pickle', 'F004T8_input54.pickle', 'F006T6_input88.pickle', 'M005T8_input19.pickle']
base_path = '/gscratch/ubicomp/girishvn/rppg/rppg_datasets/PreprocessedData/BP4D_Big144RawStd_Small9DiffNorm_LabelDiffNorm_ClipLength3_AU_Subset'
fname = os.path.join(base_path, trial_arr[4])
# file_list_path = os.path.join('/gscratch/ubicomp/girishvn/rppg/rppg_datasets/PreprocessedData/DataFileLists/BP4D_Clip3_AUSubset_3Fold', 'BP4D_Big144RawStd_Small9DiffNorm_LabelDiffNorm_ClipLength3_AUSubset_Rand_Split1_Train.csv')
# df = pd.read_csv(file_list_path)
# fname = df['input_files'][file2plot]

print('')
print(fname)
print('')

with open(fname, 'rb') as handle:
    d = pickle.load(handle)
print(d.keys())

# d = format_data_shape(d)

big = d[0]
big = np.moveaxis(big, [3], [1])
big = torch.from_numpy(big)
big = big.double()
d[0] = big

small = d[1]
small = np.moveaxis(small, [3], [1])
small = torch.from_numpy(small)
small = small.double()
d[1] = small

# Define Model
#model = BigSmallV2AttnV1(out_size=13)
# model = BigSmallV3AttnV1(out_size=13)
model  = BigSmallSlowFastWTSM(out_size=14, n_segment=3)

old_state_dict = torch.load(model_path)
new_state_dict = remove_data_parallel(old_state_dict)
model.load_state_dict(new_state_dict)
model = model.double()

model.eval()
with torch.no_grad():
    out1, out2, out3, battn, sattn = model(d)


# Plot Attn:
plot_attn_and_input(d[0], battn, sattn)

print('DONE')
