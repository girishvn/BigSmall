# NOTE: This was not how flops and parameters were counted for the paper. 
# Those were done using the flopth: https://github.com/vra/flopth

import torch
import torch.nn as nn
from torchsummary import summary
from collections import OrderedDict

from neural_methods.model.BigSmallBaselines import BigSmallBaseModel, SmallPathwayBaseModel, BigPathwayBaseModel
from neural_methods.model.BigSmallSlowFast import BigSmallSlowFast, BigSmallSlowFastTSM, BigSmallSlowFastWTSM, BigSmallSlowFastWTSM_SummedFeatMaps

def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    return new_state_dict

# BigSmall SlowFast w/ WTSM
# model_path = './PreTrainedModels/BP4D_Big144RawStd_Small9DiffNorm_LabelDiffNorm_ClipLength3_AU_Subset/BP4D_BSSF_WTSM_Clip3_Split1_Epoch2.pth'
# model = BigSmallSlowFastWTSM(out_size=13, n_segment=3)

print('')
print('======================')
print('======= MODEL ========')
print('======================')
print(model)

print('')
print('=================================')
print('======= Params By Layer ========')
print('=================================')
running_total = 0
for name, param in model.named_parameters():
    running_total += param.numel()
    print(name, param.numel())

print('')
print('=================================')
print('========= Total Params ==========')
print('=================================')
total_params = sum(param.numel() for param in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Total Params: ', total_params)
print('Trainable Params: ', trainable_params)



