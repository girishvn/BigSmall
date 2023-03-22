import numpy as np
import csv 
import glob
import os
import pandas as pd



def combine_to_make_train(df1_path, df2_path, out_path):
    
    # read in df1
    df1 = pd.read_csv(df1_path)
    split1 = df1['input_files'].to_list()
    print('d1 len', len(split1))

    # read in df2
    df2 = pd.read_csv(df2_path)
    split2 = df2['input_files'].to_list()
    print('d2 len', len(split2))

    split_flist = split1 + split2
    print('comb len', len(split_flist))
    print('')

    file_list_df = pd.DataFrame(split_flist, columns = ['input_files'])   
    file_list_df.to_csv(out_path)
    print('Files in split:', len(split_flist))
    print('')
    return len(split_flist)




##########################################################
#################### CONFIGS TO CHANGE ###################
##########################################################

# TODO SET THESE PATHS TO MATCH THOSE CONFIGURED IN bp4d_datasplit_test.py
flist_basepath = '/gscratch/ubicomp/girishvn/rppg/rppg_datasets/PreprocessedData/DataFileLists/BP4D_BigSmall_VariousClipLen_AUSubset_3Fold/'
split1_test_path = os.path.join(flist_basepath, 'BP4D_Big144RawStd_Small9DiffNorm_LabelDiffNorm_ClipLen3_AUSubset_Rand_Split1_Test.csv')
split1_train_path = os.path.join(flist_basepath,'BP4D_Big144RawStd_Small9DiffNorm_LabelDiffNorm_ClipLen3_AUSubset_Rand_Split1_Train.csv')
split2_test_path = os.path.join(flist_basepath,'BP4D_Big144RawStd_Small9DiffNorm_LabelDiffNorm_ClipLen3_AUSubset_Rand_Split2_Test.csv')
split2_train_path = os.path.join(flist_basepath,'BP4D_Big144RawStd_Small9DiffNorm_LabelDiffNorm_ClipLen3_AUSubset_Rand_Split2_Train.csv')
split3_test_path = os.path.join(flist_basepath,'BP4D_Big144RawStd_Small9DiffNorm_LabelDiffNorm_ClipLen3_AUSubset_Rand_Split3_Test.csv')
split3_train_path = os.path.join(flist_basepath,'BP4D_Big144RawStd_Small9DiffNorm_LabelDiffNorm_ClipLen3_AUSubset_Rand_Split3_Train.csv')



##########################################################
########################### MAIN #########################
##########################################################

combine_to_make_train(split2_test_path, split3_test_path, split1_train_path)
combine_to_make_train(split1_test_path, split3_test_path, split2_train_path)
combine_to_make_train(split1_test_path, split2_test_path, split3_train_path)


