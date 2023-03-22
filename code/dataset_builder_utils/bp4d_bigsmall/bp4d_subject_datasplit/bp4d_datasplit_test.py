import numpy as np
import csv 
import glob
import os
import pandas as pd
import random

def make_subj_split_flist(split, data_path, out_path):
    
    split_flist = []
    for subj in split:
        flist = glob.glob(os.path.join(data_path, '*' + subj + 'T*.pickle'))
        split_flist += flist

    file_list_df = pd.DataFrame(split_flist, columns = ['input_files'])   
    file_list_df.to_csv(out_path)

    print('Files in split:', len(split_flist))
    return len(split_flist)


def get_subj_list_from_file(fpath):

    subj_list = []
    # read in csv file 
    df = pd.read_csv(fpath)
    for i, row in df.iterrows():
        f = row['subjects']
        subj_list.append(f)

    print('Subjects in split:', len(subj_list))
    subj_list.sort()
    return subj_list



##########################################################
#################### CONFIGS TO CHANGE ###################
##########################################################

# TODO SET THIS DATA PATH TO PREPROCESSED DATA
data_path = '/gscratch/ubicomp/girishvn/rppg/rppg_datasets/PreprocessedData/BP4DPlus_Big144RawStd_Small9DiffNorm_ClipLen3_AUSubset'

# TODO SET THIS PATH TO THE BASE PATH TO SAVE DATA SPLIT FILE LISTS
flist_basepath = '/gscratch/ubicomp/girishvn/rppg/rppg_datasets/PreprocessedData/DataFileLists/BP4DPlus_Big144RawStd_Small9DiffNorm_ClipLen3_AUSubset_3Fold'



##########################################################
########################### MAIN #########################
##########################################################

# grab splits from saved csv file
split1 = get_subj_list_from_file("./dataset_builder_utils/bp4d_bigsmall/bp4d_subject_datasplit/" + "Split1_Test_Subjects.csv")
split2 = get_subj_list_from_file("./dataset_builder_utils/bp4d_bigsmall/bp4d_subject_datasplit/" + "Split2_Test_Subjects.csv")
split3 = get_subj_list_from_file("./dataset_builder_utils/bp4d_bigsmall/bp4d_subject_datasplit/" + "Split3_Test_Subjects.csv")

# CHANGE THINGS HERE
csv_path1 = os.path.join(flist_basepath, 'BP4D_Big144RawStd_Small9DiffNorm_LabelDiffNorm_ClipLen3_AUSubset_Rand_Split1_Test.csv')
make_subj_split_flist(split1, data_path, csv_path1)
print('')
csv_path2 = os.path.join(flist_basepath, 'BP4D_Big144RawStd_Small9DiffNorm_LabelDiffNorm_ClipLen3_AUSubset_Rand_Split2_Test.csv')
make_subj_split_flist(split2, data_path, csv_path2)
print('')
csv_path3 = os.path.join(flist_basepath, 'BP4D_Big144RawStd_Small9DiffNorm_LabelDiffNorm_ClipLen3_AUSubset_Rand_Split3_Test.csv')
make_subj_split_flist(split3, data_path, csv_path3)




