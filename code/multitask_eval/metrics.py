import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

import multitask_eval.signal_metrics as signal_metrics 



# TODO CAN I GET RID OF THIS??? CANT FIND WHERE IT IS USED
def read_label(data):
    df = pd.read_csv("label/{0}_Comparison.csv".format(data))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


# TODO CAN I GET RID OF THIS??? CANT FIND WHERE IT IS USED
def read_hr_label(feed_dict, index):
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def reform_data_from_dict(data):
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)
    return np.reshape(sort_data.cpu(), (-1))


def calculate_bvp_metrics(predictions, labels, config):

    metric_dict = dict()

    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()

    for index in predictions.keys():
        #prediction = reform_data_from_dict(predictions[index])
        #label = reform_data_from_dict(labels[index])

        prediction = predictions[index]
        label = labels[index]

        if config.DATA.TRAIN.PREPROCESS.LABEL_TYPE == "Standardized" or config.DATA.TRAIN.PREPROCESS.LABEL_TYPE == "Raw":
            diff_flag_test = False
        elif config.DATA.TRAIN.PREPROCESS.LABEL_TYPE == "Normalized":
            diff_flag_test = True
        else:
            raise ValueError("Not supported label type in testing!")

        # GET FUNDEMENTAL HR FREQUENCY
        gt_hr_fft, pred_hr_fft = signal_metrics.calc_fundementalfreq_fft(
            prediction, label, signal='pulse', diff_flag=diff_flag_test, fs=config.DATA.TEST.FS)
        gt_hr_peak, pred_hr_peak = signal_metrics.calc_fundementalfreq_peak(
            prediction, label, signal='pulse', diff_flag=diff_flag_test, fs=config.DATA.TEST.FS)

        gt_hr_fft_all.append(gt_hr_fft)
        predict_hr_fft_all.append(pred_hr_fft)
        predict_hr_peak_all.append(pred_hr_peak)
        gt_hr_peak_all.append(gt_hr_peak)

    predict_hr_peak_all = np.array(predict_hr_peak_all)
    predict_hr_fft_all = np.array(predict_hr_fft_all)
    gt_hr_peak_all = np.array(gt_hr_peak_all)
    gt_hr_fft_all = np.array(gt_hr_fft_all)
    
    for metric in config.MODEL_SPECS.TEST.BVP_METRICS:
        val = -1 # metric value

        if metric == "MAE":
            if config.MODEL_SPECS.TEST.EVALUATION_METHOD == "FFT":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                val = MAE_FFT
                print("FFT MAE (FFT Label):{0}".format(MAE_FFT))
            elif config.MODEL_SPECS.TEST.EVALUATION_METHOD == "peak detection":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                val = MAE_PEAK
                print("Peak MAE (Peak Label):{0}".format(MAE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "RMSE":
            if config.MODEL_SPECS.TEST.EVALUATION_METHOD == "FFT":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                val = RMSE_FFT
                print("FFT RMSE (FFT Label):{0}".format(RMSE_FFT))
            elif config.MODEL_SPECS.TEST.EVALUATION_METHOD == "peak detection":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                val = RMSE_PEAK
                print("PEAK RMSE (Peak Label):{0}".format(RMSE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "MAPE":
            if config.MODEL_SPECS.TEST.EVALUATION_METHOD == "FFT":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                val = MAPE_FFT
                print("FFT MAPE (FFT Label):{0}".format(MAPE_FFT))
            elif config.MODEL_SPECS.TEST.EVALUATION_METHOD == "peak detection":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                val = MAPE_PEAK
                print("PEAK MAPE (Peak Label):{0}".format(MAPE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "Pearson":
            if config.MODEL_SPECS.TEST.EVALUATION_METHOD == "FFT":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                val = Pearson_FFT[0][1]
                print("FFT Pearson (FFT Label):{0}".format(Pearson_FFT[0][1]))
            elif config.MODEL_SPECS.TEST.EVALUATION_METHOD == "peak detection":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                val = Pearson_PEAK[0][1]
                print("PEAK Pearson  (Peak Label):{0}".format(Pearson_PEAK[0][1]))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        else:
            raise ValueError("Wrong Test Metric Type")

        metric_dict[metric] = val # set the value w/ key metric

    print('')
    return metric_dict



def calculate_resp_metrics(predictions, labels, config):

    metric_dict = dict()

    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()

    for index in predictions.keys():
        #prediction = reform_data_from_dict(predictions[index])
        #label = reform_data_from_dict(labels[index]) 

        prediction = predictions[index]
        label = labels[index]

        if config.DATA.TRAIN.PREPROCESS.LABEL_TYPE == "Standardized" or config.DATA.TRAIN.PREPROCESS.LABEL_TYPE == "Raw":
            diff_flag_test = False
        elif config.DATA.TRAIN.PREPROCESS.LABEL_TYPE == "Normalized":
            diff_flag_test = True
        else:
            raise ValueError("Not supported label type in testing!")

        # GET FUNDEMENTAL HR FREQUENCY
        gt_hr_fft, pred_hr_fft = signal_metrics.calc_fundementalfreq_fft(
            prediction, label, signal='resp', diff_flag=diff_flag_test, fs=config.DATA.TEST.FS)
        gt_hr_peak, pred_hr_peak = signal_metrics.calc_fundementalfreq_peak(
            prediction, label, signal='resp', diff_flag=diff_flag_test, fs=config.DATA.TEST.FS)

        gt_hr_fft_all.append(gt_hr_fft)
        predict_hr_fft_all.append(pred_hr_fft)
        predict_hr_peak_all.append(pred_hr_peak)
        gt_hr_peak_all.append(gt_hr_peak)

    gt_hr_fft_all = np.array(gt_hr_fft_all)
    predict_hr_fft_all = np.array(predict_hr_fft_all)
    gt_hr_peak_all = np.array(gt_hr_peak_all)
    predict_hr_peak_all = np.array(predict_hr_peak_all)

    # Remove examples that are nan for whatever reason... TODO: Figure out the reason... lol 
    gt_peak_nan = np.isnan(gt_hr_peak_all)
    pred_peak_nan = np.isnan(predict_hr_peak_all)
    peak_nan = np.logical_or(gt_peak_nan, pred_peak_nan)
    gt_hr_peak_all = gt_hr_peak_all[~peak_nan]
    predict_hr_peak_all = predict_hr_peak_all[~peak_nan]

    print('')
    print(gt_hr_fft_all)
    print('')
    print(predict_hr_fft_all)
    print('')
    
    for metric in config.MODEL_SPECS.TEST.RESP_METRICS:
        val = -1 # metric value

        if metric == "MAE":
            if config.MODEL_SPECS.TEST.EVALUATION_METHOD == "FFT":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                val = MAE_FFT
                print("FFT MAE (FFT Label):{0}".format(MAE_FFT))
            elif config.MODEL_SPECS.TEST.EVALUATION_METHOD == "peak detection":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                val = MAE_PEAK
                print("Peak MAE (Peak Label):{0}".format(MAE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "RMSE":
            if config.MODEL_SPECS.TEST.EVALUATION_METHOD == "FFT":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                val = RMSE_FFT
                print("FFT RMSE (FFT Label):{0}".format(RMSE_FFT))
            elif config.MODEL_SPECS.TEST.EVALUATION_METHOD == "peak detection":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                val = RMSE_PEAK
                print("PEAK RMSE (Peak Label):{0}".format(RMSE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "MAPE":
            if config.MODEL_SPECS.TEST.EVALUATION_METHOD == "FFT":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                val = MAPE_FFT
                print("FFT MAPE (FFT Label):{0}".format(MAPE_FFT))
            elif config.MODEL_SPECS.TEST.EVALUATION_METHOD == "peak detection":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                val = MAPE_PEAK
                print("PEAK MAPE (Peak Label):{0}".format(MAPE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "Pearson":
            if config.MODEL_SPECS.TEST.EVALUATION_METHOD == "FFT":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                val = Pearson_FFT[0][1]
                print("FFT Pearson (FFT Label):{0}".format(Pearson_FFT[0][1]))
            elif config.MODEL_SPECS.TEST.EVALUATION_METHOD == "peak detection":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                val = Pearson_PEAK[0][1]
                print("PEAK Pearson  (Peak Label):{0}".format(Pearson_PEAK[0][1]))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        else:
            raise ValueError("Wrong Test Metric Type")
        

        metric_dict[metric] = val # set the value w/ key metric

    print('')
    return metric_dict



def calculate_au_metrics(preds, labels, config):

    metrics_dict = dict()

    all_trial_preds = []
    all_trial_labels = []

    for T in labels.keys():
        all_trial_preds.append(preds[T])
        all_trial_labels.append(labels[T])

    all_trial_preds = np.concatenate(all_trial_preds, axis=0)
    all_trial_labels = np.concatenate(all_trial_labels, axis=0)

    for metric in config.MODEL_SPECS.TEST.AU_METRICS:

        if metric == '12AUF1':

            named_AU = ['AU01', 'AU02', 'AU04', 'AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU23', 'AU24']
            AU_data = dict()
            AU_data['labels'] = dict()
            AU_data['preds'] = dict()

            for i in range(len(named_AU)):
                AU_data['labels'][named_AU[i]] = all_trial_labels[:, i, 0]
                AU_data['preds'][named_AU[i]] = all_trial_preds[:, i]

            # Calculate F1
            f1_dict = dict()  
            avg_f1 = 0
            acc_dict = dict()
            avg_acc = 0  
            print('')
            print('=== AU F1 ===')
            print('AU | F1')
            print('AU | F1 | Avg Val | Avg Label Val')
            for au in named_AU:
                preds = np.array(AU_data['preds'][au])
                preds[preds < 0.5] = 0
                preds[preds >= 0.5] = 1
                labels = np.array(AU_data['labels'][au])
                f1 = f1_score(labels, preds)*100
                f1_dict[au] = (f1, np.sum(preds)/len(preds), np.sum(labels)/len(labels))
                avg_f1 += f1
                print(au, f1, np.sum(preds)/len(preds), np.sum(labels)/len(labels))

                # get AU accuracy
                acc = sum(1 for x,y in zip(preds,labels) if x == y) / len(labels)
                acc_dict[au] = acc
                avg_acc += acc

            # Save Dictionary
            metrics_dict['12AUF1'] = f1_dict
            metrics_dict['12AUAcc'] = acc_dict

            print('Average F1:', avg_f1/len(named_AU))
            print('Average Acc:', avg_acc/len(named_AU))

        else:
            print('This AU metric does not exit')

    # Return F1 Metrics
    return metrics_dict




# def calculate_au_metrics(preds, labels, config):

#     metrics_dict = dict()

#     for metric in config.MODEL_SPECS.TEST.AU_METRICS:

#         if metric == '12AUF1':

#             named_AU = ['AU01', 'AU02', 'AU04', 'AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU23', 'AU24']
#             AU_data = dict()
#             AU_data['labels'] = dict()
#             AU_data['preds'] = dict()

#             for au in named_AU:
#                 AU_data['labels'][au] = []
#                 AU_data['preds'][au] = []

#             for T in labels.keys(): # for trials
#                 for C in labels[T]: # for chunk in trial
#                     l = labels[T][C][0,:,0].tolist() # TODO what to do if chunk len is 3
#                     p = preds[T][C][0,:].tolist()
                    
#                     for i in range(len(l)):
#                         AU_data['labels'][named_AU[i]].append(l[i])
#                         AU_data['preds'][named_AU[i]].append(p[i])

#             # Calculate F1
#             f1_dict = dict()     
#             print('')
#             print('=== AU F1 ===')
#             print('AU | F1')
#             print('AU | F1 | Avg Val | Avg Label Val')
#             for au in named_AU:
#                 preds = np.array(AU_data['preds'][au])
#                 preds[preds < 0.5] = 0
#                 preds[preds >= 0.5] = 1
#                 labels = np.array(AU_data['labels'][au])
#                 f1 = f1_score(labels, preds)*100

#                 #f1_dict[au] = f1
#                 #print(au, f1)
#                 f1_dict[au] = (f1, np.sum(preds)/len(preds), np.sum(labels)/len(labels))
#                 print(au, f1, np.sum(preds)/len(preds), np.sum(labels)/len(labels))
                
            
#             # Save Dictionary
#             metrics_dict['12AUF1'] = f1_dict

#         else:
#             print('This AU metric does not exit')

#     # Return F1 Metrics
#     return metrics_dict


def calculate_DISFA_au_metrics(preds, labels, config):

    print('IN DISFA METRICS')

    metrics_dict = dict()

    all_trial_preds = []
    all_trial_labels = []

    for T in labels.keys():
        all_trial_preds.append(preds[T])
        all_trial_labels.append(labels[T])

    disfa_idx = [0, 1, 2, 4, 6, 7, 8]
    all_trial_preds = np.concatenate(all_trial_preds, axis=0)
    all_trial_preds = all_trial_preds[:, disfa_idx]
    all_trial_labels = np.concatenate(all_trial_labels, axis=0)

    #TODO
    # print('')
    # print(all_trial_preds.shape) # GIRISH TO DO
    # print(all_trial_labels.shape) # GIRISH TO DO
    # raise ValueError('GIRISH KILL')

    for metric in config.MODEL_SPECS.TEST.AU_METRICS:

        if metric == '12AUF1':

            named_AU = ['AU01', 'AU02', 'AU04', 'AU06','AU12', 'AU15', 'AU17']
            AU_data = dict()
            AU_data['labels'] = dict()
            AU_data['preds'] = dict()

            for i in range(len(named_AU)):
                AU_data['labels'][named_AU[i]] = all_trial_labels[:, i, 0]
                AU_data['preds'][named_AU[i]] = all_trial_preds[:, i]

            # Calculate F1
            f1_dict = dict()  
            avg_f1 = 0
            acc_dict = dict()
            avg_acc = 0  
            print('')
            print('=== AU F1 ===')
            print('AU | F1')
            print('AU | F1 | Avg Val | Avg Label Val')
            for au in named_AU:
                preds = np.array(AU_data['preds'][au])
                preds[preds < 0.5] = 0
                preds[preds >= 0.5] = 1
                labels = np.array(AU_data['labels'][au])
                f1 = f1_score(labels, preds)*100
                f1_dict[au] = (f1, np.sum(preds)/len(preds), np.sum(labels)/len(labels))
                avg_f1 += f1
                print(au, f1, np.sum(preds)/len(preds), np.sum(labels)/len(labels))

                # get AU accuracy
                acc = sum(1 for x,y in zip(preds,labels) if x == y) / len(labels)
                acc_dict[au] = acc
                avg_acc += acc

            # Save Dictionary
            metrics_dict['DISFA_AUF1'] = f1_dict
            metrics_dict['DISFA_AUAcc'] = acc_dict

            print('Average F1:', avg_f1/len(named_AU))
            print('Average Acc:', avg_acc/len(named_AU))

        else:
            print('This AU metric does not exit')

    # Return F1 Metrics
    return metrics_dict