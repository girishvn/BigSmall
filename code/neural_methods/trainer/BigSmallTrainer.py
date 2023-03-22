import logging
import os
from collections import OrderedDict
from tqdm import tqdm
import pickle

import torch
import torch.optim as optim

# BASE TRAINER CLASS
from neural_methods.trainer.BaseTrainer import BaseTrainer

import numpy as np
from neural_methods import loss
from neural_methods.loss.pcgrad import PCGrad

# METRICS
from multitask_eval.metrics import calculate_bvp_metrics, calculate_resp_metrics, calculate_au_metrics

# MODELS
from neural_methods.model.BigSmall_models import BigPathwayBaseModel, SmallPathwayBaseModel
from neural_methods.model.literature_models import DRML, AlexNet, DeepPhys, MTTS_CAN_SMALL

class BigSmallTrainer(BaseTrainer):

    def define_model(self, config):

        # BigSmall Single Task Baselines
        # model = BigPathwayBaseModel(out_size=len(config.DATA.LABELS.USED_LABELS))
        # model = SmallPathwayBaseModel(out_size=len(config.DATA.LABELS.USED_LABELS))

        # Literature Baseline Models
        model = DRML(class_number=12)
        # model = AlexNet(num_classes=12)
        # model = DeepPhys()
        # model = MTTS_CAN_SMALL(frame_depth=3)

        return model



    def format_data_shape(self, data, labels):
        # reshape big data
        data_big = data[0]
        N, D, C, H, W = data_big.shape
        data_big = data_big.view(N * D, C, H, W)

        # reshape small data
        data_small = data[1]
        N, D, C, H, W = data_small.shape
        data_small = data_small.view(N * D, C, H, W)

        # reshape labels 
        if len(labels.shape) != 3: # this training format requires labels that are of shape N_label, D_label, C_label
            labels = torch.unsqueeze(labels, dim=-1)
        N_label, D_label, C_label = labels.shape
        labels = labels.view(N_label * D_label, C_label)

        # TODO If using TSM modules - reshape for GPU - change how this is used...
        if self.using_TSM:
            data_big = data_big[:(N * D) // self.base_len * self.base_len]
            data_small = data_small[:(N * D) // self.base_len * self.base_len]
            labels = labels[:(N * D) // self.base_len * self.base_len]

        data[0] = data_big
        data[1] = data_small
        labels = torch.unsqueeze(labels, dim=-1)

        return data, labels


    
    def send_data_to_device(self, data, labels):
        big_data = data[0].to(self.device)
        small_data = data[1].to(self.device)
        labels = labels.to(self.device)
        data = (big_data, small_data)
        return data, labels



    # adapted from: https://gist.github.com/IAmSuyogJadhav/bc388a871eda982ee0cf781b82227283
    def remove_data_parallel(self, old_state_dict):
        new_state_dict = OrderedDict()
        for k, v in old_state_dict.items():
            if 'module' in k:
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            else:
                name = k
                new_state_dict[name] = v

        return new_state_dict



    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')

        state_dict = self.model.state_dict()
        state_dict = self.remove_data_parallel(state_dict) # remove data parallel structure from state dict

        torch.save(state_dict, model_path)
        print('Saved Model Path: ', model_path)
        print('')



    def reform_data_from_dict(self, data, flatten):
        sort_data = sorted(data.items(), key=lambda x: x[0])
        sort_data = [i[1] for i in sort_data]
        sort_data = torch.cat(sort_data, dim=0)

        if flatten:
            sort_data = np.reshape(sort_data.cpu(), (-1))
        else:
            sort_data = np.array(sort_data.cpu())

        return sort_data



    def reform_preds_labels(self, predictions, labels, flatten=True):
        for index in predictions.keys():
            predictions[index] = self.reform_data_from_dict(predictions[index], flatten=flatten)
            labels[index] = self.reform_data_from_dict(labels[index], flatten=flatten)

        return predictions, labels



    def save_output_data(self):
        with open(self.save_data_path, 'wb') as handle:
                pickle.dump(self.data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



    

    def __init__(self, config, data_loader):

        print('')
        print('Init BigSmall SingleTask Trainer')
        print('')

        self.config = config # save config file

        # SET UP GPU COMPUTE DEVICE (GPU OR CPU)
        if torch.cuda.is_available() and config.NUM_OF_GPU_TRAIN > 0:
            self.device = torch.device(config.DEVICE) # set device to primary GPU
            self.num_of_gpu = config.NUM_OF_GPU_TRAIN # set number of used GPUs
        else:
            self.device = "cpu" # if no GPUs set device is CPU
            self.num_of_gpu = 0 # no GPUs used

        # DEFINING MODEL
        self.model = self.define_model(config) # define the model

        if torch.cuda.device_count() > 1 and config.NUM_OF_GPU_TRAIN > 1: # distribute model across GPUs
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN))) # data parallel model

        self.model = self.model.to(self.device) # send model to primary GPU

        # TRAINING PARAMETERS
        self.batch_size = config.MODEL_SPECS.TRAIN.BATCH_SIZE
        self.max_epoch_num = config.MODEL_SPECS.TRAIN.EPOCHS
        self.LR = config.MODEL_SPECS.TRAIN.LR
        self.num_train_batches = len(data_loader["train"])

        # Set Loss and Optimizer
        self.criterion = loss.loss_utils.set_loss(loss_name=config.MODEL_SPECS.TRAIN.LOSS_NAME, device=self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.MODEL_SPECS.TRAIN.LR, weight_decay=0)

        # Learning Schedulers and Tools 
        self.one_cycle_lr  = config.MODEL_SPECS.TRAIN.OCLR_SCHEDULER 

        if self.one_cycle_lr:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.LR, 
                                                                 epochs=self.max_epoch_num, 
                                                                 steps_per_epoch=self.num_train_batches)

        # MODEL INFO (SAVED MODEL DIR, CHUNK LEN, BEST EPOCH)
        self.model_dir = config.MODEL_SPECS.MODEL.MODEL_DIR
        self.model_file_name = config.MODEL_SPECS.TRAIN.MODEL_FILE_NAME
        self.chunk_len = config.DATA.TRAIN.PREPROCESS.CHUNK_LENGTH

        self.run_validation = self.config.MODEL_SPECS.VALID.RUN_VALIDATION
        self.model_to_use =  self.config.MODEL_SPECS.TEST.MODEL_TO_USE # either 'last_epoch' or 'best_epoch'
        self.used_epoch = 0

        # SAVED OUTPUT LOGGING INFO
        self.save_data = config.SAVE_DATA.SAVE_DATA
        self.save_train = config.SAVE_DATA.SAVE_TRAIN
        self.save_test = config.SAVE_DATA.SAVE_TEST
        self.save_metrics = config.SAVE_DATA.SAVE_METRICS

        self.save_data_path = config.SAVE_DATA.PATH
        self.data_dict = dict() # dictionary to save
        self.data_dict['config'] = self.config # save config file

        # INDICES CORRESPONDING TO USED LABELS 
        # Get indexes corresponding to used labels
        self.label_idx_train = self.get_label_idxs(config.DATA.TRAIN.LABELS.LABEL_LIST, config.DATA.TRAIN.LABELS.USED_LABELS)
        self.label_idx_valid = self.get_label_idxs(config.DATA.VALID.LABELS.LABEL_LIST, config.DATA.VALID.LABELS.USED_LABELS)
        self.label_idx_test = self.get_label_idxs(config.DATA.TEST.LABELS.LABEL_LIST, config.DATA.TEST.LABELS.USED_LABELS)
        print('Used Labels:', config.DATA.TRAIN.LABELS.USED_LABELS)
        print('Corresponding Indices:', self.label_idx_train)
        print('')

        # TODO Find Better Way To Integrate TSM Into This...
        self.using_TSM = False
        if self.using_TSM:
            self.frame_depth = 3 # 3 # default for TSCAN is 10 - consider changing later...
            self.base_len = self.num_of_gpu * self.frame_depth   



    def train(self, data_loader):
        """Model Training"""

        if data_loader["train"] is None:
            raise ValueError("No data for train")

        print('Starting Training Routine')
        print('')

        # Init min validation loss as infinity
        min_valid_loss = np.inf # minimum validation loss

        # ARRAYS TO SAVE (LOSS ARRAYS)
        train_loss_dict = dict()
        val_loss_dict = dict()

        # ITERATE THROUGH EPOCHS
        for epoch in range(self.max_epoch_num):
            print(f"====Training Epoch: {epoch}====")

            # INIT PARAMS FOR TRAINING
            running_loss = 0.0 # tracks avg loss over mini batches of 100
            train_loss = []
            self.model.train() # put model in train mode

            # MODEL TRAINING
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)

                # GATHER AND FORMAT BATCH DATA
                data, labels = batch[0], batch[1]
                data, labels = self.format_data_shape(data, labels)
                data, labels = self.send_data_to_device(data, labels)

                # FOWARD AND BACK PROPOGATE THROUGH MODEL
                self.optimizer.zero_grad()
                out = self.model(data)
                loss = self.criterion(out, labels[:, self.label_idx_train, 0])
                loss.backward()
                self.optimizer.step() # Step the optimizer
                if self.one_cycle_lr: # If one cycle learning rate scheduler
                    self.scheduler.step() # Step the schedulers

                # UPDATE RUNNING LOSS AND PRINTED TERMINAL OUTPUT
                train_loss.append(loss.item())

                running_loss += loss.item()
                if idx % 100 == 99: # print every 100 mini-batches
                    print(f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                tbar.set_postfix({"loss:": loss.item(), "lr:": self.optimizer.param_groups[0]["lr"]})

            # APPEND EPOCH LOSS LIST TO TRAINING LOSS DICTIONARY
            train_loss_dict[epoch] = train_loss

            print('')

            # SAVE MODEL FOR THIS EPOCH
            self.save_model(epoch)

            # VALIDATION (ENABLED)
            if self.run_validation or self.model_to_use == 'best_epoch':
                
                 # Get validation losses
                valid_loss = self.valid(data_loader)
                val_loss_dict[epoch] = valid_loss
                print('validation loss: ', valid_loss)

                 # Update used model
                if self.model_to_use == 'best_epoch' and (valid_loss < min_valid_loss):
                    self.used_epoch = epoch
                    min_valid_loss = valid_loss
                    print("Update best model! Best epoch: {}".format(self.used_epoch))
                    print('')
                elif self.model_to_use == 'last_epoch':
                    self.used_epoch = epoch
                    min_valid_loss = valid_loss

                print('')

            # VALIDATION (NOT ENABLED)
            else: 
                self.used_epoch = epoch

        # IF SAVING OUTPUT DATA
        if self.save_data:
            self.data_dict['train_loss'] = train_loss_dict
            self.data_dict['val_loss'] = val_loss_dict

        # PRINT MODEL TO BE USED FOR TESTING
        print("Used model trained epoch:{}, val_loss:{}".format(self.used_epoch, min_valid_loss))
        print('')



    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""

        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print("===Validating===")

        # INIT PARAMS FOR VALIDATION
        valid_loss = []
        self.model.eval()

        # MODEL VALIDATION
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")

                # GATHER AND FORMAT BATCH DATA
                data, labels = valid_batch[0], valid_batch[1]
                data, labels = self.format_data_shape(data, labels)
                data, labels = self.send_data_to_device(data, labels)

                out = self.model(data)
                loss = self.criterion(out, labels[:, self.label_idx_valid, 0]) 
                valid_loss.append(loss.item())
                vbar.set_postfix(loss=loss.item())

        valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)


    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""

        print("===Testing===")
        print('')

        # SETUP
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        # Change chunk length to be test chunk length
        # self.chunk_len = config.DATA.TEST.PREPROCESS.CHUNK_LENGTH 
        self.chunk_len = 180 # TODO Hardcoded

        # ARRAYS TO SAVE (PREDICTIONS AND METRICS ARRAYS)
        preds_dict = dict()
        labels_dict = dict()

        # IF ONLY_TEST MODE LOAD PRETRAINED MODEL
        if self.config.TOOLBOX_MODE == "only_test":
            model_path = self.config.MODEL_SPECS.TEST.MODEL_PATH
            print("Testing uses pretrained model!")
            print('Model path:', model_path)
            if not os.path.exists(model_path):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")

        # IF USING MODEL FROM TRAINING
        else:
            model_path = os.path.join(self.model_dir, 
                                           self.model_file_name + '_Epoch' + str(self.used_epoch) + '.pth')
            print("Testing uses non-pretrained model!")
            print('Model path:', model_path)
            if not os.path.exists(model_path):
                raise ValueError("Something went wrong... cant find trained model...")
        print('')
            

        # LOAD ABOVED SPECIFIED MODEL FOR TESTING
        self.model = self.define_model(self.config) # define the model
        state_dict = torch.load(model_path)
        state_dict = self.remove_data_parallel(state_dict)
        self.model.load_state_dict(state_dict)

        if torch.cuda.device_count() > 1 and self.config.NUM_OF_GPU_TRAIN > 1: # distribute model across GPUs
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.config.NUM_OF_GPU_TRAIN))) # data parallel model

        self.model = self.model.to(self.device) # send model to primary GPU
        self.model.eval() # Eval mode for test


        # MODEL TESTING
        with torch.no_grad():
            for _, test_batch in enumerate(data_loader['test']):

                # PROCESSING - ANALYSIS, METRICS, SAVING OUT DATA
                batch_size = test_batch[1].shape[0] # get batch size

                # GATHER AND FORMAT BATCH DATA
                data, labels = test_batch[0], test_batch[1]
                data, labels = self.format_data_shape(data, labels)
                data, labels = self.send_data_to_device(data, labels)

                # GET MODEL PREDICTIONS
                out = self.model(data)
                # if BCE Loss is used output needs to be passed through sigmoid (map 0-1) before binary classification
                if self.config.MODEL_SPECS.TRAIN.LOSS_NAME == 'BCEWithLogitsBP4DAU' or 'BCEWithLogits':
                    out = torch.sigmoid(out) # ADDED FOR AU TODO Improve this

                # GATHER AND SLICE LABELS USED FOR TEST DATASET
                if self.config.MODEL_SPECS.TEST.BVP_METRICS:
                    labels = labels[:, 0] # TODO FOR BVP
                else:
                    labels = labels[:, self.label_idx_test]
                
                # IF TEST PREDICTION DATA TO BE SAVED - MOVE FROM GPU TO CPU
                if self.save_data and self.save_test:
                    out = out.to('cpu')
                    labels = labels.to('cpu') 

                # ITERATE THROUGH BATCH, SORT, AND ADD TO CORRECT DICTIONARY
                for idx in range(batch_size):

                    #TODO maybe find a better way to address this... 
                    # if the labels are cut off due to TSM dataformating
                    if idx * self.chunk_len >= labels.shape[0] and self.using_TSM:
                        continue 

                    #TODO Use This For DISFA Training
                    # subj_index = test_batch[2][idx]
                    # sort_index = int(test_batch[3][idx].replace('input', ''))

                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])

                    # add subject to prediction / label arrays
                    if subj_index not in preds_dict.keys():
                        preds_dict[subj_index] = dict()
                        labels_dict[subj_index] = dict()

                    # append predictions and labels to subject dict
                    preds_dict[subj_index][sort_index] = out[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels_dict[subj_index][sort_index] = labels[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        # CALCULATE METRICS ON PREDICTIONS
        if self.config.MODEL_SPECS.TEST.BVP_METRICS: # run metrics, if not empty list
            preds_dict, labels_dict = self.reform_preds_labels(preds_dict, labels_dict)
            print('BVP Metrics:')
            bvp_metric_dict = calculate_bvp_metrics(preds_dict, labels_dict, self.config)
            if self.save_metrics:
                self.data_dict['bvp_metrics'] = bvp_metric_dict
            print('')

        if self.config.MODEL_SPECS.TEST.RESP_METRICS: # run metrics, if not empty list
            preds_dict, labels_dict = self.reform_preds_labels(preds_dict, labels_dict)
            print('Resp Metrics:')
            resp_metric_dict = calculate_resp_metrics(preds_dict, labels_dict, self.config)
            if self.save_metrics:
                self.data_dict['resp_metrics'] = resp_metric_dict
            print('')

        if self.config.MODEL_SPECS.TEST.AU_METRICS: # run metrics, if not empty list
            preds_dict, labels_dict = self.reform_preds_labels(preds_dict, labels_dict, flatten=False)
            print('AU Metrics:')
            au_metric_dict = calculate_au_metrics(preds_dict, labels_dict, self.config) 
            # au_metric_dict = calculate_DISFA_au_metrics(preds_dict, labels_dict, self.config) # Used for DISFA
            if self.save_metrics:
                self.data_dict['au_metrics'] = au_metric_dict
            print('')
        
        
        # IF TEST PREDICTION DATA TO BE SAVED - SAVE PREDICTION ARRAYS TO DATA DICT
        if self.save_data and self.save_test:
            self.data_dict['test_preds'] = preds_dict
            self.data_dict['test_labels'] = labels_dict


        # SAVE TRAIN/VAL/TEST DATA OUTPUT DATA DICT  
        if self.save_data: # save output data dictionary as pickle file
            self.save_output_data()


    
        