"""Trainer for MTTSCAN Multitask Models"""
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

#from metrics.metrics import calculate_metrics
from multitask_eval.metrics import calculate_bvp_metrics, calculate_resp_metrics, calculate_au_metrics
import pickle


class BigSmallMultitaskTrainer_MTTSCAN(BaseTrainer):

    def define_model(self, config):

        model = MTTS_CAN_SMALL(frame_depth=3)


        # TODO Find Better Way To Integrate TSM Into This...
        self.using_TSM = True
        if self.using_TSM:
            self.frame_depth = 3 # 3 # default for TSCAN is 10 - consider changing later...
            self.base_len = self.num_of_gpu * self.frame_depth   
            print("USING TIME SHIFT MODULE LOGIC")  


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
        print('Init Multitask MTTSCAN Trainer')
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
        self.criterionBVP = loss.loss_utils.set_loss(loss_name='MSE', device=self.device)
        self.criterionRESP = loss.loss_utils.set_loss(loss_name='MSE', device=self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.LR, weight_decay=0)

        # Learning Schedulers and Tools 
        self.gradient_surgery = config.MODEL_SPECS.TRAIN.GRAD_SURGERY
        self.one_cycle_lr  = config.MODEL_SPECS.TRAIN.OCLR_SCHEDULER 

        if self.gradient_surgery and self.one_cycle_lr:
            raise ValueError('Gradient Surgery AND One Cycle LR Scheduling CANNNOT both be enabled')

        if self.gradient_surgery:
            print('Using Gradient Surgery...')
            self.num_tasks = 3
            self.optimizer = PCGrad(self.optimizer) 

        elif self.one_cycle_lr:
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
        train_bvp_label_list = [label for label in config.DATA.TRAIN.LABELS.USED_LABELS if 'bvp' in label]
        valid_bvp_label_list = [label for label in config.DATA.VALID.LABELS.USED_LABELS if 'bvp' in label]
        test_bvp_label_list = [label for label in config.DATA.TEST.LABELS.USED_LABELS if 'bvp' in label]

        train_resp_label_list = [label for label in config.DATA.TRAIN.LABELS.USED_LABELS if 'resp' in label]
        valid_resp_label_list = [label for label in config.DATA.VALID.LABELS.USED_LABELS if 'resp' in label]
        test_resp_label_list = [label for label in config.DATA.TEST.LABELS.USED_LABELS if 'resp' in label]

        self.label_idx_train_bvp = self.get_label_idxs(config.DATA.TRAIN.LABELS.LABEL_LIST, train_bvp_label_list)
        self.label_idx_valid_bvp = self.get_label_idxs(config.DATA.VALID.LABELS.LABEL_LIST, valid_bvp_label_list)
        self.label_idx_test_bvp = self.get_label_idxs(config.DATA.TEST.LABELS.LABEL_LIST, test_bvp_label_list)

        self.label_idx_train_resp = self.get_label_idxs(config.DATA.TRAIN.LABELS.LABEL_LIST, train_resp_label_list)
        self.label_idx_valid_resp = self.get_label_idxs(config.DATA.VALID.LABELS.LABEL_LIST, valid_resp_label_list)
        self.label_idx_test_resp = self.get_label_idxs(config.DATA.TEST.LABELS.LABEL_LIST, test_resp_label_list)

        print('Used Labels:', config.DATA.TRAIN.LABELS.USED_LABELS)
        print('Training Indices BVP:', self.label_idx_train_bvp)
        print('Training Indices Resp:', self.label_idx_train_resp)
        print('')



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
        train_au_loss_dict = dict()
        train_bvp_loss_dict = dict()
        train_resp_loss_dict = dict()

        pretrain_loss_dict = dict()
        pretrain_bvp_loss_dict = dict()
        pretrain_resp_loss_dict = dict()

        val_loss_dict = dict()
        val_au_loss_dict = dict()
        val_bvp_loss_dict = dict()
        val_resp_loss_dict = dict()

        # ITERATE THROUGH EPOCHS
        for epoch in range(self.max_epoch_num):
        
            # INIT PARAMS FOR TRAINING
            running_loss = 0.0 # tracks avg loss over mini batches of 100
            train_loss = []
            train_au_loss = []
            train_bvp_loss = []
            train_resp_loss = []
            self.model.train() # put model in train mode

            # MODEL PRETRAINING TODO
            # if epoch == 0:
            # pretrain_loss, pretrain_bvp_loss, pretrain_resp_loss = self.pretrain(data_loader, epoch)


            print(f"====Training Epoch: {epoch}====")

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
                bvp_out, resp_out = self.model(data)

                bvp_loss = self.criterionBVP(bvp_out, labels[:, self.label_idx_train_bvp, 0]) # bvp loss
                resp_loss =  self.criterionRESP(resp_out, labels[:, self.label_idx_train_resp, 0]) # resp loss 
                loss = bvp_loss + resp_loss # sum losses 

                if self.gradient_surgery: # Apply gradient surgery
                    losses = [bvp_loss, resp_loss]
                    assert len(losses) == self.num_tasks
                    self.optimizer.pc_backward(losses)
                else: # Else if OCLR or normal
                    loss.backward()

                self.optimizer.step() # Step the optimizer
                if self.one_cycle_lr: # If one cycle learning rate scheduler
                    self.scheduler.step() # Step the scheduler

                # UPDATE RUNNING LOSS AND PRINTED TERMINAL OUTPUT AND SAVED LOSSES
                train_loss.append(loss.item())
                train_bvp_loss.append(bvp_loss.item())
                train_resp_loss.append(resp_loss.item())

                running_loss += loss.item()
                if idx % 100 == 99: # print every 100 mini-batches
                    print(f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                if self.gradient_surgery:
                    tbar.set_postfix({"loss:": loss.item(), "lr:": self.optimizer._optim.param_groups[0]["lr"]})
                else: 
                    tbar.set_postfix({"loss:": loss.item(), "lr:": self.optimizer.param_groups[0]["lr"]})

            # APPEND EPOCH LOSS LIST TO TRAINING LOSS DICTIONARY
            train_loss_dict[epoch] = train_loss
            train_au_loss_dict[epoch] = train_au_loss
            train_bvp_loss_dict[epoch] = train_bvp_loss
            train_resp_loss_dict[epoch] = train_resp_loss
            
            print('')

            # SAVE MODEL FOR THIS EPOCH
            self.save_model(epoch)

            # VALIDATION (ENABLED)
            if self.run_validation or self.model_to_use == 'best_epoch':

                # Get validation losses
                valid_loss, valid_au_loss, valid_bvp_loss, valid_resp_loss = self.valid(data_loader)
                val_loss_dict[epoch] = valid_loss
                val_au_loss_dict[epoch] = valid_au_loss
                val_bvp_loss_dict[epoch] = valid_bvp_loss
                val_resp_loss_dict[epoch] = valid_resp_loss
                print('validation loss: ', valid_loss)

                # Update used model
                if self.model_to_use == 'best_epoch' and (valid_loss < min_valid_loss):
                    min_valid_loss = valid_loss
                    self.used_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.used_epoch))
                elif self.model_to_use == 'last_epoch':
                    self.used_epoch = epoch
            
            # VALIDATION (NOT ENABLED)
            else: 
                self.used_epoch = epoch

            print('')

        # IF SAVING OUTPUT DATA
        if self.save_data:
            self.data_dict['train_loss'] = train_loss_dict
            self.data_dict['train_au_loss'] = train_au_loss_dict
            self.data_dict['train_bvp_loss'] = train_bvp_loss_dict
            self.data_dict['train_resp_loss'] = train_resp_loss_dict
            self.data_dict['val_loss'] = val_loss_dict 
            self.data_dict['val_au_loss'] = val_au_loss_dict
            self.data_dict['val_bvp_loss'] = val_bvp_loss_dict
            self.data_dict['val_resp_loss'] = val_resp_loss_dict

        # PRINT MODEL TO BE USED FOR TESTING
        print("Used model trained epoch:{}, val_loss:{}".format(self.used_epoch, min_valid_loss))
        print('')




    def pretrain(self, data_loader, epoch):
        """Model Pretraining"""

        if data_loader["pretrain"] is None:
            raise ValueError("No data for pretrain")

        print('Starting Pretraining Routine')
        print('')

        # INIT PARAMS FOR TRAINING
        running_loss = 0.0 # tracks avg loss over mini batches of 100
        pretrain_loss = []
        pretrain_bvp_loss = []
        pretrain_resp_loss = []

        self.model.train() # put model in train mode

        # MODEL TRAINING
        tbar = tqdm(data_loader["pretrain"], ncols=80)
        for idx, batch in enumerate(tbar):
            tbar.set_description("Pretrain epoch %s" % epoch)

            # GATHER AND FORMAT BATCH DATA
            data, labels = batch[0], batch[1]
            data, labels = self.format_data_shape(data, labels)
            data, labels = self.send_data_to_device(data, labels)

            # FOWARD AND BACK PROPOGATE THROUGH MODEL
            self.optimizer.zero_grad()
            au_out, bvp_out, resp_out = self.model(data)

            bvp_loss = self.criterionBVP(bvp_out, labels[:, self.label_idx_train_bvp, 0]) # bvp loss
            resp_loss =  self.criterionRESP(resp_out, labels[:, self.label_idx_train_resp, 0]) # resp loss 
            loss = bvp_loss + resp_loss # only sum bvp and resp loss (NOT AU)

            if self.gradient_surgery: # Apply gradient surgery
                losses = [au_loss, bvp_loss, resp_loss]
                assert len(losses) == self.num_tasks
                self.optimizer.pc_backward(losses)
            else: # Else if OCLR or normal
                loss.backward()

            self.optimizer.step() # Step the optimizer
            if self.one_cycle_lr: # If one cycle learning rate scheduler
                self.scheduler.step() # Step the scheduler

            # UPDATE RUNNING LOSS AND PRINTED TERMINAL OUTPUT AND SAVED LOSSES
            pretrain_loss.append(loss.item())
            pretrain_bvp_loss.append(bvp_loss.item())
            pretrain_resp_loss.append(resp_loss.item())

            running_loss += loss.item()
            if idx % 100 == 99: # print every 100 mini-batches
                print(f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
            if self.gradient_surgery:
                tbar.set_postfix({"loss:": loss.item(), "lr:": self.optimizer._optim.param_groups[0]["lr"]})
            else: 
                tbar.set_postfix({"loss:": loss.item(), "lr:": self.optimizer.param_groups[0]["lr"]})
        
        print('')
        return pretrain_loss, pretrain_bvp_loss, pretrain_resp_loss



    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""

        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print("===Validating===")

        # INIT PARAMS FOR VALIDATION
        valid_loss = []
        valid_au_loss = []
        valid_bvp_loss = []
        valid_resp_loss = []
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

                au_out, bvp_out, resp_out = self.model(data)

                #TODO AU SUBSAMLE
                # subsamle_idx_arr = np.arange(0, au_out.shape[0], self.frame_depth)
                # au_out = au_out[subsamle_idx_arr, :]
                # au_labels = labels[:, self.label_idx_valid_au, 0]
                # au_labels = au_labels[subsamle_idx_arr, :]
                # au_loss = self.criterionAU(au_out, au_labels) 
                # au_loss = au_loss
                #TODO

                au_loss = self.criterionAU(au_out, labels[:, self.label_idx_valid_au, 0]) # au loss
                bvp_loss = self.criterionBVP(bvp_out, labels[:, self.label_idx_valid_bvp, 0]) # bvp loss
                resp_loss =  self.criterionRESP(resp_out, labels[:, self.label_idx_valid_resp, 0]) # resp loss 
                loss = au_loss + bvp_loss + resp_loss # sum losses

                # APPEND VAL LOSS
                valid_loss.append(loss.item())
                valid_au_loss.append(au_loss.item())
                valid_bvp_loss.append(bvp_loss.item())
                valid_resp_loss.append(resp_loss.item())
                vbar.set_postfix(loss=loss.item())

        valid_loss = np.asarray(valid_loss)
        valid_au_loss = np.asarray(valid_au_loss)
        valid_bvp_loss = np.asarray(valid_bvp_loss)
        valid_resp_loss = np.asarray(valid_resp_loss)
        return np.mean(valid_loss), np.mean(valid_au_loss), np.mean(valid_bvp_loss), np.mean(valid_resp_loss)



    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""

        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print("===Testing===")
        print('')

        # ARRAYS TO SAVE (PREDICTIONS AND METRICS ARRAYS)
        preds_dict_au = dict()
        labels_dict_au = dict()
        preds_dict_bvp = dict()
        labels_dict_bvp = dict()
        preds_dict_resp = dict()
        labels_dict_resp = dict()

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

        ####################################################################
        ######################## BVP / RESP METRICS ########################
        ####################################################################

        # # WIPE BVP and RESP DICTS
        preds_dict_bvp = dict()
        labels_dict_bvp = dict()
        preds_dict_resp = dict()
        labels_dict_resp = dict()

        print('TESTING ON FULL BP4D')

        # MODEL TESTING
        with torch.no_grad():
            for _, test_batch in enumerate(data_loader['pretrain']):

                # PROCESSING - ANALYSIS, METRICS, SAVING OUT DATA
                batch_size = test_batch[1].shape[0] # get batch size

                # GATHER AND FORMAT BATCH DATA
                data, labels = test_batch[0], test_batch[1]
                data, labels = self.format_data_shape(data, labels)
                data, labels = self.send_data_to_device(data, labels)

                # GET MODEL PREDICTIONS
                bvp_out, resp_out = self.model(data)

                TEST_BVP = False
                if len(self.label_idx_test_bvp) > 0: # if test dataset has BVP
                    TEST_BVP = True
                    labels_bvp = labels[:, 0] # TODO use bpwave as label for BVP pseudo input
                else: # if not set whole BVP labels array to -1
                    labels_bvp = np.ones((batch_size, len(self.label_idx_train_bvp)))
                    labels_bvp = -1 * labels_bvp

                TEST_RESP = False
                if len(self.label_idx_test_resp) > 0: # if test dataset has BVP
                    TEST_RESP = True
                    labels_resp = labels[:, self.label_idx_test_resp]
                else: # if not set whole BVP labels array to -1
                    labels_resp = np.ones((batch_size, len(self.label_idx_train_resp)))
                    labels_resp = -1 * labels_resp
                
                # IF TEST PREDICTION DATA TO BE SAVED - MOVE FROM GPU TO CPU
                if self.save_data and self.save_test:
                    bvp_out = bvp_out.to('cpu')
                    labels_bvp = labels_bvp.to('cpu')
                    resp_out = resp_out.to('cpu')
                    labels_resp = labels_resp.to('cpu')

                # ITERATE THROUGH BATCH, SORT, AND ADD TO CORRECT DICTIONARY
                for idx in range(batch_size):

                    #TODO maybe find a better way to address this...
                    # if the labels are cut off due to TSM dataformating
                    self.chunk_len = 60 # TODO Hardcoded to test on entire dataset
                    if idx * self.chunk_len >= labels.shape[0] and self.using_TSM:
                        continue 

                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])

                    # add subject to prediction / label arrays
                    if subj_index not in preds_dict_bvp.keys():
                        preds_dict_bvp[subj_index] = dict()
                        labels_dict_bvp[subj_index] = dict()
                        preds_dict_resp[subj_index] = dict()
                        labels_dict_resp[subj_index] = dict()

                    # append predictions and labels to subject dict
                    preds_dict_bvp[subj_index][sort_index] = bvp_out[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels_dict_bvp[subj_index][sort_index] = labels_bvp[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    preds_dict_resp[subj_index][sort_index] = resp_out[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels_dict_resp[subj_index][sort_index] = labels_resp[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        ####################################################################
        ####################################################################
        ####################################################################

        # REFORM DATA
        preds_dict_bvp, labels_dict_bvp = self.reform_preds_labels(preds_dict_bvp, labels_dict_bvp)
        preds_dict_resp, labels_dict_resp = self.reform_preds_labels(preds_dict_resp, labels_dict_resp)

        # CALCULATE METRICS ON PREDICTIONS
        if self.config.MODEL_SPECS.TEST.BVP_METRICS: # run metrics, if not empty list
            print('BVP Metrics:')
            bvp_metric_dict = calculate_bvp_metrics(preds_dict_bvp, labels_dict_bvp, self.config)
            if self.save_metrics:
                self.data_dict['bvp_metrics'] = bvp_metric_dict
            print('')

        if self.config.MODEL_SPECS.TEST.RESP_METRICS: # run metrics, if not empty list
            print('Resp Metrics:')
            resp_metric_dict = calculate_resp_metrics(preds_dict_resp, labels_dict_resp, self.config)
            if self.save_metrics:
                self.data_dict['resp_metrics'] = resp_metric_dict
            print('')

        
        # IF TEST PREDICTION DATA TO BE SAVED - SAVE PREDICTION ARRAYS TO DATA DICT
        if self.save_data and self.save_test:
            self.data_dict['test_preds_au'] = preds_dict_au
            self.data_dict['test_labels_au'] = labels_dict_au
            self.data_dict['test_preds_bvp'] = preds_dict_bvp
            self.data_dict['test_labels_bvp'] = labels_dict_bvp
            self.data_dict['test_preds_resp'] = preds_dict_resp
            self.data_dict['test_labels_resp'] = labels_dict_resp

        # IF TRAIN PREDICTION DATA TO BE SAVED - SAVE PREDICTION ARRAYS TO DATA DICT
        if self.save_data and self.save_train:
            print('SAVING PREDICTIONS ON TRAINING DATA IS CURRENTLY COMMENTED OUT')
            # comment out if training preds NOT wanted
            # self.get_train_preds(data_loader) # TODO Comment out if you dont want to save train data predictions

        # SAVE TRAIN/VAL/TEST DATA OUTPUT DATA DICT  
        if self.save_data: # save output data dictionary as pickle file
            self.save_output_data()


