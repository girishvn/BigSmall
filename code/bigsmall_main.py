""" The main function of BigSmall Multitask deep learning pipeline."""

import argparse
import random
import time

import numpy as np
import torch
from config import get_config
from dataset import data_loader
from neural_methods import trainer
from signal_methods.signal_predictor import signal_predict
from torch.utils.data import DataLoader

# Set Random Seeds
RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator()
g.manual_seed(RANDOM_SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="configs/PURE_PURE_UBFC_TSCAN_BASIC.yaml", 
                        type=str, help="The name of the model.")

    return parser


def train_test(config, data_loader_dict):
    """Trains the model."""

    # Define Model to Train / Test
    if config.MODEL_SPECS.MODEL.NAME == 'BigSmall':
        model_trainer = trainer.BigSmallTrainer.BigSmallTrainer(config, data_loader_dict)
    elif config.MODEL_SPECS.MODEL.NAME == 'BigSmallMultitask':
        model_trainer = trainer.BigSmallMultitaskTrainer.BigSmallMultitaskTrainer(config, data_loader_dict)
    elif config.MODEL_SPECS.MODEL.NAME == 'BigSmallMultitask_Pretrain':
        model_trainer = trainer.BigSmallMultitaskTrainer_Pretrain.BigSmallMultitaskTrainer_Pretrain(config, data_loader_dict)
    elif config.MODEL_SPECS.MODEL.NAME == 'BigSmallMultitaskTrainer_MTTSCAN':
        model_trainer = trainer.BigSmallMultitaskTrainer_MTTSCAN.BigSmallMultitaskTrainer_MTTSCAN(config, data_loader_dict)


    else:
        raise ValueError('Your Model is Not Supported  Yet!')

    if config.TOOLBOX_MODE == "train_and_test": # Train model if requested
        model_trainer.train(data_loader_dict)
    
    model_trainer.test(data_loader_dict) # Test model


def signal_method_inference(config, data_loader):
    if not config.SIGNAL_SPECS.METHOD:
        raise ValueError("Please set signal method in yaml!")
    for signal_method in config.SIGNAL_SPECS.METHOD:
        if signal_method == "POS":
            signal_predict(config, data_loader, "POS")
        elif signal_method == "CHROM":
            signal_predict(config, data_loader, "CHROM")
        elif signal_method == "ICA":
            signal_predict(config, data_loader, "ICA")
        elif signal_method == "GREEN":
            signal_predict(config, data_loader, "GREEN")
        elif signal_method == "LGI":
            signal_predict(config, data_loader, "LGI")
        elif signal_method == "PBV":
            signal_predict(config, data_loader, "PBV")
        else:
            raise ValueError("Not supported signal method!")


# MAIN EXECUTION SCRIPT START
if __name__ == "__main__":

    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # configurations.
    config = get_config(args)
    print(config)

    data_loader_dict = dict()
    if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "only_test" or config.TOOLBOX_MODE == "only_preprocess":
        # neural method dataloader
        # train_loader
        if config.DATA.TRAIN.DATASET == "COHFACE":
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, and SCAMPS.")
        elif config.DATA.TRAIN.DATASET == "UBFC":
            train_loader = data_loader.UBFCLoader.UBFCLoader
        elif config.DATA.TRAIN.DATASET == "PURE":
            train_loader = data_loader.PURELoader.PURELoader
        elif config.DATA.TRAIN.DATASET == "SCAMPS":
            train_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.DATA.TRAIN.DATASET == "BP4D":
            train_loader = data_loader.BP4DLoader.BP4DLoader

        elif config.DATA.TRAIN.DATASET == "BP4DBigSmall":
            train_loader = data_loader.BigSmallLoader.BP4DBigSmallLoader
        elif config.DATA.TRAIN.DATASET == "PUREBigSmall":
            train_loader = data_loader.BigSmallLoader.PUREBigSmallLoader
        elif config.DATA.TRAIN.DATASET == "UBFCBigSmall":
            train_loader = data_loader.BigSmallLoader.UBFCBigSmallLoader

        elif config.DATA.TRAIN.DATASET == "MULTIDATASET":
            train_loader = data_loader.MultiDatasetLoader.MultiDatasetLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, and SCAMPS.")

        # valid_loader
        if config.DATA.VALID.DATASET == "UBFC":
            valid_loader = data_loader.UBFCLoader.UBFCLoader
        elif config.DATA.VALID.DATASET == "PURE":
            valid_loader = data_loader.PURELoader.PURELoader
        elif config.DATA.VALID.DATASET == "SCAMPS":
            valid_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.DATA.VALID.DATASET == "BP4D":
            valid_loader = data_loader.BP4DLoader.BP4DLoader

        elif config.DATA.VALID.DATASET == "BP4DBigSmall":
            valid_loader = data_loader.BigSmallLoader.BP4DBigSmallLoader
        elif config.DATA.VALID.DATASET == "PUREBigSmall":
            valid_loader = data_loader.BigSmallLoader.PUREBigSmallLoader
        elif config.DATA.VALID.DATASET == "UBFCBigSmall":
            valid_loader = data_loader.BigSmallLoader.UBFCBigSmallLoader

        elif config.DATA.VALID.DATASET == "MULTIDATASET":
            valid_loader = data_loader.MultiDatasetLoader.MultiDatasetLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, and SCAMPS.")

        # test_loader
        if config.DATA.TEST.DATASET == "UBFC":
            test_loader = data_loader.UBFCLoader.UBFCLoader
        elif config.DATA.TEST.DATASET == "PURE":
            test_loader = data_loader.PURELoader.PURELoader
        elif config.DATA.TEST.DATASET == "SCAMPS":
            test_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.DATA.TEST.DATASET == "BP4D":
            test_loader = data_loader.BP4DLoader.BP4DLoader

        elif config.DATA.TEST.DATASET == "BP4DBigSmall":
            test_loader = data_loader.BigSmallLoader.BP4DBigSmallLoader
        elif config.DATA.TEST.DATASET == "PUREBigSmall":
            test_loader = data_loader.BigSmallLoader.PUREBigSmallLoader
        elif config.DATA.TEST.DATASET == "UBFCBigSmall":
            test_loader = data_loader.BigSmallLoader.UBFCBigSmallLoader

        elif config.DATA.TEST.DATASET == "MULTIDATASET":
            test_loader = data_loader.MultiDatasetLoader.MultiDatasetLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, and SCAMPS.")

        # Pretrain loader
        if config.DATA.PRETRAIN.DATASET == "BP4DBigSmall":
            pretrain_loader = data_loader.BigSmallLoader.BP4DBigSmallLoader 
        elif config.DATA.PRETRAIN.DATASET == "PUREBigSmall":
            pretrain_loader = data_loader.BigSmallLoader.PUREBigSmallLoader
        elif config.DATA.PRETRAIN.DATASET == "UBFCBigSmall":
            pretrain_loader = data_loader.BigSmallLoader.UBFCBigSmallLoader

        # Train loader
        if config.DATA.TRAIN.DATA_PATH:
            train_data_loader = train_loader(
                name="train",
                data_path=config.DATA.TRAIN.DATA_PATH,
                config_data=config.DATA.TRAIN,
                total_config=config)
            data_loader_dict['train'] = DataLoader(
                dataset=train_data_loader,
                num_workers=8,
                batch_size=config.MODEL_SPECS.TRAIN.BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g
            )
        else:
            data_loader_dict['train'] = None

        # Valid loader
        if config.DATA.VALID.DATA_PATH:
            valid_data = valid_loader(
                name="valid",
                data_path=config.DATA.VALID.DATA_PATH,
                config_data=config.DATA.VALID, 
                total_config=config)
            data_loader_dict["valid"] = DataLoader(
                dataset=valid_data,
                num_workers=8,
                batch_size=config.MODEL_SPECS.TRAIN.BATCH_SIZE,  # batch size for val is the same as train
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=g
            )
        else:
            data_loader_dict['valid'] = None

        # Test loader
        if config.DATA.TEST.DATA_PATH:
            test_data = test_loader(
                name="test",
                data_path=config.DATA.TEST.DATA_PATH,
                config_data=config.DATA.TEST, 
                total_config=config)
            data_loader_dict["test"] = DataLoader(
                dataset=test_data,
                num_workers=8,
                batch_size=config.MODEL_SPECS.TEST.BATCH_SIZE, # switch this to TEST/TRAIN #TODO There is a bug in the batch size of the test loader...
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=g
            )
        else:
            data_loader_dict['test'] = None

        # Pretrain loader
        if config.DATA.PRETRAIN.DATA_PATH and (config.MODEL_SPECS.MODEL.NAME == 'BigSmallMultitask_Pretrain' or config.MODEL_SPECS.MODEL.NAME == 'BigSmallMultitaskTrainer_MTTSCAN'):
            pretrain_data_loader = pretrain_loader(
                name="pretrain",
                data_path=config.DATA.PRETRAIN.DATA_PATH,
                config_data=config.DATA.PRETRAIN,
                total_config=config)
            data_loader_dict['pretrain'] = DataLoader(
                dataset=pretrain_data_loader,
                num_workers=8,
                batch_size=config.MODEL_SPECS.PRETRAIN.BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g
            )
        else:
            data_loader_dict['pretrain'] = None
        

    elif config.TOOLBOX_MODE == "signal_method":
        # signal method dataloader
        if config.DATA.SIGNAL.DATASET == "UBFC":
            signal_loader = data_loader.UBFCLoader.UBFCLoader
        elif config.DATA.SIGNAL.DATASET == "PURE":
            signal_loader = data_loader.PURELoader.PURELoader
        elif config.DATA.SIGNAL.DATASET == "SCAMPS":
            signal_loader = data_loader.SCAMPSLoader.SCAMPSLoader


        elif config.DATA.SIGNAL.DATASET == "BP4D":
            signal_loader = data_loader.BP4DLoader.BP4DLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, and SCAMPS.")

        signal_data = signal_loader(
            name="signal",
            data_path=config.DATA.SIGNAL.DATA_PATH,
            config_data=config.DATA.SIGNAL,
            total_config=config)
        data_loader_dict["signal"] = DataLoader(
            dataset=signal_data,
            num_workers=8,
            batch_size=1,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g
        )

    else:
        raise ValueError("Unsupported toolbox_mode! Currently support train_and_test, only_test, \
                          only_preprocess, or signal_method.")


    if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "only_test":
        train_test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "signal_method":
        signal_method_inference(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "only_preprocess":
        print('')
        print('Done preprocessing dataset... Goodbye!')
    else:
        print("TOOLBOX_MODE only supports 'train_and_test', 'only_test', or 'signal_method'")
