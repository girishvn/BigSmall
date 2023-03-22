import torch
#from dataset.data_loader.BaseLoader import Neg_PearsonLoss
from neural_methods.loss import NegPearsonLoss
from neural_methods.loss import PhysNetNegPearsonLoss


def set_loss(loss_name, device):

    loss = None

    if loss_name == 'MSE':
        loss = torch.nn.MSELoss()

    # DOES NOT WORK YET - NEED TO FIND AWAY TO PIPE SIGMOID INTO BCE # TODO Girish
    elif loss_name == 'BCE' and False: # Binary Cross Entropy Loss
        loss = torch.nn.BCELoss()

    elif loss_name == 'BCEWithLogits': # Binary Cross Entropy Loss
        loss = torch.nn.BCEWithLogitsLoss()

    # BP4D AU Specific
    elif loss_name == 'BCEWithLogitsBP4DAU': # Binary Cross Entropy Loss
        print('Using Loss of Type: BCEWithLogitsBP4DAU')

        G = 1
        AU_weights = [9.64177799975028*G, 11.738006277088626*G, 16.771059216013345*G, 1.045773675935816,
                           0.5324451157020336, 0.5636523749243216, 0.7537758755504341, 0.6870546318289786,
                           8.505911220165068*G, 6.935754189944134*G, 5.030993489951882*G, 25.0045766590389*G]

        AU_weights = torch.as_tensor(AU_weights).to(device)
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=AU_weights)

    else: 
        raise ValueError('Loss not supported, suppoert loss functions include: \
                                     MSE, BCE, BCEWithLogitsBP4DAU')


    return loss