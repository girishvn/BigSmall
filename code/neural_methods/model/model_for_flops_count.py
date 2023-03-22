import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt 



# Big Small Base Model
class BigSmallBaseSinglePoolModel(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, 
                 dropout_rate1=0.25, dropout_rate2=0.5, dropout_rate3=0.5, pool_size=(2, 2),
                 nb_dense1=2048, nb_dense2=128, out_size=1):

        super(BigSmallBaseSinglePoolModel, self).__init__()

        print('')
        print('Init 3 Task BigSmallBaseModel')
        print('')

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.dropout_rate3 = dropout_rate3
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense1 = nb_dense1
        self.nb_dense2 = nb_dense2
        self.out_size = out_size
        self.out_size_au = 12 # TODO - Girish hardcoded
        self.out_size_bvp = 1 # TODO - Girish hardcoded
        self.out_size_resp = 1 # TODO - Girish hardcoded


        # Big Convolutional Layers
        self.big_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv5 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv6 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)

        # Big Avg Pooling / Dropout Layers
        self.big_avg_pooling1 = nn.AvgPool2d(self.pool_size)
        self.big_dropout1 = nn.Dropout(self.dropout_rate1)
        self.big_avg_pooling2 = nn.AvgPool2d(self.pool_size)
        self.big_dropout2 = nn.Dropout(self.dropout_rate2)
        self.big_avg_pooling3 = nn.AvgPool2d((4,4))
        self.big_dropout3 = nn.Dropout(self.dropout_rate3)
        
        # Small Convolutional Layers
        self.small_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.small_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.small_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.small_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)

        # Shared Layers
        # only cat

        # AU Fully Connected Layers 
        self.au_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.au_fc2 = nn.Linear(self.nb_dense2, self.out_size_au, bias=True)

        # BVP Fully Connected Layers 
        self.bvp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.bvp_fc2 = nn.Linear(self.nb_dense2, self.out_size_bvp, bias=True)

        # Resp Fully Connected Layers 
        self.resp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.resp_fc2 = nn.Linear(self.nb_dense2, self.out_size_resp, bias=True)


    def forward(self, x, y, params=None):

        big_input = x # big res 
        small_input = y # small res

        # Big Conv block 1
        b1 = nn.functional.relu(self.big_conv1(big_input))
        b2 = nn.functional.relu(self.big_conv2(b1))
        b3 = self.big_avg_pooling1(b2)
        b4 = self.big_dropout1(b3)

        # Big Conv block 2
        b5 = nn.functional.relu(self.big_conv3(b4))
        b6 = nn.functional.relu(self.big_conv4(b5))
        b7 = self.big_avg_pooling2(b6)
        b8 = self.big_dropout2(b7)

        # Big Conv block 3
        b9 = nn.functional.relu(self.big_conv5(b8))
        b10 = nn.functional.relu(self.big_conv6(b9))
        b11 = self.big_avg_pooling3(b10)
        b12 = self.big_dropout3(b11)

        # Small Conv block 1
        s1 = nn.functional.relu(self.small_conv1(small_input))
        s2 = nn.functional.relu(self.small_conv2(s1))

        # Small Conv block 2
        s3 = nn.functional.relu(self.small_conv3(s2))
        s4 = nn.functional.relu(self.small_conv4(s3))
        
        # Shared Layers
        concat = b12 + s4 # concat two layers (double the filters)
        share1 = concat.view(concat.size(0), -1) # flatten entire tensors

        # AU Output Layers
        aufc1 = nn.functional.relu(self.au_fc1(share1))
        au_out = self.au_fc2(aufc1)

        # BVP Output Layers
        bvpfc1 = nn.functional.relu(self.bvp_fc1(share1))
        bvp_out = self.bvp_fc2(bvpfc1)

        # Resp Output Layers
        respfc1 = nn.functional.relu(self.resp_fc1(share1))
        resp_out = self.resp_fc2(respfc1)

        return au_out, bvp_out, resp_out



class BigPathwaySinglePoolBaseModel(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, 
                 dropout_rate1=0.25, dropout_rate2=0.5, dropout_rate3=0.5, pool_size=(2, 2),
                 nb_dense1=2048, nb_dense2=128, out_size=1):

        super(BigPathwaySinglePoolBaseModel, self).__init__()

        print('')
        print('Init BigPathwaySinglePoolBaseModel')
        print('')

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.dropout_rate3 = dropout_rate3
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense1 = nb_dense1
        self.nb_dense2 = nb_dense2
        self.out_size = out_size

        # Big Convolutional Layers
        self.big_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv5 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv6 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)

        # Big Avg Pooling / Dropout Layers
        self.big_avg_pooling1 = nn.AvgPool2d(self.pool_size)
        self.big_dropout1 = nn.Dropout(self.dropout_rate1)
        self.big_avg_pooling2 = nn.AvgPool2d(self.pool_size)
        self.big_dropout2 = nn.Dropout(self.dropout_rate2)
        self.big_avg_pooling3 = nn.AvgPool2d((4,4))
        self.big_dropout3 = nn.Dropout(self.dropout_rate3)

        # Fully Connected Layers 
        self.out_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.out_fc2 = nn.Linear(self.nb_dense2, self.out_size, bias=True)


    def forward(self, inputs, params=None):

        big_input = inputs # big res 

        # Big Conv block 1
        b1 = nn.functional.relu(self.big_conv1(big_input))
        b2 = nn.functional.relu(self.big_conv2(b1))
        b3 = self.big_avg_pooling1(b2)
        b4 = self.big_dropout1(b3)

        # Big Conv block 2
        b5 = nn.functional.relu(self.big_conv3(b4))
        b6 = nn.functional.relu(self.big_conv4(b5))
        b7 = self.big_avg_pooling2(b6)
        b8 = self.big_dropout2(b7)

        # Big Conv block 3
        b9 = nn.functional.relu(self.big_conv5(b8))
        b10 = nn.functional.relu(self.big_conv6(b9))
        b11 = self.big_avg_pooling3(b10)
        b12 = self.big_dropout3(b11)
        
        # Output Layers
        outflat = b12.view(b12.size(0), -1) # flatten entire tensors
        outfc1 = nn.functional.relu(self.out_fc1(outflat))
        out = self.out_fc2(outfc1)

        return out



# Small Pathway Base Model
class SmallPathwayBaseModel(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, 
                 dropout_rate1=0.25, dropout_rate2=0.5, dropout_rate3=0.5, pool_size=(2, 2),
                 nb_dense1=2048, nb_dense2=128, out_size=1):

        super(SmallPathwayBaseModel, self).__init__()

        print('')
        print('Init SmallPathwayBaseModel')
        print('')

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.dropout_rate3 = dropout_rate3
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense1 = nb_dense1
        self.nb_dense2 = nb_dense2
        self.out_size = out_size

        
        # Small Convolutional Layers
        self.small_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.small_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.small_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.small_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)

        # Fully Connected Layers 
        self.out_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.out_fc2 = nn.Linear(self.nb_dense2, self.out_size, bias=True)


    def forward(self, inputs, params=None):

        small_input = inputs # small res

        # Small Conv block 1
        s1 = nn.functional.relu(self.small_conv1(small_input))
        s2 = nn.functional.relu(self.small_conv2(s1))

        # Small Conv block 2
        s3 = nn.functional.relu(self.small_conv3(s2))
        s4 = nn.functional.relu(self.small_conv4(s3))

        # Output Layers
        outflat = s4.view(s4.size(0), -1) # flatten entire tensors
        outfc1 = nn.functional.relu(self.out_fc1(outflat))
        out = self.out_fc2(outfc1)

        return out





# Big Small Base Model
class BigSmallConcatFusionModel(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, 
                 dropout_rate1=0.25, dropout_rate2=0.5, dropout_rate3=0.5, pool_size=(2, 2),
                 nb_dense1=2048, nb_dense2=128, out_size=1):

        super(BigSmallConcatFusionModel, self).__init__()

        print('')
        print('Init 3 Task BigSmallConcatFusionModel')
        print('')

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.dropout_rate3 = dropout_rate3
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense1 = nb_dense1
        self.nb_dense2 = nb_dense2
        self.out_size = out_size
        self.out_size_au = 12 # TODO - Girish hardcoded
        self.out_size_bvp = 1 # TODO - Girish hardcoded
        self.out_size_resp = 1 # TODO - Girish hardcoded


        # Big Convolutional Layers
        self.big_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv5 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv6 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)

        # Big Avg Pooling / Dropout Layers
        self.big_avg_pooling1 = nn.AvgPool2d(self.pool_size)
        self.big_dropout1 = nn.Dropout(self.dropout_rate1)
        self.big_avg_pooling2 = nn.AvgPool2d(self.pool_size)
        self.big_dropout2 = nn.Dropout(self.dropout_rate2)
        self.big_avg_pooling3 = nn.AvgPool2d((4,4))
        self.big_dropout3 = nn.Dropout(self.dropout_rate3)
        
        # Small Convolutional Layers
        self.small_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.small_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.small_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.small_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)

        # Shared Layers
        # only cat

        # AU Fully Connected Layers 
        self.au_fc1 = nn.Linear(10368, self.nb_dense2, bias=True)
        self.au_fc2 = nn.Linear(self.nb_dense2, self.out_size_au, bias=True)

        # BVP Fully Connected Layers 
        self.bvp_fc1 = nn.Linear(10368, self.nb_dense2, bias=True)
        self.bvp_fc2 = nn.Linear(self.nb_dense2, self.out_size_bvp, bias=True)

        # Resp Fully Connected Layers 
        self.resp_fc1 = nn.Linear(10368, self.nb_dense2, bias=True)
        self.resp_fc2 = nn.Linear(self.nb_dense2, self.out_size_resp, bias=True)


    def forward(self, x, y, params=None):

        big_input = x # big res 
        small_input = y # small res

        # Big Conv block 1
        b1 = nn.functional.relu(self.big_conv1(big_input))
        b2 = nn.functional.relu(self.big_conv2(b1))
        b3 = self.big_avg_pooling1(b2)
        b4 = self.big_dropout1(b3)

        # Big Conv block 2
        b5 = nn.functional.relu(self.big_conv3(b4))
        b6 = nn.functional.relu(self.big_conv4(b5))
        b7 = self.big_avg_pooling2(b6)
        b8 = self.big_dropout2(b7)

        # Big Conv block 3
        b9 = nn.functional.relu(self.big_conv5(b8))
        b10 = nn.functional.relu(self.big_conv6(b9))
        b11 = self.big_avg_pooling3(b10)
        b12 = self.big_dropout3(b11)

        # Small Conv block 1
        s1 = nn.functional.relu(self.small_conv1(small_input))
        s2 = nn.functional.relu(self.small_conv2(s1))

        # Small Conv block 2
        s3 = nn.functional.relu(self.small_conv3(s2))
        s4 = nn.functional.relu(self.small_conv4(s3))
        
        # Shared Layers
        concat = torch.cat((b12, s4), dim=1) # sum layers
        share1 = concat.view(concat.size(0), -1) # flatten entire tensors

        # AU Output Layers
        aufc1 = nn.functional.relu(self.au_fc1(share1))
        au_out = self.au_fc2(aufc1)

        # BVP Output Layers
        bvpfc1 = nn.functional.relu(self.bvp_fc1(share1))
        bvp_out = self.bvp_fc2(bvpfc1)

        # Resp Output Layers
        respfc1 = nn.functional.relu(self.resp_fc1(share1))
        resp_out = self.resp_fc2(respfc1)

        return au_out, bvp_out, resp_out






# Big Small Base Model
class BigSmall_Big2SmallFusion(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, 
                 dropout_rate1=0.25, dropout_rate2=0.5, dropout_rate3=0.5, pool_size=(2, 2),
                 nb_dense1=2048, nb_dense2=128, out_size=1):

        super(BigSmall_Big2SmallFusion, self).__init__()

        print('')
        print('Init 3 Task BigSmall_Big2SmallFusion')
        print('')

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.dropout_rate3 = dropout_rate3
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense1 = nb_dense1
        self.nb_dense2 = nb_dense2
        self.out_size = out_size
        self.out_size_au = 12 # TODO - Girish hardcoded
        self.out_size_bvp = 1 # TODO - Girish hardcoded
        self.out_size_resp = 1 # TODO - Girish hardcoded


        # Big Convolutional Layers
        self.big_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv5 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv6 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)

        # Big Avg Pooling / Dropout Layers
        self.big_avg_pooling1 = nn.AvgPool2d(self.pool_size)
        self.big_dropout1 = nn.Dropout(self.dropout_rate1)
        self.big_avg_pooling2 = nn.AvgPool2d(self.pool_size)
        self.big_dropout2 = nn.Dropout(self.dropout_rate2)
        self.big_avg_pooling3 = nn.AvgPool2d((4,4))
        self.big_dropout3 = nn.Dropout(self.dropout_rate3)
        
        # Small Convolutional Layers
        self.small_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.small_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.small_conv3 = nn.Conv2d(self.nb_filters1*2, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.small_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)

        # Shared Layers
        # only cat

        # AU Fully Connected Layers 
        self.au_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.au_fc2 = nn.Linear(self.nb_dense2, self.out_size_au, bias=True)

        # BVP Fully Connected Layers 
        self.bvp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.bvp_fc2 = nn.Linear(self.nb_dense2, self.out_size_bvp, bias=True)

        # Resp Fully Connected Layers 
        self.resp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.resp_fc2 = nn.Linear(self.nb_dense2, self.out_size_resp, bias=True)


    def forward(self, x, y, params=None):

        big_input = x # big res 
        small_input = y # small res

        # Big Conv block 1
        b1 = nn.functional.relu(self.big_conv1(big_input))
        b2 = nn.functional.relu(self.big_conv2(b1))
        b3 = self.big_avg_pooling1(b2)
        b4 = self.big_dropout1(b3)

        # Big Conv block 2
        b5 = nn.functional.relu(self.big_conv3(b4))
        b6 = nn.functional.relu(self.big_conv4(b5))
        b7 = self.big_avg_pooling2(b6)
        b8 = self.big_dropout2(b7)

        # Big Conv block 3
        b9 = nn.functional.relu(self.big_conv5(b8))
        b10 = nn.functional.relu(self.big_conv6(b9))
        b11 = self.big_avg_pooling3(b10)
        b12 = self.big_dropout3(b11)

        # Small Conv block 1
        s1 = nn.functional.relu(self.small_conv1(small_input))
        s2 = nn.functional.relu(self.small_conv2(s1))

        # Small Conv block 2
        s2 = torch.cat((s2, s2), dim=1)
        s3 = nn.functional.relu(self.small_conv3(s2))
        s4 = nn.functional.relu(self.small_conv4(s3))
        
        # Shared Layers
        concat = b12 + s4 # concat two layers (double the filters)
        share1 = concat.view(concat.size(0), -1) # flatten entire tensors

        # AU Output Layers
        aufc1 = nn.functional.relu(self.au_fc1(share1))
        au_out = self.au_fc2(aufc1)

        # BVP Output Layers
        bvpfc1 = nn.functional.relu(self.bvp_fc1(share1))
        bvp_out = self.bvp_fc2(bvpfc1)

        # Resp Output Layers
        respfc1 = nn.functional.relu(self.resp_fc1(share1))
        resp_out = self.resp_fc2(respfc1)

        return au_out, bvp_out, resp_out


    

    # Big Small Base Model
class BigSmall_Small2BigFusion(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, 
                 dropout_rate1=0.25, dropout_rate2=0.5, dropout_rate3=0.5, pool_size=(2, 2),
                 nb_dense1=2048, nb_dense2=128, out_size=1):

        super(BigSmall_Small2BigFusion, self).__init__()

        print('')
        print('Init 3 Task BigSmall_Small2BigFusion')
        print('')

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.dropout_rate3 = dropout_rate3
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense1 = nb_dense1
        self.nb_dense2 = nb_dense2
        self.out_size = out_size
        self.out_size_au = 12 # TODO - Girish hardcoded
        self.out_size_bvp = 1 # TODO - Girish hardcoded
        self.out_size_resp = 1 # TODO - Girish hardcoded


        # Big Convolutional Layers
        self.big_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv3 = nn.Conv2d(self.nb_filters1*2, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv5 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv6 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)

        # Big Avg Pooling / Dropout Layers
        self.big_avg_pooling1 = nn.AvgPool2d(self.pool_size)
        self.big_dropout1 = nn.Dropout(self.dropout_rate1)
        self.big_avg_pooling2 = nn.AvgPool2d(self.pool_size)
        self.big_dropout2 = nn.Dropout(self.dropout_rate2)
        self.big_avg_pooling3 = nn.AvgPool2d((4,4))
        self.big_dropout3 = nn.Dropout(self.dropout_rate3)
        
        # Small Convolutional Layers
        self.small_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.small_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.small_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.small_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)

        # Shared Layers
        # only cat

        # AU Fully Connected Layers 
        self.au_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.au_fc2 = nn.Linear(self.nb_dense2, self.out_size_au, bias=True)

        # BVP Fully Connected Layers 
        self.bvp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.bvp_fc2 = nn.Linear(self.nb_dense2, self.out_size_bvp, bias=True)

        # Resp Fully Connected Layers 
        self.resp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.resp_fc2 = nn.Linear(self.nb_dense2, self.out_size_resp, bias=True)


    def forward(self, x, y, params=None):

        big_input = x # big res 
        small_input = y # small res

        # Big Conv block 1
        b1 = nn.functional.relu(self.big_conv1(big_input))
        b2 = nn.functional.relu(self.big_conv2(b1))
        b3 = self.big_avg_pooling1(b2)
        b4 = self.big_dropout1(b3)


        b4 = torch.cat((b4, b4), dim=1)
        # Big Conv block 2
        b5 = nn.functional.relu(self.big_conv3(b4))
        b6 = nn.functional.relu(self.big_conv4(b5))
        b7 = self.big_avg_pooling2(b6)
        b8 = self.big_dropout2(b7)

        # Big Conv block 3
        b9 = nn.functional.relu(self.big_conv5(b8))
        b10 = nn.functional.relu(self.big_conv6(b9))
        b11 = self.big_avg_pooling3(b10)
        b12 = self.big_dropout3(b11)

        # Small Conv block 1
        s1 = nn.functional.relu(self.small_conv1(small_input))
        s2 = nn.functional.relu(self.small_conv2(s1))

        # Small Conv block 2
        s3 = nn.functional.relu(self.small_conv3(s2))
        s4 = nn.functional.relu(self.small_conv4(s3))
        
        # Shared Layers
        concat = b12 + s4 # concat two layers (double the filters)
        share1 = concat.view(concat.size(0), -1) # flatten entire tensors

        # AU Output Layers
        aufc1 = nn.functional.relu(self.au_fc1(share1))
        au_out = self.au_fc2(aufc1)

        # BVP Output Layers
        bvpfc1 = nn.functional.relu(self.bvp_fc1(share1))
        bvp_out = self.bvp_fc2(bvpfc1)

        # Resp Output Layers
        respfc1 = nn.functional.relu(self.resp_fc1(share1))
        resp_out = self.resp_fc2(respfc1)

        return au_out, bvp_out, resp_out