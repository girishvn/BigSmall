import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt 


#####################################################
############ Wrapping Time Shift Module #############
#####################################################
class WTSM(nn.Module):
    def __init__(self, n_segment=3, fold_div=3):
        super(WTSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, -1, :fold] = x[:, 0, :fold] # wrap left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, 0, fold: 2 * fold] = x[:, -1, fold: 2 * fold]  # wrap right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # no shift for final fold
        return out.view(nt, c, h, w)



#####################################################
################# Time Shift Module #################
#####################################################
class TSM(nn.Module):
    def __init__(self, n_segment=3, fold_div=3):
        super(TSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out.view(nt, c, h, w)

    

#########################################################
################# Attn Mask For Plotting #################
#########################################################
# class Attention_mask(nn.Module):
#     def __init__(self):
#         super(Attention_mask, self).__init__()

#     def forward(self, x):
#         xsum = torch.sum(x, dim=2, keepdim=True)
#         xsum = torch.sum(xsum, dim=3, keepdim=True)
#         xshape = tuple(x.size())
#         return x / xsum * xshape[2] * xshape[3] * 0.5

#     def get_config(self):
#         """May be generated manually. """
#         config = super(Attention_mask, self).get_config()
#         return config

def Attn_mask(x):
    xsum = torch.sum(x, dim=2, keepdim=True)
    xsum = torch.sum(xsum, dim=3, keepdim=True)
    xshape = tuple(x.size())
    return x / xsum * xshape[2] * xshape[3] * 0.5




#######################################################################################
############################### BigSmall SlowFast w/ WTSM #############################
#######################################################################################
class BigSmallSlowFastWTSM(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, 
                 dropout_rate1=0.25, dropout_rate2=0.5, dropout_rate3=0.5, pool_size=(2, 2),
                 nb_dense1=2048, nb_dense2=128, out_size=1, n_segment=3):

        super(BigSmallSlowFastWTSM, self).__init__()

        print('')
        print('Init 3 HERERREE Task BigSmallSlowFastWTSM')
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
        self.out_size_au = 12 # TODO -  hardcoded
        self.out_size_bvp = 1 # TODO -  hardcoded
        self.out_size_resp = 1 # TODO -  hardcoded

        self.n_segment = n_segment

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

        # TSM layers
        self.TSM_1 = WTSM(n_segment=self.n_segment)
        self.TSM_2 = WTSM(n_segment=self.n_segment)
        self.TSM_3 = WTSM(n_segment=self.n_segment)
        self.TSM_4 = WTSM(n_segment=self.n_segment)
        
        # Small Convolutional Layers
        self.small_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1,1), bias=True)

        # Shared Layers
        # only sum

        # AU Fully Connected Layers 
        self.au_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.au_fc2 = nn.Linear(self.nb_dense2, self.out_size_au, bias=True)

        # BVP Fully Connected Layers 
        self.bvp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.bvp_fc2 = nn.Linear(self.nb_dense2, self.out_size_bvp, bias=True)

        # Resp Fully Connected Layers 
        self.resp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.resp_fc2 = nn.Linear(self.nb_dense2, self.out_size_resp, bias=True)


    def forward(self, inputs, params=None):

        big_input = inputs[0] # big res 
        small_input = inputs[1] # small res

        # reshape Big 
        nt, c, h, w = big_input.size()
        n_batch = nt // self.n_segment
        big_input = big_input.view(n_batch, self.n_segment, c, h, w)
        big_input = torch.moveaxis(big_input, 1, 2) # color channel to idx 1, sequence channel to idx 2

        mode = 'FirstBig' # 'FirstBig', 'AvgBig', 'Random Big'

        # take Big 1st Frame ONLY 
        if mode == 'FirstBig':
            big_input = big_input[:, :, 0, :, :] # use only first frame in sequences 
        elif mode == 'AvgBig':
            big_input = torch.sum(big_input, dim=2) / self.n_segment # avg across color channel

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

        # Reformat Big Shape For Concat w/ Small Branch
        b13 = torch.stack((b12, b12, b12), 2) #TODO: this is hardcoded for num_segs = 3: change this...
        b14 = torch.moveaxis(b13, 1, 2)
        bN, bD, bC, bH, bW = b14.size()
        b15 = b14.reshape(int(bN*bD), bC, bH, bW)

        # Small Conv block 1
        s1 = self.TSM_1(small_input)
        s2 = nn.functional.relu(self.small_conv1(s1))
        s3 = self.TSM_2(s2)
        s4 = nn.functional.relu(self.small_conv2(s3))

        # Small Conv block 2
        s5 = self.TSM_3(s4)
        s6 = nn.functional.relu(self.small_conv3(s5))
        s7 = self.TSM_4(s6)
        s8 = nn.functional.relu(self.small_conv4(s7))

        # Shared Layers
        concat = b15 + s8 # sum layers

        # share1 = concat.view(concat.size(0), -1) # flatten entire tensors
        share1 = concat.reshape(concat.size(0), -1)

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



#######################################################################################
########################### BigSmall SlowFast w/ TSM Model ############################
#######################################################################################
class BigSmallSlowFastTSM(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, 
                 dropout_rate1=0.25, dropout_rate2=0.5, dropout_rate3=0.5, pool_size=(2, 2),
                 nb_dense1=2048, nb_dense2=128, out_size=1, n_segment=3):

        super(BigSmallSlowFastTSM, self).__init__()

        print('')
        print('Init 3 Task BigSmallSlowFastTSM')
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
        self.out_size_au = 12 # TODO -  hardcoded
        self.out_size_bvp = 1 # TODO -  hardcoded
        self.out_size_resp = 1 # TODO -  hardcoded

        self.n_segment = n_segment

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

        # TSM layers
        self.TSM_1 = TSM(n_segment=self.n_segment)
        self.TSM_2 = TSM(n_segment=self.n_segment)
        self.TSM_3 = TSM(n_segment=self.n_segment)
        self.TSM_4 = TSM(n_segment=self.n_segment)
        
        # Small Convolutional Layers
        self.small_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1,1), bias=True)

        # Shared Layers
        # only sum

        # AU Fully Connected Layers 
        self.au_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.au_fc2 = nn.Linear(self.nb_dense2, self.out_size_au, bias=True)

        # BVP Fully Connected Layers 
        self.bvp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.bvp_fc2 = nn.Linear(self.nb_dense2, self.out_size_bvp, bias=True)

        # Resp Fully Connected Layers 
        self.resp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.resp_fc2 = nn.Linear(self.nb_dense2, self.out_size_resp, bias=True)


    def forward(self, inputs, params=None):

        big_input = inputs[0] # big res 
        small_input = inputs[1] # small res

        # reshape Big 
        nt, c, h, w = big_input.size()
        n_batch = nt // self.n_segment
        big_input = big_input.view(n_batch, self.n_segment, c, h, w)
        big_input = torch.moveaxis(big_input, 1, 2) # color channel to idx 1, sequence channel to idx 2

        mode = 'FirstBig' # 'FirstBig', 'AvgBig', 'Random Big'

        # take Big 1st Frame ONLY 
        if mode == 'FirstBig':
            big_input = big_input[:, :, 0, :, :] # use only first frame in sequences 
        elif mode == 'AvgBig':
            big_input = torch.sum(big_input, dim=2) / self.n_segment # avg across color channel

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
        
        # Reformat Big Shape For Concat w/ Small Branch
        b13 = torch.stack((b12, b12, b12), 2)
        b14 = torch.moveaxis(b13, 1, 2)
        bN, bD, bC, bH, bW = b14.size()
        b15 = b14.reshape(int(bN*bD), bC, bH, bW)

        # Small Conv block 1
        s1 = self.TSM_1(small_input)
        s2 = nn.functional.relu(self.small_conv1(s1))
        s3 = self.TSM_2(s2)
        s4 = nn.functional.relu(self.small_conv2(s3))

        # Small Conv block 2
        s5 = self.TSM_3(s4)
        s6 = nn.functional.relu(self.small_conv3(s5))
        s7 = self.TSM_4(s6)
        s8 = nn.functional.relu(self.small_conv4(s7))

        # Shared Layers
        concat = b15 + s8 # sum pathways
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



#######################################################################################
############################### BigSmall SlowFast Model ###############################
#######################################################################################
class BigSmallSlowFast(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, 
                 dropout_rate1=0.25, dropout_rate2=0.5, dropout_rate3=0.5, pool_size=(2, 2),
                 nb_dense1=2048, nb_dense2=128, out_size=1, n_segment=3):

        super(BigSmallSlowFast, self).__init__()

        print('')
        print('Init 3 Task BigSmallSlowFast')
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
        self.out_size_au = 12 # TODO -  hardcoded
        self.out_size_bvp = 1 # TODO -  hardcoded
        self.out_size_resp = 1 # TODO -  hardcoded

        self.n_segment = n_segment

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
        self.small_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1,1), bias=True)

        # Shared Layers
        # only sum

        # AU Fully Connected Layers 
        self.au_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.au_fc2 = nn.Linear(self.nb_dense2, self.out_size_au, bias=True)

        # BVP Fully Connected Layers 
        self.bvp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.bvp_fc2 = nn.Linear(self.nb_dense2, self.out_size_bvp, bias=True)

        # Resp Fully Connected Layers 
        self.resp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.resp_fc2 = nn.Linear(self.nb_dense2, self.out_size_resp, bias=True)


    def forward(self, inputs, params=None):

        big_input = inputs[0] # big res 
        small_input = inputs[1] # small res

        # reshape Big 
        nt, c, h, w = big_input.size()
        n_batch = nt // self.n_segment
        big_input = big_input.view(n_batch, self.n_segment, c, h, w)
        big_input = torch.moveaxis(big_input, 1, 2) # color channel to idx 1, sequence channel to idx 2

        mode = 'FirstBig' # 'FirstBig', 'AvgBig', 'Random Big'

        # take Big 1st Frame ONLY 
        if mode == 'FirstBig':
            big_input = big_input[:, :, 0, :, :] # use only first frame in sequences 
        elif mode == 'AvgBig':
            big_input = torch.sum(big_input, dim=2) / self.n_segment # avg across color channel

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
        
        # Reformat Big Shape For Concat w/ Small Branch
        b13 = torch.stack((b12, b12, b12), 2)
        b14 = torch.moveaxis(b13, 1, 2)
        bN, bD, bC, bH, bW = b14.size()
        b15 = b14.reshape(int(bN*bD), bC, bH, bW)

        # Small Conv block 1
        s1 = nn.functional.relu(self.small_conv1(small_input))
        s2 = nn.functional.relu(self.small_conv2(s1))

        # Small Conv block 2
        s3 = nn.functional.relu(self.small_conv3(s2))
        s4 = nn.functional.relu(self.small_conv4(s3))

        # Shared Layers
        concat = b15 + s4 # sum pathways
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



#######################################################################################
############################### Big Small w/ WTSM Model ###############################
#######################################################################################
class BigSmallWTSM(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, 
                 dropout_rate1=0.25, dropout_rate2=0.5, dropout_rate3=0.5, pool_size=(2, 2),
                 nb_dense1=2048, nb_dense2=128, out_size=1, n_segment=3):

        super(BigSmallWTSM, self).__init__()

        print('')
        print('Init 3 Task BigSmallWTSM')
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
        self.out_size_au = 12 # TODO -  hardcoded
        self.out_size_bvp = 1 # TODO -  hardcoded
        self.out_size_resp = 1 # TODO -  hardcoded


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

        # TSM layers
        self.TSM_1 = WTSM(n_segment=n_segment)
        self.TSM_2 = WTSM(n_segment=n_segment)
        self.TSM_3 = WTSM(n_segment=n_segment)
        self.TSM_4 = WTSM(n_segment=n_segment)
        
        # Small Convolutional Layers
        self.small_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.small_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.small_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.small_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)

        # Shared Layers
        # only Sum

        # AU Fully Connected Layers 
        self.au_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.au_fc2 = nn.Linear(self.nb_dense2, self.out_size_au, bias=True)

        # BVP Fully Connected Layers 
        self.bvp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.bvp_fc2 = nn.Linear(self.nb_dense2, self.out_size_bvp, bias=True)

        # Resp Fully Connected Layers 
        self.resp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.resp_fc2 = nn.Linear(self.nb_dense2, self.out_size_resp, bias=True)


    def forward(self, inputs, params=None):

        big_input = inputs[0] # big res 
        small_input = inputs[1] # small res

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
        s1 = self.TSM_1(small_input)
        s2 = nn.functional.relu(self.small_conv1(s1))
        s3 = self.TSM_2(s2)
        s4 = nn.functional.relu(self.small_conv2(s3))

        # Small Conv block 2
        s5 = self.TSM_3(s4)
        s6 = nn.functional.relu(self.small_conv3(s5))
        s7 = self.TSM_4(s6)
        s8 = nn.functional.relu(self.small_conv4(s7))
        
        # Shared Layers
        concat = b12 + s8 # sum two layers
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



#######################################################################################
################################# Big Small Base Model ################################
#######################################################################################

class BigSmallBaseModel(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, 
                 dropout_rate1=0.25, dropout_rate2=0.5, dropout_rate3=0.5, pool_size=(2, 2),
                 nb_dense1=2048, nb_dense2=128, out_size=1):

        super(BigSmallBaseModel, self).__init__()

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
        self.out_size_au = 12 # TODO -  hardcoded
        self.out_size_bvp = 1 # TODO -  hardcoded
        self.out_size_resp = 1 # TODO -  hardcoded


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
        # only sum

        # AU Fully Connected Layers 
        self.au_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.au_fc2 = nn.Linear(self.nb_dense2, self.out_size_au, bias=True)

        # BVP Fully Connected Layers 
        self.bvp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.bvp_fc2 = nn.Linear(self.nb_dense2, self.out_size_bvp, bias=True)

        # Resp Fully Connected Layers 
        self.resp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.resp_fc2 = nn.Linear(self.nb_dense2, self.out_size_resp, bias=True)


    def forward(self, inputs, params=None):

        big_input = inputs[0] # big res 
        small_input = inputs[1] # small res

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
        concat = b12 + s4 # sum layers
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



#######################################################################################
################################## Big Pathway Base  ##################################
#######################################################################################
class BigPathwayBaseModel(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, 
                 dropout_rate1=0.25, dropout_rate2=0.5, dropout_rate3=0.5, pool_size=(2, 2),
                 nb_dense1=2048, nb_dense2=128, out_size=1):

        super(BigPathwayBaseModel, self).__init__()

        print('')
        print('Init BigPathwayBaseModel')
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

        big_input = inputs[0] # big res 
        small_input = inputs[1] # small res

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



#######################################################################################
################################# Small Pathway Base  #################################
#######################################################################################
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

        big_input = inputs[0] # big res 
        small_input = inputs[1] # small res

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


#######################################################################################
#######################################################################################
############################## ALTERNATIVE FUSION MODELS  #############################
#######################################################################################
#######################################################################################

#############################################################################################################################
############################### BigSmall SlowFast w/ WTSM w/ Concat Fusion Instead of Summation #############################
#############################################################################################################################
class BSSFWTSM_ConcatFusion(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, 
                 dropout_rate1=0.25, dropout_rate2=0.5, dropout_rate3=0.5, pool_size=(2, 2),
                 nb_dense1=2048, nb_dense2=128, out_size=1, n_segment=3):

        super(BSSFWTSM_ConcatFusion, self).__init__()

        print('')
        print('Init 3 Task BSSFWTSM_ConcatFusion')
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
        self.out_size_au = 12 # TODO -  hardcoded
        self.out_size_bvp = 1 # TODO -  hardcoded
        self.out_size_resp = 1 # TODO -  hardcoded

        self.n_segment = n_segment

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

        # TSM layers
        self.TSM_1 = WTSM(n_segment=self.n_segment)
        self.TSM_2 = WTSM(n_segment=self.n_segment)
        self.TSM_3 = WTSM(n_segment=self.n_segment)
        self.TSM_4 = WTSM(n_segment=self.n_segment)
        
        # Small Convolutional Layers
        self.small_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1,1), bias=True)

        # Shared Layers
        # only sum

        # AU Fully Connected Layers 
        self.au_fc1 = nn.Linear(10368, self.nb_dense2, bias=True)
        self.au_fc2 = nn.Linear(self.nb_dense2, self.out_size_au, bias=True)

        # BVP Fully Connected Layers 
        self.bvp_fc1 = nn.Linear(10368, self.nb_dense2, bias=True)
        self.bvp_fc2 = nn.Linear(self.nb_dense2, self.out_size_bvp, bias=True)

        # Resp Fully Connected Layers 
        self.resp_fc1 = nn.Linear(10368, self.nb_dense2, bias=True)
        self.resp_fc2 = nn.Linear(self.nb_dense2, self.out_size_resp, bias=True)


    def forward(self, inputs, params=None):

        big_input = inputs[0] # big res 
        small_input = inputs[1] # small res

        # reshape Big 
        nt, c, h, w = big_input.size()
        n_batch = nt // self.n_segment
        big_input = big_input.view(n_batch, self.n_segment, c, h, w)
        big_input = torch.moveaxis(big_input, 1, 2) # color channel to idx 1, sequence channel to idx 2

        mode = 'FirstBig' # 'FirstBig', 'AvgBig', 'Random Big'

        # take Big 1st Frame ONLY 
        if mode == 'FirstBig':
            big_input = big_input[:, :, 0, :, :] # use only first frame in sequences 
        elif mode == 'AvgBig':
            big_input = torch.sum(big_input, dim=2) / self.n_segment # avg across color channel

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
        
        # Reformat Big Shape For Concat w/ Small Branch
        b13 = torch.stack((b12, b12, b12), 2)
        b14 = torch.moveaxis(b13, 1, 2)
        bN, bD, bC, bH, bW = b14.size()
        b15 = b14.reshape(int(bN*bD), bC, bH, bW)

        # Small Conv block 1
        s1 = self.TSM_1(small_input)
        s2 = nn.functional.relu(self.small_conv1(s1))
        s3 = self.TSM_2(s2)
        s4 = nn.functional.relu(self.small_conv2(s3))

        # Small Conv block 2
        s5 = self.TSM_3(s4)
        s6 = nn.functional.relu(self.small_conv3(s5))
        s7 = self.TSM_4(s6)
        s8 = nn.functional.relu(self.small_conv4(s7))

        # Shared Layers
        concat = torch.cat((b15, s8), dim=1) # sum layers

        # share1 = concat.view(concat.size(0), -1) # flatten entire tensors
        share1 = concat.reshape(concat.size(0), -1)

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



###########################################################################################################################################
############################### BigSmall SlowFast w/ WTSM w/ Bir Dir Lateral Connection After First Conv Block ############################
###########################################################################################################################################


class BSSFWTSM_BiDirLatConn(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, 
                 dropout_rate1=0.25, dropout_rate2=0.5, dropout_rate3=0.5, pool_size=(2, 2),
                 nb_dense1=2048, nb_dense2=128, out_size=1, n_segment=3):

        super(BSSFWTSM_BiDirLatConn, self).__init__()

        print('')
        print('Init 3 Task BSSFWTSM_BiDirLatConn')
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
        self.out_size_au = 12 # TODO -  hardcoded
        self.out_size_bvp = 1 # TODO -  hardcoded
        self.out_size_resp = 1 # TODO -  hardcoded

        self.n_segment = n_segment

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

        # TSM layers
        self.TSM_1 = WTSM(n_segment=self.n_segment)
        self.TSM_2 = WTSM(n_segment=self.n_segment)
        self.TSM_3 = WTSM(n_segment=self.n_segment)
        self.TSM_4 = WTSM(n_segment=self.n_segment)
        
        # Small Convolutional Layers
        self.small_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv3 = nn.Conv2d(self.nb_filters1*2, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1,1), bias=True)

        # Shared Layers
        # only sum

        # AU Fully Connected Layers 
        self.au_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.au_fc2 = nn.Linear(self.nb_dense2, self.out_size_au, bias=True)

        # BVP Fully Connected Layers 
        self.bvp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.bvp_fc2 = nn.Linear(self.nb_dense2, self.out_size_bvp, bias=True)

        # Resp Fully Connected Layers 
        self.resp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.resp_fc2 = nn.Linear(self.nb_dense2, self.out_size_resp, bias=True)


    def forward(self, inputs, params=None):

        big_input = inputs[0] # big res 
        small_input = inputs[1] # small res

        # reshape Big 
        nt, c, h, w = big_input.size()
        n_batch = nt // self.n_segment
        big_input = big_input.view(n_batch, self.n_segment, c, h, w)
        big_input = torch.moveaxis(big_input, 1, 2) # color channel to idx 1, sequence channel to idx 2

        mode = 'FirstBig' # 'FirstBig', 'AvgBig', 'Random Big'

        # take Big 1st Frame ONLY 
        if mode == 'FirstBig':
            big_input = big_input[:, :, 0, :, :] # use only first frame in sequences 
        elif mode == 'AvgBig':
            big_input = torch.sum(big_input, dim=2) / self.n_segment # avg across color channel

        # Big Conv block 1
        b1 = nn.functional.relu(self.big_conv1(big_input))
        b2 = nn.functional.relu(self.big_conv2(b1))
        b3 = self.big_avg_pooling1(b2)
        b4 = self.big_dropout1(b3)

        # Small Conv block 1
        s1 = self.TSM_1(small_input)
        s2 = nn.functional.relu(self.small_conv1(s1))
        s3 = self.TSM_2(s2)
        s4 = nn.functional.relu(self.small_conv2(s3))

        # Big + Small Bi Directional Lateral Connection
        # Spatially downsample big
        # Temporal downsample small

        # Big2Small Conn
        b2s1 = nn.functional.avg_pool2d(b4, (8,8))
        b2s2 = torch.stack((b2s1, b2s1, b2s1), 2) # this is hardcoded for the clip length... need to change this eventually
        b2s3 = torch.moveaxis(b2s2, 1, 2)
        bN, bD, bC, bH, bW = b2s3.size()
        b2s4 = b2s3.reshape(int(bN*bD), bC, bH, bW)
        b2s5 = torch.cat((b2s4, s4), 1) # concat accross channels

        # Small2Big Conn
        nt, c, h, w = s4.size()
        n_batch = nt // self.n_segment
        s2b1 = s4.view(n_batch, self.n_segment, c, h, w)
        s2b2 = torch.moveaxis(s2b1, 1, 2) # color channel to idx 1, sequence channel to idx 2
        s2b3 = torch.mean(s2b2,dim=2)
        s2b4 = nn.functional.interpolate(s2b3, scale_factor=(8,8))
        s2b5 = torch.cat((s2b4, b4), 1) # concat accross channels

        # Big Conv block 2
        b5 = nn.functional.relu(self.big_conv3(s2b5))
        b6 = nn.functional.relu(self.big_conv4(b5))
        b7 = self.big_avg_pooling2(b6)
        b8 = self.big_dropout2(b7)

        # Big Conv block 3
        b9 = nn.functional.relu(self.big_conv5(b8))
        b10 = nn.functional.relu(self.big_conv6(b9))
        b11 = self.big_avg_pooling3(b10)
        b12 = self.big_dropout3(b11)
        
        # Reformat Big Shape For Concat w/ Small Branch
        b13 = torch.stack((b12, b12, b12), 2)
        b14 = torch.moveaxis(b13, 1, 2)
        bN, bD, bC, bH, bW = b14.size()
        b15 = b14.reshape(int(bN*bD), bC, bH, bW)

        # Small Conv block 2
        s5 = self.TSM_3(b2s5)
        s6 = nn.functional.relu(self.small_conv3(s5))
        s7 = self.TSM_4(s6)
        s8 = nn.functional.relu(self.small_conv4(s7))

        # Shared Layers
        concat = b15 + s8 # sum layers

        # share1 = concat.view(concat.size(0), -1) # flatten entire tensors
        share1 = concat.reshape(concat.size(0), -1)

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


    
###########################################################################################################################################
############################## BigSmall SlowFast w/ WTSM w/ Big2Small Lateral Connection After First Conv Block ###########################
###########################################################################################################################################


class BSSFWTSM_Big2SmallLatConn(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, 
                 dropout_rate1=0.25, dropout_rate2=0.5, dropout_rate3=0.5, pool_size=(2, 2),
                 nb_dense1=2048, nb_dense2=128, out_size=1, n_segment=3):

        super(BSSFWTSM_Big2SmallLatConn, self).__init__()

        print('')
        print('Init 3 Task BSSFWTSM_Big2SmallLatConn')
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
        self.out_size_au = 12 # TODO -  hardcoded
        self.out_size_bvp = 1 # TODO -  hardcoded
        self.out_size_resp = 1 # TODO -  hardcoded

        self.n_segment = n_segment

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

        # TSM layers
        self.TSM_1 = WTSM(n_segment=self.n_segment)
        self.TSM_2 = WTSM(n_segment=self.n_segment)
        self.TSM_3 = WTSM(n_segment=self.n_segment)
        self.TSM_4 = WTSM(n_segment=self.n_segment)
        
        # Small Convolutional Layers
        self.small_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv3 = nn.Conv2d(self.nb_filters1*2, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1,1), bias=True)

        # Shared Layers
        # only sum

        # AU Fully Connected Layers 
        self.au_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.au_fc2 = nn.Linear(self.nb_dense2, self.out_size_au, bias=True)

        # BVP Fully Connected Layers 
        self.bvp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.bvp_fc2 = nn.Linear(self.nb_dense2, self.out_size_bvp, bias=True)

        # Resp Fully Connected Layers 
        self.resp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.resp_fc2 = nn.Linear(self.nb_dense2, self.out_size_resp, bias=True)


    def forward(self, inputs, params=None):

        big_input = inputs[0] # big res 
        small_input = inputs[1] # small res

        # reshape Big 
        nt, c, h, w = big_input.size()
        n_batch = nt // self.n_segment
        big_input = big_input.view(n_batch, self.n_segment, c, h, w)
        big_input = torch.moveaxis(big_input, 1, 2) # color channel to idx 1, sequence channel to idx 2

        mode = 'FirstBig' # 'FirstBig', 'AvgBig', 'Random Big'

        # take Big 1st Frame ONLY 
        if mode == 'FirstBig':
            big_input = big_input[:, :, 0, :, :] # use only first frame in sequences 
        elif mode == 'AvgBig':
            big_input = torch.sum(big_input, dim=2) / self.n_segment # avg across color channel

        # Big Conv block 1
        b1 = nn.functional.relu(self.big_conv1(big_input))
        b2 = nn.functional.relu(self.big_conv2(b1))
        b3 = self.big_avg_pooling1(b2)
        b4 = self.big_dropout1(b3)

        # Small Conv block 1
        s1 = self.TSM_1(small_input)
        s2 = nn.functional.relu(self.small_conv1(s1))
        s3 = self.TSM_2(s2)
        s4 = nn.functional.relu(self.small_conv2(s3))

        # Big2Small Conn
        b2s1 = nn.functional.avg_pool2d(b4, (8,8))
        b2s2 = torch.stack((b2s1, b2s1, b2s1), 2) # this is hardcoded for the clip length... need to change this eventually
        b2s3 = torch.moveaxis(b2s2, 1, 2)
        bN, bD, bC, bH, bW = b2s3.size()
        b2s4 = b2s3.reshape(int(bN*bD), bC, bH, bW)
        b2s5 = torch.cat((b2s4, s4), 1) # concat accross channels

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
        
        # Reformat Big Shape For Concat w/ Small Branch
        b13 = torch.stack((b12, b12, b12), 2)
        b14 = torch.moveaxis(b13, 1, 2)
        bN, bD, bC, bH, bW = b14.size()
        b15 = b14.reshape(int(bN*bD), bC, bH, bW)

        # Small Conv block 2
        s5 = self.TSM_3(b2s5)
        s6 = nn.functional.relu(self.small_conv3(s5))
        s7 = self.TSM_4(s6)
        s8 = nn.functional.relu(self.small_conv4(s7))

        # Shared Layers
        concat = b15 + s8 # sum layers

        # share1 = concat.view(concat.size(0), -1) # flatten entire tensors
        share1 = concat.reshape(concat.size(0), -1)

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



###########################################################################################################################################
############################### BigSmall SlowFast w/ WTSM w/ Bir Dir Lateral Connection After First Conv Block ############################
###########################################################################################################################################


class BSSFWTSM_Small2BigLatConn(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, 
                 dropout_rate1=0.25, dropout_rate2=0.5, dropout_rate3=0.5, pool_size=(2, 2),
                 nb_dense1=2048, nb_dense2=128, out_size=1, n_segment=3):

        super(BSSFWTSM_Small2BigLatConn, self).__init__()

        print('')
        print('Init 3 Task BSSFWTSM_Small2BigLatConn')
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
        self.out_size_au = 12 # TODO -  hardcoded
        self.out_size_bvp = 1 # TODO -  hardcoded
        self.out_size_resp = 1 # TODO -  hardcoded

        self.n_segment = n_segment

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

        # TSM layers
        self.TSM_1 = WTSM(n_segment=self.n_segment)
        self.TSM_2 = WTSM(n_segment=self.n_segment)
        self.TSM_3 = WTSM(n_segment=self.n_segment)
        self.TSM_4 = WTSM(n_segment=self.n_segment)
        
        # Small Convolutional Layers
        self.small_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1,1), bias=True)

        # Shared Layers
        # only sum

        # AU Fully Connected Layers 
        self.au_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.au_fc2 = nn.Linear(self.nb_dense2, self.out_size_au, bias=True)

        # BVP Fully Connected Layers 
        self.bvp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.bvp_fc2 = nn.Linear(self.nb_dense2, self.out_size_bvp, bias=True)

        # Resp Fully Connected Layers 
        self.resp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.resp_fc2 = nn.Linear(self.nb_dense2, self.out_size_resp, bias=True)


    def forward(self, inputs, params=None):

        big_input = inputs[0] # big res 
        small_input = inputs[1] # small res

        # reshape Big 
        nt, c, h, w = big_input.size()
        n_batch = nt // self.n_segment
        big_input = big_input.view(n_batch, self.n_segment, c, h, w)
        big_input = torch.moveaxis(big_input, 1, 2) # color channel to idx 1, sequence channel to idx 2

        mode = 'FirstBig' # 'FirstBig', 'AvgBig', 'Random Big'

        # take Big 1st Frame ONLY 
        if mode == 'FirstBig':
            big_input = big_input[:, :, 0, :, :] # use only first frame in sequences 
        elif mode == 'AvgBig':
            big_input = torch.sum(big_input, dim=2) / self.n_segment # avg across color channel

        # Big Conv block 1
        b1 = nn.functional.relu(self.big_conv1(big_input))
        b2 = nn.functional.relu(self.big_conv2(b1))
        b3 = self.big_avg_pooling1(b2)
        b4 = self.big_dropout1(b3)

        # Small Conv block 1
        s1 = self.TSM_1(small_input)
        s2 = nn.functional.relu(self.small_conv1(s1))
        s3 = self.TSM_2(s2)
        s4 = nn.functional.relu(self.small_conv2(s3))

        # Big + Small Bi Directional Lateral Connection
        # Spatially downsample big
        # Temporal downsample small

        # Small2Big Conn
        nt, c, h, w = s4.size()
        n_batch = nt // self.n_segment
        s2b1 = s4.view(n_batch, self.n_segment, c, h, w)
        s2b2 = torch.moveaxis(s2b1, 1, 2) # color channel to idx 1, sequence channel to idx 2
        s2b3 = torch.mean(s2b2,dim=2)
        s2b4 = nn.functional.interpolate(s2b3, scale_factor=(8,8))
        s2b5 = torch.cat((s2b4, b4), 1) # concat accross channels

        # Big Conv block 2
        b5 = nn.functional.relu(self.big_conv3(s2b5))
        b6 = nn.functional.relu(self.big_conv4(b5))
        b7 = self.big_avg_pooling2(b6)
        b8 = self.big_dropout2(b7)

        # Big Conv block 3
        b9 = nn.functional.relu(self.big_conv5(b8))
        b10 = nn.functional.relu(self.big_conv6(b9))
        b11 = self.big_avg_pooling3(b10)
        b12 = self.big_dropout3(b11)
        
        # Reformat Big Shape For Concat w/ Small Branch
        b13 = torch.stack((b12, b12, b12), 2)
        b14 = torch.moveaxis(b13, 1, 2)
        bN, bD, bC, bH, bW = b14.size()
        b15 = b14.reshape(int(bN*bD), bC, bH, bW)

        # Small Conv block 2
        s5 = self.TSM_3(s4)
        s6 = nn.functional.relu(self.small_conv3(s5))
        s7 = self.TSM_4(s6)
        s8 = nn.functional.relu(self.small_conv4(s7))

        # Shared Layers
        concat = b15 + s8 # sum layers

        # share1 = concat.view(concat.size(0), -1) # flatten entire tensors
        share1 = concat.reshape(concat.size(0), -1)

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



#######################################################################################
#######################################################################################
################################# EXPERIMENTAL MODELS  ################################
#######################################################################################
#######################################################################################

#######################################################################################
############################### BigSmall SlowFast w/ WTSM #############################
#######################################################################################

class BigSmallSlowFastWTSM_GrayScaleBig(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, 
                 dropout_rate1=0.25, dropout_rate2=0.5, dropout_rate3=0.5, pool_size=(2, 2),
                 nb_dense1=2048, nb_dense2=128, out_size=1, n_segment=3):

        super(BigSmallSlowFastWTSM_GrayScaleBig, self).__init__()

        print('')
        print('Init 3 Task BigSmallSlowFastWTSM_GrayScaleBig')
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
        self.out_size_au = 12 # TODO -  hardcoded
        self.out_size_bvp = 1 # TODO -  hardcoded
        self.out_size_resp = 1 # TODO -  hardcoded

        self.n_segment = n_segment

        # Big Convolutional Layers
        self.big_conv1 = nn.Conv2d(int(self.in_channels/3), self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
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

        # TSM layers
        self.TSM_1 = WTSM(n_segment=self.n_segment)
        self.TSM_2 = WTSM(n_segment=self.n_segment)
        self.TSM_3 = WTSM(n_segment=self.n_segment)
        self.TSM_4 = WTSM(n_segment=self.n_segment)
        
        # Small Convolutional Layers
        self.small_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1,1), bias=True)

        # Shared Layers
        # only sum

        # AU Fully Connected Layers 
        self.au_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.au_fc2 = nn.Linear(self.nb_dense2, self.out_size_au, bias=True)

        # BVP Fully Connected Layers 
        self.bvp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.bvp_fc2 = nn.Linear(self.nb_dense2, self.out_size_bvp, bias=True)

        # Resp Fully Connected Layers 
        self.resp_fc1 = nn.Linear(5184, self.nb_dense2, bias=True)
        self.resp_fc2 = nn.Linear(self.nb_dense2, self.out_size_resp, bias=True)


    def forward(self, inputs, params=None):

        big_input = inputs[0] # big res 
        small_input = inputs[1] # small res

        # gray scale big
        big_input = torch.mean(big_input, 1, keepdim=True) 


        # reshape Big 
        nt, c, h, w = big_input.size()
        n_batch = nt // self.n_segment
        big_input = big_input.view(n_batch, self.n_segment, c, h, w)
        big_input = torch.moveaxis(big_input, 1, 2) # color channel to idx 1, sequence channel to idx 2

        mode = 'FirstBig' # 'FirstBig', 'AvgBig', 'Random Big'

        # take Big 1st Frame ONLY 
        if mode == 'FirstBig':
            big_input = big_input[:, :, 0, :, :] # use only first frame in sequences 
        elif mode == 'AvgBig':
            big_input = torch.sum(big_input, dim=2) / self.n_segment # avg across color channel

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

        # Reformat Big Shape For Concat w/ Small Branch
        b13 = torch.stack((b12, b12, b12), 2) #TODO: this is hardcoded for num_segs = 3: change this...
        b14 = torch.moveaxis(b13, 1, 2)
        bN, bD, bC, bH, bW = b14.size()
        b15 = b14.reshape(int(bN*bD), bC, bH, bW)

        # Small Conv block 1
        s1 = self.TSM_1(small_input)
        s2 = nn.functional.relu(self.small_conv1(s1))
        s3 = self.TSM_2(s2)
        s4 = nn.functional.relu(self.small_conv2(s3))

        # Small Conv block 2
        s5 = self.TSM_3(s4)
        s6 = nn.functional.relu(self.small_conv3(s5))
        s7 = self.TSM_4(s6)
        s8 = nn.functional.relu(self.small_conv4(s7))

        # Shared Layers
        concat = b15 + s8 # sum layers
        # concat = b13 + s8 #TODO temp upsample...

        # share1 = concat.view(concat.size(0), -1) # flatten entire tensors
        share1 = concat.reshape(concat.size(0), -1)

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