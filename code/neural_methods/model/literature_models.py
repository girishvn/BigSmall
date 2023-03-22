import torch
from torch import nn
import torchvision.transforms as T



#######################################################################################
######################################## DRML  ########################################
#######################################################################################

class RegionLayer(nn.Module):
    def __init__(self, in_channels, grid=(8, 8)):
        super(RegionLayer, self).__init__()

        self.in_channels = in_channels
        self.grid = grid

        self.region_layers = dict()

        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                module_name = 'region_conv_%d_%d' % (i, j)
                self.region_layers[module_name] = nn.Sequential(
                    nn.BatchNorm2d(self.in_channels),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                              kernel_size=3, stride=1, padding=1)
                )
                self.add_module(name=module_name, module=self.region_layers[module_name])

    def forward(self, x):
        """
        :param x:   (b, c, h, w)
        :return:    (b, c, h, w)
        """

        batch_size, _, height, width = x.size()

        input_row_list = torch.split(x, split_size_or_sections=height//(self.grid[0]-1), dim=2)
        output_row_list = []

        for i, row in enumerate(input_row_list):
            input_grid_list_of_a_row = torch.split(row, split_size_or_sections=width//(self.grid[1]-1), dim=3)
            output_grid_list_of_a_row = []

            for j, grid in enumerate(input_grid_list_of_a_row):
                module_name = 'region_conv_%d_%d' % (i, j)
                # print(module_name)
                # print(i,j)
                grid = self.region_layers[module_name](grid.contiguous()) + grid
                output_grid_list_of_a_row.append(grid)

            output_row = torch.cat(output_grid_list_of_a_row, dim=3)
            output_row_list.append(output_row)


        output = torch.cat(output_row_list, dim=2)
        return output


class DRML(nn.Module):
    def __init__(self, class_number=12):
        super(DRML, self).__init__()

        print('Init DRML... pls god work...')

        self.class_number = class_number

        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=11, stride=1),
            RegionLayer(in_channels=32, grid=(8, 8)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=8, stride=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=8,),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=6, stride=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=6400, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=2048, out_features=class_number)
        )

    def forward(self, data):
        """
        :param x:   (b, c, h, w)
        :return:    (b, class_number)
        """

        x = data[0]
        batch_size = x.size(0)
        output = self.extractor(x)
        output = output.view(batch_size, -1)
        output = self.classifier(output)
        return output

#######################################################################################
####################################### AlexNet  ######################################
#######################################################################################
class AlexNet(nn.Module):
    def __init__(self, num_classes = 12, dropout = 0.5): #0.5
        super().__init__()

        print('INIT AU AlexNet')

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2304, 4096), # assuming 144x144 input
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
        )

    def forward(self, data):

        x = data[0]
        x = self.features(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        return out





#######################################################################################
####################################### MTTSCAN  ######################################
#######################################################################################

class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config

class TSM(nn.Module):
    def __init__(self, n_segment=10, fold_div=3):
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

class MTTS_CAN_SMALL(nn.Module):
    """MTTS_CAN is the multi-task (respiration) version of TS-CAN"""


    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, frame_depth=20):
        super(MTTS_CAN_SMALL, self).__init__()

        print('init MTTS_CAN_SMALL')

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        # TSM layers
        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)
        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        # Apperance branch convs
        self.apperance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.apperance_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.apperance_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=(0, 0),bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=(0, 0),bias=True)
        self.attn_mask_2 = Attention_mask()

        # Dropout layers
        self.dropout_4_y = nn.Dropout(self.dropout_rate2)
        self.dropout_4_r = nn.Dropout(self.dropout_rate2)

        # Dense layers
        self.final_dense_1_y = nn.Linear(5184, self.nb_dense, bias=True)
        self.final_dense_2_y = nn.Linear(self.nb_dense, 1, bias=True)
        self.final_dense_1_r = nn.Linear(5184, self.nb_dense, bias=True)
        self.final_dense_2_r = nn.Linear(self.nb_dense, 1, bias=True)

    def forward(self, inputs, params=None):

        big = inputs[0]
        small = inputs[1]

        raw_input = torch.zeros_like(small)
        diff_input = small

        transform = T.Resize((9,9))
        for i in range(big.shape[0]):
            # iterate through batch
            raw_input[i,:,:,:] = transform(big[i,:,:,:])

        diff_input = self.TSM_1(diff_input)
        d1 = torch.tanh(self.motion_conv1(diff_input))
        d1 = self.TSM_2(d1)
        d2 = torch.tanh(self.motion_conv2(d1))

        r1 = torch.tanh(self.apperance_conv1(raw_input))
        r2 = torch.tanh(self.apperance_conv2(r1))

        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        # d3 = self.avg_pooling_1(gated1)
        # d4 = self.dropout_1(d3)

        # r3 = self.avg_pooling_2(r2)
        # r4 = self.dropout_2(r3)

        d4 = self.TSM_3(gated1)
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = self.TSM_4(d5)
        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.apperance_conv3(r2))
        r6 = torch.tanh(self.apperance_conv4(r5))

        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        # d7 = self.avg_pooling_3(gated2)
        # d8 = self.dropout_3(d7)
        d9 = gated2.view(gated2.size(0), -1)

        d10 = torch.tanh(self.final_dense_1_y(d9))
        d11 = self.dropout_4_y(d10)
        out_y = self.final_dense_2_y(d11)

        d10 = torch.tanh(self.final_dense_1_r(d9))
        d11 = self.dropout_4_r(d10)
        out_r = self.final_dense_2_r(d11)

        return out_y, out_r



#######################################################################################
####################################### DEEPPHYS  ######################################
#######################################################################################

class DeepPhys(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, out_size=1, img_size=36):
        """Definition of DeepPhys.
        Args:
          in_channels: the number of input channel. Default: 3
          img_size: height/width of each frame. Default: 36.
        Returns:
          DeepPhys model.
        """
        super(DeepPhys, self).__init__()

        print("INIT DEEPPHYS")

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        self.out_size = out_size
        
        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        # Apperance branch convs
        self.apperance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.apperance_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.apperance_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()
        # Dropout layers
        self.dropout_4 = nn.Dropout(self.dropout_rate2)
        # Dense layers
        self.final_dense_1 = nn.Linear(5184, self.nb_dense, bias=True)
        self.final_dense_2 = nn.Linear(self.nb_dense, self.out_size, bias=True)


    def forward(self, inputs, params=None):

        big = inputs[0]
        small = inputs[1]

        raw_input = torch.zeros_like(small)
        diff_input = small

        transform = T.Resize((9,9))
        for i in range(big.shape[0]):
            raw_input[i,:,:,:] = transform(big[i,:,:,:])

        d1 = torch.tanh(self.motion_conv1(diff_input))
        d2 = torch.tanh(self.motion_conv2(d1))

        r1 = torch.tanh(self.apperance_conv1(raw_input))
        r2 = torch.tanh(self.apperance_conv2(r1))

        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d5 = torch.tanh(self.motion_conv3(gated1))
        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.apperance_conv3(r2))
        r6 = torch.tanh(self.apperance_conv4(r5))

        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d9 = gated2.view(gated2.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        out = self.final_dense_2(d11)

        return out
