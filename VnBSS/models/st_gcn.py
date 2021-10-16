import torch
import torch.nn as nn
import torch.nn.functional as F

from .gconv import ConvTemporalGraphical
from .graph import Graph


def init_eiw(x):
    B, T, J = x.shape
    x = x.unsqueeze(2).expand(B, T, J, J)
    x = torch.min(x, x.transpose(2, 3))
    x = x.unsqueeze(2).expand(B, T, 3, J, J)
    return x


class FiLM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bias = nn.Linear(in_channels, out_channels)
        self.scale = nn.Linear(in_channels, out_channels)

    def forward(self, x, c, *args):
        return x * self.scale(c).view(*args) + self.bias(c).view(*args)



class ST_GCN(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence, [X,Y,C]
            :math:`V_{in}` is the number of graph nodes, NUMBER OF JOINTS
            :math:`M_{in}` is the number of instance in a frame. NUMBER OF PEOPLE
    """

    def __init__(self,
                 in_channels,
                 num_class,
                 graph_cfg,
                 mode='mode A',
                 classifier=False,
                 input_type='x',
                 confidence_attention=False,
                 **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A,
                         dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)
        self.classifier = classifier
        self.input_type = input_type
        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 5
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        if kwargs.get('bn_momentum') is not None:
            del kwargs['bn_momentum']
        kwargs['edge_importance_weighting'] = kwargs.get('edge_importance_weighting')
        kwargs['A'] = A
        kwargs['num_class'] = num_class
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        if mode == 'mode A':
            self.st_gcn_networks = nn.ModuleList((
                st_gcn_block(in_channels, 64,
                             kernel_size, 1,
                             residual=False, **kwargs0),
                st_gcn_block(64, 64, kernel_size, 1, **kwargs),
                st_gcn_block(64, 128, kernel_size, 2, **kwargs),
                st_gcn_block(128, 128, kernel_size, 1, **kwargs),
                st_gcn_block(128, 256, kernel_size, 2, **kwargs),
                st_gcn_block(256, 256, kernel_size, 1, **kwargs),
            ))
        elif mode == 'mode B':
            self.st_gcn_networks = nn.ModuleList((
                st_gcn_block(in_channels, 32,
                             kernel_size, 1,
                             residual=False, **kwargs0),
                st_gcn_block(32, 32, kernel_size, 1, **kwargs),
                st_gcn_block(32, 64, kernel_size, 2, **kwargs),
                # st_gcn_block(64, 64, kernel_size, 1, **kwargs),
                st_gcn_block(64, 64, kernel_size, 1, **kwargs),
                st_gcn_block(64, 128, kernel_size, 2, **kwargs),
                # st_gcn_block(128, 128, kernel_size, 1, **kwargs),
                st_gcn_block(128, 128, kernel_size, 1, **kwargs),
                st_gcn_block(128, 256, kernel_size, 2, **kwargs),
                st_gcn_block(256, 256, kernel_size, 1, **kwargs),
            ))
        else:
            raise NotImplementedError

        # fcn for prediction
        self.num_class = num_class
        if self.classifier:
            self.fcn = nn.Conv2d(3 * N, num_class, kernel_size=1)
        if confidence_attention:
            self.attention = nn.Sequential(
                torch.nn.Conv1d(1, 3, 5, 2, bias=False),
                nn.ReLU(True),
                torch.nn.Conv1d(3, 9, 5, 2, padding=3, padding_mode='reflect', bias=False),
                nn.Sigmoid()
            )
        else:
            self.attention = None

    def forward(self, x, *args):
        args=list(args)
        if x.shape[1] == 3:
            args.append(x[:, 2, ...])
        x, c = self.extract_feature(x, *args)

        if self.classifier:
            return x, c
        else:
            return x

    def extract_feature(self, x, *args):
        if self.attention is not None:
            attention = torch.nn.functional.interpolate(
                x[:, 2, :, :, 0].transpose(1, 2), size=13, mode='linear').transpose(1, 2).unsqueeze(1)
        x = self.T(x, self.input_type)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad

        for gcn in self.st_gcn_networks:
            x, _ = gcn(x, self.A, *args)
        if self.attention is not None:
            x = x * attention
        if self.classifier:
            # global pooling

            c = F.avg_pool2d(x, x.size()[2:])
            c = c.view(N, M, -1, 1, 1).mean(dim=1)

            # prediction
            c = self.fcn(c)
            c = c.view(c.size(0), -1)
        else:
            c = None
        return x, c

    @staticmethod
    def T(x, inputtype):
        N, C, T, V, M = x.size()
        if inputtype == 'x':
            return x
        elif inputtype == 'x1p':
            y = torch.cat((torch.cuda.FloatTensor(N, C, 1, V, M).zero_(),
                           x[:, :, 1:] - x[:, :, :-1]), 2)
            y[:, 2, 1:, ...] = x[:, 2, 1:, ...]
        elif inputtype == 'x2p':

            y = torch.cat((torch.cuda.FloatTensor(N, C, 1, V, M).zero_(),
                           x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],
                           torch.cuda.FloatTensor(N, C, 1, V, M).zero_()), 2)
            y[:, 2, 1:-1, ...] = x[:, 2, 1:-1, ...]
        elif inputtype == 'x1':
            y = torch.cat([x[:, :2, 1:] - x[:, :2, :-1], torch.min(x[:, 2, 1:], x[:, 2, 1:]).unsqueeze(1)], dim=1)
        elif inputtype == 'x2':
            y = torch.cat([x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],
                           torch.min(torch.min(x[:, 2, 1:-1], x[:, 2, 2:]), x[:, 2, :-2]).unsqueeze(1)], dim=1)
        return y


class st_gcn_block(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True,
                 edge_importance_weighting='static',
                 A=None,
                 num_class=None,
                 activation='relu'):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.ctype = edge_importance_weighting
        self.activation = activation
        if edge_importance_weighting == 'static':
            self.edge_importance = 1.
            self.edge_importance_weighting = True
        elif edge_importance_weighting == 'dynamic':
            self.edge_importance_weighting = True
            self.edge_importance = nn.Parameter(torch.ones(A.shape))
        elif edge_importance_weighting == 'categorical':
            self.edge_importance_f = nn.Linear(num_class, A.nelement())

            self.edge_importance_weighting = False
        elif edge_importance_weighting == 'temporal':
            self.edge_importance_weighting = False
            self.edge_importance_f = FiLM(num_class + A.shape[-1], A.nelement())
        elif edge_importance_weighting == 'static_temporal':
            self.edge_importance_weighting = False
            self.edge_importance_f = FiLM(num_class, A.nelement())
        elif edge_importance_weighting == 'dynamic_temporal':
            self.edge_importance_weighting = False
            self.edge_importance_f = FiLM(num_class, A.nelement())
        else:
            raise ValueError('edge_importance_weighting (%s) not implemented')

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A, *args):
        if self.edge_importance_weighting:
            A = A * self.edge_importance
        else:

            if self.ctype == 'temporal':
                B, C, T, J = x.shape
                c = torch.nn.functional.interpolate(args[1].transpose(1, 2)[..., 0], size=T, mode='linear').transpose(1,
                                                                                                                      2)
                exp_onehot = args[0].unsqueeze(1).expand(B, T, args[0].shape[-1])
                edge_importance = self.edge_importance_f(init_eiw(c), torch.cat([c, exp_onehot], dim=-1), B, T,
                                                         A.shape[0], J, J)
            elif self.ctype == 'static_temporal':
                B, C, T, J = x.shape
                c = torch.nn.functional.interpolate(args[1].transpose(1, 2)[..., 0], size=T, mode='linear').transpose(1,
                                                                                                                      2)
                edge_importance = init_eiw(c)
            elif self.ctype == 'dynamic_temporal':
                B, C, T, J = x.shape
                c = torch.nn.functional.interpolate(args[1].transpose(1, 2)[..., 0], size=T, mode='linear').transpose(1,
                                                                                                                      2)
                exp_onehot = args[0].unsqueeze(1).expand(B, T, args[0].shape[-1])
                edge_importance = self.edge_importance_f(init_eiw(c), exp_onehot, B, T,
                                                         A.shape[0], J, J)
            elif self.ctype == 'categorical':

                edge_importance = self.edge_importance_f(args[0]).view(x.shape[0], *A.shape)
            A = A * F.relu(edge_importance)
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        if self.activation == 'relu':
            return self.relu(x), A
        else:
            return x, A
