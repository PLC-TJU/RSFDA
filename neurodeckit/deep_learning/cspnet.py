'''
#####################################################################################################################
Discription: 

This file constructs fundamental modules, encompassing the BiMap layer, the graph BiMap layer, the Riemannian Batch Normalization, 
the ReEig layer, and LogEig layer. 

These layers constitute the two geometric models (Tensor-CSPNet and Graph-CSPNet) found in model.py. 

There are two types of weight parameters for initialization as follows:

1. functional.StiefelParameter(th.empty(self._h, self._ni, self._no, dtype = dtype, device = device))

   In this instance, the parameter class is typically nn.Parameter. The backpropagations originate from the subsequent sources:

        BiMap/Graph BiMap <------  nn.Parameter; 

        BatchNormSPD      <------  nn.Parameter; 

2. geoopt.ManifoldParameter(th.empty(self._h, self._ni, self._no), manifold = ..)

     In this case, the parameter class is invoked from the "geoopt" package. Since the weights in BiMap and Riemannian Batch Normalization 
     reside on Stiefel and SPD manifolds, the backpropagations stem from the following sources:

        BiMap/Graph BiMap <------ geoopt.ManifoldParameter(,manifold=geoopt.CanonicalStiefel()); 

        BatchNormSPD      <------ geoopt.ManifoldParameter(,manifold=geoopt.SymmetricPositiveDefinite()) 

3. The objective of the SPDIncreaseDim() class is to augment the dimension of weights W, for instance, from (3, 3) to (5, 5), with the 
    expanding dimension being the identity matrix I_2.

#######################################################################################################################
'''

import torch as th
import torch.nn as nn
from torch.autograd import Function as F
from . import functional
import numpy as np
import geoopt
from .base import SkorchNet2

dtype =th.double
device=th.device('cpu')

class SPDIncreaseDim(nn.Module):

    def __init__(self, input_size, output_size):

        super(SPDIncreaseDim, self).__init__()

        self.register_buffer('eye', th.eye(output_size, input_size))

        add = np.asarray([0] * input_size + [1] * (output_size-input_size), dtype=np.float32)

        self.register_buffer('add', th.from_numpy(np.diag(add)))

    def forward(self, input):

        eye    = self.eye.unsqueeze(0).unsqueeze(0).to(input.dtype)

        eye    = eye.expand(input.size(0), input.size(1), -1, -1)

        add    = self.add.unsqueeze(0).unsqueeze(0).to(input.dtype)

        add    = add.expand(input.size(0), input.size(1), -1, -1)

        output = th.add(add, th.matmul(eye, th.matmul(input, eye.transpose(2,3))))

        return output


class BiMap(nn.Module):

    def __init__(self,h,ni,no):
        super(BiMap, self).__init__()

        self._h = h

        self.increase_dim = None

        if no <= ni:
            self._ni, self._no = ni, no
        else:
            self._ni, self._no = no, no
            self.increase_dim  = SPDIncreaseDim(ni,no)

        self._W =functional.StiefelParameter(th.empty(self._h, self._ni, self._no, dtype=dtype, device=device))
        #self._W = geoopt.ManifoldParameter(th.empty(self._h, self._ni, self._no), manifold=geoopt.CanonicalStiefel())

        self._init_bimap_parameter()

    def _init_bimap_parameter(self):

        for i in range(self._h):
            v  = th.empty(self._ni, self._ni, dtype = self._W.dtype, device = self._W.device).uniform_(0., 1.)
            vv = th.svd(v.matmul(v.t()))[0][:, :self._no]
            self._W.data[i] = vv

    def _bimap_multiplication(self, X):

        batch_size, channels_in, n_in, _ = X.shape

        P = th.zeros(batch_size, self._h, self._no, self._no, dtype = X.dtype, device = X.device)

        for c in range(self._h):
            P[:,c,:,:] = self._W[c, :, :].t().matmul(X[:,c,:,:]).matmul(self._W[c, :, :])

        return P

    def forward(self, X):

        if self.increase_dim:
            return self._bimap_multiplication(self.increase_dim(X))
        else:
            return self._bimap_multiplication(X)


class Graph_BiMap(nn.Module):

    def __init__(self,h,ni,no,P):
        super(Graph_BiMap, self).__init__()

        self._h = h
        self.increase_dim = None
        self._P = P

        if no <= ni:
            self._ni, self._no = ni, no
        else:
            self._ni, self._no = no, no
            self.increase_dim = SPDIncreaseDim(ni,no)

        self._W =functional.StiefelParameter(th.empty(self._h, self._ni, self._no, dtype=dtype,device=device))
        #self._W = geoopt.ManifoldParameter(th.empty(self._h, self._ni, self._no), manifold=geoopt.CanonicalStiefel())

        self._init_bimap_parameter()

    def _init_bimap_parameter(self):

        for i in range(self._h):
            v  = th.empty(self._ni, self._ni, dtype = self._W.dtype, device = self._W.device).uniform_(0.,1.)
            vv = th.svd(v.matmul(v.t()))[0][:, :self._no]
            self._W.data[i] = vv

    def _bimap_multiplication(self, X):

        batch_size, channels_in, n_in, _ = X.shape

        P = th.zeros(batch_size, self._h, self._no, self._no, dtype = X.dtype, device = X.device)

        for c in range(self._h):
            P[:,c,:,:] = self._W[c, :, :].t().matmul(X[:,c,:,:]).matmul(self._W[c, :, :])

        return P

    def forward(self, X):
        batch_size, channel_num, dim = X.shape[0], X.shape[1], X.shape[-1]

        if self.increase_dim:
            return self._bimap_multiplication(self.increase_dim(th.matmul(self._P, X.reshape((batch_size, channel_num, -1))).reshape((batch_size, channel_num, dim, dim))))
        else:
            return self._bimap_multiplication(th.matmul(self._P, X.reshape((batch_size, channel_num, -1))).reshape((batch_size, channel_num, dim, dim)))


class BatchNormSPD(nn.Module):

    def __init__(self, momentum, n):
        super(__class__, self).__init__()

        self.momentum     = momentum

        self.running_mean = geoopt.ManifoldParameter(th.eye(n, dtype=th.double),
                                               manifold=geoopt.SymmetricPositiveDefinite(),
                                               requires_grad=False
                                               )
        self.weight       = geoopt.ManifoldParameter(th.eye(n, dtype=th.double),
                                               manifold=geoopt.SymmetricPositiveDefinite(),
                                               )
        
    def forward(self,X):

        N, h, n, n  = X.shape

        X_batched   = X.permute(2, 3, 0, 1).contiguous().view(n, n, N*h, 1).permute(2, 3, 0, 1).contiguous()

        if(self.training):

            mean = functional.BaryGeom(X_batched)

            with th.no_grad():
                self.running_mean.data = functional.geodesic(self.running_mean, mean, self.momentum)

            X_centered = functional.CongrG(X_batched, mean, 'neg')

        else:
            X_centered = functional.CongrG(X_batched, self.running_mean, 'neg')

        X_normalized   = functional.CongrG(X_centered, self.weight, 'pos')

        return X_normalized.permute(2,3,0,1).contiguous().view(n,n,N,h).permute(2,3,0,1).contiguous()

class ReEig(nn.Module):
    def forward(self,P):
        return functional.ReEig.apply(P)

class LogEig(nn.Module):
    def forward(self,P):
        return functional.LogEig.apply(P)


'''
#####################################################################################################################
Description: 

This implementation pertains to Tensor-CSPNet and Graph-CSPNet. The hyperparameters within the model are task/scenario-specific 
and employed in the paper's experiments.

            Input Shape: 
                        (batch size, time windows, frequency bands, channel No., channel No.) ----> Tensor-CSPNet;
                        (batch size, segment No., channel No., channel No.)                   ---->  Graph-CSPNet.   

            self.mlp: multilayer perception (1 layer, if false / 3 layers, if true).

            self.n_segment: time windows * frequency bands ----> Tensor-CSPNet;
                                                 segment No. ---->  Graph-CSPNet.

            self.dims: This pertains to the shape dimension (in and out) within each BiMap layer.
            
                        For instance, [20, 30, 30, 20] indicates that the first BiMap layer has an input dimension of 20,
                        and an output dimension of 30, while the second BiMap layer has an input dimension of 30 and
                        an output dimension of 20.

            self.kernel_size: This value represents the total number of temporal segments.

            self.tcn_channels: This refers to the number of output channels h in CNNs. We recommend a relatively large 
            number as a smaller one may result in a loss of discriminative information. For example, if kernel_size = 1,
            the tcn_channel = 16.
            

#######################################################################################################################
'''

@SkorchNet2
class Tensor_CSPNet(nn.Module):

    def __init__(self, kernel_size, n_segment, n_channels, n_classes = 2, mlp = False, dataset = 'KU'):
        # super(Tensor_CSPNet, self).__init__()
        super().__init__()

        self._mlp         = mlp
        self.channel_in   = n_segment
        
        classes           = n_classes
        self.dims         = [n_channels, int(n_channels*0.75)*2, int(n_channels*0.75)*2, n_channels]
        # self.kernel_size  = 3            # class Formatdata  len(self.time_seg)
        self.kernel_size  = kernel_size
        self.tcn_channels = 48
 
        # if dataset == 'KU':
        #     classes           = 2
        #     self.dims         = [20, 30, 30, 20]
        #     self.kernel_size  = 3
        #     self.tcn_channels = 48
        # elif dataset == 'BCIC':
        #     classes           = 4
        #     self.dims         = [22, 36, 36, 22]
        #     self.kernel_size  = 2
        #     self.tcn_channels = 16

        self.BiMap_Block      = self._make_BiMap_block(len(self.dims)//2)
        self.LogEig           = LogEig()

        '''
        We use 4Hz bandwith on 4Hz ~ 40Hz for the frequency segmentation, and the largest, i.e., 9 (=36/4), 
        performs the best usually. Hence, we pick self.tcn_width = 9.  
        '''

        self.tcn_width        =  9 
        self.Temporal_Block   = nn.Conv2d(1, self.tcn_channels, (self.kernel_size, self.tcn_width*self.dims[-1]**2), stride=(1, self.dims[-1]**2), padding=0).double()
        
        if self._mlp:
            self.Classifier = nn.Sequential(
            nn.Linear(self.tcn_channels, self.tcn_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.tcn_channels, self.tcn_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.tcn_channels, classes)
            ).double()
        else:
            self.Classifier = nn.Linear(self.tcn_channels, classes).double()
    
    def _make_BiMap_block(self, layer_num):
        layers = []

        if layer_num > 1:
          for i in range(layer_num-1):
            dim_in, dim_out = self.dims[2*i], self.dims[2*i+1]
            layers.append(BiMap(self.channel_in, dim_in, dim_out))
            layers.append(ReEig())
        
        dim_in, dim_out = self.dims[-2], self.dims[-1]
        layers.append(BiMap(self.channel_in, dim_in, dim_out))
        layers.append(BatchNormSPD(momentum = 0.1, n = dim_out))
        layers.append(ReEig())
          
        return nn.Sequential(*layers).double()

    def forward(self, x):

        window_num, band_num = x.shape[1], x.shape[2]

        x     = x.reshape(x.shape[0], window_num*band_num, x.shape[3], x.shape[4])

        x_csp = self.BiMap_Block(x)

        x_log = self.LogEig(x_csp)

        # NCHW Format: (batch_size, window_num*band_num, 4, 4) ---> (batch_size, 1, window_num, band_num * 4 * 4)
        x_vec = x_log.view(x_log.shape[0], 1, window_num, -1)

        y     = self.Classifier(self.Temporal_Block(x_vec).reshape(x.shape[0], -1))

        return y

@SkorchNet2
class Graph_CSPNet(nn.Module):

    def __init__(self, P, n_segment, n_channels, n_classes = 2, mlp = False, dataset = 'KU'):
        # super(Graph_CSPNet, self).__init__()
        super().__init__()
        
        self._mlp       = mlp
        self.channel_in = n_segment
        self.P          = P
        
        classes           = n_classes
        self.dims         = [n_channels, int(n_channels*0.75)*2, int(n_channels*0.75)*2, n_channels]

        # if dataset   == 'KU':
        #     classes     = 2
        #     self.dims   = [20, 30, 30, 20]
        # elif dataset == 'BCIC':
        #     classes     = 4
        #     self.dims   = [22, 36, 36, 22]

        self.Graph_BiMap_Block = self._make_Graph_BiMap_block(len(self.dims)//2)
        self.LogEig            = LogEig()
        
        if self._mlp:
            self.Classifier = nn.Sequential(
            nn.Linear(self.channel_in*self.dims[-1]**2, self.channel_in),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel_in, self.channel_in),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel_in, classes)
            ).double()
        else:
            self.Classifier = nn.Linear(self.channel_in*self.dims[-1]**2, classes).double()

    def _make_Graph_BiMap_block(self, layer_num):

        layers = []
        _I     = th.eye(self.P.shape[0], dtype=self.P.dtype, device=self.P.device)  # 确保_I的device和dtype与P的一致性 LC.Pan 2024.3.14修改
        
        if layer_num > 1:
          dim_in, dim_out = self.dims[0], self.dims[1]
          layers.append(Graph_BiMap(self.channel_in, dim_in, dim_out, self.P))
          layers.append(ReEig())

          for i in range(1, layer_num-1):
            dim_in, dim_out = self.dims[2*i], self.dims[2*i+1]
            layers.append(Graph_BiMap(self.channel_in, dim_in, dim_out, _I))
            layers.append(ReEig())

          dim_in, dim_out = self.dims[-2], self.dims[-1]
          layers.append(Graph_BiMap(self.channel_in, dim_in, dim_out, _I))
          layers.append(BatchNormSPD(momentum = 0.1, n = dim_out))
          layers.append(ReEig())

        else:
          dim_in, dim_out = self.dims[-2], self.dims[-1]
          layers.append(Graph_BiMap(self.channel_in, dim_in, dim_out, self.P))
          layers.append(BatchNormSPD(momentum = 0.1, n = dim_out))
          layers.append(ReEig())

        return nn.Sequential(*layers).double()


    def forward(self, x):

        x_csp = self.Graph_BiMap_Block(x)

        x_log = self.LogEig(x_csp)

        # NCHW Format (batch_size, window_num*band_num, 4, 4) ---> (batch_size, 1, window_num, band_num * 4 * 4)
        x_vec = x_log.view(x_log.shape[0], -1)

        y     = self.Classifier(x_vec)

        return y
