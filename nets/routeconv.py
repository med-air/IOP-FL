import functools
import numpy

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


class _routing(nn.Module):

    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(in_channels, 2*in_channels)
        self.fc2 = nn.Linear(2*in_channels, num_experts)

    def forward(self, x):
        x = torch.flatten(x)
        
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.sigmoid(x)
    

class RouteConv2D(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, name, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', client_num=4, dropout_rate=0.2):
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.kernel_size = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.dropout_rate = dropout_rate
        super(RouteConv2D, self).__init__(
            in_channels, out_channels, self.kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        
        self.use_bias = bias
        self.client_num = client_num
        self.num_experts = self.client_num + 1
        self.name=name
        self.reset_parameters()
    

    def _mix_trajectories(self, state_dicts):
        '''
        paths : [local_models_path1, ..., local_models_pathn, global model]  len: n+1
        '''
        self.reset_parameters()
        self.init_data = []
        self.init_data_bias = []
        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._routing_fn = _routing(self.in_channels, self.num_experts, self.dropout_rate)
        self.weight = Parameter(torch.Tensor(
            self.num_experts, self.out_channels, self.in_channels // self.groups, *self.kernel_size))
    
        for idx in range(self.client_num):
            local_data_weight = state_dicts[idx][self.name+'.weight']
            self.init_data.append(local_data_weight)
            if self.use_bias:
                local_data_bias = state_dicts[idx][self.name+'.bias']
                self.init_data_bias.append(local_data_bias)

        self.init_data.append(state_dicts[-1][self.name+'.weight'])
        self.init_data = torch.stack(self.init_data, dim=0)
        self.weight.data = self.init_data
        if self.use_bias:
            self.init_data_bias.append(state_dicts[-1][self.name+'.bias']) 
            self.init_data_bias = torch.stack(self.init_data_bias, dim=0)
            self.bias.data = self.init_data_bias

    def _conv_forward(self, input, weight, bias=None):
        if bias is None:
            bias = self.bias
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    def forward(self, inputs):
        b, _, _, _ = inputs.size()
        res = []
        for inpt in inputs:
            inpt = inpt.unsqueeze(0)
            pooled_inputs = self._avg_pooling(inpt)
            routing_weights = self._routing_fn(pooled_inputs)
            kernels = torch.sum(routing_weights[: ,None, None, None, None] * self.weight, 0)
            biases = torch.sum(routing_weights[:, None] * self.bias, 0) if self.use_bias else None
            out = self._conv_forward(inpt, kernels, biases)
            res.append(out)
        return torch.cat(res, dim=0)

class RouteLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, name: str, bias: bool = True, client_num=4,dropout_rate=0.2) -> None:
        super(RouteLinear,self).__init__(in_features, out_features, bias)
        self.name = name
        self.use_bias = bias
        self.client_num = client_num
        self.num_experts = self.client_num + 1
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate

        self.reset_parameters()

    def _mix_trajectories(self, state_dicts):
        '''
        paths : [local_models_path1, ..., local_models_pathn, global model]  len: n+1
        '''
        self.init_data = []
        self.init_data_bias = []
        self._avg_pooling = functools.partial(F.adaptive_avg_pool1d, output_size=1)
        self._routing_fn = _routing(self.in_features, self.num_experts, self.dropout_rate)
        
        self.weight = Parameter(torch.Tensor(
            self.num_experts, self.out_features, self.in_features))
        
        for idx in range(self.client_num):
            local_data_weight = state_dicts[idx][self.name+'.weight']
            self.init_data.append(local_data_weight)
            if self.use_bias:
                local_data_bias = state_dicts[idx][self.name+'.bias']
                self.init_data_bias.append(local_data_bias)

        self.init_data.append(state_dicts[-1][self.name+'.weight'])
        self.init_data = torch.stack(self.init_data, dim=0)
        self.weight.data = self.init_data
        if self.use_bias:
            self.init_data_bias.append(state_dicts[-1][self.name+'.bias']) 
            self.init_data_bias = torch.stack(self.init_data_bias, dim=0)
            self.bias.data = self.init_data_bias

    def forward(self, inputs):
        b, _ = inputs.size()
        res = []
        for inpt in inputs:
            inpt = inpt.unsqueeze(0)
            routing_weights = self._routing_fn(inpt)
            kernels = torch.sum(routing_weights[:, None, None] * self.weight, 0)
            biases = torch.sum(routing_weights[:, None] * self.bias, 0) if self.use_bias else None
            out = F.linear(inpt,kernels, biases)
            res.append(out)
        return torch.cat(res, dim=0)



class RouteConvTranspose2D(_ConvTransposeNd):
    

    def __init__(self, in_channels, out_channels, kernel_size, name, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', client_num=5, dropout_rate=0.2):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        
        super(RouteConvTranspose2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, _pair(0), groups, bias, padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = kernel_size
        self.client_num = client_num
        self.num_experts = self.client_num + 1
        self.dropout_rate = dropout_rate
        self.name = name
        self.use_bias = bias
        self.reset_parameters()
        
        
    
    def _mix_trajectories(self, state_dicts):
        '''
        paths : [local_models_path1, ..., local_models_pathn, global model]  len: n+1
        '''
        self.reset_parameters()
        self.init_data = []
        self.init_data_bias = []
        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._routing_fn = _routing(self.in_channels, self.num_experts, self.dropout_rate)
        self.weight = Parameter(torch.Tensor(
            self.num_experts, self.in_channels, self.out_channels // self.groups, *self.kernel_size))
        self.bias = Parameter(torch.Tensor(self.num_experts, self.out_channels))
      
        
        for idx in range(self.client_num):
            local_data_weight = state_dicts[idx][self.name+'.weight']
            self.init_data.append(local_data_weight)
            if self.use_bias:
                local_data_bias = state_dicts[idx][self.name+'.bias']
                self.init_data_bias.append(local_data_bias)

        self.init_data.append(state_dicts[-1][self.name+'.weight'])
        self.init_data = torch.stack(self.init_data, dim=0)
        self.weight.data = self.init_data
        if self.use_bias:
            self.init_data_bias.append(state_dicts[-1][self.name+'.bias']) 
            self.init_data_bias = torch.stack(self.init_data_bias, dim=0)
            self.bias.data = self.init_data_bias
       

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return None
        return F.conv_transpose2d(input, weight, bias, self.stride,
                        self.padding, 0, self.groups, self.dilation)
    
    def forward(self, inputs):
        b, _, _, _ = inputs.size()
        res = []
        for input in inputs:
            input = input.unsqueeze(0)
            pooled_inputs = self._avg_pooling(input)
            routing_weights = self._routing_fn(pooled_inputs)
            kernels = torch.sum(routing_weights[: ,None, None, None, None] * self.weight, 0)
            biases = torch.sum(routing_weights[:, None] * self.bias, 0)
            out = self._conv_forward(input, kernels, biases)
            res.append(out)
        return torch.cat(res, dim=0)
