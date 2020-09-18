
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class luna_block(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, conv_channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv3d(conv_channels, conv_channels, kernel_size=3, padding=1, bias=True)
        self.maxpool = nn.MaxPool3d(2,2)
    
    def forward(self, input_batch):
        block_out = F.relu(self.conv1(input_batch))
        block_out = F.relu(self.conv2(block_out))
        
        return self.maxpool(block_out)

class luna_model(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batchnorm = nn.BatchNorm3d(1)

        self.block1 = luna_block(in_channels, conv_channels)
        self.block2 = luna_block(conv_channels, conv_channels * 2)
        self.block3 = luna_block(conv_channels * 3, conv_channels * 4)
        self.block4 = luna_block(conv_channels * 4, conv_channels * 8)

        self.head_linear = nn.Linear(1152, 2)
        self.head_softmax = nn.Softmax(dim=1)
    
    def forward(self, input_batch):
        bn_output = self.tail_batchnorm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(bn_output)
        block_out = self.block3(bn_output)
        block_out = self.block4(bn_output)
        
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.head_linear(conv_flat)

        return linear_output, self.head_softmax(linear_output)
    
    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv3d}:
                nn.init.kaiming_normal_(m.weights, a=0, mode="fan_out", nonlinearity="relu")
            
            if m.bias is not None:
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                bound = 1 / math.sqrt(fan_out)
                nn.init.normal_(m.bias, -bound, bound)
