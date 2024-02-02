import numpy as np
import torch
from torch import nn
from torch.nn.functional import elu, instance_norm
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class PEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,):
        super(PEConv, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        x = self.conv2d(input)
        return x
    
class Cylin_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, 
                bias=True, add_pe=False, pe_down=1, pad_w=512, pad_h=256, pad_ksize=5, pad_stride=1, pad_dilation=1):
        super(Cylin_Block, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.sigmoid = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.add_pe = add_pe
        self.pe_down = pe_down
        self.pad_w = pad_w
        self.pad_h = pad_h
        self.pad_ksize = pad_ksize
        self.pad_stride = pad_stride
        self.pad_dilation = pad_dilation
        self.out_channels = out_channels

    def gated(self, mask):
        return self.sigmoid(mask)
    
    def get_pad(self, in_, ksize, stride, dilation=1):
        out_ = np.ceil(float(in_) / stride)
        return int(((out_ - 1) * stride + dilation * (ksize - 1) + 1 - in_) / 2)

    def forward(self, img, pe):
        
        ##### (before conv) circular padding along the width and then zero padding along the height #####
        
        pad1 = self.get_pad(self.pad_w, self.pad_ksize, self.pad_stride, self.pad_dilation)
        pad2 = self.get_pad(self.pad_h, self.pad_ksize, self.pad_stride, self.pad_dilation)
        input = F.pad(img, (pad1, pad1, 0, 0), mode = 'circular')
        input = F.pad(input, (0, 0, pad2, pad2))
        
        ##### (in conv) gated convolution for completion #####
        
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        x = elu(x) * self.gated(mask)
        x = self.bn(x) #x = instance_norm(x)
        
        ##### (after conv) compensate the learnable positional encoding for features #####
        
        if(self.add_pe):
            pe = F.interpolate(pe, size=(256 // self.pe_down, 512 // self.pe_down))
            pe_rep = pe.repeat(1, self.out_channels, 1, 1)
            x = torch.add(x, pe_rep)
        
        return x
    
class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)

    def forward(self, input):
        input = F.pad(input, (1, 1, 0, 0), mode = 'circular')
        input = F.pad(input, (0, 0, 1, 1))
        x = self.conv2d(input)
        return x