import torch.nn as nn
import torch
from torch.nn import functional as F

class Conv2D_WN(nn.Conv2d):
    '''Conv2D with weight normalization.
    '''
    def __init__(self, 
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ):
        super(Conv2D_WN, self).__init__(in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, groups=groups, 
            bias=bias, padding_mode=padding_mode)
        
        # set up the scale variable in weight normalization
        self.wn_scale = nn.Parameter(torch.ones(out_channels), requires_grad=True)
        for i in range(self.weight.size(0)):
            self.wn_scale.data[i] = torch.norm(self.weight.data[i])

    def forward(self, input):
        w = F.normalize(self.weight, dim=(1,2,3))
        # print(w.shape)
        # print(torch.norm(w[0]))
        # print(self.wn_scale)
        w = w * self.wn_scale.view(-1,1,1,1)
        return F.conv2d(input, w, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)