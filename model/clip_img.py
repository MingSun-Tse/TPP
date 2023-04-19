import torch
import torch.nn as nn

#import os
#import sys
#from tkinter.tix import IMAGE
#from tqdm import tqdm
#
#p1 = os.path.abspath("")
#p2 = os.path.dirname(__file__)
#sys.path.append(os.path.join(p1, p2, '../CLIP/clip'))
#import clip


class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model.visual
        
    def forward(self, x):
        x = self.model(x)#.half()
        x = x / x.norm(dim = -1, keepdim = True)
        return x

class TextCLIP(nn.Module):
    def __init__(self, model) :
            super(TextCLIP, self).__init__()
            self.model = model

    def forward(self, x):
        return self.model.encode_text(x)

#def clip_vit():
#    net, _ = clip.load('ViT-B/16')
#    return ImageCLIP(model = net)
#
#def clip_rn():
#    net, _ = clip.load('RN101')
#    return ImageCLIP(model = net)


