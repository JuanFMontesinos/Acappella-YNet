import numpy as np
from torch import nn


def normalize_max(waveform):
    if np.abs(waveform).max() != 0:
        waveform_out = waveform / np.abs(waveform).max()
    else:
        waveform_out = waveform

    return waveform_out


def unet_params(model: nn.Module):
    unet_modules = ['encoder', 'decoder', 'final_conv', 'pool', 'scaling', 'bias']
    for n, p in model.named_children():
        if n in unet_modules:
            yield from p.parameters()


class ParamFinder:
    def __init__(self):
        self.unet_params = ['encoder', 'decoder', 'final_conv', 'pool', 'scaling', 'bias']
        self.motion_net_params = ['motion_net']
        self.graph_net_params = ['graph_net']

    def unet(self, model: nn.Module):
        for n, p in model.named_children():
            if n in self.unet_params:
                yield from p.parameters()

    def others(self, model: nn.Module):
        for n, p in model.named_children():
            if (n not in self.unet_params) and (n not in self.motion_net_params) and (n not in self.graph_net_params):
                yield from p.parameters()
