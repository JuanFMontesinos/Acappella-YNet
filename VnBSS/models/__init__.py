import inspect
import os
from copy import copy
import torch
from ..utils import ParamFinder


def torch_version():
    string = torch.__version__
    segments = string.split('.')
    return int(segments[0]), int(segments[1])


__all__ = ['ModelConstructor', 'y_net_mr', 'y_net_gr', 'y_net_g', 'y_net_m']

llcp = {
    # Audiovisual flags
    "video_enabled": False,
    "llcp_enabled": True,
    "skeleton_enabled": False,
    "remix_input": False,
    "remix_coef": 0.5,
}
llcp_r = {
    # Audiovisual flags
    "video_enabled": False,
    "llcp_enabled": True,
    "skeleton_enabled": False,
    "remix_input": True,
    "remix_coef": 0.5,
}

y_net_m = {
    # Audiovisual flags
    "video_enabled": True,
    "llcp_enabled": False,
    "skeleton_enabled": False,
    # U-Net args
    "layer_distribution": [32, 64, 128, 256, 512],
    "activation": None,
    "remix_input": False,
    "remix_coef": 0.5,
    "video_temporal_features": 16,
    "architecture": "sop",
    "mode": "upsample",
}
u_net = {
    # Audiovisual flags
    "video_enabled": False,
    "llcp_enabled": False,
    "skeleton_enabled": False,
    # U-Net args
    "layer_distribution": [32, 64, 128, 256, 512],
    "activation": None,
    "remix_input": False,
    "remix_coef": 0.5,
    "video_temporal_features": 16,
    "architecture": "sop",
    "mode": "upsample",
}
y_net_r = {
    # Audiovisual flags
    "video_enabled": True,
    "llcp_enabled": False,
    "skeleton_enabled": False,
    "architecture": "sop",
    "mode": "upsample",
    # U-Net args
    "layer_distribution": [32, 64, 128, 256, 512],
    "activation": None,
    "remix_input": True,
    "remix_coef": 0.5,
    "video_temporal_features": 16
}
y_net_g = {
    # Audiovisual flags
    "video_enabled": False,
    "llcp_enabled": False,
    "skeleton_enabled": True,
    "architecture": "sop",
    "mode": "upsample",
    # U-Net args
    "layer_distribution": [32, 64, 128, 256, 512],
    "activation": None,
    "remix_input": False,
    "remix_coef": 0.5,
    "video_temporal_features": 16
}
y_net_gr = {
    # Audiovisual flags
    "video_enabled": False,
    "llcp_enabled": False,
    "skeleton_enabled": True,
    "architecture": "sop",
    "mode": "upsample",
    # U-Net args
    "layer_distribution": [32, 64, 128, 256, 512],
    "activation": None,
    "remix_input": True,
    "remix_coef": 0.5,
    "video_temporal_features": 16
}


class ModelConstructor:
    def __init__(self, *,
                 debug: dict,
                 loss_criterion: str,  # L1, MSE, BCE
                 # Fourier Transform flags
                 log_sp_enabled: bool, mel_enabled: bool, complex_enabled: bool,
                 weighted_loss: bool, loss_on_mask: bool, binary_mask: bool,
                 downsample_coarse: bool, downsample_interp: bool,
                 audio_length: int, audio_samplerate: int, hop_length: int,
                 n_mel: int, n_fft: int, sp_freq_shape: int, mean, std,
                 n=1):
        args = {}
        for arg in inspect.getfullargspec(ModelConstructor.__init__).kwonlyargs:
            if arg != 'self':
                args.update({arg: locals()[arg]})

        self.common_kwargs = args
        self.model = None

    def prepare(self, model):
        self.model = model
        return self

    def update(self, **kwargs):
        self.common_kwargs.update(kwargs)
        return self

    def build(self):

        return self._build_dev()

    def _build_dev(self):
        constructor = getattr(self, f'_{self.model}')()
        iter_func = getattr(self, f'_iter_{self.model}')

        model = constructor(**self.common_kwargs)

        iter_params = iter_func(model)
        return iter_params, model, self.common_kwargs

    def _llcp(self):
        from .llcp_net import LlcpNet
        return LlcpNet

    def _llcp_r(self):
        from .llcp_net import LlcpNet
        return LlcpNet

    def _y_net_m(self):
        from .y_net import YNet
        return YNet

    def _u_net(self):
        from .y_net import YNet
        return YNet

    def _y_net_r(self):
        from .y_net import YNet
        return YNet

    def _y_net_g(self):
        from .y_net import YNet
        return YNet

    def _y_net_gr(self):
        from .y_net import YNet
        return YNet

    def _iter_llcp(self, model):
        iterable = [{"params": model.llcp.parameters()}]
        return iterable

    def _iter_llcp_r(self, model):
        iterable = [{"params": model.llcp.parameters()}]
        return iterable

    def _iter_y_net_e_legacy(self, model):
        iterable = [{"params": ParamFinder().unet(model)},
                    {"params": ParamFinder().others(model)},
                    {"params": model.motion_net.parameters()}]
        return iterable

    def _iter_y_net_m_legacy(self, model):
        iterable = [{"params": ParamFinder().unet(model)},
                    {"params": ParamFinder().others(model)},
                    {"params": model.motion_net.parameters()}]
        return iterable

    def _iter_y_net_r_legacy(self, model):
        iterable = [{"params": ParamFinder().unet(model)},
                    {"params": ParamFinder().others(model)},
                    {"params": model.motion_net.parameters()}]
        return iterable

    def _iter_y_net_g(self, model):
        iterable = [{"params": ParamFinder().unet(model)},
                    {"params": ParamFinder().others(model)},
                    {"params": model.graph_net.parameters()}]
        if self.common_kwargs['video_enabled']:
            iterable.append({"params": model.motion_net.parameters()})
        return iterable

    def _iter_y_net_gr(self, model):
        iterable = [{"params": ParamFinder().unet(model)},
                    {"params": ParamFinder().others(model)},
                    {"params": model.graph_net.parameters()}]
        if self.common_kwargs['video_enabled']:
            iterable.append({"params": model.motion_net.parameters()})
        return iterable

    def _iter_y_net_m(self, model):
        iterable = [{"params": ParamFinder().unet(model)},
                    {"params": ParamFinder().others(model)}]

        if self.common_kwargs['video_enabled']:
            iterable.append({"params": model.motion_net.parameters()})
        return iterable

    def _iter_y_net_r(self, model):
        iterable = [{"params": ParamFinder().unet(model)},
                    {"params": ParamFinder().others(model)}]

        if self.common_kwargs['video_enabled']:
            iterable.append({"params": model.motion_net.parameters()})
        return iterable

    def _iter_u_net(self, model):
        iterable = [{"params": ParamFinder().unet(model)},
                    {"params": ParamFinder().others(model)}]
        return iterable


"""
CODE ADDED TO EASE MODEL USE
USE THESE MODELS FOR INFERENCE ONLY
"""


def copyupdt(original: dict, *args):
    assert isinstance(original, dict)
    new_dic = copy(original)
    for arg in args:
        assert isinstance(arg, dict)
        new_dic.update(arg)
    return new_dic


_DEFAULT_CFG = {
    "loss_criterion": "MSE",
    "log_sp_enabled": False,
    "mel_enabled": False,
    "complex_enabled": True,
    "weighted_loss": True,
    "loss_on_mask": True,
    "binary_mask": False,
    "downsample_coarse": True,
    "downsample_interp": False,
    "mean": [0.43216, 0.394666, 0.37645],
    "std": [0.22803, 0.22145, 0.216989],
    "audio_length": (4 * 2 ** 14 - 1),
    "audio_samplerate": 16384,
    "n_fft": 1022,
    "n_mel": 80,
    "sp_freq_shape": 1022 // 2 + 1,
    "hop_length": 256
}
_BASE_Y_NET_CFG = {
    "dropout": False,
    "skeleton_pooling": "AdaptativeAP",
    "multitask_pooling": "AdaptativeMP",
    "graph_kwargs": {
        "graph_cfg": {
            "layout": "acappella",
            "strategy": "spatial",
            "max_hop": 1,
            "dilation": 1
        },
        "edge_importance_weighting": "dynamic",
        "mode": "mode B",
        "dropout": False
    },
    "single_frame_enabled": False,
    "single_emb_enabled": False,
    "video_enabled": False,
    "llcp_enabled": False,
    "skeleton_enabled": False,
    "video_temporal_features": 16, "layer_kernels": "sssstt",
    "layer_distribution": [32, 64, 128, 256, 256, 256, 512],
    "activation": None,
    "remix_input": False,
    "remix_coef": 0.5,
    "architecture": "sop",
    "mode": "upsample",
    "multitask": {
        "enabled": False,
        "loss": "ContrastiveLoss",
        "independent_encoder": True
    },
    'white_metrics': False
}
_4BLOCK_Y_NET = copyupdt(_BASE_Y_NET_CFG,
                         {"layer_kernels": "ssstt",
                          "layer_distribution": [32, 64, 128, 256, 256, 256, 512]}
                         )
WEIGHTS_PATH = './model_weights'
DEBUG = {'isnan': True, 'ds_autogen': False, "overfit": False, 'verbose': False}


#################################
def download_google(fileID, dst):
    from google_drive_downloader import GoogleDriveDownloader as gdd
    gdd.download_file_from_google_drive(file_id=fileID, dest_path=dst)


def y_net_mr(debug_dict=DEBUG, pretrained=True, n=1):
    constructor = ModelConstructor(debug=debug_dict, n=n, **_DEFAULT_CFG)
    kwargs = copy(_BASE_Y_NET_CFG)
    kwargs['video_enabled'] = True
    iter_param, model, model_kwargs = constructor.prepare('y_net_m').update(**kwargs).build()
    if pretrained:
        from torch import load
        if not os.path.exists(WEIGHTS_PATH):
            os.mkdir(WEIGHTS_PATH)
        path = os.path.join(WEIGHTS_PATH, 'y_net_mr7.pth')
        download_google('1dEBboJPEJSMrIZSxMTSBlQzWPzsdmUdm', path)
        state_dict = load(path, map_location=lambda storage, loc: storage)
        if torch_version()[0] >= 1 and torch_version()[1] >= 10:
            state_dict['sp2mel.fb'] = torch.rand(201, 80)
        model.load_state_dict(state_dict, strict=True)
    return model


def y_net_m(debug_dict=DEBUG, pretrained=True, n=1):
    constructor = ModelConstructor(debug=debug_dict, n=n, **_DEFAULT_CFG)
    kwargs = copy(_BASE_Y_NET_CFG)
    kwargs['video_enabled'] = True
    iter_param, model, model_kwargs = constructor.prepare('y_net_m').update(**kwargs).build()
    if pretrained:
        from torch import load
        if not os.path.exists(WEIGHTS_PATH):
            os.mkdir(WEIGHTS_PATH)
        path = os.path.join(WEIGHTS_PATH, 'y_net_m7.pth')
        download_google('1HP1O9JiDJx5LuaJA3exYVvnd6oOcITHN', path)
        state_dict = load(path, map_location=lambda storage, loc: storage)
        if torch_version()[0] >= 1 and torch_version()[1] >= 10:
            state_dict['sp2mel.fb'] = torch.rand(201, 80)
        model.load_state_dict(state_dict, strict=True)
    return model


def y_net_gr(debug_dict=DEBUG, pretrained=True, n=1):
    constructor = ModelConstructor(debug=debug_dict, n=n, **_DEFAULT_CFG)
    kwargs = copy(_BASE_Y_NET_CFG)
    kwargs['skeleton_enabled'] = True
    iter_param, model, model_kwargs = constructor.prepare('y_net_g').update(**kwargs, n=n).build()
    if pretrained:
        from torch import load
        if not os.path.exists(WEIGHTS_PATH):
            os.mkdir(WEIGHTS_PATH)
        path = os.path.join(WEIGHTS_PATH, 'y_net_gr7.pth')
        download_google('1GN1bGyp1TlZkkgSQMwI_CGOskWXNLjxd', path)
        state_dict = load(path, map_location=lambda storage, loc: storage)
        if torch_version()[0] >= 1 and torch_version()[1] >= 10:
            state_dict['sp2mel.fb'] = torch.rand(201, 80)
        model.load_state_dict(state_dict, strict=True)
    return model


def y_net_g(debug_dict=DEBUG, pretrained=True, n=1):
    constructor = ModelConstructor(debug=debug_dict, n=n, **_DEFAULT_CFG)
    kwargs = copy(_BASE_Y_NET_CFG)
    kwargs['skeleton_enabled'] = True
    iter_param, model, model_kwargs = constructor.prepare('y_net_g').update(**kwargs, n=n).build()
    if pretrained:
        from torch import load
        if not os.path.exists(WEIGHTS_PATH):
            os.mkdir(WEIGHTS_PATH)
        path = os.path.join(WEIGHTS_PATH, 'y_net_g7.pth')
        download_google('1WJ8vEh6TWeKPE2nQKBXLXygrVwk7LHVm', path)
        state_dict = load(path, map_location=lambda storage, loc: storage)
        if torch_version()[0] >= 1 and torch_version()[1] >= 10:
            state_dict['sp2mel.fb'] = torch.rand(201, 80)
        model.load_state_dict(state_dict, strict=True)
    return model
