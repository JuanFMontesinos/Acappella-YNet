from torchvision.models.video.resnet import VideoResNet as VideoResNet_or, model_urls, BasicBlock, BasicStem, \
    Conv3DSimple, Conv3DNoTemporal

try:
    from torchvision.models.utils import load_state_dict_from_url
except:
    from torchvision._internally_replaced_utils import load_state_dict_from_url  # pytorch>=1.10

from torch import nn


class VideoResNet(VideoResNet_or):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # 256 feats
        # x = self.layer4(x) # 512 feats
        x = self.avgpool(x)
        return x


def _video_resnet(arch, pretrained=False, progress=True, **kwargs):
    model = VideoResNet(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def mc3_18(pretrained=False, progress=True, **kwargs):
    """Constructor for 18 layer Mixed Convolution network as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: MC3 Network definition
    """
    return _video_resnet('mc3_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DSimple] + [Conv3DNoTemporal] * 3,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)
