from typing import List
import torch

from numpy import float32
from torch import nn, istft
from torch.nn.functional import binary_cross_entropy_with_logits, l1_loss, mse_loss
from torchaudio.transforms import Spectrogram, MelScale, InverseMelScale

from .llcp import Video_Model
from .resnet import mc3_18
from .st_gcn import ST_GCN
from .unet import UNet

__all__ = ['YNet']
K = 10


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


class FiLM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FiLM, self).__init__()
        self.scaling = nn.Linear(in_channels, out_channels)
        self.bias = nn.Linear(in_channels, out_channels)

    def forward(self, visual_features, audio_features):
        vf_scale = self.scaling(visual_features).transpose(1, 2).unsqueeze(2)  # Shape: N,256,1,4
        vf_bias = self.bias(visual_features).transpose(1, 2).unsqueeze(2)  # Shape: N,256,1,4
        return audio_features * vf_scale + vf_bias


class YNet(UNet):
    def __init__(self, *,
                 debug: dict,
                 loss_criterion: str,  # L1, MSE, BCE
                 remix_input: bool, remix_coef: float,
                 # Fourier transform params
                 audio_length: int, audio_samplerate: int, mean, std,
                 n_fft: int, hop_length: int, n_mel: int, sp_freq_shape: int,
                 # Fourier Transform flags
                 log_sp_enabled: bool, mel_enabled: bool, complex_enabled: bool,
                 weighted_loss: bool, loss_on_mask: bool, binary_mask: bool,
                 downsample_coarse: bool, downsample_interp: bool,
                 # Audiovisual flags
                 video_enabled: bool,
                 video_temporal_features: int,
                 llcp_enabled: bool,
                 skeleton_enabled: bool,
                 # Appearence flags
                 single_frame_enabled: bool,
                 single_emb_enabled: bool,
                 # U-Net args
                 layer_distribution: List[int],
                 activation,
                 # Graph args
                 skeleton_pooling: str,
                 graph_kwargs: dict,

                 # Multitask args
                 multitask: dict,  # {'enabled': False, 'loss': 'ContrastiveLoss', 'independent_encoder': False},
                 multitask_pooling: str,
                 white_metrics: bool,
                 **kwargs):
        # This defines the U-Net (parent class)
        super(YNet, self).__init__(layer_distribution,
                                   2 if complex_enabled else 1,
                                   film=None,
                                   useBN=True,
                                   activation=activation,
                                   input_channels=2 if complex_enabled else 1,
                                   **kwargs)
        # Flags
        self.transformer_enabled = False  # Deleted from code for simplicity

        self.video_enabled = video_enabled
        self.video_temporal_features = video_temporal_features
        self.llcp_enabled = llcp_enabled
        self.sk_enabled = skeleton_enabled
        self.single_frame_enabled = single_frame_enabled
        self.single_emb_enabled = single_emb_enabled

        self.log_sp_enabled = log_sp_enabled
        self.mel_enabled = mel_enabled
        self.complex_enabled = complex_enabled
        self.multitask = multitask['enabled']
        self.loss_on_mask = loss_on_mask
        self.binary_mask = binary_mask
        self.downsample_coarse = downsample_coarse
        self.downsample_interp = downsample_interp
        self.weighted_loss = weighted_loss
        self.loss_criterion = loss_criterion
        self.debug = debug
        self.feat_num = 0
        self.remix_input = remix_input
        self.remix_coef = remix_coef
        self.white_metrics = white_metrics

        self._audio_samplerate = audio_samplerate
        self._n = kwargs['n']
        self._audio_length = audio_length * self._n
        self._n_fft = n_fft
        self._n_mel = n_mel
        self._sp_freq_shape = sp_freq_shape
        self._hop_length = hop_length
        self._mean = mean
        self._std = std
        self._assert_flags()

        self._define_video_network()
        self._define_graph_network(skeleton_pooling, graph_kwargs)
        self._define_fourier_operators()
        self._define_multitask_layers(multitask, layer_distribution, activation, multitask_pooling, **kwargs)
        self._define_appearence_network()
        self.enabled = {'video_enabled': video_enabled,
                        'llcp_enabled': llcp_enabled,
                        'single_frame_enabled': single_frame_enabled,
                        'single_emb_enabled': single_emb_enabled,
                        'skeleton_enabled': skeleton_enabled}
        if video_enabled or skeleton_enabled or llcp_enabled:
            # FiLM to fuse video and audio features
            N = self.dim[-2]
            self.fusion_layer = FiLM(self.feat_num, N)  # 512 or 512+256

    def _assert_flags(self):
        assert not (self.single_emb_enabled and self.single_frame_enabled), \
            f'Appearance frame  cannot be applied together with embedding apperance'
        assert not (self.complex_enabled and self.log_sp_enabled), \
            f'log scale cannot be applied to complex spectrograms'
        assert not (self.complex_enabled and self.binary_mask), \
            f'binary mask and  complex mask are mutually exclusive'
        assert not (self.complex_enabled and self.mel_enabled), \
            f'mel transform cannot be applied to complex spectrograms'

        assert not (self.mel_enabled and self.log_sp_enabled), \
            f'mel transform cannot be applied together with log scale'

        assert not (self.downsample_coarse and self.downsample_interp), \
            f'Downsample cannot be coarse and interpolated at the same time'
        assert not (self.remix_input and self.binary_mask), \
            f'Binary mask is not implemented for more than 2 src'
        assert self.loss_criterion in ['MSE', 'L1', 'BCE']

    def _define_fourier_operators(self):
        # FOURIER TRANSFORMS (for waveform to spectrogram)
        try:
            self.wav2sp_train = Spectrogram(n_fft=self._n_fft, power=None, hop_length=self._hop_length,
                                            return_complex=False)
        except TypeError:
            self.wav2sp_train = Spectrogram(n_fft=self._n_fft, power=None, hop_length=self._hop_length)

        self.sp2mel = MelScale(sample_rate=self._audio_samplerate, n_mels=self._n_mel)
        self.mel2sp = InverseMelScale(n_stft=self._sp_freq_shape, n_mels=self._n_mel,
                                      sample_rate=self._audio_samplerate)
        # self.istft = partial(istft, n_fft=self._n_fft, hop_length=self._hop_length, length=self._audio_length,
        #                      window=torch.hann_window(self._n_fft).cuda())  # Not ready for multiGPU
        self._window = torch.hann_window(self._n_fft)

    def istft(self, x):
        return istft(x, n_fft=self._n_fft, hop_length=self._hop_length, length=self._audio_length, window=self._window)

    def wav2sp(self, src):  # Func should be eventually deleted
        return self.wav2sp_train(src)

    def sp2wav(self, inference_mask, mixture):
        upsample = self.downsample_interp or self.downsample_coarse
        if inference_mask.is_complex() and upsample:
            inference_mask = torch.view_as_real(inference_mask).permute(0, 3, 1, 2)
        elif self.binary_mask:
            inference_mask = inference_mask.unsqueeze(1)
        if self.downsample_coarse:
            inference_mask = torch.nn.functional.upsample(inference_mask, scale_factor=(2, 1), mode='nearest').squeeze(
                1)
            if self.complex_enabled:
                inference_mask = torch.view_as_complex(inference_mask.permute(0, 2, 3, 1).contiguous())
        if self.downsample_interp:
            raise NotImplementedError
        estimated_sp = inference_mask * mixture
        estimated_wav = self.istft(estimated_sp)
        return estimated_wav, estimated_sp

    def _define_video_network(self):
        if self.video_enabled or self.llcp_enabled:
            # Mixted convolutional network for video a analysis
            self.motion_net = Video_Model(upsample=False) if self.llcp_enabled else mc3_18(pretrained=True)
            self.feat_num += 256
            self.temporal_pooling = nn.AdaptiveAvgPool2d((self.video_temporal_features * self._n, None))

    def _define_graph_network(self, pooling, graph_kwargs):
        if self.sk_enabled:
            self.feat_num += 256
            # Graph convolutional network for skeleton analysis
            if pooling == 'AdaptativeAP':
                self.pool = nn.AdaptiveAvgPool2d((None, 1))
            elif pooling == 'AdaptativeMP':
                self.pool = nn.AdaptiveMaxPool2d((None, 1))
            elif pooling == 'linear':
                self.pool = nn.Linear(self.graph_net.heads[0].graph.num_node, 1, bias=False)
            else:
                raise ValueError(
                    'VnNet pooling type: %s not implemented. Choose between AdaptativeMP,AdaptativeMP or linear' % pooling)

            if graph_kwargs['graph_cfg']['layout'] == 'upperbody_with_hands':
                in_channels = 3
            elif graph_kwargs['graph_cfg']['layout'] == 'acappella':
                in_channels = 2
            else:
                raise NotImplementedError
            self.graph_net = ST_GCN(in_channels=in_channels, num_class=7, **graph_kwargs)

    def _define_appearence_network(self):
        if not (self.single_emb_enabled or self.single_emb_enabled):
            return None
        from facenet_pytorch import InceptionResnetV1
        if self.single_frame_enabled:
            appearance_net = InceptionResnetV1(pretrained='vggface2')
            appearance_net.last_bn = nn.Identity()
        elif self.single_emb_enabled:
            appearance_net = nn.Identity()
        self.appearance_net = nn.Sequential(appearance_net,
                                            nn.Linear(512, self.feat_num), nn.ReLU())

    def _define_multitask_layers(self, multitask, layer_distribution, activation, pooling, **kwargs):
        N = self.dim[-2]
        if self.multitask:
            assert (
                    self.video_enabled or self.llcp_enabled or self.sk_enabled), f'Video or skeleton has to be enabled to use multitask'
            if multitask['independent_encoder']:
                self.mt_encoder = self.init_mt_encoder(layer_distribution,
                                                       2 if self.complex_enabled else 1,
                                                       film=None,
                                                       useBN=True,
                                                       activation=activation,
                                                       input_channels=2 if self.complex_enabled else 1,
                                                       **kwargs)
            else:
                self.mt_encoder = self.encoder
            self.multitask_loss = self._init_mt_loss(multitask['loss'])
            self.sp_relu = nn.ReLU(inplace=False)
            if pooling == 'AdaptativeAP':
                self.sp_pooling = nn.AdaptiveAvgPool2d((None, 1))
            elif pooling == 'AdaptativeMP':
                self.sp_pooling = nn.AdaptiveMaxPool2d((None, 1))

    def _init_mt_loss(self, loss):
        if loss == 'ContrastiveLoss':
            return ContrastiveLoss()  # Criterion: 1--> Sync samples, 0--> unsync samples
        elif loss == 'L1Loss':
            return nn.L1Loss()
        elif loss == 'L2Loss':
            return nn.MSELoss()
        else:
            raise ValueError(f'Multitask loss should be L1Loss,L2Loss or ContrastiveLoss'
                             f' but {loss} found.')

    def init_mt_encoder(self, *args, **kwargs):
        model = UNet(*args, **kwargs).encoder
        return model

    def forward(self, inputs: dict, real_sample=False):
        """
        Inputs contains the following keys:
           audio: the main audio waveform of shape N,M
           audio_acmt: the secondary audio waveform of shame N,M
           src: If using inference on real mixtures, the mixture audio waveform of shape N,M
           sk: skeletons of shape N,C,T,V,M which is batch, channels, temporal, joints, n_people
           video: video
        """

        output = {'logits_mask': None,
                  'inference_mask': None,
                  'loss_mask': None,
                  'gt_mask': None,
                  'separation_loss': None,
                  'alignment_loss': None,
                  'estimated_sp': None,
                  'estimated_wav': None}

        self.n_sources = 3 if self.remix_input else 2
        with torch.no_grad():
            sk = inputs.get('landmarks')
            v = inputs.get('llcp_embedding').permute(0, 2, 1) if self.llcp_enabled else inputs.get('video')
            frames = inputs.get('single_frame')

            # Inference for a real mixture
            if real_sample:
                self.n_sources = 2
                # Inference in case of a real sample
                sp_mix_raw = self.wav2sp(inputs['src']) / self.n_sources
                spm = sp_mix_raw  # Mock up real sample inference

                if self.downsample_coarse:
                    sp_mix = sp_mix_raw[:, ::2, ...]  # BxFxTx2
                elif self.downsample_interp:
                    raise NotImplementedError
                else:
                    sp_mix = sp_mix_raw
            else:

                srcm = inputs['audio']
                srcs = inputs['audio_acmt']

                # Computing STFT
                spm = self.wav2sp(srcm)  # Spectrogram main BxFxTx2
                sps = self.wav2sp(srcs)  # Spectrogram secondary BxFxTx2

                if self.remix_input:
                    B = spm.shape[0]  # Batch elements
                    ndim = spm.ndim
                    coef = (torch.rand(B, *[1 for _ in range(ndim - 1)], device=spm.device) < self.remix_coef).byte()
                    sources = [spm, sps, spm.flip(0) * coef]
                else:
                    sources = [spm, sps]
                sp_mix_raw = sum(sources) / self.n_sources
                # Downsampling to save memory
                if self.downsample_coarse:
                    spm = spm[:, ::2, ...]
                    sps = sps[:, ::2, ...]
                    sp_mix = sp_mix_raw[:, ::2, ...]  # BxFxTx2
                elif self.downsample_interp:
                    raise NotImplementedError
                else:
                    sp_mix = sp_mix_raw

                mag = sp_mix.norm(dim=-1)  # Magnitude spectrogram BxFxT

                if self.weighted_loss:
                    weight = torch.log1p(mag)
                    weight = torch.clamp(weight, 1e-3, 10)
            if self.complex_enabled:
                x = sp_mix.permute(0, 3, 1, 2)
                sp0 = spm.permute(0, 3, 1, 2)
            elif self.log_sp_enabled:
                epsilon = 1e-4
                x = (mag + epsilon).log().unsqueeze(1)
                sp0 = (spm.norm(dim=-1) + epsilon).log().unsqueeze(1)
                sp1 = (sps.norm(dim=-1) + epsilon).log().unsqueeze(1)

            elif self.mel_enabled:
                x = self.sp2mel(mag).unsqueeze(1)
                sp0 = spm.norm(dim=-1).unsqueeze(1)
                raise NotImplementedError('Option not implemented in depth. Draft written.')
            else:
                x = mag.unsqueeze(1)
                sp0 = spm.norm(dim=-1).unsqueeze(1)
                sp1 = sps.norm(dim=-1).unsqueeze(1)
        if self.white_metrics:
            output['raw_mix_wav'] = self.istft(sp_mix_raw)
            return output
        pred = self.core_forward(skeleton=sk, sp_mix=x,
                                 video=v, sp0=sp0,
                                 frames=frames)
        logits_mask = pred['mask'] if self.complex_enabled else pred['mask'].squeeze(
            1)  # Predicted mask shape is BxCxFxT
        output['logits_mask'] = logits_mask

        # Masks are typically bounded when used as loss
        # Two types are computed:
        # loss_mask is the one which would be used in a loss function
        # inference_mask is the one that, being multiplied by the mixture, gives the indepoendent source
        if self.loss_on_mask and not real_sample:

            if self.complex_enabled:
                loss_mask = self.tanh(logits_mask).permute(0, 2, 3, 1)
                gt_mask = self.complex_mask(spm, sp_mix)

            elif self.binary_mask:
                gt_mask = self.hard_binary_mask(sp0, sp1).squeeze(1)  # Opt 1: hard binary mask
                # gt_mask = self.complex_mask(spm, sp_mix) # Opt 1: soft binary mask
                # We can still use log here cos log is monotonically growing
                loss_mask = logits_mask
            else:
                raise NotImplementedError(f'Not tested')
                loss_mask = torch.relu(logits_mask)
            output['loss_mask'] = loss_mask
            output['gt_mask'] = gt_mask

        if not self.training or (self.training and not self.loss_on_mask):
            # Inference or training but loss on the spectrogram (A or B)
            if self.binary_mask:  # A or B for binary masks
                inference_mask = torch.sigmoid(logits_mask)
            elif self.complex_enabled:  # Case A or B for complex numbers
                inference_mask = torch.view_as_complex(logits_mask.permute(0, 2, 3, 1).contiguous())
            else:  # Case A or B for ratio masks (no complex no binary-->ratio)
                inference_mask = logits_mask
                raise NotImplementedError('Branch not tested')
            inference_mask = self.n_sources * inference_mask
            target_sp = inference_mask * torch.view_as_complex(sp_mix)
            output['inference_mask'] = inference_mask

        if not real_sample:

            if self.loss_on_mask:
                loss = self.compute_loss(loss_mask, gt_mask, weight=weight if self.weighted_loss else None)
            else:
                loss = self.compute_loss(target_sp, spm, weight=weight if self.weighted_loss else None)

            if self.multitask:
                sp = self.sp_relu(
                    pred['ind_end_feats'])  # Apply relu because the last layer of video part is a relu
                sp = self.sp_pooling(sp)[..., 0].permute(0, 2, 1)
                multitask_loss = self.multitask_loss(pred['visual_features'], sp)
                output['alignment_loss'] = multitask_loss
            output['separation_loss'] = loss

        # Reconstructing wav signal
        if not self.training:
            # Upsampling must be carried out on the mask, NOT the spectrogram
            # https://www.juanmontesinos.com/posts/2021/02/08/bss-masking/
            if self.binary_mask:
                inference_mask = (inference_mask > 0.5).float()
            estimated_wav, estimated_sp = self.sp2wav(inference_mask.detach(), torch.view_as_complex(sp_mix_raw))
            raw_mix_wav = self.istft(sp_mix_raw)
            output['estimated_sp'] = estimated_sp
            output['estimated_wav'] = estimated_wav
            output['raw_mix_wav'] = raw_mix_wav
            output['mix_sp'] = torch.view_as_complex(sp_mix_raw)
        return output

    def core_forward(self, *, skeleton, sp_mix, video, sp0, frames):
        sp = sp0.clone().detach()
        x = sp_mix.clone().detach()
        # skeleton = skeleton.clone().detach() if skeleton is not None else None
        video = video.clone().detach() if video is not None else None
        frames = frames.clone().detach() if frames is not None else None
        dense_visual_enabled = self.video_enabled or self.llcp_enabled
        visual_enabled = dense_visual_enabled or self.sk_enabled

        if self.single_frame_enabled:
            frames_shape = frames.shape
            frames = frames.permute(0, 2, 1, 3, 4).reshape(-1, *frames_shape[2:])
            aprc_feats = self.appearance_net(frames).reshape(*frames_shape[:2], self.feat_num).max(dim=1)[0]
        elif self.single_emb_enabled:
            aprc_feats = self.appearance_net(frames).max(dim=1)[0]
        if visual_enabled:
            if self.sk_enabled:
                sk_features = self.graph_net(skeleton)  # Shape: N,256,4, 18/21/58
                sk_features = self.pool(sk_features).squeeze(3)
                sk_features = torch.nn.functional.interpolate(sk_features, size=16 * self._n)
                sk_features.transpose_(1, 2)  # Shape: N,4,256
            if dense_visual_enabled:
                motion_features = self.motion_net(video) if self.llcp_enabled else self.motion_net(video)[..., 0]
                motion_features = self.temporal_pooling(motion_features).squeeze(-1)
                # Shape: N,256,100
                motion_features.transpose_(1, 2)

            if dense_visual_enabled and self.sk_enabled:
                visual_features = torch.cat([sk_features, motion_features], dim=2)
                if self.transformer_enabled:
                    visual_features = self.resampling_layer(visual_features)
            elif dense_visual_enabled:
                visual_features = motion_features
            elif self.sk_enabled:
                visual_features = sk_features
            if self.single_emb_enabled or self.single_emb_enabled:
                visual_features = visual_features + aprc_feats.unsqueeze(1)
        if self.printing:
            print('UNet input size {0}'.format(x.size()))
        to_cat_vector = []
        for i in range(len(self.dim) - 1):
            if self.printing:
                print('Forward Prop through DownConv block {}'.format(i))

            to_cat, x = self.encoder[i](x)
            if self.multitask:
                _, sp = self.mt_encoder[i](sp)
            to_cat_vector.append(to_cat)
            if i == len(self.dim) - 2 and visual_enabled:
                x = self.fusion_layer(visual_features, x)
        for i in self.vec:
            if self.printing:
                print('Concatenating and Building  UpConv Block {}'.format(i))
            if i == 0:

                x = self.decoder[i](x)
            else:

                x = self.decoder[i](x, to_cat_vector[-i])
        x = self.final_conv(x)
        if self.activation is not None:
            x = self.final_act(x)
        if self.printing:
            print('UNet Output size {}'.format(x.size()))
        output = {'mask': x, 'ind_end_feats': None, 'visual_features': None}
        if self.multitask:
            output['ind_end_feats'] = sp
        if visual_enabled:
            output['visual_features'] = visual_features
        return output

    def compute_loss(self, pred, gt, weight):

        if self.loss_on_mask and self.binary_mask:  # Loss on mask when it's binary uses BCE, otherwise norms
            assert pred.shape == gt.shape, 'Mask computation: Ground truth and predictions has to be the same shape'
            if self.weighted_loss:
                loss = binary_cross_entropy_with_logits(pred, gt, weight)
            else:
                loss = binary_cross_entropy_with_logits(pred, gt)
        else:
            if self.complex_enabled or not self.loss_on_mask:
                # Complex mask and gt shape BxFxTxC, weight unsqueezing required for broadcasting
                # The same applied for direct estimation, in which the mask multiplies the mixture as real + imag
                # However ratio masks are applied over the magnitude so no broadcasting is used
                weight = weight.unsqueeze(-1)
                pred = torch.view_as_real(pred) if not self.loss_on_mask else pred
            assert pred.shape == gt.shape, 'Mask computation: Ground truth and predictions has to be the same shape'
            if self.loss_criterion.lower() == 'mse':
                if self.weighted_loss:
                    loss = (weight * (pred - gt).pow(2)).mean()
                else:
                    loss = mse_loss(pred, gt)
            elif self.loss_criterion.lower() == 'l1':
                if self.weighted_loss:
                    loss = (weight * (pred - gt).abs()).mean()
                else:
                    loss = l1_loss(pred, gt)
            elif self.loss_criterion.lower() == 'l1':
                raise Exception('I fucked  it up with the if statements')
        return loss

    @staticmethod
    @torch.no_grad()
    def hard_binary_mask(sp0, sp1):
        return (sp0 > sp1).float()

    @staticmethod
    def soft_binary_mask(sp0, sp_mix):
        return (sp0 >= sp_mix / 2).float()

    @staticmethod
    def tanh(x):
        # *(1-torch.exp(-C * x))/(1+torch.exp(-C * x))
        # Compute this formula but using torch.tanh to deal with asymptotic values
        # Manually coded at https://github.com/vitrioil/Speech-Separation/blob/master/src/models/complex_mask_utils.py
        return K * torch.tanh(x)

    @staticmethod
    def itanh(x):
        return -torch.log((K - x) / (K + x))

    @torch.no_grad()
    def complex_mask(self, sp0, sp_mix, eps=torch.finfo(torch.float32).eps):
        # Bibliography about complex masks
        # http://homes.sice.indiana.edu/williads/publication_files/williamsonetal.cRM.2016.pdf
        assert sp0.shape == sp_mix.shape
        sp0 = torch.view_as_complex(sp0)
        sp_mix = torch.view_as_complex(sp_mix) + eps
        mask = torch.view_as_real(sp0 / sp_mix) / self.n_sources
        mask_bounded = self.tanh(mask)
        return mask_bounded

    # VISUALIZATION TOOLS
    # CONVENTION OF BATCH FIRST
    @torch.no_grad()
    def save_video(self, batch_idx, video, path, undo_statistics=True):
        from imageio import mimwrite
        if undo_statistics:
            mean = torch.tensor(self._mean)[None, None, None, :]
            std = torch.tensor(self._std)[None, None, None, :]
        video = video[batch_idx]

        if undo_statistics:
            video = video.permute(1, 2, 3, 0).detach().cpu()
            video = ((video * std + mean) * 255).byte().numpy()
        mimwrite(path, [x for x in video],
                 fps=25)

    @torch.no_grad()
    def save_landmarks(self, batch_idx, landmarks, path, sequence=[]):
        from ..utils import overlay_sk as sk
        from imageio import mimwrite
        landmarks = landmarks[batch_idx]
        video = list(sk.get_video_array(landmarks, self.graph_net.graph.edge, 1, sequence=sequence))
        mimwrite(path, [x for x in video],
                 fps=25)

    @torch.no_grad()
    def save_audio(self, batch_idx, waveform, path):
        assert waveform.ndim == 2
        from scipy.io.wavfile import write
        write(path, self._audio_samplerate, waveform[batch_idx].detach().cpu().numpy())

    @torch.no_grad()
    def save_loss_mask(self, batch_idx, loss_mask, gt_mask, path):
        if self.binary_mask:
            loss_mask = torch.sigmoid(loss_mask[batch_idx]).detach().cpu().numpy()
            fig, ax = plt.subplots(nrows=1, ncols=3)
            ax[0].imshow(loss_mask)
            ax[0].set(title='Predicted Sigmoid Mask')
            ax[0].label_outer()
            ax[1].imshow((loss_mask >= 0.5).astype(float32))
            ax[1].set(title='Predicted Binary Mask')
            ax[1].label_outer()
            im = ax[2].imshow(gt_mask[batch_idx].cpu().numpy())
            ax[2].set(title='Ground Truth Mask')
            ax[2].label_outer()
            cbaxes = fig.add_axes([0.28, 0.9, 0.49, 0.03])
            fig.colorbar(im, ax=ax, cax=cbaxes, orientation="horizontal")
            fig.tight_layout()
            fig.savefig(path, dpi=fig.dpi)

        if self.complex_enabled:
            gt_mask = torch.view_as_complex(gt_mask[batch_idx].detach().cpu())
            gt_mask_mag = gt_mask.abs().numpy()
            gt_mask_real = gt_mask.real.numpy()
            gt_mask_imag = gt_mask.imag.numpy()

            fig, ax = plt.subplots(nrows=2, ncols=3, constrained_layout=True)
            fig.set_size_inches(14, 8, forward=True)
            fig.subplots_adjust(right=0.8, )
            ax[0][0].set(title='Magnitude GT mask ')
            imm = ax[0][0].imshow(gt_mask_mag)
            ax[0][1].set(title='Real GT mask ')
            imr = ax[0][1].imshow(gt_mask_real)
            ax[0][2].set(title='Imag GT mask ')
            ax[0][2].imshow(gt_mask_imag)
            ax[0][2].label_outer()

            loss_mask = torch.view_as_complex(loss_mask[batch_idx].detach().cpu().contiguous())
            loss_mask_mag = loss_mask.abs().numpy()
            loss_mask_real = loss_mask.real.numpy()
            loss_mask_imag = loss_mask.imag.numpy()
            ax[1][0].set(title='Magnitude pred mask ')
            ax[1][0].imshow(loss_mask_mag)
            ax[1][1].set(title='Real pred mask ')
            ax[1][1].imshow(loss_mask_real)
            ax[1][2].set(title='Imag pred mask ')
            ax[1][2].imshow(loss_mask_imag)
            ax[1][2].label_outer()
            cbaxes = fig.add_axes([0.32, 0.12, 0.025, 0.75])
            cbaxes2 = fig.add_axes([0.64, 0.12, 0.025, 0.75])
            fig.colorbar(imm, ax=ax[:, 0], cax=cbaxes)
            fig.colorbar(imr, ax=ax[:, 1:], cax=cbaxes2)
            fig.tight_layout()
            fig.savefig(path, dpi=fig.dpi, bbox_inches='tight')
        plt.close('all')
