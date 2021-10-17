from functools import partial
import os
import shutil

import torch

import matplotlib.pyplot as plt

from flerken.framework.framework import Trainer
from flerken.framework.meters import TensorStorage, TensorHandler, get_nested_meter, get_loss_meter
from flerken.utils.losses import SI_SDR
from flerken.utils import BaseDict
from torch_mir_eval import bss_eval_sources

from .utils.overlay_sk import get_video_array

__all__ = ['Trainer']


@torch.no_grad()
def bss_eval(reference_sources, estimated_sources, compute_permutation=False):
    # sources are required not to be silent
    # Filtering out silent sources
    indices = torch.arange(reference_sources.shape[0])
    energy_rs = reference_sources.sum(dim=-1).abs().min(dim=-1)[0]
    energy_es = estimated_sources.sum(dim=-1).abs().min(dim=-1)[0]
    rs_idx = indices[energy_rs == 0].tolist()
    es_idx = indices[energy_es == 0].tolist()
    forbidden_indices = set(rs_idx + es_idx)
    indices = set(indices.tolist())
    indices = list(indices - forbidden_indices)
    sdr = torch.tensor([torch.Tensor([float('NaN')]) for _ in range(reference_sources.shape[0])])
    sir = torch.tensor([torch.Tensor([float('NaN')]) for _ in range(reference_sources.shape[0])])
    sar = torch.tensor([torch.Tensor([float('NaN')]) for _ in range(reference_sources.shape[0])])

    rs = reference_sources[indices]
    es = estimated_sources[indices]
    sdr_pred, sir_pred, sar_pred, _ = bss_eval_sources(rs, es, compute_permutation=compute_permutation)

    sdr[indices] = sdr_pred[:, 0].detach().cpu()
    sir[indices] = sir_pred[:, 0].detach().cpu()
    sar[indices] = sar_pred[:, 0].detach().cpu()

    condition = torch.isnan(sdr) | torch.isnan(sir) | torch.isnan(sar) | \
                torch.isinf(sdr) | torch.isinf(sir) | torch.isinf(sar)

    return ({'sdr': sdr[~condition].mean().detach().cpu(),
             'sir': sir[~condition].mean().detach().cpu(),
             'sar': sar[~condition].mean().detach().cpu()}, {'sdr': sdr.detach().cpu(),
                                                             'sir': sir.detach().cpu(),
                                                             'sar': sar.detach().cpu()})


class Trainer(Trainer):
    def __init__(self, main_device, model: torch.nn.Module, dataparallel: bool, input_shape, *,
                 debug, multitask, n_epochs, criterion, initializable_layers,
                 dump_iteration_files, white_metrics):
        super(Trainer, self).__init__(main_device, model, dataparallel, input_shape)
        self.debug = debug
        self.multitask = multitask
        self.EPOCHS = n_epochs
        self.criterion = criterion
        self.dump_files = dump_iteration_files
        self.white_metrics = white_metrics
        self._model.initializable_layers = initializable_layers

    def loader_mapping(self, x):
        (inputs, a, b, c) = x
        if isinstance(inputs, (list, tuple)):

            return {'gt': None, 'inputs': inputs, 'vs': {'trace': [x.trace for x in a], 'datasets': b, 'indices': c}}
        else:
            return {'gt': None, 'inputs': [inputs], 'vs': {'trace': [x.trace for x in a], 'datasets': b, 'indices': c}}

    def backpropagate(self, debug):
        self.optimizer.zero_grad()
        if self.model.complex_enabled and self.model.loss_on_mask:
            if torch.isnan(self.loss).any() or torch.isinf(self.loss).any():
                return None
        elif debug['isnan']:
            assert not torch.isnan(self.loss).any()
            assert not torch.isinf(self.loss).any()
        self.loss.backward()
        self.gradients()
        self.optimizer.step()
    # def gradients(self):
    #     torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)

    def hook(self, vrs):
        pred, separation_loss, alignment_loss = vrs['pred']['inference_mask'], \
                                                vrs['pred']['separation_loss'], \
                                                vrs['pred']['alignment_loss']

        if alignment_loss is not None:
            weighted_loss, coef = vrs['pred']['weighted_loss'], \
                                  vrs['pred']['loss_coef']

            self.IO.writer.add_scalars(f'{self.state}_MTWloss',
                                       {'loss_mt': weighted_loss[1].detach(),
                                        'loss_sep': weighted_loss[0].detach()},
                                       global_step=self.absolute_iter)

            self.IO.writer.add_scalars(f'{self.state}_loss_coef',
                                       {'mt': coef[1].detach(),
                                        'sep': coef[0].detach()},
                                       global_step=self.absolute_iter)

            self.IO.writer.add_scalars(f'{self.state}_MTloss',
                                       {'loss_mt': alignment_loss.detach(),
                                        'loss_sep': separation_loss.detach()},
                                       global_step=self.absolute_iter)

        if (self.dump_files['enabled'] and self.dump_files[self.state]['enabled'] and \
            (self.absolute_iter % self.dump_files[self.state]['iter_freq']) == 0 and \
            (self.epoch % self.dump_files[self.state]['epoch_freq'] == 0)) or self.dump_files['force']:
            dump_path = os.path.join(self.IO.workdir, self.state, str(self.epoch))
            audios = vrs['inputs'][0]['audio']
            audios_acmt = vrs['inputs'][0].get('audio_acmt')
            videos = vrs['inputs'][0].get('video')
            landmarks = vrs['inputs'][0].get('landmarks')  # N,C,T,V,M  batch, channels, temporal, joints, n_people
            traces = vrs['vs']['trace']
            estimated_wav = vrs['pred']['estimated_wav']
            inference_mask = vrs['pred']['inference_mask']
            loss_mask = vrs['pred']['loss_mask']

            if self.model.binary_mask:
                loss_mask = torch.sigmoid(loss_mask)
            if inference_mask is not None and self.model.complex_enabled:
                inference_mask = self.model.tanh(torch.view_as_real(inference_mask))
            for i in range(len(traces)):
                file_path = os.path.join(dump_path, f'{self.epoch:03d}it_{self.absolute_iter:06d}_el{i:02d}')
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                trace = BaseDict(traces[i])
                if not self.white_metrics:
                    al_loss = vrs['pred']['alignment_loss']
                    al_loss = al_loss.item() if al_loss is not None else 'not_computed'
                    loss = BaseDict({'separation_loss': vrs['pred']['separation_loss'].item(),
                                     'alignment_loss': al_loss})
                    trace['loss'] = loss
                try:
                    trace['sdr'] = vrs['bss_per_sample']['sdr'][i].item()
                    trace['sar'] = vrs['bss_per_sample']['sar'][i].item()
                    trace['sir'] = vrs['bss_per_sample']['sir'][i].item()
                    trace['si-sdr'] = vrs['bss']['si-sdr'].item()
                except KeyError:
                    pass
                trace.write(os.path.join(file_path, f'metadata.json'))
                if self.white_metrics:
                    continue
                if self.dump_files['audio']:
                    self.model.save_audio(i, audios, os.path.join(file_path, 'waveform_main.wav'))
                if videos is not None and self.dump_files['video']:
                    self.model.save_video(i, videos, os.path.join(file_path, 'video.mp4'))
                if (landmarks is not None) and self.dump_files['landmarks']:
                    landmarks_as_videos = []
                    for ld in landmarks:
                        ld = ld.permute(1, 0, 2, 3)[..., 0]
                        fake = ld.new_ones(ld.shape[0], 1, ld.shape[-1])
                        ld = torch.cat([ld, fake], dim=1)
                        video_i = torch.stack([torch.from_numpy(x)
                                               for x in get_video_array(ld, self.model.graph_net.graph.edge, 1, )])
                        landmarks_as_videos.append(video_i)

                    landmarks_as_videos = torch.stack(landmarks_as_videos)
                    self.model.save_video(i, landmarks_as_videos, os.path.join(file_path, 'landmarks.mp4'),
                                          undo_statistics=False)
                if (audios_acmt is not None) and self.dump_files['audio']:
                    self.model.save_audio(i, audios_acmt, os.path.join(file_path, 'waveform_slave.wav'))
                    try:
                        self.model.save_audio(i, vrs['pred']['raw_mix_wav'],
                                              os.path.join(file_path, 'waveform_mix.wav'))
                    except KeyError:
                        pass

                if (estimated_wav is not None) and self.dump_files['audio']:
                    self.model.save_audio(i, estimated_wav, os.path.join(file_path, 'waveform_estimated.wav'))
                if (loss_mask is not None) and self.dump_files['masks']:
                    gt_mask = vrs['pred']['gt_mask']
                    self.model.save_loss_mask(i, loss_mask, gt_mask, os.path.join(file_path, 'loss_mask.png'))

                if (inference_mask is not None) and self.dump_files['masks']:
                    self.model.save_loss_mask(i, inference_mask, inference_mask,
                                              os.path.join(file_path, 'inference_mask.png'))
                plt.close('all')

    def init_metrics(self):
        self.init_loss()
        self.metrics['bss_eval'] = get_bss_meter()
        self.set_tensor_scalar_item('si-sdr/ds')

        self.set_tensor_scalar_item('sdr/ds')
        self.set_tensor_scalar_item('sar/ds')
        self.set_tensor_scalar_item('sir/ds')

        if self.multitask:
            self.set_tensor_scalar_item('loss_mt')
            self.set_tensor_scalar_item('loss_sep')

    def pred_mapping(self, x):
        if 'train' not in self.state:
            f = SI_SDR()
            audio = x['inputs'][0]['audio']
            audio_acmt = x['inputs'][0]['audio_acmt']
            mixture = x['pred']['raw_mix_wav']
            estimated_audio = x['pred']['estimated_wav']
            gt_audios = torch.stack([audio, audio_acmt], dim=1)
            if self.white_metrics:
                estimated_audios = torch.stack([mixture, mixture], dim=1)
            else:
                estimated_audio_acmt = 2 * mixture - estimated_audio
                estimated_audios = torch.stack([estimated_audio, estimated_audio_acmt], dim=1)
            real_mean, real_all = bss_eval(gt_audios, estimated_audios)
            if self.white_metrics:
                real_mean['si-sdr'] = f(audio, mixture)
            else:
                real_mean['si-sdr'] = f(audio, estimated_audio)
            x['bss_per_sample'] = real_all
            x['bss'] = self.alloc(real_mean, device='cpu')

        return x


def get_bss_meter():
    f = SI_SDR()

    @torch.no_grad()
    def pred2sisdr(bss_real):
        return bss_real['si-sdr']

    @torch.no_grad()
    def gt2sisdr(bss_oracle, bss_real):
        return bss_oracle['si-sdr']

    def pred2sdr(bss_real):
        return bss_real['sdr']

    def pred2sir(bss_real):
        return bss_real['sir']

    def pred2sar(bss_real):
        return bss_real['sar']

    handlers = {}
    handlers['bss'] = lambda x: x

    handlers['si-sdr'] = pred2sisdr

    handlers['sdr'] = pred2sdr

    handlers['sir'] = pred2sir

    handlers['sar'] = pred2sar

    opt = {'bss': {'type': 'input', 'store': 'list'},
           'si-sdr': {'type': 'output', 'store': 'list'},
           'sdr': {'type': 'output', 'store': 'list'},
           'sar': {'type': 'output', 'store': 'list'},
           'sir': {'type': 'output', 'store': 'list'},
           }
    return get_nested_meter(
        partial(TensorStorage, handlers=handlers, opt=opt, on_the_fly=True,
                redirect={'si-sdr': 'si-sdr/ds',
                          'sdr': 'sdr/ds',
                          'sir': 'sir/ds',
                          'sar': 'sar/ds',
                          }), 1)
