import acappella_info
import inspect
import numpy as np
import os
import torch
from flerken.datasets import AVDataset
from flerken.datasets.audiovisual_dataset import AudioReader
from flerken.utils import BaseDict
from random import randint, choice, shuffle, seed as set_seed
from scipy.io.wavfile import read
from warnings import warn

from .readers import *


def np_check_sing(x):
    return np.any(np.isnan(x)) or np.any(np.isinf(x))


def torch_check_sing(x):
    return torch.isnan(x).any() or torch.isinf(x).any()


class BaseDataHandler(AVDataset):
    """
    Audiovisual Base Data handler
    """

    def __init__(self,
                 # Audiovisual tasks
                 multitask: bool,
                 # Flags
                 video_enabled: bool, skeleton_enabled: bool,
                 # Keys
                 dataset_path, audio_exclude, audio_key, video_key, skeleton_key,
                 # Dataset_args
                 debug: bool, yield_mode: str, visualization,
                 audio_video_rate, audio_length, n_video_frames, n_skeleton_frames,
                 **kwargs):
        super(BaseDataHandler, self).__init__(dataset_path, in_memory=True, as_generator=False, exclude=audio_exclude,
                                              debug=debug,
                                              yield_mode=yield_mode,
                                              **kwargs)
        self._rs2idx = {x: i for i, x in enumerate(self.filemanager.resources)}
        self.N_SEC_FRAMES = 10

        # Flags
        self.multitask = multitask
        self.video_enabled = video_enabled
        self.skeleton_enabled = skeleton_enabled
        self.crop_mouth_enabled = False  # Flag to be used by inherited classes
        self.single_frame_enabled = False  # Flag to be used by inherited classes
        self.single_emb_enabled = False  # Flag to be used by inherited classes
        self.savgol_enabled = False
        # Keys
        self.ad_key = audio_key
        self.vd_key = video_key
        self.skeleton_key = skeleton_key

        # audiovisual opts
        self.AV_RATE = audio_video_rate
        self.AUDIO_LENGTH = audio_length
        self.N_VIDEO_FRAMES = n_video_frames
        self.N_SK_FRAMES = n_skeleton_frames

        self.vis = visualization

    def get_idx_kwargs(self, idx: int):
        """
        Given an index, this function generates the kwargs for the each modality reader to do his job.
        """
        kw = {'audio': {'offset': round(idx * self.AV_RATE), 'length': self.AUDIO_LENGTH},
              self.ad_key: {'offset': round(idx * self.AV_RATE), 'length': self.AUDIO_LENGTH}}
        if self.video_enabled or self.single_frame_enabled:
            kw[self.vd_key] = {'offset': idx, 'length': self.N_VIDEO_FRAMES}
            kw['videos'] = {'offset': idx, 'length': self.N_VIDEO_FRAMES}
        if self.skeleton_enabled:
            kw['landmarks'] = {'offset': idx, 'length': self.N_VIDEO_FRAMES}
        if self.llcp_enabled or self.single_emb_enabled:
            kw[self.llcp_key] = {'offset': idx, 'length': self.N_VIDEO_FRAMES}
        return kw

    def valid_interval(self, init_ts, sample_timestamps, length):
        is_valid = False
        for sample_timestamp in sample_timestamps:
            init = sample_timestamp[0]
            fin = sample_timestamp[1]
            if (init <= init_ts <= fin) and (init <= (init_ts + length - 1) <= fin):
                is_valid = True
                break
        return is_valid

    def precompute_epoch(self, *, batch_size: int, n: int, overfit: bool, seed: int = None, n_sources=1, classes=None,
                         sequential_inference=False):
        """
        Predifines which  elements are going to be loaded before computing the epoch
        """
        traces = []
        timestamps = acappella_info.get_timestamps()
        if sequential_inference:
            idx = self._rs2idx['frames']
            _, trace_example = self.getitem(0, n_sources, [], trazability=True,
                                            repeat_class=True, classes=classes)
            for j, sample in enumerate(self.filemanager.indexable):
                video_path = sample[idx]
                sample_id = os.path.basename(video_path)[:-4]
                sample_timestamps = timestamps[sample_id]
                T = np.load(video_path, mmap_mode='r').shape[0]
                n_elements = T // self.N_VIDEO_FRAMES
                for i in range(n_elements):
                    if self.valid_interval(i * self.N_VIDEO_FRAMES, sample_timestamps, self.N_VIDEO_FRAMES):
                        trace = {'indices': [j],
                                 'kwargs': [{'audio': {'offset': i * self.AUDIO_LENGTH, 'length': self.AUDIO_LENGTH},
                                             'frames': {'offset': i * self.N_VIDEO_FRAMES,
                                                        'length': self.N_VIDEO_FRAMES},
                                             'llcp_embed': {'offset': i * self.N_VIDEO_FRAMES,
                                                            'length': self.N_VIDEO_FRAMES}}]}
                        traces.append(trace)
                    else:
                        sample_id
        else:
            set_seed(seed)
            while len(traces) < n:
                for idx in range(len(self)):
                    _, trace = self.getitem(idx, n_sources, [], trazability=True,
                                            repeat_class=True, classes=classes)
                    traces.append(trace)
                    if len(traces) > n:
                        break
            if overfit:
                r = len(traces) / batch_size
                traces_overfit = traces[:batch_size]
                traces = []
                for i in range(int(r)):
                    traces = traces + traces_overfit
            else:
                shuffle(traces)
            n_samples = (len(traces) // batch_size) * batch_size
            traces = traces[:n_samples]
        return traces

    @property
    def yield_mode(self):
        return self.filemanager.yield_mode


class AudiosetDataHandler(BaseDataHandler):
    def __init__(self, *,
                 # Audiovisual tasks
                 onehot,
                 # Acappella kwargs
                 root, audio_exclude, audio_key, audio_length,
                 # Dataset_args
                 debug: bool, yield_mode: str, visualization, **kwargs):

        video_enabled = False
        skeleton_enabled = False
        multitask = False

        video_key = None
        skeleton_key = None

        n_skeleton_frames = None
        n_video_frames = None
        audio_video_rate = None
        super(AudiosetDataHandler, self).__init__(multitask,
                                                  video_enabled, skeleton_enabled,
                                                  root, audio_exclude, audio_key, video_key, skeleton_key,
                                                  debug, yield_mode, visualization,
                                                  audio_video_rate, audio_length, n_video_frames, n_skeleton_frames,
                                                  **kwargs)
        if self.skeleton_enabled:
            sk_dict = BaseDict().load(self.sk_dict_path)
            for key in self.filemanager.info.keys():
                self.filemanager.info[key]['skeleton_npy_indices'] = sk_dict[key]

        if onehot is None:
            self.is_classinformed = False
            len(self.filemanager)
            self._zeros = torch.zeros(len(self.filemanager.clusters))
            self.class2idx = {x: y for x, y in
                              zip(self.filemanager.clusters.keys(), range(len(self.filemanager.clusters)))}
        else:
            self.is_classinformed = True
            self._zeros = torch.zeros(len(onehot)).float()
            self.class2idx = onehot

        self.reader.init_reader(**self.get_reader_kwargs())

        self.info = BaseDict()
        for f in self.filemanager.indexable:
            key = f[0].split('/')[-1].split('.')[0]
            self.info[key] = len(read(f[0], mmap=True)[1])

    def class2onehot(self, cat):
        x = self._zeros.clone()
        x[self.class2idx[cat]] = 1.
        return x

    def sample_idx(self, idx):
        if self.yield_mode == 'yield_module':
            path = self.filemanager[idx][self._rs2idx[self.ad_key]]
        elif self.yield_mode == 'yield_file':
            path = self.filemanager[idx][self._rs2idx[self.ad_key]]
        key = path.split('/')[-1].split('.')[0]
        N = self.info[key]
        kw = {'audio': {'offset': round(randint(0, int(N - 1.2 * self.AUDIO_LENGTH))), 'length': self.AUDIO_LENGTH},
              self.ad_key: {'offset': round(randint(0, int(N - 1.2 * self.AUDIO_LENGTH))), 'length': self.AUDIO_LENGTH}}
        return kw

    def get_reader_kwargs(self):
        return {self.ad_key: AudioReader()}


class AcappellaDataHandler(BaseDataHandler):
    def __init__(self, *,
                 # Audiovisual tasks
                 multitask: bool, handle_empty_stamps: bool, mouth_shape: tuple,
                 # Flags
                 video_enabled: bool, skeleton_enabled: bool,
                 llcp_enabled: bool,
                 crop_mouth: bool,
                 savgol_filter_enabled: bool,
                 single_frame_enabled: bool,
                 single_emb_enabled: bool,
                 # Acappella kwargs
                 dataset_path, audio_exclude, audio_key, video_key, skeleton_key,
                 # Llcp kwargs
                 llcp_key,
                 audio_video_rate, audio_length, n_video_frames, n_skeleton_frames,
                 # Dataset_args
                 debug: bool, yield_mode: str, visualization, **kwargs):
        args = {}
        for arg in inspect.getfullargspec(AcappellaDataHandler.__init__).kwonlyargs:
            if arg != 'self':
                args.update({arg: locals()[arg]})
        super(AcappellaDataHandler, self).__init__(multitask,
                                                   video_enabled, skeleton_enabled,
                                                   dataset_path, audio_exclude, audio_key, video_key, skeleton_key,
                                                   debug, yield_mode, visualization,
                                                   audio_video_rate, audio_length, n_video_frames, n_skeleton_frames,
                                                   **kwargs)
        self.llcp_enabled = llcp_enabled
        self.single_frame_enabled = single_frame_enabled
        self.single_emb_enabled = single_emb_enabled
        self.llcp_key = llcp_key
        self.crop_mouth_enabled = crop_mouth and video_enabled
        self.savgol_enabled = skeleton_enabled and savgol_filter_enabled
        self.mouth_shape = mouth_shape
        self.reader.init_reader(**self.get_reader_kwargs())
        from acappella_info import get_timestamps
        stamps = get_timestamps()
        self.info = BaseDict()
        for key in stamps:
            if key not in audio_exclude:
                self.info[key] = {'stamps': stamps[key]}
        self._check_stamps(**args)

    def _check_stamps(self, handle_empty_stamps, *args, **kwargs):
        """
        This method ensures each sample has the minimum duration required

        """
        def gather_good(stamps):
            return [x for x in stamps if x[1] - x[0] > self.N_VIDEO_FRAMES + self.N_SEC_FRAMES]

        def is_empty(x):
            return not bool(x)

        exclude = []
        for key in self.info:
            stamp = self.info[key]['stamps']
            stamp = gather_good(stamp)

            if is_empty(stamp):
                warn('Sample %s contains no stamps. Exclude it from further initializations.' % key)
                if key not in kwargs['audio_exclude']:
                    exclude.append(key)
            else:
                self.info[key]['stamps'] = stamp

        kwargs['audio_exclude'] += exclude
        kwargs['handle_empty_stamps'] = False
        if not is_empty(exclude):
            self.__init__(*args, **kwargs)

        self._exclude_empty_stamps = exclude

    def load_llcp_embedding(self, trace):
        llcp_embedding = self.getitem(trace, 1, [self.llcp_key])
        llcp_embedding = llcp_embedding[0]
        return llcp_embedding

    def sample_idx(self, idx):
        if self.yield_mode == 'yield_module':
            path = self.filemanager[idx][self._rs2idx[self.vd_key]]
            key = path.split('/')[-1]
        elif self.yield_mode == 'yield_file':
            path = self.filemanager[idx][self._rs2idx[self.vd_key]]
            key = path.split('/')[-1].split('.')[0]

        stamp0, stamp1 = choice(self.info[key]['stamps'])
        stamp = randint(stamp0, stamp1 - self.N_VIDEO_FRAMES - self.N_SEC_FRAMES)

        return self.get_idx_kwargs(stamp)

    def get_reader_kwargs(self):
        """
        This method defines the reader associated to each modality
        """
        out = {self.ad_key: AudioReader()}
        if self.skeleton_enabled:
            out.update({'landmarks': NumpyFrameReader()})
        if self.video_enabled or self.single_frame_enabled:
            out.update({self.vd_key: NumpyFrameReader()})
        if self.llcp_enabled or self.single_emb_enabled:
            out.update({self.llcp_key: NumpyFrameReader()})
        return out
