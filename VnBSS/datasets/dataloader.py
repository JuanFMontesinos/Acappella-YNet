import collections
import numpy as np
import re
import torch
from flerken.audio import np_int2float
from flerken.datasets.pytorch_pipes import FlerkenDataset
from random import choice
from scipy.signal import savgol_filter
from torch._six import string_classes
from torch.utils.data import DataLoader

from ..utils import normalize_max

"""
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def default_convert(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, collections.abc.Mapping):
        return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        return [default_convert(d) for d in data]
    else:
        return data


#
#
def default_collate(batch):
    """
    Puts each data field into a tensor with outer dimension batch size.
    This is a direct copy from pytorch code. The pytorch source code changes from version 1.8 onwards.
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    else:
        return batch
    raise TypeError(default_collate_err_msg_format.format(elem_type))


"""
    Here it ends pytorch's licensing
"""


class TraceWrapper:
    def __init__(self, trace):
        self.trace = trace


def pytorch_dataloader(dataset, batch_size):
    # NUM_WORKERS = cpu_count()
    NUM_WORKERS = 0
    return DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=NUM_WORKERS,
                      collate_fn=default_collate)


class DataManager(FlerkenDataset):
    def custom_ops(self, data: dict, traces: dict, datasets_used: dict, idx: int):
        rsc2idx = {x: y for x, y in
                   zip(self.acappella_av['query_elements'], range(len(self.acappella_av['query_elements'])))}
        audio = normalize_max(np_int2float(data['acappella_visual'][rsc2idx['audio']][0]))
        audio_acmt = normalize_max(np_int2float(data['audioset'][0][0]))
        if self.loudness_coef is not None:
            if isinstance(self.loudness_coef, float):
                loudness_coef = self.loudness_coef
            elif isinstance(self.loudness_coef, str):
                if self.loudness_coef == 'random':
                    loudness_coef = choice([0.25, 0.5, 0.75, 1.])
                else:
                    raise NotImplementedError

            rms_audio = self.rms(audio) or 1
            rms_audio_acmt = self.rms(audio_acmt) or 1
            audio = loudness_coef * audio / rms_audio
            audio_acmt = audio_acmt / rms_audio_acmt
            max_val = max(np.abs(audio).max(), np.abs(audio_acmt).max())
            audio /= max_val
            audio_acmt /= max_val
        output = {'audio': torch.from_numpy(audio), 'audio_acmt': torch.from_numpy(audio_acmt)}

        read_video = self.acappella.video_enabled or self.acappella.single_frame_enabled
        read_video_only = self.acappella.video_enabled and not self.acappella.single_frame_enabled
        read_frame_only = not self.acappella.video_enabled and self.acappella.single_frame_enabled
        read_both = self.acappella.video_enabled and self.acappella.single_frame_enabled
        if read_video:
            video = data['acappella_visual'][rsc2idx['frames']][0]
            if read_frame_only:
                video = video[[20, 50, 80]]
            video = self.normalize_video(video)
            video = torch.from_numpy(video)
            video = video.permute(0, 3, 1, 2)
            if read_both:
                frames = video[[20, 50, 80]]
            elif read_frame_only:
                frames = video
            if self.acappella.crop_mouth_enabled and self.acappella.video_enabled:
                video = video[..., 60:100, 20:80]
                video = torch.nn.functional.upsample_bilinear(video, size=self.acappella.mouth_shape)

        if self.acappella.video_enabled:
            output['video'] = video.permute(1, 0, 2, 3).float().contiguous()
        if self.acappella.single_frame_enabled:
            output['single_frame'] = frames.permute(1, 0, 2, 3).float().contiguous()
        if self.acappella.skeleton_enabled:
            savgol = data['acappella_visual'][rsc2idx['landmarks']][0].copy().astype(np.float32)
            if self.acappella.savgol_enabled:
                savgol = savgol_filter(savgol,
                                       window_length=self.acappella._savgol_size,
                                       polyorder=self.acappella._savgol_order,
                                       axis=0)
            output['landmarks'] = torch.from_numpy(savgol).permute(2, 0, 1).unsqueeze(
                -1)
        if self.acappella.llcp_enabled:
            output['llcp_embedding'] = torch.from_numpy(
                data['acappella_visual'][rsc2idx['llcp_embed']][0].copy().astype(np.float32))
        if self.acappella.single_emb_enabled:
            output['single_frame'] = torch.from_numpy(
                data['acappella_visual'][rsc2idx['llcp_embed']][0][[20, 50, 80]].copy().astype(np.float32))
        return output, TraceWrapper(traces), datasets_used, idx

    def rms(self, x):
        return np.sqrt(np.mean(x ** 2))

    @property
    def statistics(self):
        return self._mean, self._std

    @statistics.setter
    def statistics(self, val):
        self._mean = np.array(val[0])
        self._std = np.array(val[1])

    def normalize_video(self, array):
        return (array / 255. - self._mean) / self._std

    def undo_normalization(self, array):
        return array * self._std + self._mean

    @property
    def acappella(self):
        return self.acappella_av['dataset']

    @property
    def loudness_coef(self):
        return self._k

    @loudness_coef.setter
    def loudness_coef(self, val):
        self._k = val
