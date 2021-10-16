import inspect
import json
import os
from copy import copy
from random import choices

from .dataloader import DataManager, pytorch_dataloader

__all__ = ['DataloaderConstructor', ]

lang_ex = {'english': ['Spanish', 'Hindi', 'Others'],
           'spanish': ['English', "Hindi", 'Others'],
           'hindi': ['English', 'Spanish', 'Others'],
           'others': ['English', 'Spanish', 'Hindi']}


def check_keys(dataset):
    for i, rs in enumerate(dataset.filemanager.indexable):
        rs = [x for x in rs if isinstance(x, str)]
        key = rs[dataset.rsr_idx].split('/')[-1].split('.')[0]
        assert all([key in x for x in rs]), f'Autogen Dataset key mismatch {key}, element # {i}'


class DataloaderConstructor:
    def __new__(cls, obj, *args, **kwargs):
        if obj is None:
            obj = super(DataloaderConstructor, cls).__new__(cls)
            obj.init(*args, **kwargs)
        else:
            new_kwargs = obj.__default_kwargs__
            new_kwargs.update(kwargs)
            obj.init(**new_kwargs)
        return obj

    def init(self, *, dataset_paths, debug, n_iterations, batch_size, audio_key, video_key, llcp_key, skeleton_key,
             audio_video_rate, audio_length, n_video_frames, n_skeleton_frames,
             trace_init=None):
        self._key_kwargs = ['audio_key', 'video_key', 'llcp_key', 'skeleton_key']
        self._av_kwargs = ["audio_video_rate", "audio_length", "n_video_frames", "n_skeleton_frames", ]
        args = {}
        self.key_kwargs = {}
        self.av_kwargs = {}
        self.__default_kwargs__ = {}
        for arg in inspect.getfullargspec(DataloaderConstructor.init).kwonlyargs:
            if arg != 'self':
                new_arg = {arg: locals()[arg]}
                self.__default_kwargs__.update(new_arg)
                if arg in self._key_kwargs:
                    self.key_kwargs.update(new_arg)
                elif arg in self._av_kwargs:
                    self.av_kwargs.update(new_arg)

        self.common_kwargs = copy(self.__default_kwargs__)
        self.paths = dataset_paths

        self.mode = None
        self.dataset_list = []
        self.traces = {}
        self.debug = debug

        self.n_samples = n_iterations * batch_size
        self.traces_init = json.load(open(trace_init)) if trace_init is not None else None

    def _build_assertions(self):
        assert len(self.dataset_list) > 0, f'No datasets has been added to the stack'
        assert self.mode is not None, f'Dataset mode should be train,val,test... but None found'

    def set_mode(self, value):
        self.mode = value
        return self

    def add_acappella(self, exclude,
                      multitask, is_enabled, crop_mouth, mouth_shape, savgol_kwargs={'enabled':False},
                      dataset_name='acappella_av', cluster_name='acappella_visual'):
        dataset_acappella = self._acappella(dataset_path=self.paths['acappella'],
                                            exclude=exclude, nsources=1, cluster_name=cluster_name,
                                            dataset_name=dataset_name,
                                            savgol_kwargs=savgol_kwargs,
                                            crop_mouth=crop_mouth, mouth_shape=mouth_shape,
                                            multitask=multitask,
                                            is_enabled=is_enabled
                                            )
        self.main_traces = dataset_acappella['dataset'].precompute_epoch(batch_size=self.common_kwargs["batch_size"],
                                                                         n=self.n_samples,
                                                                         overfit=self.debug["overfit"])
        self.dataset_list.append(dataset_acappella)
        return self

    def build(self, balanced_sampling, mean, std):
        self._build_assertions()
        # Should be extended for more instances
        dataset = DataManager(*self.dataset_list, n_iterations=self.common_kwargs['n_iterations'],
                              batch_size=self.common_kwargs["batch_size"],
                              balanced_sampling=balanced_sampling)
        dataset.statistics = (mean, std)
        if self.traces_init is not None:
            dataset.traces = self.traces_init
            traces_formatted = self.traces_init
        else:
            traces_idx = choices(list(self.traces.keys()), cum_weights=dataset.sampling_probs['audioset'],
                                 k=dataset.n_iterations * self.common_kwargs["batch_size"])
            traces_formatted = {'acappella_visual': [{'acappella_av': x} for x in self.main_traces]}
            traces = [{y: self.traces[y][i]} for y, i in
                      zip(traces_idx, range(dataset.n_iterations * self.common_kwargs["batch_size"]))]
            traces_formatted['audioset'] = traces
        dataloader = pytorch_dataloader(dataset, batch_size=self.common_kwargs["batch_size"])
        return dataloader, traces_formatted

    def add_audioset(self, *categories, n_sources, auto=True):
        if 'train' in self.mode:
            suffix = 'unbalanced_train'
        elif 'test' in self.mode:
            suffix = 'test'
        elif 'val' in self.mode:
            suffix = 'eval'
        else:
            raise NotImplementedError(f'Mode {self.mode} not implemented for audioset')
        cat_path = os.path.join(self.paths['audioset'], suffix, self.common_kwargs['audio_key'])
        if auto:
            categories = os.listdir(cat_path)
        for cat in categories:
            exclude_i = os.listdir(cat_path)
            idx = exclude_i.index(cat)
            exclude_i.pop(idx)
            dataset_audioset_i = self._audioset(dataset_path=os.path.join(self.paths['audioset'], suffix)
                                                , exclude=exclude_i,
                                                debug=self.common_kwargs['debug'],
                                                nsources=n_sources,
                                                cluster_name='audioset', dataset_name=f'audioset_{cat}')

            self.traces[f'audioset_{cat}'] = dataset_audioset_i['dataset'].precompute_epoch(
                batch_size=self.common_kwargs["batch_size"],
                n=self.n_samples,
                overfit=self.common_kwargs['debug']["overfit"])
            self.dataset_list.append(dataset_audioset_i)
        return self

    def _audioset(self, exclude, debug, nsources, cluster_name, dataset_name, dataset_path):
        from ..dataset_helpers import AudiosetDataHandler
        dataset_audioset = AudiosetDataHandler(onehot=None, root=dataset_path,
                                               audio_exclude=exclude, audio_key=self.common_kwargs['audio_key'],
                                               debug=debug, yield_mode='yield_file', visualization=[],
                                               audio_length=self.av_kwargs['audio_length'])
        dataset_audioset = {'name': dataset_name, 'dataset': dataset_audioset,
                            'query_elements': [self.common_kwargs['audio_key']],
                            'cluster': cluster_name, 'nsources': nsources}

        return dataset_audioset

    def _acappella(self, dataset_path, exclude, nsources, cluster_name, dataset_name, is_enabled,
                   crop_mouth, multitask, mouth_shape, savgol_kwargs):

        from ..dataset_helpers import AcappellaDataHandler
        import os
        exclusion_list = []
        exclusion_list.extend(exclude)
        if 'val' in self.mode:
            suffix = 'val_seen'
        elif 'test' in self.mode:
            args = ['test']
            if 'unseen' in self.mode:
                args.append('unseen')
                suffix = 'test_unseen'
            elif 'seen' in self.mode:
                args.append('seen')
                suffix = 'test_seen'
            else:
                raise NotImplementedError(f'Test must be either seen or unseen.')
            if 'english' in self.mode.lower():
                exclusion_list.extend(lang_ex['english'])
            elif 'spanish' in self.mode.lower():
                exclusion_list.extend(lang_ex['spanish'])
            elif 'hindi' in self.mode.lower():
                exclusion_list.extend(lang_ex['hindi'])
            elif 'others' in self.mode.lower():
                exclusion_list.extend(lang_ex['others'])
            if 'male' in self.mode.lower():
                exclusion_list.append('Female')
            elif 'female' in self.mode.lower():
                exclusion_list.append('Male')
            # else:
            #    assert 'mixed' in self.mode.lower(), 'Test must be male, female or mixed'

        elif self.mode == 'train':
            suffix = 'train'
        else:
            raise NotImplementedError

        dataset_acappella = AcappellaDataHandler(multitask=multitask, handle_empty_stamps=True,
                                                 mouth_shape=tuple(mouth_shape),
                                                 crop_mouth=crop_mouth,
                                                 savgol_filter_enabled=savgol_kwargs['enabled'],
                                                 audio_exclude=exclusion_list,
                                                 yield_mode='yield_file', debug=self.debug, visualization=[],
                                                 dataset_path=os.path.join(dataset_path, suffix),
                                                 **self.key_kwargs,
                                                 **self.av_kwargs,
                                                 **is_enabled
                                                 )
        query_elements = [self.key_kwargs['audio_key']]
        if is_enabled["video_enabled"] or is_enabled['single_frame_enabled']:
            query_elements.append(self.key_kwargs['video_key'])
        if is_enabled['llcp_enabled'] or is_enabled['single_emb_enabled']:
            query_elements.append(self.key_kwargs['llcp_key'])
        if is_enabled['skeleton_enabled']:
            query_elements.append(self.key_kwargs['skeleton_key'])
        if savgol_kwargs['enabled']:
            dataset_acappella._savgol_order = savgol_kwargs['order']
            dataset_acappella._savgol_size = savgol_kwargs['size']
        dataset_acappella = {'name': dataset_name, 'dataset': dataset_acappella, 'nsources': nsources,
                             'query_elements': query_elements, 'cluster': cluster_name}
        return dataset_acappella
