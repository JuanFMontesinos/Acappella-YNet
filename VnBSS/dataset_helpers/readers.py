import os
import numpy as np

__all__ = ['SKReader', 'SPReader', 'class_reader', 'ID_reader', 'NumpyFrameReader']


class SKReader(object):
    def __init__(self, path, shape, filemanager, N, dtype):
        self.npy = np.memmap(path, dtype=dtype, mode='r', shape=shape)
        self.filemanager = filemanager
        self.N = N

    def __call__(self, path, offset, length):
        return self.npy[offset:offset + length]


class SPReader(object):
    def __init__(self, filemanager):
        self.filemanager = filemanager

    def __call__(self, path, offset, length):
        key = ID_reader(path)
        path = os.path.join(path, key + '.npy')
        npy = np.load(path, mmap_mode='r')
        return npy[:, offset:length + offset, :]


class NumpyFrameReader:
    def __init__(self, source=None):
        if source is not None:
            self.tree = {}
            for path in source:
                key = path.split('/')[-1].split('.')[0]
                self.tree[key] = np.load(path, mmap_mode='r')
            self.from_memory = True
        else:
            self.tree = None
            self.from_memory = False

    def __call__(self, path: str, offset: int, length: int):
        if self.from_memory:
            key = path.split('/')[-1].split('.')[0]
            return self.tree[key][offset:offset + length]
        else:
            return np.load(path, mmap_mode='r')[offset:offset + length]

    def __getitem__(self, item):
        return self.tree[item]


def class_reader(path):
    key = path.split('/')[-2]
    return key


def ID_reader(path):
    key = path.split('/')[-1]
    return key
