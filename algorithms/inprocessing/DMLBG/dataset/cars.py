# from .base import *
import scipy.io

# class Cars(BaseDataset):
#     def __init__(self, root, mode, transform = None):
#         self.root = root + '/car'
#         self.mode = mode
#         self.transform = transform
#         if self.mode == 'train':
#             self.classes = range(0,98)
#         elif self.mode == 'eval':
#             self.classes = range(98,196)
                
#         BaseDataset.__init__(self, self.root, self.mode, self.transform)
#         annos_fn = 'cars_annos.mat'
#         cars = scipy.io.loadmat(os.path.join(self.root, annos_fn))
#         ys = [int(a[5][0] - 1) for a in cars['annotations'][0]]
#         im_paths = [a[0][0] for a in cars['annotations'][0]]
#         index = 0
#         for im_path, y in zip(im_paths, ys):
#             if y in self.classes: # choose only specified classes
#                 self.im_paths.append(os.path.join(self.root, im_path))
#                 self.ys.append(y)
#                 self.I += [index]
#                 index += 1

import os
import scipy.io
import numpy as np

from .base import BaseDataset   # ✅ 정확히 이렇게 import해야 함

def _mat_scalar(x):
    # numpy array로 감싸진 값들을 재귀적으로 벗김
    import numpy as np
    while isinstance(x, np.ndarray):
        x = x[0]
    return x

class Cars(BaseDataset):
    def __init__(self, root, mode, transform=None):
        self.root = root + '/car'
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.classes = range(0, 98)
        elif self.mode == 'eval':
            self.classes = range(98, 196)

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        annos_fn = 'cars_annos.mat'
        cars = scipy.io.loadmat(os.path.join(self.root, annos_fn))
        ann = cars['annotations'][0]

        ys = [int(_mat_scalar(a[5]) - 1) for a in ann]
        im_paths = [_mat_scalar(a[0]) for a in ann]

        index = 0
        for im_path, y in zip(im_paths, ys):
            if y in self.classes:  # choose only specified classes
                self.im_paths.append(os.path.join(self.root, im_path))
                self.ys.append(y)
                self.I += [index]
                index += 1
