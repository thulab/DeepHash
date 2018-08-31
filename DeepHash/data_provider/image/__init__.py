# =============================================================================
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the NUS-WIDE binary file format."""

import os
import cv2
import numpy as np


# Process images of this size. Note that this differs from the original nus-wide
# image size of 224 x 224. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.

# Global constants describing the NUS-WIDE data set.


class Dataset(object):
    def __init__(self, modal, data_root, path, train=True):
        self.lines = open(path, 'r').readlines()
        self.data_root = data_root
        self.n_samples = len(self.lines)
        self.train = train
        assert modal == 'img'
        self.modal = 'img'
        self._img = [0] * self.n_samples
        self._label = [0] * self.n_samples
        self._load = [0] * self.n_samples
        self._load_num = 0
        self._status = 0
        self.data = self.img_data
        self.all_data = self.img_all_data

    def get_img(self, i):
        path = os.path.join(self.data_root, self.lines[i].strip().split()[0])
        return cv2.resize(cv2.imread(path), (256, 256))

    def get_label(self, i):
        return [int(j) for j in self.lines[i].strip().split()[1:]]

    def img_data(self, indexes):
        if self._status:
            return (self._img[indexes, :], self._label[indexes, :])
        else:
            ret_img = []
            ret_label = []
            for i in indexes:
                try:
                    if self.train:
                        if not self._load[i]:
                            self._img[i] = self.get_img(i)
                            self._label[i] = self.get_label(i)
                            self._load[i] = 1
                            self._load_num += 1
                        ret_img.append(self._img[i])
                        ret_label.append(self._label[i])
                    else:
                        self._label[i] = self.get_label(i)
                        ret_img.append(self.get_img(i))
                        ret_label.append(self._label[i])
                except Exception as e:
                    print('cannot open {}, exception: {}'.format(self.lines[i].strip(), e))

            if self._load_num == self.n_samples:
                self._status = 1
                self._img = np.asarray(self._img)
                self._label = np.asarray(self._label)
            return (np.asarray(ret_img), np.asarray(ret_label))

    def img_all_data(self):
        if self._status:
            return (self._img, self._label)

    def get_labels(self):
        for i in range(self.n_samples):
            if self._label[i] is not list:
                self._label[i] = [int(j)
                                  for j in self.lines[i].strip().split()[1:]]
        return np.asarray(self._label)


def import_train(data_root, img_tr):
    '''
    return (img_tr, txt_tr)
    '''
    return (Dataset('img', data_root, img_tr, train=True))


def import_validation(data_root, img_te, img_db):
    '''
    return (img_te, txt_te, img_db, txt_db)
    '''
    return (Dataset('img', data_root, img_te, train=False),
            Dataset('img', data_root, img_db, train=False))
