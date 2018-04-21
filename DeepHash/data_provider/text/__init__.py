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

import h5py
import numpy as np


# Process images of this size. Note that this differs from the original nus-wide
# image size of 224 x 224. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.

# Global constants describing the NUS-WIDE data set.

class Dataset(object):
    def __init__(self, modal, path, train=True):
        self.lines = open(path, 'r').readlines()
        self.n_samples = len(self.lines)
        self.train = train
        if modal == 'txt':
            self.modal = 'txt'
            self._txt = [0] * self.n_samples
            self._label = [0] * self.n_samples
            self._load = [0] * self.n_samples
            self._load_num = 0
            self._status = 0
            self.data = self.txt_data
            self.all_data = self.txt_all_data

    def txt_data(self, index):
        if self._status:
            return (self._txt[index, :], self._label[index, :])
        else:
            ret_txt = []
            ret_label = []
            for i in index:
                try:
                    if self.train:
                        if not self._load[i]:
                            self._txt[i] = h5py.File(self.lines[i].split('\n')[0], 'r')["data"][0]
                            self._label[i] = [int(j) for j in h5py.File(self.lines[i].split('\n')[0], 'r')['label1'][0]]
                            self._load[i] = 1
                            self._load_num += 1
                        ret_txt.append(self._txt[i])
                        ret_label.append(self._label[i])
                    else:
                        # self._label[i] = [int(j) for j in h5py.File(self.lines[i].split('\n')[0], 'r')['label1'][0]]
                        f = h5py.File(self.lines[i].split('\n')[0], 'r')
                        ret_txt.append(f["data"][0])
                        ret_label.append([int(j) for j in f['label1'][0]])
                        f.close()
                except:
                    print('cannot open', self.lines[i].split('\n')[0])

            if self._load_num == self.n_samples:
                self._status = 1
                self._txt = np.reshape(np.asarray(self._txt), (self.n_samples, len(self._txt[0])))
                self._label = np.asarray(self._label)
            return np.reshape(np.asarray(ret_txt), (len(index), -1)), np.asarray(ret_label)

    def txt_all_data(self):
        if self._status:
            return (self._txt, self._label)

    def get_labels(self):
        for i in range(self.n_samples):
            if self._label[i] == 0:
                if self.modal == 'img':
                    self._label[i] = [int(j) for j in self.lines[i].strip().split()[1:]]
                elif self.modal == 'txt':
                    f = h5py.File(self.lines[i].split('\n')[0], 'r')
                    self._label[i] = [int(j) for j in f['label1'][0]]
                    f.close()
        return np.asarray(self._label)


def import_train(txt_tr):
    return (Dataset('txt', txt_tr, train=True))


def import_validation(txt_te, txt_db):
    return (Dataset('txt', txt_te, train=False),
            Dataset('txt', txt_db, train=False))
