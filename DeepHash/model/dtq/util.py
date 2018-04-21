import numpy as np
import math
from distance.npversion import distance

class Dataset(object):
    def __init__(self, dataset, output_dim, code_dim):
        self._dataset = dataset
        self.n_samples = dataset.n_samples
        self._train = dataset.train
        self._output = np.zeros((self.n_samples, output_dim), dtype=np.float32)
        self._codes = np.zeros((self.n_samples, code_dim), dtype=np.float32)
        self._triplets = np.array([])
        self._trip_index_in_epoch = 0
        self._index_in_epoch = 0
        self._epochs_complete = 0
        self._perm = np.arange(self.n_samples)
        np.random.shuffle(self._perm)
        return

    def update_triplets(self, margin, n_part=10, dist_type='euclidean2', select_strategy='margin'):
        """
        :param select_strategy: hard, all, margin
        :param dist_type: distance type, e.g. euclidean2, cosine
        :param margin: triplet margin parameter
        :n_part: number of part to split data
        """
        n_samples = self.n_samples
        np.random.shuffle(self._perm)
        embedding = self._output[self._perm[:n_samples]]
        labels = self._dataset.get_labels()[self._perm[:n_samples]]
        n_samples_per_part = int(math.ceil(n_samples / n_part))
        triplets = []
        for i in range(n_part):
            start = n_samples_per_part * i
            end = min(n_samples_per_part * (i+1), n_samples)
            dist = distance(embedding[start:end], pair=True, dist_type=dist_type)
            for idx_anchor in range(0, end - start):
                label_anchor = np.copy(labels[idx_anchor+start, :])
                label_anchor[label_anchor==0] = -1
                all_pos = np.where(np.any(labels[start:end] == label_anchor, axis=1))[0]
                all_neg = np.array(list(set(range(end-start)) - set(all_pos)))

                if select_strategy == 'hard':
                    idx_pos = all_pos[np.argmax(dist[idx_anchor, all_pos])]
                    if idx_pos == idx_anchor:
                        continue
                    idx_neg = all_neg[np.argmin(dist[idx_anchor, all_neg])]
                    triplets.append((idx_anchor + start, idx_pos + start, idx_neg + start))
                    continue

                for idx_pos in all_pos:
                    if idx_pos == idx_anchor:
                        continue

                    if select_strategy == 'all':
                        selected_neg = all_neg
                    elif select_strategy == 'margin':
                        selected_neg = all_neg[np.where(dist[idx_anchor, all_neg] - dist[idx_anchor, idx_pos] < margin)[0]]

                    if selected_neg.shape[0] > 0:
                        idx_neg = np.random.choice(selected_neg)
                        triplets.append((idx_anchor + start, idx_pos + start, idx_neg + start))
        self._triplets = np.array(triplets)
        np.random.shuffle(self._triplets)

        # assert
        anchor = labels[self._triplets[:, 0]]
        mapper = lambda anchor, other: np.any(anchor * (anchor == other), -1)
        assert(np.all(mapper(anchor, labels[self._triplets[:, 1]])))
        assert(np.all(np.invert(anchor, labels[self._triplets[:, 2]])))
        return

    def next_batch_triplet(self, batch_size):
        """
        Args:
          batch_size
        Returns:
          data, label, codes
        """
        start = self._trip_index_in_epoch
        self._trip_index_in_epoch += batch_size
        if self._trip_index_in_epoch > self.triplets.shape[0]:
            start = 0
            self._trip_index_in_epoch = batch_size
        end = self._trip_index_in_epoch

        # stack index of anchors, positive, negetive to one array
        arr = self.triplets[start:end]
        idx = self._perm[np.concatenate([arr[:, 0], arr[:, 1], arr[:, 2]], axis=0)]
        data, label = self._dataset.data(idx)

        return data, label, self._codes[idx]

    def next_batch(self, batch_size):
        """
        Args:
          batch_size
        Returns:
          [batch_size, (n_inputs)]: next batch images, by stacking anchor, positive, negetive
          [batch_size, n_class]: next batch labels
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.n_samples:
            if self._train:
                self._epochs_complete += 1
                start = 0
                self._index_in_epoch = batch_size
            else:
                # Validation stage only process once
                start = self.n_samples - batch_size
                self._index_in_epoch = self.n_samples
        end = self._index_in_epoch

        data, label = self._dataset.data(self._perm[start:end])
        return (data, label, self._codes[self._perm[start: end], :])

    def next_batch_output_codes(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # Another epoch finish
        if self._index_in_epoch > self.n_samples:
            if self._train:
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
            else:
                # Validation stage only process once
                start = self.n_samples - batch_size
                self._index_in_epoch = self.n_samples
        end = self._index_in_epoch

        return (self._output[self._perm[start: end], :],
                self._codes[self._perm[start: end], :])

    def feed_batch_output(self, batch_size, output):
        start = self._index_in_epoch - batch_size
        end = self._index_in_epoch
        self._output[self._perm[start:end], :] = output
        return

    def feed_batch_triplet_output(self, batch_size, triplet_output):
        anchor, pos, neg = np.split(triplet_output, 3, axis=0)
        start = self._trip_index_in_epoch - batch_size
        end = self._trip_index_in_epoch
        idx = self._perm[self._triplets[start:end, :]]
        self._output[idx[:, 0]] = anchor
        self._output[idx[:, 1]] = pos
        self._output[idx[:, 2]] = neg
        return

    def feed_batch_codes(self, batch_size, codes):
        """
        Args:
          batch_size
          [batch_size, n_output]
        """
        start = self._index_in_epoch - batch_size
        end = self._index_in_epoch
        self._codes[self._perm[start:end], :] = codes
        return

    @property
    def output(self):
        return self._output

    @property
    def codes(self):
        return self._codes

    @property
    def triplets(self):
        return self._triplets

    @property
    def label(self):
        return self._dataset.get_labels()

    def finish_epoch(self):
        self._index_in_epoch = 0
