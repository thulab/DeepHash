import numpy as np

class Dataset(object):
    def __init__(self, dataset, output_dim, code_dim):
        print ("Initializing Dataset")
        self._dataset = dataset
        self.n_samples = dataset.n_samples
        self._train = dataset.train
        self._output = np.zeros((self.n_samples, output_dim), dtype=np.float32)
        self._codes = np.zeros((self.n_samples, code_dim), dtype=np.float32)

        self._perm = np.arange(self.n_samples)
        np.random.shuffle(self._perm)
        self._index_in_epoch = 0
        self._epochs_complete = 0
        print ("Dataset already")
        return

    def next_batch(self, batch_size):
        """
        Args:
          batch_size
        Returns:
          [batch_size, (n_inputs)]: next batch images
          [batch_size, n_class]: next batch labels
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # Another epoch finish
        if self._index_in_epoch > self.n_samples:
            if self._train:
                # Training stage need repeating get batch
                self._epochs_complete += 1
                # Shuffle the data
                np.random.shuffle(self._perm)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
            else:
                # Validation stage only process once
                start = self.n_samples - batch_size
                self._index_in_epoch = self.n_samples
        end = self._index_in_epoch

        data, label = self._dataset.data(self._perm[start:end])
        return (data, label, self.codes[self._perm[start: end], :])

    def next_batch_output_codes(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # Another epoch finish
        if self._index_in_epoch > self.n_samples:
            if self._train:
                # Shuffle the data
                np.random.shuffle(self._perm)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
            else:
                # Validation stage only process once
                start = self.n_samples - batch_size
                self._index_in_epoch = self.n_samples
        end = self._index_in_epoch

        return (self.output[self._perm[start: end], :],
                self.codes[self._perm[start: end], :])

    def feed_batch_output(self, batch_size, output):
        start = self._index_in_epoch - batch_size
        end = self._index_in_epoch
        self.output[self._perm[start:end], :] = output
        return

    def feed_batch_codes(self, batch_size, codes):
        """
        Args:
          batch_size
          [batch_size, n_output]
        """
        start = self._index_in_epoch - batch_size
        end = self._index_in_epoch
        self.codes[self._perm[start:end], :] = codes
        return

    @property
    def output(self):
        return self._output

    @property
    def codes(self):
        return self._codes

    @property
    def label(self):
        return self._dataset.get_labels()

    def finish_epoch(self):
        self._index_in_epoch = 0
        np.random.shuffle(self._perm)

