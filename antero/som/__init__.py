import pickle
from pathlib import Path

import numpy as np


class _BaseSOM:
    def __init__(self, shape: tuple, features: int, *_,
                 max_epochs: int = None, init: str = 'uniform', learning_rate: float = 0.1):
        """
        Self-organising map.

        :param shape: map dimensions
        :param features: number of input features
        :param _: used to force calling with keyword arguments below
        :param max_epochs: used to scale the neighbourhood and learning rate functions
        :param init: method of weight initialisation. 'uniform' for drawing from an uniform
            distribution between 0..1, 'normal' for drawing from X~N(0,1)
        :param learning_rate: initial learning rate multiplier
        """
        self._weights = None

        self._shape = shape
        self._features = features
        self._neighbour_shape = (len(shape),) + tuple(1 for _ in shape) + (-1,)

        self._epochs = 0
        self._max_epochs = max_epochs
        self._initial_lr = learning_rate

        if init not in ['uniform', 'normal']:
            raise AssertionError('Unknown weights initialiser type "%s"!' % init)

        self._initialiser = init

    @property
    def weights(self) -> np.ndarray:
        if self._weights is None:
            raise ValueError('Map not fitted!')
        return self._weights

    @property
    def shape(self) -> tuple: return self._shape

    @property
    def n_nodes(self) -> int: return int(np.prod(self.shape))

    @property
    def features(self) -> int: return self._features

    @property
    def epochs(self) -> int: return self._epochs

    @property
    def max_epochs(self) -> int: return self._max_epochs

    def project(self, data: np.ndarray) -> np.ndarray:
        """
        Project data onto the map.

        :param data: samples
        :return: node indices
        """
        diff = self.weights - data
        dist = np.sum(diff ** 2, axis=-1, keepdims=True)
        return np.array(np.unravel_index(
            np.argmin(dist.reshape((-1, data.shape[0])), axis=0), self.shape
        ))

    def _gather_indices(self, indices: np.ndarray) -> np.ndarray:
        """
        Count occurrence of indices.

        :param indices: indices to array
        :return: array with index counts
        """
        heat = np.zeros(self.shape)
        np.add.at(heat, tuple(indices), 1)
        return heat

    def heatmap(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Count occurrence of indices per label.

        :param x: data samples
        :param y: true numerical labels
        :return: array with index counts
        """
        indices = self.project(x)

        if y is None:
            return self._gather_indices(indices)
        else:
            heats = np.zeros((y.max() + 1,) + self.shape)
            for i in range(y.max() + 1):
                heats[i] = self._gather_indices(indices[..., np.where(y == i)])
            return heats

    def labelmap(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Label each node with the most frequent class.

        :param x: data samples
        :param y: true numerical labels
        :return: label map
        """
        heats = self.heatmap(x, y)
        winner = np.argmax(heats, axis=0).astype(np.float)
        empty = np.where(heats.sum(axis=0) == 0)

        winner[empty] = np.nan
        return winner

    def save(self, path: Path) -> None:
        """
        Save self as pickled object.

        :param path: file name
        :return: None
        """
        d = {
            'w': self._weights,
            'e': self._epochs,
            'm': self._max_epochs,
            'r': self._initial_lr,
            'i': self._initialiser
        }
        with open(str(path), 'wb') as f:
            pickle.dump(d, f)

    @classmethod
    def load(cls, path: Path) -> '_BaseSOM':
        with open(str(path), 'rb') as f:
            d = pickle.load(f)

        som = cls(d['w'].shape[:-2], d['w'].shape[-1], max_epochs=d['m'], init=d['i'], learning_rate=d['r'])
        som._weights = d['w']
        som._epochs = d['e']
        return som

    def __repr__(self):
        return ''.join(str(s) for s in [
            str(type(self)),
            '\nShape: ', self.shape,
            '\nFeatures: ', self.features,
            '\nEpochs: ', self.epochs, '/', self.max_epochs,
            '\nInitial learning rate: ', self._initial_lr,
            '\nInitialised with ', self._initialiser, ' distribution.',
        ])
