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
    def weights(self):
        if self._weights is None:
            raise ValueError('Map not fitted!')
        return self._weights

    @property
    def shape(self): return self._shape

    @property
    def n_nodes(self): return int(np.prod(self.shape))

    @property
    def features(self): return self._features

    @property
    def epochs(self): return self._epochs

    @property
    def max_epochs(self): return self._max_epochs

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
