import numpy as np

from tqdm import tqdm
from antero.som import _BaseSOM


class SelfOrganisingMap(_BaseSOM):
    def _on_first_train(self):
        self.neighbourhood_f = lambda r, t: np.exp(
            - (2 * r / max(self.shape)) ** 2
            * (self.max_epochs / (self.max_epochs - t)) ** 3
        )

        self.learning_rate = lambda t: np.exp(-4 * t / self.max_epochs)

        self.indices = np.expand_dims(np.indices(self.shape), axis=-1)

        if self._initialiser == 'uniform':
            self._weights = np.random.rand(*self.shape, 1, self.features)
        elif self._initialiser == 'normal':
            self._weights = np.random.normal(size=(*self.shape, 1, self.features))

    def _idx_distances(self, points: np.ndarray) -> np.ndarray:
        """
        Index distance to other points on a map.

        :param points: map coordinates
        :return: distance matrix with map shape
        """
        diff = self.indices - points.reshape(self._neighbour_shape)
        return np.linalg.norm(diff, axis=0)

    def train(self, x: np.ndarray, epochs: int, batch_size: int = 1) -> None:
        """
        Train SOM with batches. Count epochs starting from first train call.

        :param x: training data
        :param epochs: number of epochs to train
        :param batch_size: number of training examples per step
        :return: None
        """
        if self._weights is None:
            self._on_first_train()

        if x.shape[0] % batch_size != 0:
            raise ValueError('Bad batch_size, last batch would be incomplete!')

        for i in tqdm(range(epochs)):
            for batch in range(x.shape[0] // batch_size):
                data = x[batch*batch_size:(batch+1)*batch_size]
                epoch = self.epochs + i
                diff = self.weights - data
                dist = np.sum(diff ** 2, axis=-1, keepdims=True)
                winner = np.array(np.unravel_index(
                    np.argmin(dist.reshape((-1, data.shape[0])), axis=0), self.shape
                ))
                factor = self.neighbourhood_f(self._idx_distances(winner), epoch)
                update = diff * self.learning_rate(epoch) * np.expand_dims(factor, axis=-1)
                self._weights -= self._initial_lr * np.sum(update, axis=-2, keepdims=True)

        # Record elapsed epochs
        self._epochs += epochs
