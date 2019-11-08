import numpy as np

from tqdm import tqdm
from sklearn.utils import shuffle as shuffle_data
from antero.som import _BaseSOM, load


def _make_learning_rate(max_epochs: int, lr_decay: float = 1) -> callable:
    c = -4 / max_epochs * lr_decay

    def _learning_rate(epoch: int) -> float:
        return np.exp(c * epoch)
    return _learning_rate


def _make_neighbourhood(
        shape: tuple,
        max_epochs: int,
        nbh_width: float = 1,
        nbh_decay: float = 1
) -> callable:
    c = 4 / max(shape) / nbh_width

    def _neighbourhood(r: np.ndarray, epoch: int):
        return np.exp(
            - (c * r / (1 - epoch / max_epochs) ** nbh_decay) ** 2
        )
    return _neighbourhood


class SelfOrganisingMap(_BaseSOM):
    def _init_weights(self):
        if self._initialiser == 'uniform':
            self._weights = np.random.rand(*self.shape, 1, self.features)
        elif self._initialiser == 'normal':
            self._weights = np.random.normal(size=(*self.shape, 1, self.features))

    def _init_members(self):
        self.neighbourhood = _make_neighbourhood(
            self.shape,
            self.max_epochs,
            self._neighbourhood_width,
            self._neighbourhood_decay
        )
        self.learning_rate = _make_learning_rate(
            self.max_epochs,
            self._learning_rate_decay
        )
        self.indices = np.expand_dims(np.indices(self.shape), axis=-1)

    def _on_first_train(self) -> None:
        """
        Initialise functions, indices and weights.

        :return: None
        """
        self._init_members()
        self._init_weights()

    def _idx_distances(self, points: np.ndarray) -> np.ndarray:
        """
        Index distance to other points on a map.

        :param points: map coordinates
        :return: distance matrix with map shape
        """
        diff = self.indices - points.reshape(self._neighbour_shape)
        return np.linalg.norm(diff, axis=0)

    def train(
            self,
            x: np.ndarray,
            epochs: int,
            batch_size: int = 1,
            shuffle: bool = False,
            verbose: bool = False
    ) -> None:
        """
        Train SOM with batches. Count epochs starting from first train call.

        :param x: training data
        :param epochs: number of epochs to train
        :param batch_size: number of training examples per step
        :param shuffle: shuffle data each epoch
        :param verbose: display a progress bar
        :return: None
        """
        if self._weights is None:
            self._on_first_train()

        if x.shape[0] % batch_size != 0:
            raise ValueError('Bad batch_size, last batch would be incomplete!')

        for i in tqdm(range(epochs)) if verbose else range(epochs):
            epoch = self.epochs + i
            rate = self.learning_rate(epoch)
            if shuffle:
                x = shuffle_data(x)

            for batch in range(x.shape[0] // batch_size):
                data = x[batch*batch_size:(batch+1)*batch_size]
                diff = self.weights - data
                dist = np.sum(diff ** 2, axis=-1, keepdims=True)
                winner = np.array(np.unravel_index(
                    np.argmin(dist.reshape((-1, data.shape[0])), axis=0), self.shape
                ))
                factor = self.neighbourhood(self._idx_distances(winner), epoch)
                update = diff * rate * np.expand_dims(factor, axis=-1)
                self._weights -= self._initial_lr * np.mean(update, axis=-2, keepdims=True)

        # Record elapsed epochs
        self._epochs += epochs

    @classmethod
    def load(cls, path) -> 'SelfOrganisingMap':
        som = load(cls, path)
        som._init_members()
        return som
