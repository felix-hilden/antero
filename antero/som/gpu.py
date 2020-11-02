import numpy as np
import tensorflow as tf

from tqdm import tqdm
from antero.som import _BaseSOM


def _learning_rate(
        epoch,
        max_epochs: int,
        lr_decay: float = 1
):
    return tf.cast(tf.exp(-4 / max_epochs * lr_decay * epoch), tf.float64)


def _neighbourhood(
        shape: tuple,
        r,
        epoch,
        max_epochs: int,
        nbh_width: float = 1,
        nbh_decay: float = 1
):
    c = 4 / max(shape) / nbh_width
    return tf.exp(
        - (c * r / (1 - epoch / max_epochs) ** nbh_decay) ** 2
    )


class SelfOrganisingMap(_BaseSOM):
    def initialiser(self, shape):
        if self._weights is None:
            if self._initialiser == 'uniform':
                return tf.random_uniform_initializer()(shape=shape)
            elif self._initialiser == 'normal':
                return tf.random_normal_initializer()(shape=shape)
        else:
            return tf.convert_to_tensor(self._weights)

    def train(
            self,
            x: np.ndarray,
            epochs: int,
            batch_size: int = 1,
            shuffle: bool = False,
            verbose: bool = False
    ) -> None:
        """
        Train SOM.

        Note that this method is unoptimised, so it is slow compared to CPU.

        :param x: training data
        :param epochs: number of epochs to train
        :param batch_size: number of training examples per step
        :param shuffle: shuffle data in training
        :param verbose: display a progress bar
        :return: None
        """
        x = x.astype(np.float64)

        batches = x.shape[0] // batch_size
        if x.shape[0] % batch_size != 0:
            raise ValueError('Bad batch_size, last batch would be incomplete!')

        indices = tf.convert_to_tensor(np.expand_dims(
            np.indices(self.shape, dtype=np.float64), axis=-1
        ), dtype=tf.float64)
        weights = tf.cast(tf.Variable(
            self.initialiser(shape=(*self.shape, 1, self.features))
        ), tf.float64)

        data = tf.data.Dataset.from_tensor_slices(x)
        if shuffle:
            data = data.shuffle(buffer_size=x.shape[0])
        data = data.repeat(epochs)
        data = data.batch(batch_size, drop_remainder=True)
        data_it = iter(data)

        for i in tqdm(range(epochs)) if verbose else range(epochs):
            curr_epoch = self.epochs + i
            for _ in range(batches):
                batch = next(data_it)
                diff = weights - batch
                dist = tf.reduce_sum(diff ** 2, axis=-1, keepdims=True)
                winner_ix = tf.argmin(tf.reshape(dist, (self.n_nodes, batch.shape[0])), axis=0)
                winner = tf.unravel_index(winner_ix, self.shape)
                winner = tf.cast(tf.reshape(winner, shape=self._neighbour_shape), tf.float64)
                idx_dist = tf.norm(indices - winner, axis=0)

                l_rate = _learning_rate(
                    curr_epoch,
                    self.max_epochs,
                    self._learning_rate_decay
                )
                n_hood = _neighbourhood(
                    self.shape,
                    idx_dist,
                    curr_epoch,
                    self.max_epochs,
                    self._neighbourhood_width,
                    self._neighbourhood_decay
                )
                update = diff * l_rate * tf.expand_dims(n_hood, axis=-1)
                mean_update = tf.reduce_mean(update, axis=-2, keepdims=True)
                weights = weights - self._initial_lr * mean_update

        self._weights = weights.numpy()
        self._epochs += epochs

