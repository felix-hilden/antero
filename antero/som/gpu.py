import numpy as np
import tensorflow as tf

from tqdm import tqdm
from antero.som import _BaseSOM


def _learning_rate(epoch: tf.placeholder, max_epochs: int):
    with tf.name_scope('learning_rate'):
        return tf.exp(-4 * epoch / max_epochs)


def _neighbourhood(r: tf.placeholder, epoch: tf.placeholder, max_epochs: int, size: int):
    with tf.name_scope('neighbourhood'):
        return tf.exp(
            - (2 * r / size) ** 2
            * (max_epochs / (max_epochs - epoch)) ** 3
        )


class SelfOrganisingMap(_BaseSOM):
    @property
    def initialiser(self):
        if self._weights is None:
            if self._initialiser == 'uniform':
                return tf.random_uniform_initializer
            elif self._initialiser == 'normal':
                return tf.random_normal_initializer
        else:
            return tf.convert_to_tensor(self._weights)

    def train(self, x: np.ndarray, epochs: int, batch_size: int = 1,
              shuffle: bool = False, verbose: bool = False) -> None:
        """
        Create training graph and train SOM.

        :param x: training data
        :param epochs: number of epochs to train
        :param batch_size: number of training examples per step
        :param shuffle: shuffle data in training
        :param verbose: display a progress bar
        :return: None
        """
        graph = tf.Graph()
        sess = tf.Session(graph=graph)

        x = x.astype(np.float64)

        batches = x.shape[0] // batch_size
        if x.shape[0] % batch_size != 0:
            raise ValueError('Bad batch_size, last batch would be incomplete!')

        # Construct graph
        with graph.as_default():
            indices = tf.convert_to_tensor(np.expand_dims(
                np.indices(self.shape, dtype=np.float64), axis=-1
            ))
            weights = tf.get_variable(
                'weights', (*self.shape, 1, self.features), initializer=self.initialiser, dtype=tf.float64
            )
            curr_epoch = tf.placeholder(dtype=tf.int64, shape=())

            with tf.name_scope('data'):
                data = tf.data.Dataset.from_tensor_slices(x)
                if shuffle:
                    data = data.shuffle(buffer_size=10000)
                data = data.repeat(epochs)
                data = data.batch(batch_size, drop_remainder=True)
                data = data.make_one_shot_iterator().get_next()

            def train_loop(w):
                with tf.name_scope('winner'):
                    diff = w - data
                    dist = tf.reduce_sum(diff ** 2, axis=-1, keepdims=True)
                    w_ix = tf.argmin(tf.reshape(dist, (self.n_nodes, data.shape[0])), axis=0)
                    winner_op = tf.convert_to_tensor(tf.unravel_index(w_ix, self.shape))

                with tf.name_scope('update'):
                    idx_diff = indices - tf.reshape(tf.cast(
                        winner_op, dtype=tf.float64
                    ), shape=self._neighbour_shape)
                    idx_dist = tf.norm(idx_diff, axis=0)

                    l_rate = _learning_rate(curr_epoch, self.max_epochs)
                    n_hood = _neighbourhood(
                        idx_dist, curr_epoch, self.max_epochs, max(self.shape)
                    )

                    update = diff * l_rate * tf.expand_dims(n_hood, axis=-1)
                    return w - self._initial_lr * tf.reduce_mean(update, axis=-2, keepdims=True)

            n_weights = tf.while_loop(lambda _: True, train_loop, (weights,), maximum_iterations=batches)
            update_op = weights.assign(n_weights)

            init = tf.global_variables_initializer()

        sess.run(init)

        for i in tqdm(range(epochs)) if verbose else range(epochs):
            sess.run(update_op, feed_dict={
                curr_epoch: self.epochs + i
            })

        self._weights = sess.run(weights)
        self._epochs += epochs

