import numpy as np
import pandas as pd


class OneHotEncoder:
    """
    Simple one-hot encoder.

    Does not handle unseen categories: will default to the first category.
    Does not invert all-zero arrays: will default to the first category.
    Does not handle NaN data.

    Example:
        >>> oh = OneHotEncoder()
        >>> oh.fit(np.array(['a', 'b', 'c', 'd']))
        >>> oh.transform(np.array(['a', 'c', 'd', 'a']))
        >>> oh.inverse(np.array([[0, 1, 0, 0]]))
    """
    def __init__(self):
        self._categories = None

    @property
    def categories(self) -> np.ndarray:
        if self._categories is None:
            raise ValueError('Encoder not fitted!')
        return self._categories

    @categories.setter
    def categories(self, categories) -> None:
        self._categories = categories

    @property
    def n_categories(self) -> int:
        return len(self.categories)

    def fit(self, categories: np.ndarray) -> None:
        """
        Fit the encoder with the unique elements in categories.

        :param categories: np.ndarray
        :return: None
        """
        self.categories = np.unique(categories)

    def transform(self, samples: np.ndarray) -> np.ndarray:
        """
        Transform samples into their one-hot encoding.

        :param samples: np.ndarray
        :return: encoding
        """
        return self.transform_from_labels(self.transform_to_labels(samples))

    def transform_to_labels(self, samples: np.ndarray) -> np.ndarray:
        """
        Transform samples to labels (numericals).

        :param samples: np.ndarray
        :return: labels
        """
        arr = np.argwhere(self.categories == samples.reshape(-1, 1))
        labels = np.zeros((samples.size,), dtype=int)
        labels[arr[:, 0]] = arr[:, 1]
        return labels.reshape(samples.shape)

    def transform_from_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Transform labels to one-hot encoding.

        :param labels: np.ndarray
        :return: encoding
        """
        return np.eye(self.n_categories)[labels]

    def inverse_from_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Invert labels to original categories.

        :param labels: np.ndarray
        :return: categories
        """
        return self.categories[labels]

    @staticmethod
    def inverse_to_labels(encoded: np.ndarray) -> np.ndarray:
        """
        Invert one-hot encoding to label values

        :param encoded: np.ndarray
        :return: labels
        """
        return np.argmax(encoded, axis=-1)

    def inverse(self, encoded: np.ndarray) -> np.ndarray:
        """
        Invert one-hot encoding to original categories.

        :param encoded: np.ndarray
        :return: categories
        """
        return self.inverse_from_labels(self.inverse_to_labels(encoded))


class NanHotEncoder(OneHotEncoder):
    """
    One-hot encoder that handles NaN values.

    Does handle NaN data, ignores unseen categories (all zero) and inverts all zero rows.
    Only accepts and returns 1-dimensional data (pd.Series) as samples (categories).

    Example:
        >>> nh = NanHotEncoder()
        >>> nh.fit(np.array(['a', 'b', 'c', 'd']))
        >>> nh.transform(pd.Series([np.nan, 'c', 'd', 'a']))
        >>> nh.inverse(np.array([[0, 0, 0, 0], [0, 0, 1, 0]]))
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def _mask_assign(shape: tuple, mask: np.ndarray, values: np.ndarray, init: float=np.nan) -> np.ndarray:
        array = np.full(shape, init)
        array[mask] = values
        return array

    def transform_from_labels(self, labels: np.ndarray) -> np.ndarray:
        nans = np.isnan(labels)
        encoded = super(NanHotEncoder, self).transform_from_labels(labels[~nans].astype(int))
        return self._mask_assign(labels.shape + (self.n_categories,), ~nans, encoded, init=0)

    def inverse_to_lables(self, encoded: np.ndarray) -> np.ndarray:
        nans = np.sum(encoded, axis=-1) == 0
        inverted = super(NanHotEncoder, self).inverse_to_labels(encoded[~nans].astype(int))
        return self._mask_assign(encoded.shape[:-1], ~nans, inverted)

    def transform_to_labels(self, samples: pd.Series) -> np.ndarray:
        mask = samples.isnull() | ~samples.isin(self.categories)
        labels = super(NanHotEncoder, self).transform_to_labels(samples[~mask].values)
        return self._mask_assign(samples.values.shape, ~mask.values, labels)

    def inverse_from_labels(self, labels: np.ndarray) -> pd.Series:
        series = pd.Series(labels.ravel())
        inverted = super(NanHotEncoder, self).inverse_from_labels(series.dropna().values.astype(int))
        series[~series.isnull()] = inverted
        return series

    def transform(self, samples: pd.Series) -> np.ndarray:
        return self.transform_from_labels(self.transform_to_labels(samples))

    def inverse(self, encoded: np.ndarray) -> pd.Series:
        return self.inverse_from_labels(self.inverse_to_labels(encoded))