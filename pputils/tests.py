import unittest
from pputils.categorical import OneHotEncoder, NanHotEncoder, CatHotEncoder
from pputils.exceptions import ProgrammingError

import numpy as np
import pandas as pd


def array_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a == b) | ((a != a) & (b != b))


class TestOneHotEncoder(unittest.TestCase):
    str_categories = np.array(['a', 'b', 'c', 'd'])

    def setUp(self):
        self.oh = OneHotEncoder().fit(self.str_categories)

    def test_fit(self):
        self.assertTrue(np.all(self.str_categories == self.oh.categories))

    def test_transform_to_labels(self):
        samples = np.array([[['a', 'c'], ['b', 'c']], [['d', 'd'], ['a', 'd']]])
        result = np.array([[[0, 2], [1, 2]], [[3, 3], [0, 3]]])
        self.assertTrue(np.all(self.oh.transform_to_labels(samples) == result))

    def test_transform_from_labels(self):
        labels = np.array([[0, 2], [1, 3]])
        result = np.array([[[1, 0, 0, 0], [0, 0, 1, 0]], [[0, 1, 0, 0], [0, 0, 0, 1]]])
        self.assertTrue(np.all(self.oh.transform_from_labels(labels) == result))

    def test_inverse_from_labels(self):
        labels = np.array([[[0, 2], [1, 2]], [[3, 3], [0, 3]]])
        result = np.array([[['a', 'c'], ['b', 'c']], [['d', 'd'], ['a', 'd']]])
        self.assertTrue(np.all(self.oh.inverse_from_labels(labels) == result))

    def test_inverse_to_labels(self):
        encoded = np.array([[[1, 0, 0, 0], [0, 0, 1, 0]], [[0, 1, 0, 0], [0, 0, 0, 1]]])
        result = np.array([[0, 2], [1, 3]])
        self.assertTrue(np.all(self.oh.inverse_to_labels(encoded) == result))


class TestNanHotEncoder(unittest.TestCase):
    categories = np.array(['a', 'b', 'c', 'd'])

    def setUp(self):
        self.nh = NanHotEncoder().fit(self.categories)

    def test_transform_to_labels(self):
        samples = pd.Series(['a', 'c', np.nan, 'c', 'd', np.nan, 'a', 'd'])
        result = np.array([0, 2, np.nan, 2, 3, np.nan, 0, 3])
        self.assertTrue(np.all(array_equal(self.nh.transform_to_labels(samples), result)))

    def test_transform_from_labels(self):
        labels = np.array([[0, np.nan], [np.nan, 3]])
        result = np.array([[[1, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 1]]])
        self.assertTrue(np.all(array_equal(self.nh.transform_from_labels(labels), result)))

    def test_inverse_from_labels(self):
        labels = np.array([0, 2, np.nan, 2, 3, np.nan, 0, 3])
        result = pd.Series(['a', 'c', np.nan, 'c', 'd', np.nan, 'a', 'd'])
        self.assertTrue(self.nh.inverse_from_labels(labels).equals(result))

    def test_inverse_to_labels(self):
        encoded = np.array([[[1, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 1]]])
        result = np.array([[0, np.nan], [np.nan, 3]])
        self.assertTrue(np.all(array_equal(self.nh.inverse_to_lables(encoded), result)))

    def test_novel_classes(self):
        samples = pd.Series(['a', 'f', np.nan, 'd'])
        result = np.array([[1, 0, 0, 0], [0, 0, 0, 0],  [0, 0, 0, 0], [0, 0, 0, 1]])
        self.assertTrue(np.all(array_equal(self.nh.transform(samples), result)))


class TestCatHotEncoder(unittest.TestCase):
    series = pd.Series(pd.Categorical([np.nan, 'c', 'd', 'a', 'b', 'c', 'c']))

    def setUp(self):
        self.ch = CatHotEncoder().fit(self.series)

    def test_transform_to_labels(self):
        with self.assertRaises(ProgrammingError):
            self.ch.transform_to_labels(self.series)

    def test_transform_from_labels(self):
        labels = np.array([[0, -1], [-1, 3]])
        result = np.array([[[1, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 1]]])
        self.assertTrue(np.all(array_equal(self.ch.transform_from_labels(labels), result)))

    def test_inverse_from_labels(self):
        with self.assertRaises(ProgrammingError):
            self.ch.transform_to_labels(self.series)

    def test_inverse_to_labels(self):
        encoded = np.array([[[1, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 1]]])
        result = np.array([[0, -1], [-1, 3]])
        self.assertTrue(np.all(array_equal(self.ch.inverse_to_lables(encoded), result)))

    def test_novel_classes(self):
        samples = pd.Series(pd.Categorical(['a', 'f', np.nan, 'd']))
        result = np.array([[1, 0, 0, 0], [0, 0, 0, 0],  [0, 0, 0, 0], [0, 0, 0, 1]])
        self.assertTrue(np.all(array_equal(self.ch.transform(samples), result)))


if __name__ == '__main__':
    oh_test = TestOneHotEncoder()
    nh_test = TestNanHotEncoder()
    ch_test = TestCatHotEncoder()
    test = unittest.TestSuite()
    test.addTests([oh_test, nh_test, ch_test])
    res = unittest.TestResult()
    test.run(res)
