import unittest

import numpy as np

import optable._core


class TestTemporalTargetEncode(unittest.TestCase):
    def setUp(self):
        self.encoder = optable._core.TargetEncoder()

    def test_temporal_encode(self):
        targets = np.array([1, 0, 0, 1]).astype(np.float32)
        ids = np.array([0, 1, 1, 0]).astype(np.int32)
        sorted_index = np.array([2, 1, 3, 0]).astype(np.int32)
        time = np.array([3, 1, 0, 2])

        ret = self.encoder.temporal_encode(
            targets, ids, time, sorted_index, 25)

        np.testing.assert_allclose(
            ret, np.array([13.5 / 26, 12.5 / 26, 0.5, 0.5]))

    def test_encode(self):
        targets = np.array([1, 0, 0, 1, 0, 1]).astype(np.float32)
        ids = np.array([0, 1, 1, 0, 0, 1]).astype(np.int32)

        ret = self.encoder.encode(
            targets, ids, 20)

        np.testing.assert_allclose(
            ret, np.array([11/22, 11/22, 11/22, 11/22, 12/22, 10/22]))
