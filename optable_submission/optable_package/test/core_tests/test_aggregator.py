import unittest

import numpy as np

import optable._core


class TestAggregator(unittest.TestCase):
    def setUp(self):
        self.aggregator = optable._core.Aggregator()

    def test_one_many_one(self):
        dst_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
        src_id_0 = np.array([0, 1, 2, 3, 4, 5]).astype(np.int32)
        dst_id_0 = np.array([0, 1, 0, 0, 2, 4, 5, 5]).astype(np.int32)
        src_id_1 = np.array([0, 3, 2, 2, 4, 4, 7, 8]).astype(np.int32)
        dst_id_1 = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.int32)
        src_id_for_each_relation = [src_id_0, src_id_1]
        dst_id_for_each_relation = [dst_id_0, dst_id_1]
        time_0 = np.array([0.5, 0.7, 0.0, 0.9, 0.2, 0.1]).astype(np.float32)
        time_2 = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ).astype(np.float32)
        time_for_each_table = {0: time_0, 2: time_2}
        sorted_index_for_each_table = \
            {key: np.argsort(time_for_each_table[key])
             for key in time_for_each_table}
        src_is_unique_for_each_relation = \
            [len(np.unique(src_id)) == len(src_id)
             for src_id in src_id_for_each_relation]
        dst_is_unique_for_each_relation = \
            [len(np.unique(dst_id)) == len(dst_id)
             for dst_id in dst_id_for_each_relation]

        ret = self.aggregator.aggregate(
            dst_data, time_for_each_table, sorted_index_for_each_table,
            src_id_for_each_relation, dst_id_for_each_relation,
            src_is_unique_for_each_relation, dst_is_unique_for_each_relation,
            "mean", "mean")

        np.testing.assert_allclose(
            np.isnan(ret), [False, False, True, True, True, True]
        )
        np.testing.assert_allclose(
            ret[[0, 1]], [7 / 3, 4]
        )

    def test_many_one_many(self):
        dst_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
        src_id_0 = np.array([0, 1, 0, 3, 2, 2]).astype(np.int32)
        dst_id_0 = np.array([0, 1, 2, 3]).astype(np.int32)
        src_id_1 = np.array([4, 3, 2, 1]).astype(np.int32)
        dst_id_1 = np.array([1, 1, 0, 2, 2, 4, 3, 4, 4, 2]).astype(np.int32)
        src_id_for_each_relation = [src_id_0, src_id_1]
        dst_id_for_each_relation = [dst_id_0, dst_id_1]
        time_0 = np.array([0.5, 0.7, 0.0, 0.9, 0.2, 0.1]).astype(np.float32)
        time_2 = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ).astype(np.float32)
        time_for_each_table = {0: time_0, 2: time_2}
        sorted_index_for_each_table = \
            {key: np.argsort(time_for_each_table[key])
             for key in time_for_each_table}
        src_is_unique_for_each_relation = \
            [len(np.unique(src_id)) == len(src_id)
             for src_id in src_id_for_each_relation]
        dst_is_unique_for_each_relation = \
            [len(np.unique(dst_id)) == len(dst_id)
             for dst_id in dst_id_for_each_relation]

        ret = self.aggregator.aggregate(
            dst_data, time_for_each_table, sorted_index_for_each_table,
            src_id_for_each_relation, dst_id_for_each_relation,
            src_is_unique_for_each_relation, dst_is_unique_for_each_relation,
            "mean", "mean")

        np.testing.assert_allclose(
            np.isnan(ret), [True, False, True, False, True, True]
        )
        np.testing.assert_allclose(
            ret[[1, 3]], [7, 3 / 2]
        )


class TestMergeSortedIndex(unittest.TestCase):
    def test_normal(self):
        for i in range(10):
            time_0 = np.random.normal(size=10)
            time_1 = np.random.normal(size=10)
            sorted_index_0 = np.argsort(time_0)
            sorted_index_1 = np.argsort(time_1)

            for is_src_priority in [True, False]:
                merged_is_src, merged_sorted_index = \
                    optable._core.merge_sorted_index(
                        time_0, time_1, sorted_index_0, sorted_index_1,
                        is_src_priority)

                times = np.stack([time_0, time_1])
                merged_times = \
                    times[1 - merged_is_src, merged_sorted_index]

                self.assertTrue((
                    merged_times[1:] - merged_times[:-1] >= 0).all())
                np.testing.assert_allclose(
                    merged_sorted_index[merged_is_src == 1], sorted_index_0)
                np.testing.assert_allclose(
                    merged_sorted_index[merged_is_src == 0], sorted_index_1)

    def test_nan(self):
        for i in range(10):
            nan_array = np.array([np.nan for i in range(10)])
            time_0 = np.random.normal(size=10)
            time_0 = np.concatenate([time_0, nan_array])
            np.random.shuffle(time_0)
            time_1 = np.random.normal(size=10)
            time_1 = np.concatenate([time_1, nan_array])
            np.random.shuffle(time_1)
            sorted_index_0 = np.argsort(time_0)
            sorted_index_1 = np.argsort(time_1)

            for is_src_priority in [True, False]:
                merged_is_src, merged_sorted_index = \
                    optable._core.merge_sorted_index(
                        time_0, time_1, sorted_index_0, sorted_index_1,
                        is_src_priority)
                times = np.stack([time_0, time_1])
                merged_times = \
                    times[1 - merged_is_src, merged_sorted_index]

                self.assertTrue(np.isnan(merged_times[-20:]).all())
                self.assertTrue(
                    (merged_times[1:-20] - merged_times[:-21] >= 0).all())
                np.testing.assert_allclose(
                    merged_sorted_index[merged_is_src == 1], sorted_index_0)
                np.testing.assert_allclose(
                    merged_sorted_index[merged_is_src == 0], sorted_index_1)

    def test_priority(self):
        time_0 = [0, 1, 2, 5]
        time_1 = [1, 2, 3, 4]
        sorted_index_0 = np.argsort(time_0)
        sorted_index_1 = np.argsort(time_1)
        merged_is_src, merged_sorted_index = \
            optable._core.merge_sorted_index(
                time_0, time_1, sorted_index_0, sorted_index_1, True)
        np.testing.assert_allclose(
            merged_sorted_index, [0, 1, 0, 2, 1, 2, 3, 3])
        np.testing.assert_allclose(
            merged_is_src, [1, 1, 0, 1, 0, 0, 0, 1])

        merged_is_src, merged_sorted_index = \
            optable._core.merge_sorted_index(
                time_0, time_1, sorted_index_0, sorted_index_1, False)
        np.testing.assert_allclose(
            merged_sorted_index, [0, 0, 1, 1, 2, 2, 3, 3])
        np.testing.assert_allclose(
            merged_is_src, [1, 0, 1, 0, 1, 0, 0, 1])


class TestSquashSplitedRelations(unittest.TestCase):
    def test_double(self):
        for i in range(10):
            src_id_0 = np.array([0, 1, 0, 3, 2, 2]).astype(np.int32)
            dst_id_0 = np.array([0, 1, 2, 3]).astype(np.int32)
            src_id_1 = np.array([4, 3, 2, 1]).astype(np.int32)
            dst_id_1 = np.array(
                [1, 1, 0, 2, 2, 4, 3, 4, 4, 2]).astype(np.int32)

            squashed_id = optable._core.squash_splited_relations(
                [src_id_0, src_id_1], [dst_id_0, dst_id_1])
            np.testing.assert_allclose(
                squashed_id, np.array([4, 3, 4, 1, 2, 2]))

    def test_triple(self):
        for i in range(10):
            src_id_0 = np.array([0, 1, 2, 3])
            dst_id_0 = np.array([2, 1, 3, 0])
            src_id_1 = np.array([3, 2, 1, 0])
            dst_id_1 = np.array([0, 3, 2, 1])
            src_id_2 = np.array([4, 3, 2, 1])
            dst_id_2 = np.array([1, 1, 0, 2, 2, 4, 3, 4, 4, 2])

            squashed_id = optable._core.squash_splited_relations(
                [src_id_0, src_id_1], [dst_id_0, dst_id_1])
            squashed_id = optable._core.squash_splited_relations(
                [src_id_0, src_id_1, src_id_2], [dst_id_0, dst_id_1, dst_id_2])

    def test_not_exist_double(self):
        for i in range(10):
            src_id_0 = np.array([0, 1, 0, 3, 2, 2, -1]).astype(np.int32)
            dst_id_0 = np.array([0, 1, 2, 3]).astype(np.int32)
            src_id_1 = np.array([4, 3, 2, 1]).astype(np.int32)
            dst_id_1 = np.array(
                [1, 1, 0, 2, 2, 4, 3, 4, 4, 2]).astype(np.int32)

            squashed_id = optable._core.squash_splited_relations(
                [src_id_0, src_id_1], [dst_id_0, dst_id_1])
            np.testing.assert_allclose(
                squashed_id, np.array([4, 3, 4, 1, 2, 2, -1]))


class TestGetSrcTimeAndSortedIndex(unittest.TestCase):
    def test_normal(self):
        dst_time = np.random.normal(size=10)
        src_id = np.array([0, 1, 0, 3, 3, 2]).astype(np.int32)
        dst_id = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.int32)

        src_time_and_sorted_index = \
            optable._core.get_src_time_and_sorted_index(
                dst_time, src_id, dst_id
            )
        src_time, src_sorted_index = src_time_and_sorted_index

        self.assertTrue(
            (src_time[src_sorted_index][1:]
             - src_time[src_sorted_index][:-1] >= 0).all())
        np.testing.assert_allclose(
            np.sort(src_sorted_index), np.arange(6))

    def test_only_src_exist(self):
        dst_time = np.random.normal(size=10)
        src_id = np.array([0, 1, 0, 3, 3, -1]).astype(np.int32)
        dst_id = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.int32)

        src_time_and_sorted_index = \
            optable._core.get_src_time_and_sorted_index(
                dst_time, src_id, dst_id
            )
        src_time, src_sorted_index = src_time_and_sorted_index

        self.assertTrue(np.isnan(src_time[5]))
        self.assertEqual(src_sorted_index[5], 5)
        self.assertTrue(
            (src_time[src_sorted_index][1:-1]
             - src_time[src_sorted_index][:-2] >= 0).all())
        np.testing.assert_allclose(
            np.sort(src_sorted_index), np.arange(6))


class TestNotTemporalToOneAggregate(unittest.TestCase):
    def test_normal(self):
        for i in range(10):
            dst_data = np.array([1, 2, 3, 4])
            src_id = np.array([-1, 0, 1, 2])
            dst_id = np.array([0, 1, 2, 3])

            src_data = optable._core.not_temporal_to_one_aggregate(
                dst_data,
                src_id, dst_id
            )
            np.testing.assert_allclose(
                np.isnan(src_data), [True, False, False, False])
            np.testing.assert_allclose(
                src_data[[1, 2, 3]], [1, 2, 3]
            )


class TestTemporalToOneAggregate(unittest.TestCase):
    def test_normal(self):
        for i in range(10):
            dst_data = np.array([1, 2, 3, 4])
            src_id = np.array([-1, 0, 1, 2])
            dst_id = np.array([0, 1, 2, 3])
            src_time = np.array([5, 6, 7, 0])
            dst_time = np.array([1, 2, 3, 4])
            src_sorted_index = np.argsort(src_time)
            dst_sorted_index = np.argsort(dst_time)

            src_data = optable._core.temporal_to_one_aggregate(
                dst_data,
                src_id, dst_id,
                src_time, dst_time,
                src_sorted_index, dst_sorted_index
            )
            print("src_data", src_data)
            np.testing.assert_allclose(
                np.isnan(src_data), [True, False, False, True])
            np.testing.assert_allclose(
                src_data[[1, 2]], [1, 2]
            )


class TestNotTemporalToManyAggregate(unittest.TestCase):
    def test_sum(self):
        dst_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        src_id = np.array([-1, 0, 1, 2, 3])
        dst_id = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

        src_data = optable._core.not_temporal_to_many_aggregate(
            dst_data,
            src_id, dst_id,
            "sum"
        )
        print("src_data", src_data)
        np.testing.assert_allclose(
            np.isnan(src_data), [False, False, False, False, False]
        )
        np.testing.assert_allclose(
            src_data[[0, 1, 2, 3, 0]], [0, 18, 12, 15, 0]
        )

    def test_mean(self):
        dst_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        src_id = np.array([-1, 0, 1, 2, 3])
        dst_id = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

        src_data = optable._core.not_temporal_to_many_aggregate(
            dst_data,
            src_id, dst_id,
            "mean"
        )
        np.testing.assert_allclose(
            np.isnan(src_data), [True, False, False, False, True]
        )
        np.testing.assert_allclose(
            src_data[[1, 2, 3]], [4.5, 4, 5]
        )


class TestTemporalToManyAggregate(unittest.TestCase):
    def test_sum(self):
        dst_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        src_id = np.array([-1, 0, 1, 2, 3])
        dst_id = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        src_time = np.array([5, 6, 7, 0, -1])
        dst_time = np.array([1, 2, 3, 4, 5, 5, 4, 3, 2, 1])
        src_sorted_index = np.argsort(src_time)
        dst_sorted_index = np.argsort(dst_time)

        src_data = optable._core.temporal_to_many_aggregate(
            dst_data,
            src_id, dst_id,
            src_time, dst_time,
            src_sorted_index, dst_sorted_index,
            "sum"
        )
        np.testing.assert_allclose(
            np.isnan(src_data), [False, False, False, False, False]
        )
        np.testing.assert_allclose(
            src_data[[0, 1, 2, 3, 4]], [0, 18, 12, 0, 0]
        )

    def test_mean(self):
        dst_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        src_id = np.array([-1, 0, 1, 2, 3])
        dst_id = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        src_time = np.array([5, 6, 7, 0, -1])
        dst_time = np.array([1, 2, 3, 4, 5, 5, 4, 3, 2, 1])
        src_sorted_index = np.argsort(src_time)
        dst_sorted_index = np.argsort(dst_time)

        src_data = optable._core.temporal_to_many_aggregate(
            dst_data,
            src_id, dst_id,
            src_time, dst_time,
            src_sorted_index, dst_sorted_index,
            "mean"
        )
        np.testing.assert_allclose(
            np.isnan(src_data), [True, False, False, True, True]
        )
        np.testing.assert_allclose(
            src_data[[1, 2]], [4.5, 4]
        )
