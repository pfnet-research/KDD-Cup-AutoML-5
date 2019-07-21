import pandas as pd
import numpy as np
from sklearn import metrics

from optable.synthesis import manipulation
from optable.dataset import feature_types
from optable.dataset import relation_types
from optable import _core


def get_dst_is_substance_unique(path, dataset, col):
    ret = []
    if len(path.relations) >= 1:
        for r1, r2 in zip(path.relations[:-1], path.relations[1:]):
            if r1.type.dst_is_unique:
                ret.append(True)
            else:
                if r1.type == relation_types.many_to_many \
                   and r1.src == r1.dst:
                    ret.append(False)
                elif dataset.tables[r1.dst].has_cache(
                    ('substance_to_one_columns', r1.dst_id)
                ):
                    if not dataset.tables[r1.dst].get_cache(
                        ('substance_to_one_columns', r1.dst_id)
                    )[r2.src_id]:
                        ret.append(True)
                else:
                    ret.append(False)

        last_r = path.relations[-1]
        if last_r.type.dst_is_unique:
            ret.append(True)
        else:
            if last_r.type == relation_types.many_to_many \
               and last_r.src == last_r.dst:
                ret.append(False)
            elif dataset.tables[last_r.dst].has_cache(
                ('substance_to_one_columns', last_r.dst_id)
            ):
                if not dataset.tables[last_r.dst].get_cache(
                    ('substance_to_one_columns', last_r.dst_id)
                )[col]:
                    ret.append(True)
            else:
                ret.append(False)

    return ret


class AggregationManipulation(manipulation.Manipulation):
    def __init__(self, path, dataset, col, name, last_agg, other_agg, is_cat):
        self.__path = path
        self.__dataset = dataset
        self.__col = col
        self.__name = name
        self.__last_agg = last_agg
        self.__other_agg = other_agg
        self.__is_cat = is_cat

        super(AggregationManipulation, self).__init__()

    def __repr__(self):
        return "{} {} {}".format(self.__name, self.__path, self.__col)

    @property
    def path(self):
        return self.__path

    @property
    def dataset(self):
        return self.__dataset

    @property
    def col(self):
        return self.__col

    def calculate_priority(self):
        pass

    def calculate_size(self):
        return 1

    def meta_feature_size():
        pass

    def meta_feature(self):
        pass

    def meta_feature_name():
        pass

    def synthesis(self):
        if len(self.__path) == 0:
            return

        if self.__is_cat:
            new_data_name = "{}{}_{}_{}".format(
                feature_types.aggregate_processed_categorical.prefix,
                self.__name, self.__path, self.__col)
        else:
            new_data_name = "{}{}_{}_{}".format(
                feature_types.aggregate_processed_numerical.prefix,
                self.__name, self.__path, self.__col)

        dst_table = self.dataset.tables[self.__path.dst]
        dst_df = dst_table.df
        dst_data = dst_df[self.__col].values
        time_for_each_table = {
            table_idx: self.dataset.tables[table_name].hour_time_data
            for table_idx, table_name in enumerate(self.__path.table_names)
            if self.dataset.tables[table_name].has_time}
        sorted_index_for_each_table = {
            table_idx: self.dataset.tables[table_name].sorted_time_index
            for table_idx, table_name in enumerate(self.__path.table_names)
            if self.dataset.tables[table_name].has_time}
        src_id_for_each_relation = [
            self.dataset.tables[rel.src].df[rel.src_id].values
            for rel in self.__path.relations
        ]
        dst_id_for_each_relation = [
            self.dataset.tables[rel.dst].df[rel.dst_id].values
            for rel in self.__path.relations
        ]
        src_is_unique_for_each_relation = [
            rel.type.src_is_unique
            for rel in self.__path.relations
        ]
        """
        dst_is_unique_for_each_relation = [
            rel.type.dst_is_unique
            for rel in self.__path.relations
        ]
        """
        dst_is_unique_for_each_relation = get_dst_is_substance_unique(
            self.__path, self.__dataset, self.__col)

        new_data = _core.Aggregator().aggregate(
            dst_data, time_for_each_table, sorted_index_for_each_table,
            src_id_for_each_relation, dst_id_for_each_relation,
            src_is_unique_for_each_relation, dst_is_unique_for_each_relation,
            self.__last_agg, self.__other_agg)

        if self.__is_cat:
            self.__dataset.tables[self.__path.src].set_new_data(
                new_data, new_data_name)
        else:
            train_size = np.isfinite(self.__dataset.target).sum()
            train_isfinite = np.isfinite(new_data[:train_size])
            if not train_isfinite.any():
                return
            score = metrics.roc_auc_score(
                self.__dataset.target[:train_size][train_isfinite],
                new_data[:train_size][train_isfinite])
            score = np.abs(score - 0.5)
            if score > 0.001:
                self.__dataset.tables[self.__path.src].set_new_data(
                    new_data, new_data_name)
