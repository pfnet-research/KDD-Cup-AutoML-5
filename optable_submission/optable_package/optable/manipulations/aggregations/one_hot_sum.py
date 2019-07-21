import collections
import numpy as np
from scipy import stats

from optable.synthesis import manipulation
from optable.synthesis import manipulation_candidate
from optable.dataset import feature_types
from optable import _core


class OneHotSumManipulation(manipulation.Manipulation):
    def __init__(self, path, dataset, col, value):
        self.__path = path
        self.__dataset = dataset
        self.__col = col
        self.__value = value

        super(OneHotSumManipulation, self).__init__()

    def __repr__(self):
        return "One Hot Sum {} {} {}".format(
            self.__path, self.__col, self.__value)

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
        return 0.5 + 0.2 * self.path.not_deeper_count \
            + 0.5 * self.path.substance_to_many_count(self.dataset, self.col) \
            * self.path.to_many_path_priority(self.dataset, self.col)

    def meta_feature_size():
        return 3

    def meta_feature(self):
        to_many_meta = self.path.substance_to_many_count(
            self.dataset, self.col) \
            * self.path.to_many_path_priority(
            self.dataset, self.col)
        return [1, self.path.not_deeper_count, to_many_meta]

    def meta_feature_name():
        return [
            "OneHotSum-Constant",
            "OneHotSum-NotDeeperCount",
            "OneHotSum-ToManyMeta"
        ]

    def calculate_size(self):
        return 1

    def synthesis(self):
        self.__recursive_synthesis(self.__path)

    def __recursive_synthesis(self, path):
        if len(self.__path) == 0:
            return

        new_data_name = "{}OneHotSum_{}_{}_{}".format(
            feature_types.aggregate_processed_numerical.prefix,
            path, self.__col, self.__value)
        dst_table = self.dataset.tables[self.__path.dst]

        if dst_table.has_cache(
            ("categorical_manager", self.__col)
        ):
            categorical_manager = dst_table.get_cache(
                ("categorical_manager", self.__col)
            )
        else:
            processing_data = \
                dst_table.df[self.__col].fillna("").astype(str).values
            categorical_manager = \
                _core.CategoricalManager(processing_data)
            dst_table.set_cache(
                ("categorical_manager", self.__col),
                categorical_manager
            )

        dst_data = categorical_manager.is_array(self.__value)
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
        dst_is_unique_for_each_relation = [
            rel.type.dst_is_unique
            for rel in self.__path.relations
        ]

        new_data = _core.Aggregator().aggregate(
            dst_data, time_for_each_table, sorted_index_for_each_table,
            src_id_for_each_relation, dst_id_for_each_relation,
            src_is_unique_for_each_relation, dst_is_unique_for_each_relation,
            "sum", "sum")
        self.__dataset.tables[self.__path.src].set_new_data(
            new_data, new_data_name)


class OneHotSumCandidate(manipulation_candidate.ManipulationCandidate):
    def search(self, path, dataset):
        if path.is_to_many:
            dst_table = dataset.tables[path.dst]
            ret = []
            for col in dst_table.df.columns:
                if path.is_substance_to_one_with_col(dataset, col):
                    continue
                ftype = dst_table.ftypes[col]
                if ftype == feature_types.categorical \
                   or ftype == feature_types.c_processed_categorical:
                    dst_data = dataset.tables[path.dst].df[col].values

                    if dst_table.has_cache(
                        ("categorical_manager", col)
                    ):
                        categorical_manager = dst_table.get_cache(
                            ("categorical_manager", col)
                        )
                    else:
                        processing_data = \
                            dst_table.df[col].fillna("").astype(str).values
                        categorical_manager = \
                            _core.CategoricalManager(processing_data)
                        dst_table.set_cache(
                            ("categorical_manager", col),
                            categorical_manager
                        )
                    if dst_table.nunique[col] == 2:
                        mode = categorical_manager.most_common(1)[0][0]
                        ret.append(OneHotSumManipulation(
                            path, dataset, col, mode))
                    else:
                        for value, freq in categorical_manager.most_common(5):
                            if freq > 1:
                                ret.append(OneHotSumManipulation(
                                    path, dataset, col, value))
            return ret
        else:
            return []
