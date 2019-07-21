import numpy as np

from optable.synthesis import manipulation
from optable.synthesis import manipulation_candidate
from optable.dataset import feature_types
from optable import _core


# TODO: もうちょっとto-many, to-oneうまく扱う、このままだと抜けが出てくる
# TODO: dstのlenで判断しているのもよくない
# TODO: 重複分を省いて早くした方が良い


class MultiCategoricalTargetEncodingManipulation(manipulation.Manipulation):
    def __init__(self, path, dataset, col):
        self.__path = path
        self.__dataset = dataset
        self.__col = col

        super(MultiCategoricalTargetEncodingManipulation, self).__init__()

    def __repr__(self):
        return "MultiCategoricalTargetEncoding {} {}".format(
            self.__path, self.__col)

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
        return 0.8 + 0.9 * self.path.substance_to_many_count(
            self.dataset, self.col)

    def calculate_size(self):
        return 1

    def meta_feature_size():
        return 2

    def meta_feature(self):
        return [
            1,
            self.path.substance_to_many_count(self.dataset, self.col)
        ]

    def meta_feature_name():
        return [
            "MultiCategoricalTargetEncoding-Constant",
            "MultiCategoricalTargetEncoding-SubstanceToManyCount"
        ]

    def synthesis(self):
        dst_table = self.dataset.tables[self.path.dst]

        if dst_table.has_cache(
            ("multi_categorical_manager", self.__col)
        ):
            multi_categorical_manager = dst_table.get_cache(
                ("multi_categorical_manager", self.__col)
            )
        else:
            multi_categorical_string_values = \
                dst_table.df[self.col].fillna("").values
            multi_categorical_manager = \
                _core.MultiCategoricalManager(multi_categorical_string_values)
            dst_table.set_cache(
                ("multi_categorical_manager", self.__col),
                multi_categorical_manager
            )

        to_one_path = None
        to_many_path = None
        for i in range(len(self.__path), -1, -1):
            if self.__path[i:].is_substance_to_one_with_col(
               self.__dataset, self.__col):
                to_one_path = self.__path[i:]
                to_many_path = self.__path[:i]

        # to_one identify of dst_induces
        if len(to_one_path) > 0:
            dst_induces = np.arange(len(dst_table.df))
            time_for_each_table = {
                table_idx: self.dataset.tables[table_name].hour_time_data
                for table_idx, table_name in enumerate(to_one_path.table_names)
                if self.dataset.tables[table_name].has_time}
            sorted_index_for_each_table = {
                table_idx: self.dataset.tables[table_name].sorted_time_index
                for table_idx, table_name in enumerate(to_one_path.table_names)
                if self.dataset.tables[table_name].has_time}
            src_id_for_each_relation = [
                self.dataset.tables[rel.src].df[rel.src_id].values
                for rel in to_one_path.relations
            ]
            dst_id_for_each_relation = [
                self.dataset.tables[rel.dst].df[rel.dst_id].values
                for rel in to_one_path.relations
            ]
            src_is_unique_for_each_relation = [
                rel.type.src_is_unique
                for rel in to_one_path.relations
            ]
            dst_is_unique_for_each_relation = [
                rel.type.dst_is_unique
                for rel in to_one_path.relations
            ]
            dst_induces = _core.Aggregator().aggregate(
                dst_induces, time_for_each_table, sorted_index_for_each_table,
                src_id_for_each_relation, dst_id_for_each_relation,
                src_is_unique_for_each_relation,
                dst_is_unique_for_each_relation,
                "last", "last")
            dst_induces = dst_induces.astype(np.int32)
            dst_induces[dst_induces < 0] = -1
        else:
            dst_table = self.dataset.tables[to_one_path.dst]
            dst_induces = np.arange(len(dst_table.df))
            dst_induces = dst_induces.astype(np.int32)
            dst_induces[dst_induces < 0] = -1

        # target encoding
        dst_table = self.dataset.tables[to_many_path.dst]
        if not dst_table.has_pseudo_target:
            return

        if dst_table.has_time:
            sorted_index = dst_table.sorted_time_index
            if dst_table.has_hist_time_data:
                time_data = dst_table.hist_time_data
            else:
                time_data = dst_table.time_data
            new_data = multi_categorical_manager \
                .temporal_target_encode_with_dst_induces(
                    dst_table.pseudo_target,
                    dst_induces,
                    time_data,
                    sorted_index,
                    multi_categorical_manager.unique_word_num,
                    multi_categorical_manager.row_num
                )
        else:
            new_data = multi_categorical_manager \
                .target_encode_with_dst_induces(
                    dst_table.pseudo_target,
                    dst_induces,
                    multi_categorical_manager.unique_word_num,
                    multi_categorical_manager.row_num
                )

        # to_many_aggregate or set
        if len(to_many_path) == 0:
            new_data_name = "{}MultiCategoricalTargetEncoding_{}_{}".format(
                feature_types.aggregate_processed_numerical.prefix,
                self.__path, self.__col)
            self.__dataset.tables[to_many_path.src].set_new_data(
                new_data, new_data_name)
        else:
            # to_many_aggregate
            dst_data = new_data

            time_for_each_table = {
                table_idx: self.dataset.tables[table_name].hour_time_data
                for table_idx, table_name
                in enumerate(to_many_path.table_names)
                if self.dataset.tables[table_name].has_time}
            sorted_index_for_each_table = {
                table_idx: self.dataset.tables[table_name].sorted_time_index
                for table_idx, table_name
                in enumerate(to_many_path.table_names)
                if self.dataset.tables[table_name].has_time}
            src_id_for_each_relation = [
                self.dataset.tables[rel.src].df[rel.src_id].values
                for rel in to_many_path.relations
            ]
            dst_id_for_each_relation = [
                self.dataset.tables[rel.dst].df[rel.dst_id].values
                for rel in to_many_path.relations
            ]
            src_is_unique_for_each_relation = [
                rel.type.src_is_unique
                for rel in to_many_path.relations
            ]
            dst_is_unique_for_each_relation = [
                rel.type.dst_is_unique
                for rel in to_many_path.relations
            ]

            new_data = _core.Aggregator().aggregate(
                dst_data, time_for_each_table, sorted_index_for_each_table,
                src_id_for_each_relation, dst_id_for_each_relation,
                src_is_unique_for_each_relation,
                dst_is_unique_for_each_relation,
                "mean", "mean")

            new_data_name = "{}MultiCategoricalTargetEncoding_{}_{}".format(
                feature_types.aggregate_processed_numerical.prefix,
                self.__path, self.__col)

            self.__dataset.tables[to_many_path.src].set_new_data(
                new_data, new_data_name)


class MultiCategoricalTargetEncodingCandidate(
    manipulation_candidate.ManipulationCandidate
):
    def search(self, path, dataset):
        dst_table = dataset.tables[path.dst]
        if not dst_table.has_pseudo_target:
            return []
        ret = []
        if path.not_deeper_count == 0:
            for col in dst_table.df.columns:
                ftype = dst_table.ftypes[col]
                if ftype == feature_types.multi_categorical:
                    ret.append(MultiCategoricalTargetEncodingManipulation(
                        path, dataset, col))
        return ret
