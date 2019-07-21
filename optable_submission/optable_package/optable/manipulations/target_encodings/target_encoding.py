import numpy as np
from sklearn import metrics

from optable.synthesis import manipulation
from optable.synthesis import manipulation_candidate
from optable.dataset import feature_types
from optable import _core
from optable.manipulations.manip_utils import auc_selection


# TODO: もうちょっとto-many, to-oneうまく扱う、このままだと抜けが出てくる
# TODO: dstのlenで判断しているのもよくない


class TargetEncodingManipulation(manipulation.Manipulation):
    def __init__(self, path, dataset, col):
        self.__path = path
        self.__dataset = dataset
        self.__col = col

        super(TargetEncodingManipulation, self).__init__()

    def __repr__(self):
        return "TargetEncoding {} {}".format(self.__path, self.__col)

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
        return 0.7 + 0.8 * self.path.substance_to_many_count(
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
            "TargetEncoding-Constant",
            "TargetEncoding-SubstanceToManyCount"
        ]

    def synthesis(self):
        dst_table = self.dataset.tables[self.path.dst]

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

        to_one_path = None
        to_many_path = None
        for i in range(len(self.__path), -1, -1):
            if self.__path[i:].is_substance_to_one_with_col(
               self.__dataset, self.__col):
                to_one_path = self.__path[i:]
                to_many_path = self.__path[:i]

        # to_one identify
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

        targets = dst_table.pseudo_target

        if dst_table.has_time:
            sorted_index = dst_table.sorted_time_index
            if dst_table.has_hist_time_data:
                time_data = dst_table.hist_time_data
            else:
                time_data = dst_table.time_data
            new_data = categorical_manager \
                .temporal_target_encode_with_dst_induces(
                    targets, dst_induces, time_data, sorted_index,
                    categorical_manager.unique_num
                )
        else:
            new_data = \
                categorical_manager.target_encode_with_dst_induces(
                    targets, dst_induces,
                    categorical_manager.unique_num
                )

        if len(to_many_path) > 0:
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

        new_data_name = "{}TargetEncoding_{}_{}".format(
            feature_types.aggregate_processed_numerical.prefix,
            self.__path, self.__col)
        train_size = np.isfinite(self.__dataset.target).sum()
        train_isfinite = np.isfinite(new_data[:train_size])
        if (len(np.unique(
                new_data[:train_size][train_isfinite])
                ) <= 1):
            return

        if not auc_selection.numerical_adversarial_auc_select(
            self.__dataset, new_data, 0.2
        ):
            return

        self.__dataset.tables[to_many_path.src].set_new_data(
            new_data, new_data_name)


class TargetEncodingCandidate(manipulation_candidate.ManipulationCandidate):
    def search(self, path, dataset):
        if path.not_deeper_count == 0:
            dst_table = dataset.tables[path.dst]
            if not dst_table.has_pseudo_target:
                return []
            ret = []
            for col in dst_table.df.columns:
                ftype = dst_table.ftypes[col]
                if ftype == feature_types.categorical:
                    # or ftype == feature_types.mc_processed_categorical:
                    if len(path) == 0:
                        if (
                            3 * np.log(len(dst_table.df))
                            < dst_table.nunique[col]
                            and dst_table.nunique[col] <=
                            0.1 * len(dst_table.df)
                        ):
                            ret.append(TargetEncodingManipulation(
                                path, dataset, col))
                    elif dst_table.nunique[col] < 0.1 * len(dst_table.df):
                        ret.append(TargetEncodingManipulation(
                            path, dataset, col))
                elif ftype == feature_types.multi_categorical:
                    if dst_table.has_cache(("substance_categorical", col)) \
                       and dst_table.get_cache(("substance_categorical", col)):
                        if (3 * np.log(len(dst_table.df))
                            < dst_table.nunique[col]
                            or (len(path) > 0)) \
                           and dst_table.nunique[col] \
                           < 0.1 * len(dst_table.df):
                            ret.append(TargetEncodingManipulation(
                                path, dataset, col))

            return ret
        else:
            return []
