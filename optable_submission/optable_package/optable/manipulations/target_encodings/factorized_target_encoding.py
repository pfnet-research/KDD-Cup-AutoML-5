import numpy as np

from optable.synthesis import manipulation
from optable.synthesis import manipulation_candidate
from optable.dataset import feature_types
from optable import _core


# TODO: もうちょっとto-many, to-oneうまく扱う、このままだと抜けが出てくる
# TODO: dstのlenで判断しているのもよくない


class FactorizedTargetEncodingManipulation(manipulation.Manipulation):
    def __init__(self, path, dataset, col1, col2):
        self.__path = path
        self.__dataset = dataset
        self.__col1 = col1
        self.__col2 = col2

        super(FactorizedTargetEncodingManipulation, self).__init__()

    def __repr__(self):
        return "FactorizedTargetEncoding {} {} {}".format(
            self.__path, self.__col1, self.__col2)

    @property
    def path(self):
        return self.__path

    @property
    def dataset(self):
        return self.__dataset

    @property
    def col1(self):
        return self.__col1

    @property
    def col2(self):
        return self.__col2

    def calculate_priority(self):
        return 2.7 + 0.5 * self.path.substance_to_many_count(
            self.dataset, self.col1) + 0.5 * self.path.substance_to_many_count(
                self.dataset, self.col2)

    def calculate_size(self):
        return 1

    def meta_feature_size():
        return 3

    def meta_feature(self):
        return [
            1,
            self.path.substance_to_many_count(self.dataset, self.col1),
            self.path.substance_to_many_count(self.dataset, self.col2)
        ]

    def meta_feature_name():
        return [
            "FactorizedTargetEncoding-Constant",
            "FactorizedTargetEncoding-SubstanceToManyCount1",
            "FactorizedTargetEncoding-SubstanceToManyCount2",
        ]

    def synthesis(self):
        to_one_path = None
        to_many_path = None
        for i in range(len(self.__path), -1, -1):
            if self.__path[i:].is_substance_to_one_with_col(
               self.__dataset, self.__col1) \
               and self.__path[i:].is_substance_to_one_with_col(
                  self.__dataset, self.__col2):
                to_one_path = self.__path[i:]
                to_many_path = self.__path[:i]

        # to_one identify
        if len(to_one_path) > 0:
            dst_table = self.dataset.tables[to_one_path.dst]
            dst_data1 = dst_table.df[self.col1].values
            dst_data2 = dst_table.df[self.col2].values
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
            ids1 = _core.Aggregator().aggregate(
                dst_data1, time_for_each_table, sorted_index_for_each_table,
                src_id_for_each_relation, dst_id_for_each_relation,
                src_is_unique_for_each_relation,
                dst_is_unique_for_each_relation,
                "last", "last")
            ids2 = _core.Aggregator().aggregate(
                dst_data2, time_for_each_table, sorted_index_for_each_table,
                src_id_for_each_relation, dst_id_for_each_relation,
                src_is_unique_for_each_relation,
                dst_is_unique_for_each_relation,
                "last", "last")
            ids1 = ids1.astype(np.int32)
            ids1[ids1 < 0] = -1
            ids2 = ids2.astype(np.int32)
            ids2[ids2 < 0] = -1
        else:
            dst_table = self.dataset.tables[to_one_path.dst]
            ids1 = dst_table.df[self.col1].values
            ids2 = dst_table.df[self.col2].values
            ids1 = ids1.astype(np.int32)
            ids1[ids1 < 0] = -1
            ids2 = ids2.astype(np.int32)
            ids2[ids2 < 0] = -1

        # target encoding
        dst_table = self.dataset.tables[to_many_path.dst]
        if not dst_table.has_pseudo_target:
            return

        targets = dst_table.pseudo_target
        encoder = _core.FactorizedTargetEncoder()
        k1 = len(np.unique(ids1))
        k2 = len(np.unique(ids2))
        k0 = k1 * k2
        if dst_table.has_hist_time_data:
            sorted_index = dst_table.sorted_time_index
            time_data = dst_table.hist_time_data
            new_data = encoder.temporal_encode(
                targets, ids1, ids2, time_data, sorted_index, k0, k1, k2)
        elif dst_table.has_time:
            sorted_index = dst_table.sorted_time_index
            time_data = dst_table.time_data
            new_data = encoder.temporal_encode(
                targets, ids1, ids2, time_data, sorted_index, k0, k1, k2)
        else:
            new_data = encoder.encode(targets, ids1, ids2, k0, k1, k2)

        if len(to_many_path) == 0:
            new_data_name = "{}FactorizedTargetEncoding_{}_{}_{}".format(
                feature_types.aggregate_processed_numerical.prefix,
                self.__path, self.__col1, self.__col2)
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

            new_data_name = "{}FactorizedTargetEncoding_{}_{}_{}".format(
                feature_types.aggregate_processed_numerical.prefix,
                self.__path, self.__col1, self.__col2)

            train_size = np.isfinite(self.dataset.target).sum()
            from sklearn import metrics
            print("Factorized")
            print(metrics.roc_auc_score(
                self.dataset.target[:train_size][
                    np.isfinite(new_data[:train_size])],
                new_data[:train_size][np.isfinite(new_data[:train_size])]))

            self.__dataset.tables[to_many_path.src].set_new_data(
                new_data, new_data_name)


class FactorizedTargetEncodingCandidate(
    manipulation_candidate.ManipulationCandidate
):
    def search(self, path, dataset):
        if path.not_deeper_count == 0:
            dst_table = dataset.tables[path.dst]
            if not dst_table.has_pseudo_target:
                return []
            ret = []
            categorical_cols = []
            for col in dst_table.df.columns:
                ftype = dst_table.ftypes[col]
                if ftype == feature_types.categorical:
                    if path.is_substance_to_one_with_col(dataset, col):
                        continue
                    if 1 < dst_table.nunique[col] \
                       and dst_table.nunique[col] < np.sqrt(len(dst_table.df)):
                        categorical_cols.append(col)

            for col_idx1, col1 in enumerate(categorical_cols):
                for col_idx2, col2 in enumerate(categorical_cols):
                    if col_idx1 < col_idx2:
                        ret.append(FactorizedTargetEncodingManipulation(
                            path, dataset, col1, col2))
            return ret
        else:
            return []
