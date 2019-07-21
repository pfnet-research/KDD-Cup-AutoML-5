import pandas as pd

from optable.synthesis import manipulation
from optable.synthesis import manipulation_candidate
from optable.dataset import feature_types
from optable import _core
from optable.manipulations.manip_utils import auc_selection


class IdentifyManipulation(manipulation.Manipulation):
    def __init__(self, path, dataset, col, is_cat):
        self.__path = path
        self.__dataset = dataset
        self.__col = col
        self.__is_cat = is_cat

        super(IdentifyManipulation, self).__init__()

    def __repr__(self):
        return "Identify {} {}".format(self.__path, self.__col)

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
        return 0

    def calculate_size(self):
        return 1

    def meta_feature_size():
        return 0

    def meta_feature(self):
        return []

    def meta_feature_name():
        return []

    def synthesis(self):
        if len(self.__path) == 0:
            return

        if self.__is_cat:
            new_data_name = "{}Identify_{}_{}".format(
                feature_types.aggregate_processed_categorical.prefix,
                self.__path, self.__col)
        else:
            new_data_name = "{}Identify_{}_{}".format(
                feature_types.aggregate_processed_numerical.prefix,
                self.__path, self.__col)

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
        dst_is_unique_for_each_relation = [
            rel.type.dst_is_unique
            for rel in self.__path.relations
        ]

        new_data = _core.Aggregator().aggregate(
            dst_data, time_for_each_table, sorted_index_for_each_table,
            src_id_for_each_relation, dst_id_for_each_relation,
            src_is_unique_for_each_relation, dst_is_unique_for_each_relation,
            "last", "last")

        """
        if (
            not self.__is_cat
            and not auc_selection.numerical_adversarial_auc_select(
                self.__dataset, new_data, 0.2
            )
        ):
            return
        """

        self.__dataset.tables[self.__path.src].set_new_data(
            new_data, new_data_name)


class IdentifyCandidate(manipulation_candidate.ManipulationCandidate):
    def search(self, path, dataset):
        if len(path) > 0:
            dst_table = dataset.tables[path.dst]
            ret = []
            for col in dst_table.df.columns:
                if path.is_substance_to_one_with_col(dataset, col) \
                   and path.self_mm_count == 0:
                    ftype = dst_table.ftypes[col]
                    if (
                        ftype == feature_types.numerical
                        or ftype == feature_types.mc_processed_numerical
                        or ftype == feature_types.c_processed_numerical
                        or ftype == feature_types.t_processed_numerical
                    ):
                        ret.append(IdentifyManipulation(
                            path, dataset, col, False))
                    elif (
                        ftype == feature_types.categorical
                        or ftype == feature_types.c_processed_categorical
                        # or ftype == feature_types.n_processed_categorical
                        # or ftype == feature_types.mc_processed_categorical
                        # or ftype == feature_types.t_processed_categorical
                    ):
                        # if dst_table.nunique[col] < 0.3 * len(dst_table.df):
                        if dst_table.nunique[col] < dataset.max_cat_nunique:
                            ret.append(IdentifyManipulation(
                                path, dataset, col, True))
            return ret
        else:
            return []
