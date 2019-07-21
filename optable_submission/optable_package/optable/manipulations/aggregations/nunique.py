from optable.synthesis import manipulation_candidate
from optable.dataset import feature_types
from optable.manipulations.aggregations import aggregation


class NuniqueManipulation(aggregation.AggregationManipulation):
    def __init__(self, path, dataset, col):
        super(NuniqueManipulation, self).__init__(
            path, dataset, col, "Nunique", "nunique", "max", False)

    def calculate_priority(self):
        return 0.5 + 0.3 * self.path.not_deeper_count \
            + 0.2 * self.path.substance_to_many_count(self.dataset, self.col) \
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
            "NUnique-Constant",
            "NUnique-NotDeeperCount",
            "NUnique-ToManyMeta"
        ]


class NuniqueCandidate(manipulation_candidate.ManipulationCandidate):
    def search(self, path, dataset):
        if path.is_to_many:
            dst_table = dataset.tables[path.dst]
            ret = []
            for col in dst_table.df.columns:
                if path.is_substance_to_one_with_col(dataset, col):
                    continue
                ftype = dst_table.ftypes[col]
                if ftype == feature_types.categorical:
                    # or ftype == feature_types.mc_processed_categorical:
                    # or ftype == feature_types.c_processed_categorical \
                    # or ftyep == feature_types.n_processed_categorical:
                    if dst_table.nunique[col] < 0.3 * len(dst_table.df):
                        ret.append(NuniqueManipulation(path, dataset, col))
            return ret
        else:
            return []
