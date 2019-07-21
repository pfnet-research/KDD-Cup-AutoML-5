from optable.synthesis import manipulation_candidate
from optable.dataset import feature_types
from optable.manipulations.aggregations import aggregation


class MinManipulation(aggregation.AggregationManipulation):
    def __init__(self, path, dataset, col):
        super(MinManipulation, self).__init__(
            path, dataset, col, "Min", "min", "min", False)

    def calculate_priority(self):
        return 1.6 + 0.5 * self.path.not_deeper_count \
            + 0.7 * self.path.substance_to_many_count(self.dataset, self.col) \
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
            "Min-Constant",
            "Min-NotDeeperCount",
            "Min-ToManyMeta"
        ]


class MinCandidate(manipulation_candidate.ManipulationCandidate):
    def search(self, path, dataset):
        if path.is_to_many:
            dst_table = dataset.tables[path.dst]
            ret = []
            for col in dst_table.df.columns:
                if path.is_substance_to_one_with_col(dataset, col):
                    continue
                ftype = dst_table.ftypes[col]
                if ftype == feature_types.numerical:
                    ret.append(MinManipulation(path, dataset, col))
            return ret
        else:
            return []
