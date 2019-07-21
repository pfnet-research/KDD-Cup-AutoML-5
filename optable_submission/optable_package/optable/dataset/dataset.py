import collections
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import metrics

from optable.dataset import relation_types
from optable.dataset import feature_types


class Dataset(object):
    def __init__(self):
        self.__tables = {}
        self.__relations = []
        self.__accessibilities = {}
        self.__depths = {}
        self.__main_table_name = None
        self.__max_depth = None
        self.__target = None
        self.__train_size = None
        self.__test_size = None
        self.__max_cat_nunique = None
        self.__minority_count = None

    def __copy__(self):
        dst_dataset = Dataset()
        for key in self.__tables:
            dst_dataset.set_table(key, self.__tables[key])
        for relation in self.__relations:
            dst_dataset.set_relation(relation)
        for key in self.__accessibilities:
            dst_dataset.set_accessibility(
                key[0], key[1], self.__accessibilities[key])
        return dst_dataset

    @property
    def tables(self):
        return self.__tables

    @property
    def relations(self):
        return self.__relations

    @property
    def accessibilities(self):
        return self.__accessibilities

    @property
    def main_table_name(self):
        return self.__main_table_name

    @property
    def max_depth(self):
        return self.__max_depth

    @property
    def depths(self):
        return self.__depths

    @property
    def target(self):
        return self.__target

    @property
    def max_cat_nunique(self):
        return self.__max_cat_nunique

    @property
    def minority_count(self):
        return self.__minority_count

    @property
    def train_size(self):
        return self.__train_size

    @property
    def test_size(self):
        return self.__test_size

    def set_table(self, name, table, is_main_table=False):
        self.__tables[name] = table
        if is_main_table:
            self.__main_table_name = name

    def set_relation(self, relation):
        self.__relations.append(relation)

    def set_accessibility(self, src, dst, accessibility):
        self.__accessibilities[(src, dst)] = accessibility

    def reset_indexers(self):
        self.__indexers = {}

    def depth_search(self):
        assert self.__main_table_name is not None
        queue = collections.deque([self.__main_table_name])
        self.__depths[self.__main_table_name] = 0
        while queue:
            table_name = queue.popleft()
            for relation in self.relations:
                if relation.src == table_name:
                    next_table_name = relation.dst
                elif relation.dst == table_name:
                    next_table_name = relation.src
                else:
                    continue
                if next_table_name not in self.__depths:
                    self.__depths[next_table_name] = \
                        self.__depths[table_name] + 1
                    queue.append(next_table_name)
        self.__max_depth = max([self.__depths[k] for k in self.__depths])

    def set_target(self, target):
        self.__target = target
        self.__minority_count = min([
            int((target == 1).sum()),
            int((target == 0).sum())
        ])
        self.__max_cat_nunique = min([
            int(np.sqrt((target == 1).sum())),
            int(np.sqrt((target == 0).sum()))
        ])

    def set_train_size(self, train_size):
        self.__train_size = train_size
        self.__test_size = len(self.__target) - train_size

    def clear_cache_of_table(self):
        for table_name in self.__tables:
            self.__tables[table_name].clear_cache()

    def make_pseudo_target(self):
        assert self.__main_table_name is not None
        queue = collections.deque([self.__main_table_name])
        pseudo_target_dist = {table_name: None for table_name in self.__tables}
        stacked_pseudo_target_dist = {
            table_name: [] for table_name in self.__tables
        }
        pseudo_target_dist[self.__main_table_name] = self.__target
        stacked_pseudo_target_dist[self.__main_table_name] = [self.__target]
        while queue:
            table_name = queue.popleft()
            for relation in self.relations:
                if relation.src == table_name:
                    pass
                elif relation.dst == table_name:
                    relation = relation.inverse
                else:
                    continue
                next_table_name = relation.dst
                prev_id_data = \
                    self.__tables[table_name].df[relation.src_id]
                next_id_data = \
                    self.__tables[next_table_name].df[relation.dst_id]
                if self.__depths[next_table_name] > self.__depths[table_name]:
                    prev_df = pd.DataFrame({
                        "target": pseudo_target_dist[table_name],
                        "id": prev_id_data})
                    next_df = pd.DataFrame({
                        "id": next_id_data
                    })
                    prev_df = prev_df[np.isfinite(prev_df["target"].values)]
                    if relation.type == relation_types.many_to_one \
                       or relation.type == relation_types.many_to_many:
                        mean_target = prev_df["target"].mean()
                        k = 1000
                        new_pseudo_target = next_df.merge(
                            (
                                (prev_df.groupby("id").sum() + k * mean_target)
                                / (prev_df.groupby("id").count() + k)
                            ).reset_index(),
                            how="left"
                        )["target"].values
                        new_pseudo_target[np.isnan(new_pseudo_target)] = \
                            mean_target
                    else:
                        new_pseudo_target = next_df.merge(
                            prev_df,
                            how="left"
                        )["target"].values
                    stacked_pseudo_target_dist[next_table_name].append(
                        new_pseudo_target
                    )
                    pseudo_target_dist[next_table_name] = np.nanmean(
                        np.stack(stacked_pseudo_target_dist[next_table_name]),
                        axis=0
                    )
                    queue.append(next_table_name)

        leak_columns = \
            {table_name:
             set([]) for table_name in pseudo_target_dist}
        for relation in self.relations:
            leak_columns[relation.src].add(relation.src_id)
            leak_columns[relation.dst].add(relation.dst_id)

        for table_name in pseudo_target_dist:
            pseudo_target = pseudo_target_dist[table_name]
            if table_name == "main":
                self.__tables[table_name].set_pseudo_target(pseudo_target)
            else:
                surrogate_data = None
                surrogate_col = ""
                surrogate_score = 0
                for col in self.tables[table_name].df.columns:
                    if col in leak_columns[table_name]:
                        continue
                    if "tfidf" in col:
                        continue
                    tmp_data = None
                    if self.tables[table_name].ftypes[col] \
                       == feature_types.n_processed_categorical \
                       or self.tables[table_name].ftypes[col] \
                       == feature_types.categorical:
                        tmp_data = self.tables[table_name].df[col].values
                        tmp_data = (tmp_data == stats.mode(tmp_data)[0]) * 1
                    elif (
                        self.tables[table_name].ftypes[col]
                        == feature_types.numerical
                        or self.tables[table_name].ftypes[col]
                        == feature_types.mc_processed_numerical
                        or self.tables[table_name].ftypes[col]
                        == feature_types.c_processed_numerical
                    ):
                        tmp_data = self.tables[table_name].df[col].values
                        tmp_data = (tmp_data > np.nanmedian(tmp_data)) * 1

                    if tmp_data is not None:
                        finite = \
                            np.isfinite(pseudo_target) * np.isfinite(tmp_data)
                        finite = np.where(finite)[0]
                        if len(finite) > 20000:
                            selected = np.random.choice(finite, 20000)
                        else:
                            selected = finite
                        if len(np.unique(tmp_data[selected])) == 2:
                            score = metrics.roc_auc_score(
                                tmp_data[selected], pseudo_target[selected])
                            score = np.abs(score - 0.5)
                            if score > surrogate_score:
                                surrogate_score = score
                                surrogate_data = tmp_data
                                surrogate_col = col
                if surrogate_data is not None:
                    print("set_surrogate_data", table_name, surrogate_col,
                          surrogate_score)
                    self.__tables[table_name].set_pseudo_target(
                        surrogate_data)

    def make_adversarial_count(self):
        assert self.__main_table_name is not None
        queue = collections.deque([self.__main_table_name])
        adversarial_true_count_dict = \
            {table_name: None for table_name in self.__tables}
        adversarial_total_count_dict = \
            {table_name: None for table_name in self.__tables}
        adversarial_true_count_dict[self.__main_table_name] \
            = np.isfinite(self.__target).astype(np.int64)
        adversarial_total_count_dict[self.__main_table_name] \
            = np.ones_like(self.__target).astype(np.int64)

        while queue:
            table_name = queue.popleft()
            for relation in self.relations:
                if relation.src == table_name:
                    pass
                elif relation.dst == table_name:
                    relation = relation.inverse
                else:
                    continue
                next_table_name = relation.dst
                prev_id_data = \
                    self.__tables[table_name].df[relation.src_id]
                next_id_data = \
                    self.__tables[next_table_name].df[relation.dst_id]

                if self.__depths[next_table_name] > self.__depths[table_name]:
                    prev_df = pd.DataFrame({
                        "adversarial_true_count":
                            adversarial_true_count_dict[table_name],
                        "adversarial_total_count":
                            adversarial_total_count_dict[table_name],
                        "id": prev_id_data})
                    next_df = pd.DataFrame({
                        "id": next_id_data
                    })
                    if relation.type == relation_types.many_to_one \
                       or relation.type == relation_types.many_to_many:
                        new_adversarial_true_count = next_df.merge(
                            prev_df.groupby("id")["adversarial_true_count"]
                            .sum().reset_index(),
                            how="left"
                        )["adversarial_true_count"].fillna(0).values
                        new_adversarial_total_count = next_df.merge(
                            prev_df.groupby("id")["adversarial_total_count"]
                            .sum().reset_index(),
                            how="left"
                        )["adversarial_total_count"].fillna(0).values
                    else:
                        new_adversarial_true_count = next_df.merge(
                            prev_df,
                            how="left"
                        ).fillna(0)["adversarial_total_count"].values
                        new_adversarial_total_count = next_df.merge(
                            prev_df,
                            how="left"
                        ).fillna(0)["adversarial_total_count"].values
                    if (
                        adversarial_true_count_dict[next_table_name]
                        is not None
                    ):
                        old_mean = \
                            adversarial_true_count_dict[next_table_name] \
                            / adversarial_total_count_dict[next_table_name]
                        new_mean = \
                            new_adversarial_true_count \
                            / new_adversarial_total_count
                        if np.nanstd(new_mean) > np.nanstd(old_mean):
                            adversarial_true_count_dict[next_table_name] \
                                = new_adversarial_true_count
                            adversarial_total_count_dict[next_table_name] \
                                = new_adversarial_total_count
                    else:
                        adversarial_true_count_dict[next_table_name] \
                            = new_adversarial_true_count
                        adversarial_total_count_dict[next_table_name] \
                            = new_adversarial_total_count
                    queue.append(next_table_name)

        for table_name in self.__tables:
            self.__tables[table_name].set_adversarial_count(
                adversarial_true_count_dict[table_name],
                adversarial_total_count_dict[table_name]
            )
