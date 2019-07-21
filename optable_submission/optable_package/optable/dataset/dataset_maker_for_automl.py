import collections

import numpy as np
import pandas as pd
from sklearn import preprocessing  # LabelEncoder

import optable.dataset.dataset as dataset_mod
import optable.dataset.table as table_mod
import optable.dataset.relation as relation_mod
from optable.dataset import feature_types
from optable.dataset import relation_types
from optable.dataset import union_find_node
from optable.processing import label_encoder_for_multi_fit
from optable import CONSTANT
from optable import _core


class DatasetMakerForAutoML(object):
    def __init__(self):
        pass

    def make(self, Xs, X_test, y, info):
        dataset = dataset_mod.Dataset()

        main_train_df = Xs[CONSTANT.MAIN_TABLE_NAME]
        main_test_df = X_test

        main_df = pd.concat([main_train_df, main_test_df], axis=0)
        main_df.index = range(len(main_df))
        min_time = main_df[info['time_col']].min()

        target = np.concatenate([y, np.full(len(main_test_df), np.nan)])
        dataset.set_target(target)
        dataset.set_train_size(len(main_train_df))

        # TODO: 難解になりすぎた・・・refactoring
        relation_col_ids = self.get_relation_col_ids(info)
        label_encoders = {
            group_id: _core.LabelEncoderForMultiFit()
            for group_id in relation_col_ids}
        for group_id in relation_col_ids:
            for table_name, col in relation_col_ids[group_id]:
                if table_name == 'main':
                    label_encoders[group_id].fit(main_df.loc[:, col].fillna("").values)
                else:
                    label_encoders[group_id].fit(Xs[table_name].loc[:, col].fillna("").values)

        label_encoders_of_main = {}
        for group_id_ in relation_col_ids:
            for table_name_, col_ in relation_col_ids[group_id_]:
                if table_name_ == 'main':
                    label_encoders_of_main[col_] = label_encoders[group_id_]
        main_table = table_mod.Table(
            main_df, info['time_col'], label_encoders=label_encoders_of_main,
            min_time=min_time)
        dataset.set_table(
            CONSTANT.MAIN_TABLE_NAME, main_table, is_main_table=True)

        for table_name in Xs:
            if table_name != CONSTANT.MAIN_TABLE_NAME:
                label_encoders_of_table = {}
                for group_id_ in relation_col_ids:
                    for table_name_, col_ in relation_col_ids[group_id_]:
                        if table_name_ == table_name:
                            label_encoders_of_table[col_] = \
                                label_encoders[group_id_]
                if info['time_col'] in Xs[table_name]:
                    table = table_mod.Table(
                        Xs[table_name], info['time_col'],
                        label_encoders=label_encoders_of_table,
                        min_time=min_time)
                else:
                    table = table_mod.Table(
                        Xs[table_name],
                        label_encoders=label_encoders_of_table,
                        min_time=min_time)
                dataset.set_table(table_name, table)

        interval_for_hist = (
            np.nanmax(dataset.tables['main'].time_data)
            - np.nanmin(dataset.tables['main'].time_data)
            ) / (int(np.sqrt(len(dataset.tables['main'].df))))
        for table_name in Xs:
            if dataset.tables[table_name].has_time:
                dataset.tables[table_name].set_interval_for_hist(
                    interval_for_hist)

        for rel in info['relations']:
            for key in rel['key']:
                src, dst = rel['table_A'], rel['table_B']
                dataset.set_relation(
                    relation_mod.Relation(src, dst, key, key, rel['type']))

        # self many-to-many
        for table_name in Xs:
            for columns in dataset.tables[table_name].df.columns:
                if (
                    dataset.tables[table_name].ftypes[columns]
                    == feature_types.categorical
                ):
                    if (
                        dataset.tables[table_name].nunique[columns]
                        > np.sqrt(len(dataset.tables[table_name].df))
                        and dataset.tables[table_name].nunique[columns]
                        < len(dataset.tables[table_name].df) / 2
                        and dataset.tables[table_name].ftypes[columns]
                        == feature_types.categorical
                    ):
                        dataset.set_relation(
                            relation_mod.Relation(
                                table_name, table_name, columns, columns,
                                'many_to_many'))

        dataset.depth_search()
        dataset.make_pseudo_target()
        # dataset.make_adversarial_count()

        self.make_substance_to_one(dataset)

        return dataset

    def get_relation_col_ids(self, info):
        relation_cols_set = set([])
        relation_union_find_nodes = {}
        now_group_id = 0
        relation_col_ids = {}
        for rel in info['relations']:
            for key in rel['key']:
                src, dst = rel['table_A'], rel['table_B']
                if (src, key) not in relation_cols_set:
                    relation_cols_set.add((src, key))
                    relation_union_find_nodes[(src, key)] = \
                        union_find_node.UnionFindNode(
                            group_id=now_group_id,
                            value=(src, key),
                        )
                    now_group_id += 1
                if (dst, key) not in relation_cols_set:
                    relation_cols_set.add((dst, key))
                    relation_union_find_nodes[(dst, key)] = \
                        union_find_node.UnionFindNode(
                            group_id=now_group_id,
                            value=(dst, key),
                        )
                    now_group_id += 1
                relation_union_find_nodes[(src, key)].unite(
                    relation_union_find_nodes[(dst, key)]
                )
        for key in relation_union_find_nodes:
            group_id = relation_union_find_nodes[key].root().group_id
            if group_id not in relation_col_ids:
                relation_col_ids[group_id] = []
            relation_col_ids[group_id].append(key)
        return relation_col_ids

    def make_substance_to_one(self, dataset):
        for relation_ in dataset.relations:
            for relation in [relation_, relation_.inverse]:
                if relation.type == relation_types.many_to_many \
                   or relation.type == relation_types.one_to_many:
                    if dataset.tables[relation.dst].has_cache(
                        ("substance_to_one_relation", relation.dst_id)
                    ):
                        continue
                    print(relation)
                    processing_data = \
                        dataset.tables[relation.dst].df[relation.dst_id].values
                    processing_data = \
                        processing_data[np.isfinite(processing_data)]
                    counter = collections.Counter(processing_data)
                    substance_to_one_relation = \
                        np.median(
                            [freq for key, freq in
                             counter.most_common()]
                        ) == 1
                    nunique_df = []
                    for key, freq in counter.most_common(10):
                        nunique_df.append(
                            dataset.tables[relation.dst]
                            .df[dataset.tables[relation.dst]
                                .df[relation.dst_id] == key]
                            .nunique())
                    substance_to_one_columns = (
                        pd.DataFrame(nunique_df) <= 1).all(axis=0)
                    dataset.tables[relation.dst].set_cache(
                        ("substance_to_one_relation", relation.dst_id),
                        substance_to_one_relation
                    )
                    dataset.tables[relation.dst].set_cache(
                        ("substance_to_one_columns", relation.dst_id),
                        substance_to_one_columns
                    )
