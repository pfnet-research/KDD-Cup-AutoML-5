import collections
import threading
import gc
import traceback

import pandas as pd
import numpy as np

from optable.dataset import feature_types
from optable import _core


class Table(object):
    """avalble for only automl data frame
    """
    def __init__(self, df, time_col=None, label_encoders={}, min_time=None):
        self.__df = df
        self.__time_col = time_col
        self.__min_time = min_time
        self.__cache = {}
        self.__pseudo_target = None
        self.__adversarial_true_count = None
        self.__adversarial_total_count = None
        self.__new_data = {}
        if self.__time_col is not None:
            time_data = self.__df[self.__time_col]
            time_data.index = range(len(time_data))

            if min_time is None:
                raise ValueError("min_time is None")

            time_data = time_data - min_time

            time_data = time_data.astype(int).values
            time_data = time_data / 1e9
            second_time_data = time_data.astype(int)
            minute_time_data = second_time_data // 60
            hour_time_data = minute_time_data // 60
            day_time_data = hour_time_data // 24

            second_time_data = second_time_data.astype(np.float32)
            minute_time_data = minute_time_data.astype(np.float32)
            hour_time_data = hour_time_data.astype(np.float32)
            day_time_data = day_time_data.astype(np.float32)
            time_data = time_data.astype(np.float32)

            """
            time_data[time_data < 0] = np.nan
            second_time_data[second_time_data < 0] = np.nan
            minute_time_data[minute_time_data < 0] = np.nan
            hour_time_data[hour_time_data < 0] = np.nan
            day_time_data[day_time_data < 0] = np.nan
            """

            self.__time_data = time_data
            self.__second_time_data = second_time_data
            self.__minute_time_data = minute_time_data
            self.__hour_time_data = hour_time_data
            self.__day_time_data = day_time_data

            self.__sorted_time_index = \
                np.argsort(time_data).astype(np.int32)
        else:
            self.__sorted_time_index = None
        self.__hist_time_data = None
        self.__ftypes = pd.Series(
            self.__automl_df_to_ftypes(), self.__df.dtypes.index)

        self.__label_encoders = label_encoders
        self.__tfidf_vectorizers = {}

        self.__preprocess()

        self.__ftypes = pd.Series(
            self.__automl_df_to_ftypes(), self.__df.dtypes.index)
        self.__nunique = pd.Series(
            [self.__df[col].nunique() for col in self.__df],
            self.__df.dtypes.index)

        self.__set_new_data_lock = threading.Lock()

    @property
    def ftypes(self):
        return self.__ftypes

    @property
    def df(self):
        return self.__df

    @property
    def sorted_time_index(self):
        return self.__sorted_time_index

    @property
    def time_data(self):
        return self.__time_data

    @property
    def second_time_data(self):
        return self.__second_time_data

    @property
    def minute_time_data(self):
        return self.__minute_time_data

    @property
    def hour_time_data(self):
        return self.__hour_time_data

    @property
    def day_time_data(self):
        return self.__day_time_data

    @property
    def has_time(self):
        if self.__time_col is None:
            return False
        return True

    def get_lightgbm_df(self, max_cat_nunique=30):
        columns = []
        col_idx = []
        cat_idx = []
        idx = 0
        lightgbm_feature_types = [
            feature_types.numerical,
            feature_types.categorical,
            feature_types.mc_processed_numerical,
            feature_types.c_processed_numerical,
            feature_types.t_processed_numerical,
            feature_types.n_processed_categorical,
            feature_types.mc_processed_categorical,
            feature_types.c_processed_categorical,
            feature_types.t_processed_categorical,
            feature_types.aggregate_processed_numerical,
            feature_types.aggregate_processed_categorical
        ]
        cat_feature_types = [
            feature_types.categorical,
            feature_types.aggregate_processed_categorical,
            feature_types.n_processed_categorical,
            feature_types.mc_processed_categorical,
            feature_types.c_processed_categorical,
            feature_types.t_processed_categorical,
        ]
        for col_i, col in enumerate(self.__df.columns):
            for ftype in lightgbm_feature_types:
                if col.startswith(ftype.prefix):
                    if ftype in cat_feature_types:
                        if self.__nunique[col] <= max_cat_nunique:
                            cat_idx.append(idx)
                            columns.append(col)
                            col_idx.append(col_i)
                            idx += 1
                    else:
                        columns.append(col)
                        col_idx.append(col_i)
                        idx += 1
                    break
        return self.__df.take(col_idx, axis=1, is_copy=False), cat_idx

    def set_ftypes(self, ftypes):
        if isinstance(ftypes, list):
            self.__ftypes[:] = ftypes
        elif isinstance(ftypes, dict):
            for k in ftypes:
                self.__ftypes[k] = ftypes[k]

    @property
    def nunique(self):
        return self.__nunique

    def set_new_data(self, data, name):
        self.__set_new_data_lock.acquire()
        if name in self.__df.columns or name in self.__new_data:
            print("duplicated", name)

        try:
            self.__new_data[name] = data
        except Exception as e:
            print(name)
            traceback.print_exc()
        finally:
            self.__set_new_data_lock.release()

    @property
    def new_data_size(self):
        return len(self.__new_data)

    def get_new_data(self):
        cat_feature_types = [
            feature_types.categorical,
            feature_types.aggregate_processed_categorical,
            feature_types.n_processed_categorical,
            feature_types.mc_processed_categorical,
            feature_types.c_processed_categorical,
            feature_types.t_processed_categorical,
        ]
        is_cat = [
            feature_types.column_name_to_ftype(key)
            in cat_feature_types for key in self.__new_data]
        return [self.__new_data[key] for key in self.__new_data], is_cat

    def clear_new_data(self):
        self.__new_data = {}

    def confirm_new_data(self):
        new_df = pd.DataFrame(self.__new_data)
        for name in self.__new_data:
            prefix = "{}_".format(name.split("_")[0])
            self.__ftypes[name] = feature_types.prefix_to_ftype[prefix]
            self.__nunique[name] = new_df[name].nunique()
        self.__new_data = {}
        gc.collect()
        self.__df = pd.concat([self.__df, new_df], axis=1)
        gc.collect()

    def test_concat(self, test_df):
        pass

    def __preprocess(self):
        cols_of_each_ftype = self.cols_of_each_ftype

        # numericalでnuniqueが低いものはcategoricalに
        """
        if len(self.__df) > 1000:
            columns = self.__df.columns
            for col in columns:
                if self.__ftypes[col] == feature_types.numerical:
                    if self.__df[col].nunique() <= 10:
                        self.__df["{}{}".format(
                            feature_types.categorical.prefix, col,
                        )] = self.__df[col].astype(str)
                        self.__df.drop(col, axis=1, inplace=True)
                        print("numerical {} change to categorical".format(col))
            self.__ftypes = pd.Series(
                self.__automl_df_to_ftypes(), self.__df.dtypes.index)
        """
        import time

        new_data = {}
        columns = self.__df.columns
        for col in columns:
            start = time.time()
            if self.__ftypes[col] == feature_types.time:
                # Time preprocess
                self.__df[col] = pd.to_datetime(self.__df[col])
                """
                # time numericalize
                if self.__min_time is not None:
                    self.__df["{}numericalized_{}".format(
                        feature_types.t_processed_numerical.prefix, col,
                    )] = ((self.__df[col] - self.__min_time).astype(int)
                          / 1e9).astype(np.float32)
                else:
                    self.__df["{}numericalized_{}".format(
                        feature_types.t_processed_numerical.prefix, col,
                    )] = (self.__df[col].astype(int)
                          / 1e9).astype(np.float32)
                """

                max_min_time_diff = self.__df[col].max() - self.__df[col].min()
                # time hour
                if max_min_time_diff > pd.Timedelta('2 hours'):
                    new_data["{}hour_{}".format(
                        feature_types.t_processed_numerical.prefix, col,
                    )] = self.__df[col].dt.hour.values.astype(np.float32)
                # time year
                if max_min_time_diff > pd.Timedelta('500 days'):
                    new_data["{}year_{}".format(
                        feature_types.t_processed_numerical.prefix, col,
                    )] = self.__df[col].dt.year.values.astype(np.float32)
                # time doy
                if max_min_time_diff > pd.Timedelta('100 days'):
                    new_data["{}doy_{}".format(
                        feature_types.t_processed_numerical.prefix, col,
                    )] = self.__df[col].dt.dayofyear.values.astype(np.float32)
                # time dow
                if max_min_time_diff > pd.Timedelta('2 days'):
                    new_data["{}dow_{}".format(
                        feature_types.t_processed_numerical.prefix, col,
                    )] = self.__df[col].dt.dayofweek.values.astype(np.float32)
                # weekend
                if max_min_time_diff > pd.Timedelta('2 days'):
                    new_data["{}id_weekend_{}".format(
                        feature_types.t_processed_categorical.prefix, col,
                    )] = (self.__df[col].dt.dayofweek >= 5).astype(np.int32)
                # time zone
                if max_min_time_diff > pd.Timedelta('8 hours'):
                    new_data["{}time_zone_{}".format(
                        feature_types.t_processed_categorical.prefix, col,
                    )] = (self.__df[col].dt.hour.values // 6).astype(np.int32)

                self.__df[col] = (
                    (self.__df[col] - self.__min_time).astype(
                        int) / 1e9).astype(np.float32)

            elif self.__ftypes[col] == feature_types.categorical:
                # categorical preprocess
                processing_data = \
                    self.__df[col].fillna("").values
                categorical_manager = \
                    _core.CategoricalManager(processing_data)
                self.set_cache(
                    ("categorical_manager", col),
                    categorical_manager
                )
                if col in self.__label_encoders:
                    self.__df[col] = self.__label_encoders[col].transform(
                        processing_data
                    ).astype(np.int32)
                else:
                    self.__df[col] = categorical_manager.label()

                # frequency encoding
                new_data["{}frequency_{}".format(
                    feature_types.c_processed_numerical.prefix, col
                )] = categorical_manager.frequency()

                if self.has_time:
                    # processing_data = self.__df[col].values
                    """
                    new_data["{}neighbor_nunique_{}".format(
                        feature_types.c_processed_numerical.prefix, col
                    )] = _core.not_temporal_to_many_aggregate(
                        np.roll(processing_data, -1),
                        processing_data, processing_data, 'nunique') \
                        / _core.not_temporal_to_many_aggregate(
                        np.ones_like(processing_data),
                        processing_data, processing_data, 'sum')

                    new_data["{}time_variance_{}".format(
                        feature_types.c_processed_numerical.prefix, col
                    )] = _core.not_temporal_to_many_aggregate(
                        np.arange(len(processing_data)),
                        processing_data, processing_data, 'variance')
                    """
                    """
                    new_data["{}neighbor_count_{}".format(
                        feature_types.c_processed_numerical.prefix, col
                    )] = categorical_manager.sequential_count_encoding(
                        self.__sorted_time_index,
                        len(self.__df) // 30)
                    """

                if categorical_manager.has_null:
                    new_data["{}_is_null_{}".format(
                        feature_types.c_processed_categorical.prefix, col
                    )] = categorical_manager.is_null()

            elif self.__ftypes[col] == feature_types.multi_categorical:
                # multi categorical preprocess
                processing_data = \
                    self.__df[col].fillna("").values
                multi_categorical_manager = \
                    _core.MultiCategoricalManager(processing_data)
                self.set_cache(
                    ("multi_categorical_manager", col),
                    multi_categorical_manager
                )

                counter = collections.Counter(processing_data)
                if np.median([value for key, value
                              in counter.most_common()]) > 1:
                    self.set_cache(
                        ("substance_categorical", col),
                        True
                    )
                    categorical_manager = \
                        _core.CategoricalManager(processing_data)
                    self.set_cache(
                        ("categorical_manager", col),
                        categorical_manager
                    )
                    # frequency encoding
                    """
                    self.__df["{}frequency_{}".format(
                        feature_types.c_processed_numerical.prefix, col
                    )] = categorical_manager.frequency()
                    """
                else:
                    self.set_cache(
                        ("substance_categorical", col),
                        False
                    )

                # length
                # nunique
                # duplicated
                length = multi_categorical_manager.length()
                nunique = multi_categorical_manager.nunique()
                # duplicated = length - nunique
                duplicated = multi_categorical_manager.duplicates()
                new_data["{}length_{}".format(
                    feature_types.mc_processed_numerical.prefix, col
                )] = length
                new_data["{}nunique_{}".format(
                    feature_types.mc_processed_numerical.prefix, col
                )] = nunique
                new_data["{}duplicated_{}".format(
                    feature_types.mc_processed_numerical.prefix, col
                )] = duplicated

                # max_count
                # min_count
                new_data["{}max_count_{}".format(
                    feature_types.mc_processed_numerical.prefix, col
                )] = multi_categorical_manager.max_count()
                new_data["{}min_count_{}".format(
                    feature_types.mc_processed_numerical.prefix, col
                )] = multi_categorical_manager.min_count()

                # mode
                new_data["{}mode_{}".format(
                    feature_types.mc_processed_categorical.prefix, col
                )] = multi_categorical_manager.mode().astype(int)

                # max_tfidf_words
                """
                new_data["{}max_tfidf_words_{}".format(
                    feature_types.mc_processed_categorical.prefix, col
                )] = multi_categorical_manager.max_tfidf_words().astype(int)
                """

                # hashed tf-idf
                """
                multi_categorical_manager.calculate_hashed_tfidf(10)
                for vectorized_idx in range(10):
                    self.__df["{}hashed_tfidf_{}_{}".format(
                        feature_types.mc_processed_numerical.prefix, col,
                        vectorized_idx,
                    )] = multi_categorical_manager.get_hashed_tfidf(
                        vectorized_idx)
                """

                # tf-idf vectorize
                """
                for vectorized_idx in range(10):
                    new_data["{}tfidf_{}_{}".format(
                        feature_types.mc_processed_numerical.prefix, col,
                        vectorized_idx,
                    )] = multi_categorical_manager.tfidf(vectorized_idx)
                """
                for vectorized_idx in range(10):
                    new_data["{}count_{}_{}".format(
                        feature_types.mc_processed_numerical.prefix, col,
                        vectorized_idx,
                    )] = multi_categorical_manager.count(vectorized_idx)

                # svd
                """
                svd_values = \
                    multi_categorical_manager.truncated_svd(10, False, False)
                """
                """
                tfidf_values = multi_categorical_manager.get_tfidf_matrix()
                from sklearn.decomposition import TruncatedSVD
                svd_values = TruncatedSVD(
                    n_components=10, random_state=10, algorithm='arpack',
                    n_iter=5).fit_transform(tfidf_values)
                """
                """
                for svd_idx in range(10):
                    new_data["{}svd_{}_{}".format(
                        feature_types.mc_processed_numerical.prefix, col,
                        svd_idx,
                    )] = svd_values[:, svd_idx]
                """
                self.__df.drop(col, axis=1, inplace=True)
                del processing_data
                self.__df[col] = ""
                gc.collect()

            elif self.__ftypes[col] == feature_types.numerical:
                # numerical preprocess
                if pd.isnull(self.__df[col]).all():
                    continue

                if (
                    len(np.unique(self.__df[col].values[
                        np.isfinite(self.__df[col].values)]
                    )) == 1
                ):
                    self.__df.drop(col, axis=1, inplace=True)
                    continue
                """
                mode, mode_count = \
                    collections.Counter(
                        self.__df[col].values[
                            np.isfinite(self.__df[col].values)]
                        ).most_common(1)[0]
                mode_freq = mode_count / len(self.__df)
                if mode_freq >= 1:
                    self.__df.drop(col, axis=1, inplace=True)
                    continue

                if mode_freq > 0.1:
                    new_data["{}_is_mode_{}".format(
                        feature_types.n_processed_categorical.prefix, col
                    )] = (self.__df[col].values == mode).astype(np.int32)
                """
                if pd.isnull(self.__df[col]).any():
                    new_data["{}_is_null_{}".format(
                        feature_types.n_processed_categorical.prefix, col
                    )] = pd.isnull(self.__df[col]).astype(np.int32)
                self.__df[col] = self.__df[col].astype(np.float32)
            
            print(col, time.time() - start)

        new_data = pd.DataFrame(new_data)
        self.__df = pd.concat([self.__df, new_data], axis=1)

    def __automl_df_to_ftypes(self):
        ftypes = {}
        for col in self.__df.columns:
            prefix = "{}_".format(col.split("_")[0])
            ftypes[col] = feature_types.prefix_to_ftype[prefix]
        return ftypes

    @property
    def cols_of_each_ftype(self):
        cols_of_each_ftype = {ftype: [] for ftype in feature_types.ftypes}
        for col in self.__df:
            cols_of_each_ftype[self.__ftypes[col]].append(col)
        return cols_of_each_ftype

    def has_cache(self, key):
        return key in self.__cache

    def get_cache(self, key):
        if self.has_cache(key):
            return self.__cache[key]
        else:
            return None

    def set_cache(self, key, value):
        self.__cache[key] = value

    @property
    def cache_keys(self):
        return self.__cache.keys()

    def clear_cache(self):
        self.__cache = {}
        gc.collect()
        _core.malloc_trim(0)

    @property
    def pseudo_target(self):
        return self.__pseudo_target

    @property
    def has_pseudo_target(self):
        return (self.__pseudo_target is not None)

    def set_pseudo_target(self, pseudo_target):
        self.__pseudo_target = pseudo_target

    @property
    def has_adversarial_count(self):
        return (self.__adversarial_true_count is not None)

    @property
    def adversarial_true_count(self):
        return self.__adversarial_true_count

    @property
    def adversarial_total_count(self):
        return self.__adversarial_total_count

    def set_adversarial_count(self, true_count, total_count):
        self.__adversarial_true_count = true_count
        self.__adversarial_total_count = total_count

    @property
    def has_hist_time_data(self):
        return self.__hist_time_data is not None

    @property
    def hist_time_data(self):
        return self.__hist_time_data

    def set_interval_for_hist(self, interval):
        hist_time_data = self.__time_data // interval
        hist_time_data = hist_time_data.astype(np.float32)
        hist_time_data[hist_time_data < 0] = np.nan
        self.__hist_time_data = hist_time_data
