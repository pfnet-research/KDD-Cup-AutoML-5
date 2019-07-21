import gc
import os

import numpy as np
import pandas as pd


DEVELOP_MODE = True

if DEVELOP_MODE:
    import resource
    # 15.5 GB
    memory_limit = 15.5 * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_RSS, (memory_limit, memory_limit))
else:
    os.system("conda install -y cmake")
    os.system("apt-get update")
    os.system("unzip {}/optable.zip".format(os.path.dirname(__file__)))
    os.system("cd optable_package/deb"
              "&& dpkg -i pkg-config_0.29-4+b1_amd64.deb")
    os.system("cd optable_package/deb"
              "&& dpkg -i gcc-8-base_8.3.0-7_amd64.deb")
    os.system("cd optable_package/deb"
              "&& dpkg -i libstdc++6_8.3.0-7_amd64.deb")
    os.system("cd optable_package/deb"
              "&& dpkg -i libboost1.62-dev_1.62.0+dfsg-10+b1_amd64.deb")
    os.system("cd optable_package/deb"
              "&& dpkg -i libboost-math1.62.0_1.62.0+dfsg-10+b1_amd64.deb")
    os.system("cd optable_package/deb"
              "&& dpkg -i libboost-math1.62-dev_1.62.0+dfsg-10+b1_amd64.deb")
    os.system("cd optable_package/deb"
              "&& dpkg -i libeigen3-dev_3.3.7-1_all.deb")
    # os.system("apt-get install -y libboost-dev")
    # os.system("apt-get update")
    # os.system("apt-get install -y libboost-math-dev")
    # os.system("apt-get update")
    # os.system("apt-get install -y python3.6-dev")
    # os.system("apt-get update")
    # os.system("apt-get install -y libeigen3-dev")
    os.system("echo \"\" > /usr/share/dpkg/no-pie-link.specs")
    os.system("echo \"\" > /usr/share/dpkg/no-pie-compile.specs")
    os.system("pip3 install pybind11==2.3.0")
    os.system("cd optable_package && pip3 install .")
    os.system("pip3 install lightgbm==2.2.3")
    # os.system("cd optable_package/PFLightGBM && mkdir build && cd build "
    #           "&& cmake .. && make -j4 "
    #           "&& cd ../python-package && python3 setup.py install")
    os.system("pip3 install timeout-decorator==0.4.1")
    os.system("pip3 install optable_package/optuna.zip")
    os.system("pip3 install psutil==5.6.3")


import optable  # NOQA
import timeout_decorator  # NOQA


class Model:
    def __init__(self, info):
        self.info = info
        self.Xs = None
        self.y = None
        self.timer = optable.Timer(
            self.info["time_budget"], self.info["time_budget"])

    def fit(self, Xs, y, time_remain):
        self.Xs = Xs
        self.y = y
        self.timer.update_time_remain(time_remain)
        self.timer.print_memory_usage()

    def _predict(self, X_test, time_remain):
        self.timer.update_time_remain(time_remain)
        print("predict time remain:", time_remain)
        self.timer.print_memory_usage()

        # make dataset
        self.timer.print("start making dataset")
        train_size = len(self.Xs[optable.CONSTANT.MAIN_TABLE_NAME])
        test_size = len(X_test)
        dataset_maker = optable.DatasetMakerForAutoML()
        dataset = dataset_maker.make(self.Xs, X_test, self.y, self.info)
        self.timer.print("finished making dataset")

        self.timer.print_memory_usage()

        # synthesis
        self.timer.print("start synthesis")
        synthesizer = \
            optable.Synthesizer(
                dataset, timer=self.timer, priority_perturbation=0.0)
        max_feature_size = min([int(4.5e8 / train_size),
                                int(8e8 / (train_size + test_size)),
                                int(np.log10(train_size) * 70 + 200)])

        synthesizer.synthesis(
            max_feature_size, self.timer.ratio_remain_time(0.3))
        self.timer.print("finished synthesis")
        self.timer.print_memory_usage()

        dataset.clear_cache_of_table()
        gc.collect()
        optable._core.malloc_trim(0)
        self.timer.print_memory_usage()

        main_table = dataset.tables[
            optable.CONSTANT.MAIN_TABLE_NAME]
        main_table.confirm_new_data()
        del dataset_maker, dataset
        del synthesizer
        optable._core.malloc_trim(0)
        gc.collect()
        self.timer.print_memory_usage()

        max_cat_nunique = min([
            int(np.power(self.y.values.sum(), 0.3)),
            int(np.power((1 - self.y.values).sum(), 0.3))])
        feature, cat_idx = main_table.get_lightgbm_df(max_cat_nunique)
        main_sorted_time_index = main_table.sorted_time_index
        # num_idx = [col_idx for col_idx in range(feature.shape[1])
        #            if col_idx not in cat_idx]

        del main_table
        optable._core.malloc_trim(0)
        gc.collect()

        self.timer.print_memory_usage()

        train_feature = feature.iloc[:train_size]
        test_feature = feature.iloc[train_size:]
        train_sorted_time_index = \
            main_sorted_time_index[main_sorted_time_index < train_size]
        self.timer.print("train_feature shape {}".format(train_feature.shape))
        self.timer.print("test_feature shape {}".format(test_feature.shape))

        # feature selection
        self.timer.print("start feature selection")
        selector = optable.AdversarialAUCSelector(
            train_size, threshold=0.3, max_ratio=0.5)
        selected = selector.fit(
            feature.values, self.y.values,
            main_sorted_time_index, cat_idx)

        del feature, selector
        gc.collect()
        self.timer.print("finished feature selection")
        self.timer.print("{} / {} is selected".format(
            selected.sum(), len(selected)))
        drop_columns = train_feature.columns[np.logical_not(selected)]
        self.timer.print("droped {}".format(drop_columns))
        self.timer.print_memory_usage()

        train_feature = train_feature.iloc[:, selected]
        gc.collect()
        test_feature = test_feature.iloc[:, selected]
        gc.collect()

        train_feature = train_feature.values
        gc.collect()
        train_feature = train_feature.astype(dtype=np.float32)
        gc.collect()

        test_feature = test_feature.values
        gc.collect()
        test_feature = test_feature.astype(dtype=np.float32)
        gc.collect()

        is_cat = np.zeros(len(selected)).astype(bool)
        is_cat[cat_idx] = True
        cat_idx = np.where(is_cat[selected])[0].tolist()

        gc.collect()
        self.timer.print_memory_usage()

        # hyper parameter optimization
        self.timer.print("start hyper parameter optimization")

        """
        if np.abs(0.5 - self.y.values.mean()) > 0.49:
            mode = "logloss"
        else:
            mode = "auc"
        """
        mode = "auc"

        ohp_searcher = optable.OptunaHyperParamsSearcher(
            mode=mode, timer=self.timer)
        # ohp_searcher = optable.TimeSplitHyperParamsSearcher(
        #     mode=mode, timer=self.timer)
        params_list = ohp_searcher.fit(
            train_feature, self.y.values,
            train_sorted_time_index, cat_idx)
        self.timer.print("finished hyper parameter optimization")

        del ohp_searcher
        del self.Xs, X_test
        gc.collect()
        self.timer.print_memory_usage()
        self.timer.print("start model training and prediction")
        lgb_model = optable.LightGBMCVAsWellAsPossible(
            params_list, self.timer, n_split=10, max_model=10)
        # lgb_model = optable.TimeSplitLightGBMCV(
        #     params_list, self.timer, n_split=5, max_model=10)

        gc.collect()
        self.timer.print_memory_usage()

        predicted = lgb_model.fit_predict(
            train_feature, self.y.values, test_feature, cat_idx)
        self.timer.print("finished model training and prediction")

        del train_feature, test_feature, lgb_model, self.y
        gc.collect()
        self.timer.print_memory_usage()
        return pd.Series(predicted)

    def predict(self, X_test, time_remain):
        ret = self._predict(X_test, time_remain)
        gc.collect()
        optable._core.malloc_trim(0)
        gc.collect()
        self.timer.print_memory_usage()
        return ret
