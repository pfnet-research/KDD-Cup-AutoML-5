import gc

import numpy as np
import pandas as pd
from sklearn import metrics

from optable.learning import learner
from optable import _core


class AdversarialAUCSelector(learner.Learner):
    def __init__(self, train_size, threshold=0.2, max_ratio=0.1):
        self.train_size = train_size
        self.threshold = threshold
        self.max_ratio = max_ratio

    def fit(self, X, y, sorted_time_index, categorical_feature):
        test_size = len(X) - self.train_size
        train_size = self.train_size
        adversarial_label = np.concatenate([
            np.zeros(train_size).astype(int),
            np.ones(test_size).astype(int)])
        if len(adversarial_label) > 2000:
            choiced, choiced_adversarial_label = self.stratified_data_sample(
                np.arange(len(adversarial_label)), adversarial_label, 20000)
        else:
            choiced = np.arange(len(adversarial_label))
            choiced_adversarial_label = adversarial_label
        auc_by_feature = []
        for col_idx in range(X.shape[1]):
            if col_idx in categorical_feature:
                estimate = _core.TargetEncoder().not_loo_encode(
                    adversarial_label, X[:, col_idx].astype(np.int32), 1
                )
                choiced_estimate = estimate[choiced]
                try:
                    auc_by_feature.append(metrics.roc_auc_score(
                        choiced_adversarial_label[
                            np.isfinite(choiced_estimate)],
                        choiced_estimate[np.isfinite(choiced_estimate)]))
                except ValueError as e:
                    auc_by_feature.append(0.5)
            else:
                data = X[:, col_idx]
                choiced_data = data[choiced]
                try:
                    auc_by_feature.append(metrics.roc_auc_score(
                        choiced_adversarial_label[np.isfinite(choiced_data)],
                        choiced_data[np.isfinite(choiced_data)]))
                except ValueError as e:
                    auc_by_feature.append(0.5)
        auc_by_feature = np.array(auc_by_feature)
        selected = np.abs(auc_by_feature - 0.5) < self.threshold
        if selected.mean() < self.max_ratio:
            selected = (
                np.abs(auc_by_feature - 0.5)
                < np.percentile(np.abs(auc_by_feature - 0.5),
                                100 - 100 * self.max_ratio))

        if self.train_size > 30000:
            train_choiced = np.random.choice(
                self.train_size, 30000, replace=False)
        else:
            train_choiced = np.arange(self.train_size)

        if test_size > 30000:
            test_choiced = np.random.choice(
                test_size, 30000, replace=False)
        else:
            test_choiced = np.arange(test_size)
        test_choiced += self.train_size

        numeric_feature = [col_idx for col_idx in range(X.shape[1])
                           if col_idx not in categorical_feature]
        std_score = \
            np.nanstd(X[train_choiced][:, numeric_feature], axis=0) \
            / np.nanstd(X[test_choiced][:, numeric_feature], axis=0)
        selected[np.array(numeric_feature)[np.isnan(std_score)]] = False
        selected[np.array(numeric_feature)[std_score > 3]] = False
        selected[np.array(numeric_feature)[std_score < 1/3]] = False

        nunique_score = \
            pd.DataFrame(X[train_choiced][:, categorical_feature]).nunique() \
            / pd.DataFrame(X[test_choiced][:, categorical_feature]).nunique()
        nunique_score = nunique_score.values
        selected[np.array(categorical_feature)[
            np.isnan(nunique_score)]] = False
        selected[np.array(categorical_feature)[nunique_score > 3]] = False
        selected[np.array(categorical_feature)[nunique_score < 1/3]] = False
        return selected
