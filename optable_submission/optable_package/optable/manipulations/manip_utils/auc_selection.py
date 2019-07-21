import numpy as np
from sklearn import model_selection, metrics


def stratified_data_sample(X, y, n=30000):
    if len(X) > n:
        X_sample, _, y_sample, _ = model_selection.train_test_split(
            X, y, train_size=n, stratify=y, random_state=1)
    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample


def numerical_adversarial_auc_select(dataset, data, threshold=0.2):
    adversarial_label = np.concatenate([
        np.zeros(dataset.train_size).astype(int),
        np.ones(dataset.test_size).astype(int)])
    choiced, choiced_adversarial_label = stratified_data_sample(
        np.arange(len(adversarial_label)), adversarial_label, 20000)
    choiced_data = data[choiced]
    choice_isfinite = np.isfinite(choiced_data)

    try:
        auc = metrics.roc_auc_score(
            choiced_adversarial_label[choice_isfinite],
            choiced_data[choice_isfinite])
    except ValueError as e:
        return False

    if np.abs(auc - 0.5) > threshold:
        return False
    return True
