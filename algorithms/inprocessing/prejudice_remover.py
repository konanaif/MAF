import sys, os
import codecs

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

from aif360.algorithms.inprocessing import PrejudiceRemover as aifPrejudiceRemover

from MAF.algorithms.preprocessing.optim_preproc_helpers.data_prepro_function import (
    load_preproc_data_adult,
    load_preproc_data_german,
    load_preproc_data_compas,
)
from MAF.metric import common_utils


class PrejudiceRemover:
    def __init__(self, dataset_name="compas", protected="race"):
        self.dataset_name = dataset_name
        self.protected = protected
        np.random.seed(1)
        self._load_and_preprocess_data()

    def _load_and_preprocess_data(self):
        loaders = {
            "adult": load_preproc_data_adult,
            "german": load_preproc_data_german,
            "compas": load_preproc_data_compas,
        }
        if self.dataset_name not in loaders:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.dataset_orig = loaders[self.dataset_name]([self.protected])
        self.dataset_orig_train, self.dataset_orig_test = self.dataset_orig.split(
            [0.7], shuffle=True, seed=1
        )

        self.privileged_groups = [{self.protected: 1}]
        self.unprivileged_groups = [{self.protected: 0}]

    def fit(self):
        pr = aifPrejudiceRemover()
        pr.fit(self.dataset_orig_train)
        dataset_yhat = pr.predict(self.dataset_orig_test)
        return dataset_yhat

    def baseline_fit(self):
        scale_orig = StandardScaler()
        X_train = scale_orig.fit_transform(self.dataset_orig_train.features)
        y_train = self.dataset_orig_train.labels.ravel()

        X_test = scale_orig.transform(self.dataset_orig_test.features)
        y_test = self.dataset_orig_test.labels.ravel()

        lmod = LogisticRegression()
        lmod.fit(X_train, y_train)
        y_test_pred = lmod.predict(X_test)

        pred_dataset = self.dataset_orig_test.copy(deepcopy=True)
        pred_dataset.labels = y_test_pred
        return pred_dataset

    def compute_metrics(self, dataset):
        return common_utils.compute_metrics(
            self.dataset_orig_test,
            dataset,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )

    def run(self):
        lr_pred = self.baseline_fit()
        pr_pred = self.fit()

        metrics_orig = self.compute_metrics(lr_pred)
        metrics_transform = self.compute_metrics(pr_pred)
        return metrics_orig, metrics_transform


if __name__ == "__main__":
    pr = PrejudiceRemover(dataset_name="adult", protected="sex")
    metrics_orig, metrics_transf = pr.run()
    print("Metrics for original data:")
    print(metrics_orig)
    print("\nMetrics for transformed data:")
    print(metrics_transf)
