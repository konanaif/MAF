import sys, os
import numpy as np
from sklearn.preprocessing import MaxAbsScaler

from aif360.algorithms.inprocessing import MetaFairClassifier as aifMetaFairClassifier
from MAF.algorithms.preprocessing.optim_preproc_helpers.data_prepro_function import (
    load_preproc_data_adult,
    load_preproc_data_german,
    load_preproc_data_compas,
)
from MAF.metric import common_utils


class MetaFairClassifier:
    def __init__(self, dataset_name="adult", protected="race", fairness_type="fdr"):
        self.dataset_name = dataset_name
        self.protected = protected
        self.fairness_type = fairness_type
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

        scaler = MaxAbsScaler()
        self.dataset_orig_train.features = scaler.fit_transform(
            self.dataset_orig_train.features
        )
        self.dataset_orig_test.features = scaler.transform(
            self.dataset_orig_test.features
        )

        self.privileged_groups = [{self.protected: 1}]
        self.unprivileged_groups = [{self.protected: 0}]

    def fit(self, tau):
        mfc = aifMetaFairClassifier(
            tau=tau, sensitive_attr=self.protected, type=self.fairness_type
        )
        mfc.fit(self.dataset_orig_train)
        test_pred = mfc.predict(self.dataset_orig_test)
        return test_pred

    def compute_metrics(self, dataset):
        return common_utils.compute_metrics(
            self.dataset_orig_test,
            dataset,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )

    def run(self):
        orig_test_pred = self.fit(tau=0)
        metrics_orig = self.compute_metrics(orig_test_pred)

        transf_test_pred = self.fit(tau=0.7)
        metrics_transform = self.compute_metrics(transf_test_pred)

        return metrics_orig, metrics_transform


if __name__ == "__main__":
    mfc = MetaFairClassifier(dataset_name="compas", protected="sex")
    metrics_orig, metrics_transf = mfc.run()
    print("Metrics for original data:")
    print(metrics_orig)
    print("\nMetrics for transformed data:")
    print(metrics_transf)
