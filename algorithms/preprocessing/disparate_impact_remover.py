import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import (
    DisparateImpactRemover as aifDisparateImpactRemover,
)

from MAF.metric import common_utils
from MAF.datamodule.dataset import AdultDataset, GermanDataset, CompasDataset


class DisparateImpactRemover:
    def __init__(
        self,
        dataset_name: str = "adult",
        protected: str = "sex",
        repair_level: float = 1.0,
    ):
        self.dataset_name = dataset_name
        self.protected = protected
        self.repair_level = repair_level
        np.random.seed(1)

    def load_and_preprocess_data(self):
        if self.dataset_name == "adult":
            self.dataset_orig = AdultDataset()
        elif self.dataset_name == "german":
            self.dataset_orig = GermanDataset()
        elif self.dataset_name == "compas":
            self.dataset_orig = CompasDataset()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.dataset_orig_train, self.dataset_orig_vt = self.dataset_orig.split(
            [0.7], shuffle=True, seed=1
        )
        scaler = MinMaxScaler(copy=False)

        train, test = self.dataset_orig.split([0.7], shuffle=True, seed=1)
        train.features = scaler.fit_transform(train.features)
        test.features = scaler.transform(test.features)
        index = train.feature_names.index(self.protected)

        self.privileged_groups = [{self.protected: 1}]
        self.unprivileged_groups = [{self.protected: 0}]

        return train, test, index

    def fit(self, train, test):
        di = aifDisparateImpactRemover(repair_level=self.repair_level)
        transf_train = di.fit_transform(train)
        transf_test = di.fit_transform(test)
        return transf_train, transf_test

    def baseline_fit(self, train, index):
        X_train = np.delete(train.features, index, axis=1)
        y_train = train.labels.ravel()
        lmod = LogisticRegression(class_weight="balanced", solver="liblinear")
        lmod.fit(X_train, y_train)
        return lmod

    def compute_metrics(self, model, test, index):
        test_pred = test.copy()
        X_test_original = np.delete(test.features, index, axis=1)
        test_pred.labels = model.predict(X_test_original)
        metrics = common_utils.compute_metrics(
            test,
            test_pred,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )
        metrics["protected"] = self.protected
        return metrics

    def run(self):
        train, test, index = self.load_and_preprocess_data()
        orig_model = self.baseline_fit(train, index)
        orig_metrics = self.compute_metrics(orig_model, test, index)

        transf_train, transf_test = self.fit(train, test)
        transf_model = self.baseline_fit(transf_train, index)
        transf_metrics = self.compute_metrics(transf_model, test, index)
        return orig_metrics, transf_metrics


if __name__ == "__main__":
    dpir = DisparateImpactRemover(
        dataset_name="adult", protected="sex", repair_level=1.0
    )
    orig_metrics, transf_metrics = dpir.run()
    print("Metrics for original data:")
    print(orig_metrics)
    print("\nMetrics for transformed data:")
    print(transf_metrics)
