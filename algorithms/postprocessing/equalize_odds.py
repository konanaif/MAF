import os, sys
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
from aif360.metrics import ClassificationMetric
from MAF.datamodule.dataset import AdultDataset, GermanDataset, CompasDataset
from MAF.algorithms.preprocessing.optim_preproc_helpers.data_prepro_function import (
    load_preproc_data_adult,
    load_preproc_data_german,
    load_preproc_data_compas,
)
from MAF.metric import common_utils


class EqOdds:
    def __init__(self, dataset_name: str = "adult", protected: str = "sex") -> None:
        self.best_class_thres = 0.0
        self.dataset_name = dataset_name
        self.protected = protected
        self.randseed = 1
        np.random.seed(1)
        self.load_and_preprocess_data()

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
        self.dataset_orig_valid, self.dataset_orig_test = self.dataset_orig_vt.split(
            [0.5], shuffle=True, seed=1
        )

        self.privileged_groups = [{self.protected: 1}]
        self.unprivileged_groups = [{self.protected: 0}]

    def preprocess_data(self, dataset, scaler, model, class_thresh=0.5):
        X = scaler.transform(dataset.features)
        y_pred_prob = model.predict_proba(X)[:, 1]

        dataset_pred = dataset.copy(deepcopy=True)
        dataset_pred.scores = y_pred_prob.reshape(-1, 1)

        y_pred = np.zeros_like(dataset_pred.labels)
        y_pred[y_pred_prob >= class_thresh] = dataset_pred.favorable_label
        y_pred[~(y_pred_prob >= class_thresh)] = dataset_pred.unfavorable_label
        dataset_pred.labels = y_pred

        return dataset_pred

    def baseline_fit(self):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(self.dataset_orig_train.features)
        y_train = self.dataset_orig_train.labels.ravel()

        lmod = LogisticRegression()
        lmod.fit(X_train, y_train)

        orig_train_pred = self.preprocess_data(self.dataset_orig_train, scaler, lmod)
        orig_valid_pred = self.preprocess_data(self.dataset_orig_valid, scaler, lmod)
        orig_test_pred = self.preprocess_data(self.dataset_orig_test, scaler, lmod)

        return orig_train_pred, orig_valid_pred, orig_test_pred

    def fit(self, orig_valid_pred, orig_test_pred):
        eo = EqOddsPostprocessing(
            privileged_groups=self.privileged_groups,
            unprivileged_groups=self.unprivileged_groups,
            seed=self.randseed,
        )
        eo = eo.fit(self.dataset_orig_valid, orig_valid_pred)

        transf_valid_pred = eo.predict(orig_valid_pred)
        transf_test_pred = eo.predict(orig_test_pred)
        return transf_valid_pred, transf_test_pred

    def compute_metrics(self, pred):
        metrics = common_utils.compute_metrics(
            self.dataset_orig_test,
            pred,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )
        return metrics

    def run(self):
        orig_train_pred, orig_valid_pred, orig_test_pred = self.baseline_fit()
        metrics_orig = self.compute_metrics(orig_test_pred)
        transf_valid_pred, transf_test_pred = self.fit(orig_valid_pred, orig_test_pred)
        metrics_transf = self.compute_metrics(transf_test_pred)
        return metrics_orig, metrics_transf


if __name__ == "__main__":
    eo = EqOdds(dataset_name="adult", protected="sex")
    metrics_orig, metrics_transf = eo.run()
    print("Metrics for original data:")
    print(metrics_orig)
    print("\nMetrics for transformed data:")
    print(metrics_transf)
