import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing as aifReweighing
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import (
    load_preproc_data_adult,
    load_preproc_data_german,
    load_preproc_data_compas,
)

from MAF.algorithms.preprocessing.optim_preproc_helpers.data_prepro_function import (
    load_preproc_data_adult,
    load_preproc_data_german,
    load_preproc_data_compas,
)
from MAF.datamodule.dataset import AdultDataset, GermanDataset, CompasDataset
from MAF.metric import common_utils


class Reweighing:
    def __init__(self, dataset_name: str = "adult", protected: str = "sex"):
        self.dataset_name = dataset_name
        self.protected = protected
        np.random.seed(1)
        self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        dataset_loaders = {
            "adult": load_preproc_data_adult,
            "german": load_preproc_data_german,
            "compas": load_preproc_data_compas,
        }

        if self.dataset_name not in dataset_loaders:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.dataset_orig = dataset_loaders[self.dataset_name]()
        self.dataset_orig_train, self.dataset_orig_vt = self.dataset_orig.split(
            [0.7], shuffle=True, seed=1
        )
        self.dataset_orig_valid, self.dataset_orig_test = self.dataset_orig_vt.split(
            [0.5], shuffle=True, seed=1
        )

        self.privileged_groups = [{self.protected: 1}]
        self.unprivileged_groups = [{self.protected: 0}]

    def fit(self):
        rw = aifReweighing(
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )
        rw.fit(self.dataset_orig_train)
        transf_data = rw.transform(self.dataset_orig_train)
        return transf_data

    def baseline_fit(self, dataset):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(dataset.features)
        y_train = dataset.labels.ravel()
        lmod = LogisticRegression()
        lmod.fit(X_train, y_train, sample_weight=dataset.instance_weights)
        return lmod, scaler

    def predict(self, dataset, model, scaler):
        dataset_pred = dataset.copy(deepcopy=True)
        X = scaler.transform(dataset_pred.features)
        dataset_pred.scores = model.predict_proba(X)[:, self.pos_ind].reshape(-1, 1)
        return dataset_pred

    def evaluate(self, model, scaler):
        dataset_orig_valid_pred = self.predict(self.dataset_orig_valid, model, scaler)
        num_thresh = 100
        ba_arr = np.zeros(num_thresh)
        class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
        for idx, class_thresh in enumerate(class_thresh_arr):

            fav_inds = dataset_orig_valid_pred.scores > class_thresh
            dataset_orig_valid_pred.labels[
                fav_inds
            ] = dataset_orig_valid_pred.favorable_label
            dataset_orig_valid_pred.labels[
                ~fav_inds
            ] = dataset_orig_valid_pred.unfavorable_label

            classified_metric_orig_valid = ClassificationMetric(
                self.dataset_orig_valid,
                dataset_orig_valid_pred,
                unprivileged_groups=self.unprivileged_groups,
                privileged_groups=self.privileged_groups,
            )

            ba_arr[idx] = 0.5 * (
                classified_metric_orig_valid.true_positive_rate()
                + classified_metric_orig_valid.true_negative_rate()
            )

        best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
        best_class_thresh = class_thresh_arr[best_ind]

        print("Best balanced accuracy (no reweighing) = %.4f" % np.max(ba_arr))
        print(
            "Optimal classification threshold (no reweighing) = %.4f"
            % best_class_thresh
        )
        return best_class_thresh

    def compute_metrics(self, dataset, best_class_thresh):
        fav_inds = dataset.scores > best_class_thresh
        dataset.labels[fav_inds] = dataset.favorable_label
        dataset.labels[~fav_inds] = dataset.unfavorable_label

        metrics = common_utils.compute_metrics(
            self.dataset_orig_test,
            dataset,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )
        return metrics

    def run(self):
        orig_model, orig_scaler = self.baseline_fit(self.dataset_orig_train)
        self.pos_ind = np.where(
            orig_model.classes_ == self.dataset_orig_train.favorable_label
        )[0][0]
        orig_best_class_thresh = self.evaluate(orig_model, orig_scaler)
        orig_test_pred = self.predict(self.dataset_orig_test, orig_model, orig_scaler)
        orig_metrics = self.compute_metrics(orig_test_pred, orig_best_class_thresh)

        transf_data = self.fit()
        transf_model, transf_scaler = self.baseline_fit(transf_data)
        transf_test_pred = self.predict(
            self.dataset_orig_test, transf_model, transf_scaler
        )
        transf_metrics = self.compute_metrics(transf_test_pred, orig_best_class_thresh)
        return orig_metrics, transf_metrics


if __name__ == "__main__":
    rw = Reweighing(dataset_name="adult", protected="sex")
    orig_metrics, transf_metrics = rw.run()
    print("Metrics for original data:")
    print(orig_metrics)
    print("\nMetrics for transformed data:")
    print(transf_metrics)
