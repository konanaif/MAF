import sys
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.lfr import LFR

from MAF.algorithms.preprocessing.optim_preproc_helpers.data_prepro_function import (
    load_preproc_data_adult,
    load_preproc_data_german,
    load_preproc_data_compas,
)
from MAF.datamodule.dataset import AdultDataset, GermanDataset, CompasDataset
from MAF.metric import common_utils


class LearningFairRepresentation:
    def __init__(self, dataset_name="adult", protected="sex"):
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
        lfr = LFR(
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
            k=10,
            Ax=0.1,
            Ay=1.0,
            Az=2.0,
            verbose=1,
        )
        lfr = lfr.fit(self.dataset_orig_train, maxiter=5000, maxfun=5000)
        transf_data = lfr.transform(self.dataset_orig_train)
        return transf_data

    def baseline_fit(self, dataset, with_weights=True):
        scale = StandardScaler()
        X_train = scale.fit_transform(dataset.features)
        y_train = dataset.labels.ravel()
        lmod = LogisticRegression(class_weight="balanced", solver="liblinear")
        lmod.fit(X_train, y_train)
        return lmod, scale

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
        metrics["protected"] = self.protected
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
    lfr = LearningFairRepresentation(dataset_name="adult", protected="sex")
    metrics_orig, metrics_transf = lfr.run()
    print("Metrics for original data:")
    print(metrics_orig)
    print("\nMetrics for transformed data:")
    print(metrics_transf)
