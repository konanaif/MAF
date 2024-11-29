import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.optim_preproc import (
    OptimPreproc as aifOptimPreproc,
)
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools

from MAF.algorithms.preprocessing.optim_preproc_helpers.data_prepro_function import (
    load_preproc_data_adult,
    load_preproc_data_german,
    load_preproc_data_compas,
    get_optim_options,
)
from MAF.datamodule.dataset import AdultDataset, GermanDataset, CompasDataset
from MAF.metric import common_utils


class OptimPreproc:
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

        self.dataset_orig = dataset_loaders[self.dataset_name]([self.protected])
        self.dataset_orig_train, self.dataset_orig_vt = self.dataset_orig.split(
            [0.7], shuffle=True, seed=1
        )
        self.dataset_orig_valid, self.dataset_orig_test = self.dataset_orig_vt.split(
            [0.5], shuffle=True, seed=1
        )
        self.privileged_groups = [{self.protected: 1}]
        self.unprivileged_groups = [{self.protected: 0}]
        self.optim_options = get_optim_options(
            dataset_name=self.dataset_name, protected=self.protected
        )

    def fit(self):
        op = aifOptimPreproc(
            OptTools,
            self.optim_options,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )

        op.fit(self.dataset_orig_train)

        transf_train = op.transform(self.dataset_orig_train, transform_Y=True)
        transf_train = self.dataset_orig_train.align_datasets(transf_train)

        dataset_orig_test = transf_train.align_datasets(self.dataset_orig_test)

        transf_test = op.transform(dataset_orig_test, transform_Y=True)
        transf_test = dataset_orig_test.align_datasets(transf_test)
        return transf_train, transf_test

    def baseline_fit(self, dataset):
        scale = StandardScaler()
        X_train = scale.fit_transform(dataset.features)
        y_train = self.dataset_orig_train.labels.ravel()

        lmod = LogisticRegression()
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
        return metrics

    def run(self):
        orig_model, orig_scaler = self.baseline_fit(self.dataset_orig_train)
        self.pos_ind = np.where(
            orig_model.classes_ == self.dataset_orig_train.favorable_label
        )[0][0]
        orig_best_class_thresh = self.evaluate(orig_model, orig_scaler)
        orig_test_pred = self.predict(self.dataset_orig_test, orig_model, orig_scaler)
        orig_metrics = self.compute_metrics(orig_test_pred, orig_best_class_thresh)

        transf_train, transf_test = self.fit()
        transf_model, transf_scaler = self.baseline_fit(transf_train)
        transf_test_pred = self.predict(
            self.dataset_orig_test, transf_model, transf_scaler
        )
        transf_metrics = self.compute_metrics(transf_test_pred, orig_best_class_thresh)
        return orig_metrics, transf_metrics


if __name__ == "__main__":
    op = OptimPreproc(dataset_name="adult", protected="race")
    orig_metrics, transf_metrics = op.run()
    print("Metrics for original data:")
    print(orig_metrics)
    print("\nMetrics for transformed data:")
    print(transf_metrics)
