import os, sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from aif360.metrics import ClassificationMetric
from aif360.algorithms.postprocessing.reject_option_classification import (
    RejectOptionClassification,
)

from MAF.datamodule.dataset import AdultDataset, GermanDataset, CompasDataset
from MAF.algorithms.preprocessing.optim_preproc_helpers.data_prepro_function import (
    load_preproc_data_adult,
    load_preproc_data_german,
    load_preproc_data_compas,
)
from MAF.metric import common_utils


class RejectOptionClassifier:
    def __init__(
        self,
        dataset_name: str = "compas",
        protected: str = "sex",
        metric_ub: float = 0.05,
        metric_lb: float = -0.05,
    ) -> None:
        self.metric_ub = metric_ub
        self.metric_lb = metric_lb
        self.dataset_name = dataset_name
        self.protected = protected
        np.random.seed(1)
        self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        if self.dataset_name == "adult":
            self.dataset_orig = load_preproc_data_adult()
        elif self.dataset_name == "german":
            self.dataset_orig = load_preproc_data_german()
        elif self.dataset_name == "compas":
            self.dataset_orig = load_preproc_data_compas()
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

    def baseline_fit(self):
        dataset_orig_train_pred = self.dataset_orig_train.copy(deepcopy=True)
        dataset_orig_valid_pred = self.dataset_orig_valid.copy(deepcopy=True)
        dataset_orig_test_pred = self.dataset_orig_test.copy(deepcopy=True)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(self.dataset_orig_train.features)
        y_train = self.dataset_orig_train.labels.ravel()

        lmod = LogisticRegression()
        lmod.fit(X_train, y_train)
        y_train_pred = lmod.predict(X_train)

        fav_idx = np.where(lmod.classes_ == self.dataset_orig_train.favorable_label)[0][
            0
        ]

        dataset_orig_train_pred.labels = y_train_pred

        X_valid = scaler.transform(dataset_orig_valid_pred.features)
        y_valid = dataset_orig_valid_pred.labels
        dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[
            :, fav_idx
        ].reshape(-1, 1)

        X_test = scaler.transform(dataset_orig_test_pred.features)
        y_test = dataset_orig_test_pred.labels
        dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:, fav_idx].reshape(
            -1, 1
        )

        return dataset_orig_train_pred, dataset_orig_valid_pred, dataset_orig_test_pred

    def get_thresh(self, dataset_orig_valid_pred):
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

        print(
            "Best balanced accuracy (no fairness constraints) = %.4f" % np.max(ba_arr)
        )
        print(
            "Optimal classification threshold (no fairness constraints) = %.4f"
            % best_class_thresh
        )
        return best_class_thresh

    def fit(self, dataset_orig_valid_pred, dataset_orig_test_pred):
        roc = RejectOptionClassification(
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
            low_class_thresh=0.01,
            high_class_thresh=0.99,
            num_class_thresh=100,
            num_ROC_margin=50,
            metric_ub=self.metric_ub,
            metric_lb=self.metric_lb,
        )
        roc = roc.fit(self.dataset_orig_valid, dataset_orig_valid_pred)

        print(
            "Optimal classification threshold (with fairness constraints) = %.4f"
            % roc.classification_threshold
        )
        print("Optimal ROC margin = %.4f" % roc.ROC_margin)

        dataset_transf_valid_pred = roc.predict(dataset_orig_valid_pred)
        dataset_transf_test_pred = roc.predict(dataset_orig_test_pred)
        return dataset_transf_valid_pred, dataset_transf_test_pred

    def compute_metrics(self, origin, pred, best_class_thresh):
        fav_inds = pred.scores > best_class_thresh
        pred.labels[fav_inds] = pred.favorable_label
        pred.labels[~fav_inds] = pred.unfavorable_label
        metrics = common_utils.compute_metrics(
            origin,
            pred,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )
        return metrics

    def run(self):
        (
            dataset_orig_train_pred,
            dataset_orig_valid_pred,
            dataset_orig_test_pred,
        ) = self.baseline_fit()
        best_class_thresh = self.get_thresh(dataset_orig_valid_pred)
        metrics_orig = self.compute_metrics(
            self.dataset_orig_test, dataset_orig_test_pred, best_class_thresh
        )
        dataset_transf_valid_pred, dataset_transf_test_pred = self.fit(
            dataset_orig_valid_pred, dataset_orig_test_pred
        )
        metrics_transf = common_utils.compute_metrics(
            self.dataset_orig_test,
            dataset_transf_test_pred,
            self.unprivileged_groups,
            self.privileged_groups,
        )
        return metrics_orig, metrics_transf


if __name__ == "__main__":
    roc = RejectOptionClassifier(
        dataset_name="compas", protected="sex", metric_ub=0.05, metric_lb=-0.05
    )
    metrics_orig, metrics_transf = roc.run()
    print("Metrics for original data:")
    print(metrics_orig)
    print("\nMetrics for transformed data:")
    print(metrics_transf)
