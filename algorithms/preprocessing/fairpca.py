import os, sys
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.datasets import StandardDataset

from MAF.algorithms.preprocessing.optim_preproc_helpers.data_prepro_function import (
    load_preproc_data_adult,
    load_preproc_data_german,
    load_preproc_data_compas,
)
from MAF.metric import common_utils


class MeanCovarianceMatchingFairPCA:
    def __init__(
        self,
        unprivileged_groups,
        privileged_groups,
        target_unfair_dim: int = 5,
        target_pca_dim: int = 5,
        n_iter_unfair: int = 100,
        n_iter_pca: int = 100,
    ):

        # (Un)privileged groups: list(dict)
        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups
        protected_attribute_names = sum(
            (list(d.keys()) for d in self.privileged_groups), start=[]
        )
        protected_attribute_names += sum(
            (list(d.keys()) for d in self.unprivileged_groups), start=[]
        )
        self.protected_attribute_names = set(protected_attribute_names)
        # Fair PCA parameters
        self.target_unfair_dim = target_unfair_dim
        self.target_pca_dim = target_pca_dim
        self.n_iter_unfair = n_iter_unfair
        self.n_iter_pca = n_iter_pca

    def fit_direct(self, dataset: StandardDataset):
        X, priv_cond, unpriv_cond = self._preprocess_dataset(dataset)
        if self.target_unfair_dim >= 0:
            # Mean matching
            mean_group = [X[priv_cond].mean(0), X[unpriv_cond].mean(0)]
            mean_diff = mean_group[1] - mean_group[0]
            norm_mean_diff = np.linalg.norm(mean_diff)
            mean_diff_direction = (
                np.zeros_like(mean_diff)
                if np.isclose(norm_mean_diff, 0)
                else mean_diff / norm_mean_diff
            )
            if np.isclose(norm_mean_diff, 0):
                warnings.warn("ALREADY MEAN MATCHED")
            if self.target_unfair_dim <= 0:
                unfair_directions = np.reshape(mean_diff_direction, (-1, 1))
            else:
                # Covariance matching up to top-m eigenspace (m=`target_unfair_dim`) in magnitude
                cov_group = [np.cov(X[priv_cond].T), np.cov(X[unpriv_cond].T)]
                cov_diff = cov_group[1] - cov_group[0]
                eigval_cov_diff, eigvec_cov_diff = np.linalg.eigh(
                    cov_diff
                )  # direct eigendecomposition! (for a real symm matrix)
                top_indices_unfair = np.argsort(np.abs(eigval_cov_diff))[
                    -self.target_unfair_dim :
                ][
                    ::-1
                ]  # indices of top-m eigenvalues (in magnitude)
                cov_diff_top_directions = eigvec_cov_diff[:, top_indices_unfair]
                proj_mean_diff = mean_diff - cov_diff_top_directions @ (
                    cov_diff_top_directions.T @ mean_diff
                )
                norm_proj_mean_diff = np.linalg.norm(proj_mean_diff)
                if np.isclose(norm_proj_mean_diff, 0):
                    warnings.warn("MEAN DIFF ∈ CovDiffDirections")
                    unfair_directions = cov_diff_top_directions
                else:
                    proj_mean_diff_direction = np.reshape(
                        proj_mean_diff / norm_proj_mean_diff, (-1, 1)
                    )
                    unfair_directions = np.concatenate(
                        [cov_diff_top_directions, proj_mean_diff_direction], -1
                    )
        # Constrained PCA
        cov_total = np.cov(X.T)
        if self.target_unfair_dim >= 0:
            proj_cov = cov_total - unfair_directions @ (unfair_directions.T @ cov_total)
            proj_cov -= (proj_cov @ unfair_directions) @ unfair_directions.T
        else:
            proj_cov = cov_total
        eigval, eigvec = np.linalg.eigh(
            proj_cov
        )  # direct eigendecomposition! (for a real symm matrix)
        top_indices_pca = np.argsort(np.abs(eigval))[-self.target_pca_dim :][
            ::-1
        ]  # indices of top-k eigenvalues (in magnitude)
        self.loading_matrix = eigvec[
            :, top_indices_pca
        ]  # PCA output: (d times k) dimensional

    def fit_power_method(self, dataset: StandardDataset):
        X, priv_cond, unpriv_cond = self._preprocess_dataset(dataset)
        _, d = X.shape
        # initial iterates
        V, _ = np.linalg.qr(np.random.randn(d, self.target_pca_dim))
        if self.target_unfair_dim > 0:
            W, _ = np.linalg.qr(np.random.randn(d, self.target_unfair_dim))
        if self.target_unfair_dim >= 0:
            # Mean matching
            mean_group = [X[priv_cond].mean(0), X[unpriv_cond].mean(0)]
            mean_diff = mean_group[1] - mean_group[0]
            norm_mean_diff = np.linalg.norm(mean_diff)
            mean_diff_direction = (
                np.zeros_like(mean_diff)
                if np.isclose(norm_mean_diff, 0)
                else mean_diff / norm_mean_diff
            )
            if np.isclose(norm_mean_diff, 0):
                warnings.warn("ALREADY MEAN MATCHED")
            if self.target_unfair_dim <= 0:
                unfair_directions = np.reshape(mean_diff_direction, (-1, 1))
            else:
                # Covariance matching up to top-m eigenspace (m=`target_unfair_dim`) in magnitude
                # Based on Power Method:
                for _ in range(self.n_iter_unfair):
                    W, _ = np.linalg.qr(
                        X[priv_cond].T @ X[priv_cond] @ W
                        - X[unpriv_cond].T @ X[unpriv_cond] @ W
                    )
                proj_mean_diff = mean_diff - W @ (W.T @ mean_diff)
                norm_proj_mean_diff = np.linalg.norm(proj_mean_diff)
                if np.isclose(norm_proj_mean_diff, 0):
                    warnings.warn("MEAN DIFF ∈ CovDiffDirections")
                    unfair_directions = W
                else:
                    proj_mean_diff_direction = np.reshape(
                        proj_mean_diff / norm_proj_mean_diff, (-1, 1)
                    )
                    unfair_directions = np.concatenate(
                        [W, proj_mean_diff_direction], -1
                    )
        # Constrained PCA
        # Based on Power Method:
        for _ in range(self.n_iter_pca):
            if self.target_unfair_dim >= 0:
                V -= unfair_directions @ (unfair_directions.T @ V)
            V = X.T @ (X @ V)
            if self.target_unfair_dim >= 0:
                V -= unfair_directions @ (unfair_directions.T @ V)
            V, _ = np.linalg.qr(V)
        self.loading_matrix = V

    def transform(self, dataset: StandardDataset):
        self._check_fitted()
        dataset_transformed = dataset.copy(deepcopy=True)
        d, k = self.loading_matrix.shape
        unprotected_features = np.array(
            [
                (fname not in self.protected_attribute_names)
                for fname in dataset.feature_names
            ]
        )
        assert np.sum(unprotected_features) == d
        X = dataset.features[:, unprotected_features]
        X_proj = X @ self.loading_matrix
        dataset_transformed.features = X_proj
        dataset_transformed.feature_names = list(range(k))
        return dataset_transformed

    def _preprocess_dataset(self, dataset: StandardDataset):
        unprotected_features = np.array(
            [
                (fname not in self.protected_attribute_names)
                for fname in dataset.feature_names
            ]
        )
        X = dataset.features[:, unprotected_features]  # delete protected attributes
        # Conditioning
        priv_cond = compute_boolean_conditioning_vector(
            dataset.protected_attributes,
            dataset.protected_attribute_names,
            condition=self.privileged_groups,
        )
        unpriv_cond = compute_boolean_conditioning_vector(
            dataset.protected_attributes,
            dataset.protected_attribute_names,
            condition=self.unprivileged_groups,
        )
        # Centering & scaling dataset
        scaler = StandardScaler(with_std=False)
        X = scaler.fit_transform(X)
        return X, priv_cond, unpriv_cond

    def _check_fitted(self):
        return getattr(self, "loading_matrix", None) is not None


class MeanCovarianceMatchingFairPCAWithClassifier:
    def __init__(
        self,
        dataset_name: str = "adult",
        protected: str = "sex",
        target_unfair_dim: int = 5,
        target_pca_dim: int = 5,
        algorithm_mode: str = "power_method",
        n_iter_unfair: int = 100,
        n_iter_pca: int = 100,
    ):

        np.random.seed(1)
        self.dataset_name = dataset_name
        self.protected = protected
        self.target_unfair_dim = target_unfair_dim
        self.target_pca_dim = target_pca_dim
        algorithm_mode_list = ["power_method", "direct"]
        if algorithm_mode not in algorithm_mode_list:
            raise ValueError(
                f"Unsupported algorithm_mode: '{algorithm_mode}' (not in {algorithm_mode_list})"
            )
        self.algorithm_mode = algorithm_mode
        self.n_iter_unfair = (
            n_iter_unfair  # used only when algorithm_mode = 'power_method'
        )
        self.n_iter_pca = n_iter_pca  # used only when algorithm_mode = 'power_method'
        self.load_and_preprocess_data()
        self.transform = None  # PCA module

    def load_and_preprocess_data(self):
        dataset_loaders = {
            "adult": load_preproc_data_adult,
            "german": load_preproc_data_german,
            "compas": load_preproc_data_compas,
        }
        if self.dataset_name not in dataset_loaders:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        self.dataset_orig: StandardDataset = dataset_loaders[self.dataset_name]()
        self.dataset_orig_train, self.dataset_orig_vt = self.dataset_orig.split(
            [0.7], shuffle=True, seed=1
        )
        self.dataset_orig_valid, self.dataset_orig_test = self.dataset_orig_vt.split(
            [0.5], shuffle=True, seed=1
        )
        self.privileged_groups = [{self.protected: 1}]
        self.unprivileged_groups = [{self.protected: 0}]

    def baseline_fit(self, dataset: StandardDataset):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(dataset.features)
        y_train = dataset.labels.ravel()
        lmod = LogisticRegression()
        lmod.fit(X_train, y_train, sample_weight=dataset.instance_weights)
        return lmod, scaler

    def predict(
        self,
        dataset: StandardDataset,
        model: LogisticRegression,
        scaler: StandardScaler,
    ):
        dataset_pred = dataset.copy(deepcopy=True)
        X = scaler.transform(dataset_pred.features)
        dataset_pred.scores = model.predict_proba(X)[:, self.pos_ind].reshape(-1, 1)
        return dataset_pred

    def evaluate(self, model: LogisticRegression, scaler: StandardScaler):
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
        return best_class_thresh

    def compute_metrics(
        self,
        dataset_true: StandardDataset,
        dataset_pred: StandardDataset,
        best_class_thresh,
    ):
        fav_inds = dataset_pred.scores > best_class_thresh
        dataset_pred.labels[fav_inds] = dataset_pred.favorable_label
        dataset_pred.labels[~fav_inds] = dataset_pred.unfavorable_label
        metrics = common_utils.compute_metrics(
            dataset_true,
            dataset_pred,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )
        return metrics

    def get_fair_pca_transform(self):
        self.transform = MeanCovarianceMatchingFairPCA(
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
            target_unfair_dim=self.target_unfair_dim,
            target_pca_dim=self.target_pca_dim,
            n_iter_unfair=self.n_iter_unfair,
            n_iter_pca=self.n_iter_pca,
        )
        if self.algorithm_mode == "direct":
            self.transform.fit_direct(self.dataset_orig_train)
            print("Fit Direct completed.")
        if self.algorithm_mode == "power_method":
            self.transform.fit_power_method(self.dataset_orig_train)
            print("Fit Power Method completed.")
        print("Transform initialized:", self.transform is not None)

    def run(self):
        """Runner of MeanCovarianceMatchingFairPCAWithClassifier class.

        Return:
        - metrics_orig (OrderedDict): performance & fairness metrics for original dataset (naïve logistic regression)
        - metrics_transf (OrderedDict): performance & fairness metrics for transformed dataset with Fair PCA
        """
        # baseline
        model_orig, scaler_orig = self.baseline_fit(self.dataset_orig_train)
        self.pos_ind = np.where(
            model_orig.classes_ == self.dataset_orig_train.favorable_label
        )[0][0]
        best_class_thresh_orig = self.evaluate(model_orig, scaler_orig)
        orig_dataset_test_pred = self.predict(
            self.dataset_orig_test, model_orig, scaler_orig
        )
        metrics_orig = self.compute_metrics(
            self.dataset_orig_test, orig_dataset_test_pred, best_class_thresh_orig
        )
        # fair preprocessing
        transform = self.get_fair_pca_transform()

        transf_data_train = self.transform.transform(self.dataset_orig_train)
        transf_data_test = self.transform.transform(self.dataset_orig_test)

        model_transf, scaler_transf = self.baseline_fit(transf_data_train)
        transf_dataset_test_pred = self.predict(
            transf_data_test, model_transf, scaler_transf
        )
        metrics_transf = self.compute_metrics(
            transf_data_test, transf_dataset_test_pred, best_class_thresh_orig
        )
        return metrics_orig, metrics_transf


if __name__ == "__main__":
    fairpca = MeanCovarianceMatchingFairPCAWithClassifier(
        dataset_name="adult",
        protected="sex",
        target_unfair_dim=0,
        target_pca_dim=4,
        algorithm_mode="direct",
    )

    metrics_orig, metrics_transf = fairpca.run()
    print("Metrics for original data:")
    print(metrics_orig)
    print("\nMetrics for transformed data:")
    print(metrics_transf)
