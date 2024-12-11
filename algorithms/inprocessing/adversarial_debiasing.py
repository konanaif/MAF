import os, sys
from aif360.algorithms.inprocessing.adversarial_debiasing import (
    AdversarialDebiasing as aifAdversarialDebiasing,
)
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
from MAF.algorithms.preprocessing.optim_preproc_helpers.data_prepro_function import (
    load_preproc_data_adult,
    load_preproc_data_german,
    load_preproc_data_compas,
)
from MAF.metric import common_utils

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


class AdversarialDebiasing:
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

        scaler = MaxAbsScaler()
        self.dataset_orig_train.features = scaler.fit_transform(
            self.dataset_orig_train.features
        )
        self.dataset_orig_test.features = scaler.transform(
            self.dataset_orig_test.features
        )

        self.privileged_groups = [{self.protected: 1}]
        self.unprivileged_groups = [{self.protected: 0}]

    def ad_fit(self, scope_name, debias):
        tf.disable_eager_execution()
        sess = tf.Session()
        ad = aifAdversarialDebiasing(
            privileged_groups=self.privileged_groups,
            unprivileged_groups=self.unprivileged_groups,
            scope_name=scope_name,
            debias=debias,
            sess=sess,
        )
        ad.fit(self.dataset_orig_train)
        pred_test = ad.predict(self.dataset_orig_test)
        sess.close()
        tf.reset_default_graph()
        return pred_test

    def compute_metrics(self, dataset):
        return common_utils.compute_metrics(
            self.dataset_orig_test,
            dataset,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )

    def run(self):
        orig_test_pred = self.ad_fit(debias=False, scope_name="plain_classifier")
        metrics_orig = self.compute_metrics(orig_test_pred)

        transf_test_pred = self.ad_fit(debias=True, scope_name="debiased_classifier")
        metrics_transform = self.compute_metrics(transf_test_pred)

        return metrics_orig, metrics_transform


if __name__ == "__main__":
    advdebias = AdversarialDebiasing(dataset_name="compas", protected="sex")
    metrics_orig, metrics_transf = advdebias.run()
    print("Metrics for original data:")
    print(metrics_orig)
    print("\nMetrics for transformed data:")
    print(metrics_transf)
