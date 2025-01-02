from aif360.algorithms.inprocessing import GerryFairClassifier as aifGerryFairClassifier
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import tree
from MAF.algorithms.preprocessing.optim_preproc_helpers.data_prepro_function import (
    load_preproc_data_adult,
    load_preproc_data_german,
    load_preproc_data_compas,
)
from MAF.metric import common_utils
from MAF.utils.common import fix_seed

fix_seed(1)


class GerryFairClassifier:
    def __init__(self, dataset_name: str = "adult", protected: str = "sex") -> None:
        self.dataset_name = dataset_name
        self.protected = protected
        self.C = 100
        self.gamma = 0.005
        self.print_flag = True
        self.max_iterations = 500
        self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        loaders = {
            "adult": load_preproc_data_adult,
            "german": load_preproc_data_german,
            "compas": load_preproc_data_compas,
        }
        if self.dataset_name not in loaders:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.dataset_orig = loaders[self.dataset_name]([self.protected])
        self.dataset_orig_train, self.dataset_orig_vt = self.dataset_orig.split(
            [0.7], shuffle=True, seed=1
        )
        self.dataset_orig_valid, self.dataset_orig_test = self.dataset_orig_vt.split(
            [0.5], shuffle=True, seed=1
        )

        self.privileged_groups = [{self.protected: 1}]
        self.unprivileged_groups = [{self.protected: 0}]

    def fit(self):
        gfc = aifGerryFairClassifier(
            C=self.C,
            printflag=self.print_flag,
            gamma=self.gamma,
            fairness_def="FP",
            max_iters=self.max_iterations,
            heatmapflag=False,
        )

        gfc.fit(dataset=self.dataset_orig_train, early_termination=True)
        yhat = gfc.predict(self.dataset_orig_test, threshold=False)
        return yhat

    def baseline_fit(self):
        predictor = tree.DecisionTreeRegressor(max_depth=3)
        baseline_gfc = aifGerryFairClassifier(
            C=100,
            printflag=False,
            gamma=1,
            predictor=predictor,
            max_iters=self.max_iterations,
        )

        baseline_gfc.fit(dataset=self.dataset_orig_train, early_termination=True)
        preds = baseline_gfc.predict(self.dataset_orig_test, threshold=False)
        return preds

    def compute_metrics(self, dataset):
        return common_utils.compute_metrics(
            self.dataset_orig_test,
            dataset,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )

    def run(self):
        lr_pred = self.baseline_fit()
        gfc_pred = self.fit()
        metrics_orig = self.compute_metrics(lr_pred)
        metrics_transform = self.compute_metrics(gfc_pred)

        return metrics_orig, metrics_transform


if __name__ == "__main__":
    gfc = GerryFairClassifier(dataset_name="compas", protected="sex")
    metrics_orig, metrics_transform = gfc.run()
    print("Metrics for original data:")
    print(metrics_orig)
    print("\nMetrics for transformed data:")
    print(metrics_transform)
