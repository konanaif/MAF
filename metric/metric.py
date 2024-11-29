import sys
import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from sklearn.manifold import TSNE

from aif360.metrics import DatasetMetric, BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from MAF.datamodule.dataset import aifData


def set_privileged_groups(data_name: str):
    if (data_name == "compas") or (data_name == "adult"):
        privileged_groups = [{"sex": 1.0}]
        unprivileged_groups = [{"sex": 0.0}]
    elif data_name == "german":
        privileged_groups = [{"sex": 0.0}]
        unprivileged_groups = [{"sex": 1.0}]
    elif data_name == "pubfig":
        privileged_groups = [{"Heavy Makeup": 1.0}]
        unprivileged_groups = [{"Heavy Makeup": 0.0}]
    elif data_name == "celeba":
        privileged_groups = [{"Male": 1.0}]
        unprivileged_groups = [{"Male": 0.0}]
    return privileged_groups, unprivileged_groups


class DataMetric(BinaryLabelDatasetMetric):
    def __init__(self, dataset, data_name):
        self.dataset = dataset
        self.data_name = data_name
        privileged_groups, unprivileged_groups = set_privileged_groups(data_name)
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups


def compute_tsne(dataset: aifData, sample_size: int = 10):
    print("T-SNE analysis start")
    priv_val = dataset.privileged_protected_attributes[0][0]
    unpriv_val = dataset.unprivileged_protected_attributes[0][0]

    df = dataset.convert_to_dataframe()[0]
    df_priv = df.loc[df[dataset.protected_attribute_names[0]] == priv_val]
    df_unpriv = df.loc[df[dataset.protected_attribute_names[0]] == unpriv_val]
    ds_priv = aifData(
        df=df_priv,
        label_name=dataset.label_names[0],
        favorable_classes=[dataset.favorable_label],
        protected_attribute_names=dataset.protected_attribute_names,
        privileged_classes=dataset.privileged_protected_attributes,
    )
    ds_unpriv = aifData(
        df=df_unpriv,
        label_name=dataset.label_names[0],
        favorable_classes=[dataset.favorable_label],
        protected_attribute_names=dataset.protected_attribute_names,
        privileged_classes=dataset.privileged_protected_attributes,
    )

    priv_sample = random.sample(ds_priv.features.tolist(), k=sample_size)
    priv_sample = np.array(priv_sample)
    unpriv_sample = random.sample(ds_unpriv.features.tolist(), k=sample_size)
    unpriv_sample = np.array(unpriv_sample)

    # T-SNE analysis
    tsne_priv = TSNE(
        n_components=2, learning_rate="auto", init="random", perplexity=5
    ).fit_transform(priv_sample)
    tsne_priv = tsne_priv.tolist()

    tsne_unpriv = TSNE(
        n_components=2, learning_rate="auto", init="random", perplexity=5
    ).fit_transform(unpriv_sample)
    tsne_unpriv = tsne_unpriv.tolist()

    return tsne_priv, tsne_unpriv


def get_baseline_result(trainset, testset, baseline: str = "svm"):
    if baseline == "svm":
        baseline = svm.SVC(random_state=777)

    baseline.fit(trainset.features, trainset.labels.ravel())
    prediction = testset.copy()
    prediction.labels = baseline.predict(testset.features)
    return prediction


def get_metrics(dataset, data_name: str):
    """
    dataset: CompasDataset, GermanDataset, AdultDataset
    """
    tsne_priv, tsne_unpriv = compute_tsne(dataset)
    data_metric = DataMetric(dataset=dataset, data_name=data_name)

    traindata, testdata = dataset.split([0.7], shuffle=True)
    test_prediction = get_baseline_result(trainset=traindata, testset=testdata)

    privileged_groups, unprivileged_groups = set_privileged_groups(data_name)
    cls_metric = ClassificationMetric(
        testdata,
        test_prediction,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )

    perfm = dict(
        TPR=cls_metric.true_positive_rate(),
        TNR=cls_metric.true_negative_rate(),
        FPR=cls_metric.false_positive_rate(),
        FNR=cls_metric.false_negative_rate(),
        PPV=cls_metric.positive_predictive_value(),
        NPV=cls_metric.negative_predictive_value(),
        FDR=cls_metric.false_discovery_rate(),
        FOR=cls_metric.false_omission_rate(),
        ACC=cls_metric.accuracy(),
    )

    clsmetrics = dict(
        error_rate=round(cls_metric.error_rate(), 3),
        average_odds_difference=round(cls_metric.average_odds_difference(), 3),
        average_abs_odds_difference=round(cls_metric.average_abs_odds_difference(), 3),
        selection_rate=round(cls_metric.selection_rate(), 3),
        disparate_impact=round(cls_metric.disparate_impact(), 3),
        statistical_parity_difference=round(
            cls_metric.statistical_parity_difference(), 3
        ),
        generalized_entropy_index=round(cls_metric.generalized_entropy_index(), 3),
        theil_index=round(cls_metric.theil_index(), 3),
        equal_opportunity_difference=round(
            cls_metric.equal_opportunity_difference(), 3
        ),
    )
    metrics = {
        "data": {
            "protected": dataset.protected_attribute_names[0],
            "privileged": {
                "num_negatives": data_metric.num_negatives(privileged=True),
                "num_positives": data_metric.num_positives(privileged=True),
                "TSNE": tsne_priv,
            },
            "unprivileged": {
                "num_negatives": data_metric.num_negatives(privileged=False),
                "num_positives": data_metric.num_positives(privileged=False),
                "TSNE": tsne_unpriv,
            },
            "base_rate": round(data_metric.base_rate(), 3),
            "statistical_parity_difference": round(
                data_metric.statistical_parity_difference(), 3
            ),
            "consistency": round(data_metric.consistency()[0], 4),
        },
        "performance": {
            "recall": round(perfm["TPR"], 3),
            "true_negative_rate": round(perfm["TNR"], 3),
            "false_positive_rate": round(perfm["FPR"], 3),
            "false_negative_rate": round(perfm["FNR"], 3),
            "precision": round(perfm["PPV"], 3),
            "negative_predictive_value": round(perfm["NPV"], 3),
            "false_discovery_rate": round(perfm["FDR"], 3),
            "false_omission_rate": round(perfm["FOR"], 3),
            "accuracy": round(perfm["ACC"], 3),
        },
        "classify": {
            "error_rate": clsmetrics["error_rate"],
            "average_odds_difference": clsmetrics["average_odds_difference"]
            if not pd.isna(clsmetrics["average_odds_difference"])
            else 0.0,
            "average_abs_odds_difference": clsmetrics["average_abs_odds_difference"]
            if not pd.isna(clsmetrics["average_abs_odds_difference"])
            else 0.0,
            "selection_rate": clsmetrics["selection_rate"],
            "disparate_impact": clsmetrics["disparate_impact"]
            if not pd.isna(clsmetrics["disparate_impact"])
            else 0.0,
            "statistical_parity_difference": clsmetrics["statistical_parity_difference"]
            if not pd.isna(clsmetrics["statistical_parity_difference"])
            else 0.0,
            "generalized_entropy_index": clsmetrics["generalized_entropy_index"],
            "theil_index": clsmetrics["theil_index"],
            "equal_opportunity_difference": clsmetrics["equal_opportunity_difference"]
            if not pd.isna(clsmetrics["equal_opportunity_difference"])
            else 0.0,
        },
    }
    return metrics
