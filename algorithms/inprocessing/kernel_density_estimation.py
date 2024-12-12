import math
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from MAF.datamodule.dataset import RawDataSet
from MAF.metric import common_utils
from MAF.algorithms.preprocessing.optim_preproc_helpers.data_prepro_function import (
    load_preproc_data_adult,
    load_preproc_data_german,
    load_preproc_data_compas,
)
from MAF.utils.common import fix_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Train on [[[  {}  ]]] device.".format(device))
fix_seed(1)


def mapping(class_vector):
    class_vector = class_vector.ravel()

    cls2val = np.unique(class_vector)
    val2cls = dict(zip(cls2val, range(len(cls2val))))

    converted_vector = [val2cls[v] for v in class_vector]

    return cls2val, val2cls, converted_vector


def normal_pdf(x):
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


def normal_cdf(y, h=0.01, tau=0.5):
    # Approximation of Q-function given by Lopez-Benitez & Casadevall (2011)
    # based on a second-order exponential function & Q(x) = 1 - Q(-x):
    Q_fn = lambda x: torch.exp(-0.4920 * x**2 - 0.2887 * x - 1.1893)

    m = len(y)
    y_prime = (tau - y) / h
    summation = (
        torch.sum(Q_fn(y_prime[y_prime > 0]))
        + torch.sum(1 - Q_fn(torch.abs(y_prime[y_prime < 0])))
        + 0.5 * len(y_prime[y_prime == 0])
    )

    return summation / m


def Huber_loss(x, delta):
    if abs(x) < delta:
        return (x**2) / 2
    else:
        return delta * (x.abs() - delta / 2)


def Huber_loss_derivative(x, delta):
    if x > delta:
        return delta
    elif x < -delta:
        return -delta
    return x


def get_fairness_metrics(Y, Z, Ytilde, classes, protect_attrs):
    DDP = 0
    DEO = 0
    for y in classes:
        Pr_Ytilde_y = (Ytilde == y).mean()
        Ytilde_y_given_Y_y = np.logical_and(Ytilde == y, Y == y)
        for z in range(n_sensitive_attrs):
            DDP += abs(
                np.logical_and(Ytilde == y, Z == z).mean() / (Z == z).mean()
                - Pr_Ytilde_y
            )
            DEO += abs(
                np.logical_and(Ytilde_y_given_Y_y == y, Z == z).mean()
                / np.logical_and(Y == y, Z == z).mean()
                - Ytilde_y_given_Y_y.mean() / (Y == y).mean()
            )
    return DDP, DEO


class BCELossAccuracy:
    def __init__(self):
        self.loss_function = nn.BCELoss()

    @staticmethod
    def accuracy(y_hat, labels):
        with torch.no_grad():
            y_tilde = (y_hat > 0.5).int()
            accuracy = (y_tilde == labels.int()).float().mean().item()
        return accuracy

    def __call__(self, y_hat, labels):
        loss = self.loss_function(y_hat, labels)
        accuracy = self.accuracy(y_hat, labels)
        return loss, accuracy


class CELossAccuracy:
    def __init__(self):
        self.loss_function = nn.CrossEntropyLoss()

    @staticmethod
    def accuracy(y_hat, labels):
        with torch.no_grad():
            y_tilde = y_hat.argmax(axis=1)
            accuracy = (y_tilde == labels).float().mean().item()
        return accuracy

    def __call__(self, y_hat, labels):
        loss = self.loss_function(y_hat, labels)
        accuracy = self.accuracy(y_hat, labels)
        return loss, accuracy


class FairnessLoss:
    def __init__(
        self, h, tau, delta, notion, n_classes, n_sensitive_attrs, sensitive_attr
    ):
        self.h = h
        self.tau = tau
        self.delta = delta
        self.fairness_notion = notion
        self.n_classes = n_classes
        self.n_sensitive_attrs = n_sensitive_attrs
        self.sensitive_attr = sensitive_attr

        if self.n_classes > 2:
            self.tau = 0.5

        assert self.fairness_notion in ["DP", "EO"]

    def DDP_loss(self, y_hat, Z):
        m = y_hat.shape[0]
        backward_loss = 0
        logging_loss = 0

        if self.n_classes == 2:
            Pr_Ytilde1 = normal_cdf(y_hat.detach(), self.h, self.tau)
            for z in self.sensitive_attr:
                Pr_Ytilde1_Z = normal_cdf(y_hat.detach()[Z == z], self.h, self.tau)
                m_z = Z[Z == z].shape[0]

                Prob_diff_Z = Pr_Ytilde1_Z - Pr_Ytilde1

                _dummy = torch.dot(
                    normal_pdf((self.tau - y_hat.detach()[Z == z]) / self.h).view(-1),
                    y_hat[Z == z].view(-1),
                ) / (self.h * m_z) - torch.dot(
                    normal_pdf((self.tau - y_hat.detach()) / self.h).view(-1),
                    y_hat.view(-1),
                ) / (
                    self.h * m
                )

                _dummy *= Huber_loss_derivative(Prob_diff_Z, self.delta)

                backward_loss += _dummy

                logging_loss += Huber_loss(Prob_diff_Z, self.delta)

        else:
            idx_set = list(range(self.n_classes)) if self.n_classes > 2 else [0]
            for y in idx_set:
                Pr_Ytilde1 = normal_cdf(y_hat[:, y].detach(), self.h, self.tau)
                for z in self.sensitive_attr:
                    Pr_Ytilde1_Z = normal_cdf(y_hat[:, y].detach(), self.h, self.tau)
                    m_z = Z[Z == z].shape[0]

                    Prob_diff_Z = Pr_Ytilde1_Z - Pr_Ytilde1
                    _dummy = Huber_loss_derivative(Prob_diff_Z, self.delta)
                    _dummy *= torch.dot(
                        normal_pdf(
                            (self.tau - y_hat[:, y].detach()[Z == z]) / self.h
                        ).view(-1),
                        y_hat[:, y][Z == z].view(-1),
                    ) / (self.h * m_z) - torch.dot(
                        normal_pdf((self.tau - y_hat[:, y].detach()) / self.h).view(-1),
                        y_hat[:, y].view(-1),
                    ) / (
                        self.h * m
                    )

                    backward_loss += _dummy
                    logging_loss += Huber_loss(Prob_diff_Z, self.delta).item()

        return backward_loss, logging_loss

    def DEO_loss(self, y_hat, Y, Z):
        backward_loss = 0
        logging_loss = 0

        if self.n_classes == 2:
            for y in [0, 1]:
                Pr_Ytilde1_Y = normal_cdf(y_hat[Y == y].detach(), self.h, self.tau)
                m_y = (Y == y).sum().item()
                for z in self.sensitive_attr:
                    Pr_Ytilde1_YZ = normal_cdf(
                        y_hat[torch.logical_and(Y == y, Z == z)].detach(),
                        self.h,
                        self.tau,
                    )
                    m_zy = torch.logical_and(Y == y, Z == z).sum().item()

                    Prob_diff_Z = Pr_Ytilde1_YZ - Pr_Ytilde1_Y
                    _dummy = Huber_loss_derivative(Prob_diff_Z, self.delta)
                    _dummy *= torch.dot(
                        normal_pdf(
                            (
                                self.tau
                                - y_hat[torch.logical_and(Y == y, Z == z)].detach()
                            )
                            / self.h
                        ).view(-1),
                        y_hat[torch.logical_and(Y == y, Z == z)].view(-1),
                    ) / (self.h * m_zy) - torch.dot(
                        normal_pdf((self.tau - y_hat[Y == y].detach()) / self.h).view(
                            -1
                        ),
                        y_hat[torch.logical_and(Y == y, Z == z)].view(-1),
                    ) / (
                        self.h * m_y
                    )

                    backward_loss += _dummy
                    logging_loss += Huber_loss(Prob_diff_Z, self.delta).item()
        else:
            for y in range(self.n_classes):
                Pr_Ytilde1_Y = normal_cdf(
                    y_hat[:, y][Y == y].detach(), self.h, self.tau
                )
                m_y = (Y == y).sum().item()
                for z in self.sensitive_attr:
                    Pr_Ytilde1_YZ = normal_cdf(
                        y_hat[:, y][torch.logical_and(Y == y, Z == z)].detach(),
                        self.h,
                        self.tau,
                    )
                    m_zy = torch.logical_and(Y == y, Z == z).sum().item()

                    Prob_diff_Z = Pr_Ytilde1_YZ - Pr_Ytilde1_Y
                    _dummy = Huber_loss_derivative(Prob_diff_Z, self.delta)
                    _dummy *= torch.dot(
                        normal_pdf(
                            (
                                self.tau
                                - y_hat[:, y][
                                    torch.logical_and(Y == y, Z == z)
                                ].detach()
                            )
                            / self.h
                        ).view(-1),
                        y_hat[:, y][torch.logical_and(Y == y, Z == z)].view(-1),
                    ) / (self.h * m_zy) - torch.dot(
                        normal_pdf(
                            (self.tau - y_hat[:, y][Y == y].detach()) / self.h
                        ).view(-1),
                        y_hat[:, y][Y == y].view(-1),
                    ) / (
                        self.h * m_y
                    )

                backward_loss += _dummy
                logging_loss += Huber_loss(Prob_diff_Z, self.delta).item()

        return backward_loss, logging_loss

    def __call__(self, y_hat, Y, Z):
        if self.fairness_notion == "DP":
            return self.DDP_loss(y_hat, Z)
        else:
            return self.DEO_loss(y_hat, Y, Z)


class KDEDataset(Dataset):
    def __init__(self, X, y, z):
        self.cls2val_t, self.val2cls_t, target = mapping(y)
        self.cls2val_b, self.val2cls_b, bias = mapping(z)

        self.X = torch.Tensor(X).to(device)
        self.y = torch.Tensor(y).type(torch.long).to(device)
        self.z = torch.Tensor(z).type(torch.long).to(device)

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.z[idx]


class KDEParameters:
    def __init__(
        self,
        fairness_type: str = "DP",
        batch_size: int = 64,
        n_epoch: int = 20,
        learning_rate: float = 0.01,
        h: float = 0.01,
        tau: float = 0.5,
        delta: float = 0.5,
        l: float = 0.1,
        seed: int = 777,
        model=None,
    ):
        self.fairness_type = fairness_type
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate
        self.h = h
        self.tau = tau
        self.delta = delta
        self.l = l
        self.seed = seed
        self.model = model


class KDEModel:
    def __init__(self, params: KDEParameters):
        self.params = params
        self.ce_loss = CELossAccuracy()

    def train(self, train_data):
        self.train_data = KDEDataset(
            X=train_data.feature, y=train_data.target, z=train_data.bias
        )
        self.n_class = len(np.unique(train_data.target))
        self.n_protect = len(np.unique(train_data.bias))

        if self.params.model is None:
            self.model = nn.Linear(train_data.feature.shape[-1], self.n_class)

            self.model = self.model.to(device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.params.learning_rate
        )
        self.fairness_loss = FairnessLoss(
            self.params.h,
            self.params.tau,
            self.params.delta,
            self.params.fairness_type,
            self.n_class,
            self.n_protect,
            np.unique(self.train_data.y),
        )

        trainloader = DataLoader(
            self.train_data,
            batch_size=self.params.batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.model.train()

        print("Train model start.")
        for ep in range(self.params.n_epoch):
            for idx, (X, y, z) in enumerate(trainloader):
                self.optimizer.zero_grad()

                pred = self.model(X)
                tilde = torch.round(pred.detach().reshape(-1))
                p_loss, acc = self.ce_loss(pred.squeeze(), y.squeeze())
                f_loss, f_loss_item = self.fairness_loss(pred, y, z)
                cost = (1 - self.params.l) * p_loss + self.params.l * f_loss

                if (torch.isnan(cost)).any():
                    continue

                cost.backward()
                self.optimizer.step()

                if (idx + 1) % 10 == 0 or (idx + 1) == len(trainloader):
                    print(
                        "Epoch [{}/{}], Batch [{}/{}], Cost: {:.4f}".format(
                            ep + 1,
                            self.params.n_epoch,
                            idx + 1,
                            len(trainloader),
                            cost.item(),
                        ),
                        end="\r",
                    )

        print("Train model done.")

    def evaluation(self, test_data):
        self.test_data = KDEDataset(
            X=test_data.feature, y=test_data.target, z=test_data.bias
        )
        testloader = DataLoader(
            self.test_data,
            batch_size=self.params.batch_size,
            shuffle=False,
            drop_last=False,
        )

        try:
            self.model.eval()
        except:
            return "Setting model is required."

        print("Evaluation start.")
        prediction = []
        for X, y, z in testloader:
            pred = self.model(X)
            pred = pred.argmax(dim=1)
            prediction.append(pred)

        prediction = torch.cat(prediction)
        print("Evaluation done.")
        return prediction


class KernelDensityEstimation:
    def __init__(self, dataset_name="compas", protected="race"):
        self.dataset_name = dataset_name
        self.protected = protected
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

        self.privileged_groups = [{self.protected: 1}]
        self.unprivileged_groups = [{self.protected: 0}]

        self.protected_label = self.dataset_orig_train.protected_attribute_names[0]
        self.protected_idx = self.dataset_orig_train.feature_names.index(
            self.protected_label
        )
        self.biased_train = self.dataset_orig_train.features[:, self.protected_idx]
        self.biased_test = self.dataset_orig_test.features[:, self.protected_idx]

    def fit(self, params: KDEParameters):
        train_data = RawDataSet(
            x=self.dataset_orig_train.features,
            y=self.dataset_orig_train.labels,
            z=self.biased_train,
        )
        test_data = RawDataSet(
            x=self.dataset_orig_test.features,
            y=self.dataset_orig_test.labels,
            z=self.biased_test,
        )

        kde = KDEModel(params)
        kde.train(train_data)
        pred = kde.evaluation(test_data)
        pred_dataset = self.dataset_orig_test.copy(deepcopy=True)
        pred_dataset.labels = np.array(pred)
        return pred_dataset

    def baseline_fit(self):
        scale_orig = StandardScaler()
        X_train = scale_orig.fit_transform(self.dataset_orig_train.features)
        y_train = self.dataset_orig_train.labels.ravel()

        X_test = scale_orig.transform(self.dataset_orig_test.features)
        y_test = self.dataset_orig_test.labels.ravel()

        lmod = LogisticRegression()
        lmod.fit(X_train, y_train)
        y_test_pred = lmod.predict(X_test)

        pred_dataset = self.dataset_orig_test.copy(deepcopy=True)
        pred_dataset.labels = y_test_pred
        return pred_dataset

    def compute_metrics(self, dataset):
        return common_utils.compute_metrics(
            self.dataset_orig_test,
            dataset,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )

    def run(self, params: KDEParameters):
        lr_pred = self.baseline_fit()
        pr_pred = self.fit(params)

        metrics_orig = self.compute_metrics(lr_pred)
        metrics_transform = self.compute_metrics(pr_pred)
        return metrics_orig, metrics_transform


if __name__ == "__main__":
    kde_params = KDEParameters()
    kde = KernelDensityEstimation(dataset_name="adult", protected="sex")
    metrics_orig, metrics_transf = kde.run(kde_params)
    print("Metrics for original data:")
    print(metrics_orig)
    print("\nMetrics for transformed data:")
    print(metrics_transf)
