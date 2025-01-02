import argparse, os, sys, json
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
)
from sklearn.preprocessing import MinMaxScaler

from MAF.algorithms.config.load_yaml_config import yaml_config_hook

parent_dir = os.environ["PYTHONPATH"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FairLoss(nn.Module):
    def __init__(self, alg="sipm"):
        super(FairLoss, self).__init__()
        self.alg = alg

    def forward(self, proj0, proj1):
        proj0, proj1 = proj0.flatten(), proj1.flatten()
        mean0, mean1 = proj0.mean(), proj1.mean()
        loss = (mean0 - mean1).abs()

        return loss


class EncoderNet(nn.Module):
    def __init__(self, num_layer, input_dim, rep_dim, acti):
        super(EncoderNet, self).__init__()

        if acti == "relu":
            self.acti = nn.ReLU()
        elif acti == "leakyrelu":
            self.acti = nn.LeakyReLU()
        elif acti == "softplus":
            self.acti = nn.SoftPlus()

        self.net = nn.ModuleList()
        for i in range(num_layer + 1):
            if i == 0:
                self.net.append(nn.Linear(input_dim, rep_dim))
            else:
                self.net.append(self.acti)
                self.net.append(nn.Linear(rep_dim, rep_dim))

    def forward(self, x):
        for _, fc in enumerate(self.net):
            x = fc(x)
        return x


class HeadNet(nn.Module):
    def __init__(self, num_layer, input_dim, rep_dim, acti):
        super(HeadNet, self).__init__()

        if acti == "relu":
            self.acti = nn.ReLU()
        elif acti == "leakyrelu":
            self.acti = nn.LeakyReLU()
        elif acti == "softplus":
            self.acti = nn.SoftPlus()
        elif acti == "sigmoid":
            self.acti = nn.Sigmoid()

        self.net = nn.ModuleList()
        for i in range(num_layer + 1):
            if i == num_layer:
                self.net.append(nn.Linear(rep_dim, 1))
            else:
                self.net.append(nn.Linear(rep_dim, rep_dim))
                self.net.append(self.acti)

    def forward(self, x):
        for _, fc in enumerate(self.net):
            x = fc(x)
        return x, torch.sigmoid(x)


class DecoderNet(nn.Module):
    def __init__(self, num_layer, input_dim, rep_dim, acti):
        super(DecoderNet, self).__init__()

        if acti == "relu":
            self.acti = nn.ReLU()
        elif acti == "leakyrelu":
            self.acti = nn.LeakyReLU()
        elif acti == "softplus":
            self.acti = nn.SoftPlus()

        self.net = nn.ModuleList()
        for i in range(num_layer + 1):
            if i == num_layer:
                self.net.append(nn.Linear(rep_dim, input_dim))
                self.net.append(nn.Sigmoid())
            else:
                self.net.append(nn.Linear(rep_dim, rep_dim))
                self.net.append(self.acti)

    def forward(self, x):
        for _, fc in enumerate(self.net):
            x = fc(x)
        return x


class MLP(nn.Module):
    def __init__(self, num_layer, input_dim, rep_dim, acti):
        super(MLP, self).__init__()

        self.num_layer = num_layer
        self.input_dim = input_dim
        self.rep_dim = rep_dim
        self.acti = acti

        self.encoder = EncoderNet(
            self.num_layer, self.input_dim, self.rep_dim, acti="leakyrelu"
        )
        self.head = HeadNet(self.num_layer, self.input_dim, self.rep_dim, self.acti)
        self.decoder = DecoderNet(
            self.num_layer, self.input_dim, self.rep_dim, acti="leakyrelu"
        )

    # freezing and melting
    def freeze(self):
        for para in self.parameters():
            para.requires_grad = False

    def melt(self):
        for para in self.parameters():
            para.requires_grad = True

    def melt_head_only(self):
        for para in self.encoder.parameters():
            para.requires_grad = False
        for para in self.decoder.parameters():
            para.requires_grad = False
        for para in self.head.parameters():
            para.requires_grad = True

    def replace_head(self):
        self.head = HeadNet(self.num_layer, self.input_dim, self.rep_dim, self.acti)


class MLPLinear(nn.Module):
    def __init__(self, num_layer, input_dim, rep_dim, acti):
        super(MLPLinear, self).__init__()

        self.num_layer = num_layer
        self.input_dim = input_dim
        self.rep_dim = rep_dim
        self.acti = acti

        self.encoder = EncoderNet(
            self.num_layer, self.input_dim, self.rep_dim, acti="leakyrelu"
        )
        self.head = HeadNet(0, self.input_dim, self.rep_dim, self.acti)
        self.decoder = DecoderNet(
            self.num_layer, self.input_dim, self.rep_dim, acti="leakyrelu"
        )

    # freezing and melting
    def freeze(self):
        for para in self.parameters():
            para.requires_grad = False

    def melt(self):
        for para in self.parameters():
            para.requires_grad = True

    def melt_head_only(self):
        for para in self.encoder.parameters():
            para.requires_grad = False
        for para in self.decoder.parameters():
            para.requires_grad = False
        for para in self.head.parameters():
            para.requires_grad = True

    def replace_head(self):
        self.head = HeadNet(self.num_layer, self.input_dim, self.rep_dim, self.acti)


class MLPSmooth(nn.Module):
    def __init__(self, num_layer, head_num_layer, input_dim, rep_dim, acti):
        super(MLPSmooth, self).__init__()

        self.num_layer = num_layer
        self.head_num_layer = head_num_layer
        self.input_dim = input_dim
        self.rep_dim = rep_dim
        self.acti = acti

        self.encoder = EncoderNet(
            self.num_layer, self.input_dim, self.rep_dim, acti="leakyrelu"
        )
        self.head = HeadNet(
            self.head_num_layer, self.input_dim, self.rep_dim, self.acti
        )
        self.decoder = DecoderNet(
            self.num_layer, self.input_dim, self.rep_dim, acti="leakyrelu"
        )

    # freezing and melting
    def freeze(self):
        for para in self.parameters():
            para.requires_grad = False

    def melt(self):
        for para in self.parameters():
            para.requires_grad = True

    def melt_head_only(self):
        for para in self.encoder.parameters():
            para.requires_grad = False
        for para in self.decoder.parameters():
            para.requires_grad = False
        for para in self.head.parameters():
            para.requires_grad = True

    def replace_head(self):
        self.head = HeadNet(self.num_layer, self.input_dim, self.rep_dim, self.acti)


class AudModel(nn.Module):
    def __init__(self, rep_dim):
        super(AudModel, self).__init__()

        # aud fc layer
        self.aud = nn.ModuleList()
        self.aud.append(nn.Linear(rep_dim, 1))
        self.aud.append(nn.Sigmoid())

    def forward(self, x):
        for _, fc in enumerate(self.aud):
            x = fc(x)
        return x

    def freeze(self):
        for para in self.parameters():
            para.requires_grad = False

    def melt(self):
        for para in self.parameters():
            para.requires_grad = True


class Trainer:
    def __init__(self) -> None:
        pass

    def _loss(
        self, x, y, s, lmda, lmdaF, lmdaR, model, aud_model, criterion, fair_criterion
    ):

        # to train
        model.train()
        aud_model.train()

        # weights and flattening
        y, s = y.flatten(), s.int().flatten()

        # feeding
        z = model.encoder(x)
        _, preds = model.head(z)

        # task loss
        task_loss = criterion(preds.flatten(), y)

        # fair loss
        fair_loss = 0.0
        if lmdaF > 0.0:
            z0, z1 = z[s == 0], z[s == 1]
            aud_z0, aud_z1 = aud_model(z0), aud_model(z1)
            fair_loss = fair_criterion(aud_z0, aud_z1)

        # recon loss
        recon_loss = 0.0
        if lmdaR > 0.0:
            recon = model.decoder(z)
            recon_loss = ((x - recon) ** 2).sum(dim=1).mean()

        # all loss
        loss = lmda * task_loss
        if lmdaF > 0.0:
            loss += lmdaF * fair_loss
        if lmdaR > 0.0:
            loss += lmdaR * recon_loss

        return loss

    def _train(
        self,
        train_loader,
        lmda,
        lmdaF,
        lmdaR,
        model,
        aud_model,
        criterion,
        optimizer,
        fair_criterion,
        fair_optimizer,
        aud_steps,
    ):

        losses = 0.0
        n_train = 0
        for x, y, s in train_loader:

            # initialization
            batch_size = x.size(0)
            n_train += batch_size
            x, y, s = x.to(DEVICE), y.to(DEVICE), s.to(DEVICE)

            # train encoder + head
            model.melt()
            aud_model.freeze()

            loss = self._loss(
                x, y, s, lmda, lmdaF, lmdaR, model, aud_model, criterion, fair_criterion
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # technical
            optimizer.step()
            losses += loss.item() * batch_size

            # train adversarial network
            if lmdaF > 0:
                model.freeze()
                aud_model.melt()
                for _ in range(aud_steps):
                    loss = self._loss(
                        x,
                        y,
                        s,
                        lmda,
                        lmdaF,
                        lmdaR,
                        model,
                        aud_model,
                        criterion,
                        fair_criterion,
                    )
                    loss *= -1
                    fair_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(aud_model.parameters(), 5.0)
                    fair_optimizer.step()

        return round(losses / n_train, 5)

    def _finetune(
        self,
        train_loader,
        lmda,
        lmdaF,
        lmdaR,
        model,
        aud_model,
        criterion,
        optimizer,
        fair_criterion,
    ):

        # only lmda = 1.0
        lmda, lmdaF, lmdaR = 1.0, 0.0, 0.0
        xs, zs, ys, ss = [], [], [], []
        for x, y, s in train_loader:
            # initialization
            x, y, s = x.to(DEVICE), y.to(DEVICE), s.to(DEVICE)
            # train encoder + head
            model.melt_head_only()
            aud_model.freeze()
            loss = self._loss(
                x, y, s, lmda, lmdaF, lmdaR, model, aud_model, criterion, fair_criterion
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class Evaluator:
    def __init__(self) -> None:
        pass

    def _eval(
        self,
        dataset=None,
        model=None,
        aud_model=None,
        lmda=None,
        lmdaF=None,
        lmdaR=None,
        criterion=None,
        fair_criterion=None,
    ):

        # to eval
        model.eval()
        trainer = Trainer()
        x, y, s = dataset
        x, y, s = x.to(DEVICE), y.to(DEVICE), s.to(DEVICE)
        with torch.no_grad():
            z = model.encoder(x)
            logits, preds = model.head(z)
            recon = model.decoder(z)
            logits = logits.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()

            loss = 0.0
            if criterion is not None:
                loss = trainer._loss(
                    x,
                    y,
                    s,
                    lmda,
                    lmdaF,
                    lmdaR,
                    model,
                    aud_model,
                    criterion,
                    fair_criterion,
                )
                loss = loss.item()

        s = s.flatten().cpu().numpy().astype(int)
        y = y.flatten().cpu().numpy().astype(int)

        """ utility """
        preds = preds.flatten()
        pred_labels = (preds > 0.5).astype(int)
        pred_result = pred_labels.tolist()

        # acc
        acc = accuracy_score(y, pred_labels)
        # bacc
        bacc = balanced_accuracy_score(y, pred_labels)
        # ap
        ap = average_precision_score(y, preds)

        """ fairness """
        preds0, preds1 = preds[s == 0], preds[s == 1]
        taus = np.arange(0.0, 1.0, 0.01)
        # dp
        dp = (preds0 > 0.5).mean() - (preds1 > 0.5).mean()
        dp = abs(dp)
        # mdp
        mdp = preds0.mean() - preds1.mean()
        mdp = abs(mdp)
        # vdp
        vdp = preds0.std() ** 2 - preds1.std() ** 2
        vdp = abs(vdp)
        # sdp
        dps = []
        for tau in taus:
            tau_dp = (preds0 > tau).mean() - (preds1 > tau).mean()
            dps.append(abs(tau_dp))
        sdp = np.mean(dps)

        return {
            "loss": loss,
            "acc": acc,
            "bacc": bacc,
            "ap": ap,
            "dp": dp,
            "mdp": mdp,
            "vdp": vdp,
            "sdp": sdp,
        }, pred_result


class StandardDataset:
    def __init__(self):
        self.protected_attribute_name = ""
        self.privileged_classes = []
        self.fair_variables = []

    def process(
        self,
        train,
        test,
        protected_attribute_name,
        privileged_classes,
        missing_value=[],
        features_to_drop=[],
        categorical_features=[],
        favorable_classes=[],
        normalize=True,
    ):
        cols = [
            x
            for x in train.columns
            if x
            not in (
                features_to_drop
                + [protected_attribute_name]
                + categorical_features
                + ["result"]
            )
        ]

        result = []
        for df in [train, test]:
            # drop nan values
            df = df.replace(missing_value, np.nan)
            df = df.dropna(axis=0)

            # drop useless features
            df = df.drop(columns=features_to_drop)

            # create one-hot encoding of categorical features
            df = pd.get_dummies(df, columns=categorical_features, prefix_sep="=")

            # map protected attributes to privileged or unprivileged
            pos = np.logical_or.reduce(
                np.equal.outer(privileged_classes, df[protected_attribute_name].values)
            )
            df.loc[pos, protected_attribute_name] = 1
            df.loc[~pos, protected_attribute_name] = 0
            df[protected_attribute_name] = df[protected_attribute_name].astype(int)

            # set binary labels
            pos = np.logical_or.reduce(
                np.equal.outer(favorable_classes, df["result"].values)
            )
            df.loc[pos, "result"] = 1
            df.loc[~pos, "result"] = 0
            df["result"] = df["result"].astype(int)

            result.append(df)

        # standardize numeric columns
        for col in cols:
            data = result[0][col].tolist()
            mean = np.mean(data)
            std = np.std(data)
            result[0][col] = (result[0][col] - mean) / std
            result[1][col] = (result[1][col] - mean) / std

        train = result[0]
        test = result[1]
        for col in train.columns:
            if col not in test.columns:
                test[col] = 0
        cols = train.columns
        test = test[cols]
        assert all(
            train.columns[i] == test.columns[i] for i in range(len(train.columns))
        )

        return train, test


class SIPMDataset(StandardDataset):
    def __init__(self, dataname: str):
        super(SIPMDataset, self).__init__()
        if dataname == "compas":
            self.protected_attribute_name = "race"
            self.privileged_classes = ["Caucasian"]
            filedir = parent_dir + "/MAF/data/compas/"

            if not os.path.exists(filedir):
                os.makedirs(filedir)

            if not os.path.exists(os.path.join(filedir, "compas_train.csv")):
                df = pd.read_csv(filedir + "compas-scores-two-years.csv")
                df = df.loc[df["days_b_screening_arrest"] <= 30]
                df = df.loc[df["days_b_screening_arrest"] >= -30]
                df = df.loc[df["is_recid"] != -1]
                df = df.loc[df["c_charge_degree"] != "O"]
                df = df.loc[df["score_text"] != "N/A"]

                categorical_features = ["sex", "age_cat", "c_charge_degree"]
                cols = [
                    "c_charge_degree",
                    "race",
                    "age_cat",
                    "sex",
                    "priors_count",
                    "days_b_screening_arrest",
                    "decile_score",
                    "two_year_recid",
                ]
                df = df[cols].copy()
                df = df.rename(columns={"two_year_recid": "result"})
                df.sample(frac=1, random_state=0)
                self.test = df.tail(df.shape[0] // 10 * 3)
                self.train = df.head(df.shape[0] - self.test.shape[0])
                self.train, self.test = super().process(
                    self.train,
                    self.test,
                    categorical_features=categorical_features,
                    features_to_drop=[],
                    missing_value=["?"],
                    favorable_classes=[1],
                    protected_attribute_name=self.protected_attribute_name,
                    privileged_classes=self.privileged_classes,
                )
                self.train.sample(frac=1, random_state=0)
                n = self.train.shape[0]
                self.val = self.train.tail(n // 10 * 2)
                self.train = self.train.head(n - self.val.shape[0])

                self.train.to_csv(os.path.join(filedir, "compas_train.csv"), index=None)
                self.val.to_csv(os.path.join(filedir, "compas_val.csv"), index=None)
                self.test.to_csv(os.path.join(filedir, "compas_test.csv"), index=None)
            else:
                self.train = pd.read_csv(
                    os.path.join(filedir, "compas_train.csv"), index_col=False
                )
                self.val = pd.read_csv(
                    os.path.join(filedir, "compas_val.csv"), index_col=False
                )
                self.test = pd.read_csv(
                    os.path.join(filedir, "compas_test.csv"), index_col=False
                )
            columns = self.train.columns.values
            self.fair_variables = [ele for ele in columns if "c_charge_degree" in ele]

        elif dataname == "adult":

            def preprocess(df):
                def group_edu(x):
                    if x <= 5:
                        return "<6"
                    elif x >= 13:
                        return ">12"
                    else:
                        return x

                def age_cut(x):
                    if x >= 70:
                        return ">=70"
                    else:
                        return x

                def group_race(x):
                    if x == "White":
                        return 1.0
                    else:
                        return 0.0

                # Cluster education and age attributes.
                # Limit education range
                df["education-num"] = df["education-num"].apply(lambda x: group_edu(x))
                df["education-num"] = df["education-num"].astype("category")

                # Limit age range
                df["age"] = df["age"].apply(lambda x: x // 10 * 10)
                df["age"] = df["age"].apply(lambda x: age_cut(x))

                # Group race
                df["race"] = df["race"].apply(lambda x: group_race(x))
                return df

            self.protected_attribute_name = "sex"
            self.privileged_classes = ["Male"]
            filedir = parent_dir + "/MAF/data/adult/"

            if not os.path.exists(filedir + "adult_train.csv"):
                print("Generating adult train/val/test dataset")
                self.train = pd.read_csv(filedir + "adult.data", header=None)
                self.test = pd.read_csv(filedir + "adult.test", header=None)
                columns = [
                    "age",
                    "workclass",
                    "fnlwgt",
                    "education",
                    "education-num",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "capital-gain",
                    "capital-loss",
                    "hours-per-week",
                    "native-country",
                    "result",
                ]
                self.train.columns = columns
                self.test.columns = columns
                self.train = preprocess(self.train)
                self.test = preprocess(self.test)

                categorical_features = [
                    "workclass",
                    "education",
                    "age",
                    "race",
                    "education-num",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "native-country",
                ]

                self.train, self.test = self.process(
                    self.train,
                    self.test,
                    protected_attribute_name=self.protected_attribute_name,
                    privileged_classes=self.privileged_classes,
                    missing_value=["?"],
                    features_to_drop=["fnlwgt"],
                    categorical_features=categorical_features,
                    favorable_classes=[">50K", ">50K."],
                )
                self.train.sample(frac=1, random_state=0)
                n = self.train.shape[0]
                self.val = self.train.tail(n // 10 * 2)
                self.train = self.train.head(n - self.val.shape[0])
                self.train.to_csv(os.path.join(filedir, "adult_train.csv"), index=None)
                self.val.to_csv(os.path.join(filedir, "adult_val.csv"), index=None)
                self.test.to_csv(os.path.join(filedir, "adult_test.csv"), index=None)
            else:
                self.train = pd.read_csv(
                    os.path.join(filedir, "adult_train.csv"), index_col=False
                )
                self.val = pd.read_csv(
                    os.path.join(filedir, "adult_val.csv"), index_col=False
                )
                self.test = pd.read_csv(
                    os.path.join(filedir, "adult_test.csv"), index_col=False
                )

            columns = self.train.columns.values
            self.fair_variables = [ele for ele in columns if "occupation" in ele]

    def process(
        self,
        train,
        test,
        protected_attribute_name,
        privileged_classes,
        missing_value=[],
        features_to_drop=[],
        categorical_features=[],
        favorable_classes=[],
        normalize=True,
    ):
        cols = [
            x
            for x in train.columns
            if x
            not in (
                features_to_drop
                + [protected_attribute_name]
                + categorical_features
                + ["result"]
            )
        ]

        result = []
        for df in [train, test]:
            # drop nan values
            df = df.replace(missing_value, np.nan)
            df = df.dropna(axis=0)

            # drop useless features
            df = df.drop(columns=features_to_drop)

            # create one-hot encoding of categorical features
            df = pd.get_dummies(df, columns=categorical_features, prefix_sep="=")

            # map protected attributes to privileged or unprivileged
            pos = np.logical_or.reduce(
                np.equal.outer(privileged_classes, df[protected_attribute_name].values)
            )
            df.loc[pos, protected_attribute_name] = 1
            df.loc[~pos, protected_attribute_name] = 0
            df[protected_attribute_name] = df[protected_attribute_name].astype(int)

            # set binary labels
            pos = np.logical_or.reduce(
                np.equal.outer(favorable_classes, df["result"].values)
            )
            df.loc[pos, "result"] = 1
            df.loc[~pos, "result"] = 0
            df["result"] = df["result"].astype(int)

            result.append(df)

        # standardize numeric columns
        for col in cols:
            data = result[0][col].tolist()
            mean = np.mean(data)
            std = np.std(data)
            result[0][col] = (result[0][col] - mean) / std
            result[1][col] = (result[1][col] - mean) / std

        train = result[0]
        test = result[1]
        for col in train.columns:
            if col not in test.columns:
                test[col] = 0
        cols = train.columns
        test = test[cols]
        assert all(
            train.columns[i] == test.columns[i] for i in range(len(train.columns))
        )

        return train, test


class SIPMLFR:
    def __init__(
        self,
        dataname: str = "compas",
        scaling: int = 1,
        batch_size: int = 512,
        epochs: int = 300,
        opt: str = "Adam",
        model_lr: float = 0.02,
        aud_lr: float = 0.02,
        aud_steps: int = 2,
        acti: str = "leakyrelu",
        num_layer: int = 1,
        head_net: str = "linear",
        aud_dim: int = 0,
        eval_freq: int = 10,
    ):

        # initialization hyps
        self.scaling = bool(scaling)
        self.batch_size = batch_size
        loaders = {
            "adult": SIPMDataset(dataname="adult"),
            "compas": SIPMDataset(dataname="compas"),
        }
        self.dataset = loaders[dataname]
        rep_dim = {"adult": 60, "compas": 8}
        self.rep_dim = rep_dim[dataname]
        self.epochs = epochs
        self.opt = opt
        self.aud_opt = opt
        self.model_lr, self.aud_lr = model_lr, aud_lr
        self.aud_steps = aud_steps
        self.acti = acti
        self.num_layer = num_layer
        self.head_net = head_net
        self.aud_num_layer = num_layer
        self.aud_dim = aud_dim
        self.eval_freq = eval_freq

        self.config_path = f"batch-{self.batch_size}_epoch-{self.epochs}_opt-{self.opt}_lr-{self.model_lr}_advopt-{self.aud_opt}_advlr-{self.aud_lr}_advstep-{self.aud_steps}_repdim-{self.rep_dim}_head-{self.head_net}_advlayer-{self.aud_num_layer}_advdim-{self.aud_dim}/"
        self.results_path = f"result/sipm_lfr/{dataname}/" + self.config_path
        self.model_path = f"model/sipm_lfr/{dataname}/" + self.config_path
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

        train_x, train_y, train_s = self.preprocess(mode="train")
        self.train_dataloader = self.to_dataloader(x=train_x, y=train_y, s=train_s)
        self.val_dataset = self.preprocess(mode="val")
        self.test_dataset = self.preprocess(mode="test")
        self.input_dim = self.test_dataset[0].size(1)

    def preprocess(self, mode: str):
        if mode == "train":
            df = self.dataset.train
        elif mode == "val":
            df = self.dataset.val
        elif mode == "test":
            df = self.dataset.test

        x_idx = df.columns.values.tolist()
        x_idx.remove("result")
        scaler = MinMaxScaler()
        if self.scaling:
            x = torch.from_numpy(scaler.fit_transform(df[x_idx].values)).type(
                torch.float
            )
        else:
            x = torch.from_numpy(df[x_idx].values).type(torch.float)

        y = torch.from_numpy(df["result"].values).flatten().type(torch.float)
        s = (
            torch.from_numpy(df[self.dataset.protected_attribute_name].values)
            .flatten()
            .type(torch.float)
        )
        return x, y, s

    def to_dataloader(self, x, y, s):
        tensor_dataset = TensorDataset(x, y, s)
        return DataLoader(
            tensor_dataset,
            sampler=RandomSampler(tensor_dataset),
            batch_size=self.batch_size,
        )

    def learning(self, i, seed, lmda, lmdaF, lmdaR):
        """initialization"""
        if i == 0:
            if lmda > 0:
                self.model_path = self.model_path + f"sup/fair-{lmdaF}/"
            else:
                self.results_path = self.results_path + f"unsup/fair-{lmdaF}/"
            os.makedirs(self.model_path, exist_ok=True)
            os.makedirs(self.results_path, exist_ok=True)

        """ models """
        model = MLP(
            num_layer=self.num_layer,
            input_dim=self.input_dim,
            rep_dim=self.rep_dim,
            acti=self.acti,
        )
        if self.head_net == "linear":
            model = MLPLinear(
                num_layer=self.num_layer,
                input_dim=self.input_dim,
                rep_dim=self.rep_dim,
                acti=self.acti,
            )
        elif self.head_net[0].isdigit() and (self.head_net[1:] == "smooth"):
            model = MLPSmooth(
                num_layer=self.num_layer,
                head_num_layer=int(self.head_net[0]),
                input_dim=self.input_dim,
                rep_dim=self.rep_dim,
                acti="sigmoid",
            )
        elif self.head_net[0].isdigit() and (self.head_net[1:] == "mlp"):
            model = MLP(
                num_layer=self.num_layer,
                input_dim=self.input_dim,
                rep_dim=self.rep_dim,
                acti="relu",
            )
        else:
            raise ValueError("only linear, mlp, smooth classifiers are provided!")

        model = model.to(DEVICE)
        aud_model = AudModel(rep_dim=self.rep_dim).to(DEVICE)
        print(model)
        print(aud_model)

        """ criterion and optimizers """
        criterion = nn.BCELoss().to(DEVICE)
        optimizer = getattr(torch.optim, self.opt)(model.parameters(), lr=self.model_lr)
        fair_criterion = FairLoss().to(DEVICE)
        fair_optimizer = getattr(torch.optim, self.aud_opt)(
            aud_model.parameters(), lr=self.aud_lr
        )

        """ train """
        best_epoch = [0]  # initial
        best_val = [-1e10]  # initial
        train_loss = []
        trainer = Trainer()
        evaluator = Evaluator()
        print(":::: Training ::::")
        for epoch in range(self.epochs):
            print("aud_steps", self.aud_steps)
            loss = trainer._train(
                train_loader=self.train_dataloader,
                lmda=lmda,
                lmdaF=lmdaF,
                lmdaR=lmdaR,
                model=model,
                aud_model=aud_model,
                criterion=criterion,
                optimizer=optimizer,
                fair_criterion=fair_criterion,
                fair_optimizer=fair_optimizer,
                aud_steps=self.aud_steps,
            )
            train_loss.append(loss)
            # print
            print(f"EPOCH[{epoch+1}/{self.epochs}]: loss {loss}", end="\r")
            # val
            if epoch % self.eval_freq == 0:
                val_stats, val_pred_result = evaluator._eval(
                    self.val_dataset,
                    model,
                    aud_model,
                    lmda,
                    lmdaF,
                    lmdaR,
                    criterion,
                    fair_criterion,
                )
                # check best
                if lmda == 0.0:
                    check = -val_stats["loss"]
                else:
                    check = val_stats["acc"]
                    if lmdaF > 0.0:
                        check = val_stats["acc"] - val_stats["dp"]
                if check > best_val[-1]:
                    best_epoch.append(epoch)
                    best_val.append(check)
                    torch.save(
                        model.state_dict(),
                        os.path.join(self.model_path, f"model-best.pth"),
                    )
                    # print
                    if lmda > 0:
                        print(
                            f"BEST at {epoch+1} with validation | acc: {val_stats['acc']}, DP: {val_stats['dp']}"
                        )
                    else:
                        print(
                            f"BEST at {epoch+1} with validation | loss: {val_stats['loss']}"
                        )

        """ fine tune """
        print("::: Fine-tuning :::")
        model.melt_head_only()
        for finetune_epoch in range(100):
            finetune_epoch += self.epochs
            trainer._finetune(
                self.train_dataloader,
                lmda,
                lmdaF,
                lmdaR,
                model,
                aud_model,
                criterion,
                optimizer,
                fair_criterion,
            )
            finetune_val_stats, val_pred_result = evaluator._eval(
                dataset=self.val_dataset, model=model
            )
            # check best
            check = finetune_val_stats["acc"]
            if lmdaF > 0.0:
                check = finetune_val_stats["acc"] - finetune_val_stats["dp"]
            if check > best_val[-1]:
                best_epoch.append(finetune_epoch)
                best_val.append(check)
                torch.save(
                    model.state_dict(), os.path.join(self.model_path, f"model-best.pth")
                )
                print(
                    f"BEST at {finetune_epoch+1} with validation | acc: {finetune_val_stats['acc']}, DP: {finetune_val_stats['dp']}"
                )

    def inference(self, when):
        # inference
        best_model = MLP(
            num_layer=self.num_layer,
            input_dim=self.input_dim,
            rep_dim=self.rep_dim,
            acti=self.acti,
        ).to(DEVICE)
        if self.head_net == "linear":
            best_model = MLPLinear(
                num_layer=self.num_layer,
                input_dim=self.input_dim,
                rep_dim=self.rep_dim,
                acti=self.acti,
            ).to(DEVICE)
        elif self.head_net[0].isdigit() and (self.head_net[1:] == "smooth"):
            best_model = MLPSmooth(
                num_layer=self.num_layer,
                head_num_layer=int(self.head_net[0]),
                input_dim=self.input_dim,
                rep_dim=self.rep_dim,
                acti=self.acti,
            ).to(DEVICE)
        elif self.head_net[0].isdigit() and (self.head_net[1:] == "mlp"):
            best_model = MLPSmooth(
                num_layer=self.num_layer,
                head_num_layer=int(self.head_net[0]),
                input_dim=self.input_dim,
                rep_dim=self.rep_dim,
                acti="relu",
            ).to(DEVICE)
        else:
            raise ValueError("only linear, mlp, smooth classifiers are provided!")

        best_model.load_state_dict(
            torch.load(
                os.path.join(self.model_path, f"model-{when}.pth"), weights_only=True
            )
        )
        best_model.eval()
        evaluator = Evaluator()
        test_stats, test_pred_result = evaluator._eval(
            dataset=self.test_dataset, model=best_model
        )
        print("BEST test results:")
        print(test_stats)

        return test_stats

    def run(
        self,
        run_five: int = 1,
        lmda: float = 0.0,
        lmdaF: float = 0.0,
        lmdaR: float = 1.0,
    ):
        seeds = [2021, 2022, 2023, 2024, 2025] if bool(run_five) else [2021]
        stats = {}
        for i, seed in enumerate(seeds):
            print(f"::: STEP {i+1} with seed {seed} :::")
            self.learning(i, seed, lmda, lmdaF, lmdaR)
            stat = self.inference(when="best")
            print("stat", stat)
            if i == 0:
                stats = deepcopy(stat)
                for key in stats.keys():
                    stats[key] = [stats[key]]
            else:
                for key in stats.keys():
                    stats[key].append(stat[key])
        save_result(self.results_path, stats)
        return stats


def save_result(path, stats):
    # mean and median
    mean_stats, median_stats = deepcopy(stats), deepcopy(stats)
    for key in stats.keys():
        mean_stats[key] = np.mean(stats[key]).item()
        median_stats[key] = np.median(stats[key]).item()

    # save
    with open(os.path.join(path, f"mean_result.json"), "w") as f:
        f.write(json.dumps(mean_stats, indent=4))
        f.close()
    with open(os.path.join(path, f"median_result.json"), "w") as f:
        f.write(json.dumps(median_stats, indent=4))
        f.close()


def sIPM_LFR_fit(
    config_path: str = parent_dir
    + "/MAF/algorithms/config/sipm_lfr/sipm_lfr_config.yaml",
):

    config = yaml_config_hook(config_path)
    dataname = config["dataset"]
    scaling = config["scaling"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    opt = config["opt"]
    model_lr = config["model_lr"]
    aud_lr = config["aud_lr"]
    aud_steps = config["aud_steps"]
    acti = config["acti"]
    num_layer = config["num_layer"]
    head_net = config["head_net"]
    aud_dim = config["aud_dim"]
    eval_freq = config["eval_freq"]
    run_five = config["run_five"]
    lmda = config["lmda"]
    lmdaF = config["lmdaF"]
    lmdaR = config["lmdaR"]

    runner = SIPMLFR(
        dataname,
        scaling,
        batch_size,
        epochs,
        opt,
        model_lr,
        aud_lr,
        aud_steps,
        acti,
        num_layer,
        head_net,
        aud_dim,
        eval_freq,
    )

    result = runner.run(run_five, lmda, lmdaF, lmdaR)
    return result


if __name__ == "__main__":
    result = sIPM_LFR_fit()
    print("Result", result)
