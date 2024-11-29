import os, sys
import torch
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")

from MAF.metric import common_utils
from MAF.datamodule.dataset import PubFigDataset, aifData, RawDataSet

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("Train on [[[  {}  ]]] device.".format(device))


def mapping(class_vector):
    # Flatten
    class_vector = class_vector.ravel()

    cls2val = np.unique(class_vector)
    val2cls = dict(zip(cls2val, range(len(cls2val))))

    converted_vector = [val2cls[v] for v in class_vector]

    return cls2val, val2cls, converted_vector


class EMA:
    def __init__(self, label, alpha=0.9):
        self.label = label
        self.alpha = alpha
        self.parameter = torch.zeros(label.size(0))
        self.updated = torch.zeros(label.size(0))

    def update(self, data, index):
        self.parameter[index] = (
            self.alpha * self.parameter[index]
            + (1 - self.alpha * self.updated[index]) * data
        )
        self.updated[index] = 1

    def max_loss(self, label):
        label_index = np.where(self.label == label)[0]
        return self.parameter[label_index].max()


class MultiDimAverageMeter(object):
    def __init__(self, dims):
        self.dims = dims
        self.cum = torch.zeros(np.prod(dims))
        self.cnt = torch.zeros(np.prod(dims))
        self.idx_helper = torch.arange(np.prod(dims), dtype=torch.long).reshape(*dims)

    def add(self, vals, idxs):
        flattened_idx = torch.stack(
            [self.idx_helper[tuple(idxs[i])] for i in range(idxs.size(0))],
            dim=0,
        )
        self.cum.index_add_(0, flattened_idx, vals.view(-1).float())
        self.cnt.index_add_(
            0, flattened_idx, torch.ones_like(vals.view(-1), dtype=torch.float)
        )

    def get_mean(self):
        return (self.cum / self.cnt).reshape(*self.dims)

    def reset(self):
        self.cum.zero_()
        self.cnt.zero_()


class LfFDataset(Dataset):
    def __init__(self, X, y, z):
        self.X = torch.Tensor(X).to(device)
        self.y = torch.Tensor(y).type(torch.long).to(device)
        self.z = torch.Tensor(z).type(torch.long).to(device)

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, i):
        X = self.X[i] / 255
        X = X.view(X.size(0), -1)

        return X, self.y[i], self.z[i]


class MLP(nn.Module):
    def __init__(self, num_classes, input_size):
        super(MLP, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(100, num_classes)

    def forward(self, x, return_feat=False):
        x = x.view(x.size(0), -1) / 255
        feat = self.feature(x)
        x = feat.clone()
        x = self.classifier(x)

        if return_feat:
            return x, feat
        else:
            return x


class GeneralizedCELoss(nn.Module):
    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        loss_weight = (Yg.squeeze().detach() ** self.q) * self.q
        loss = F.cross_entropy(logits, targets, reduction="none") * loss_weight

        return loss


class LfFmodel:
    def __init__(
        self,
        rawdata,
        n_epoch,
        batch_size,
        learning_rate,
        image_shape,
        attr_idx_list=[],
        train_size=0.8,
        seed=777,
    ):
        # Data
        train_X, test_X, train_y, test_y, train_z, test_z = train_test_split(
            rawdata.feature_only,
            rawdata.target,
            rawdata.bias,
            train_size=train_size,
            random_state=seed,
        )

        train_attr = np.column_stack(
            (train_X[:, attr_idx_list], train_z.reshape(-1, 1), train_y.reshape(-1, 1))
        )
        self.train_attr = torch.LongTensor(train_attr).to(device)
        self.train_target_attr = self.train_attr[:, -1]
        self.train_bias_attr = self.train_attr[:, -2]

        test_attr = np.column_stack(
            (test_X[:, attr_idx_list], test_z.reshape(-1, 1), test_y.reshape(-1, 1))
        )
        self.test_attr = torch.LongTensor(test_attr).to(device)
        self.test_target_attr = self.test_attr[:, -1]
        self.test_bias_attr = self.test_attr[:, -2]

        self.attr_dims = []
        self.attr_dims.append(torch.max(self.train_target_attr).item() + 1)
        self.attr_dims.append(torch.max(self.train_bias_attr).item() + 1)

        self.train_dataset = LfFDataset(train_X, train_y, train_z)
        self.test_dataset = LfFDataset(test_X, test_y, test_z)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        self.batch_size = batch_size
        self.n_epoch = n_epoch
        # self.learning_rate = learning_rate

        self.classes = np.unique(rawdata.target)
        self.n_class = len(self.classes)

        self.protects = np.unique(rawdata.bias)
        self.n_protect = len(self.protects)

        self.input_size = np.prod(np.array(image_shape))

        self.model_b = MLP(self.n_class, self.input_size)
        self.model_d = MLP(self.n_class, self.input_size)

        self.optimizer_b = torch.optim.Adam(self.model_b.parameters(), lr=learning_rate)
        self.optimizer_d = torch.optim.Adam(self.model_d.parameters(), lr=learning_rate)

        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.bias_criterion = GeneralizedCELoss()

    def train(self):
        print("Training start")
        for ep in range(self.n_epoch):
            self.model_b.train()
            self.model_d.train()

            num_updated = 0

            for idx, (X, y, z) in enumerate(self.train_loader):
                sample_loss_ema_b = EMA(y, alpha=0.7)
                sample_loss_ema_d = EMA(y, alpha=0.7)

                logit_b = self.model_b(X)
                logit_d = self.model_d(X)

                loss_b = self.criterion(logit_b, y).cpu().detach()
                loss_d = self.criterion(logit_d, y).cpu().detach()
                loss_per_sample_b = loss_b
                loss_per_sample_d = loss_d

                index_arr = np.array(range(self.batch_size)) + idx
                index_arr = np.where(index_arr < len(index_arr), index_arr, 0)

                sample_loss_ema_b.update(loss_b, index_arr)
                sample_loss_ema_d.update(loss_d, index_arr)

                loss_b = sample_loss_ema_b.parameter[index_arr].clone().detach()
                loss_d = sample_loss_ema_d.parameter[index_arr].clone().detach()

                label_cpu = y.cpu()

                for c in y.unique():
                    class_index = np.where(label_cpu == c)[0]
                    max_loss_b = sample_loss_ema_b.max_loss(c)
                    max_loss_d = sample_loss_ema_d.max_loss(c)
                    loss_b[class_index] /= max_loss_b
                    loss_d[class_index] /= max_loss_d

                loss_weight = loss_b / (loss_b + loss_d + 1e-8)

                loss_b_update = self.bias_criterion(logit_b, y)
                loss_d_update = self.criterion(logit_d, y) * loss_weight

                loss = loss_b_update.mean() + loss_d_update.mean()

                num_updated += loss_weight.mean().item() * X.size(0)

                self.optimizer_b.zero_grad()
                self.optimizer_d.zero_grad()
                loss.backward()
                self.optimizer_b.step()
                self.optimizer_d.step()

            valid_attrwise_accs_b, _, _, _ = self.evaluate(self.model_b)
            valid_attrwise_accs_d, bias_np, label_np, pred_np = self.evaluate(
                self.model_d
            )
            valid_accs_b = torch.mean(valid_attrwise_accs_b)
            valid_accs_d = torch.mean(valid_attrwise_accs_d)

            print(
                "Epoch {}/{} :  Loss {:.04f} | valid_accuracy_b {:.04f} | valid_accuracy_d {:.04f}".format(
                    ep, self.n_epoch, loss, valid_accs_b, valid_accs_d
                )
            )
        print("Training end")
        return self.model_d

    def evaluate(self, model):
        model.eval()
        acc = 0
        attrwise_acc_meter = MultiDimAverageMeter(self.attr_dims)
        predict_list = []
        label_list = []
        bias_list = []
        for idx, (X, y, z) in enumerate(self.test_loader):
            with torch.no_grad():
                logit = model(X)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == y).long()

            label_list += list(y.numpy())
            bias_list += list(z.numpy())
            predict_list += list(pred.numpy())

            attr = torch.LongTensor(
                np.column_stack((y.reshape(-1, 1), z.reshape(-1, 1)))
            ).to(device)
            # print(attr)
            # print(correct)
            # print(self.attr_dims)

            attrwise_acc_meter.add(correct.cpu(), attr.cpu())
            # attrwise_acc_meter.add(correct.cpu(), attr)

        accs = attrwise_acc_meter.get_mean()

        return accs, bias_list, label_list, predict_list


class LearningFromFairness:
    def __init__(self, dataset_name="pubfig", protected="Heavy Makeup"):
        self.dataset_name = dataset_name
        self.protected = protected
        self.load_and_preprocess_data()
        self.device = device
        self.image_shape = (3, 64, 64)
        self.batch_size = 32
        self.num_epochs = 20
        self.z_dim = 20
        self.learning_rate = 0.01

    def load_and_preprocess_data(self):
        if self.dataset_name == "pubfig":
            self.pubfig = PubFigDataset()
            self.dataset = self.pubfig.to_dataset()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.data = self.dataset["aif_dataset"]
        self.privileged_groups = [{self.protected: 1}]
        self.unprivileged_groups = [{self.protected: 0}]

        fltn_img = np.array(
            [img.ravel() for img in self.dataset["image_list"]], dtype="int"
        )
        self.rds = RawDataSet(
            x=fltn_img, y=self.dataset["target"], z=self.dataset["bias"]
        )

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.z_train,
            self.z_test,
        ) = train_test_split(
            self.rds.feature_only,
            self.rds.target,
            self.rds.bias,
            train_size=0.8,
            random_state=777,
        )

        testset = pd.DataFrame(self.X_test)
        testset[self.protected] = self.z_test
        testset[self.dataset["aif_dataset"].label_names[0]] = self.y_test

        self.dataset_orig = aifData(
            df=testset,
            label_name=self.dataset["aif_dataset"].label_names[0],
            favorable_classes=[self.dataset["aif_dataset"].favorable_label],
            protected_attribute_names=self.dataset[
                "aif_dataset"
            ].protected_attribute_names,
            privileged_classes=self.dataset[
                "aif_dataset"
            ].privileged_protected_attributes,
        )

    def baseline_fit(self):
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        pred = model.predict(self.X_test)
        return pred

    def lff_fit(self):
        lff = LfFmodel(
            self.rds,
            self.num_epochs,
            self.batch_size,
            self.learning_rate,
            self.image_shape,
        )
        lff.train()
        # Prediction
        _, _, _, pred = lff.evaluate(lff.model_d)
        pred = np.array(pred).ravel()
        return pred

    def compute_metrics(self, dataset_pred):
        test_pred = self.dataset_orig.copy()
        test_pred.labels = dataset_pred
        return common_utils.compute_metrics(
            self.dataset_orig,
            test_pred,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )

    def run(self):
        lr_pred = self.baseline_fit()
        lff_pred = self.lff_fit()
        print(len(lff_pred))
        metrics_orig = self.compute_metrics(lr_pred)
        metrics_transf = self.compute_metrics(lff_pred)
        return metrics_orig, metrics_transf


if __name__ == "__main__":
    lff_run = LearningFromFairness(dataset_name="pubfig")
    metrics_orig, metrics_transf = lff_run.run()
    print("Metrics for original data:")
    print(metrics_orig)
    print("\nMetrics for transformed data:")
    print(metrics_transf)
