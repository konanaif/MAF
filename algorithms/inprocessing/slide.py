import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import confusion_matrix
from collections import OrderedDict

from MAF.algorithms.preprocessing.optim_preproc_helpers.data_prepro_function import (
    load_preproc_data_adult,
    load_preproc_data_german,
    load_preproc_data_compas,
)


class SlideFairClassifier:
    def __init__(
        self,
        dataset_name: str = "adult",
        protected: str = "sex",
        use_cuda=False,
        **kwargs,
    ):
        self.dataset_name = dataset_name
        self.protected = protected
        self.use_cuda = use_cuda
        np.random.seed(1)
        torch.random.manual_seed(1)
        torch.manual_seed(1)
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

        self.dataset_orig_train.labels = self.dataset_orig_train.labels.flatten()
        self.dataset_orig_test.labels = self.dataset_orig_test.labels.flatten()

        self.dataset_orig_train.sensitives = self.dataset_orig_train.features[
            :, np.array(self.dataset_orig_train.feature_names) == self.protected
        ].flatten()
        self.dataset_orig_test.sensitives = self.dataset_orig_test.features[
            :, np.array(self.dataset_orig_test.feature_names) == self.protected
        ].flatten()

        self.input_dim = self.dataset_orig_train.features.shape[1]

        # make torch dataloaders (x, y, s)
        self.train_dset = torch.utils.data.TensorDataset(
            torch.from_numpy(self.dataset_orig_train.features).float(),
            torch.from_numpy(self.dataset_orig_train.labels).long(),
            torch.from_numpy(self.dataset_orig_train.sensitives).long(),
        )
        self.test_dset = torch.utils.data.TensorDataset(
            torch.from_numpy(self.dataset_orig_test.features).float(),
            torch.from_numpy(self.dataset_orig_test.labels).long(),
            torch.from_numpy(self.dataset_orig_test.sensitives).long(),
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dset, shuffle=True, drop_last=True, batch_size=128
        )
        self.traineval_loader = torch.utils.data.DataLoader(
            self.train_dset, shuffle=False, drop_last=False, batch_size=128
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dset, shuffle=False, drop_last=False, batch_size=128
        )

    def _compute_slide_penalty(self, pred, gamma=0.7, tau=0.1):
        term1_ = torch.relu(pred - gamma) / tau
        term2_ = torch.relu(pred - gamma - tau) / tau
        loss = term1_ - term2_
        return loss.mean()

    def _get_colored_probs(self, model, inputs, sensitives):
        logits = model(inputs)
        probs = torch.softmax()
        pn0_, pn1_ = probs[:, 0][sensitives == 0], probs[:, 0][sensitives == 1]

        return pn0_, pn1_

    def _set_optimization(self):
        model = nn.Sequential(
            nn.Linear(self.input_dim, 50), nn.ReLU(), nn.Linear(50, 2)
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.999
        )
        configs = {
            "criterion": criterion,
            "optimizer": optimizer,
            "scheduler": scheduler,
        }
        return model, configs

    def _train_single_iter(self, model, configs, inputs, labels, sensitives, reg=0.0):
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)
        loss = configs["criterion"](logits, labels)
        util_loss = loss.item()
        if reg > 0.0:
            fair_0_ = self._compute_slide_penalty(probs[sensitives == 0])
            fair_1_ = self._compute_slide_penalty(probs[sensitives == 1])
            loss += reg * (fair_0_ - fair_1_).abs()
        configs["optimizer"].zero_grad()
        loss.backward()
        configs["optimizer"].step()
        configs["scheduler"].step()
        return model, loss.item(), util_loss

    def _eval_single_iter(self, model):
        test_preds = []
        for inputs, labels, sensitives in self.test_loader:
            if self.use_cuda:
                inputs, labels, sensitives = (
                    inputs.cuda(),
                    labels.cuda(),
                    sensitives.cuda(),
                )
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).detach().cpu().numpy()
            test_preds.append(preds)
        test_preds = np.concatenate(test_preds)
        return test_preds

    def fit(self, epochs=200, reg=0.0, scope_name="debiased_classifier"):

        if scope_name == "debiased_classifier":
            assert reg > 0.0, ValueError("reg should be larger than 0 if debiasing!")
        elif scope_name == "plain_classifier":
            assert reg == 0.0, ValueError("reg should be 0 if plain!")
        model, configs = self._set_optimization()

        final_model, final_loss = None, 1e8
        for epoch in range(epochs):
            cnt, epoch_loss, epoch_util_loss = 0, 0.0, 0.0
            model = model.cuda() if self.use_cuda else model
            model.train()
            for inputs, labels, sensitives in self.train_loader:
                if self.use_cuda:
                    inputs, labels, sensitives = (
                        inputs.cuda(),
                        labels.cuda(),
                        sensitives.cuda(),
                    )
                model, loss, util_loss = self._train_single_iter(
                    model, configs, inputs, labels, sensitives, reg
                )
                cnt += inputs.shape[0]
                epoch_loss += loss * inputs.shape[0]
                epoch_util_loss += util_loss * inputs.shape[0]
            epoch_loss /= cnt
            epoch_util_loss /= cnt
            if epoch_loss < final_loss:
                final_model = model
                final_loss = epoch_util_loss
            print(
                f"{scope_name} | [{epoch+1}/{epochs}] loss: {round(epoch_loss, 4)}",
                end="\r",
            )

        final_model.eval()
        with torch.no_grad():
            test_pred = self._eval_single_iter(final_model)

        return test_pred

    def compute_metrics(self, preds):
        sensitives = self.dataset_orig_test.sensitives
        labels = self.dataset_orig_test.labels
        acc = (labels == preds).astype(float).mean()
        dp = np.abs(
            preds[sensitives == 0].astype(float).mean()
            - preds[sensitives == 1].astype(float).mean()
        )
        metrics = {"acc": acc, "dp": dp}
        return metrics

    def run(self, reg=3.0):
        orig_test_pred = self.fit(epochs=100, reg=0.0, scope_name="plain_classifier")
        metrics_orig = self.compute_metrics(orig_test_pred)
        transf_test_pred = self.fit(
            epochs=100, reg=reg, scope_name="debiased_classifier"
        )
        metrics_transform = self.compute_metrics(transf_test_pred)
        return metrics_orig, metrics_transform


if __name__ == "__main__":
    use_cuda = False
    reg = 3.0
    slide = SlideFairClassifier(
        dataset_name="adult", protected="sex", use_cuda=use_cuda
    )
    metrics_orig, metrics_transf = slide.run(reg)
    print("Metrics for original data:")
    print(metrics_orig)
    print("\nMetrics for transformed data:")
    print(metrics_transf)
