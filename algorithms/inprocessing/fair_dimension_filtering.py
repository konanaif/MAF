import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings("ignore")

from MAF.metric import common_utils
from MAF.utils.common import fix_seed
from MAF.datamodule.dataset import CelebADataset, aifData

fix_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Trained on [[[  {}  ]]] device.".format(device))


class MultiDimAverageMeter(object):
    def __init__(self, dims=None):
        if dims != None:
            self.dims = dims
            self.cum = torch.zeros(np.prod(dims))
            self.cnt = torch.zeros(np.prod(dims))
            self.idx_helper = torch.arange(np.prod(dims), dtype=torch.long).reshape(
                *dims
            )
        else:
            self.dims = None
            self.cum = torch.tensor(0.0)
            self.cnt = torch.tensor(0.0)

    def add(self, vals, idxs=None):
        if self.dims:
            flattened_idx = torch.stack(
                [self.idx_helper[tuple(idxs[i])] for i in range(idxs.size(0))],
                dim=0,
            )
            self.cum.index_add_(0, flattened_idx, vals.view(-1).float())
            self.cnt.index_add_(
                0, flattened_idx, torch.ones_like(vals.view(-1), dtype=torch.float)
            )
        else:
            self.cum += vals.sum().float()
            self.cnt += vals.numel()

    def get_mean(self):
        if self.dims:
            return (self.cum / self.cnt).reshape(*self.dims)
        else:
            return self.cum / self.cnt

    def reset(self):
        if self.dims:
            self.cum.zero_()
            self.cnt.zero_()
        else:
            self.cum = torch.tensor(0.0)
            self.cnt = torch.tensor(0.0)


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > torch.mean(input).item()).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        sigmoid_output = torch.sigmoid(input)
        sigmoid_grad = sigmoid_output * (1 - sigmoid_output)

        positive_grad = torch.clamp(grad_output, max=0)

        grad_input = positive_grad * sigmoid_grad

        grad_input = torch.clamp(grad_input, min=0)

        return grad_input


class MaskingModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MaskingModel, self).__init__()
        self.mask_scores = nn.Parameter(torch.randn(input_dim) * 0.001)
        self.classifier = nn.Linear(input_dim, output_dim, bias=False)
        torch.nn.init.sparse_(self.classifier.weight, sparsity=0.02, std=0.01)

    def forward(self, x):
        if self.training:
            mask = STEFunction.apply(F.sigmoid(self.mask_scores))
            out = self.classifier(x * mask)
        else:
            out = self.classifier(x)
        return out


class Filter_Net(nn.Module):
    def __init__(self, output_dim):
        super(Filter_Net, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        input_dim = self.backbone.fc.in_features
        self.output_dim = output_dim
        self.backbone.fc = MaskingModel(input_dim, output_dim=self.output_dim)

    def forward(self, x):
        tilde_z = self.backbone(x)
        return tilde_z


class DFDataset(Dataset):
    def __init__(self, IMG_DIR, ATTR_DIR_, META_DIR, split="train"):
        self.IMG_DIR = IMG_DIR
        ATTR_DIR = ATTR_DIR_.replace(".txt", ".csv")
        self.meta_data = pd.read_csv(ATTR_DIR)
        self.split_df = pd.read_csv(META_DIR)
        self.meta_data["split"] = self.split_df["partition"]
        self.split_dict = {"train": 0, "val": 1, "test": 2}
        self.meta_data = self.meta_data[
            self.meta_data["split"] == self.split_dict[split]
        ]
        self.image_idx = self.meta_data["image_id"].values
        self.label_idx = self.meta_data["Blond_Hair"].values
        self.sens_idx = self.meta_data["Male"].values
        self.attr = np.vstack((self.label_idx, self.sens_idx)).T

        class_counts = torch.bincount(torch.from_numpy(self.attr[:, 0]).cpu())
        class_weights = 1.0 / class_counts.float()
        self.class_weights = class_weights / class_weights.sum()

        if split == "train":
            self.transform = T.Compose(
                [
                    T.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = T.Compose(
                [
                    T.Resize((256, 256)),
                    T.CenterCrop((224, 224)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self):
        return self.image_idx.size

    def __getitem__(self, idx):
        img = Image.open(self.IMG_DIR + "/" + self.image_idx[idx])
        img = self.transform(img)
        attr = self.attr[idx]

        return (img, attr, idx)


class DimFiltering:
    def __init__(
        self,
        train_loader,
        valid_loader,
        test_loader,
        n_epoch,
        batch_size,
        learning_rate,
        patience,
        weight_decay,
        momentum,
        weight,
        device,
        seed=42,
    ):

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device
        self.model_dir = os.environ["PYTHONPATH"] + "/MAF/model/FairFiltering/"

        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.weight = weight.to(device)

        self.seed = seed
        self.n_class = 2
        self.n_protect = 2

        self.model = Filter_Net(output_dim=self.n_class)

        # TODO: Baseline model(Resnet50)
        self.baseline = models.resnet50(pretrained=True)
        in_features = self.baseline.fc.in_features
        self.baseline.fc = nn.Linear(in_features, self.n_class, bias=False)
        # TODO: criterion
        self.criterion = nn.CrossEntropyLoss(weight=self.weight, reduction="none")

    def train(self, models_):
        print("Train start")
        file_path = self.model_dir
        optimizer = torch.optim.SGD(
            models_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epoch
        )
        models_ = models_.to(self.device)

        BEST_SCORE: float = 0
        PATIENCE: int = 0
        for ep in range(self.n_epoch):
            models_.train()
            pbar = tqdm.tqdm(self.train_loader)

            for idx, data in enumerate(pbar):
                optimizer.zero_grad()
                img, attr, _ = data
                img = img.to(device)
                attr = attr.to(device)
                target = attr[
                    :, 0
                ]  # Target: 0번째 column, Protected Attribute: 1번째 column

                pred = models_(img)
                loss = self.criterion(pred, target)

                loss_for_update = loss.mean() + (self.weight_decay) * (
                    torch.norm(models_.fc.classifier.weight.data, p=1)
                )

                loss_for_update.backward()
                optimizer.step()

                # print log
                pbar.set_postfix(
                    epoch=f"{ep}/{self.n_epoch}",
                    loss="{:.4f}".format(loss.mean().detach().cpu()),
                )
            pbar.close()
            scheduler.step()

            # TODO: validate & save model
            models_.eval()
            total = 0
            correct_sum = 0
            group_acc_meter = MultiDimAverageMeter(
                dims=(self.n_class, self.n_protect)
            )  # 각 그룹에 대한 정확도 계산
            with torch.no_grad():
                pbar = tqdm.tqdm(self.valid_loader)
                for idx, data in enumerate(pbar):
                    img, attr, _ = data
                    img = img.to(device)
                    attr = attr.to(device)
                    target = attr[:, 0]

                    pred = models_(img)
                    loss = self.criterion(pred, target)

                    correct = (pred.argmax(dim=1) == target).long()
                    group_acc_meter.add(correct.cpu(), attr.cpu())

                    total += img.size(0)
                    correct_sum += torch.sum(correct).item()
                    acc = correct_sum / total
                    pbar.set_postfix(
                        loss="{:.4f}, acc = {:.4f}".format(
                            loss.mean().detach().cpu(), acc
                        )
                    )
                pbar.close()

            group_accs = group_acc_meter.get_mean()
            worst_group_acc = group_accs.min().item()

            # 조기 종료 모듈
            if worst_group_acc > BEST_SCORE:
                BEST_SCORE = worst_group_acc
                print(
                    "*" * 15,
                    "Best Score: {:.4f}".format(worst_group_acc * 100),
                    "*" * 15,
                )
                state_dict = {
                    "best score": worst_group_acc,
                    "state_dict": models_.state_dict(),
                }
                if isinstance(models_, Filter_Net):
                    torch.save(state_dict, self.model_dir + "Filter_model.th")
                else:
                    torch.save(state_dict, self.model_dir + "baseline.th")
                PATIENCE = 0
            else:
                PATIENCE += 1
            if PATIENCE > self.patience:
                break

        print("Train done.")

    def evaluation(self, models_):
        if isinstance(models_, Filter_Net):
            state_dict = torch.load(self.model_dir + "Filter_model.th")
        else:
            state_dict = torch.load(self.model_dir + "baseline.th")
        models_.load_state_dict(state_dict["state_dict"], strict=True)
        models_.eval()
        models_ = models_.to(self.device)
        print("*" * 15, "Test Start", "*" * 15)
        # TODO: Test -> load model
        predictions = []
        attrs = []
        pbar = tqdm.tqdm(self.test_loader)
        print("self.test_loader", len(self.test_loader))
        with torch.no_grad():
            for idx, (data) in enumerate(pbar):
                img, attr, _ = data
                img = img.to(device)
                attr = attr.to(device)
                target = attr[:, 0]

                pred = models_(img)
                correct = (pred.argmax(dim=1) == target).long()

                predictions.append(correct.cpu())
                attrs.append(attr.cpu())
            attrs = torch.cat(attrs)
            predictions = torch.cat(predictions)

        print("Evaluation finished.")
        return predictions, attrs


class FairDimFilter:
    def __init__(self, dataset_name="celeba", protected="Male"):
        self.dataset_name = dataset_name
        self.protected = protected
        self.image_shape = (3, 224, 224)
        self.model_dir = os.environ["PYTHONPATH"] + "/MAF/model/FairFiltering/"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 100
        self.n_epoch = 100
        self.learning_rate = 0.003
        self.patience = 20
        self.momentum = 0.9
        self.weight_decay = 1e-04

        self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        if self.dataset_name == "celeba":
            self.celeba = CelebADataset()
            self.celeba.to_dataset()  # 데이터 전처리 및 csv 저장
        elif self.dataset_name == "other_dataset":
            self.other_dataset = OtherDataset()
            self.dataset = self.other_dataset.to_dataset()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        # TODO: CelebA 데이터셋 불러오기(train, valid, test)
        self.train_dataset = DFDataset(
            self.celeba.IMAGE_DIR,
            self.celeba.ATTR_FILE,
            self.celeba.META_DATA,
            split="train",
        )
        self.valid_dataset = DFDataset(
            self.celeba.IMAGE_DIR,
            self.celeba.ATTR_FILE,
            self.celeba.META_DATA,
            split="val",
        )
        self.test_dataset = DFDataset(
            self.celeba.IMAGE_DIR,
            self.celeba.ATTR_FILE,
            self.celeba.META_DATA,
            split="test",
        )

        # TODO: 데이터 로더 생성(train, valid, test)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
        )

    def dim_filter_fit(self):
        # TODO: 1. main function -> class로 구현  & 2. train function call하기
        self.dim_filter = DimFiltering(
            self.train_loader,
            self.valid_loader,
            self.test_loader,
            self.n_epoch,
            self.batch_size,
            self.learning_rate,
            self.patience,
            self.weight_decay,
            self.momentum,
            self.train_dataset.class_weights,
            self.device,
        )
        filter_model_path = self.model_dir + "Filter_model.th"
        if os.path.exists(filter_model_path):
            print("Loading existing filter model...")
            self.dim_filter.model.load_state_dict(
                torch.load(filter_model_path), strict=False
            )
        else:
            print("Training new filter model...")
            self.dim_filter.train(self.dim_filter.model)
        pred, attrs = self.dim_filter.evaluation(self.dim_filter.model)
        return pred, attrs

    def baseline_fit(self):
        self.dim_filter = DimFiltering(
            self.train_loader,
            self.valid_loader,
            self.test_loader,
            self.n_epoch,
            self.batch_size,
            self.learning_rate,
            self.patience,
            self.weight_decay,
            self.momentum,
            self.train_dataset.class_weights,
            self.device,
        )
        baseline_path = self.model_dir + "baseline.th"
        if os.path.exists(baseline_path):
            print("Loading existing baseline model...")
            self.dim_filter.model.load_state_dict(
                torch.load(baseline_path), strict=False, map_location=device
            )
        else:
            print("Training new baseline model...")
            self.dim_filter.train(self.dim_filter.baseline)

        preds, attrs = self.dim_filter.evaluation(self.dim_filter.baseline)
        return preds, attrs

    def compute_metrics(self, pred, attrs):
        accuracy_meter = MultiDimAverageMeter(
            [self.dim_filter.n_class, self.dim_filter.n_protect]
        )
        accuracy_meter.add(pred, attrs)
        return torch.min(accuracy_meter.get_mean()).item()

    def run(self):
        lr_pred, attrs = self.baseline_fit()
        dim_filter_pred, attrs = self.dim_filter_fit()
        metrics_org = {}
        metrics_transf = {}

        metrics_org["acc"] = self.compute_metrics(lr_pred, attrs)
        metrics_transf["acc"] = self.compute_metrics(dim_filter_pred, attrs)
        return metrics_org, metrics_transf


if __name__ == "__main__":
    fdf = FairDimFilter(dataset_name="celeba")
    metrics_orig, metrics_transf = fdf.run()
    print("Metrics for original data:")
    print(metrics_orig)
    print("\nMetrics for transformed data:")
    print(metrics_transf)
