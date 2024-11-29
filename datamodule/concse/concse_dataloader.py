from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler
import torch


class Tokenization(Dataset):
    def __init__(self, data, tokenizer, max_length, labels=None):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        encoding = self.tokenizer(
            str(data),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        results = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten(),
        }
        if self.labels is not None:
            labels = self.labels[item]
            labels = torch.tensor(labels, dtype=torch.float)
            results["labels"] = labels

        return results


class Iterator:
    def __init__(self, df, tokenizer, train_config, args):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = train_config.max_length
        self.batch_size = train_config.batch_size
        self.random_seed = args.random_seed

    def _get_sampler(self, ds):
        return RandomSampler(
            ds, generator=torch.Generator().manual_seed(self.random_seed)
        )

    def train_en_sent1_loader(self):
        ds = Tokenization(
            data=self.df["sentence1"].to_numpy(),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        return DataLoader(ds, batch_size=self.batch_size, sampler=self._get_sampler(ds))

    def train_en_sent2_loader(self):
        ds = Tokenization(
            data=self.df["sentence2"].to_numpy(),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        return DataLoader(ds, batch_size=self.batch_size, sampler=self._get_sampler(ds))

    def train_en_hard_neg_loader(self):
        ds = Tokenization(
            data=self.df["hard_neg"].to_numpy(),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        return DataLoader(ds, batch_size=self.batch_size, sampler=self._get_sampler(ds))

    def train_cross_sent1_loader(self):
        ds = Tokenization(
            data=self.df["cross1"].to_numpy(),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        return DataLoader(ds, batch_size=self.batch_size, sampler=self._get_sampler(ds))

    def train_cross_sent2_loader(self):
        ds = Tokenization(
            data=self.df["cross2"].to_numpy(),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        return DataLoader(ds, batch_size=self.batch_size, sampler=self._get_sampler(ds))

    def train_cross_hard_neg_loader(self):
        ds = Tokenization(
            data=self.df["cross_hard_neg"].to_numpy(),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        return DataLoader(ds, batch_size=self.batch_size, sampler=self._get_sampler(ds))

    def eval_en_sent1_loader(self):  # stsb
        ds = Tokenization(
            data=self.df["sentence1"].to_numpy(),
            labels=self.df["label"].to_numpy(),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        return DataLoader(ds, batch_size=self.batch_size, sampler=self._get_sampler(ds))

    def eval_en_sent2_loader(self):  # stsb
        ds = Tokenization(
            data=self.df["sentence2"].to_numpy(),
            labels=self.df["label"].to_numpy(),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        return DataLoader(ds, batch_size=self.batch_size, sampler=self._get_sampler(ds))

    def eval_cross_sent1_loader(self):  # stsb
        ds = Tokenization(
            data=self.df["cross1"].to_numpy(),
            labels=self.df["label"].to_numpy(),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        return DataLoader(ds, batch_size=self.batch_size, sampler=self._get_sampler(ds))

    def eval_cross_sent2_loader(self):  # stsb
        ds = Tokenization(
            data=self.df["cross2"].to_numpy(),
            labels=self.df["label"].to_numpy(),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        return DataLoader(ds, batch_size=self.batch_size, sampler=self._get_sampler(ds))


class CustomDataLoader:
    def __init__(self, iterator, train_config, args, iter_type):
        self.iterator = iterator
        self.train_config = train_config
        self.args = args
        self.iter_type = iter_type
        self.reset()

    def reset(self):
        if self.iter_type == "train" and (
            self.args.lang_type == "en2en" or self.args.lang_type == "en2cross"
        ):
            self.sent1_loader = iter(self.iterator.train_en_sent1_loader())
            self.sent2_loader = iter(self.iterator.train_en_sent2_loader())
            self.hard_neg_loader = iter(self.iterator.train_en_hard_neg_loader())
        elif self.iter_type == "valid" and (
            self.args.lang_type == "en2en" or self.args.lang_type == "en2cross"
        ):
            self.sent1_loader = iter(self.iterator.eval_en_sent1_loader())
            self.sent2_loader = iter(self.iterator.eval_en_sent2_loader())
            self.hard_neg_loader = None
        elif self.iter_type == "test" and (self.args.lang_type == "en2en"):
            self.sent1_loader = iter(self.iterator.eval_en_sent1_loader())
            self.sent2_loader = iter(self.iterator.eval_en_sent2_loader())
            self.hard_neg_loader = None
        elif self.iter_type == "test" and (self.args.lang_type == "en2cross"):
            self.sent1_loader = iter(self.iterator.eval_cross_sent1_loader())
            self.sent2_loader = iter(self.iterator.eval_cross_sent2_loader())
            self.hard_neg_loader = None

        elif (
            self.iter_type == "train"
            and (self.args.lang_type == "cross2cross")
            and self.args.method != "ConCSE"
        ):
            self.sent1_loader = iter(self.iterator.train_cross_sent1_loader())
            self.sent2_loader = iter(self.iterator.train_cross_sent2_loader())
            self.hard_neg_loader = iter(self.iterator.train_cross_hard_neg_loader())
        elif self.iter_type == "valid" and (self.args.lang_type == "cross2cross"):
            self.sent1_loader = iter(self.iterator.eval_cross_sent1_loader())
            self.sent2_loader = iter(self.iterator.eval_cross_sent2_loader())
            self.hard_neg_loader = None
        elif self.iter_type == "test" and (self.args.lang_type == "cross2cross"):
            self.sent1_loader = iter(self.iterator.eval_cross_sent1_loader())
            self.sent2_loader = iter(self.iterator.eval_cross_sent2_loader())
            self.hard_neg_loader = None

        elif self.iter_type == "train" and self.args.method == "ConCSE":
            self.sent1_loader = iter(self.iterator.train_en_sent1_loader())
            self.sent2_loader = iter(self.iterator.train_en_sent2_loader())
            self.en_hard_neg_loader = iter(self.iterator.train_en_hard_neg_loader())
            self.cross1_loader = iter(self.iterator.train_cross_sent1_loader())
            self.cross2_loader = iter(self.iterator.train_cross_sent2_loader())
            self.cross_hard_neg_loader = iter(
                self.iterator.train_cross_hard_neg_loader()
            )

    def __len__(self):
        return len(self.sent1_loader)

    def __iter__(self):
        return self

    def __next__(self):

        if self.iter_type == "train" and self.args.method == "ConCSE":
            try:
                sent1_batch = next(self.sent1_loader)
                sent2_batch = next(self.sent2_loader)
                en_hard_neg_batch = next(self.en_hard_neg_loader)
                cross1_batch = next(self.cross1_loader)
                cross2_batch = next(self.cross2_loader)
                cross_hard_neg_batch = next(self.cross_hard_neg_loader)
            except StopIteration:
                raise
            return (
                sent1_batch,
                sent2_batch,
                en_hard_neg_batch,
                cross1_batch,
                cross2_batch,
                cross_hard_neg_batch,
            )

        else:
            try:
                sent0_batch = next(self.sent1_loader)
                sent1_batch = next(self.sent2_loader)
                hard_neg_batch = (
                    next(self.hard_neg_loader) if self.iter_type == "train" else None
                )
            except StopIteration:
                raise

            if self.iter_type == "train":
                return sent0_batch, sent1_batch, hard_neg_batch
            else:  # eval일때
                return sent0_batch, sent1_batch


class CustomCollator:
    def __init__(self, data_loaders, args, iter_type):
        self.data_loaders = data_loaders
        self.args = args
        self.iter_type = iter_type

    def __len__(self):
        return len(self.data_loaders)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_type == "train" and self.args.method == "ConCSE":
            (
                sent1_batch,
                sent2_batch,
                en_hard_neg_batch,
                cross1_batch,
                cross2_batch,
                cross_hard_neg_batch,
            ) = next(self.data_loaders)
            input_ids = torch.stack(
                [
                    sent1_batch["input_ids"],
                    sent2_batch["input_ids"],
                    en_hard_neg_batch["input_ids"],
                    cross1_batch["input_ids"],
                    cross2_batch["input_ids"],
                    cross_hard_neg_batch["input_ids"],
                ],
                dim=1,
            )
            attention_mask = torch.stack(
                [
                    sent1_batch["attention_mask"],
                    sent2_batch["attention_mask"],
                    en_hard_neg_batch["attention_mask"],
                    cross1_batch["attention_mask"],
                    cross2_batch["attention_mask"],
                    cross_hard_neg_batch["attention_mask"],
                ],
                dim=1,
            )
            token_type_ids = torch.stack(
                [
                    sent1_batch["token_type_ids"],
                    sent2_batch["token_type_ids"],
                    en_hard_neg_batch["token_type_ids"],
                    cross1_batch["token_type_ids"],
                    cross2_batch["token_type_ids"],
                    cross_hard_neg_batch["token_type_ids"],
                ],
                dim=1,
            )
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }

        else:
            if self.iter_type == "train":
                sent0_batch, sent1_batch, hard_neg_batch = next(self.data_loaders)
                input_ids = torch.stack(
                    [
                        sent0_batch["input_ids"],
                        sent1_batch["input_ids"],
                        hard_neg_batch["input_ids"],
                    ],
                    dim=1,
                )
                attention_mask = torch.stack(
                    [
                        sent0_batch["attention_mask"],
                        sent1_batch["attention_mask"],
                        hard_neg_batch["attention_mask"],
                    ],
                    dim=1,
                )
                token_type_ids = torch.stack(
                    [
                        sent0_batch["token_type_ids"],
                        sent1_batch["token_type_ids"],
                        hard_neg_batch["token_type_ids"],
                    ],
                    dim=1,
                )
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                }
            else:
                sent0_batch, sent1_batch = next(self.data_loaders)
                input_ids = torch.stack(
                    [sent0_batch["input_ids"], sent1_batch["input_ids"]], dim=1
                )
                attention_mask = torch.stack(
                    [sent0_batch["attention_mask"], sent1_batch["attention_mask"]],
                    dim=1,
                )
                token_type_ids = torch.stack(
                    [sent0_batch["token_type_ids"], sent1_batch["token_type_ids"]],
                    dim=1,
                )
                labels = sent0_batch["labels"]
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                    "labels": labels,
                }
