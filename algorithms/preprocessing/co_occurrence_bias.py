import json
import random, sys, os
import tqdm
import numpy as np

from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

keywords_to_remove = {"?": "?", ":": ":", "!": "!", ".": ".", ",": ",", ";": ";"}
for w in stopwords.words("english"):
    keywords_to_remove[w] = w


def text_normalization_without_lemmatization(text):
    result = []
    tokens = word_tokenize(text)

    for token in tokens:
        token_low = token.lower()
        if token_low in keywords_to_remove:
            continue
        result.append(token_low)
    return result


class CooccurrenceMatrix:
    def __init__(
        self,
        pretraining_dataset_name: str = "pile",
        data_statistics_dir: str = os.environ["PYTHONPATH"]
        + "/MAF/data/co-occurrence-bias/data_statistics",
    ):
        with open(
            f"{data_statistics_dir}/entity_set/merged/all_subjects.json", "r"
        ) as fin:
            self.subject_idx = json.load(fin)
        with open(
            f"{data_statistics_dir}/entity_set/merged/all_objects.json", "r"
        ) as fin:
            self.object_idx = json.load(fin)
        with open(
            f"{data_statistics_dir}/entity_set/merged/all_entities.json", "r"
        ) as fin:
            self.entity_idx = json.load(fin)

        self.subject_inverted_idx = {v: k for k, v in self.subject_idx.items()}
        self.object_inverted_idx = {v: k for k, v in self.object_idx.items()}
        self.entity_inverted_idx = {v: k for k, v in self.entity_idx.items()}

        self.cooccurrence_matrix = np.load(
            f"{data_statistics_dir}/cooccurrence_matrix/{pretraining_dataset_name}/cooccurrence_matrix.npy"
        )
        self.occurrence_matrix = np.load(
            f"{data_statistics_dir}/occurrence_matrix/{pretraining_dataset_name}/occurrence_matrix.npy"
        )

    def count(self, word):
        idx = self.get_entity_idx(word)
        if idx is not None:
            return self.occurrence_matrix[idx].item()
        else:
            return -1

    def coo_count(self, subj, obj):
        s_idx = self.get_subject_idx(subj)
        o_idx = self.get_object_idx(obj)
        if s_idx is not None and o_idx is not None:
            return self.cooccurrence_matrix[s_idx][o_idx].item()
        else:
            return -1

    def get_subject_idx(self, word):
        return self.subject_idx.get(word, None)

    def get_object_idx(self, word):
        return self.object_idx.get(word, None)

    def get_entity_idx(self, word):
        return self.entity_idx.get(word, None)

    def get_subject(self, idx):
        return self.subject_inverted_idx.get(idx, "<empty>")

    def get_object(self, idx):
        return self.object_inverted_idx.get(idx, "<empty>")

    def get_entity(self, idx):
        return self.entity_inverted_idx.get(idx, "<empty>")


class CooccurrenceDebiasing:
    def __init__(
        self,
        dataset_name: str = "LAMA_TREx",
        pretraining_dataset_name: str = "pile",
        protected=None,
        data_dir: str = os.environ["PYTHONPATH"] + "/MAF/data/co-occurrence-bias",
    ):

        random.seed(1)
        np.random.seed(1)

        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.pretraining_dataset_name = pretraining_dataset_name
        self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        dataset_loaders = {"LAMA_TREx": self.load_lama_trex}
        if self.dataset_name not in dataset_loaders:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        print("Loading..")
        # no test set
        self.dataset_orig_train = dataset_loaders[self.dataset_name]()
        self.coo_matrix = (
            CooccurrenceMatrix(pretraining_dataset_name=self.pretraining_dataset_name)
            if self.pretraining_dataset_name is not None
            else None
        )
        print("Loading Complete!")

    def load_lama_trex(self):
        with open(f"{self.data_dir}/LAMA_TREx/train.json", "r") as fin:
            f_train = json.load(fin)
        return f_train

    def debiasing_with_undersampling_fit(self):
        uid_prob_per_rel = defaultdict(list)

        for example in tqdm.tqdm(
            self.dataset_orig_train, desc="Calculating Cooccurence.."
        ):
            rel = example["rel_id"]
            subj = example["subj"]
            obj = example["output"]
            subj = " ".join(text_normalization_without_lemmatization(subj))
            obj = " ".join(text_normalization_without_lemmatization(obj))

            subj_count = self.coo_matrix.count(subj)
            obj_count = self.coo_matrix.count(obj)
            subj_obj_count = self.coo_matrix.coo_count(subj, obj)

            cond_prob = subj_obj_count / subj_count if subj_count > 0 else 0

            uid_prob_per_rel[rel].append((example["uid"], cond_prob))

        for rel in uid_prob_per_rel:
            uid_prob_per_rel[rel] = sorted(
                uid_prob_per_rel[rel], key=lambda x: x[1], reverse=True
            )

        filtering_ratios = [0.1, 0.3, 0.5]

        for filtering_ratio in tqdm.tqdm(
            filtering_ratios, desc="Making debiased datasets.."
        ):
            random_filtered_uids = []
            condprob_filtered_uids = []

            for rel in uid_prob_per_rel:
                random_filtered_idx = np.random.choice(
                    range(len(uid_prob_per_rel[rel])),
                    size=int(len(uid_prob_per_rel[rel]) * filtering_ratio),
                    replace=False,
                )
                condprob_filtered_idx = list(
                    range(int(len(uid_prob_per_rel[rel]) * filtering_ratio))
                )

                for idx, uid_prob in enumerate(uid_prob_per_rel[rel]):
                    if idx not in random_filtered_idx:
                        random_filtered_uids.append(uid_prob[0])
                    if idx not in condprob_filtered_idx:
                        condprob_filtered_uids.append(uid_prob[0])

            random_filtered_dataset = []
            condprob_filtered_dataset = []

            for example in self.dataset_orig_train:
                uid = example["uid"]
                if uid in random_filtered_uids:
                    random_filtered_dataset.append(example)
                if uid in condprob_filtered_uids:
                    condprob_filtered_dataset.append(example)

            with open(
                f"{self.data_dir}/{self.dataset_name}/train_{self.pretraining_dataset_name}_random_filtered_"
                + str(filtering_ratio)
                + ".json",
                "w",
            ) as fout:
                json.dump(random_filtered_dataset, fout)
            with open(
                f"{self.data_dir}/{self.dataset_name}/train_{self.pretraining_dataset_name}_debiased_"
                + str(filtering_ratio)
                + ".json",
                "w",
            ) as fout:
                json.dump(condprob_filtered_dataset, fout)

        print(
            f"End! The debiased dataset is stored in: {self.data_dir}/{self.dataset_name}/"
        )
        return condprob_filtered_dataset

    def run(self):
        # Note that this function need
        debiased_data = self.debiasing_with_undersampling_fit()

        return debiased_data


if __name__ == "__main__":
    cooc_db = CooccurrenceDebiasing()
    debiased_data = cooc_db.run()
