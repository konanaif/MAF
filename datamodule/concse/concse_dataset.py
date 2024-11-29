import pandas as pd
import sys

parent_dir = sys.path[1]


class LoadNLI:
    @staticmethod
    def en_train():
        train_path = (
            parent_dir + "/MAF/data/Koglish_dataset/Koglish_NLI/en_train.csv"
        )
        train_data = pd.read_csv(train_path, sep=",")
        return train_data

    @staticmethod
    def cross_train():
        train_path = (
            parent_dir + "/MAF/data/Koglish_dataset/Koglish_NLI/cs_train.csv"
        )
        train_data = pd.read_csv(train_path, sep=",")
        return train_data


class LoadSTSB:
    @staticmethod
    def en_valid():
        valid_path = (
            parent_dir + "/MAF/data/Koglish_dataset/Koglish_STS/STS_B/en_valid.csv"
        )
        valid_data = pd.read_csv(valid_path, sep=",")
        return valid_data

    def en_test():
        test_path = (
            parent_dir + "/MAF/data/Koglish_dataset/Koglish_STS/STS_B/cs_test.csv"
        )
        test_data = pd.read_csv(test_path, sep=",")
        return test_data

    @staticmethod
    def cross_valid():
        valid_path = (
            parent_dir + "/MAF/data/Koglish_dataset/Koglish_STS/STS_B/cs_valid.csv"
        )
        valid_data = pd.read_csv(valid_path, sep=",")
        return valid_data

    def cross_test():
        test_path = (
            parent_dir + "/MAF/data/Koglish_dataset/Koglish_STS/STS_B/cs_test.csv"
        )
        test_data = pd.read_csv(test_path, sep=",")
        return test_data
