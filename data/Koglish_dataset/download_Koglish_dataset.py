import os
from datasets import load_dataset
import pandas as pd


####=====Download Koglish_GLUE datset=====####
class LoadSST2:
    @staticmethod
    def get_english_dataset():
        save_dir = "./Koglish_GLUE/SST2"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_GLUE",
            data_files={
                "train": "SST2/en_train.csv",
                "validation": "SST2/en_valid.csv",
                "test": "SST2/en_test.csv",
            },
        )

        train_data = dataset["train"]
        valid_data = dataset["validation"]
        test_data = dataset["test"]

        train_save_path = os.path.join(save_dir, "en_train.csv")
        valid_save_path = os.path.join(save_dir, "en_valid.csv")
        test_save_path = os.path.join(save_dir, "en_test.csv")

        train_data.to_csv(train_save_path, index=False)
        valid_data.to_csv(valid_save_path, index=False)
        test_data.to_csv(test_save_path, index=False)

        print(
            f"English Train data of Koglih_GLUE on ****SST2**** saved to {train_save_path}"
        )
        print(
            f"English Validation data of Koglih_GLUE on ****SST2**** saved to {valid_save_path}"
        )
        print(
            f"English Test data Koglih_GLUE on ****SST2**** saved to {test_save_path}"
        )

        return train_data, valid_data, test_data

    @staticmethod
    def get_code_swhiched_dataset():
        save_dir = "./Koglish_GLUE/SST2"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_GLUE",
            data_files={
                "train": "SST2/cs_train.csv",
                "validation": "SST2/cs_valid.csv",
                "test": "SST2/cs_test.csv",
            },
        )

        train_data = dataset["train"]
        valid_data = dataset["validation"]
        test_data = dataset["test"]

        train_save_path = os.path.join(save_dir, "cs_train.csv")
        valid_save_path = os.path.join(save_dir, "cs_valid.csv")
        test_save_path = os.path.join(save_dir, "cs_test.csv")

        train_data.to_csv(train_save_path, index=False)
        valid_data.to_csv(valid_save_path, index=False)
        test_data.to_csv(test_save_path, index=False)

        print(
            f"Code-Switched Train data of Koglih_GLUE on ****SST2**** saved to {train_save_path}"
        )
        print(
            f"Code-Switched Validation data of Koglih_GLUE on ****SST2**** saved to {valid_save_path}"
        )
        print(
            f"Code-Switched Test data Koglih_GLUE on ****SST2**** saved to {test_save_path}"
        )

        return train_data, valid_data, test_data


class LoadMRPC:
    @staticmethod
    def get_english_dataset():
        save_dir = "./Koglish_GLUE/MRPC"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_GLUE",
            data_files={
                "train": "MRPC/en_train.csv",
                "validation": "MRPC/en_valid.csv",
                "test": "MRPC/en_test.csv",
            },
        )

        train_data = dataset["train"]
        valid_data = dataset["validation"]
        test_data = dataset["test"]

        train_save_path = os.path.join(save_dir, "en_train.csv")
        valid_save_path = os.path.join(save_dir, "en_valid.csv")
        test_save_path = os.path.join(save_dir, "en_test.csv")

        train_data.to_csv(train_save_path, index=False)
        valid_data.to_csv(valid_save_path, index=False)
        test_data.to_csv(test_save_path, index=False)

        print(
            f"English Train data of Koglih_GLUE on ****MRPC**** saved to {train_save_path}"
        )
        print(
            f"English Validation data of Koglih_GLUE on ****MRPC**** saved to {valid_save_path}"
        )
        print(
            f"English Test data Koglih_GLUE on ****MRPC**** saved to {test_save_path}"
        )

        return train_data, valid_data, test_data

    @staticmethod
    def get_code_swhiched_dataset():
        save_dir = "./Koglish_GLUE/MRPC"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_GLUE",
            data_files={
                "train": "MRPC/cs_train.csv",
                "validation": "MRPC/cs_valid.csv",
                "test": "MRPC/cs_test.csv",
            },
        )

        train_data = dataset["train"]
        valid_data = dataset["validation"]
        test_data = dataset["test"]

        train_save_path = os.path.join(save_dir, "cs_train.csv")
        valid_save_path = os.path.join(save_dir, "cs_valid.csv")
        test_save_path = os.path.join(save_dir, "cs_test.csv")

        train_data.to_csv(train_save_path, index=False)
        valid_data.to_csv(valid_save_path, index=False)
        test_data.to_csv(test_save_path, index=False)

        print(
            f"Code-Switched Train data of Koglih_GLUE on ****MRPC**** saved to {train_save_path}"
        )
        print(
            f"Code-Switched Validation data of Koglih_GLUE on ****MRPC**** saved to {valid_save_path}"
        )
        print(
            f"Code-Switched Test data Koglih_GLUE on ****MRPC**** saved to {test_save_path}"
        )

        return train_data, valid_data, test_data


class LoadCOLA:
    @staticmethod
    def get_english_dataset():
        save_dir = "./Koglish_GLUE/COLA"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_GLUE",
            data_files={
                "train": "COLA/en_train.csv",
                "validation": "COLA/en_valid.csv",
                "test": "COLA/en_test.csv",
            },
        )

        train_data = dataset["train"]
        valid_data = dataset["validation"]
        test_data = dataset["test"]

        train_save_path = os.path.join(save_dir, "en_train.csv")
        valid_save_path = os.path.join(save_dir, "en_valid.csv")
        test_save_path = os.path.join(save_dir, "en_test.csv")

        train_data.to_csv(train_save_path, index=False)
        valid_data.to_csv(valid_save_path, index=False)
        test_data.to_csv(test_save_path, index=False)

        print(
            f"English Train data of Koglih_GLUE on ****COLA**** saved to {train_save_path}"
        )
        print(
            f"English Validation data of Koglih_GLUE on ****COLA**** saved to {valid_save_path}"
        )
        print(
            f"English Test data Koglih_GLUE on ****COLA**** saved to {test_save_path}"
        )

        return train_data, valid_data, test_data

    @staticmethod
    def get_code_swhiched_dataset():
        save_dir = "./Koglish_GLUE/COLA"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_GLUE",
            data_files={
                "train": "COLA/cs_train.csv",
                "validation": "COLA/cs_valid.csv",
                "test": "COLA/cs_test.csv",
            },
        )

        train_data = dataset["train"]
        valid_data = dataset["validation"]
        test_data = dataset["test"]

        train_save_path = os.path.join(save_dir, "cs_train.csv")
        valid_save_path = os.path.join(save_dir, "cs_valid.csv")
        test_save_path = os.path.join(save_dir, "cs_test.csv")

        train_data.to_csv(train_save_path, index=False)
        valid_data.to_csv(valid_save_path, index=False)
        test_data.to_csv(test_save_path, index=False)

        print(
            f"Code-Switched Train data of Koglih_GLUE on ****COLA**** saved to {train_save_path}"
        )
        print(
            f"Code-Switched Validation data of Koglih_GLUE on ****COLA**** saved to {valid_save_path}"
        )
        print(
            f"Code-Switched Test data Koglih_GLUE on ****COLA**** saved to {test_save_path}"
        )

        return train_data, valid_data, test_data


class LoadRTE:
    @staticmethod
    def get_english_dataset():
        save_dir = "./Koglish_GLUE/RTE"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_GLUE",
            data_files={
                "train": "RTE/en_train.csv",
                "validation": "RTE/en_valid.csv",
                "test": "RTE/en_test.csv",
            },
        )

        train_data = dataset["train"]
        valid_data = dataset["validation"]
        test_data = dataset["test"]

        train_save_path = os.path.join(save_dir, "en_train.csv")
        valid_save_path = os.path.join(save_dir, "en_valid.csv")
        test_save_path = os.path.join(save_dir, "en_test.csv")

        train_data.to_csv(train_save_path, index=False)
        valid_data.to_csv(valid_save_path, index=False)
        test_data.to_csv(test_save_path, index=False)

        print(
            f"English Train data of Koglih_GLUE on ****RTE**** saved to {train_save_path}"
        )
        print(
            f"English Validation data of Koglih_GLUE on ****RTE**** saved to {valid_save_path}"
        )
        print(f"English Test data Koglih_GLUE on ****RTE**** saved to {test_save_path}")

        return train_data, valid_data, test_data

    @staticmethod
    def get_code_swhiched_dataset():
        save_dir = "./Koglish_GLUE/RTE"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_GLUE",
            data_files={
                "train": "RTE/cs_train.csv",
                "validation": "RTE/cs_valid.csv",
                "test": "RTE/cs_test.csv",
            },
        )

        train_data = dataset["train"]
        valid_data = dataset["validation"]
        test_data = dataset["test"]

        train_save_path = os.path.join(save_dir, "cs_train.csv")
        valid_save_path = os.path.join(save_dir, "cs_valid.csv")
        test_save_path = os.path.join(save_dir, "cs_test.csv")

        train_data.to_csv(train_save_path, index=False)
        valid_data.to_csv(valid_save_path, index=False)
        test_data.to_csv(test_save_path, index=False)

        print(
            f"Code-Switched Train data of Koglih_GLUE on ****RTE**** saved to {train_save_path}"
        )
        print(
            f"Code-Switched Validation data of Koglih_GLUE on ****RTE**** saved to {valid_save_path}"
        )
        print(
            f"Code-Switched Test data Koglih_GLUE on ****RTE**** saved to {test_save_path}"
        )

        return train_data, valid_data, test_data


class LoadSTSB:
    @staticmethod
    def get_english_dataset():
        save_dir = "./Koglish_GLUE/STS_B"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_GLUE",
            data_files={
                "train": "STS_B/en_train.csv",
                "validation": "STS_B/en_valid.csv",
                "test": "STS_B/en_test.csv",
            },
        )

        train_data = dataset["train"]
        valid_data = dataset["validation"]
        test_data = dataset["test"]

        train_save_path = os.path.join(save_dir, "en_train.csv")
        valid_save_path = os.path.join(save_dir, "en_valid.csv")
        test_save_path = os.path.join(save_dir, "en_test.csv")

        train_data.to_csv(train_save_path, index=False)
        valid_data.to_csv(valid_save_path, index=False)
        test_data.to_csv(test_save_path, index=False)

        print(
            f"English Train data of Koglih_GLUE on ****STS_B**** saved to {train_save_path}"
        )
        print(
            f"English Validation data of Koglih_GLUE on ****STS_B**** saved to {valid_save_path}"
        )
        print(
            f"English Test data Koglih_GLUE on ****STS_B**** saved to {test_save_path}"
        )

        return train_data, valid_data, test_data

    @staticmethod
    def get_code_swhiched_dataset():
        save_dir = "./Koglish_GLUE/STS_B"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_GLUE",
            data_files={
                "train": "STS_B/cs_train.csv",
                "validation": "STS_B/cs_valid.csv",
                "test": "STS_B/cs_test.csv",
            },
        )

        train_data = dataset["train"]
        valid_data = dataset["validation"]
        test_data = dataset["test"]

        train_save_path = os.path.join(save_dir, "cs_train.csv")
        valid_save_path = os.path.join(save_dir, "cs_valid.csv")
        test_save_path = os.path.join(save_dir, "cs_test.csv")

        train_data.to_csv(train_save_path, index=False)
        valid_data.to_csv(valid_save_path, index=False)
        test_data.to_csv(test_save_path, index=False)

        print(
            f"Code-Switched Train data of Koglih_GLUE on ****STS_B**** saved to {train_save_path}"
        )
        print(
            f"Code-Switched Validation data of Koglih_GLUE on ****STS_B**** saved to {valid_save_path}"
        )
        print(
            f"Code-Switched Test data Koglih_GLUE on ****STS_B**** saved to {test_save_path}"
        )

        return train_data, valid_data, test_data


class LoadQNLI:
    @staticmethod
    def get_english_dataset():
        # Koglish_GLUE 폴더 경로 설정
        save_dir = "./Koglish_GLUE/QNLI"

        # 폴더가 없으면 생성
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 데이터셋 로드
        dataset = load_dataset(
            "Jangyeong/Koglish_GLUE",
            data_files={
                "train": "QNLI/en_train.csv",
                "validation": "QNLI/en_valid.csv",
                "test": "QNLI/en_test.csv",
            },
        )
        # 각각의 데이터셋을 가져오기
        train_data = dataset["train"]
        valid_data = dataset["validation"]
        test_data = dataset["test"]

        # 각각의 데이터셋을 지정한 경로에 CSV 파일로 저장
        train_save_path = os.path.join(save_dir, "en_train.csv")
        valid_save_path = os.path.join(save_dir, "en_valid.csv")
        test_save_path = os.path.join(save_dir, "en_test.csv")

        # Dataset을 CSV 파일로 저장
        train_data.to_csv(train_save_path, index=False)
        valid_data.to_csv(valid_save_path, index=False)
        test_data.to_csv(test_save_path, index=False)

        print(
            f"English Train data of Koglih_GLUE on ****QNLI**** saved to {train_save_path}"
        )
        print(
            f"English Validation data of Koglih_GLUE on ****QNLI**** saved to {valid_save_path}"
        )
        print(
            f"English Test data Koglih_GLUE on ****QNLI**** saved to {test_save_path}"
        )

        return train_data, valid_data, test_data

    @staticmethod
    def get_code_swhiched_dataset():
        # Koglish_GLUE 폴더 경로 설정
        save_dir = "./Koglish_GLUE/QNLI"

        # 폴더가 없으면 생성
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 데이터셋 로드
        dataset = load_dataset(
            "Jangyeong/Koglish_GLUE",
            data_files={
                "train": "QNLI/cs_train.csv",
                "validation": "QNLI/cs_valid.csv",
                "test": "QNLI/cs_test.csv",
            },
        )
        # 각각의 데이터셋을 가져오기
        train_data = dataset["train"]
        valid_data = dataset["validation"]
        test_data = dataset["test"]

        # 각각의 데이터셋을 지정한 경로에 CSV 파일로 저장
        train_save_path = os.path.join(save_dir, "cs_train.csv")
        valid_save_path = os.path.join(save_dir, "cs_valid.csv")
        test_save_path = os.path.join(save_dir, "cs_test.csv")

        # Dataset을 CSV 파일로 저장
        train_data.to_csv(train_save_path, index=False)
        valid_data.to_csv(valid_save_path, index=False)
        test_data.to_csv(test_save_path, index=False)

        print(
            f"Code-Switched Train data of Koglih_GLUE on ****QNLI**** saved to {train_save_path}"
        )
        print(
            f"Code-Switched Validation data of Koglih_GLUE on ****QNLI**** saved to {valid_save_path}"
        )
        print(
            f"Code-Switched Test data Koglih_GLUE on ****QNLI**** saved to {test_save_path}"
        )

        return train_data, valid_data, test_data


####=====Download Koglish_NLI datset=====####
class LoadNLI:
    @staticmethod
    def get_english_dataset():
        save_dir = "./Koglish_NLI"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_NLI", data_files={"en_train": "en_train.csv"}
        )
        en_train_data = dataset["en_train"]
        en_train_save_path = os.path.join(save_dir, "en_train.csv")
        en_train_data.to_csv(en_train_save_path, index=False)
        print(f"English Train data of Koglish_NLI saved to {en_train_save_path}")
        return en_train_data

    @staticmethod
    def get_code_swhiched_dataset():
        save_dir = "./Koglish_NLI"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_NLI",
            data_files={
                "cs_train": "cs_train.csv",
            },
        )
        cs_train_data = dataset["cs_train"]
        cs_train_save_path = os.path.join(save_dir, "cs_train.csv")
        cs_train_data.to_csv(cs_train_save_path, index=False)
        print(f"Code-Switched train data of Koglish_NLI saved to {cs_train_save_path}")
        return cs_train_data


####=====Download Koglish_STS(STS12~STS16, STSB,SICK) datset=====####
class LoadSTSB_for_ConCSE:
    @staticmethod
    def get_english_dataset():
        save_dir = "./Koglish_STS/STS_B"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_STS",
            data_files={
                "en_valid": "STS_B/en_valid.csv",
                "en_test": "STS_B/en_test.csv",
            },
        )
        en_valid_data = dataset["en_valid"]
        en_test_data = dataset["en_test"]

        en_valid_save_path = os.path.join(save_dir, "en_valid.csv")
        en_test_save_path = os.path.join(save_dir, "en_test.csv")

        en_valid_data.to_csv(en_valid_save_path, index=False)
        en_test_data.to_csv(en_test_save_path, index=False)

        print(f"English Train data of Koglish_STS/STS_B saved to {en_valid_save_path}")
        print(f"English Train data of Koglish_STS/STS_B saved to {en_test_save_path}")

        return en_valid_data, en_test_data

    @staticmethod
    def get_code_swhiched_dataset():
        save_dir = "./Koglish_STS/STS_B"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_STS",
            data_files={
                "cs_valid": "STS_B/cs_valid.csv",
                "cs_test": "STS_B/cs_test.csv",
            },
        )
        cs_valid_data = dataset["cs_valid"]
        cs_test_data = dataset["cs_test"]

        cs_valid_save_path = os.path.join(save_dir, "cs_valid.csv")
        cs_test_save_path = os.path.join(save_dir, "cs_test.csv")

        cs_valid_data.to_csv(cs_valid_save_path, index=False)
        cs_test_data.to_csv(cs_test_save_path, index=False)

        print(
            f"Code-Switched Valid,Test data of Koglish_STS/STS_B saved to {cs_valid_save_path}"
        )
        print(
            f"Code-Switched Valid,Test data of Koglish_STS/STS_B saved to {cs_test_save_path}"
        )

        return en_valid_data, en_test_data


class LoadSTS12:
    @staticmethod
    def get_english_dataset():
        save_dir = "./Koglish_STS/STS_12"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_STS",
            data_files={
                "en_valid": "STS_12/en_valid.csv",
                "en_test": "STS_12/en_test.csv",
            },
        )
        en_valid_data = dataset["en_valid"]
        en_test_data = dataset["en_test"]

        en_valid_save_path = os.path.join(save_dir, "en_valid.csv")
        en_test_save_path = os.path.join(save_dir, "en_test.csv")

        en_valid_data.to_csv(en_valid_save_path, index=False)
        en_test_data.to_csv(en_test_save_path, index=False)

        print(f"English Train data of Koglish_STS/STS_12 saved to {en_valid_save_path}")
        print(f"English Train data of Koglish_STS/STS_12 saved to {en_test_save_path}")

        return en_valid_data, en_test_data

    @staticmethod
    def get_code_swhiched_dataset():
        save_dir = "./Koglish_STS/STS_12"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_STS",
            data_files={
                "cs_valid": "STS_12/cs_valid.csv",
                "cs_test": "STS_12/cs_test.csv",
            },
        )
        cs_valid_data = dataset["cs_valid"]
        cs_test_data = dataset["cs_test"]

        cs_valid_save_path = os.path.join(save_dir, "cs_valid.csv")
        cs_test_save_path = os.path.join(save_dir, "cs_test.csv")

        cs_valid_data.to_csv(cs_valid_save_path, index=False)
        cs_test_data.to_csv(cs_test_save_path, index=False)

        print(
            f"Code-Switched Valid,Test data of Koglish_STS/STS_12 saved to {cs_valid_save_path}"
        )
        print(
            f"Code-Switched Valid,Test data of Koglish_STS/STS_12 saved to {cs_test_save_path}"
        )

        return cs_valid_data, cs_test_data


class LoadSTS13:
    @staticmethod
    def get_english_dataset():
        save_dir = "./Koglish_STS/STS_13"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_STS",
            data_files={
                "en_valid": "STS_13/en_valid.csv",
                "en_test": "STS_13/en_test.csv",
            },
        )
        en_valid_data = dataset["en_valid"]
        en_test_data = dataset["en_test"]

        en_valid_save_path = os.path.join(save_dir, "en_valid.csv")
        en_test_save_path = os.path.join(save_dir, "en_test.csv")

        en_valid_data.to_csv(en_valid_save_path, index=False)
        en_test_data.to_csv(en_test_save_path, index=False)

        print(f"English Train data of Koglish_STS/STS_13 saved to {en_valid_save_path}")
        print(f"English Train data of Koglish_STS/STS_13 saved to {en_test_save_path}")

        return en_valid_data, en_test_data

    @staticmethod
    def get_code_swhiched_dataset():
        save_dir = "./Koglish_STS/STS_13"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_STS",
            data_files={
                "cs_valid": "STS_13/cs_valid.csv",
                "cs_test": "STS_13/cs_test.csv",
            },
        )
        cs_valid_data = dataset["cs_valid"]
        cs_test_data = dataset["cs_test"]

        cs_valid_save_path = os.path.join(save_dir, "cs_valid.csv")
        cs_test_save_path = os.path.join(save_dir, "cs_test.csv")

        cs_valid_data.to_csv(cs_valid_save_path, index=False)
        cs_test_data.to_csv(cs_test_save_path, index=False)

        print(
            f"Code-Switched Valid,Test data of Koglish_STS/STS_13 saved to {cs_valid_save_path}"
        )
        print(
            f"Code-Switched Valid,Test data of Koglish_STS/STS_13 saved to {cs_test_save_path}"
        )

        return cs_valid_data, cs_test_data


class LoadSTS14:
    @staticmethod
    def get_english_dataset():
        save_dir = "./Koglish_STS/STS_14"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_STS",
            data_files={
                "en_valid": "STS_14/en_valid.csv",
                "en_test": "STS_14/en_test.csv",
            },
        )
        en_valid_data = dataset["en_valid"]
        en_test_data = dataset["en_test"]

        en_valid_save_path = os.path.join(save_dir, "en_valid.csv")
        en_test_save_path = os.path.join(save_dir, "en_test.csv")

        en_valid_data.to_csv(en_valid_save_path, index=False)
        en_test_data.to_csv(en_test_save_path, index=False)

        print(f"English Train data of Koglish_STS/STS_14 saved to {en_valid_save_path}")
        print(f"English Train data of Koglish_STS/STS_14 saved to {en_test_save_path}")

        return en_valid_data, en_test_data

    @staticmethod
    def get_code_swhiched_dataset():
        save_dir = "./Koglish_STS/STS_14"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_STS",
            data_files={
                "cs_valid": "STS_14/cs_valid.csv",
                "cs_test": "STS_14/cs_test.csv",
            },
        )
        cs_valid_data = dataset["cs_valid"]
        cs_test_data = dataset["cs_test"]

        cs_valid_save_path = os.path.join(save_dir, "cs_valid.csv")
        cs_test_save_path = os.path.join(save_dir, "cs_test.csv")

        cs_valid_data.to_csv(cs_valid_save_path, index=False)
        cs_test_data.to_csv(cs_test_save_path, index=False)

        print(
            f"Code-Switched Valid,Test data of Koglish_STS/STS_14 saved to {cs_valid_save_path}"
        )
        print(
            f"Code-Switched Valid,Test data of Koglish_STS/STS_14 saved to {cs_test_save_path}"
        )

        return cs_valid_data, cs_test_data


class LoadSTS15:
    @staticmethod
    def get_english_dataset():
        save_dir = "./Koglish_STS/STS_15"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_STS",
            data_files={
                "en_valid": "STS_15/en_valid.csv",
                "en_test": "STS_15/en_test.csv",
            },
        )
        en_valid_data = dataset["en_valid"]
        en_test_data = dataset["en_test"]

        en_valid_save_path = os.path.join(save_dir, "en_valid.csv")
        en_test_save_path = os.path.join(save_dir, "en_test.csv")

        en_valid_data.to_csv(en_valid_save_path, index=False)
        en_test_data.to_csv(en_test_save_path, index=False)

        print(f"English Train data of Koglish_STS/STS_15 saved to {en_valid_save_path}")
        print(f"English Train data of Koglish_STS/STS_15 saved to {en_test_save_path}")

        return en_valid_data, en_test_data

    @staticmethod
    def get_code_swhiched_dataset():
        save_dir = "./Koglish_STS/STS_15"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_STS",
            data_files={
                "cs_valid": "STS_15/cs_valid.csv",
                "cs_test": "STS_15/cs_test.csv",
            },
        )
        cs_valid_data = dataset["cs_valid"]
        cs_test_data = dataset["cs_test"]

        cs_valid_save_path = os.path.join(save_dir, "cs_valid.csv")
        cs_test_save_path = os.path.join(save_dir, "cs_test.csv")

        cs_valid_data.to_csv(cs_valid_save_path, index=False)
        cs_test_data.to_csv(cs_test_save_path, index=False)

        print(
            f"Code-Switched Valid,Test data of Koglish_STS/STS_15 saved to {cs_valid_save_path}"
        )
        print(
            f"Code-Switched Valid,Test data of Koglish_STS/STS_15 saved to {cs_test_save_path}"
        )

        return cs_valid_data, cs_test_data


class LoadSTS16:
    @staticmethod
    def get_english_dataset():
        save_dir = "./Koglish_STS/STS_16"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_STS",
            data_files={
                "en_valid": "STS_16/en_valid.csv",
                "en_test": "STS_16/en_test.csv",
            },
        )
        en_valid_data = dataset["en_valid"]
        en_test_data = dataset["en_test"]

        en_valid_save_path = os.path.join(save_dir, "en_valid.csv")
        en_test_save_path = os.path.join(save_dir, "en_test.csv")

        en_valid_data.to_csv(en_valid_save_path, index=False)
        en_test_data.to_csv(en_test_save_path, index=False)

        print(f"English Train data of Koglish_STS/STS_16 saved to {en_valid_save_path}")
        print(f"English Train data of Koglish_STS/STS_16 saved to {en_test_save_path}")

        return en_valid_data, en_test_data

    @staticmethod
    def get_code_swhiched_dataset():
        save_dir = "./Koglish_STS/STS_16"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_STS",
            data_files={
                "cs_valid": "STS_16/cs_valid.csv",
                "cs_test": "STS_16/cs_test.csv",
            },
        )
        cs_valid_data = dataset["cs_valid"]
        cs_test_data = dataset["cs_test"]

        cs_valid_save_path = os.path.join(save_dir, "cs_valid.csv")
        cs_test_save_path = os.path.join(save_dir, "cs_test.csv")

        cs_valid_data.to_csv(cs_valid_save_path, index=False)
        cs_test_data.to_csv(cs_test_save_path, index=False)

        print(
            f"Code-Switched Valid,Test data of Koglish_STS/STS_16 saved to {cs_valid_save_path}"
        )
        print(
            f"Code-Switched Valid,Test data of Koglish_STS/STS_16 saved to {cs_test_save_path}"
        )

        return cs_valid_data, cs_test_data


class LoadSICK:
    @staticmethod
    def get_english_dataset():
        save_dir = "./Koglish_STS/SICK"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_SICK",
            data_files={"en_valid": "en_valid.csv", "en_test": "en_test.csv"},
        )
        en_valid_data = dataset["en_valid"]
        en_test_data = dataset["en_test"]

        en_valid_save_path = os.path.join(save_dir, "en_valid.csv")
        en_test_save_path = os.path.join(save_dir, "en_test.csv")

        en_valid_data.to_csv(en_valid_save_path, index=False)
        en_test_data.to_csv(en_test_save_path, index=False)

        print(f"English Train data of Koglish_STS/SICK saved to {en_valid_save_path}")
        print(f"English Train data of Koglish_STS/SICK saved to {en_test_save_path}")

        return en_valid_data, en_test_data

    @staticmethod
    def get_code_swhiched_dataset():
        save_dir = "./Koglish_STS/SICK"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = load_dataset(
            "Jangyeong/Koglish_SICK",
            data_files={"cs_valid": "cs_valid.csv", "cs_test": "cs_test.csv"},
        )
        cs_valid_data = dataset["cs_valid"]
        cs_test_data = dataset["cs_test"]

        cs_valid_save_path = os.path.join(save_dir, "cs_valid.csv")
        cs_test_save_path = os.path.join(save_dir, "cs_test.csv")

        cs_valid_data.to_csv(cs_valid_save_path, index=False)
        cs_test_data.to_csv(cs_test_save_path, index=False)

        print(
            f"Code-Switched Valid,Test data of Koglish_STS/SICK saved to {cs_valid_save_path}"
        )
        print(
            f"Code-Switched Valid,Test data of Koglish_STS/SICK saved to {cs_test_save_path}"
        )

        return cs_valid_data, cs_test_data


if __name__ == "__main__":
    ####=====Download Koglish_GLUE datset=====####
    loader = LoadSST2()
    en_train_data, en_valid_data, en_test_data = loader.get_english_dataset()
    cs_train_data, cs_valid_data, cs_test_data = loader.get_code_swhiched_dataset()

    loader = LoadMRPC()
    en_train_data, en_valid_data, en_test_data = loader.get_english_dataset()
    cs_train_data, cs_valid_data, cs_test_data = loader.get_code_swhiched_dataset()

    loader = LoadCOLA()
    en_train_data, en_valid_data, en_test_data = loader.get_english_dataset()
    cs_train_data, cs_valid_data, cs_test_data = loader.get_code_swhiched_dataset()

    loader = LoadRTE()
    en_train_data, en_valid_data, en_test_data = loader.get_english_dataset()
    cs_train_data, cs_valid_data, cs_test_data = loader.get_code_swhiched_dataset()

    loader = LoadSTSB()
    en_train_data, en_valid_data, en_test_data = loader.get_english_dataset()
    cs_train_data, cs_valid_data, cs_test_data = loader.get_code_swhiched_dataset()

    loader = LoadQNLI()
    en_train_data, en_valid_data, en_test_data = loader.get_english_dataset()
    cs_train_data, cs_valid_data, cs_test_data = loader.get_code_swhiched_dataset()

    ####=====Download Koglish_NLI datset=====####
    loader = LoadNLI()
    en_train_data = loader.get_english_dataset()
    cs_train_data = loader.get_code_swhiched_dataset()

    ####=====Download Koglish_STS datset=====####
    loader = LoadSTSB_for_ConCSE()
    en_valid_data, en_test_data = loader.get_english_dataset()
    cs_valid_data, cs_test_data = loader.get_code_swhiched_dataset()

    loader = LoadSTS12()
    en_valid_data, en_test_data = loader.get_english_dataset()
    cs_valid_data, cs_test_data = loader.get_code_swhiched_dataset()

    loader = LoadSTS13()
    en_valid_data, en_test_data = loader.get_english_dataset()
    cs_valid_data, cs_test_data = loader.get_code_swhiched_dataset()

    loader = LoadSTS14()
    en_valid_data, en_test_data = loader.get_english_dataset()
    cs_valid_data, cs_test_data = loader.get_code_swhiched_dataset()

    loader = LoadSTS15()
    en_valid_data, en_test_data = loader.get_english_dataset()
    cs_valid_data, cs_test_data = loader.get_code_swhiched_dataset()

    loader = LoadSTS16()
    en_valid_data, en_test_data = loader.get_english_dataset()
    cs_valid_data, cs_test_data = loader.get_code_swhiched_dataset()

    loader = LoadSICK()
    en_valid_data, en_test_data = loader.get_english_dataset()
    cs_valid_data, cs_test_data = loader.get_code_swhiched_dataset()
