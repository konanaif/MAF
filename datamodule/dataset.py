import os, sys, glob
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter
import torch
from torch.utils.data import Dataset
from PIL import Image
import requests

from aif360.datasets import StandardDataset, AdultDataset

parent_dir = os.environ["PYTHONPATH"]


class aifData(StandardDataset):
    def __init__(
        self,
        df,
        label_name,
        favorable_classes,
        protected_attribute_names,
        privileged_classes,
        instance_weights_name="",
        scores_name="",
        categorical_features=[],
        features_to_keep=[],
        features_to_drop=[],
        na_values=[],
        custom_preprocessing=None,
        metadata=None,
    ):

        super(aifData, self).__init__(
            df=df,
            label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop,
            na_values=na_values,
            custom_preprocessing=custom_preprocessing,
            metadata=metadata,
        )


class RawDataSet:
    def __init__(self, filename=None, **kwargs):
        if filename:
            # Detect file type
            extention = os.path.splitext(filename)[-1]
        else:
            extention = ""

        ##### CASE 1 : numpy object #####
        if extention == ".npy":
            ## Required parameters
            #    - target column index (int) : target_col_idx
            #    - bias column index (int) : bias_col_idx
            ## Optional parameters
            #    - prediction column index (int) : pred_col_idx

            # Check required parameters
            try:
                target_col_idx = kwargs["target_col_idx"]
                bias_col_idx = kwargs["bias_col_idx"]
            except:
                print("ERROR! You need to pass the required parameters.")
                print("Please check the [target_col_idx] and [bias_col_idx].")
                raise

            # Load the raw file
            loaded_arr = np.load(filename)

            # Make features
            self.target = loaded_arr[:, target_col_idx]
            self.bias = loaded_arr[:, bias_col_idx]

            # Check optional parameters
            if "pred_col_idx" in kwargs.keys():
                pred_col_idx = kwargs["pred_col_idx"]
                self.predict = loaded_arr[:, pred_col_idx]
                self.feature = np.delete(
                    loaded_arr, [target_col_idx, pred_col_idx], axis=1
                )
                self.feature_only = np.delete(
                    loaded_arr, [bias_col_idx, target_col_idx, pred_col_idx], axis=1
                )
            else:
                self.predict = (
                    np.zeros_like(loaded_arr[:, 0]) - 1
                )  # [-1, -1, -1, ..., -1]
                self.feature = np.delete(loaded_arr, [target_col_idx], axis=1)
                self.feature_only = np.delete(
                    loaded_arr, [bias_col_idx, target_col_idx], axis=1
                )
        #################################

        ##### CASE 2 : framed table #####
        elif extention in [".csv", ".tsv"]:
            ## Required parameters
            #    - target column name (str) : target_col_name
            #    - bias column name (str) : bias_col_name
            ## Optional parameters
            #    - header (int) : table header (column name) row index
            #    - seperator (str) : table seperator
            #    - prediction column name (str) : pred_col_name
            #    - categorical column name list (list) : cate_cols

            # Check the required parameters
            try:
                target_col_name = kwargs["target_col_name"]
                bias_col_name = kwargs["bias_col_name"]
            except:
                print("ERROR! You need to pass the required parameters.")
                print("Please check the [target_col_idx] and [bias_col_idx].")
                raise

            # Check the optional parameters
            header = kwargs["header"] if "header" in kwargs.keys() else 0  # default 0
            seperator = (
                kwargs["seperator"] if "seperator" in kwargs.keys() else ","
            )  # default ,
            cate_cols = (
                kwargs["cate_cols"] if "cate_cols" in kwargs.keys() else []
            )  # default []

            # Load the table file
            loaded_df = pd.read_table(filename, sep=seperator, header=header)

            # Preprocess categorical columns
            converted_df = self.convert_categorical(loaded_df, cate_cols)

            # Make features
            self.target = converted_df.loc[:, target_col_name].to_numpy()
            self.bias = converted_df.loc[:, bias_col_name].to_numpy()
            if "pred_col_name" in kwargs.keys():
                pred_col_name = kwargs["pred_col_name"]
                self.predict = converted_df.loc[:, pred_col_name].to_numpy()
                self.feature = converted_df.drop(
                    columns=[target_col_name, pred_col_name]
                ).to_numpy()
                self.feature_only = converted_df.drop(
                    columns=[bias_col_name, target_col_name, pred_col_name]
                ).to_numpy()
            else:
                self.predict = (
                    np.zeros_like(converted_df[target_col_name]) - 1
                )  # [-1, -1, -1, ..., -1]
                self.feature = converted_df.drop(columns=[target_col_name]).to_numpy()
                self.feature_only = converted_df.drop(
                    columns=[bias_col_name, target_col_name]
                ).to_numpy()

        else:
            try:
                self.feature = kwargs["x"]
                self.bias = kwargs["z"]
                self.target = kwargs["y"]
                self.feature_only = kwargs["x"]
            except:
                print("Input file : {}\t\t\tExtention : {}".format(filename, extention))
                raise Exception("FILE ERROR!! Only [npy, csv, tsv] extention required.")

    # Convert categorical values to numerical (integer) values on pandas.DataFrame
    def convert_categorical(self, dataframe, category_list):
        temp = dataframe.copy()

        for cate in category_list:
            categories = Counter(temp[cate])
            c2i = {}
            i = 0
            for c, f in categories.items():
                c2i[c] = i
            temp[cate] = temp[cate].map(lambda x: c2i[x])
        return temp


class CelebADataset(StandardDataset):
    ROOT = parent_dir + "/MAF/data/celeba/"
    IMAGE_DIR = ROOT + "img_align_celeba"
    ATTR_FILE = ROOT + "list_attr_celeba.txt"
    META_DATA = ROOT + "list_eval_partition.csv"

    def __init__(self):
        if not os.path.exists(self.IMAGE_DIR) or not os.path.exists(self.ATTR_FILE):
            self.download()

    def download(self):
        if not os.path.exists(self.ROOT):
            os.makedirs(self.ROOT)

        img_file = os.path.join(self.ROOT, "img_align_celeba.zip")
        attr_file = self.ATTR_FILE

        def download_file(url, output):
            if not os.path.exists(output):
                try:
                    gdown.download(url, output, quiet=False)
                    print(f"Successfully downloaded: {output}")
                except Exception as e:
                    print(f"Error downloading {url}: {str(e)}")
                    return False
            else:
                print(f"File already exists: {output}")
            return True

        # 이미지 다운 경로
        img_url = "https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM"
        attr_url = "https://drive.google.com/uc?id=0B7EVK8r0v71pblRyaVFSWGxPY0U"

        if download_file(img_url, img_file) and download_file(attr_url, attr_file):
            try:
                with zipfile.ZipFile(img_file, "r") as zip_ref:
                    zip_ref.extractall(self.IMAGE_DIR)
                print("Successfully extracted images.")
            except Exception as e:
                print(f"Error extracting images: {str(e)}")
        else:
            print("Download failed. Please try again.")

    def to_dataset(self):
        img_files = glob.glob(self.IMAGE_DIR + "/*.jpg")
        print(f"celeba image files {len(img_files)}")
        img_keys = []
        img_list = []
        for ifn in tqdm(img_files):
            try:
                img = Image.open(ifn).convert("RGB").resize((224, 224))
            except:
                continue

            img = np.asarray(img)
            if img.size == 3 * 224 * 224:
                key = os.path.basename(ifn)
                img_keys.append(key)
                img_list.append(img)

        with open(self.ATTR_FILE, "r") as f:
            lines = f.readlines()

        attribute_names = lines[1].strip().split()

        data = []
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) > 1:
                filename = parts[0]
                attrs = [1 if int(x) > 0 else 0 for x in parts[1:]]
                data.append([filename] + attrs)

        attribute = pd.DataFrame(data, columns=["image_id"] + attribute_names)
        attribute = attribute[attribute["image_id"].isin(img_keys)]

        TARGET_NAME = "Blond_Hair"
        BIAS_NAME = "Male"
        SELECTED_COLUMNS = ["image_id", TARGET_NAME, BIAS_NAME]

        print("creating selected_df")
        selected_df = attribute[SELECTED_COLUMNS]
        selected_df.to_csv(self.ATTR_FILE.replace(".txt", ".csv"), index=True)

        # Convert Target and Bias to categorical
        def categorize(score):
            if score > 0:
                return 1
            else:
                return 0

        vfunc = np.vectorize(categorize)
        target_vect = attribute[TARGET_NAME].to_numpy()
        target_vect = vfunc(target_vect)
        bias_vect = attribute[BIAS_NAME].to_numpy()
        bias_vect = vfunc(bias_vect)

        # Make images to DataFrame (for using aif360)
        temp = [im.ravel() for im in img_list]
        temp_df = pd.DataFrame(temp)

        # Add column
        temp_df[TARGET_NAME] = target_vect
        temp_df[BIAS_NAME] = bias_vect

        # Make dataset
        dataset = aifData(
            df=temp_df,
            label_name=TARGET_NAME,
            favorable_classes=[1],
            protected_attribute_names=[BIAS_NAME],
            privileged_classes=[[1]],
        )

        print("finish to created dataset")
        return dataset


compas_mappings = {
    "label_maps": [{1.0: "Did recid.", 0.0: "No recid."}],
    "protected_attribute_maps": [
        {0.0: "Male", 1.0: "Female"},
        {1.0: "Caucasian", 0.0: "Not Caucasian"},
    ],
}


def compas_preprocessing(df):
    """Perform the same preprocessing as the original analysis:
    https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    """
    return df[
        (df.days_b_screening_arrest <= 30)
        & (df.days_b_screening_arrest >= -30)
        & (df.is_recid != -1)
        & (df.c_charge_degree != "O")
        & (df.score_text != "N/A")
    ]


class CompasDataset(StandardDataset):
    def __init__(
        self,
        filepath=parent_dir + "/MAF/data/compas/compas-scores-two-years.csv",
        label_name="two_year_recid",
        favorable_classes=[0],
        protected_attribute_names=["sex", "race"],
        privileged_classes=[["Female"], ["Caucasian"]],
        instance_weights_name=None,
        categorical_features=["age_cat", "c_charge_degree", "c_charge_desc"],
        features_to_keep=[
            "sex",
            "age",
            "age_cat",
            "race",
            "juv_fel_count",
            "juv_misd_count",
            "juv_other_count",
            "priors_count",
            "c_charge_degree",
            "c_charge_desc",
            "two_year_recid",
        ],
        features_to_drop=[],
        na_values=[],
        custom_preprocessing=compas_preprocessing,
        metadata=compas_mappings,
    ):

        try:
            df = pd.read_csv(filepath, index_col="id", na_values=na_values)
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following file:")
            print(
                "\n\thttps://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
            )
            print("\nand place it, as-is, in the folder:")
            print("\n\t{}\n".format(filepath))
            import sys

            sys.exit(1)

        super(CompasDataset, self).__init__(
            df=df,
            label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop,
            na_values=na_values,
            custom_preprocessing=custom_preprocessing,
            metadata=metadata,
        )


german_mappings = {
    "label_maps": [{1.0: "Good Credit", 0.0: "Bad Credit"}],
    "protected_attribute_maps": [
        {1.0: "Male", 0.0: "Female"},
        {1.0: "Old", 0.0: "Young"},
    ],
}


def german_preprocessing(df):
    """Adds a derived sex attribute based on personal_status."""
    # TODO: ignores the value of privileged_classes for 'sex'
    status_map = {
        "A91": "male",
        "A93": "male",
        "A94": "male",
        "A92": "female",
        "A95": "female",
    }
    df["sex"] = df["personal_status"].replace(status_map)

    return df


class GermanDataset(StandardDataset):
    """German credit Dataset.
    See :file:`aif360/data/raw/german/README.md`.
    """

    def __init__(
        self,
        filepath=parent_dir + "/MAF/data/german/german.data",
        label_name="credit",
        favorable_classes=[1],
        protected_attribute_names=["sex", "age"],
        privileged_classes=[["male"], lambda x: x > 25],
        instance_weights_name=None,
        categorical_features=[
            "status",
            "credit_history",
            "purpose",
            "savings",
            "employment",
            "other_debtors",
            "property",
            "installment_plans",
            "housing",
            "skill_level",
            "telephone",
            "foreign_worker",
        ],
        features_to_keep=[],
        features_to_drop=["personal_status"],
        na_values=[],
        custom_preprocessing=german_preprocessing,
        metadata=german_mappings,
    ):
        """See :obj:`StandardDataset` for a description of the arguments.
        By default, this code converts the 'age' attribute to a binary value
        where privileged is `age > 25` and unprivileged is `age <= 25` as
        proposed by Kamiran and Calders [1]_.
        References:
                .. [1] F. Kamiran and T. Calders, "Classifying without
                   discriminating," 2nd International Conference on Computer,
                   Control and Communication, 2009.
        Examples:
                In some cases, it may be useful to keep track of a mapping from
                `float -> str` for protected attributes and/or labels. If our use
                case differs from the default, we can modify the mapping stored in
                `metadata`:
                >>> label_map = {1.0: 'Good Credit', 0.0: 'Bad Credit'}
                >>> protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
                >>> gd = GermanDataset(protected_attribute_names=['sex'],
                ... privileged_classes=[['male']], metadata={'label_map': label_map,
                ... 'protected_attribute_maps': protected_attribute_maps})
                Now this information will stay attached to the dataset and can be
                used for more descriptive visualizations.
        """

        # as given by german.doc
        column_names = [
            "status",
            "month",
            "credit_history",
            "purpose",
            "credit_amount",
            "savings",
            "employment",
            "investment_as_income_percentage",
            "personal_status",
            "other_debtors",
            "residence_since",
            "property",
            "age",
            "installment_plans",
            "housing",
            "number_of_credits",
            "skill_level",
            "people_liable_for",
            "telephone",
            "foreign_worker",
            "credit",
        ]
        try:
            df = pd.read_csv(
                filepath, sep=" ", header=None, names=column_names, na_values=na_values
            )
            df["credit"] = df["credit"] - 1
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following files:")
            print(
                "\n\thttps://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
            )
            print(
                "\thttps://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc"
            )
            print("\nand place them, as-is, in the folder:")
            print("\n\t{}\n".format(filepath))
            import sys

            sys.exit(1)

        super(GermanDataset, self).__init__(
            df=df,
            label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop,
            na_values=na_values,
            custom_preprocessing=custom_preprocessing,
            metadata=metadata,
        )


adult_mappings = {
    "label_maps": [{1.0: ">50K", 0.0: "<=50K"}],
    "protected_attribute_maps": [
        {1.0: "White", 0.0: "Non-white"},
        {1.0: "Male", 0.0: "Female"},
    ],
}


class AdultDataset(StandardDataset):
    """Adult Census Income Dataset.
    See :file:`aif360/data/raw/adult/README.md`.
    """

    def __init__(
        self,
        file_directory=parent_dir + "/MAF/data/adult",
        label_name="income-per-year",
        favorable_classes=[">50K", ">50K."],
        protected_attribute_names=["race", "sex"],
        privileged_classes=[["White"], ["Male"]],
        instance_weights_name=None,
        categorical_features=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "native-country",
        ],
        features_to_keep=[],
        features_to_drop=["fnlwgt"],
        na_values=["?"],
        custom_preprocessing=None,
        metadata=adult_mappings,
    ):
        """See :obj:`StandardDataset` for a description of the arguments.
        Examples:
                The following will instantiate a dataset which uses the `fnlwgt`
                feature:
                >>> from aif360.datasets import AdultDataset
                >>> ad = AdultDataset(instance_weights_name='fnlwgt',
                ... features_to_drop=[])
                WARNING:root:Missing Data: 3620 rows removed from dataset.
                >>> not np.all(ad.instance_weights == 1.)
                True
                To instantiate a dataset which utilizes only numerical features and
                a single protected attribute, run:
                >>> single_protected = ['sex']
                >>> single_privileged = [['Male']]
                >>> ad = AdultDataset(protected_attribute_names=single_protected,
                ... privileged_classes=single_privileged,
                ... categorical_features=[],
                ... features_to_keep=['age', 'education-num'])
                >>> print(ad.feature_names)
                ['education-num', 'age', 'sex']
                >>> print(ad.label_names)
                ['income-per-year']
                Note: the `protected_attribute_names` and `label_name` are kept even
                if they are not explicitly given in `features_to_keep`.
                In some cases, it may be useful to keep track of a mapping from
                `float -> str` for protected attributes and/or labels. If our use
                case differs from the default, we can modify the mapping stored in
                `metadata`:
                >>> label_map = {1.0: '>50K', 0.0: '<=50K'}
                >>> protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
                >>> ad = AdultDataset(protected_attribute_names=['sex'],
                ... categorical_features=['workclass', 'education', 'marital-status',
                ... 'occupation', 'relationship', 'native-country', 'race'],
                ... privileged_classes=[['Male']], metadata={'label_map': label_map,
                ... 'protected_attribute_maps': protected_attribute_maps})
                Note that we are now adding `race` as a `categorical_features`.
                Now this information will stay attached to the dataset and can be
                used for more descriptive visualizations.
        """

        train_path = os.path.join(file_directory, "adult.data")
        test_path = os.path.join(file_directory, "adult.test")
        # as given by adult.names
        column_names = [
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
            "income-per-year",
        ]
        try:
            train = pd.read_csv(
                train_path,
                header=None,
                names=column_names,
                skipinitialspace=True,
                na_values=na_values,
            )
            test = pd.read_csv(
                test_path,
                header=0,
                names=column_names,
                skipinitialspace=True,
                na_values=na_values,
            )
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following files:")
            print(
                "\n\thttps://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            )
            print(
                "\thttps://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
            )
            print(
                "\thttps://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names"
            )
            print("\nand place them, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(file_directory)))
            import sys

            sys.exit(1)

        df = pd.concat([test, train], ignore_index=True)

        super(AdultDataset, self).__init__(
            df=df,
            label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop,
            na_values=na_values,
            custom_preprocessing=custom_preprocessing,
            metadata=metadata,
        )


class PubFigDataset(StandardDataset):
    ROOT = parent_dir + "/MAF/data/pubfig/image"
    ATTRIBUTE_FILE = parent_dir + "/MAF/data/pubfig/pubfig_attributes.txt"
    URL_FILE = parent_dir + "/MAF/data/pubfig/dev_urls.txt"

    def __init__(self):
        if not os.path.exists(self.ATTRIBUTE_FILE):
            print(
                "The attribute file [[ {} ]] is not in directory".format(
                    self.ATTRIBUTE_FILE
                )
            )
            print("It will be downloaded...")
            response = requests.get(
                "https://www.cs.columbia.edu/CAVE/databases/pubfig/download/pubfig_attributes.txt"
            )
            with open(self.ATTRIBUTE_FILE, "wb") as f:
                for data in tqdm(response.iter_content()):
                    f.write(data)
            print("Downloaded successfully", end="\n\n")

        if not os.path.exists(self.URL_FILE):
            print(
                "The attribute file [[ {} ]] is not in directory".format(self.URL_FILE)
            )
            print("It will be downloaded...")
            response = requests.get(
                "https://www.cs.columbia.edu/CAVE/databases/pubfig/download/dev_urls.txt"
            )
            with open(self.URL_FILE, "wb") as f:
                for data in tqdm(response.iter_content()):
                    f.write(data)
            print("Downloaded successfully", end="\n\n")

    def download(self):
        # Read the files with pandas.DataFrame
        imgurl_df = pd.read_table(
            self.URL_FILE,
            sep="\t",
            skiprows=[0, 1],
            names=["person", "imagenum", "url", "rect", "md5sum"],
        )
        attr_df = pd.read_table(
            self.ATTRIBUTE_FILE,
            sep="\t",
            skiprows=[0, 1],
            names=[
                "person",
                "imagenum",
                "Male",
                "Asian",
                "White",
                "Black",
                "Baby",
                "Child",
                "Youth",
                "Middle Aged",
                "Senior",
                "Black Hair",
                "Blond Hair",
                "Brown Hair",
                "Bald",
                "No Eyewear",
                "Eyeglasses",
                "Sunglasses",
                "Mustache",
                "Smiling",
                "Frowning",
                "Chubby",
                "Blurry",
                "Harsh Lighting",
                "Flash",
                "Soft Lighting",
                "Outdoor",
                "Curly Hair",
                "Wavy Hair",
                "Straight Hair",
                "Receding Hairline",
                "Bangs",
                "Sideburns",
                "Fully Visible Forehead",
                "Partially Visible Forehead",
                "Obstructed Forehead",
                "Bushy Eyebrows",
                "Arched Eyebrows",
                "Narrow Eyes",
                "Eyes Open",
                "Big Nose",
                "Pointy Nose",
                "Big Lips",
                "Mouth Closed",
                "Mouth Slightly Open",
                "Mouth Wide Open",
                "Teeth Not Visible",
                "No Beard",
                "Goatee",
                "Round Jaw",
                "Double Chin",
                "Wearing Hat",
                "Oval Face",
                "Square Face",
                "Round Face",
                "Color Photo",
                "Posed Photo",
                "Attractive Man",
                "Attractive Woman",
                "Indian",
                "Gray Hair",
                "Bags Under Eyes",
                "Heavy Makeup",
                "Rosy Cheeks",
                "Shiny Skin",
                "Pale Skin",
                "5 o' Clock Shadow",
                "Strong Nose-Mouth Lines",
                "Wearing Lipstick",
                "Flushed Face",
                "High Cheekbones",
                "Brown Eyes",
                "Wearing Earrings",
                "Wearing Necktie",
                "Wearing Necklace",
            ],
        )

        # Merge two DataFrame using (person & imagenum)
        imgurl_df["key"] = imgurl_df["person"] + "_" + imgurl_df["imagenum"].astype(str)
        attr_df["key"] = attr_df["person"] + "_" + attr_df["imagenum"].astype(str)
        merged = imgurl_df.merge(attr_df, how="inner")

        # Before download the images,
        # make directory for saving the images
        if not os.path.isdir(self.ROOT):
            os.mkdir(self.ROOT)
        print("The pubfig images will be downloaded on [[ {} ]]".format(self.ROOT))

        # Download images
        print("{} images downloaded from source urls".format(len(merged)))
        print("Download start...")

        for iuk in tqdm(zip(merged.index, merged["url"], merged["key"])):
            idx, url, key = iuk
            fn = key + ".jpg"
            filepath = os.path.join(self.ROOT, fn)

            if os.path.exists(filepath):
                continue

            try:
                response = requests.get(url, timeout=2)
            except requests.exceptions.Timeout:
                # Timed out
                # The image link blocked or removed
                merged = merged.drop(index=idx)
                continue
            except:
                # Unknown URL
                # Remove the row on 'merged' DataFrame
                merged = merged.drop(index=idx)
                continue

            with open(filepath, "wb") as img:
                for data in response.iter_content():
                    img.write(data)
                response.close()
        print(
            "All images {} download on {} successfully".format(len(merged), self.ROOT)
        )
        merged.to_csv(
            parent_dir + "/MAF/data/pubfig/pubfig_attr_merged.csv",
            index=False,
            encoding="utf-8",
        )

    def to_dataset(self):
        img_files = glob.glob(parent_dir + "/MAF/data/pubfig/image/*")

        # Load the images
        img_keys = []
        img_list = []
        for ifn in tqdm(img_files):
            try:
                img = Image.open(ifn).resize((64, 64))
            except:
                print(f'"{ifn}" cannot be resized by 64x64')
                continue
            img = np.asarray(img)

            if img.size == 12288:
                key = os.path.basename(ifn).replace(".jpg", "")
                img_list.append(img)
                img_keys.append(key)

        # Load the attribute file
        attribute = pd.read_csv(
            parent_dir + "/MAF/data/pubfig/pubfig_attr_merged.csv", encoding="utf-8"
        )
        attribute = attribute[attribute["key"].isin(img_keys)]

        TARGET_NAME = "Male"
        BIAS_NAME = "Heavy Makeup"

        # Convert Target and Bias to categorical
        def categorize(score):
            if score > 0:
                return 1
            else:
                return 0

        vfunc = np.vectorize(categorize)
        target_vect = attribute[TARGET_NAME].to_numpy()

        target_vect = vfunc(target_vect)
        bias_vect = attribute[BIAS_NAME].to_numpy()
        bias_vect = vfunc(bias_vect)

        # Make images to DataFrame (for using aif360)
        temp = [im.ravel() for im in img_list]
        temp_df = pd.DataFrame(temp)

        # Add column
        temp_df[TARGET_NAME] = target_vect
        temp_df[BIAS_NAME] = bias_vect

        # Make dataset
        dataset = aifData(
            df=temp_df,
            label_name=TARGET_NAME,
            favorable_classes=[1],
            protected_attribute_names=[BIAS_NAME],
            privileged_classes=[[1]],
        )

        result = {
            "aif_dataset": dataset,
            "image_list": img_list,
            "attribute": attribute,
            "target": target_vect,
            "bias": bias_vect,
        }

        return result
