# sample.py  (스페이스 4칸 들여쓰기, aif360 없어도 동작)
import os
import sys
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image
from collections import Counter
import torch

# ==== 경로 상수: sample.py 파일 위치 기준 절대경로 ====
_THIS_DIR   = os.path.dirname(__file__)                      # .../code
_DATA_ROOT  = os.path.abspath(os.path.join(_THIS_DIR, 'data'))  # .../code/data
_DEP_DIR    = os.path.join(_DATA_ROOT, 'dep')                # .../code/data/dep
_PUBFIG_DIR = os.path.join(_DEP_DIR, 'pubfig')               # .../code/data/dep/pubfig
_ATTR_FILE  = os.path.join(_DEP_DIR, 'pubfig_attributes.txt')
_URL_FILE   = os.path.join(_DEP_DIR, 'dev_urls.txt')
_MERGED_CSV = os.path.join(_DEP_DIR, 'pubfig_attr_merged.csv')

# ==== aif360 임포트: 없으면 shim 사용 ====
try:
    from aif360.datasets import StandardDataset
    _HAS_AIF360 = True
except ImportError:
    _HAS_AIF360 = False

if _HAS_AIF360:
    class aifData(StandardDataset):
        def __init__(self, df, label_name, favorable_classes,
                     protected_attribute_names, privileged_classes,
                     instance_weights_name='', scores_name='',
                     categorical_features=None, features_to_keep=None,
                     features_to_drop=None, na_values=None,
                     custom_preprocessing=None, metadata=None):
            if categorical_features is None: categorical_features = []
            if features_to_keep is None: features_to_keep = []
            if features_to_drop is None: features_to_drop = []
            if na_values is None: na_values = []
            super(aifData, self).__init__(
                df=df, label_name=label_name,
                favorable_classes=favorable_classes,
                protected_attribute_names=protected_attribute_names,
                privileged_classes=privileged_classes,
                instance_weights_name=instance_weights_name,
                categorical_features=categorical_features,
                features_to_keep=features_to_keep,
                features_to_drop=features_to_drop,
                na_values=na_values,
                custom_preprocessing=custom_preprocessing,
                metadata=metadata
            )
else:
    class aifData:
        """
        aif360 없이도 FFD/DML 코드가 쓰는 최소 필드/메서드만 제공.
        - .features (N, D+1)  # 마지막 컬럼은 protected z
        - .labels (N, 1)
        - .protected_attributes (N, 1)
        - .label_names, .favorable_label
        - .protected_attribute_names, .privileged_protected_attributes
        - .split([ratio], shuffle=True, seed=1) -> (train, test)
        - .instance_names (선택)
        """
        def __init__(self, df, label_name, favorable_classes,
                     protected_attribute_names, privileged_classes, **kwargs):
            self.label_names = [label_name]
            self.favorable_label = int(favorable_classes[0])
            self.protected_attribute_names = protected_attribute_names
            self.privileged_protected_attributes = privileged_classes

            prot = protected_attribute_names[0]
            pixel_cols = [c for c in df.columns if c not in [label_name, prot]]

            X = df[pixel_cols].to_numpy()
            z = df[prot].to_numpy().astype(int).reshape(-1, 1)
            y = df[label_name].to_numpy().astype(int).reshape(-1, 1)

            # FFD/dep 래퍼 호환: features의 마지막 컬럼이 z 이어야 함
            self.features = np.concatenate([X, z], axis=1)
            self.labels = y
            self.protected_attributes = z
            self.instance_names = df.index.astype(str).tolist()

        def copy(self, deepcopy=False):
            import copy
            return copy.deepcopy(self) if deepcopy else copy.copy(self)

        def split(self, ratios, shuffle=True, seed=1):
            assert len(ratios) == 1, "Expected single ratio list, e.g., [0.7]"
            r = float(ratios[0])
            N = self.features.shape[0]
            idx = np.arange(N)
            if shuffle:
                rng = np.random.RandomState(seed)
                rng.shuffle(idx)
            n_tr = int(N * r)
            tr_idx, te_idx = idx[:n_tr], idx[n_tr:]

            def _subset(idxs):
                sub = aifData.__new__(aifData)
                sub.label_names = self.label_names[:]
                sub.favorable_label = self.favorable_label
                sub.protected_attribute_names = self.protected_attribute_names[:]
                sub.privileged_protected_attributes = self.privileged_protected_attributes
                sub.features = self.features[idxs]
                sub.labels = self.labels[idxs]
                sub.protected_attributes = self.protected_attributes[idxs]
                sub.instance_names = [self.instance_names[i] for i in idxs]
                return sub

            return _subset(tr_idx), _subset(te_idx)

# ========= (옵션) RawDataSet: 필요 없으면 지워도 무방 =========
class RawDataSet:
    def __init__(self, filename=None, **kwargs):
        if filename:
            ext = os.path.splitext(filename)[-1]
        else:
            ext = ''

        if ext == '.npy':
            try:
                target_col_idx = kwargs['target_col_idx']
                bias_col_idx = kwargs['bias_col_idx']
            except KeyError:
                raise ValueError('Need target_col_idx and bias_col_idx')
            arr = np.load(filename)
            self.target = arr[:, target_col_idx]
            self.bias = arr[:, bias_col_idx]
            if 'pred_col_idx' in kwargs:
                p = kwargs['pred_col_idx']
                self.predict = arr[:, p]
                self.feature = np.delete(arr, [target_col_idx, p], axis=1)
                self.feature_only = np.delete(arr, [bias_col_idx, target_col_idx, p], axis=1)
            else:
                self.predict = np.zeros_like(arr[:, 0]) - 1
                self.feature = np.delete(arr, [target_col_idx], axis=1)
                self.feature_only = np.delete(arr, [bias_col_idx, target_col_idx], axis=1)

        elif ext in ['.csv', '.tsv']:
            try:
                target_col_name = kwargs['target_col_name']
                bias_col_name = kwargs['bias_col_name']
            except KeyError:
                raise ValueError('Need target_col_name and bias_col_name')
            header = kwargs.get('header', 0)
            sep = kwargs.get('seperator', ',')
            cate_cols = kwargs.get('cate_cols', [])

            df = pd.read_table(filename, sep=sep, header=header)
            df = self.convert_categorical(df, cate_cols)
            self.target = df[target_col_name].to_numpy()
            self.bias = df[bias_col_name].to_numpy()
            if 'pred_col_name' in kwargs:
                pc = kwargs['pred_col_name']
                self.predict = df[pc].to_numpy()
                self.feature = df.drop(columns=[target_col_name, pc]).to_numpy()
                self.feature_only = df.drop(columns=[bias_col_name, target_col_name, pc]).to_numpy()
            else:
                self.predict = np.zeros_like(df[target_col_name]) - 1
                self.feature = df.drop(columns=[target_col_name]).to_numpy()
                self.feature_only = df.drop(columns=[bias_col_name, target_col_name]).to_numpy()
        else:
            try:
                self.feature = kwargs['x']
                self.bias = kwargs['z']
                self.target = kwargs['y']
                self.feature_only = kwargs['x']
            except KeyError:
                raise ValueError('Only .npy/.csv/.tsv or explicit x,y,z are supported.')

    def convert_categorical(self, dataframe, category_list):
        temp = dataframe.copy()
        for cate in category_list:
            cats = Counter(temp[cate])
            c2i = {c: i for i, (c, _) in enumerate(cats.items())}
            temp[cate] = temp[cate].map(lambda x: c2i[x])
        return temp

# ========= PubFigDataset =========
class PubFigDataset:
    ROOT = _PUBFIG_DIR
    ATTRIBUTE_FILE = _ATTR_FILE
    URL_FILE = _URL_FILE
    MERGED_CSV = _MERGED_CSV

    def __init__(self):
        # 디렉토리 준비
        os.makedirs(os.path.dirname(self.ATTRIBUTE_FILE), exist_ok=True)
        os.makedirs(self.ROOT, exist_ok=True)

        if not os.path.exists(self.ATTRIBUTE_FILE):
            print(f'[{self.ATTRIBUTE_FILE}] not found. Downloading...')
            r = requests.get('https://www.cs.columbia.edu/CAVE/databases/pubfig/download/pubfig_attributes.txt')
            with open(self.ATTRIBUTE_FILE, 'wb') as f:
                for chunk in r.iter_content():
                    f.write(chunk)
            print('Downloaded.\n')

        if not os.path.exists(self.URL_FILE):
            print(f'[{self.URL_FILE}] not found. Downloading...')
            r = requests.get('https://www.cs.columbia.edu/CAVE/databases/pubfig/download/dev_urls.txt')
            with open(self.URL_FILE, 'wb') as f:
                for chunk in r.iter_content():
                    f.write(chunk)
            print('Downloaded.\n')

    def download(self):
        imgurl_df = pd.read_table(self.URL_FILE, sep='\t', skiprows=[0, 1],
                                  names=['person', 'imagenum', 'url', 'rect', 'md5sum'])
        attr_df = pd.read_table(self.ATTRIBUTE_FILE, sep='\t', skiprows=[0, 1], names=[
            'person','imagenum','Male','Asian','White','Black','Baby','Child','Youth','Middle Aged','Senior',
            'Black Hair','Blond Hair','Brown Hair','Bald','No Eyewear','Eyeglasses','Sunglasses','Mustache',
            'Smiling','Frowning','Chubby','Blurry','Harsh Lighting','Flash','Soft Lighting','Outdoor',
            'Curly Hair','Wavy Hair','Straight Hair','Receding Hairline','Bangs','Sideburns',
            'Fully Visible Forehead','Partially Visible Forehead','Obstructed Forehead','Bushy Eyebrows',
            'Arched Eyebrows','Narrow Eyes','Eyes Open','Big Nose','Pointy Nose','Big Lips','Mouth Closed',
            'Mouth Slightly Open','Mouth Wide Open','Teeth Not Visible','No Beard','Goatee','Round Jaw',
            'Double Chin','Wearing Hat','Oval Face','Square Face','Round Face','Color Photo','Posed Photo',
            'Attractive Man','Attractive Woman','Indian','Gray Hair','Bags Under Eyes','Heavy Makeup',
            'Rosy Cheeks','Shiny Skin','Pale Skin',"5 o' Clock Shadow",'Strong Nose-Mouth Lines',
            'Wearing Lipstick','Flushed Face','High Cheekbones','Brown Eyes','Wearing Earrings',
            'Wearing Necktie','Wearing Necklace'
        ])

        imgurl_df['key'] = imgurl_df['person'] + '_' + imgurl_df['imagenum'].astype(str)
        attr_df['key'] = attr_df['person'] + '_' + attr_df['imagenum'].astype(str)
        merged = imgurl_df.merge(attr_df, how='inner')

        print(f'Downloading {len(merged)} images to {self.ROOT} ...')
        for idx, url, key in tqdm(zip(merged.index, merged['url'], merged['key']), total=len(merged)):
            fn = key + '.jpg'
            fp = os.path.join(self.ROOT, fn)
            if os.path.exists(fp):
                continue
            try:
                r = requests.get(url, timeout=2)
                with open(fp, 'wb') as imgf:
                    for chunk in r.iter_content():
                        imgf.write(chunk)
            except Exception:
                merged = merged.drop(index=idx)
                continue

        os.makedirs(os.path.dirname(self.MERGED_CSV), exist_ok=True)
        merged.to_csv(self.MERGED_CSV, index=False, encoding='utf-8')
        print(f'Done. CSV -> {self.MERGED_CSV}')

    def to_dataset(self):
        img_files = glob(os.path.join(self.ROOT, '*'))

        img_keys, img_list = [], []
        for ifn in tqdm(img_files):
            try:
                img = Image.open(ifn).resize((64, 64))
            except Exception:
                print(f'"{ifn}" cannot be resized by 64x64')
                continue
            img = np.asarray(img)
            if img.size == 64 * 64 * 3:
                key = os.path.basename(ifn).replace('.jpg', '')
                img_list.append(img)
                img_keys.append(key)

        attribute = pd.read_csv(self.MERGED_CSV, encoding='utf-8')
        attribute = attribute[attribute['key'].isin(img_keys)]

        TARGET_NAME = 'Male'
        BIAS_NAME = 'Heavy Makeup'

        to01 = np.vectorize(lambda s: 1 if s > 0 else 0)
        target_vect = to01(attribute[TARGET_NAME].to_numpy())
        bias_vect = to01(attribute[BIAS_NAME].to_numpy())

        temp = [im.ravel() for im in img_list]
        temp_df = pd.DataFrame(temp)
        temp_df[TARGET_NAME] = target_vect
        temp_df[BIAS_NAME] = bias_vect

        dataset = aifData(
            df=temp_df,
            label_name=TARGET_NAME,
            favorable_classes=[1],
            protected_attribute_names=[BIAS_NAME],
            privileged_classes=[[1]],
        )

        return {
            'aif_dataset': dataset,
            'image_list': img_list,
            'attribute': attribute,
            'target': target_vect,
            'bias': bias_vect
        }
