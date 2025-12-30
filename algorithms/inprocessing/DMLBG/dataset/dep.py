# dataset/pubfig_aif.py
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import sys
# FFD 스크립트에서 쓰던 경로 계승: sample 모듈이 위로 2단계에 있다고 가정
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

# 이제 datamodule.dataset에서 가져오기
from datamodule.dataset import PubFigDataset, aifData

class dep(Dataset):
    """
    PubFigDataset/aifData 기반 AIF360 스타일 데이터를
    DML 트레이너가 요구하는 PyTorch Dataset 인터페이스로 감싼 래퍼.
    - mode: 'train' | 'eval'
    - transform: dataset.utils.make_transform(...) 그대로 사용
    """
    def __init__(self, root, mode='train', transform=None, seed=1,
                 image_shape=(3, 64, 64), protected_name='Heavy Makeup'):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.image_shape = tuple(image_shape)  # (C,H,W)
        self.protected_name = protected_name

        # 1) 원본 PubFig 로딩 (너 스크립트 로직과 동일)
        self.pubfig = PubFigDataset()
        # 샘플 폴더 확인 후 필요 시 다운로드
        if not os.path.isdir(os.path.join(self.root, 'Sample', 'pubfig')) and not os.path.isdir('./Sample/pubfig'):
            # self.pubfig.download() 는 내부 경로 기준이므로,
            # 루트 기준으로 동작 안 하면 여기서 한 번 호출해 주는 정도로만.
            self.pubfig.download()

        # 2) dict 형태 데이터셋 획득
        self.dataset = self.pubfig.to_dataset()  # {'aif_dataset', 'image_list', ...}

        # 3) AIF dataset 분할 (train/eval)
        aif = self.dataset['aif_dataset']
        self.privileged_groups = [{self.protected_name: 1}]
        self.unprivileged_groups = [{self.protected_name: 0}]
        # 70/30 split, seed 고정(FFD 코드와 동일)
        self.dataset_train, self.dataset_test = aif.split([0.7], shuffle=True, seed=seed)

        # 4) 모드에 맞는 뷰 선택
        if self.mode == 'train':
            self.split = self.dataset_train
        elif self.mode == 'eval':
            self.split = self.dataset_test
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # 5) (이미지, 라벨) 준비
        # - features: (N, 64*64*3 + 1) 형태로 가정(마지막 컬럼이 보호속성)
        # - labels: (N, 1)
        # - instance_names가 있으면 인덱싱에 사용 가능(옵션)
        X = self.split.features
        y = self.split.labels.ravel().astype(int)

        # 보호속성 컬럼을 제외한 픽셀만 추출
        if X.ndim != 2:
            raise RuntimeError(f"Expected 2D features, got {X.shape}")
        C, H, W = self.image_shape
        pixels = X[:, :-1]  # 마지막 컬럼은 protected attribute

        # (N, C*H*W) -> (N, H, W, C) -> PIL.Image
        # AIF 데이터가 0~255 정수라고 가정 (FFD 코드에서 dtype='int'로 flatten 했음)
        if pixels.shape[1] != C * H * W:
            raise RuntimeError(f"Feature length {pixels.shape[1]} != C*H*W {C*H*W}")
        self._imgs_np = pixels.reshape(-1, C, H, W)  # (N, C, H, W)
        # PIL은 (H, W, C) 를 선호하므로 on-the-fly로 변환

        # 6) 라벨 0..C-1 보장
        classes = np.unique(y)
        self._class_map = {c: i for i, c in enumerate(sorted(classes))}
        self.targets = [self._class_map[c] for c in y]  # BalancedSampler 호환용
        self._num_classes = len(self._class_map)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # (C, H, W) uint8 → PIL → transform → tensor
        img_chw = self._imgs_np[idx]
        if img_chw.dtype != np.uint8:
            # 안전하게 uint8로 변환 (0~255 범위로 클램프)
            arr = np.clip(img_chw, 0, 255).astype(np.uint8)
        else:
            arr = img_chw
        # (C,H,W) -> (H,W,C)
        img_hwc = np.transpose(arr, (1, 2, 0))
        img = Image.fromarray(img_hwc)

        if self.transform is not None:
            img = self.transform(img)

        label = self.targets[idx]
        return img, label

    # DML 트레이너가 호출
    def nb_classes(self):
        return self._num_classes
