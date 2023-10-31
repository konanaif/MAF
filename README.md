# MAF

MSIT AI Fair(MAF)는 과학기술정보통신부 “인공지능 모델과 학습데이터의 편향성 분석-탐지-완화-제거 지원 프레임워크 개발(2019-2022)” 국가과제의 일환으로 개발된 프레임워크로 인공지능(AI)의 공정성을 진단하고 편향성을 교정하는 진단 시스템입니다.

MAF는 데이터 편향성과 알고리즘 편향성을 측정 및 완화하는 것을 목표로 합니다. MAF는 IBM에서 공개한 AI Fairness 360(AIF360)의 브랜치로 시작하여 AIF360의 기본 기능을 담고 있으며, 과제 수행 기간 중 컨소시엄 내에서 개발된 편향성 완화 알고리즘의 추가, 지원 데이터 형식 추가, CPU 환경 지원 추가 등의 기능을 계속 확장하고 있습니다.

MAF 패키지는 python 환경에서 사용할 수 있습니다.

MAF 패키지에는 다음이 포함됩니다
1. 편향성을 테스트할 데이터 세트
2. 모델에 대한 메트릭 세트 및 메트릭에 대한 설명
3. 데이터 세트 및 모델의 편향을 완화하는 알고리즘
      * 연구소 알고리즘은 음성, 언어, 금융, 의뢰 시스템, 의료, 채용, 치안, 광고, 법률, 문화, 방송 등 광범위한 분야에서 활용하기 위해 설계되었습니다.
   
지속적인 확장 가능성을 두고 패키지를 개발했으며, 이 프레임워크는 개발 진행중입니다.

# Bias mitigation algorithms
MAF는 AIF360의 workflow를 차용했으므로, 처리 과정은 Pre/In/Post Processing의 3단계를 가집니다.

### Pre-Processing Algorithms
* Disparate_Impact_Remover(From AIF360)
* Learning_Fair_Representation(From AIF360)
* Reweighing(From AIF360)
### In-Processing Algorithms
* Gerry_Fair_Classifier(From AIF360)
* Meta_Fair_Classifier(From AIF360)
* Prejudice_Remover(From AIF360)
### Post-Processing Algorithms
* Calibrated_EqOdds(From AIF360)
* EqualizedOdds(From AIF360)
* RejectOption(From AIF360)
### Sota Algorithms
* FairBatch
* FairFeatureDistillation(Image only)
* FairnessVAE(Image only)
* KernelDensityEstimator
* LearningFromFairness(Image only)

### Algorithms to add
* fair-manifold-pca


# Supported fairness metrics
### Data metrics
* Number of nagatives (privileged)
* Number of positives (privileged)
* Number of negatives (unprivileged)
* Number of positives (unprivileged)
* Base rate
* Statistical parity difference
* Consistency
### Classification metrics
* Error rate
* Average odds difference
* Average abs odds difference
* Selection rate
* Disparate impact
* Statistical parity difference
* Generalized entropy index
* Theil index
* Equal opportunity difference

# Setup
Supported Python Configurations:

| OS      | Python version |
| ------- | -------------- |
| macOS   | 3.8 – 3.11     |
| Ubuntu  | 3.8 – 3.11     |
| Windows | 3.8 – 3.11     |

### (Optional) Create a virtual environment
MAF에는 시스템의 다른 프로젝트와 충돌할 수 있는 많은 Python 패키지의 특정 버전이 필요합니다. 종속성이 안전하게 설치되도록 하려면 anaconda 가상 환경을 사용하는 것이 좋습니다.


### Install with pip
이 저장소의 최신 버전을 복제합니다.
```bash
git clone https://github.com/konanaif/MAF.git
```

