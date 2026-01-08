# MSIT AI Fair(MAF)

MSIT AI Fair(MAF)는 과학기술정보통신부 “점차 강화되고 있는 윤리 정책에 발맞춰 유연하게 진화하는 인공지능 기술 개발 연구(2022~2026)“ 국가과제의 일환으로, 인공지능(AI)의 공정성을 진단하고 편향성을 교정하는 진단 시스템입니다. 과거 “인공지능 모델과 학습데이터의 편향성 분석-탐지-완화-제거 지원 프레임워크 개발(2019-2022)” 국가과제 결과물의 연장선으로, 지속적으로 확장·개발되고 있습니다.

MAF는 데이터 편향성과 알고리즘 편향성을 측정 및 완화하는 것을 목표로 합니다. MAF는 IBM에서 공개한 AI Fairness 360(AIF360)의 브랜치로 시작하여 AIF360의 기본 기능을 담고 있으며, 과제 수행 기간 중 컨소시엄 내에서 개발된 편향성 완화 알고리즘의 추가, 지원 데이터 형식 추가, CPU 환경 지원 추가 등의 기능을 계속 확장하고 있습니다.

MAF 패키지는 python 환경에서 사용할 수 있습니다.

MAF 패키지에는 다음이 포함됩니다.
1. 모델에 대한 메트릭 세트 및 메트릭에 대한 설명
2. 데이터 세트 및 모델의 편향을 완화하는 알고리즘
      * 연구소 알고리즘은 음성, 언어, 금융, 의뢰 시스템, 의료, 채용, 치안, 광고, 법률, 문화, 방송 등 광범위한 분야에서 활용하기 위해 설계되었습니다.

확장 가능성을 두고 패키지를 개발하였으며 지속적으로 개발 및 업데이트를 진행 중입니다.

# Framework Outline
MAF는 크게 algorithms, benchmark, metric의 세 파트로 이루어져 있습니다.
algorithms 파트는 편향성 완화와 관련된 다양한 알고리즘이 포함되어있으며, AIF360의 분류를 따라 알고리즘을 Pre/In/Post Processing 3가지로 분류하고 있습니다. benchmark 파트는 편향성 완화와 관련된 각 benckmark를 테스트 해볼 수 있는 모듈들로 구성되어있으며, metric 파트는 편향성 측정 지표와 관련한 모듈들로 구성되어있습니다.

algorithms, benchmark, metric 각 파트의 구성은 다음과 같습니다.

## Algorithms
AIF360의 알고리즘 및 편향성 완화와 관련한 최신 연구들을 포함하고 있습니다.
### Pre-processing Algorithms
* AIF360
  * Disparate Impact Remover
  * Learning Fair Representation
  * Reweighing
  * Optim Preproc
  * Learning Fair Representation
* SOTA Algorithm
  * Co-occurrence-bias [📃paper](https://aclanthology.org/2023.findings-emnlp.518.pdf) [💻 code](https://github.com/CheongWoong/impact_of_cooccurrence)
  * Fair Streaming PCA [📃paper](https://arxiv.org/abs/2310.18593) [💻 code](https://github.com/HanseulJo/fair-streaming-pca/?tab=readme-ov-file)
  * Representative Heuristic [📚 data](https://github.com/jongwonryu/RH)
  * Fair Batch [📃paper](https://arxiv.org/abs/2012.01696) [💻 code](https://github.com/yuji-roh/fairbatch)

### In-processing Algorithms
* AIF360
  * Gerry Fair Classifier
  * Meta Fair Classifier
  * Prejudice Remover
  * Exponentiated Gradient Reduction
  * Adversarial Debiasing
* SOTA Algorithm
  * ConCSE [📃paper](https://arxiv.org/abs/2409.00120) [💻 code](https://github.com/jjy961228/ConCSE?tab=readme-ov-file)
  * INTapt [📃paper](https://arxiv.org/abs/2305.16371)
  * Fair Dimension Filtering
  * Fairness Through Matching [💻 code]("https://github.com/kwkimonline/FTM)
  * Fair Feature Distillation [📃paper](https://arxiv.org/abs/2106.04411) [💻 code](https://github.com/DQle38/Fair-Feature-Distillation-for-Visual-Recognition)
  * SLIDE [📃paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608022002891) [💻 code](https://github.com/kwkimonline/SLIDE?tab=readme-ov-file)
  * sIPM-LFR  [📃paper](https://arxiv.org/abs/2202.02943) [💻 code](https://github.com/kwkimonline/sIPM-LFR)
  * Learning From Fairness [📃paper](https://arxiv.org/abs/2007.02561) [💻 code](https://github.com/alinlab/LfF)
  * Fairness VAE [📃paper](https://arxiv.org/abs/2007.03775) [💻 code](https://github.com/sungho-CoolG/Fairness-VAE)
  * Kernel Density Estimator [📃paper](https://proceedings.neurips.cc/paper/2020/hash/ac3870fcad1cfc367825cda0101eee62-Abstract.html) [💻 code](https://github.com/Gyeongjo/FairClassifier_using_KDE)
  * DMLBG
  * FairASR [📃paper](https://arxiv.org/abs/2506.10747) [💻 code](https://github.com/JongSuk1/FairASR)

### Post-processing Algorithms
* AIF360
  * Calibrated EqOdds
  * Equalized Odds
  * Reject Option Classifier
* SOTA Algorithm
  * Causal Path Tracing [📃paper](TBA) [💻 code](TBA)
  * EMBER [📃paper](https://arxiv.org/abs/2410.20774) [💻 code](https://github.com/DongryeolLee96/EMBER)

## Benchmark
편향성과 관련한 benchmark에 대한 연구들을 포함하고 있습니다.

* KoBBQ [📃paper](https://arxiv.org/abs/2307.16778) [💻 code](https://github.com/naver-ai/KoBBQ?tab=readme-ov-file)
* CREHate [📃paper](https://arxiv.org/abs/2308.16705) [💻 code](https://github.com/nlee0212/CREHate)
* BBG [📃paper](https://arxiv.org/abs/2503.06987) [💻 code](https://github.com/jinjh0123/BBG)

## Metric
편향성을 측정할 수 있는 metric과 관련한 연구들을 포함하고 있습니다.

* Latte [📃paper](https://arxiv.org/pdf/2402.06900v3)

MAF에서는 AIF360에 제시된 편향성 관련 metric도 함께 지원하고 있습니다.
### Data metrics
* Number of negatives (privileged)
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

MAF의 원활한 구동을 위해서는 특정 버전의 패키지들이 필요합니다. 시스템의 다른 프로젝트와 충돌할 수 있으므로 anaconda 가상 환경 혹은 docker를 권장드립니다.

### Installation
1. 저장소 복제
    ```bash
    git clone https://github.com/konanaif/MAF.git
    ```

2. 환경 설정
   - anaconda 가상 환경 사용 시, 가상 환경 생성 후 필요 패키지를 설치합니다.
        ```bash
        conda install --file requirements.txt
        ```
    - docker 이용 시, Dockerfile을 통해 이미지를 빌드합니다. MAF의 기본 작업 공간은 workspace 입니다.
        ```bash
        docker build -f Dockerfile -t maf:v1 ..
        ```

3. 텍스트의 경우, 외부 API를 이용합니다. API 이용을 위해 별도의 KEY 설정이 필요합니다.
    ```bash
    #OPENAI API KEY 설정 예시
    export OPENAI_API_KEY = 'your_api_key'
    ```

4. 데이터 및 모델 세팅
   MAF에서는 tabular, text, image, audio의 4가지 타입의 데이터를 지원하고 있으며, 세부 리스트는 다음과 같습니다.
   - tabular
     - COMPAS [📚 data](https://github.com/propublica/compas-analysis/)
     - German credit scoring [📚 data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
     - Adult Census Income [📚 data](https://archive.ics.uci.edu/dataset/2/adult)

   - image
     - Public Figures Face Database [📚 data](https://www.cs.columbia.edu/CAVE/databases/pubfig/download/)
     - CelebA [📚 data](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
       - 해당 데이터는 datamodule/dataset.py - CelebADataset를 통해 로드됩니다. 시스템 메모리에 따라 CelebADataset 클래스 내 dask.dataframe.from_pandas() 함수의 npartitions 변수값 조정이 필요할 수 있습니다. (https://docs.dask.org/en/stable/generated/dask.dataframe.from_pandas.html?highlight=from_pandas#dask.dataframe.from_pandas)

   - text
     - KoBBQ [📚 data](https://github.com/naver-ai/KoBBQ/tree/main)
     - CREHate [📚 data](https://github.com/nlee0212/CREHate)
     - co-occurrence-bias [📚 data](https://drive.google.com/file/d/19I7ron7FycqqJqRH0vdVHW_nCAKOf5g5/view)
     - latte
        [📚 baq_questionnaire](https://pubmed.ncbi.nlm.nih.gov/24115185/)
        [📚 bbq](https://github.com/nyu-mll/BBQ)
        [📚 virtue, deontology, utilitarianism](https://www.catalyzex.com/paper/aligning-ai-with-shared-human-values/code)
        [📚 hate](https://osf.io/53tfs/)
        [📚 fair](https://paperswithcode.com/dataset/fairprism)
        [📚 proso, proso_toxic](https://paperswithcode.com/dataset/prosocialdialog)
        [📚 detox](https://github.com/s-nlp/paradetox)
        [📚 political_compass](https://www.politicalcompass.org/test/en?page=1)
     - RH [📚 data](https://github.com/jongwonryu/RH)
     - Koglish [📚 data](https://huggingface.co/Jangyeong)
     - ember [📚 data](https://github.com/DongryeolLee96/EMBER) #full data

    - audio
      - esyoon/coraal_clean_test [📚 data](https://huggingface.co/datasets/esyoon/coraal_clean_test)
      - Fair-speech Dataset [📚 data](https://ai.meta.com/datasets/speech-fairness-dataset)

    개별 다운로드가 필요한 모델은 다음과 같습니다.
    - ConCSE [📚 model](https://drive.google.com/drive/folders/1k3JDP4WfRkVTypaiL3L1RO1qeve2yvhF?usp=sharing)
    - Fair Dimension Filtering
         - Filter_model.th [📚 model](https://drive.google.com/file/d/1ZgQIBYghDpQ7lkKD3UnDwJCnJupPbkrF/view?usp=sharing)
         - baseline.th [📚 model](https://drive.google.com/file/d/14UvLw8ZQMizJgy0ZALA67m0Cmdb04Dp_/view?usp=sharing)

   개별 알고리즘에 따른 데이터 및 모델 세팅이 필요합니다. 데이터와 모델은 각각 data와 model 폴더를 생성하여 다음과 같은 구조로 설정합니다.

       4-1. data
        ```bash
            data
              ㄴadult
                ㄴadult.data
                ㄴadult.names
                ㄴadult.test
              ㄴceleba
                ㄴimg_align_celeba
                ㄴlist_attr_celeba.csv
                ㄴlist_attr_celeba.txt
                ㄴlist_eval_partition.csv
              ㄴco-occurrence-bias
                ㄴdata_statistics #링크를 통해 다운로드
                ㄴLAMA_TREx #scripts/setup/download_LAMA.sh, scripts/setup/preprocess_LAMA_TREx.sh
                ㄴoriginal_LAMA #prepare_dataset.sh
                ㄴscripts
                ㄴprepare_dataset.sh
                ㄴpreprocess_LAMA_TREx.py
              ㄴcompas
                ㄴcompas-scores-two-years.csv
              ㄴcrehate
                ㄴCREHate_CP.csv
                ㄴCREHate_SBIC.csv
              ㄴgerman
                ㄴgerman.data
              ㄴINTapt
                ㄴdownload_data_model.py #esyoon/coraal_clean_test 데이터 및 모델 저장
                ㄴesyoon___coraal_clean_test
                ㄴmodels--esyoon--INTapt-HuBERT-large-coraal-prompt-generator
                ㄴmodels--facebook--hubert-large-ls960-ft
              ㄴkobbq
                ㄴkobbq_data
                    ㄴKoBBQ_test_samples.tsv
                ㄴ0_evaluation_prompts.tsv
              ㄴKoglish_dataset
                ㄴdownload_Koglish_dataset.py #Koglish 데이터 및 모델 저장
                ㄴKoglish_STS
                ㄴKoglish_NLI
                ㄴKoglish_GLUE
              ㄴlatte
                ㄴbaq_questionnaire.csv
                ㄴbbq.csv
                ㄴdeontology.csv
                ㄴdetox.csv
                ㄴfair.csv
                ㄴhate.csv
                ㄴpolitical_compass.csv
                ㄴproso_toxic.csv
                ㄴproso.csv
                ㄴutilitarianism.csv
                ㄴvirtue.csv
              ㄴpubfig
                ㄴimage
                ㄴdev_urls.txt
                ㄴpubfig_attr_merged.csv
                ㄴpubfig_attributes.txt
              ㄴRH
                ㄴRH_dataset.xlsx
              ㄴcasual_path_tracing
                ㄴlama_trex.json
              ㄴember # 원본 repo를 통해 full data 다운로드 필수
                ㄴif
                  ㄴember_if.json
                ㄴqa
                  ㄴember_qa_gpt4.json
                  ㄴember_qa_newbing.json
         ```

        4-2. model
        ```bash
            model
              ㄴConCSE
                ㄴmbert_uncased
                ㄴxlmr_base
                ㄴxlmr_large
              ㄴFairFiltering
                ㄴbaseline.th
                ㄴFilter_model.th
        ```
