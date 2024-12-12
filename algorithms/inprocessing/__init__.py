from MAF.algorithms.inprocessing.concse import mitigate_concse
from MAF.algorithms.inprocessing.INTapt.intapt import mitigate_intapt
from MAF.algorithms.inprocessing.exponentiated_gradient_reduction import (
    ExponentiatedGradientReduction,
)
from MAF.algorithms.inprocessing.meta_classifier import MetaFairClassifier
from MAF.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from MAF.algorithms.inprocessing.prejudice_remover import PrejudiceRemover
from MAF.algorithms.inprocessing.slide import SlideFairClassifier
from MAF.algorithms.inprocessing.ftm import FTMFairClassifier
from MAF.algorithms.inprocessing.fair_dimension_filtering import FairDimFilter
from MAF.algorithms.inprocessing.fair_feature_distillation import (
    FairFeatureDistillation,
)
from MAF.algorithms.inprocessing.fairness_vae import FairnessVAE
from MAF.algorithms.inprocessing.learning_from_fairness import LearningFromFairness
from MAF.algorithms.inprocessing.kernel_density_estimation import (
    KDEParameters,
    KernelDensityEstimation,
)
