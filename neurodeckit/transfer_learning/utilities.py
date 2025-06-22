# This file contains the implementation of various transfer learning methods.
# Author: LC.Pan
# Date: 2024.6.21

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.multiclass import OneVsRestClassifier

from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
# from sklearn.linear_model import Lasso # Lasso 回归, L1 正则化
from sklearn.feature_selection import RFE # 递归特征消除
from sklearn.feature_selection import RFECV # 递归特征消除(结合交叉验证自动选择最佳特征数量)
from ..machine_learning import LassoSelector as Lasso # 序列特征选择

from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR # 逻辑回归
from sklearn.neighbors import KNeighborsClassifier as KNN # K 近邻
from sklearn.tree import DecisionTreeClassifier as DTC # 决策树
from sklearn.ensemble import RandomForestClassifier as RFC # 随机森林
from sklearn.ensemble import ExtraTreesClassifier as ETC # 极端随机森林
# from sklearn.ensemble import AdaBoostClassifier as ABC # AdaBoost
from ..ensemble_learning import AdaBoost as ABC # AdaBoost
from sklearn.ensemble import GradientBoostingClassifier as GBC # GradientBoosting
from sklearn.naive_bayes import GaussianNB as GNB # 高斯朴素贝叶斯
from sklearn.neural_network import MLPClassifier as MLP # 多层感知机

from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.classification import KNearestNeighbor as RKNN
from pyriemann.classification import SVC as RKSVM
from pyriemann.transfer import TLDummy, TLStretch, TLRotate, MDWM

from ..machine_learning import RiemannCSP as CSP
from ..machine_learning import TRCSP, DCPM, TRCA, SBLEST, TSclassifier, TS, FGDA, FgMDM
# from .mekt import MEKT
from .mekt import MEKT_supervised as MEKT #有监督的MEKT
from .mekt import MEKT_improved as MEKT_P #改进的MEKT LC.Pan 2024.09.01
from .mekt import MEKT_improved2 as MEKT_P2 #改进的MEKT LC.Pan 2024.09.01
from .kl import KEDA # 未完成版本，还需要继续改进
from .base import decode_domains, encode_datasets, TLClassifier
from .rpa import RCT, STR, ROT
from .rpa import TLCenter_online as TLCenter

__all__ = [
    'BaseEstimator', 'ClassifierMixin', 'clone', 'accuracy_score',
    'make_pipeline', 'Pipeline', 'OneVsRestClassifier', 'SelectKBest',
    'SelectPercentile', 'f_classif','mutual_info_classif', 'PCA', 'Lasso',
    'RFE', 'RFECV', 'SVC', 'LDA', 'LR', 'KNN', 'DTC', 'RFC', 'ETC', 'ABC',
    'GBC', 'GNB', 'MLP', 'Covariances', 'TS', 'FGDA', 'MDM', 'FgMDM', 'RKNN',
    'RKSVM', 'TLDummy', 'TLCenter', 'TLStretch', 'TLRotate', 'MDWM', 'CSP',
    'TRCSP', 'DCPM', 'TRCA', 'SBLEST', 'TSclassifier', 'MEKT', 'RCT', 'STR',
    'ROT', 'decode_domains', 'encode_datasets', 'TLClassifier', 
    'MEKT_P', 'MEKT_P2',
    'KEDA',
]


from ..utils import extract_dict_keys

def estimator_list():
    DPA_Methods = extract_dict_keys('transfer_learning.tl_classifier', 'TL_Classifier', 'check_dpa', 'prealignments')
    FEE_Methods = extract_dict_keys('transfer_learning.tl_classifier', 'TL_Classifier', 'check_fee', 'feature_extractions')
    FES_Methods = extract_dict_keys('transfer_learning.tl_classifier', 'TL_Classifier', 'check_fes', 'feature_selections')
    CLF_Methods = extract_dict_keys('transfer_learning.tl_classifier', 'TL_Classifier', 'check_clf', 'classifiers')
    END_Methods = extract_dict_keys('transfer_learning.tl_classifier', 'TL_Classifier', 'check_endest', 'estimators')
    END_TO_END_Methods = extract_dict_keys('transfer_learning.tl_classifier', 'TL_Classifier', 'check_endtoend', 'endtoends')
    
    DPA_Methods = [None if item == 'NONE' else item for item in DPA_Methods]
    FEE_Methods = [None if item == 'NONE' else item for item in FEE_Methods]
    FES_Methods = [None if item == 'NONE' else item for item in FES_Methods]
    CLF_Methods = [None if item == 'NONE' else item for item in CLF_Methods]
    END_Methods = [None if item == 'NONE' else item for item in END_Methods]
    END_TO_END_Methods = [None if item == 'NONE' else item for item in END_TO_END_Methods]
    
    return DPA_Methods, FEE_Methods, FES_Methods, CLF_Methods, END_Methods, END_TO_END_Methods


