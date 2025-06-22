# This file is the main file of the NeuroDecKit package.
from . import analysis
from . import loaddata
from . import deep_learning
from . import ensemble_learning
from . import machine_learning
from . import pre_processing
from . import transfer_learning
from . import utils

from .loaddata import Dataset_Left_Right_MI, Dataset_MI
from .transfer_learning import TLSplitter, encode_datasets
from .pre_processing.preprocessing import Pre_Processing
from .machine_learning.ml_classifier import ML_Classifier
from .ensemble_learning.el_classifier import EL_Classifier
from .transfer_learning.tl_classifier import TL_Classifier
from .deep_learning.dl_classifier import DL_Classifier

__all__ = [
    '__version__',
    'analysis',
    'loaddata',
    'deep_learning',
    'ensemble_learning',
    'machine_learning',
    'pre_processing',
    'transfer_learning',
    'utils',
    'Dataset_Left_Right_MI',
    'Dataset_MI',
    'TLSplitter',
    'encode_datasets',
    'Pre_Processing',
    'ML_Classifier',
    'EL_Classifier',
    'TL_Classifier',
    'DL_Classifier'
]