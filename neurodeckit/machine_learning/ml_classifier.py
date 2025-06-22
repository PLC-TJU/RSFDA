"""
ML_Classifier: Traditional Machine Learning Classifier for EEG Data
Author: LC.Pan <panlincong@tju.edu.cn.com>
Date: 2024/3/25
License: All rights reserved

备注：
1. 该文件是机器学习分类器的实现，包括CSP、LDA、SVM、FBCSP、SBLEST、TRCA等分类器。
2. 该文件仅供参考，所集成的算法源码均来自于其它项目，如scikit-learn、pyriemann、metabci等。
3. 所使用的各种算法的具体实现可能与matlab版本有所差异。
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from . import RiemannCSP as CSP
from . import Covariances as Cov
from . import Kernels as Ker
from . import Shrinkage as Shr
from . import (FB, FBCSP, FBTRCA, DSP, FBDSP, DCPM,
               MDM, FgMDM, TS, TSclassifier, RKSVM, TRCA, SBLEST, TRCSP)
from .base import generate_filterbank
from ..utils import generate_intervals, adjust_intervals
from ..deep_learning import Formatdata


class ML_Classifier(BaseEstimator, ClassifierMixin): 
    def __init__(self, model_name='CSP', **kwargs):
        self.model_name = model_name
        
        # 以下参数均可通过kwargs传入，若未传入则使用默认值
        self.kwargs = kwargs
        self.fs = kwargs.get('fs', None)
        self.cov_estimator = kwargs.get('cov_estimator', 'cov') # {'scm', 'cov', 'lwf'}
        self.shr_value = kwargs.get('shr_value', None)
        self.n_components = kwargs.get('n_components', 8)
        self.rsf_method = kwargs.get('rsf_method', None) 
        self.rsf_dim = kwargs.get('rsf_dim', None)
        self.freqband = kwargs.get('freqband', [5, 32])
        self.filterbank = kwargs.get('filterbank', generate_intervals(4, 4, (4, 40)))
        self.n_components_fb = kwargs.get('n_components_fb', 24) 
        self.device = kwargs.get('device', 'cuda')  # SBLEST only
        self.svm_kernel = kwargs.get('svm_kernel', ['linear', 'rbf']) #  {"svm_kernel": ["linear", "rbf"]}
        self.svm_C = kwargs.get('svm_C', [0.1, 1, 10]) #  {"svm_C": np.logspace(-2, 2, 10)}
        self.lda_shrinkage = kwargs.get('lda_shrinkage', ['None', 'auto']) #  {"lda_shrinkage": ['None', 'auto']}
        self.lda_solver = kwargs.get('lda_solver', ['eigen','lsqr']) #  {"lda_solver": ['eigen', 'lsqr']}
        self.kernel_fct = kwargs.get('kernel_fct', None) #  {"kernel_fct": [None, 'precomputed']}
        self.n_jobs = kwargs.get('n_jobs', None)
        
        self.param_svm = {"kernel": self.svm_kernel, "C": self.svm_C}
        self.param_lda = {"shrinkage": self.lda_shrinkage, "solver": self.lda_solver}
        
        # 传统机器学习分类器列表
        if self.model_name in ['CSP-LDA','CSP']:
            self.clf = make_pipeline(Cov(estimator=self.cov_estimator), 
                                     CSP(nfilter=self.n_components, metric='euclid'), 
                                     GridSearchCV(LDA(), self.param_lda, cv=3, n_jobs=self.n_jobs)
                                     )
        elif self.model_name == 'CSP-SVM':
            self.clf = make_pipeline(Cov(estimator=self.cov_estimator), 
                                     CSP(nfilter=self.n_components, metric='euclid'), 
                                     GridSearchCV(SVC(), self.param_svm, cv=3, n_jobs=self.n_jobs)
                                     )
        elif self.model_name in ['TRCSP-LDA','TRCSP']:
            self.clf = make_pipeline(Cov(estimator=self.cov_estimator), 
                                     TRCSP(nfilter=self.n_components, metric='riemann'), 
                                     GridSearchCV(LDA(), self.param_lda, cv=3, n_jobs=self.n_jobs)
                                     )
        elif self.model_name == 'TRCSP-SVM':
            self.clf = make_pipeline(Cov(estimator=self.cov_estimator), 
                                     TRCSP(nfilter=self.n_components, metric='riemann'), 
                                     GridSearchCV(SVC(), self.param_svm, cv=3, n_jobs=self.n_jobs)
                                     )
        elif self.model_name in ['RiemannCSP-LDA','RiemannCSP']:
            self.clf = make_pipeline(Cov(estimator=self.cov_estimator), 
                                     CSP(nfilter=self.n_components, metric='riemann'), 
                                     GridSearchCV(LDA(), self.param_lda, cv=3, n_jobs=self.n_jobs)
                                     )
        elif self.model_name == 'RiemannCSP-SVM':
            self.clf = make_pipeline(Cov(estimator=self.cov_estimator), 
                                     CSP(nfilter=self.n_components, metric='riemann'), 
                                     GridSearchCV(SVC(), self.param_svm, cv=3, n_jobs=self.n_jobs)
                                     )
        elif self.model_name in ['oFBCSP','oFBCSP-LDA','FBCSP','FBCSP-LDA']:
            if self.fs is None:
                ValueError('fs is not specified')
            filterbanks = generate_filterbank(self.filterbank,
                                             adjust_intervals(self.filterbank),
                                             srate=self.fs,
                                             order=4)
            self.clf = make_pipeline(FBCSP(n_components=self.n_components,
                                           n_mutualinfo_components=self.n_components_fb,
                                           filterbank=filterbanks), 
                                     GridSearchCV(LDA(), self.param_lda, cv=3, n_jobs=self.n_jobs)
                                     )
        elif self.model_name in ['oFBCSP-SVM','FBCSP-SVM']:
            if self.fs is None:
                ValueError('fs is not specified')
            filterbanks = generate_filterbank(self.filterbank,
                                             adjust_intervals(self.filterbank),
                                             srate=self.fs,
                                             order=4)
            self.clf = make_pipeline(FBCSP(n_components=self.n_components,
                                           n_mutualinfo_components=self.n_components_fb,
                                           filterbank=filterbanks), 
                                     GridSearchCV(SVC(), self.param_svm, cv=3, n_jobs=self.n_jobs)
                                     )
        elif self.model_name in ['FB-CSP-LDA','FB-CSP','RSF-FB-CSP-LDA','RSF-FB-CSP']:
            # some unknow bugs in these models, so we use the FBCSP-LDA model instead
            if self.fs is None:
                ValueError('fs is not specified')
            if self.model_name.startswith('RSF'):
                self.rsf_method = 'default' if (self.rsf_method is None or 
                                                self.rsf_method.lower() == 'none') else self.rsf_method
            Process = Formatdata(fs=self.fs, alg_name=self.model_name, rsf_method=self.rsf_method, 
                                 rsf_dim=self.rsf_dim)
            Basemodel = FB(make_pipeline(Cov(estimator=self.cov_estimator), CSP(nfilter=self.n_components)))
            Feaselect = SelectKBest(score_func=mutual_info_classif, k=self.n_components_fb)
            OptLDA = GridSearchCV(LDA(), self.param_lda, cv=3, n_jobs=self.n_jobs)
            self.clf = make_pipeline(Process, Basemodel, Feaselect, OptLDA)
        elif self.model_name in ['FB-CSP-SVM','RSF-FB-CSP-SVM']:
            # some unknow bugs in these models, so we use the FBCSP-SVM model instead
            if self.fs is None:
                ValueError('fs is not specified')
            if self.model_name.startswith('RSF'):
                self.rsf_method = 'default' if (self.rsf_method is None or 
                                                self.rsf_method.lower() == 'none') else self.rsf_method
                
            Process = Formatdata(fs=self.fs, alg_name=self.model_name, rsf_method=self.rsf_method, 
                                 rsf_dim=self.rsf_dim)
            Basemodel = FB(make_pipeline(Cov(estimator=self.cov_estimator), CSP(nfilter=self.n_components)))
            Feaselect = SelectKBest(score_func=mutual_info_classif, k=self.n_components_fb)
            OptSVM = GridSearchCV(SVC(), self.param_svm, cv=3, n_jobs=self.n_jobs)
            self.clf = make_pipeline(Process, Basemodel, Feaselect, OptSVM)
        elif self.model_name in ['MDM', 'MDRM']:
                self.clf = make_pipeline(Cov(estimator=self.cov_estimator), 
                                        MDM())
        elif self.model_name == 'FgMDM':
                self.clf = make_pipeline(Cov(estimator=self.cov_estimator), 
                                        FgMDM(n_jobs=self.n_jobs))
        elif self.model_name == 'TS-LDA':
            self.clf = make_pipeline(Cov(estimator=self.cov_estimator), 
                                     TS(),
                                     GridSearchCV(LDA(), self.param_lda, cv=3, n_jobs=self.n_jobs)
                                     )
        elif self.model_name == 'TS-SVM':
            self.clf = make_pipeline(Cov(estimator=self.cov_estimator), 
                                     TS(),
                                     GridSearchCV(SVC(), self.param_svm, cv=3, n_jobs=self.n_jobs)
                                     )
        elif self.model_name == 'TSLDA':
            self.clf = make_pipeline(Cov(estimator=self.cov_estimator), 
                                     TSclassifier(clf=LDA(solver='eigen', shrinkage='auto'))
                                     )
        elif self.model_name == 'TSSVM':
            self.clf = make_pipeline(Cov(estimator=self.cov_estimator), 
                                     TSclassifier(clf=SVC(kernel='linear', C=1))
                                     )
        elif self.model_name == 'TSGLM':
            self.clf = make_pipeline(Cov(estimator=self.cov_estimator), 
                                     TSclassifier()
                                     )
        elif self.model_name == 'RKSVM':
            self.clf = make_pipeline(Cov(estimator=self.cov_estimator), 
                                     RKSVM(metric='riemann', kernel_fct=self.kernel_fct)
                                     )
        elif self.model_name == 'TRCA':
            self.clf = TRCA(n_components=self.n_components)
        elif self.model_name == 'FBTRCA':
            
            if self.fs is None:
                ValueError('fs is not specified')
            filterbanks = generate_filterbank(self.filterbank,
                                             adjust_intervals(self.filterbank),
                                             srate=self.fs,
                                             order=4)
            self.clf = FBTRCA(n_components=self.n_components, filterbank=filterbanks)
        elif self.model_name == 'DSP':
            self.clf = DSP(n_components=self.n_components)
        elif self.model_name == 'FBDSP':
            if self.fs is None:
                ValueError('fs is not specified')
            filterbanks = generate_filterbank(self.filterbank,
                                             adjust_intervals(self.filterbank),
                                             srate=self.fs,
                                             order=4)            
            self.clf = FBDSP(n_components=self.n_components, filterbank=filterbanks)
        elif self.model_name == 'DCPM':
            self.clf = DCPM(n_components=self.n_components)
        elif self.model_name in ['SBLEST']:
            self.clf = SBLEST(K=2, tau=1, device=self.device) #使用默认参数
        else:
            raise ValueError('Invalid method')

    def fit(self, X, y):

        self.clf.fit(X.copy(), y.copy())
        return self
    
    def predict(self, X):
        return self.clf.predict(X.copy())
    

if __name__ == '__main__':
    from ..loaddata import Dataset_Left_Right_MI
    from joblib import Parallel, delayed
    from sklearn.model_selection import StratifiedKFold
    import time

    # Define a function to perform cross-validation manually
    def cross_validate(model, X, y, cv=5):
        kf = StratifiedKFold(n_splits=cv, random_state=42, shuffle=True)
        
        def fit_and_score(train, test):
            model.fit(X[train], y[train])
            predictions = model.predict(X[test])
            return accuracy_score(y[test], predictions)
        
        scores = Parallel(n_jobs=cv)(delayed(fit_and_score)(train, test) for train, test in kf.split(X,y))
        
        return np.array(scores)


    # Load the dataset
    dataset_name = 'BNCI2014_001'
    subject =[1]
    fs = 160
    dataset = Dataset_Left_Right_MI(dataset_name, fs)
    data, label = dataset.get_data(subject)

    # Define the algorithms to test
    algos = ['CSP-LDA', 'CSP-SVM', 'TRCSP-LDA', 'TRCSP-SVM',
         'RiemannCSP-LDA', 'RiemannCSP-SVM', 'FBCSP-LDA', 'FBCSP-SVM', 
         'MDM','FgMDM','TS-LDA','TS-SVM', 'TSLDA', 'TSSVM', 'TSGLM', 'RKSVM', 
         'TRCA','FBTRCA', 'DSP', 'FBDSP', 'DCPM', 'SBLEST']

    # Find the longest algorithm name for formatting
    max_length = len(max(algos, key=len))
    
    for algo in algos:
        start_time = time.time()
        clf = ML_Classifier(model_name=algo, fs=fs, n_components=8, n_components_fb=24, 
                            svm_kernel=['linear'], svm_C=[1], rsf_method=None, rsf_dim=8)
        scores = cross_validate(clf, data, label, cv=5)
        end_time = time.time()
        print("{:<{}} Accuracy: {:0.2f} (+/- {:0.2f}) Time: {:0.2f} s".format(
            algo, max_length + 2, scores.mean(), scores.std() * 2, end_time - start_time))
