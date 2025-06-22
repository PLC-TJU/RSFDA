## This is the RAVE algorithm implementation in Python.

# Matlab code: https://github.com/PLC-TJU/RAVE
# Note: This python implementation is not the same as the Matlab code.

# L. Pan et al., "Riemannian geometric and ensemble learning for decoding cross-session 
# motor imagery electroencephalography signals," J Neural Eng, vol. 20, no. 6, p. 066011, 
# Nov 22 2023, doi: 10.1088/1741-2552/ad0a01.

import numpy as np
import itertools

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR
# from sklearn.ensemble import StackingClassifier
# from sklearn.model_selection import cross_val_score

from joblib import Parallel, delayed
from joblib import Memory
from copy import deepcopy, copy

from ..pre_processing.preprocessing import Pre_Processing
from ..transfer_learning.tl_classifier import TL_Classifier
from ..transfer_learning import decode_domains

class EL_Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 fs_new=250,
                 timesets=[[0, 4],[0, 2],[1, 3],[2, 4],[0, 3], [1, 4]],
                # timesets=[[0, 4],[0, 2],[0.5, 2.5],[1, 3],[1.5, 3.5], [2, 4]],
                 freqsets=[[8, 30],[8, 13],[13, 18],[18, 26],[26, 30]],
                #  timesets=[[0, 4],[0, 2]],
                #  freqsets=[[8, 30],[8, 13]],
                 chansets=[None],
                 parasets=None,
                 meta_classifier=LDA(solver="lsqr", shrinkage="auto"),
                 target_domain=None,
                 njobs=1,
                 out_type='prob',
                 cv_mode=False,
                 **kwargs,
                 ):
        
        self.fs_new = fs_new
        self.timesets = timesets
        self.freqsets = freqsets
        self.chansets = chansets
        self.parasets = parasets
        self.meta_classifier = meta_classifier
        self.target_domain = target_domain
        self.njobs = njobs
        self.out_type = out_type
        self.cv_mode = cv_mode
        self.kwargs = kwargs
        self.modelsets = []
        
        self.dpa_method = kwargs.get('dpa_method', 'RA')
        self.fee_method = kwargs.get('fee_method', 'MDM')
        self.fes_method = kwargs.get('fes_method', None)
        self.clf_method = kwargs.get('clf_method', None)
        self.end_method = kwargs.get('end_method', None)
        self.ete_method = kwargs.get('ete_method', None)
        
        if self.parasets is None:
            self.parasets = list(itertools.product(self.chansets, self.timesets, self.freqsets))
            
        for channels, time_window, freq_window in self.parasets:
            pre_processor = Pre_Processing(fs_old=None,
                                           fs_new=self.fs_new, 
                                           channels=channels, 
                                           start_time=time_window[0], 
                                           end_time=time_window[1], 
                                           lowcut=freq_window[0], 
                                           highcut=freq_window[1],
                                           **self.kwargs,
                                           )
            
            Model = TL_Classifier(#dpa_method=self.dpa_method, 
                                #   fee_method=self.fee_method, 
                                #   fes_method=self.fes_method,
                                #   clf_method=self.clf_method,
                                #   end_method=self.end_method, 
                                #   ete_method=self.ete_method, 
                                  pre_est=pre_processor.process, 
                                  target_domain=self.target_domain, 
                                  tl_mode='TL',
                                  **self.kwargs,
                                  )
            
            self.modelsets.append(deepcopy(Model))
    
    def evaluate_model(self, model, fold_data):
        """
        Evaluate a model on a fold of data.

        Parameters
        ----------
        model : object
            A model object that implements the `fit` and `score` methods.
        fold_data : dict
            A dictionary containing the training and validation data for the fold.

        Returns
        -------
        score : float
            The score of the model on the validation data.
        """
        model.fit(fold_data['train_data'], fold_data['train_label'])
        if self.out_type == 'prob':
            prob = model.predict_proba(fold_data['all_data'])
            pred = np.diff(prob, axis=1).flatten()
        else:
            pred = model.predict(fold_data['all_data']) 
        return pred
    
    def train_model(self, model, X, y_enc):
        """
        Train a model on the training data.

        Parameters
        ----------
        model : object
            A model object that implements the `fit` method.
        X : ndarray, shape (n_trials, n_channels, n_times)
            Training data.
        y_enc : ndarray, shape (n_trials,)
            Encoded labels for each trial.

        Returns
        -------
        model : object
            The trained model.
        """
        model.fit(X, y_enc)
        return model
                                                 
    def fit(self, X, y_enc):
        """
        Fit the model to the training data.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            Training data.
        y : ndarray, shape (n_trials,)
            Extended labels for each trial.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        _, y_dec, domains = decode_domains(X, y_enc)
        domains = list(domains)
        # Split the data into training and validation sets
        datas = []
        if self.target_domain in domains and not self.cv_mode:
            data = {}
            train_index = [i for i, x in enumerate(domains) if x != self.target_domain]
            data['train_data'] = copy(X[train_index])
            data['train_label'] = copy(y_enc[train_index])
            data['all_data'] = copy(X)
            data['all_label'] = copy(y_enc)
            datas.append(data)
        else:
            n_splits = 5
            n_repeats = 1
            data = {}
            rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
            for train_index, test_index in rskf.split(X, y_enc):
                data['train_data'] = copy(X[train_index])
                data['train_label'] = copy(y_enc[train_index])
                data['all_data'] = copy(X)
                data['all_label'] = copy(y_enc)
                datas.append(data)
            
        results = Parallel(n_jobs=self.njobs)(
            delayed(self.evaluate_model)(self.modelsets[model_idx], datas[fold_idx])
            for model_idx in range(len(self.modelsets))
            for fold_idx in range(len(datas))
            )
        results = np.array(results).reshape(len(self.modelsets), len(datas), -1)
        labels = np.array(y_dec)
        
        # 使用完整数据集训练模型
        self.modelsets = [deepcopy(self.modelsets[i]) for i in range(len(self.modelsets))]
        trained_models = Parallel(n_jobs=self.njobs)(
            delayed(self.train_model)(self.modelsets[model_idx], X, y_enc) 
            for model_idx in range(len(self.modelsets))
            )
        self.modelsets = trained_models
        
        num_models, num_folds, num_samples = results.shape
        # 初始化 meta 特征矩阵和标签向量
        X_meta = np.zeros((num_folds * num_samples, num_models))
        y_meta = np.zeros(num_folds * num_samples, dtype=int)
        # 填充 X_meta 和 y_meta
        for fold_idx in range(num_folds):
            for sample_idx in range(num_samples):
                # 在每一个折中，对每一个样本，将所有基础模型的预测结果作为特征
                X_meta[fold_idx * num_samples + sample_idx, :] = results[:, fold_idx, sample_idx]
                # 标签直接重复 num_folds 次
                y_meta[fold_idx * num_samples + sample_idx] = labels[sample_idx]
        
        # stacking
        self.meta_classifier.fit(X_meta, y_meta)

        return self

    def predict(self, X):
        """Predict the labels for the test data.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            Test data.

        Returns
        -------
        y_pred : ndarray, shape (n_trials,)
            Predicted labels for each trial.
        """
        if self.out_type == 'prob':
            results = Parallel(n_jobs=self.njobs)(
                delayed(self.modelsets[model_idx].predict_proba)(X) 
                for model_idx in range(len(self.modelsets))
                )
            for i in range(len(results)):
                results[i] = np.diff(results[i], axis=1).flatten()
            
        else: 
            results = Parallel(n_jobs=self.njobs)(
                delayed(self.modelsets[model_idx].predict)(X) 
                for model_idx in range(len(self.modelsets)))
        
        results = np.array(results).reshape(len(self.modelsets), -1)  # (num_models, num_samples)   
        
        # 计算 meta 特征
        X_meta = results.T  # (num_samples, num_models)
        
        # 使用 meta-classifier 预测标签
        y_pred = self.meta_classifier.predict(X_meta)

        return y_pred
    
    def score(self, X, y_enc):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)           
            Test data.
        y : ndarray, shape (n_trials,)
            Extended labels for each trial.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """ 
        y_pred = self.predict(X)
        _, y_dec, _ = decode_domains(X, y_enc)
        return np.mean(y_pred == y_dec)
        
    

if __name__ == '__main__':
    from sklearn.model_selection import StratifiedShuffleSplit
    from loaddata import Dataset_Left_Right_MI
    from transfer_learning import TLSplitter, encode_datasets
    
    dataset_name = 'Pan2023'
    fs = 250
    datapath = r'E:\工作进展\小论文2023会议\数据处理python\datasets'
    # dataset = Dataset_MI(dataset_name,fs=fs,fmin=8,fmax=30,tmin=0,tmax=4,path=datapath)
    dataset = Dataset_Left_Right_MI(dataset_name,fs=fs,fmin=1,fmax=40,tmin=0,tmax=4,path=datapath)
    subjects = dataset.subject_list[:3]

    datas, labels = [], []
    for sub in subjects:
        data, label, _ = dataset.get_data([sub])
        datas.append(data)
        labels.append(label)

    # 设置交叉验证
    n_splits=5
    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=2024) #可以控制训练集的数量

    for sub in subjects:

            print(f"Subject {sub}...")
            target_domain = f'S{sub}'
            
            X, y_enc, domain = encode_datasets(datas, labels)
            print(f"data shape: {X.shape}, label shape: {y_enc.shape}")
            print(f"All Domain: {domain}")
            print(f'target_domain: {target_domain}')
            
            Model = EL_Classifier(target_domain=target_domain, njobs=-1)
            
            # tl_cv = TLSplitter(target_domain=target_domain, cv=None, no_calibration=True)
            tl_cv = TLSplitter(target_domain=target_domain, cv=cv, no_calibration=False)
            
            acc = []
            for train, test in tl_cv.split(X, y_enc):
                X_train, y_train = X[train], y_enc[train]
                X_test, y_test = X[test], y_enc[test]
                Model.fit(X_train, y_train)
                score = Model.score(X_test, y_test)
                acc.append(score)

            print(f"Subject {sub} score: {score.mean()} +/- {score.std()}")