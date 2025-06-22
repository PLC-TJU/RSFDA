"""
KL-Divergence Embedded Distribution Alignment (KEDA)
Author: LC.Pan <panlincong@tju.edu.cn.com>
Date: 2024/3/14
License: All rights reserved
"""

import numpy as np
from itertools import combinations
from scipy.stats import entropy
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KernelDensity
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.utils import mean_covariance
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.tangentspace import tangent_space
from pyriemann.utils.utils import check_weights
from . import decode_domains
import warnings
warnings.filterwarnings("ignore")

def estimate_distribution(X, bandwidth=0.1, num_points=200):
    num_points = max(num_points, X.shape[0])
    
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    # 计算核密度估计
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
    X_d = np.linspace(np.min(X), np.max(X), num_points)[:, np.newaxis]
    log_density = kde.score_samples(X_d)
    return np.exp(log_density)

def kl_divergence(p, q):
    # Normalize distributions to ensure they sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)
    # avoid division by zero
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return entropy(p, q)

def supervised_kl_div_kernel(Xs, ys, Xt, yt, d=10, bandwidth=1.0, solver = 'L-BFGS-B', maxiter=100, tol=1e-6, collect_obj_values=False):
    """
    Calculate the transformation matrix W that minimizes the KL divergence between
    the joint distributions of the source and target domains, as well as the marginal 
    and conditional probability distribution discrepancies between the source and the
    target domains for each class. 
    
    Args:
        Xs: Source domain data (num_samples, num_feature)
        ys: Source domain labels (num_samples)
        Xt: Target domain data (num_samples, num_feature)
        yt: Target domain labels (num_samples)
        d: Reduced feature dimension
        bandwidth: Kernel bandwidth for density estimation
        solver: Optimization solver for the KL divergence
        maxiter: Maximum number of iterations for the optimization solver
        collect_obj_values: Whether to collect the objective values during the optimization process
        
    Returns:
        W: Transformation matrix of size (num_feature, d)
        
        only if collect_obj_values is True:
        obj_values: List of objective values during the optimization process
        W_values: List of W values during the optimization process
    """
    nc = Xs.shape[1]
    d = min(d, nc)  # Limit d to the number of features in Xs
    # Step 1: Define the objective function to minimize
    
    
    def objective(W):
        W = W.reshape((nc, d))  # Reshape W into a matrix of size (feature_dim, d)
        Xs_transformed = np.dot(Xs, W)
        Xt_transformed = np.dot(Xt, W)
        num_points = max(200, Xs_transformed.shape[0], Xt_transformed.shape[0])
        kl_divs1, kl_divs2 = [], []
        
        # Compute KL divergence between source and target distributions
        for i in range(d):
            Xs_cls = Xs_transformed[:, i].reshape(-1, 1)
            Xt_cls = Xt_transformed[:, i].reshape(-1, 1)
            
            Ps = estimate_distribution(Xs_cls, bandwidth=bandwidth, num_points=num_points)
            Pt = estimate_distribution(Xt_cls, bandwidth=bandwidth, num_points=num_points)
            
            kl_divs1.append(kl_divergence(Ps, Pt))
        
        # Compute KL divergence between source and target distributions for each class
        classes = np.unique(ys)
        for cls in classes:
            for i in range(d):
                Xs_cls = Xs_transformed[ys == cls, i].reshape(-1, 1)
                Xt_cls = Xt_transformed[yt == cls, i].reshape(-1, 1)
                
                Ps = estimate_distribution(Xs_cls, bandwidth=bandwidth, num_points=num_points)
                Pt = estimate_distribution(Xt_cls, bandwidth=bandwidth, num_points=num_points)
                
                kl_divs1.append(kl_divergence(Ps, Pt))
                
        for i in range(d):
            for cls1, cls2 in combinations(classes, 2):
                Xs_cls1 = Xs_transformed[ys == cls1, i].reshape(-1, 1)
                Xs_cls2 = Xs_transformed[ys == cls2, i].reshape(-1, 1)
                
                Xt_cls1 = Xt_transformed[yt == cls1, i].reshape(-1, 1)
                Xt_cls2 = Xt_transformed[yt == cls2, i].reshape(-1, 1)
                
                X_cls1 = np.vstack((Xs_cls1, Xt_cls1))
                X_cls2 = np.vstack((Xs_cls2, Xt_cls2))
                
                Ps1 = estimate_distribution(X_cls1, bandwidth=bandwidth, num_points=num_points)
                Ps2 = estimate_distribution(X_cls2, bandwidth=bandwidth, num_points=num_points)
                
                kl_divs2.append(kl_divergence(Ps1, Ps2))      
        
        return np.sum(kl_divs1)/np.sum(kl_divs2)
    
    def objective2(W):
        W = W.reshape((nc, d))  # Reshape W into a matrix of size (feature_dim, d)
        Xs_transformed = np.dot(Xs, W)
        Xt_transformed = np.dot(Xt, W)
        num_points = max(200, Xs_transformed.shape[0], Xt_transformed.shape[0])
        kl_divs1= []
        
        # Compute KL divergence between source and target distributions
        for i in range(d):
            Xs_cls = Xs_transformed[:, i].reshape(-1, 1)
            Xt_cls = Xt_transformed[:, i].reshape(-1, 1)
            
            Ps = estimate_distribution(Xs_cls, bandwidth=bandwidth, num_points=num_points)
            Pt = estimate_distribution(Xt_cls, bandwidth=bandwidth, num_points=num_points)
            
            kl_divs1.append(kl_divergence(Ps, Pt))
        
        # Compute KL divergence between source and target distributions for each class
        classes = np.unique(ys)
        for cls in classes:
            for i in range(d):
                Xs_cls = Xs_transformed[ys == cls, i].reshape(-1, 1)
                Xt_cls = Xt_transformed[yt == cls, i].reshape(-1, 1)
                
                Ps = estimate_distribution(Xs_cls, bandwidth=bandwidth, num_points=num_points)
                Pt = estimate_distribution(Xt_cls, bandwidth=bandwidth, num_points=num_points)
                
                kl_divs1.append(kl_divergence(Ps, Pt))    
        
        return np.sum(kl_divs1)
    
    # Define the callback function
    W, obj_values = [], []
    def callback(xk,state=None):
        # obj_values.append(objFunc(xk))
        W.append(xk.reshape((nc, d)))
        obj_values.append(-state.fun) if solver == 'trust-constr' else obj_values.append(objective(xk))
    
    # Step 2: Initialize W randomly
    # W_init = np.random.randn(nc, d)
    W_init = np.ones((nc, d))
    
    # Step 3: Optimize W to minimize the KL divergence
    result = minimize(objective, W_init.flatten(), method=solver, tol=tol,
                      options={'maxiter': maxiter, 'disp': True,'verbose': 1}, 
                      callback=callback if collect_obj_values else None)
    
    W_optimized = result.x.reshape((nc, d))
    
    return W_optimized, (obj_values if collect_obj_values else None), (W if collect_obj_values else None)



def unsupervised_kl_div_kernel(Xs, ys, Xt, yt=None, d=10, bandwidth=1, solver = 'L-BFGS-B', maxiter=100, tol=1e-6, collect_obj_values=False):
    """
    Calculate the transformation matrix W that minimizes the KL divergence between
    the joint distributions of the source and target domains, as well as the marginal
    and conditional probability distribution discrepancies between the source and the
    target domains for each class. 
    
    Args:
        Xs: Source domain data (num_samples, num_feature)
        ys: Source domain labels (num_samples)
        Xt: Target domain data (num_samples, num_feature)
        yt: Target domain labels (num_samples), not used in this function
        d: Reduced feature dimension
        bandwidth: Kernel bandwidth for density estimation  
        solver: Optimization solver for the KL divergence
        maxiter: Maximum number of iterations for the optimization solver
        collect_obj_values: Whether to collect the objective values during the optimization process
        
    Returns:
        W: Transformation matrix of size (num_feature, d)        
        
        only if collect_obj_values is True:
        obj_values: List of objective values during the optimization process
        W_values: List of W values during the optimization process
    """
    nc = Xs.shape[1]
    d = min(d, nc)  # Limit d to the number of features in Xs
    # Step 1: Define the objective function to minimize
    def objective(W):
        W = W.reshape((nc, d))  # Reshape W into a matrix of size (feature_dim, d)
        Xs_transformed = np.dot(Xs, W)
        Xt_transformed = np.dot(Xt, W)
        num_points = max(200, Xs_transformed.shape[0], Xt_transformed.shape[0])
        kl_divs1, kl_divs2 = [], []
        
        # Compute KL divergence between source and target distributions
        for i in range(d):
            Xs_cls = Xs_transformed[:, i].reshape(-1, 1)
            Xt_cls = Xt_transformed[:, i].reshape(-1, 1)
            
            Ps = estimate_distribution(Xs_cls, bandwidth=bandwidth, num_points=num_points)
            Pt = estimate_distribution(Xt_cls, bandwidth=bandwidth, num_points=num_points)
            
            kl_divs1.append(kl_divergence(Ps, Pt))
        
        # Compute KL divergence between each class in the source domain
        classes = np.unique(ys)
        for i in range(d):
            for cls1, cls2 in combinations(classes, 2):
                Xs_cls1 = Xs_transformed[ys == cls1, i].reshape(-1, 1)
                Xs_cls2 = Xs_transformed[ys == cls2, i].reshape(-1, 1)
                
                Ps1 = estimate_distribution(Xs_cls1, bandwidth=bandwidth, num_points=num_points)
                Ps2 = estimate_distribution(Xs_cls2, bandwidth=bandwidth, num_points=num_points)
                
                kl_divs2.append(kl_divergence(Ps1, Ps2))
        
        return np.sum(kl_divs1)/np.sum(kl_divs2)
    
    # Define the callback function
    W, obj_values = [], []
    def callback(xk,state=None):
        # obj_values.append(objFunc(xk))
        W.append(xk.reshape((nc, d)))
        obj_values.append(-state.fun) if solver == 'trust-constr' else obj_values.append(objective(xk))
    
    # Step 2: Initialize W randomly
    # W_init = np.random.randn(nc, d)
    W_init = np.ones((nc, d))
    
    # Step 3: Optimize W to minimize the KL divergence
    result = minimize(objective, W_init.flatten(), method=solver, tol=tol,
                      options={'maxiter': maxiter, 'disp': True, 'verbose': 1}, 
                      callback=callback if collect_obj_values else None)
    
    W_optimized = result.x.reshape((nc, d))
    
    return W_optimized, (obj_values if collect_obj_values else None), (W if collect_obj_values else None)

class KEDA(BaseEstimator, ClassifierMixin): #基于特征空间的无监督的迁移学习方法，需要目标域数据，但是不需要标签
    """
    KL-Divergence Embedded Distribution Alignment (KEDA).
    author: LC.Pan
    Created on: 2024-03-22
    """

    def __init__(
        self,
        target_domain,
        subspace_dim: int = 10,
        bandwidth: float = 0.1,
        max_iter: int = 20,
        metric="riemann",
        selector=None,
        estimator=LDA(solver="lsqr", shrinkage="auto"),
    ):
        self.target_domain = target_domain
        self.subspace_dim = subspace_dim
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.metric = metric
        self.selector = selector
        self.estimator = estimator

    def get_feature(self, X, sample_weight=None, metric='riemann'):
        # Covariance Matrix Centroid Alignment
        M = mean_covariance(X, metric=metric, sample_weight=sample_weight)
        iM12 = invsqrtm(M)
        C = iM12 @ X @ iM12.T
        # Tangent Space Feature Extraction
        feature = tangent_space(C, np.eye(M.shape[0]), metric=metric)
        return feature
    
    def transform(self, X):
        """Obtain target domain features after KEDA transformation.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_channels).

        Returns
        -------
        target_features: ndarray
            target domain features, shape(n_trials, n_features).

        """
        feature = self.get_feature(X, metric=self.metric)
        
        if self.selector is not None:
            feature = self.selector.transform(feature)
        
        return feature @ self.W_

    def fit_transform_(self, X, y_enc, sample_weight=None):
        """Obtain source and target domain features after KEDA transformation.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_channels).
        y_enc: ndarray
            Label, shape(n_trials,).
        sample_weight: ndarray
            Sample weight, shape(n_trials,).

        Returns
        -------
        feature: ndarray
            source and target domain features, shape(n_trials, n_features).

        """
        X, y, domains = decode_domains(X, y_enc)
        sample_weight = check_weights(sample_weight, X.shape[0])
        
        Xs = X[domains != self.target_domain]
        ys = y[domains != self.target_domain]
        Xt = X[domains == self.target_domain]
        yt = y[domains == self.target_domain]
        
        featureXs = self.get_feature(
            Xs, 
            sample_weight=sample_weight[domains != self.target_domain], 
            metric=self.metric
            )
        featureXt = self.get_feature(
            Xt, 
            metric=self.metric
            )
        
        # 特征选择
        if self.selector is not None:
            featureXs_selected = self.selector.fit_transform(featureXs, ys)
            featureXt_selected = self.selector.transform(featureXt)
        else:
            featureXs_selected = featureXs
            featureXt_selected = featureXt
        
        # # 无监督迁移学习
        # self.W_, _, _ = unsupervised_kl_div_kernel(
        #     featuresXs_selected, ys, featuresXt_selected, yt, 
        #     d=self.subspace_dim, bandwidth=1)
        
        # 监督迁移学习
        self.W_, _, _ = supervised_kl_div_kernel(
            featureXs_selected, ys, featureXt_selected, yt, 
            d=self.subspace_dim, bandwidth=1, 
            maxiter=self.max_iter, tol=1e-3)

        source_features = featureXs_selected @ self.W_
        target_features = featureXt_selected @ self.W_
        feature = np.zeros((len(domains), source_features.shape[-1]))
        feature[domains != self.target_domain] = source_features
        feature[domains == self.target_domain] = target_features
        
        return feature, y # 返回源域特征和目标域特征和标签（有监督）
        # return source_features, ys # 只返回源域特征和标签(无监督)
    
    def fit(self, X, y_enc, sample_weight=None):
        """Fit the model with X and y.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_channels).
        y_enc: ndarray
            Label, shape(n_trials,).
        sample_weight: ndarray
            Sample weight, shape(n_trials,).

        Returns
        -------
        self: object
            Returns the instance itself. 

        """
        features, y = self.fit_transform_(X, y_enc, sample_weight)
        self.classes_ = np.unique(y)
        self.model_ = self.estimator.fit(features, y)
        return self
    
    def predict(self, X):
        """Predict the target domain labels.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_channels).

        Returns
        -------
        y_pred: ndarray
            Predicted target domain labels, shape(n_trials,).

        """

        y_pred = self.model_.predict(self.transform(X))

        return y_pred
    
    def score(self, X, y_enc):
        """Calculate the accuracy of the model.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_channels).
        y_enc: ndarray
            Label, shape(n_trials,).

        Returns
        -------
        score: float
            Accuracy of the model.

        """
        _, y_true, _ = decode_domains(X, y_enc)
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)

# Example usage
if __name__ == '__main__':
    # Xs = np.random.randn(100, 20)  # Source domain data (100 samples, 20 features)
    # ys = np.random.randint(0, 2, 100)  # Source domain labels (binary classification)
    # Xt = np.random.randn(80, 20)   # Target domain data (80 samples, 20 features)
    # yt = np.random.randint(0, 2, 80)  # Target domain labels (binary classification)
    # d = 10  # Reduced feature dimension

    # W = unsupervised_kl_div_kernel(Xs, ys, Xt, yt, d)
    # print("Optimized Transformation Matrix W:", W)
    
    import sys
    from sklearn.pipeline import make_pipeline
    from pyriemann.estimation import Covariances as Cov
    folder_path = 'NeuroDecKit'
    if folder_path not in sys.path:
        sys.path.append(folder_path)
    
    from loaddata import Dataset_Left_Right_MI
    from machine_learning import RiemannCSP as CSP
    dataset_name = 'Pan2023'
    fs = 250
    datapath = r'E:\工作进展\小论文2023会议\数据处理python\datasets'
    # dataset = Dataset_MI(dataset_name,fs=fs,fmin=8,fmax=30,tmin=0,tmax=4,path=datapath)
    dataset = Dataset_Left_Right_MI(dataset_name,fs=fs,fmin=1,fmax=40,tmin=0,tmax=4,path=datapath)

    Xs, ys, _ = dataset.get_data([1])
    Xt, yt, _ = dataset.get_data([2])
    d = 10  # Reduced feature dimension
    
    clf = make_pipeline(Cov(), CSP(nfilter=20))
    Xs_csp = clf.fit_transform(Xs, ys)
    Xt_csp = clf.transform(Xt)

    # W = supervised_kl_div_kernel(Xs_csp, ys, Xt_csp, yt, d)
    W, obj_values, _ = unsupervised_kl_div_kernel(Xs_csp, ys, Xt_csp, yt, d, maxiter=1000, collect_obj_values=True)
    print("Optimized Transformation Matrix W:", W)
    
    # 绘制收敛曲线
    import matplotlib.pyplot as plt
    plt.plot(obj_values)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.show()
