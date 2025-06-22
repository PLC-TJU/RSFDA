"""
modiified from https://github.com/EEGdecoding/Code-SBLEST/blob/main/SBLEST_model.py <201710102248@mail.scut.edu.cn>
author: LC, Pan <panlincong@tju.edu.cn>
date: 2024-03-11
"""

import torch
import warnings
import numpy as np
from torch import reshape, norm, zeros, eye, float64, mm, inverse, log, det
from torch import linalg, diag, log
from torch import zeros, float64, mm, DoubleTensor
from sklearn.base import BaseEstimator, ClassifierMixin

warnings.filterwarnings('ignore')

def sblest_kernel(X, Y, epoch=5000, tol=1e-5, epoch_print=None, device='cpu'):
    """
    SBLEST     : Sparse Bayesina Learning for End-to-end Spatio-Temporal-filtering-based 
                 single-trial EEG classification [1]

    --- Parameters ---
    X          : M trials of Feature. [M, (K*C)^2].
    Y          : True label vector. [M, 1].
    epoch      : Number of iterations for optimization.
    tol        : Tolerance for convergence.
    epoch_print: Number of iterations for printing loss.
    device     : 'cpu' or 'cuda'

    --- Returns ---
    W          : Estimated low-rank weight matrix. [K*C, K*C].
    alpha      : Classifier weights. [L, 1].
    V          : Spatio-temporal filter matrix. [K*C, L].
                 Each column of V represents a spatio-temporal filter.

    Reference:
    [1] W. Wang et al. (2023), "Sparse Bayesian Learning for End-to-End EEG Decoding," in IEEE 
    Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 12, pp. 15632-15649
    
    """
    X = torch.tensor(X, dtype=torch.float64, device=device)
    Y = torch.tensor(Y, dtype=torch.float64, device=device)
    
    # Check properties of R
    M, D_R = X.shape  # M: number of samples; D_R: dimension of vec(R_m)
    KC = round(np.sqrt(D_R))
    Loss_old = 1e12
    threshold = 0.05

    assert D_R == KC ** 2, "ERROR: Columns of A do not align with square matrix"

    # Check if R is symmetric
    for j in range(M):
        row_cov = reshape(X[j, :], (KC, KC))
        row_cov = (row_cov + row_cov.T) / 2
        assert norm(row_cov - row_cov.T) < 1e-4, "ERROR: Measurement row does not form symmetric matrix"

    # Initializations
    U = zeros(KC, KC, dtype=float64).to(device)  # estimated low-rank matrix W initialized to be Zeros
    Psi = eye(KC, dtype=float64).to(device)  # covariance matrix of Gaussian prior distribution is initialized to be unit diagonal matrix
    lambda_noise = 1.  # variance of the additive noise set to 1

    # Optimization loop
    for i in range(epoch):

        # update B,Sigma_y,u
        RPR = zeros(M, M, dtype=float64).to(device)
        B = zeros(KC ** 2, M, dtype=float64).to(device)
        for j in range(KC):
            start = j * KC
            stop = start + KC
            Temp = mm(Psi, X[:, start:stop].T)
            B[start:stop, :] = Temp
            RPR = RPR + mm(X[:, start:stop], Temp)
        Sigma_y = RPR + lambda_noise * eye(M, dtype=float64).to(device)
        uc = mm(mm(B, inverse(Sigma_y)), Y)  # maximum a posterior estimation of uc
        Uc = reshape(uc, (KC, KC))
        U = (Uc + Uc.T) / 2
        u = U.T.flatten()  # vec operation (Torch)

        # Update Phi (dual variable of Psi)
        Phi = []
        SR = mm(inverse(Sigma_y), X)
        for j in range(KC):
            start = j * KC
            stop = start + KC
            Phi_temp = Psi - Psi @ X[:, start:stop].T @ SR[:, start:stop] @ Psi
            Phi.append(Phi_temp)

        # Update Psi
        PHI = 0
        UU = 0
        for j in range(KC):
            PHI = PHI + Phi[j]
            UU = UU + U[:, j].reshape(-1, 1) @ U[:, j].reshape(-1, 1).T
        # UU = U @ U.T
        Psi = ((UU + UU.T) / 2 + (PHI + PHI.T) / 2) / KC    # make sure Psi is symmetric

        # Update theta (dual variable of lambda)
        theta = 0
        for j in range(KC):
            start = j * KC
            stop = start + KC
            theta = theta + (Phi[j] @ X[:, start:stop].T @ X[:, start:stop]).trace()

        # Update lambda
        lambda_noise = ((norm(Y - (X @ u).reshape(-1, 1), p=2) ** 2).sum() + theta) / M

        # Convergence check
        Loss = Y.T @ inverse(Sigma_y) @ Y + log(det(Sigma_y))
        delta_loss = abs(Loss_old - Loss.cpu().numpy()) / abs(Loss_old)
        if delta_loss < tol:
            # print('EXIT: Change in loss below tolerance threshold')
            break
        Loss_old = Loss.cpu().numpy()
        if epoch_print != 0 and epoch_print is not None:
            if (i+1) % epoch_print == 0:
                print('Iterations: ', str(i+1), 
                      '  lambda: ', str(lambda_noise.cpu().numpy()), 
                      '  Loss: ', float(Loss.cpu().numpy()), 
                      '  Delta_Loss: ', float(delta_loss)
                      )

    # Eigen-decomposition of W
    W = U
    D, V_all = torch.linalg.eig(W)
    D, V_all = D.double().cpu().numpy(), V_all.double().cpu().numpy()
    idx = D.argsort()
    D = D[idx]
    V_all = V_all[:, idx]       # each column of V represents a spatio-temporal filter
    alpha_all = D

    # Determine spatio-temporal filters V and classifier weights alpha
    d = np.abs(alpha_all)
    d_max = np.max(d)
    w_norm = d / d_max      # normalize eigenvalues of W by the maximum eigenvalue
    index = np.where(w_norm > threshold)[0]    # indices of selected V according to a pre-defined threshold,.e.g., 0.05
    V = V_all[:, index]
    alpha = alpha_all[index]
    
    # # Return results as numpy arrays in CPU memory
    W = W.double().cpu().numpy()

    return W, alpha, V


def matrix_operations(A):
    """Calculate the -1/2 power of matrix A"""

    V, Q = linalg.eig(A)
    V_inverse = diag(V ** (-0.5))
    A_inverse = mm(mm(Q, V_inverse), linalg.inv(Q))

    return A_inverse.double()


def logm(A):
    """Calculate the matrix logarithm of matrix A"""

    V, Q = linalg.eig(A)  # V为特征值,Q为特征向量
    V_log = diag(log(V))
    A_logm = mm(mm(Q, V_log), linalg.inv(Q))

    return A_logm.double()


def computer_acc(predict_Y, Y_test):
    """Compute classification accuracy for test set"""

    predict_Y = predict_Y.cpu().numpy()
    Y_test = torch.squeeze(Y_test).cpu().numpy()
    total_num = len(predict_Y)
    error_num = 0

    # Compute classification accuracy for test set
    Y_predict = np.zeros(total_num)
    for i in range(total_num):
        if predict_Y[i] > 0:
            Y_predict[i] = 1
        else:
            Y_predict[i] = -1

    # Compute classification accuracy
    for i in range(total_num):
        if Y_predict[i] != Y_test[i]:
            error_num = error_num + 1

    accuracy = (total_num - error_num) / total_num
    return accuracy


def enhanced_cov(X, K, tau, Wh=None, device='cpu'):
    """
    Compute enhanced covariance matrices

    --- Parameters ---
    X         : M trials of C (channel) x T (time) EEG signals. [C, T, M].
    K         : Order of FIR filter
    tau       : Time delay parameter
    Wh        : Whitening matrix for enhancing covariance matrices.
                In training mode(train=1), Wh will be initialized as following python_code.
                In testing mode(train=0), Wh will receive the concrete value.

    --- Returns ---
    R         : Enhanced covariance matrices. [M, (K*C)^2]
    Wh        : Whitening matrix. [(K*C)^2, (K*C)^2].
    """
    # Ensure tau is a list if it's a scalar
    if isinstance(tau, int):
        tau = [tau]
    
    # Ensure X is a tensor
    X = torch.tensor(X, dtype=torch.float64, device=device)
    
    # Initialization, [KC, KC]: dimension of augmented covariance matrix
    X_order_k = None
    C, T, M = X.shape
    Cov = []
    Sig_Cov = zeros(K * C, K * C).to(device)

    for m in range(M):
        X_m = X[:, :, m]
        X_m_hat = DoubleTensor().to(device)

        # Generate augmented EEG data
        for k in range(K):
            n_delay = tau[k]
            if n_delay == 0:
                X_order_k = X_m.clone()
            else:
                X_order_k[:, 0:n_delay] = 0
                X_order_k[:, n_delay:T] = X_m[:, 0:T - n_delay].clone()
            X_m_hat = torch.cat((X_m_hat, X_order_k), 0)

        # Compute covariance matrices
        R_m = mm(X_m_hat, X_m_hat.T)

        # Trace normalization
        R_m = R_m / R_m.trace()
        Cov.append(R_m)

        Sig_Cov = Sig_Cov + R_m

    # Compute Whitening matrix (Rp).
    if Wh is None:
        Wh = Sig_Cov / M

    # Whitening, logarithm transform, and Vectorization
    Cov_whiten = zeros(M, K * C, K * C, dtype=float64).to(device)
    R = zeros(M, K * C * K * C, dtype=float64).to(device)

    for m in range(M):
        # progress_bar(m, M)

        # whitening
        Wh_inverse = matrix_operations(Wh)  # Rp^(-1/2)
        temp_cov = Wh_inverse @ Cov[m] @ Wh_inverse
        Cov_whiten[m, :, :] = (temp_cov + temp_cov.T) / 2
        R_m = logm(Cov_whiten[m, :, :])
        R_m = R_m.reshape(R_m.numel())  # column-wise vectorization
        R[m, :] = R_m
    
    # Return results as numpy arrays in CPU memory
    R = R.cpu().numpy()

    return R, Wh

class SBLEST(BaseEstimator, ClassifierMixin):
    def __init__(self, K=2, tau=[0, 1], epoch=5000, epoch_print=None, device='cpu'):
        self.K = K
        self.tau = tau
        self.epoch = epoch
        self.epoch_print = epoch_print
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.classes_ = None
        self.W = None
        self.alpha = None
        self.V = None
        self.Wh = None
    
    def fit(self, X, Y):
        X = X.transpose(1, 2, 0)    # [C, T, M] -> [T, M, C]
        self.classes_ = np.unique(Y)     
        Y = np.where(Y == np.unique(Y)[0], 1, -1)# Convert labels to -1 and 1
        Y = np.array(Y).reshape(-1, 1)
        R_train, self.Wh = enhanced_cov(X, self.K, self.tau, device=self.device)
        self.W, self.alpha, self.V = sblest_kernel(
            R_train, Y, epoch=self.epoch, epoch_print=self.epoch_print, device=self.device)
        return self 

    def transform(self, X):
        X = X.transpose(1, 2, 0)
        R_test, _ = enhanced_cov(X, self.K, self.tau, self.Wh, device=self.device)
        return R_test

    def predict(self, X):
        if self.W is None:
            raise ValueError("Model is not trained yet. Please call 'fit' with \
                             appropriate arguments before calling 'predict'.")
        R_test = self.transform(X)
        vec_W = self.W.T.flatten()
        predict_Y = R_test @ vec_W
        return np.where(predict_Y > 0, self.classes_[0], self.classes_[1])
    
    def decision_function(self, X):
        if self.W is None:
            raise ValueError("Model is not trained yet. Please call 'fit' with \
                             appropriate arguments before calling 'decision_function'.")
        R_test = self.transform(X)
        vec_W = self.W.T.flatten()
        predict_Y = R_test @ vec_W
        return -predict_Y
    
    