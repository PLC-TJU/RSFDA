# Off-line modeling program/ Transfer learning
#
# Authors: Corey Lin <panlincong@tju.edu.cn.com>
# Date: 2023/07/08
# License: BSD 3-Clause License

import numpy as np
from numpy import ndarray
from copy import deepcopy
from scipy.signal import  iircomb, butter, filtfilt
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.estimation import Covariances
from . import RiemannCSP as CSP
from . import MDM, FgMDM, TS, TRCA, DCPM, MIBIF
from ..transfer_learning import TLClassifier
from pyriemann.transfer import encode_domains, TLDummy, TLCenter, TLStretch, TLRotate

import warnings
warnings.filterwarnings('ignore')

def Modeling_Framework(Xt, yt, Xs = None, ys = None, method_flag = [[0,0],[0,0],[0,1]]):
    """
    这个函数可以建立一系列算法（预对齐+特征提取+分类器）的计算流程组合。

    参数:
    - Xt: numpy 数组，表示目标域数据，形状为 (n_trials, n_channels, n_samples)。
    - yt: numpy 数组，表示目标域标签，形状为 (n_trials,)。
    - Xs: 列表，包含多个 numpy 数组，表示源域数据。默认值为 None。
    - ys: 列表，包含多个 numpy 数组，表示源域标签。默认值为 None。
    - method_flag: 列表，表示算法组合的索引。默认值为 [[0],[0],[0]]。

        method_flag[0]: 预对齐方法的索引。
        0 : 不使用迁移学习
        1 : TLDummy
        2 : EA
        3 : RA
        4 : RPA

        method_flag[1]: 特征提取方法的索引。
        0 : CSP
        1 : MDRM
        2 : FgMDRM
        3 : TS
        4 : DCPM
        5 : TRCA

        method_flag[2]: 分类方法的索引。
        0 : SVM
        1 : LDA

    返回值:
    - Models: 列表，包含多个模型对象。
    - Infos: 列表，包含与每个模型对应的相关信息。

    示例:
    ```python
    # 我们可以使用 numpy 库来生成模拟数据。
    import numpy as np

    n_trials = 100
    n_channels = 32
    n_samples = 256

    # 目标域数据和标签
    Xt = np.random.randn(n_trials, n_channels, n_samples)
    yt = np.random.randint(0, 2, size=n_trials)

    # 源域数据和标签
    Xs = [np.random.randn(n_trials, n_channels, n_samples) for _ in range(3)]
    ys = [np.random.randint(0, 2, size=n_trials) for _ in range(3)]

    # 我们可以使用 Modeling_Framework 函数来实现这一目标。
    # 在这个示例中，我们将使用索引为 0 的预对齐方法，
    # 索引为 1 和 2 的特征提取方法，
    # 以及索引为 0 和 1 的分类方法。

    Models, Infos = Modeling_Framework(Xt, yt, Xs, ys, method_flag=[[0, 0], [1, 2], [0, 1]])

    # 现在 Models 列表中包含了多个模型对象，
    # 每个对象都对应于一种不同的算法组合。

    # 我们可以使用这些模型来预测新数据。
    for Model in Models:
        y_pred = Model.predict(X_new)

    # 此外 Infos 列表中包含了与每个模型对应的相关信息，
    # 包括它使用的预处理方法、特征提取方法和分类方法，
    # 以及在训练过程中是否发生了异常。
    for Info in Infos:
        print(Info)
    ```
    """
    pa_methods = ['noTL','TLDummy','EA','RA','RPA']
    fee_methods = ['CSP','MDRM','FgMDRM','TS','DCPM','TRCA']
    clf_methods = ['SVM','LDA']
        
    ## with TL
    if any(x != 0 for x in method_flag[0]):
        DataAll = np.concatenate([np.concatenate(Xs, axis=0), Xt], axis=0)# 拼接源域数据和目标域数据
        LabelAll = np.concatenate(ys + [yt])# 拼接源域标签和目标域标签
        domain = np.concatenate([i * np.ones(len(y), dtype=int) for i, y in enumerate(ys + [yt])])# 创建 domain 数组

        DataAll, LabelAll = encode_domains(DataAll, LabelAll, domain)

        n = len(Xs) # 不同域数据集数量
        source_names = ['{}'.format(i) for i in range(n)]# 为每个源域数据集命名
        domain_weight = {name: 1 for name in source_names}# 创建 domain_weight 字典，设置各源域的权重为1
        target_domain = str(domain[-1])# 定义目标域
        domain_weight.update({target_domain:3})# 设置目标域的权重为3
        # 归一化权重
        total = sum(domain_weight.values())
        for key in domain_weight:
            domain_weight[key] /= total
    
    ## without TL
    if 0 in method_flag[0]:
        DataAll0 = np.float32(Xt)
        LabelAll0 = np.array(yt)
        domain = np.zeros(LabelAll0.shape, dtype=int)
        target_domain = domain[-1]  
            
    #1# Data pre-alignment
    prealignments = {
        0: Pipeline(steps=[]),
        1: make_pipeline(
            Covariances(estimator='lwf'),
            TLDummy(),
        ),
        2: make_pipeline(
            Covariances(estimator='lwf'),
            TLCenter(target_domain=target_domain, metric='euclid'),
        ),
        3: make_pipeline(
            Covariances(estimator='lwf'),
            TLCenter(target_domain=target_domain, metric='riemann'),
        ),
        4: make_pipeline(
            Covariances(estimator='lwf'),
            TLCenter(target_domain=target_domain),
            TLStretch(
                target_domain=target_domain,
                final_dispersion=1,
                centered_data=True,
            ),
            TLRotate(target_domain=target_domain, metric='euclid'),
        )
    }

    if not all(i in prealignments for i in method_flag[0]):
        raise ValueError("Invalid value for method_flag[0]: Index of data pre-alignment methods.")
  
    #2# Feature extraction
    transformers = {
        0: [CSP(nfilter=6)],
        1: [MDM()],
        2: [FgMDM()],
        3: [TS(), MIBIF()],
        4: [DCPM()],
        5: [TRCA()]
    }

    if not all(i in transformers for i in method_flag[1]):
        raise ValueError("Invalid value for method_flag[1]: Index of feature extraction methods.")
    
    #3# Classification   
    classifiers = {
        0: [SVC()],
        1: [LDA(solver='eigen', shrinkage='auto')]
    }

    if not all(i in classifiers for i in method_flag[2]):
        raise ValueError("Invalid value for method_flag[2]: Index of classification methods.")
        
    # 将method_flag按照子元素的第一个子元素的值不同来划分为多份   
    method_type = {}
    for i, item in enumerate(method_flag[0]):
        key = item
        if key not in method_type:
            method_type[key] = []
        temp = []
        for j in range(len(method_flag)):
            temp.append(method_flag[j][i])
        method_type[key].append(temp)
    method_type = list(method_type.items())    
    
    #4# Model training process
    # Data pre-alignment + Feature extraction + Classification
    Models, Infos = [], []
    for i in range(len(method_type)):
        pa_index = method_type[i][0]
        if method_type[i][0] != 0:
            pa = prealignments[pa_index] 
            pa_base = clone(pa)
            transformers_ = deepcopy(transformers)
            try:
                tempDataAll = pa_base.fit_transform(DataAll, LabelAll)
                exception = False
            except:
                exception = True
        else:
            transformers_ = deepcopy(transformers)
            transformers_[0].insert(0, Covariances(estimator='cov')) #对于CSP算法选择cov方法估计协方差矩阵
            for n in range(1, 4):
                transformers_[n].insert(0, Covariances(estimator='lwf'))
            exception = False
        for j in range(len(method_type[i][1])):
            fee_index, clf_index = method_type[i][1][j][1], method_type[i][1][j][2]
            if exception:
                   Models.append(None)
            else:
                fee = make_pipeline(*transformers_[fee_index])
                clf = make_pipeline(*classifiers[clf_index])
                estimator = clone(fee)
                estimator.steps += clone(clf).steps
                if method_type[i][0] == 0:
                    try:
                        Model = clone(estimator)
                        Model.fit(DataAll0, LabelAll0)
                        Models.append(Model)
                        exception = False
                    except:
                        Models.append(None)
                        exception = True
                else:
                    clf_fur = make_pipeline(
                        TLClassifier(
                            target_domain=target_domain,
                            estimator=estimator,
                            domain_weight=domain_weight,
                        ),
                    )
                    try:
                        clf_fur_base = clone(clf_fur)
                        clf_fur_base.fit(tempDataAll, LabelAll)
                        steps = pa_base.steps + clf_fur_base.steps
                        Model = Pipeline(steps)
                        Models.append(Model)
                        exception = False
                    except:
                        Models.append(None)
                        exception = True
            Infos.append({
                'prealignment': pa_methods[pa_index],
                'feature_extraction': fee_methods[fee_index],
                'classification': clf_methods[clf_index],
                'exception': exception
            })
                
    return Models, Infos

# 分类预测
def classify(model,channel,para,X):
    fs = 250
    #通道选择
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X_ch = X[:,channel,:]
    #时间截取
    start = para[0]
    end = para[1]
    X_cut = cut_data(X_ch, fs, start, end)
    #带通滤波
    lowcut = para[2]
    highcut = para[3]
    X_filt = butter_bandpass_filter(X_cut, lowcut, highcut, fs, order=5) 
    #分类
    pred = model.predict(X_filt)
    return pred

# 50Hz陷波滤波器
def get_pre_filter(data, fs=250):
    f0 = 50
    q = 35
    b, a = iircomb(f0, q, ftype='notch', fs=fs)
    filter_data = filtfilt(b, a, data)
    return filter_data

# 时间窗截取   
def cut_data(data, fs, start, end):

    # 获取data的形状，得到样本数和采样点数
    n_samples, n_channels, n_points = data.shape
    # 计算每个采样点对应的时间，单位秒
    t = np.arange(n_points) / fs 
    # 找到在start和end之间的采样点的索引
    idx = np.logical_and(t >= start, t < end) 
    # 使用切片操作，截取data的指定时间窗内的数据
    data_new = data[:, :, idx] 

    return data_new

# 设计巴特沃斯带通滤波器
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    data = get_pre_filter(data) # 50Hz陷波
    
    nyq = 0.5 * fs # 奈奎斯特频率
    low = lowcut / nyq # 归一化低截止频率
    high = highcut / nyq # 归一化高截止频率
    b, a = butter(order, [low, high], btype='band')
    data_filtered = filtfilt(b, a, data) 

    return data_filtered

class Basemodel(BaseEstimator, ClassifierMixin):
    """A collection of various classification frameworks.   
    
    """

    def __init__(self, channel=range(16,59), para=[0,2,5,32], algorind=[[0],[0],[0]], **kwds):
        """Init."""
        self.channel = channel
        self.para = para
        self.algorind = algorind
        self.kwds = kwds
        self.fs = 250

    def fit(self, Xt: ndarray, yt: ndarray, Xs = None, ys = None):

        fs = self.fs
        if Xs is None:
            X = Xt
            flag = False
        elif isinstance(Xs, list):
            X = []
            X.append(Xt)
            X = X + Xs
            del Xt, Xs
            flag = True
        elif Xs.ndim == 4:
            Xs_list = [Xs[i] for i in range(len(Xs))]
            X = []
            X.append(Xt)
            X = X + Xs_list
            del Xt, Xs
            flag = True
        elif Xs.ndim == 3:
            Xs = np.reshape(Xs, (-1, *Xs.shape[-3:]))
            X = np.concatenate((Xt, Xs), axis=0)
            del Xt, Xs
            flag = True
        
        if not flag: #isinstance(Xs, np.ndarray):    
            if any(x != 0 for x in self.algorind[0]):
                raise ValueError(
                    "Transfer learning algorithms are not allowed when there is no input from the source dataset!")   
            else:
                #通道选择
                Xt = np.reshape(Xt, (-1, *Xt.shape[-2:]))
                Xt_ch = Xt[:,self.channel,:]
                #时间截取
                start = self.para[0]
                end = self.para[1]
                Xt_cut = cut_data(Xt_ch, fs, start, end)
                #带通滤波
                lowcut = self.para[2]
                highcut = self.para[3]
                Xt_filt = butter_bandpass_filter(Xt_cut, lowcut, highcut, fs, order=5)
                #建模
                self.model,self.info = Modeling_Framework(Xt_filt, yt, method_flag = self.algorind)
        else:
            #通道选择
            ch = self.channel
            #截取时间窗内的数据
            start = self.para[0]
            end = self.para[1]
            #带通滤波
            lowcut = self.para[2]
            highcut = self.para[3]
            all_samples = []
            for samples in X:
                samples = cut_data(samples[:,ch,:], fs, start, end)
                samples = butter_bandpass_filter(samples, lowcut, highcut, fs, order=5)
                all_samples.append(samples)

            sourceData = np.float32(all_samples[1:])
            targetData = np.float32(all_samples[0])
            del all_samples

            self.model,self.info = Modeling_Framework(targetData, yt, sourceData, ys, method_flag = self.algorind)
        
        return self

    def predict(self, X):

        fs = self.fs
        #通道选择
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X_ch = X[:,self.channel,:]
        #时间截取
        start = self.para[0]
        end = self.para[1]
        X_cut = cut_data(X_ch, fs, start, end)
        #带通滤波
        lowcut = self.para[2]
        highcut = self.para[3]
        X_filt = butter_bandpass_filter(X_cut, lowcut, highcut, fs, order=5) 
        #分类            
        if isinstance(self.model, list):            
            pred = []
            for model, info in zip(self.model, self.info):
                if not info['exception']:
                    pred.append(model.predict(X_filt))
                else:
                    pred.append(np.zeros(len(X_filt), dtype=int))
        else:
            if not self.info['exception']:
                pred = self.model.predict(X_filt)
            else:
                pred = np.zeros(len(X_filt), dtype=int)
        
        return pred
    
    