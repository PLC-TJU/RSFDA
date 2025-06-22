# 导入所需的库
from sklearn.svm import SVC  # 支持向量机
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # 线性判别分析
from sklearn.linear_model import LogisticRegression as LR  # 逻辑回归
from sklearn.neighbors import KNeighborsClassifier as KNN  # K 近邻
from sklearn.tree import DecisionTreeClassifier as DTC  # 决策树
from sklearn.ensemble import RandomForestClassifier as RFC  # 随机森林
from sklearn.ensemble import ExtraTreesClassifier as ETC  # 极端随机森林
from sklearn.ensemble import AdaBoostClassifier as ABC  # AdaBoost
from sklearn.ensemble import GradientBoostingClassifier as GBC  # 梯度提升
from sklearn.naive_bayes import GaussianNB as GNB  # 高斯朴素贝叶斯
from sklearn.neural_network import MLPClassifier as MLP  # 多层感知机

# 支持向量机 (SVC)
# 特点: 强大的分类器，适用于高维空间。
# 优点: 在复杂数据集上表现良好。
# 缺点: 计算复杂度高，内存占用大。
# 适用场景: 文本分类、图像识别等高维问题。
# 使用说明: SVC(kernel='linear', C=1.0)
svc = SVC(kernel='linear', C=1.0)

# 线性判别分析 (LDA)
# 特点: 寻找不同类别数据的最佳投影。
# 优点: 模型简单，计算效率高。
# 缺点: 假设所有类别数据具有相同协方差。
# 适用场景: 类别明显且数据线性可分的情况。
lda = LDA(solver='eigen', shrinkage='auto')

# 逻辑回归 (LR)
# 特点: 概率框架，输出可解释性强。
# 优点: 实现简单，效率高。
# 缺点: 对非线性问题表现不佳。
# 适用场景: 二分类问题。
lr = LR()

# K 近邻 (KNN)
# 特点: 基于距离的非参数方法。
# 优点: 理论简单，易于实现。
# 缺点: 对大数据集计算量大。
# 适用场景: 小数据集上的分类。
knn = KNN(n_neighbors=5)

# 决策树 (DTC)
# 特点: 易于理解和解释。
# 优点: 不需要数据预处理。
# 缺点: 容易过拟合。
# 适用场景: 需要解释模型决策过程的应用。
dtc = DTC(min_samples_split=2)

# 随机森林 (RFC)
# 特点: 集成多个决策树，提高稳定性。
# 优点: 对过拟合有抵抗力。
# 缺点: 模型较大，需要更多计算资源。
# 适用场景: 处理高维数据和大数据集。
rfc = RFC(n_estimators=50)

# 极端随机森林 (ETC)
# 特点: 在随机森林基础上增加随机性。
# 优点: 训练速度快。
# 缺点: 可能会增加方差。
# 适用场景: 需要快速模型训练的情况。
etc = ETC(n_estimators=50)

# AdaBoost (ABC)
# 特点: 通过增加之前分类错误样本的权重来提升性能。
# 优点: 准确率高。
# 缺点: 对噪声和异常值敏感。
# 适用场景: 二分类或多分类问题。
abc = ABC(n_estimators=50, learning_rate=1.0)

# 梯度提升 (GBC)
# 特点: 通过逐步添加模型减少误差。
# 优点: 在多种数据集上表现优秀。
# 缺点: 训练时间可能较长。
# 适用场景: 处理各种类型的数据，包括非平衡数据。
gbc = GBC(n_estimators=100, learning_rate=0.1)

# 高斯朴素贝叶斯 (GNB)
# 特点: 基于概率的简单分类器。
# 优点: 训练和预测速度快。
# 缺点: 假设特征之间相互独立。
# 适用场景: 文本分类和自然语言处理。
gnb = GNB()

# 多层感知机 (MLP)
# 特点: 神经网络的一种，适用于复杂模式识别。
# 优点: 能够捕捉数据中的非线性关系。
# 缺点: 需要大量数据，容易过拟合。
# 适用场景: 图像和语音识别等高级特征学习。
mlp = MLP(hidden_layer_sizes=(50,), max_iter=1000, alpha=0.0001, solver='adam')
