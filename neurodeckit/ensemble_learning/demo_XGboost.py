import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 假设X和y是您的特征和标签
X = np.random.rand(1000, 10)
y = np.random.randint(2, size=1000)

# 假设X和y是您的特征和标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置参数
params = {
    'objective': 'binary:logistic',  # 目标函数
    'max_depth': 3,                  # 树的最大深度
    'eta': 0.1,                      # 学习率
    'eval_metric': 'logloss'         # 评估指标
}

# 训练模型
bst = xgb.train(params, dtrain, num_boost_round=100)

# 预测
y_pred = bst.predict(dtest)
y_pred_binary = (y_pred > 0.5).astype(int)

# 评估模型
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"XGBoost Accuracy: {accuracy}")
