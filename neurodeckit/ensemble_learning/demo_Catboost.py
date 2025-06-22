import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 假设X和y是您的特征和标签
X = np.random.rand(1000, 10)
y = np.random.randint(2, size=1000)

# 假设X和y是您的特征和标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建CatBoost分类器
catboost = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=3, verbose=0)

# 训练模型
catboost.fit(X_train, y_train)

# 预测
y_pred = catboost.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"CatBoost Accuracy: {accuracy}")
