# Authors: Corey Lin <panlincong@tju.edu.cn.com>
# Date: 2024/07/08
# License: BSD 3-Clause License


import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

def select_weak_classifiers(predictions, y_true=None, accuracies=None, accuracy_threshold=0.0, max_models=200, method='correlation'):
    """
    选择最终保留的子模型的索引。

    参数:
    predictions (ndarray): 子模型在评估阶段获得的预测标签值，形状为 (n_models, n_samples)。
    y_true (ndarray): 实际的标签值，形状为 (n_samples,) 或 None。如果提供 accuracies 则可为 None。
    accuracies (ndarray): 所有基模型在评估阶段的预测准确率，形状为 (n_models,) 或 None。如果提供 y_true 则可为 None。
    accuracy_threshold (float): 筛选过程中需要去掉低于阈值的子模型。
    max_models (int): 保留的子模型的最大数量限制。
    method (str): 筛选方法，可选 'correlation', 'diversity', 'select_from_model', 'sequential_feature_selector'。

    返回:
    ndarray: 最终保留的子模型的索引 (n_models,)。
    """

    n_models = len(predictions)
    
    # 如果 accuracies 未提供，则计算准确率
    if accuracies is None and y_true is not None:
        accuracies = np.array([accuracy_score(y_true, predictions[i]) for i in range(n_models)])
    
    # 根据准确率阈值筛选子模型
    accuracy_mask = accuracies >= accuracy_threshold
    filtered_indices = np.where(accuracy_mask)[0]
    filtered_predictions = predictions[filtered_indices]
    filtered_accuracies = accuracies[filtered_indices]
    
    # 根据准确率排序并选择前 max_models 个子模型
    sorted_indices = np.argsort(-filtered_accuracies)
    selected_indices = sorted_indices[:max_models]
    
    if method == 'correlation' and len(selected_indices) > 1:
        # 计算模型预测之间的相关性矩阵
        distances = pdist(filtered_predictions[selected_indices], metric='correlation')
        correlation_matrix = squareform(distances)
        
        # 使用贪心算法进一步筛选相关性较小的模型
        final_selected_indices = []
        for i in selected_indices:
            if len(final_selected_indices) == 0:
                final_selected_indices.append(i)
            else:
                correlations = [correlation_matrix[i, j] for j in final_selected_indices]
                if max(correlations) < 0.5:  # 阈值可以调整
                    final_selected_indices.append(i)
            if len(final_selected_indices) == max_models:
                break
        selected_indices = final_selected_indices

    elif method == 'diversity' and len(selected_indices) > 1:
        # 示例：使用标准差作为多样性度量
        diversities = np.std(filtered_predictions[selected_indices], axis=1)
        selected_indices = selected_indices[np.argsort(diversities)[-max_models:]]

    elif method == 'select_from_model':
        # 使用 SelectFromModel 方法进行模型筛选
        selector = SelectFromModel(LogisticRegression())
        selector.fit(filtered_predictions.T, y_true)
        selected_indices = np.where(selector.get_support())[0]

    elif method == 'sequential_feature_selector':
        # 使用 SequentialFeatureSelector 方法进行模型筛选
        sfs = SequentialFeatureSelector(LogisticRegression(), n_features_to_select=max_models)
        sfs.fit(filtered_predictions.T, y_true)
        selected_indices = np.where(sfs.get_support())[0]
    
    # 生成最终保留的子模型索引
    final_indices = np.zeros(n_models, dtype=bool)
    final_indices[filtered_indices[selected_indices]] = True
    
    return final_indices

# 示例调用
# 假设你有预测标签值 predictions 和实际标签值 y_true
# predictions = np.random.randint(0, 2, (1000, 100))  # 示例数据
# y_true = np.random.randint(0, 2, 100)  # 示例数据

# selected = select_weak_classifiers(predictions, y_true=y_true, accuracy_threshold=0.6, max_models=200, method='correlation')


