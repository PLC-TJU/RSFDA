"""
Meta-Analysis Tool
Author: LC.Pan <panlincong@tju.edu.cn.com>
Date: 2024/4/17
License: BSD 3-Clause License

Introduction:
The Meta-Analysis Tool is a Python script designed to perform meta-analysis by comparing the 
significance of differences between two algorithms across multiple datasets. It uses Stouffer’s 
method to combine p-values from independent tests and provides a summary of the overall effect.

Notes:
    1> Ensure that the number of datasets and subjects in each dataset match.
    2> Adjust the parameters according to your specific research needs.

"""

import numpy as np
from scipy import stats
from statsmodels.stats import multitest

# Stouffer's method
def stouffers_method(p_values, sample_sizes):    
    """
    Calculate the combined Z-score and p-value using Stouffer's method.
    """ 
    # Convert the lists to numpy arrays
    p_values = np.array(p_values)
    sample_sizes = np.array(sample_sizes)     
    
    # Calculate the Z-scores for each p-value and weight them by the sample size
    z_scores = [-stats.norm.ppf(p/2) for p in p_values]  # for two-tailed tests
    weighted_z_scores = np.sqrt(sample_sizes) * z_scores
    combined_z = np.sum(weighted_z_scores) / np.sqrt(np.sum(sample_sizes))
    combined_p = stats.norm.sf(abs(combined_z)) * 2  # for two-tailed tests
    return combined_z, combined_p

def meta_analysis(acc_A, acc_B, test_method=None, correction_method=None, perm_cutoff=20, alternative="two-sided"):
    """
    Perform meta-analysis to compare the significance of differences between two algorithms.

    Parameters:
    acc_A (list of lists): Accuracy of algorithm A for each subject in each dataset. Should have the same length as acc_B.
    acc_B (list of lists): Accuracy of algorithm B for each subject in each dataset. Should have the same length as acc_A.
    test_method (str): The statistical test method to use ('paired_t', 'independent_t', 'wilcoxon', etc.). (default independent_t).
    correction_method (str): The multiple comparison correction method to use (None, 'bonferroni', 'fdr', etc.). (default None).
    perm_cutoff (int): threshold value for using pair t-test or Wilcoxon tests when test_method is None. (default 20).

    Returns:
    dict: A dictionary with t-values, p-values, combined Z-value, and combined p-value.
    t_values (list): t-statistics for each dataset. 
    p_values (list): The p-values for each dataset.
    combined_z (float): The combined Z-value. 
    combined_p (float): The combined p-value.
    """
    if len(acc_A)!= len(acc_B):
        raise ValueError("The number of datasets should be the same")
    
    # Perform the specified significance test for each dataset
    t_values, p_values = [], []
    sample_sizes = []
    for data_A, data_B in zip(acc_A, acc_B):
        
        if len(data_A) != len(data_B):
            raise ValueError("The number of subjects in each dataset should be the same")
        
        if test_method is None:
            _test_method = 'permutation_t' if len(data_A) < perm_cutoff else 'wilcoxon'
        else:
            _test_method = test_method
        
        if _test_method == 'paired_t':
            t_val, p_val = stats.ttest_rel(data_A, data_B, alternative=alternative)    
        elif _test_method == 'independent_t':
            t_val, p_val = stats.ttest_ind(data_A, data_B, alternative=alternative)
        elif _test_method == 'permutation_t':
            t_val, p_val = stats.ttest_ind(data_A, data_B, 
                                           permutations=10000, 
                                           random_state=42,
                                           alternative=alternative,
                                           )    
        elif _test_method == 'wilcoxon':
            t_val, p_val = stats.wilcoxon(data_A, data_B)
        else:
            raise ValueError("Unsupported test method")
        t_values.append(t_val)
        p_values.append(p_val)
        sample_sizes.append(len(data_A))

    # Apply multiple comparison correction if specified
    if correction_method == 'bonferroni':
        p_values = [min(p * len(p_values), 1.0) for p in p_values]
    elif correction_method == 'fdr':
        _, p_values, _, _ = multitest.multipletests(p_values, method='fdr_bh')
    elif correction_method is not None:
        raise ValueError("Unsupported correction method")

    combined_z, combined_p = stouffers_method(p_values, sample_sizes)

    # Return the results
    return {
        't_values': t_values,
        'p_values': p_values,
        'combined_z': combined_z,
        'combined_p': combined_p
    }

# Example usage:
# acc_A and acc_B would be lists of lists containing the accuracies for each subject in each dataset
# sample_sizes would be a list containing the number of subjects in each dataset
# results = meta_analysis(acc_A, acc_B, sample_sizes, test_method='paired_t', correction_method='bonferroni')
# print(results)

if __name__ == '__main__':
    # 假设acc_A和acc_B是长度为7的列表，每个列表子元素包含一个数据集所有受试者的分类准确率
    # 示例数据，实际应用中应替换为实际数据
    acc_A = [
        np.random.rand(9),
        np.random.rand(52),
        np.random.rand(54),
        np.random.rand(14),
        np.random.rand(106),
        np.random.rand(29),
        np.random.rand(10)
    ]
    acc_B = [
        np.random.rand(9),
        np.random.rand(52),
        np.random.rand(54),
        np.random.rand(14),
        np.random.rand(106),
        np.random.rand(29),
        np.random.rand(10)
    ]

    # 数据集的受试者数量
    sample_sizes = [9, 52, 54, 14, 106, 29, 10]
    
    results = meta_analysis(acc_A, acc_B, correction_method='bonferroni')
    
    # 打印结果    
    print("t-values:")
    [print(t) for t in results['t_values']]
    print("p-values:")
    [print(p) for p in results['p_values']]
    print("Combined Z-value:", results['combined_z'])
    print("Combined p-value:", results['combined_p'])
