# Author: LC.Pan
# Date: 2024-06-21
# Version: 0.1.0
# Description: 定义一些用于构建和管理 sklearn Pipeline 的工具函数

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from typing import Callable, Union, List

def ensure_pipeline(process: Union[Pipeline, BaseEstimator, List, Callable, None, tuple]):
    """
    确保 process 是一个 sklearn-like 的 Pipeline 或 Estimator。

    参数:
    process: 任何类型的对象

    返回:
    steps: 如果 process 是一个 sklearn-like 的 Pipeline 或 Estimator，返回步骤列表

    异常:
    ValueError: 如果 process 不能被转换为 Pipeline 或 Estimator
    
    程序操作说明：
    1> 检查 process 是否是 Pipeline，如果是，则返回其步骤。
    2> 检查 process 是否是 BaseEstimator，如果是，则使用 make_pipeline 将其转换为 Pipeline，并返回步骤。
    3> 检查 process 是否是一个步骤列表（包含元组的列表），如果是，则直接返回该列表。
    4> 检查 process 是否是一个单一的 (name, transformer) 元组，如果是，则构造一个包含该元组的列表并返回。
    5> 检查 process 是否是 Callable，如果是，则使用 FunctionTransformer 将其转换为 Pipeline，并返回步骤。
    6> 如果 process 不符合上述任何一种情况，则抛出 ValueError。
    """
    if process is None:
        return []
    if isinstance(process, Pipeline):
        return process.steps
    if isinstance(process, BaseEstimator):
        return make_pipeline(process).steps
    if isinstance(process, list):
        if all(isinstance(step, tuple) and len(step) == 2 for step in process):
            return process
        else:
            raise ValueError("The process list should contain tuples of (name, transformer)")
    if isinstance(process, tuple) and len(process) == 2:
        name, transformer = process
        if isinstance(name, str) and isinstance(transformer, (BaseEstimator, Callable)):
            return [(name, transformer)]
        else:
            raise ValueError("The tuple should be in the form (name, transformer)")
    if isinstance(process, Callable):
        return make_pipeline(FunctionTransformer(process)).steps
    else:
        raise ValueError("""%s Process should be a sklearn-like estimator, a pipeline, \
                         a list of (name, transformer) tuples, a single (name, transformer) tuple, \
                         or a callable function""" % type(process))


def combine_processes(*processes, memory=None):
    """
    合并多个 sklearn-like 的 Pipeline 或 Estimator。    

    参数:
    processes: 多个 sklearn-like 的 Pipeline 或 Estimator

    返回:
    combined_process: 合并后的 Pipeline 或 Estimator

    异常:
    ValueError: 如果 processes 中存在无法转换为 Pipeline 或 Estimator 的对象
    
    程序操作说明：
    1> 接受多个 processes 参数，将每个 process 通过 ensure_pipeline 函数转换为步骤列表。
    2> 过滤掉无效的步骤（如空的 Pipeline 步骤）。
    3> 将有效的步骤合并到一个新的 Pipeline 中，并返回。
    """
    combined_steps = []
    for process in processes:
        steps = ensure_pipeline(process)
        for name, transformer in steps:
            # 只添加有效的步骤
            if not (isinstance(transformer, Pipeline) and len(transformer.steps) == 0):
                combined_steps.append((name, transformer))
    return Pipeline(steps=combined_steps, memory=memory)


def check_pipeline_compatibility(est):
    """
    检查变量 est 是否可以用于构建 sklearn 的 Pipeline。

    参数:
    est: 任何类型的对象

    返回:
    est: 如果 est 可以用于构建 Pipeline，返回 est

    异常:
    ValueError: 如果 est 不符合 Pipeline 的要求
    """
    if isinstance(est, Pipeline):
        return est

    if isinstance(est, BaseEstimator):
        if hasattr(est, 'fit'):
            if isinstance(est, TransformerMixin):
                if hasattr(est, 'transform'):
                    return est
            else:
                return est


    