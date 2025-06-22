# Author: Pan Lincong
# Edition date: 5 Mar 2023

import numpy as np
import h5py, re, os, scipy.io
from scipy.signal import iircomb, iirnotch, butter, filtfilt, resample
import pynvml
import psutil
import subprocess
import inspect

def load_mat(filename):
    # 输入是一个字符串，表示mat文件的路径
    # 输出是一个字典，包含mat文件中的变量
    # 功能是使用h5py模块来读取v7.3格式的mat文件
    
    # 创建一个空字典result
    result = {}
    # 打开mat文件，返回一个文件对象
    f = h5py.File(filename, 'r')
    # 查看文件中的变量名，返回一个列表
    var_names = list(f.keys())
    # 对于每个变量名：
    for var_name in var_names:
        # 读取该变量，返回一个数组对象
        var_value = f[var_name]
        # 将变量名和值添加到result中
        result[var_name] = np.array(var_value)
    # 关闭文件
    f.close()
       
    # 返回result
    return result

def transpose_to_matlab_order(data):
    if isinstance(data, np.ndarray):
        # 获取数组的维度数
        num_axes = len(data.shape)
        # 生成新的轴顺序，以匹配MATLAB的列优先顺序
        new_axes = tuple(reversed(range(num_axes)))
        # 调整数组的维度顺序
        return np.transpose(data, axes=new_axes)
    return data

def fix_dims(data):
    # 检查数据是否为HDF5数据集
    if isinstance(data, h5py.Dataset):
        # 如果数据是数值类型，调整维度顺序
        if data.dtype.kind in 'iufc':  # 整数、无符号整数、浮点数、复数
            return transpose_to_matlab_order(np.array(data))
        else:
            # 对于其他类型，如字符串，可能需要不同的处理
            return data[()]
    return data

def load_mat2(filename):
    # 输入是一个字符串，表示mat文件的路径
    # 输出是一个字典，包含mat文件中的变量
    # 功能是使用h5py模块来读取v7.3格式的mat文件
    # 或使用scipy.io.loadmat模块来读取低于v7.3格式的mat文件
    
    """Load MATLAB .mat file and maintain the original dimension order.

    Parameters
    ----------
    filename : str
        Path to the .mat file.

    Returns
    -------
    dict
        Dictionary containing variables from the .mat file.
    """
    
    # 检查文件是否存在
    if not os.path.isfile(filename):
        print(f"文件 {filename} 不存在。")
        return {}

    # 尝试确定文件是否为MATLAB v7.3格式
    try:
        with h5py.File(filename, 'r') as file:
            # 如果成功，说明是v7.3格式的文件
            print(f"文件 {filename} 是MATLAB v7.3格式的.mat文件。")
            result = {}
            for var_name in file.keys():
                result[var_name] = fix_dims(file[var_name])
            return result
    except OSError:
        # 如果失败，说明可能不是v7.3格式的文件
        pass

    # 如果不是v7.3格式的文件，尝试使用scipy.io.loadmat来读取
    try:
        mat_data = scipy.io.loadmat(filename, 
                                    squeeze_me=True, 
                                    struct_as_record=False, 
                                    verify_compressed_data_integrity=False)
        # 移除不是变量的键
        mat_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}
        print(f"文件 {filename} 是低于MATLAB v7.3版本的.mat文件。")
        return mat_data
    except NotImplementedError as e:
        print(f"使用scipy.io.loadmat读取文件时出错: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

    # 如果都不成功，返回一个空字典
    return {}

def resample_data(data, fs, fs_new):
    # 设定降采样的参数
    num = data.shape[2] # 原始采样点数
    num_new = int(num * fs_new / fs) # 新的采样点数

    # 对data进行降采样
    data_resampled = resample(data, num_new, axis=2) # 返回一个长度为num_new的数组

    return data_resampled

# iirnotch和iircomb都是设计IIR陷波器的方法，但它们的参数和设计原理有所不同。
# iirnotch通常用于设计简单的二阶陷波器，而iircomb可以设计出具有多个凹槽的组合滤波器。
# 在实际应用中，iirnotch通常足以抑制单一频率的干扰，如电源线干扰。
# 如果您需要更复杂的陷波效果，比如同时抑制多个谐波，那么iircomb可能是更好的选择。

# 使用iircomb来设计陷波器时，如果fs不是f0的倍数，可能会导致设计的陷波器无法正确抑制指定频率，
# 因为iircomb函数设计的陷波器的中心频率是基于采样频率和指定频率的关系确定的。
def get_pre_filter0(data, fs=250):    
    f0 = 50
    q = 35
    b, a = iircomb(f0, q, ftype='notch', fs=fs)
    filter_data = filtfilt(b, a, data)
    return filter_data

def get_pre_filter(data, fs=250):
    f0 = 50  # 中心频率
    q = 35   # 品质因数
    b, a = iirnotch(f0, q, fs)
    filter_data = filtfilt(b, a, data)
    return filter_data

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):   
    # data = get_pre_filter(data, fs) # 50Hz notch filter
    # 设计巴特沃斯带通滤波器
    nyq = 0.5 * fs # 奈奎斯特频率
    low = lowcut / nyq # 归一化低截止频率
    high = highcut / nyq # 归一化高截止频率
    b, a = butter(order, [low, high], btype='band') # 返回滤波器的分子和分母系数

    # 对data进行滤波
    data_filtered = filtfilt(b, a, data) # 使用filtfilt函数可以避免相位延迟

    return data_filtered

def cut_data(data, fs, start, end):
    # 输入是一个三维数组data，表示原始数据，形状为(样本数, 通道数, 采样点数)
    # 一个整数fs，表示采样频率，单位Hz
    # 两个浮点数start和end，表示要截取的时间窗的起始和结束时间，单位秒
    # 输出是一个三维数组data_new，表示截取后的数据，形状为(样本数, 通道数, 新的采样点数)
    # 功能是使用切片操作和逻辑运算来截取data的任意时间窗内的数据
    
    # 获取data的形状，得到样本数和采样点数
    n_samples, n_channels, n_points = data.shape
    # 计算每个采样点对应的时间，单位秒
    t = np.arange(n_points) / fs # 返回一个长度为n_points的一维数组，表示时间序列
    # 找到在start和end之间的采样点的索引
    idx = np.logical_and(t >= start, t < end) # 返回一个长度为n_points的布尔数组，表示是否在时间窗内
    # 使用切片操作，截取data的指定时间窗内的数据
    data_new = data[:, :, idx] # 返回一个形状为(n_samples, n_channels, sum(idx))的数组
    # 返回data_new
    return data_new

# 定义一个函数，用于对多导联EEG信号进行降采样和提取MI任务态信号
def downsample_and_extract(EEG, fs_old, fs_new, window):
    """Downsample and extract multi-channel EEG signals for MI task.

    This function takes an array of multi-channel EEG signals, an original sampling frequency,
    a new sampling frequency, and a time window as inputs, and returns an array of downsampled
    and extracted EEG signals for MI task as output.
    The function uses scipy.signal.resample to downsample each channel of the EEG signals,
    and then extracts the signals within the specified time window.
    The time window is a tuple of start time and end time in seconds.

    Parameters
    ----------
    EEG : array
        The array of multi-channel EEG signals, with shape (n, c, m), where n is the number of samples,
        c is the number of channels, and m is the number of original samples per channel.
    fs_old : int
        The original sampling frequency of the EEG signals, in Hz.
    fs_new : int
        The new sampling frequency of the EEG signals, in Hz.
    window : tuple
        The time window for extracting the EEG signals for MI task, in seconds.
        It is a tuple of start time and end time, such as (2.5, 6.5).

    Returns
    -------
    all_EEG : array
        The array of downsampled and extracted EEG signals for MI task, with shape (n, c, k),
        where n is the number of samples, c is the number of channels,
        and k is the number of new samples per channel within the time window.

    Example
    -------
    >>> EEG = np.random.rand(10,64,10000)
    >>> fs_old = 1000
    >>> fs_new = 250
    >>> window = (0, 4)
    >>> all_EEG = downsample_and_extract(EEG, fs_old, fs_new, window)
    """
    
    # Define an empty list to store all samples' downsampled and extracted signals
    all_EEG = []
    # Traverse each sample's signal
    for sample in EEG:
        # Downsample the EEG signal using scipy.signal.resample function
        # Calculate the length of the downsampled signal
        length_new = int(len(sample[0]) * fs_new / fs_old)
        # Downsample each lead's signal and convert to numpy array
        EEG_new = np.array([resample(signal, length_new) for signal in sample])
        # Extract signals within specific time window,
        # assuming window is a tuple representing start time and end time in seconds
        # Calculate index corresponding to time window
        start = int(window[0] * fs_new)
        end = int(window[1] * fs_new)
        # Cut out signals within time window
        EEG_window = EEG_new[:, start:end]
        # Add result to list
        all_EEG.append(EEG_window)
    # Convert list to numpy array with dimensions sample count * channel count * time window length
    all_EEG = np.array(all_EEG)
    # Return downsampled and extracted signal
    return all_EEG

# 接受EEG样本及其标签，采样率，时间窗宽度和步长作为参数，返回一个包含划分后的样本的列表
def split_eeg(eeg, tags, fs=250, window_width=2, window_step=0.1):
    """Split EEG array into samples with different window sizes and steps.

    Args:
        eeg (numpy.ndarray): The EEG array with shape (n_samples, n_channels, n_timepoints).
        tags (numpy.ndarray): The label array with shape (n_samples,) or (n_samples, 1), 
        where each element represents the label of the corresponding sample in eeg.
        fs (int): The sampling rate of the EEG array in Hz.
        window_width (float): The width of the window in seconds.
        window_step (float): The step of the window in seconds.

    Returns:
        ndarray: samples with shape (n_samples * n_windows, n_channels, width), where 
        width is the number of timepoints in each window.
        ndarray: labels with shape (n_samples * n_windows,) or (n_samples * n_windows, 1), 
        where each element represents the label of the corresponding sample in samples.

    Raises:
        ValueError: If the window_width or window_step is not positive or larger than the 
        number of timepoints.
    
    Example:
        >>> eeg = np.random.randn(90, 64, 1000) # generate a random EEG array
        >>> label = np.random.randint(0, 2, 90) # generate a random label array
        >>> fs = 250 # set the sampling rate to 250 Hz
        >>> window_width = 1.5 # set the window width to 1.5 seconds
        >>> window_step = 0.1 # set the window step to 0.1 seconds
        >>> samples, labels = split_eeg(eeg, label, fs, window_width, window_step) # split the EEG array into samples and labels
        >>> print(len(samples)) # print the number of samples
        4050
        >>> print(len(labels)) # print the number of labels
        4050
    """
    # convert window_width and window_step from seconds to samples
    width = int(window_width * fs)
    step = int(window_step * fs)
    # get the shape of eeg
    n_samples, n_channels, n_timepoints = eeg.shape
    # initialize an empty list to store the samples
    samples, labels = [], []
    # loop through each sample
    for i in range(n_samples):
        # get the current sample and label
        sample, label = eeg[i], tags[i]
        # initialize the start and end indices of the window
        start = 0
        end = width
        # loop until the end index exceeds the number of timepoints
        while end <= n_timepoints:
            # get the current window
            window = sample[:, start:end]
            # append the window and label to the samples and labels list
            samples.append(window)
            labels.append(label)
            # update the start and end indices by adding the step size
            start += step
            end += step
    # return the samples list
    return np.array(samples), np.array(labels)

# 定义一个提取数字的函数
def extract_number(filename):
    match = re.search(r's(\d+)\.mat', filename)
    if match:
        return int(match.group(1))
    return None

# 定义一个提取关键排序信息的函数
def extract_sort_key(filename):
    # 匹配多种模式：'A01E.mat', 'S01A.mat', 'S01E.mat', 'A.mat', 'S01D1.mat'等
    match = re.search(r'([A-Za-z]+)(\d*)([A-Za-z]*)(\d*)\.mat', filename)
    if match:
        # 提取匹配的部分，并转换数字为整数以确保正确排序
        parts = match.groups()
        return (parts[0], int(parts[1]) if parts[1] else 0, parts[2], int(parts[3]) if parts[3] else 0)
    return filename

def generate_intervals(window_width = 4, step = 4, range_start_end = (4, 40)):
    """
    生成一个区间列表。

    参数:
    window_width (int): 区间的窗宽。
    step (int): 区间的步长。
    range_start_end (tuple): 区间的起止范围。

    返回:
    list of tuples: 生成的区间列表。

    功能说明:
    这个函数根据指定的窗宽、步长和起止范围生成一个区间列表。
    每个区间是一个元组，形式为(start, start + window_width)，
    其中start从range_start_end[0]开始，每次增加step，直到range_start_end[1]。

    使用说明:
    1. 指定窗宽、步长和区间的起止范围，例如：window_width=4, step=4, range_start_end=(0,40)。
    2. 调用函数并传入这些参数，例如：A = generate_intervals(4, 4, (0, 40))。
    3. 函数会返回一个生成的区间列表。
    """

    range_start, range_end = range_start_end
    return [(i, i + window_width) for i in range(range_start, range_end, step) if i + window_width <= range_end]

def adjust_intervals(intervals = generate_intervals, delta = 2):
    """
    调整区间列表中的每个元组。

    参数:
    intervals (list of tuples): 需要调整的区间列表。
    delta (int): 调整区间的增减值。

    返回:
    list of tuples: 调整后的区间列表。

    功能说明:
    这个函数接收一个区间列表和一个整数delta，然后返回一个新的区间列表。
    新列表中的每个区间都是原区间的起始值减去delta，终止值加上delta。

    使用说明:
    1. 创建一个区间列表，例如：A = [(4,8), (8,12), ...]。
    2. 调用函数并传入区间列表和delta值，例如：B = adjust_intervals(A, 2)。
    3. 函数会返回一个新的区间列表。
    """

    return [(x-delta, y+delta) for x, y in intervals]

# filterBank
def filterBank(x, fs, bands = generate_intervals(4, 4, (4, 40))):
    """
    计算带通滤波器组。

    参数:
    x (numpy.ndarray): 输入信号，形状为(samples, channels, points)。
    fs (int): 采样率。
    bands (list of tuples): 子带的频率范围列表。

    返回:
    y (numpy.ndarray): 带通滤波器组的输出信号，形状为(samples, channels, points, filters_count)。

    功能说明:
    这个函数接收输入信号x、采样率fs和子带的频率范围列表bands，然后计算带通滤波器组的输出信号。
    输出信号的形状为(samples, channels, points, filters_count)，其中filters_count为子带的数量。

    使用说明:
    1. 创建一个输入信号，例如：x = np.random.randn(10, 64, 10000)。
    2. 调用函数并传入信号、采样率和子带的频率范围列表，例如：y = filterBank(x, 250, generate_intervals(4, 4, (4, 40)))。
    3. 函数会返回带通滤波器组的输出信号。
    """
    
    # 初始化输出信号数组
    samples, channels, points = x.shape
    filters_count = len(bands)
    y = np.empty((samples, channels, points, filters_count))

    # 对每个样本和通道应用滤波器组
    for i, band in enumerate(bands):
        y[:, :, :, i] = butter_bandpass_filter(x, band[0], band[1], fs)

    return y

import pynvml

# 获取当前计算机上的独立GPU的利用率
def get_gpu_usage():
    """
    获取当前计算机上的独立GPU的使用率和内存使用率。

    返回:
        gpu_utilization (int): GPU的使用率百分比。
        memory_utilization (int): GPU内存的使用率百分比。
        如果无法获取GPU信息或没有独立GPU，则返回None。

    注意:
        这个函数假设独立GPU是列表中的第一个GPU（索引为0）。
        确保NVIDIA驱动已安装且支持NVML。
    
    # 使用示例
    gpu_usage, memory_usage = get_gpu_usage()
    if gpu_usage is not None and memory_usage is not None:
        print(f"独立GPU的使用率为: {gpu_usage}%")
        print(f"独立GPU的内存使用率为: {memory_usage}%")
    else:
        print("无法获取独立GPU的使用率或内存使用率。")
    """
    try:
        # 初始化NVML
        pynvml.nvmlInit()

        # 获取第一块GPU的句柄，索引为0
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # 使用句柄获取GPU利用率
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization = utilization.gpu

        # 使用句柄获取内存信息，并计算内存利用率
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_utilization = (memory_info.used / memory_info.total) * 100

        # 关闭NVML
        pynvml.nvmlShutdown()

        # 返回GPU利用率和内存利用率
        return gpu_utilization, memory_utilization
    
    except pynvml.NVMLError as error:
        print(f"获取GPU信息时发生错误: {error}")
        return None, None

def get_system_usage(interval=None):
    """
    获取当前计算机的CPU和内存使用率。
    
    参数:
        interval (float): 获取CPU使用率前的等待时间（秒）。

    返回:
        cpu_usage (float): CPU的使用率百分比。
        memory_usage (float): 内存的使用率百分比。

    注意:
        为了获得准确的CPU使用率，请确保在调用此函数前系统至少空闲了1秒。

    # 使用示例
    cpu_usage, memory_usage = get_system_usage()
    print(f"CPU的使用率为: {cpu_usage}%")
    print(f"内存的使用率为: {memory_usage}%")
    """

    cpu_usage = np.sum(psutil.cpu_percent(interval=interval, percpu=True))
    memory_usage = psutil.virtual_memory().percent
    return cpu_usage, memory_usage

# 定义一个函数，用于创建文件夹，如果文件夹不存在则创建
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


# pip freeze > requirements.txt  # 生成requirements.txt文件，保存当前环境的所有工具包
# 生成requirements.txt文件，仅保存当前程序运行所需的工具包
def generate_requirements(project_path=None, output_path=None):
    """
    生成指定项目的requirements.txt文件。

    参数:
    - project_path: str, 项目文件夹的路径，默认为当前工作目录。
    - output_path: str, requirements.txt文件的保存路径，默认为当前工作目录。

    返回:
    - 无

    使用说明:
    - 调用函数时，可以指定项目路径和输出路径。
    - 如果不指定路径，将使用当前工作目录。
    - 函数会在指定的输出路径生成requirements.txt文件。
    - 如果pipreqs未安装，函数会提示安装。
    - 如果输出路径中已存在requirements.txt文件，将提示用户是否覆盖。
    """

    # 设置默认路径为当前工作目录
    if project_path is None:
        project_path = os.getcwd()
    if output_path is None:
        output_path = os.getcwd()

    # 检查pipreqs是否已安装
    reqs_installed = subprocess.run(['pip', 'show', 'pipreqs'], capture_output=True, text=True)
    if "Name: pipreqs" not in reqs_installed.stdout:
        print("pipreqs未安装，请先安装pipreqs。")
        return

    # 检查requirements.txt文件是否已存在
    req_file_path = os.path.join(output_path, 'requirements.txt')
    if os.path.isfile(req_file_path):
        print(f"检测到 {req_file_path} 已存在。")
        overwrite = input("是否覆盖现有文件？(y/n): ").strip().lower()
        if overwrite != 'y':
            print("操作已取消。")
            return

    # 生成requirements.txt文件
    try:
        print("正在生成requirements.txt文件...")
        subprocess.run(['pipreqs', project_path, '--force', '--savepath', req_file_path], check=True)
        print(f"requirements.txt文件已生成在 {output_path}")
    except subprocess.CalledProcessError as e:
        print("生成requirements.txt文件时出错：", e)
        print("错误详情：", e.stderr.decode('utf-8'))  # 打印详细的错误信息
        

# 定义一个函数，用于提取类中所有字典的键值
import inspect
import ast
import textwrap
import importlib

def extract_dict_keys(module_name, class_name, func_name, dict_name):
    """
    提取类中指定函数的指定字典的键值。

    参数:
    - module_name: str, 模块名。
    - class_name: str, 类名。
    - func_name: str, 函数名。
    - dict_name: str, 字典名。

    返回:
    - list of str, 字典的键值列表。

    使用说明:
    - 调用函数时，需要指定类名、函数名和字典名。
    - 函数会在类中查找指定函数，并在函数中查找指定字典的定义。
    - 函数会返回指定字典的键值列表。
    - 如果类、函数或字典不存在，函数会抛出ValueError。
    
    示例:      
    ```python
    my_module.py:
    class MyClass:
        def func_name(self):
            dict_name = {'key1': 1, 'key2': 2, 'key3': 3}
            return dict_name
    
    keys = extract_dict_keys('my_module', 'MyClass', 'func_name', 'dict_name')
    print(keys)  # ['key1', 'key2', 'key3']
    ```
    """
    # 导入模块
    module = importlib.import_module(module_name)
    
    # 获取类对象
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ValueError(f"Class '{class_name}' not found in module '{module_name}'.")

    # 获取函数对象
    func = getattr(cls, func_name, None)
    if func is None:
        raise ValueError(f"Function '{func_name}' not found in class '{class_name}'.")

    # 获取函数的源代码
    source = inspect.getsource(func)
    
    # 处理缩进
    source = textwrap.dedent(source)
    
    # 解析源代码为AST
    tree = ast.parse(source)
    
    # 查找指定字典的定义
    class DictVisitor(ast.NodeVisitor):
        def __init__(self):
            self.keys = []
        
        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == dict_name:
                    if isinstance(node.value, ast.Dict):
                        self.keys = [k.s for k in node.value.keys if isinstance(k, ast.Str)]
            self.generic_visit(node)
    
    visitor = DictVisitor()
    visitor.visit(tree)
    
    return visitor.keys

# 定义一个函数，用于展开X和y，使X变成标准的三维样本:样本数*通道数*时间点数
def check_sample_dims(X, y):
    """
    展开X和y，使X变成标准的三维样本（不同通道成分数目*不同时间窗成分数目*多个频带成分数目*样本数）*通道数*时间点数
    并且扩展y以匹配新的样本维度。

    Parameters:
    X (np.ndarray): 输入数据，维度为(不同通道成分, 不同时间窗成分, 多个频带成分, ..., 样本数, 通道数, 时间点数)
    y (np.ndarray): 标签数据，维度为(样本数, 1)

    Returns:
    tuple: (新的X, 新的y)
        - 新的X: 维度为(新的样本数, 通道数, 时间点数)
        - 新的y: 维度为(新的样本数, 1)
    """
    # 获取输入X的维度
    input_shape = X.shape
    
    # 检查输入X的维度是否正确
    if len(input_shape) < 3:
        raise ValueError("输入X的维度不正确，至少需要3维。")
    elif  len(input_shape) == 3:  # 输入X的维度为(样本数, 通道数, 时间点数)
        return X, y

    # 样本数、通道数、时间点数
    sample_count, channel_count, time_point_count = input_shape[-3], input_shape[-2], input_shape[-1]

    # 计算新的样本数
    new_sample_count = np.prod(input_shape[:-3]) * sample_count

    # 重塑X
    new_X = X.reshape((new_sample_count, channel_count, time_point_count))

    # 扩展y
    new_y = np.repeat(y, np.prod(input_shape[:-3]), axis=0)

    return new_X, new_y
