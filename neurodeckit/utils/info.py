import json
import time
import numpy as np
import scipy.io
import os

def get_timestamp():
    """
    Returns a timestamp string in the format of "YYYY-MM-DD HH:MM:SS".

    Returns:
        str: Timestamp string.
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def update_json_header(json_filename, flag="-"*100, **kwargs):
    """
    Saves data to a JSON file.

    Args:
        results_json_filename (str): Path to the JSON file.
        **kwargs: Keyword arguments representing variable names and their corresponding values.

    Returns:
        None
    """
    time_str = kwargs.get("timestamp", get_timestamp())  # 时间戳
    kwargs['timestamp'] = time_str  # 加入时间戳信息
    is_exist = False
    
    # 读取文件内容
    if os.path.exists(json_filename):
        with open(json_filename, 'r') as file:
            lines = file.readlines()
            is_exist = True
    else:
        lines = []

    try:
        # 没有找到头部信息栏，则创建新的头部信息栏
        if flag + "\n" not in lines:
            header_dict = (
                f"file path: {json_filename}\n"
                f"{flag}\n"
                f"{flag}\n"
                "results:\n"
                )

            # 写入标题和原始内容
            with open(json_filename, 'w') as f:
                f.write(header_dict)
                f.writelines(lines)
            
            update_json_header(json_filename, flag=flag, **kwargs)

        # 找到头部信息栏的开始和结束索引
        else:
            start_index = lines.index(flag + "\n") + 1
            end_index = lines[start_index:].index(flag + "\n") + start_index

            # 将头部信息栏转换为字典
            header_dict = {}
            for line in lines[start_index:end_index]:
                key, value = line.strip().split(": ", 1)
                header_dict[key] = value

            # 使用新的键值对更新头部信息
            header_dict.update(kwargs)

            # 将更新后的头部字典转换回字符串
            updated_header = ""
            for key, value in header_dict.items():
                updated_header += f"{key}: {value}\n"
            # updated_header += flag + "\n"

            # 将更新后的头部和文件的其余内容写回文件
            with open(json_filename, 'w') as file:
                file.writelines(lines[:start_index])
                file.write(updated_header)
                file.writelines(lines[end_index:])
    except Exception as e:  
        
        print(f"Error: {e}")
        if is_exist:
            with open(json_filename, 'w') as file:
                file.writelines(lines)

# # 函数使用示例
# update_json_header(
#     'results.json',
#     sub='sub002',
#     dataset_name='datasetA',
#     fs=250,
#     freqband='1-40Hz',
#     n_chan=64,
#     nRepeat=100,
#     kfold=5,
#     seed=42,
#     methods='CSP',
#     dims=3,
#     algorithms='LDA',
#     device_use='CPU'
# )

def save_json_file(json_filename, **kwargs):
    """
    Saves data to a JSON file.

    Args:
        json_filename (str): Path to the JSON file.
        **kwargs: Keyword arguments representing variable names and their corresponding values.

    Returns:
        None
    """

    # Save the data to the file
    with open(json_filename, 'w') as f:
        json.dump(kwargs, f)
        f.write('\n')  # 换行，以便于读取时分割
        # print(f"Data saved to {json_filename}")   

def save_to_mat(file_path="data.mat", **kwargs):
    """
    Saves data to a .mat file.

    Args:
        file_path (str): Path to the .mat file.
        **kwargs: Keyword arguments representing variable names and their corresponding values.

    Returns:
        None
    """
    if os.path.exists(file_path):
        # Load existing data from the file
        existing_data = scipy.io.loadmat(file_path)
    else:
        # Create a new dictionary if the file does not exist
        existing_data = {}

    # Convert lists to arrays or cell arrays
    for key, value in kwargs.items():
        if isinstance(value, list):
            # Check if the list contains sublists (cell arrays)
            if all(isinstance(sublist, list) for sublist in value):
                # Convert cell arrays to numpy arrays
                # Determine the shape of the resulting cell array
                max_sublist_len = max(len(sublist) for sublist in value)
                num_sublists = len(value)
                value_array = np.empty((num_sublists, 1), dtype=object)
                for i, sublist in enumerate(value):
                    value_array[i, 0] = sublist
            else:
                # Convert regular lists to numpy arrays
                if any(isinstance(item, str) for item in value):
                    value_array = np.array(value, dtype=object)
                else:
                    try:
                        value_array = np.array(value, dtype=float)
                    except ValueError:
                        value_array = np.array(value, dtype=object)
                
        elif isinstance(value, np.ndarray):
            # Handle nested arrays (e.g., arrays of arrays)
            if value.ndim > 1:
                value_array = np.array(value.tolist(), dtype=object)
            else:
                value_array = value
        elif isinstance(value, str):
            # Convert string variables to numpy arrays
            value_array = np.array([value], dtype=object)
        else:
            # For other types, keep the original value
            value_array = value

        existing_data[key] = value_array

    # Save the updated data to the file
    scipy.io.savemat(file_path, existing_data, oned_as='column')

def append_to_mat_file(filename, new_data):
    """
    Appends new data to an existing .mat file.

    Args:
        filename (str): Path to the .mat file.
        new_data (dict): Dictionary containing the new data to be appended.

    Returns:
        None
    """
    # load existing data from the file
    if os.path.exists(filename):
        # Load existing data from the file
        existing_data = scipy.io.loadmat(filename)
    else:
        # Create a new dictionary if the file does not exist
        existing_data = {}

    # Append the new data to the existing data
    for key, value in new_data.items():
        existing_data[key] = value

    # Save the updated data to the file
    scipy.io.savemat(filename, existing_data, oned_as='column')