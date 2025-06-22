import os
import json

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)
        f.write('\n')  # 换行，以便于读取时分割

# 生成每个作业的唯一标识符
def generate_job_identifier(job, keys_to_include=None):
    if keys_to_include is None:
        keys_to_include = ['channels', 'start_time', 'end_time', 'lowcut', 'highcut', 'cs_method', 'nelec', 
                           'tl_mode', 'aug_method', 'algorithm', 'algorithm_id']
    # 根据关键字排序，确保唯一性
    identifier = '_'.join(str(job[key]) for key in sorted(keys_to_include))
    return identifier

# 检查已完成的计算并返回未完成的计算列表
def check_completed_jobs(filename, parasets):
    if not os.path.exists(filename):
        return parasets
    else:
        with open(filename, 'r') as f:
            lines = f.readlines()
        completed_jobs = [json.loads(line.strip()) for line in lines]
        # 使用生成的唯一标识符来标记已完成的作业
        completed_jobs_identifiers = set(generate_job_identifier(job) for job in completed_jobs)
        uncompleted_jobs = [job for job in parasets if generate_job_identifier(job) not in completed_jobs_identifiers]
        return uncompleted_jobs