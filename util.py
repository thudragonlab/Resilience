import os
from datetime import datetime

timestamp_path = ''

def mkdir(path):
    os.makedirs(path,exist_ok=True)


def set_timestamp_path(path):
    global timestamp_path
    timestamp_path = path

def record_launch_time(func):
    def get_args(*args):
        global timestamp_path
        with open(os.path.join(timestamp_path,'time_stamp.txt'), 'a') as time_stamp:
            start_time = datetime.now()
            func_name = func.__name__
            result = func(*args)
            time_stamp.write('%s  %40s  \n' % (func_name, datetime.now() - start_time))
            return result
    return get_args
