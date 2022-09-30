import os
import inspect
import sys
from datetime import datetime
from functools import wraps

timestamp_path = ''


def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_timestamp_path(path: str) -> None:
    global timestamp_path
    timestamp_path = path


def record_launch_time(func):

    @wraps(func)
    def get_args(*args):
        global timestamp_path
        with open(os.path.join(timestamp_path, 'time_stamp.txt'),
                  'a') as time_stamp:
            start_time = datetime.now()
            cal_frame = inspect.getouterframes(inspect.currentframe(), 2)
            now_func_name = sys._getframe().f_code.co_name
            layer = len(
                list(filter(lambda x: x.function == now_func_name,
                            cal_frame))) - 1
            suffix = ''
            func_name = func.__name__
            if layer > 0:
                suffix = '\t' * layer
            else:
                time_stamp.write(
                    '%s------------------- %s -------------------\n' %
                    (suffix, func_name))
                time_stamp.flush()
            result = func(*args)
            time_stamp.write('{}{:<20} {:>30}\n'.format(
                suffix, func_name, (datetime.now() - start_time).__str__()))
            time_stamp.flush()
            return result

    return get_args


def record_launch_time_and_param(*param_index):

    def time_(func=None):

        @wraps(func)
        def get_args(*args, ):
            global timestamp_path
            with open(os.path.join(timestamp_path, 'time_stamp.txt'),
                      'a') as time_stamp:
                start_time = datetime.now()
                cal_frame = inspect.getouterframes(inspect.currentframe(), 2)
                now_func_name = sys._getframe().f_code.co_name
                layer = len(
                    list(
                        filter(lambda x: x.function == now_func_name,
                               cal_frame))) - 1
                suffix = ''
                if layer > 0:
                    suffix = '\t' * layer
                func_name = func.__name__
                result = func(*args)
                time_stamp.write('{} {:<20} {:>30}'.format(
                    suffix, func_name,
                    (datetime.now() - start_time).__str__()))
                if len(param_index) != 0:
                    time_stamp.write(' {:>30}'.format(','.join(
                        [str(args[i]) for i in param_index])))
                time_stamp.write("\n")
                time_stamp.flush()
                return result

        return get_args

    return time_
