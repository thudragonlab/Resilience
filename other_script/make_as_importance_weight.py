import json
import os
import csv
import multiprocessing
from other_script.my_types import *
from typing import Dict, List, NewType, Tuple, Union
from other_script.util import mkdir, record_launch_time, record_launch_time_and_param

WEIGHT = NewType('Weight', List[Union[USER_IMPORTANT_WEIGHT,DOMAIN_IMPORTANT_WEIGHT]])


def get_radio(as_list: List[str], path: str) -> Dict[str, int]:
    inner_rank_map: Dict[str, int] = {}
    sum_rank = 0
    user_weight_list: List[Union[str, int]] = []
    with open(os.path.join(path, 'input', 'as_user_ratio.json'), 'r') as user_file:
        data = json.load(user_file)
        rank_index = 0
        for i in data:
            if i[1] in as_list:
                user_weight_list.append([i[1], i[4]])
        user_weight_list.sort(key=lambda x: x[1])
        for i in user_weight_list:
            if i[0][2:] not in inner_rank_map:
                inner_rank_map[i[0][2:]] = 0
            # print(i[1])
            rank_index += 1
            sum_rank += 1 / int(rank_index)
            inner_rank_map[i[0][2:]] += (1 / int(rank_index))

        for _as in inner_rank_map:
            # print(_as, inner_rank_map[_as] / sum_rank)
            inner_rank_map[_as] = inner_rank_map[_as] / sum_rank
        return inner_rank_map


def get_radio_domain(as_list_domain: List[str], csv_data: List[str]) -> Dict[str, int]:
    as_domain_map: Dict[str, int] = {}
    sum_rank_domain = 0
    rank_index = 0
    as_weight_list: List[Union[str, int]] = []
    for c in csv_data:
        weight = c[4]
        # weight = c[9]
        if c[2] == '':
            continue
        if c[2] not in as_list_domain:
            continue
        # print(c[2], weight)
        as_weight_list.append([c[2], weight])
    as_weight_list.sort(key=lambda x: x[1])
    for c in as_weight_list:
        if c[0] not in as_domain_map:
            as_domain_map[c[0]] = 0
        rank_index += 1
        sum_rank_domain += (1 / rank_index)
        as_domain_map[c[0]] += (1 / rank_index)

    for _key in as_domain_map:
        as_domain_map[_key] = as_domain_map[_key] / sum_rank_domain
    return as_domain_map


@record_launch_time_and_param(0)
def do_something(ccc: COUNTRY_CODE, path: ROOT_PATH, csv_data: List[str]) -> None:
    cc2as_path2 = os.path.join(path, 'output/cc2as')
    as_map: Dict[AS_CODE, WEIGHT] = {}
    result: List[Tuple[COUNTRY_CODE,USER_IMPORTANT_WEIGHT,DOMAIN_IMPORTANT_WEIGHT]] = []
    output_path = os.path.join(path, 'output/weight_data')
    mkdir(output_path)
    if os.path.exists(os.path.join(output_path, '%s.json' % ccc)):
        return
    try:
        with open(os.path.join(cc2as_path2, '%s.json' % ccc), 'r') as cc_file:
            cc_as_list: List[str] = json.load(cc_file)
            user_rank_map = get_radio(list(map(lambda x: 'AS%s' % x, cc_as_list)), path)
            domain_rank_map = get_radio_domain(cc_as_list, csv_data)
        for _as in user_rank_map:
            if _as not in as_map:
                as_map[_as] = [0, 0]
            as_map[_as][0] = user_rank_map[_as]

        for _as in domain_rank_map:
            if _as not in as_map:
                as_map[_as] = [0, 0]
            as_map[_as][1] = domain_rank_map[_as]

        for _as in as_map:
            result.append((_as,*as_map[_as]))
        with open(os.path.join(output_path, '%s.json' % ccc), 'w') as f:
            json.dump(result, f)
    except Exception as e:
        print(e)
        raise e


@record_launch_time
def make_as_importance(path: ROOT_PATH, cc_list: List[COUNTRY_CODE]) -> WEIGHT_PATH:
    '''
    path:input的父级目录
    cc_list:参与计算的国家列表

    按照cc_list生成每个国家的权重数据

    return 存储权重数据的文件夹路径
    '''
    csv_data: List[str] = []
    with open(os.path.join(path, 'input', 'normprefixrank_list-alexa_family-4_limit-all.csv'), 'r') as csv_file:
        for i in csv.reader(csv_file):
            csv_data.append(i)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for cc in cc_list:
        try:
            pool.apply_async(do_something, (cc, path, csv_data))
            # do_something (cc, path, csv_data)
        except Exception as e:
            print(e)
    pool.close()
    pool.join()
    return os.path.join(path, 'output/weight_data')
