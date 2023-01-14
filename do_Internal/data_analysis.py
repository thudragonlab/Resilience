import json
import os
from typing import Dict,List
from other_script.util import record_launch_time
from other_script.my_types import *


@record_launch_time
def as_rela_txt(dsn_path:OUTPUT_PATH, txt_path:str,asn_data:Dict[AS_CODE,int],rtree_node_min_cone:int) -> str:
    '''
    dsn_path:output路径
    txt_path:原始txt文件路径
    asn_data:as-cone关系字典
    rtree_node_min_cone:配置文件中最小cone值
    处理AS关系数据
    {as:[peer, customor], ......}
    '''
    file = open(txt_path, 'r')
    file_name = txt_path.split('/')[-1][:-4]
    # dsn_path = '/home/peizd01/for_dragon/new_data_pzd/'
    real_dst_path = ''
    result:Dict[AS_CODE,List[AS_CODE]] = {}
    try:
        for line in file:
            if line[0] != '#':
                line = line.strip().split('|')
                a = line[0]
                b = line[1]
                # 如果 a 或者 b 不在cone文件中，或者对应cone大小小于最小值
                if a not in asn_data or b not in asn_data or asn_data[a] < rtree_node_min_cone or asn_data[b] < rtree_node_min_cone:
                    continue
                if line[-1] == '1': continue
                if a not in result.keys():
                # 第一个存与a时peer关系的，第二个存a的customer
                    result[a] = [[], []]
                # -1 表示 a 是 b 的provider
                if line[-1] == '-1':
                    result[a][1].append(b)
                else:
                    result[a][0].append(b)
        real_dst_path = os.path.join(dsn_path, file_name + '.json')
        with open(real_dst_path, 'w') as f:
            json.dump(result, f)
            print(real_dst_path)
            f.close()
    finally:
        file.close()
    return real_dst_path



def as_rela_txt_dont_save(txt_path:str) -> str:
    '''
    处理AS关系数据
    {as:[peer, customor], ......}
    '''
    file = open(txt_path, 'r')
    result:Dict[str,List[str]] = {}
    try:
        for line in file:
            if line[0] != '#':
                line = line.strip().split('|')
                a = line[0]
                b = line[1]
                if line[-1] == '1': continue
                if a not in result.keys():
                    result[a] = [[], []]
                if line[-1] == '-1':
                    result[a][1].append(b)
                else:
                    result[a][0].append(b)
    finally:
        file.close()
    return result